"""Static validation for the four-module production deployment."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DEPLOY_DIR = PROJECT_ROOT / "deploy"


class DeploymentConfigurationTest(unittest.TestCase):
    def test_production_compose_contains_four_modules_and_llm_storage(self) -> None:
        body = yaml.safe_load((PRODUCTION_DEPLOY_DIR / "docker-compose.production.yml").read_text(encoding="utf-8"))
        expected = {
            "config-check",
            "postgres",
            "redis",
            "asr",
            "llm",
            "tts",
            "gateway",
            "caddy",
            "blackbox-exporter",
            "prometheus",
        }
        self.assertEqual(set(body["services"]), expected)
        for removed in ("agent", "memory", "knowledge", "qdrant"):
            self.assertNotIn(removed, body["services"])
        self.assertEqual(set(body["services"]["llm"]["depends_on"]), {"config-check", "postgres", "redis"})
        self.assertEqual(set(body["services"]["gateway"]["depends_on"]), {"config-check", "asr", "llm", "tts"})
        self.assertTrue(body["networks"]["backend"]["internal"])

    def test_internal_services_do_not_publish_host_ports(self) -> None:
        body = yaml.safe_load((PRODUCTION_DEPLOY_DIR / "docker-compose.production.yml").read_text(encoding="utf-8"))
        for name in ("config-check", "postgres", "redis", "asr", "llm", "tts", "gateway"):
            self.assertFalse(body["services"][name].get("ports"), f"{name} must not publish host ports")

    def test_production_environment_uses_four_service_discovery(self) -> None:
        environment = (PRODUCTION_DEPLOY_DIR / ".env.production.example").read_text(encoding="utf-8")
        for service_name in ("asr", "llm", "tts"):
            self.assertIn(f"http://{service_name}:", environment)
        for removed in ("GATEWAY_AGENT_URL", "KNOWLEDGE_URL", "QDRANT_URL"):
            self.assertNotIn(removed, environment)
        self.assertIn("DATABASE_URL=postgresql://", environment)
        self.assertIn("REDIS_URL=redis://redis:", environment)
        service_urls = [line for line in environment.splitlines() if line.startswith("GATEWAY_") and "_URL=" in line]
        self.assertTrue(service_urls)
        self.assertFalse(any("127.0.0.1" in line for line in service_urls))

    def test_production_images_install_audio_runtime_tools(self) -> None:
        asr_dockerfile = (PRODUCTION_DEPLOY_DIR / "Dockerfile.asr").read_text(encoding="utf-8")
        tts_dockerfile = (PRODUCTION_DEPLOY_DIR / "Dockerfile.tts").read_text(encoding="utf-8")
        self.assertIn("ffmpeg", asr_dockerfile)
        self.assertIn("ffmpeg", tts_dockerfile)
        self.assertIn("sox", tts_dockerfile)

    def test_production_images_bundle_runtime_models(self) -> None:
        asr_dockerfile = (PRODUCTION_DEPLOY_DIR / "Dockerfile.asr").read_text(encoding="utf-8")
        tts_dockerfile = (PRODUCTION_DEPLOY_DIR / "Dockerfile.tts").read_text(encoding="utf-8")
        compose = (PRODUCTION_DEPLOY_DIR / "docker-compose.production.yml").read_text(encoding="utf-8")
        self.assertIn("models/asr", asr_dockerfile)
        self.assertIn("models/tts", tts_dockerfile)
        self.assertIn("models/manifest.json", tts_dockerfile)
        self.assertIn("restore_models.py", tts_dockerfile)
        self.assertNotIn("TTS_MODEL_DIR", compose)


if __name__ == "__main__":
    unittest.main()
