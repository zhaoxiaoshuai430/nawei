"""Download or verify the local Paraformer online model using the ASR environment."""

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("config.yaml", "configuration.json", "model.pt", "tokens.json", "am.mvn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "models/asr/paraformer-online")
    args = parser.parse_args()
    destination = args.output.resolve()
    if args.download:
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(MODEL_ID, local_dir=str(destination), allow_patterns=[*REQUIRED, "seg_dict", "README.md"])
    missing = [name for name in REQUIRED if not (destination / name).is_file() or not (destination / name).stat().st_size]
    if missing:
        raise SystemExit("Missing streaming model files: " + ", ".join(missing))
    import yaml

    config = yaml.safe_load((destination / "config.yaml").read_text(encoding="utf-8"))
    if config.get("model") != "ParaformerStreaming":
        raise SystemExit("Expected a native ParaformerStreaming model, not an offline ASR model.")
    print(f"Verified native streaming model: {destination}")


if __name__ == "__main__":
    main()
