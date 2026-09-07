"""Tests for the shared Xiaowei persona prompt."""

from __future__ import annotations

import unittest

from app_protocols.persona import XIAOWEI_PERSONA_VERSION, format_response_policy, format_xiaowei_system_prompt


class XiaoweiPersonaTest(unittest.TestCase):
    def test_prompt_uses_persona_v6(self) -> None:
        prompt = format_xiaowei_system_prompt("2026-08-26")

        self.assertEqual(XIAOWEI_PERSONA_VERSION, "xiaowei-persona-v6")
        self.assertIn("persona_version: xiaowei-persona-v6", prompt)
        self.assertIn("当前日期：2026-08-26", prompt)

    def test_persona_and_response_policy_define_daily_guardian_style(self) -> None:
        persona = format_xiaowei_system_prompt()
        policy = format_response_policy(
            {
                "scene": "daily_life",
                "interaction_mode": "understand_then_practical_help",
                "companion_style": "reliable_guardian",
                "humor_level": "lively",
                "care_focus": "practical_solution_and_risk",
                "safety_note": "不得建议把备用钥匙藏在门口附近",
            }
        )

        self.assertIn("日常生活和轻松聊天要自然、有生活感", persona)
        self.assertIn("守护型伙伴", persona)
        self.assertIn("站在用户这边的守护型伙伴", policy)
        self.assertIn("自然加入一处贴合情境的幽默", policy)
        self.assertIn("风险、后果或下一步", policy)
        self.assertIn("不使用编号、加粗小标题、教程或长清单", policy)
        self.assertIn("生活场景输出契约", policy)
        self.assertIn("不得推断用户发生过几次", policy)
        self.assertIn("整篇最多出现一个问号", policy)
        self.assertIn("具体安全边界：不得建议把备用钥匙藏在门口附近", policy)
        self.assertIn("不建议把钥匙、凭据或私人信息交给不可信的人", persona)
        self.assertNotIn("\f", persona)
        self.assertNotIn("\b", persona)


if __name__ == "__main__":
    unittest.main()
