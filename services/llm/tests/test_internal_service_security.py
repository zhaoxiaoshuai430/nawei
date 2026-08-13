"""Internal LLM authentication tests for the four-service chain."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from services.llm.auth import authorize_llm_internal_request


class InternalServiceAuthenticationTest(unittest.TestCase):
    def test_llm_requires_a_strong_key_in_production(self) -> None:
        with patch.dict(os.environ, {"NAWEI_ENV": "production"}, clear=True), self.assertRaises(
            HTTPException
        ) as raised:
            authorize_llm_internal_request(None)
        self.assertEqual(raised.exception.status_code, 503)

        env = {"NAWEI_ENV": "production", "LLM_INTERNAL_API_KEY": "short"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(HTTPException) as raised:
            authorize_llm_internal_request("short")
        self.assertEqual(raised.exception.status_code, 503)

    def test_llm_rejects_an_invalid_configured_key(self) -> None:
        with patch.dict(os.environ, {"LLM_INTERNAL_API_KEY": "a" * 32}, clear=True), self.assertRaises(
            HTTPException
        ) as raised:
            authorize_llm_internal_request("wrong")
        self.assertEqual(raised.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
