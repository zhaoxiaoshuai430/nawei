"""Create an HS256 token accepted by the production Gateway."""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--expires-in", type=int, default=3600)
    args = parser.parse_args()

    secret = os.getenv("GATEWAY_JWT_SECRET", "")
    if len(secret.encode("utf-8")) < 32:
        raise SystemExit("GATEWAY_JWT_SECRET must contain at least 32 bytes")
    if args.expires_in < 60:
        raise SystemExit("--expires-in must be at least 60 seconds")

    now = int(time.time())
    payload: dict[str, object] = {
        "sub": args.user_id,
        "iat": now,
        "nbf": now,
        "exp": now + args.expires_in,
    }
    issuer = os.getenv("GATEWAY_JWT_ISSUER", "").strip()
    audience = os.getenv("GATEWAY_JWT_AUDIENCE", "").strip()
    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience

    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = b".".join((_encode_json(header), _encode_json(payload)))
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    print(f"{signing_input.decode('ascii')}.{_base64url(signature)}")


def _encode_json(value: dict[str, object]) -> bytes:
    return _base64url(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).encode("ascii")


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


if __name__ == "__main__":
    main()
