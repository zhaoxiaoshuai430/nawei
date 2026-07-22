import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    src = ROOT / "response.json"
    if len(sys.argv) > 1:
        src = Path(sys.argv[1])
    if not src.is_file() or src.stat().st_size == 0:
        raise SystemExit(f"文件为空或不存在: {src}")

    data = json.loads(src.read_text(encoding="utf-8"))
    b64 = data["audio_b64"]
    out = ROOT / "output.wav"
    out.write_bytes(base64.b64decode(b64))
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
