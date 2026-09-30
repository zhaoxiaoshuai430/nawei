#!/usr/bin/env python3
"""Validate bundled model files and restore local split weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PART_PATTERN = re.compile(r"^(?P<name>.+)\.part(?P<index>\d{3})$")


def restore_models(model_root: Path, prefix: str = "", *, remove_parts: bool = False) -> None:
    manifest_path = model_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if prefix and not entry["path"].startswith(prefix):
            continue
        target = model_root / entry["path"]
        expected_size = int(entry["size"])
        expected_sha256 = entry["sha256"]
        declared_parts = entry.get("parts", [])
        target_is_valid = (
            target.is_file()
            and target.stat().st_size == expected_size
            and _sha256(target) == expected_sha256
        )
        if not target_is_valid:
            if not declared_parts:
                raise ValueError(
                    f"Model file is missing or invalid: {target}. Restore the matching local model file before starting."
                )
            parts = [model_root / part for part in declared_parts]
            missing_parts = [part for part in parts if not part.is_file()]
            if missing_parts:
                raise FileNotFoundError(f"Missing model split parts: {', '.join(map(str, missing_parts))}")
            parts.sort(key=_part_index)

            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".restoring")
            with temporary.open("wb") as destination:
                for part in parts:
                    with part.open("rb") as source:
                        while chunk := source.read(16 * 1024 * 1024):
                            destination.write(chunk)

            if temporary.stat().st_size != expected_size or _sha256(temporary) != expected_sha256:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Restored model checksum mismatch: {target}")
            temporary.replace(target)
        if remove_parts:
            _remove_declared_parts(model_root, declared_parts)


def _remove_declared_parts(model_root: Path, declared_parts: list[str]) -> None:
    for part in declared_parts:
        (model_root / part).unlink(missing_ok=True)


def _part_index(path: Path) -> int:
    match = PART_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Invalid model part name: {path}")
    return int(match.group("index"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument("--prefix", default="", help="Only restore manifest paths under this prefix.")
    parser.add_argument(
        "--remove-parts",
        action="store_true",
        help="Delete declared split parts after the restored model passes checksum validation.",
    )
    arguments = parser.parse_args()
    restore_models(arguments.model_root.resolve(), arguments.prefix, remove_parts=arguments.remove_parts)
    print(f"Bundled models are ready under {arguments.model_root.resolve()}")


if __name__ == "__main__":
    main()
