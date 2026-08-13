"""Runtime discovery for native audio tools used by the TTS service."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def ensure_sox_on_path() -> Path | None:
    """Add an installed SoX executable to PATH before audio modules import."""

    existing = shutil.which("sox")
    if existing:
        return Path(existing)

    for candidate in _sox_candidates():
        if candidate.is_file():
            _prepend_to_path(candidate.parent)
            return candidate
    return None


def _sox_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("NAWEI_SOX_EXECUTABLE", "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())

    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        local_root = Path(local_app_data)
        candidates.append(local_root / "Microsoft" / "WinGet" / "Links" / "sox.exe")
        packages = local_root / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            candidates.extend(packages.glob("ChrisBagwell.SoX_*/*/sox.exe"))

    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        program_files = os.getenv(variable, "").strip()
        if program_files:
            candidates.extend(Path(program_files).glob("sox*/sox.exe"))
    return candidates


def _prepend_to_path(directory: Path) -> None:
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    directory_text = str(directory)
    if directory_text not in entries:
        os.environ["PATH"] = os.pathsep.join([directory_text, *entries])
