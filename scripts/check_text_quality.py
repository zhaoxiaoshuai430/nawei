"""Validate project-owned text files and local Markdown links."""

from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bat",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
}
PRUNED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runtime-tools",
    ".venv",
    "SenseVoice",
    "__pycache__",
    "cuda_env",
    "data",
    "hf_download",
    "logs",
    "output",
}
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\((?!https?://|mailto:|#)([^)]+)\)")
FORBIDDEN_PROJECT_PATH = "D:\\daima\\nawei"


def iter_text_files() -> list[Path]:
    files: list[Path] = []
    pending = [PROJECT_ROOT]
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            if path.is_dir():
                if path.name not in PRUNED_DIRECTORIES:
                    pending.append(path)
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                files.append(path)
    return sorted(files)


def validate_text(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    relative = path.relative_to(PROJECT_ROOT)
    for line_number, line in enumerate(text.splitlines(), 1):
        if "\ufffd" in line:
            issues.append(f"{relative}:{line_number}: contains Unicode replacement characters")
        if any("\ue000" <= character <= "\uf8ff" for character in line):
            issues.append(f"{relative}:{line_number}: contains private-use characters")
        if FORBIDDEN_PROJECT_PATH.lower() in line.lower():
            issues.append(f"{relative}:{line_number}: contains a machine-specific project path")
    return issues


def validate_markdown_links(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    for match in MARKDOWN_LINK_RE.finditer(text):
        raw_target = match.group(1).split("#", 1)[0].strip()
        if not raw_target:
            continue
        target = (path.parent / raw_target).resolve()
        if not target.exists():
            relative = path.relative_to(PROJECT_ROOT)
            issues.append(f"{relative}: broken local link: {raw_target}")
    return issues


def main() -> int:
    issues: list[str] = []
    files = iter_text_files()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            issues.append(f"{path.relative_to(PROJECT_ROOT)}: invalid UTF-8: {exc}")
            continue
        issues.extend(validate_text(path, text))
        if path.suffix.lower() == ".md":
            issues.extend(validate_markdown_links(path, text))

    if issues:
        print("Text quality check failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print(f"Text quality passed for {len(files)} project files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
