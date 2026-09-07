"""Text normalization and deterministic knowledge-document chunking."""

from __future__ import annotations

import re


def chunk_text(text: str, *, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunk_size = max(100, chunk_size)
    overlap = max(0, min(overlap, chunk_size // 2))
    paragraphs = [item.strip() for item in re.split(r"\n{2,}", normalized) if item.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            pieces.append(paragraph[start:end].strip())
            if end >= len(paragraph):
                break
            start = max(start + 1, end - overlap)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        prefix = chunks[-1][-overlap:].strip() if chunks and overlap else ""
        current = f"{prefix}\n{piece}".strip() if prefix else piece
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _normalize_text(text: str) -> str:
    value = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n[ \t]+", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
