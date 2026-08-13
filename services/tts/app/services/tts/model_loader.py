"""Compatibility loader for faster-qwen3-tts and current Transformers."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.predictor_graph import PredictorGraph
from faster_qwen3_tts.talker_graph import TalkerGraph
from faster_qwen3_tts.utils import suppress_flash_attn_warning


def load_faster_model(
    model_name: str,
    *,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    max_seq_len: int = 2048,
) -> FasterQwen3TTS:
    """Load the accelerated model without deprecated Transformers arguments."""

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("CUDA graphs require a CUDA device")

    with suppress_flash_attn_warning():
        from qwen_tts import Qwen3TTSModel

    base_model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    base_model.sample_rate = resolve_model_sample_rate(model_name)

    talker = base_model.model.talker
    talker_config = base_model.model.config.talker_config
    predictor = talker.code_predictor
    predictor_config = predictor.model.config

    predictor_graph = PredictorGraph(
        predictor,
        predictor_config,
        talker_config.hidden_size,
        device=device,
        dtype=dtype,
        do_sample=True,
        top_k=50,
        temperature=0.9,
    )
    talker_graph = TalkerGraph(
        talker.model,
        talker_config,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )
    return FasterQwen3TTS(
        base_model=base_model,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device=device,
        dtype=dtype,
        max_seq_len=max_seq_len,
    )


def resolve_model_sample_rate(model_name: str) -> int:
    """Read the waveform sample rate from a local checkpoint when available."""

    configured = os.getenv("TTS_OUTPUT_SAMPLE_RATE", "").strip()
    if configured:
        return int(configured)

    model_path = Path(model_name)
    candidates = (
        model_path / "speech_tokenizer" / "preprocessor_config.json",
        model_path / "speech_tokenizer" / "config.json",
        model_path / "config.json",
    )
    for candidate in candidates:
        value = _read_sample_rate(candidate)
        if value is not None:
            return value
    return 24000


def _read_sample_rate(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _find_sample_rate(document)


def _find_sample_rate(value: object) -> int | None:
    if isinstance(value, dict):
        for key in ("output_sample_rate", "sampling_rate", "sample_rate"):
            candidate = value.get(key)
            if isinstance(candidate, int) and candidate > 0:
                return candidate
        for nested in value.values():
            candidate = _find_sample_rate(nested)
            if candidate is not None:
                return candidate
    elif isinstance(value, list):
        for nested in value:
            candidate = _find_sample_rate(nested)
            if candidate is not None:
                return candidate
    return None
