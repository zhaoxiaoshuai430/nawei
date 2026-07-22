"""TTS model load / unload helpers."""
from __future__ import annotations

import gc

import torch
from faster_qwen3_tts import FasterQwen3TTS

from app.services.tts import state
from app.services.tts.presets import prime_preset_voice_cache
from app.services.voice_lab.checkpoint import is_voice_lab_model, resolve_checkpoint_path


def get_gpu_free_memory(device=0) -> float:
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return 0.0
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    used_memory = total_memory - free_memory
    free_gb = free_memory / 1024**3
    used_gb = used_memory / 1024**3
    total_gb = total_memory / 1024**3
    print(f"total_memory:{total_gb},used_memory:{used_gb},free_memory:{free_gb}")
    return free_gb


def safe_unload_model(model) -> None:
    if model is None:
        return
    for attr in ["model", "predictor_graph", "talker_graph", "_voice_prompt_cache"]:
        if hasattr(model, attr):
            delattr(model, attr)
    del model
    print("正在释放已分配显存")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model_into_cache(model_id: str) -> None:
    if model_id in state.model_cache:
        state.active_model_name = model_id
        state.model_cache.move_to_end(model_id)
        return

    state.loading = True
    try:
        free_memory = get_gpu_free_memory(0)
        if state.model_cache and (
            len(state.model_cache) >= state.model_cache_max or free_memory < 6.5
        ):
            evicted, evicted_model = state.model_cache.popitem(last=False)
            print(f"Model cache full — evicted: {evicted}")
            safe_unload_model(evicted_model)
        load_path = model_id
        if is_voice_lab_model(model_id):
            ckpt = resolve_checkpoint_path(model_id)
            if ckpt is None:
                raise FileNotFoundError(
                    f"Voice lab checkpoint not found for {model_id}. "
                    "Train with voice_lab/ or set VOICE_LAB_CHECKPOINT."
                )
            load_path = str(ckpt)
            print(f"Loading voice_lab checkpoint: {load_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        new_model = FasterQwen3TTS.from_pretrained(
            load_path,
            device=device,
            dtype=dtype,
        )
        print("Capturing CUDA graphs…")
        if device == "cuda":
            new_model._warmup(prefill_len=100)
        state.model_cache[model_id] = new_model
        state.model_cache.move_to_end(model_id)
        state.active_model_name = model_id
        if not is_voice_lab_model(model_id):
            prime_preset_voice_cache(new_model)
        print("CUDA graphs captured — model ready.")
    finally:
        state.loading = False
