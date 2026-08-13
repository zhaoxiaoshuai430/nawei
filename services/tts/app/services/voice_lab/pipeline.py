"""Voice lab pipeline status and background job runner."""
from __future__ import annotations

import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml

from app.config import REPO_ROOT, VOICE_LAB_DIR, VOICE_LAB_SPEAKER
from app.services.voice_lab.checkpoint import find_latest_checkpoint, voice_lab_model_id

SCRIPTS_DIR = VOICE_LAB_DIR / "scripts"
DATA_DIR = VOICE_LAB_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CLIPS_DIR = DATA_DIR / "clips"
MANIFEST_DIR = DATA_DIR / "manifests"
OUTPUT_DIR = VOICE_LAB_DIR / "output"
EMOTIONS_YAML = VOICE_LAB_DIR / "configs" / "emotions.yaml"
TRAIN_ENV_PY = REPO_ROOT / "train_env" / "Scripts" / "python.exe"
CUDA_ENV_PY = REPO_ROOT / "cuda_env" / "python.exe"

AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a"})
RAW_COUNT_GLOBS = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a")


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class VoiceLabJob:
    id: str
    step: str
    status: JobStatus = JobStatus.QUEUED
    log: list[str] = field(default_factory=list)
    error: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "status": self.status.value,
            "log": self.log[-200:],
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


_jobs: dict[str, VoiceLabJob] = {}
_job_lock = threading.Lock()
_active_job_id: str | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_python() -> Path:
    if TRAIN_ENV_PY.is_file():
        return TRAIN_ENV_PY
    if CUDA_ENV_PY.is_file():
        return CUDA_ENV_PY
    return Path(sys.executable)


def _load_emotions() -> dict:
    if not EMOTIONS_YAML.is_file():
        return {}
    with EMOTIONS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f).get("emotions") or {}


def list_drama_styles_for_chat() -> list[dict]:
    """Voice-lab 戏路 (lao/tu/de/…) for Chat instruct dropdown."""
    styles: list[dict] = []
    for key, emo in _load_emotions().items():
        styles.append(
            {
                "id": key,
                "label": emo.get("label") or key,
                "folder": emo.get("folder") or key,
                "instruct": (emo.get("instruct") or "").strip(),
            }
        )
    return styles


def _count_files(root: Path, pattern: str = "*") -> int:
    if not root.is_dir():
        return 0
    return sum(1 for p in root.rglob(pattern) if p.is_file())


def _count_audio_files(root: Path) -> int:
    if not root.is_dir():
        return 0
    seen: set[Path] = set()
    for pattern in RAW_COUNT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                seen.add(path.resolve())
    return len(seen)


def _validate_emotion_folder(emotion_folder: str) -> str:
    emotions = _load_emotions()
    valid_folders = {e.get("folder") or k for k, e in emotions.items()}
    folder = (emotion_folder or "neutral").strip().lower()
    if folder not in valid_folders:
        raise ValueError(f"Invalid emotion folder: {folder}")
    return folder


def _unique_dest(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename or "upload.wav").name
    dest = dest_dir / safe_name
    if not dest.exists():
        return dest
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix or ".wav"
    n = 2
    while (dest_dir / f"{stem}_{n}{suffix}").exists():
        n += 1
    return dest_dir / f"{stem}_{n}{suffix}"


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def get_pipeline_status() -> dict:
    emotions = _load_emotions()
    per_emo: dict[str, dict] = {}
    for key, emo in emotions.items():
        folder = emo.get("folder") or key
        raw_dir = RAW_DIR / folder
        clip_dir = CLIPS_DIR / folder
        clips = list(clip_dir.glob("*.wav")) if clip_dir.is_dir() else []
        transcribed = sum(1 for c in clips if c.with_suffix(".txt").is_file())
        per_emo[key] = {
            "key": key,
            "label": emo.get("label", key),
            "folder": folder,
            "min_clips": int(emo.get("min_clips", 0)),
            "raw_files": _count_audio_files(raw_dir),
            "clips": len(clips),
            "transcribed": transcribed,
            "instruct": emo.get("instruct", ""),
        }

    ckpt = find_latest_checkpoint()
    return {
        "speaker": VOICE_LAB_SPEAKER,
        "model_id": voice_lab_model_id(),
        "checkpoint": str(ckpt) if ckpt else None,
        "checkpoint_ready": ckpt is not None,
        "train_env": TRAIN_ENV_PY.is_file(),
        "vendor_official": (VOICE_LAB_DIR / "vendor" / "Qwen3-TTS" / "finetuning" / "prepare_data.py").is_file(),
        "python": str(_resolve_python()),
        "emotions": per_emo,
        "manifests": {
            "train_raw": _count_jsonl_lines(MANIFEST_DIR / "train_raw.jsonl"),
            "train_raw_instruct": _count_jsonl_lines(MANIFEST_DIR / "train_raw_instruct.jsonl"),
        },
        "active_job": _jobs[_active_job_id].to_dict() if _active_job_id and _active_job_id in _jobs else None,
    }


def get_job(job_id: str) -> VoiceLabJob | None:
    return _jobs.get(job_id)


def _run_command(job: VoiceLabJob, cmd: list[str], *, cwd: Path) -> None:
    job.status = JobStatus.RUNNING
    job.started_at = _now_iso()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            job.log.append(line.rstrip("\n"))
        code = proc.wait()
        if code != 0:
            job.status = JobStatus.ERROR
            job.error = f"exit code {code}"
        else:
            job.status = JobStatus.DONE
    except Exception as exc:
        job.status = JobStatus.ERROR
        job.error = str(exc)
        job.log.append(f"ERROR: {exc}")
    finally:
        job.finished_at = _now_iso()
        global _active_job_id
        with _job_lock:
            if _active_job_id == job.id:
                _active_job_id = None


def _command_for_step(step: str) -> tuple[list[str], Path]:
    py = _resolve_python()
    repo = REPO_ROOT
    vl = VOICE_LAB_DIR

    if step == "split":
        return [str(py), str(SCRIPTS_DIR / "01_split_audio.py")], repo
    if step == "split_classify":
        return [str(py), str(SCRIPTS_DIR / "01b_split_and_classify.py")], repo
    if step == "transcribe":
        return [str(py), str(SCRIPTS_DIR / "02_transcribe.py")], repo
    if step == "build_jsonl":
        return [str(py), str(SCRIPTS_DIR / "03_build_jsonl.py")], repo
    if step == "train_timbre":
        ps1 = SCRIPTS_DIR / "04_train.ps1"
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
            "-Mode",
            "timbre",
        ], repo
    if step == "train_instruct":
        ps1 = SCRIPTS_DIR / "04_train.ps1"
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ps1),
            "-Mode",
            "instruct",
        ], repo
    if step == "eval":
        ckpt = find_latest_checkpoint()
        if not ckpt:
            raise ValueError("No checkpoint found. Train first.")
        return [str(py), str(SCRIPTS_DIR / "07_eval_infer.py"), "--checkpoint", str(ckpt)], repo
    raise ValueError(f"Unknown step: {step}")


ALLOWED_STEPS = frozenset(
    {"split", "split_classify", "transcribe", "build_jsonl", "train_timbre", "train_instruct", "eval"}
)


def start_step(step: str) -> VoiceLabJob:
    global _active_job_id
    step = step.strip().lower()
    if step not in ALLOWED_STEPS:
        raise ValueError(f"Invalid step. Choose from: {', '.join(sorted(ALLOWED_STEPS))}")

    with _job_lock:
        if _active_job_id and _active_job_id in _jobs:
            active = _jobs[_active_job_id]
            if active.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                raise RuntimeError(f"Job already running: {active.step} ({active.id})")

    cmd, cwd = _command_for_step(step)
    job = VoiceLabJob(id=uuid.uuid4().hex[:12], step=step)
    _jobs[job.id] = job
    with _job_lock:
        _active_job_id = job.id

    thread = threading.Thread(target=_run_command, args=(job, cmd), kwargs={"cwd": cwd}, daemon=True)
    thread.start()
    return job


def import_raw_to_clips(emotion_folder: str | None = None) -> dict:
    """Move/copy files from data/raw/{emotion}/ to data/clips/ (for mis-uploaded short clips)."""
    import shutil

    emotions = _load_emotions()
    if emotion_folder:
        folders = [_validate_emotion_folder(emotion_folder)]
    else:
        folders = [e.get("folder") or k for k, e in emotions.items()]

    per_emo: dict[str, int] = {}
    for folder in folders:
        raw_dir = RAW_DIR / folder
        clip_dir = CLIPS_DIR / folder
        clip_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        if not raw_dir.is_dir():
            per_emo[folder] = 0
            continue
        for pattern in RAW_COUNT_GLOBS:
            for src in list(raw_dir.glob(pattern)):
                if not src.is_file():
                    continue
                if src.suffix.lower() == ".wav":
                    dest = _unique_dest(clip_dir, src.name)
                    shutil.move(str(src), str(dest))
                else:
                    dest = save_clip_upload(folder, src.name, src.read_bytes())
                    src.unlink(missing_ok=True)
                    dest = Path(dest)
                count += 1
        per_emo[folder] = count
    return {"status": "ok", "moved": per_emo, "total": sum(per_emo.values())}


def save_raw_upload(emotion_folder: str, filename: str, content: bytes) -> Path:
    folder = _validate_emotion_folder(emotion_folder)
    dest = _unique_dest(RAW_DIR / folder, filename or "upload.wav")
    dest.write_bytes(content)
    return dest


def save_clip_upload(emotion_folder: str, filename: str, content: bytes) -> Path:
    """Save a pre-cut clip (converted to mono 24kHz WAV under data/clips/)."""
    from app.utils.audio import AudioFormatError, bytes_to_wav_bytes

    folder = _validate_emotion_folder(emotion_folder)
    suffix = Path(filename or "upload.wav").suffix or ".wav"
    try:
        wav_bytes = bytes_to_wav_bytes(content, suffix, target_sr=24000)
    except AudioFormatError as exc:
        raise ValueError(str(exc.detail)) from exc
    wav_name = Path(filename or "upload.wav").stem + ".wav"
    dest = _unique_dest(CLIPS_DIR / folder, wav_name)
    dest.write_bytes(wav_bytes)
    return dest


def save_uploads_batch(
    *,
    emotion_folder: str,
    target: str,
    files: list[tuple[str, bytes]],
) -> dict:
    """Upload one or many files to raw/ or clips/. Returns summary for the UI."""
    target = (target or "raw").strip().lower()
    if target not in ("raw", "clips"):
        raise ValueError("target must be raw or clips")
    if not files:
        raise ValueError("请选择至少一个音频文件")

    saver = save_raw_upload if target == "raw" else save_clip_upload
    saved: list[str] = []
    errors: list[dict[str, str]] = []

    for filename, content in files:
        if not content:
            errors.append({"file": filename or "?", "error": "文件为空"})
            continue
        try:
            path = saver(emotion_folder, filename, content)
            saved.append(str(path))
        except ValueError as exc:
            errors.append({"file": filename or "?", "error": str(exc)})

    if not saved and errors:
        raise ValueError(errors[0]["error"])

    folder = _validate_emotion_folder(emotion_folder)
    raw_dir = RAW_DIR / folder
    clip_dir = CLIPS_DIR / folder
    clip_count = len(list(clip_dir.glob("*.wav"))) if clip_dir.is_dir() else 0
    return {
        "status": "ok",
        "target": target,
        "emotion": folder,
        "saved_count": len(saved),
        "error_count": len(errors),
        "paths": saved,
        "errors": errors,
        "raw_files": _count_audio_files(raw_dir),
        "clips": clip_count,
    }
