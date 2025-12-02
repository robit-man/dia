#!/usr/bin/env python3
"""
Dia REST API server.

Exposes a small Flask API for running text-to-dialogue inference with the Dia model.
Routes:
  GET  /api/v1/health    -> service + model status
  GET  /api/v1/info      -> model/config metadata
  POST /api/v1/generate  -> generate audio from text (optional audio prompt)

The server loads the model once at startup and reuses it for all requests.
Environment knobs:
  DIA_MODEL_ID: HF repo id (default: nari-labs/Dia-1.6B-0626)
  DIA_DEVICE:  force device string (cpu, cuda, mps)
  DIA_COMPUTE_DTYPE: compute dtype for the model (float16|float32|bfloat16)
  DIA_HOST / DIA_PORT: bind address/port (default: 0.0.0.0:8000)
  DIA_MAX_UPLOAD_MB: max upload size for audio prompts (default: 20)
"""

from __future__ import annotations

import base64
import io
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import platform
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------#
# Virtualenv bootstrap (create .venv, re-exec, and install lightweight deps)
# -----------------------------------------------------------------------------#

BASE = Path(__file__).resolve().parent
VENV = BASE / ".venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")
PY = BIN / ("python.exe" if os.name == "nt" else "python")
PIP = BIN / ("pip.exe" if os.name == "nt" else "pip")
TORCH_INDEX_URL = os.getenv("DIA_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu126")


def _in_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == PY.resolve()
    except Exception:
        return False


def _ensure_venv():
    if VENV.exists():
        return
    import venv

    venv.EnvBuilder(with_pip=True).create(VENV)
    subprocess.check_call([str(PY), "-m", "pip", "install", "--upgrade", "pip"])


def _ensure_deps():
    basic_need = []
    for mod, pkg in [
        ("flask", "flask"),
        ("flask_cors", "flask-cors"),
        ("soundfile", "soundfile"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except Exception:
            basic_need.append(pkg)
    if basic_need:
        subprocess.check_call([str(PIP), "install", *basic_need])

    heavy_need = []
    try:
        import torch  # noqa: F401
    except Exception:
        heavy_need.append("torch==2.6.0")
    try:
        import torchaudio  # noqa: F401
    except Exception:
        heavy_need.append("torchaudio==2.6.0")
    try:
        import dac  # noqa: F401
    except Exception:
        heavy_need.append("descript-audio-codec>=1.0.0")

    if heavy_need:
        cmd = [str(PIP), "install", *heavy_need]
        # Prefer the PyTorch wheel index for Linux/Windows when available
        if platform.system().lower() in {"linux", "windows"} and TORCH_INDEX_URL:
            cmd.extend(["--extra-index-url", TORCH_INDEX_URL])
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            # Retry without extra index to allow CPU-only installs if CUDA wheels unavailable
            try:
                subprocess.check_call([str(PIP), "install", *heavy_need])
            except subprocess.CalledProcessError as exc:
                print(f"[Dia] Failed to install heavy dependencies: {exc}", file=sys.stderr)
                raise


if not _in_venv():
    _ensure_venv()
    os.execv(str(PY), [str(PY), *sys.argv])

_ensure_deps()

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from dia.model import DEFAULT_SAMPLE_RATE, Dia

# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#

APP_START = time.time()

MODEL_ID = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
DEVICE_HINT = os.getenv("DIA_DEVICE")
COMPUTE_DTYPE_HINT = os.getenv("DIA_COMPUTE_DTYPE")

DEFAULT_CFG_SCALE = float(os.getenv("DIA_CFG_SCALE", "3.0"))
DEFAULT_TEMPERATURE = float(os.getenv("DIA_TEMPERATURE", "1.8"))
DEFAULT_TOP_P = float(os.getenv("DIA_TOP_P", "0.95"))
DEFAULT_CFG_FILTER_TOP_K = int(os.getenv("DIA_CFG_FILTER_TOP_K", "45"))
DEFAULT_SPEED = float(os.getenv("DIA_SPEED", "1.0"))

MAX_UPLOAD_MB = int(os.getenv("DIA_MAX_UPLOAD_MB", "20"))
HOST = os.getenv("DIA_HOST", "0.0.0.0")
PORT = int(os.getenv("DIA_PORT", "8000"))

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#


def _truthy(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _select_device(hint: Optional[str]) -> torch.device:
    if hint:
        return torch.device(hint)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device(DEVICE_HINT)
COMPUTE_DTYPE = COMPUTE_DTYPE_HINT or ("float16" if DEVICE.type == "cuda" else "float32")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": os.getenv("DIA_CORS_ORIGINS", "*")}})

MODEL: Optional[Dia] = None
MODEL_LOAD_ERROR: Optional[str] = None
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()


def _coerce_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    if val is None or val == "":
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _coerce_float(val: Any, default: float) -> float:
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _coerce_bool(val: Any, default: bool = False) -> bool:
    return _truthy(val, default=default)


def set_seed(seed: Optional[int]):
    """Sets RNG seeds for reproducibility."""
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _model_max_tokens() -> int:
    if MODEL and hasattr(MODEL, "config"):
        return getattr(MODEL.config.decoder_config, "max_position_embeddings", 3072)
    return 3072


def _ensure_model() -> Dia:
    global MODEL, MODEL_LOAD_ERROR
    if MODEL is not None:
        return MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        try:
            MODEL = Dia.from_pretrained(MODEL_ID, compute_dtype=COMPUTE_DTYPE, device=DEVICE)
            MODEL_LOAD_ERROR = None
            return MODEL
        except Exception as exc:  # noqa: BLE001
            MODEL_LOAD_ERROR = str(exc)
            raise


def _merge_prompt_text(text: str, prompt_transcript: str) -> str:
    text = text.strip()
    prompt_transcript = prompt_transcript.strip()
    return f"{prompt_transcript}\n{text}" if prompt_transcript else text


def _time_stretch(audio: np.ndarray, speed: float) -> np.ndarray:
    speed = max(0.1, min(speed, 5.0))
    if speed == 1.0:
        return audio.astype(np.float32)
    original_len = len(audio)
    target_len = int(original_len / speed)
    if target_len <= 0:
        return audio.astype(np.float32)
    x_original = np.arange(original_len)
    x_resampled = np.linspace(0, original_len - 1, target_len)
    stretched = np.interp(x_resampled, x_original, audio)
    return stretched.astype(np.float32)


def _audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, DEFAULT_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.getvalue()


def _serialize_model_info() -> Dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "device": str(DEVICE),
        "compute_dtype": COMPUTE_DTYPE,
        "max_tokens": _model_max_tokens(),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "loaded": MODEL is not None and MODEL_LOAD_ERROR is None,
        "load_error": MODEL_LOAD_ERROR,
        "uptime_s": round(time.time() - APP_START, 3),
    }


# -----------------------------------------------------------------------------#
# Routes
# -----------------------------------------------------------------------------#


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Dia REST API ready", "endpoints": ["/api/v1/health", "/api/v1/info", "/api/v1/generate"]})


@app.route("/api/v1/health", methods=["GET"])
def health():
    status_code = 200 if MODEL_LOAD_ERROR is None else 503
    payload = {"status": "ok" if MODEL_LOAD_ERROR is None else "degraded"}
    payload.update(_serialize_model_info())
    return jsonify(payload), status_code


@app.route("/api/v1/info", methods=["GET"])
def info():
    return jsonify(_serialize_model_info())


@app.route("/api/v1/generate", methods=["POST"])
def generate_audio():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    if request.form:
        payload.update(request.form.to_dict())

    text = (payload.get("text") or payload.get("script") or "").strip()
    audio_prompt_text = (payload.get("audio_prompt_text") or payload.get("prompt_text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    # Handle optional audio prompt (multipart file or base64)
    audio_prompt_path: Optional[str] = None
    tmp_path: Optional[str] = None
    upload = request.files.get("audio_prompt") or request.files.get("prompt_audio")
    audio_prompt_b64 = payload.get("audio_prompt_base64")

    if upload:
        suffix = Path(upload.filename or "").suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            upload.save(tmp)
            tmp_path = tmp.name
            audio_prompt_path = tmp_path
    elif audio_prompt_b64:
        try:
            raw_audio = base64.b64decode(audio_prompt_b64)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"Could not decode audio_prompt_base64: {exc}"}), 400
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(raw_audio)
            tmp_path = tmp.name
            audio_prompt_path = tmp_path

    if audio_prompt_path and not audio_prompt_text:
        return jsonify({"error": "audio_prompt_text is required when providing an audio prompt"}), 400

    prompt_text = _merge_prompt_text(text, audio_prompt_text) if audio_prompt_path else text

    max_tokens = _coerce_int(payload.get("max_tokens"), default=_model_max_tokens()) or _model_max_tokens()
    max_tokens = max(128, min(max_tokens, _model_max_tokens()))
    cfg_scale = _coerce_float(payload.get("cfg_scale"), DEFAULT_CFG_SCALE)
    temperature = _coerce_float(payload.get("temperature"), DEFAULT_TEMPERATURE)
    top_p = min(1.0, max(0.01, _coerce_float(payload.get("top_p"), DEFAULT_TOP_P)))
    cfg_filter_top_k = max(1, _coerce_int(payload.get("cfg_filter_top_k"), DEFAULT_CFG_FILTER_TOP_K) or DEFAULT_CFG_FILTER_TOP_K)
    speed = _coerce_float(payload.get("speed"), DEFAULT_SPEED)
    verbose = _coerce_bool(payload.get("verbose"), default=False)
    seed = _coerce_int(payload.get("seed"), default=None)

    try:
        model = _ensure_model()
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Model load failed")
        return jsonify({"error": f"Model unavailable: {exc}"}), 503

    try:
        seed = set_seed(seed)
        start = time.time()
        with INFER_LOCK:
            audio = model.generate(
                prompt_text,
                max_tokens=max_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                audio_prompt=audio_prompt_path,
                verbose=verbose,
                use_torch_compile=False,
            )
        elapsed = time.time() - start
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Generation failed")
        return jsonify({"error": f"Inference failed: {exc}"}), 500
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    if isinstance(audio, list):
        audio = audio[0]
    if audio is None:
        return jsonify({"error": "No audio produced"}), 500

    audio_np = np.asarray(audio, dtype=np.float32)
    audio_np = _time_stretch(audio_np, speed)
    wav_bytes = _audio_to_wav_bytes(audio_np)

    response_meta = {
        "seed": seed,
        "duration_s": round(len(audio_np) / DEFAULT_SAMPLE_RATE, 3),
        "elapsed_s": round(elapsed, 3),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "model_id": MODEL_ID,
        "device": str(DEVICE),
        "compute_dtype": COMPUTE_DTYPE,
        "max_tokens": max_tokens,
        "cfg_scale": cfg_scale,
        "temperature": temperature,
        "top_p": top_p,
        "cfg_filter_top_k": cfg_filter_top_k,
        "speed": speed,
    }

    wants_wav = "audio/" in (request.headers.get("Accept", "") or "") or request.args.get("format") == "wav"
    if wants_wav:
        return send_file(io.BytesIO(wav_bytes), mimetype="audio/wav", download_name="dia-output.wav")

    encoded_audio = base64.b64encode(wav_bytes).decode("ascii")
    response_meta["audio_base64"] = encoded_audio
    return jsonify(response_meta)


if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
