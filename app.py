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
import csv
import io
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import platform
import signal
import shutil
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
JETSON_INDEX_URL = os.getenv("DIA_JETSON_INDEX_URL", "https://developer.download.nvidia.com/compute/redist/jp/v60")
JETSON_TORCH_WHEEL = os.getenv("DIA_JETSON_TORCH_WHEEL")
JETSON_TORCHAUDIO_WHEEL = os.getenv("DIA_JETSON_TORCHAUDIO_WHEEL")

# Known Jetson torch wheels (JP6 / CUDA 12.x) by Python ABI; adjust via env overrides when needed.
JETSON_WHEELS = {
    "cp312": "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp312-cp312-linux_aarch64.whl",
    "cp311": "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+3bcc3cddb5.nv24.07.16234504-cp311-cp311-linux_aarch64.whl",
    "cp310": "https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+f70bd71a48.nv24.06.15634931-cp310-cp310-linux_aarch64.whl",
}


def _is_jetson() -> bool:
    try:
        return platform.machine().lower() == "aarch64" and Path("/etc/nv_tegra_release").exists()
    except Exception:
        return False


def _jetson_cuda_env() -> dict[str, str]:
    env = os.environ.copy()
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-12.2/lib64",
        "/usr/local/cuda-12.4/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu/tegra",
        "/usr/lib/aarch64-linux-gnu/tegra-egl",
    ]
    ld_path = env.get("LD_LIBRARY_PATH", "")
    for path in cuda_paths:
        if Path(path).exists() and path not in ld_path:
            ld_path = f"{ld_path}:{path}" if ld_path else path
    env["LD_LIBRARY_PATH"] = ld_path
    if "CUDA_HOME" not in env:
        for cuda_home in ["/usr/local/cuda-12.6", "/usr/local/cuda-12.4", "/usr/local/cuda-12.2", "/usr/local/cuda-12", "/usr/local/cuda"]:
            if Path(cuda_home).exists():
                env["CUDA_HOME"] = cuda_home
                break
    return env


def _install_jetson_cusparselt():
    """Best-effort install of cuSPARSELt required by recent PyTorch on JetPack 6."""
    lib_paths = [
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.2/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.4/lib64/libcusparseLt.so.0",
        "/usr/local/cuda-12.6/lib64/libcusparseLt.so.0",
        "/usr/lib/libcusparseLt.so.0",
    ]
    if any(Path(p).exists() for p in lib_paths):
        return
    print("[Dia] Jetson: libcusparseLt.so.0 not found; attempting install...")
    # Mirror PyTorch install script (lightweight retry)
    try:
        import tarfile
        import tempfile
        import urllib.request

        # Default to CUDA 12.x tarball for JetPack 6 (sbsa)
        version = "0.5.2.1"
        arch = "sbsa"
        name = f"libcusparse_lt-linux-{arch}-{version}-archive"
        url = f"https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-{arch}/{name}.tar.xz"
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / f"{name}.tar.xz"
            urllib.request.urlretrieve(url, archive)
            with tarfile.open(archive, "r:xz") as tar:
                tar.extractall(td)
            extracted = Path(td) / name
            cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
            lib_dst = cuda_home / "lib64"
            include_dst = cuda_home / "include"
            lib_dst.mkdir(parents=True, exist_ok=True)
            include_dst.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", str(extracted / "lib") + "/.", str(lib_dst)], check=True)
            subprocess.run(["cp", "-a", str(extracted / "include") + "/.", str(include_dst)], check=True)
            print("[Dia] Installed cuSPARSELt for Jetson.")
    except Exception as exc:
        print(f"[Dia] Jetson cuSPARSELt install skipped/failed: {exc}")


def _default_jetson_wheels() -> tuple[str | None, str | None]:
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return JETSON_WHEELS.get(py_tag), None  # torchaudio URL not bundled; expect pip to resolve from index


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
        ("pydantic", "pydantic"),
        ("huggingface_hub", "huggingface-hub"),
        ("safetensors", "safetensors"),
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
        jetson = _is_jetson()
        cuda_env = _jetson_cuda_env() if jetson else os.environ.copy()

        # Handle torch/torchaudio separately for Jetson to pick correct wheels
        torch_missing = any(pkg.startswith("torch") for pkg in heavy_need)
        ta_missing = any(pkg.startswith("torchaudio") for pkg in heavy_need)

        # Remaining packages (e.g., dac) can be installed normally
        other_need = [pkg for pkg in heavy_need if not (pkg.startswith("torch") or pkg.startswith("torchaudio"))]

        if jetson and (torch_missing or ta_missing):
            _install_jetson_cusparselt()
            torch_url, ta_url = _default_jetson_wheels()
            torch_url = JETSON_TORCH_WHEEL or torch_url
            ta_url = JETSON_TORCHAUDIO_WHEEL or ta_url

            if torch_missing:
                torch_cmd = [str(PIP), "install"]
                if torch_url:
                    torch_cmd.append(torch_url)
                else:
                    torch_cmd.extend(["torch==2.6.0", "-f", JETSON_INDEX_URL])
                try:
                    subprocess.check_call(torch_cmd, env=cuda_env)
                except subprocess.CalledProcessError:
                    print("[Dia] Jetson torch install failed; retrying CPU wheel...", file=sys.stderr)
                    subprocess.check_call([str(PIP), "install", "torch==2.6.0"], env=cuda_env)

            if ta_missing:
                ta_cmd = [str(PIP), "install"]
                if ta_url:
                    ta_cmd.append(ta_url)
                else:
                    ta_cmd.extend(["torchaudio==2.6.0", "-f", JETSON_INDEX_URL])
                try:
                    subprocess.check_call(ta_cmd, env=cuda_env)
                except subprocess.CalledProcessError:
                    print("[Dia] Jetson torchaudio install failed; retrying CPU wheel...", file=sys.stderr)
                    subprocess.check_call([str(PIP), "install", "torchaudio==2.6.0"], env=cuda_env)

        # Install remaining packages or non-Jetson heavy deps
        remaining = []
        if not jetson:
            remaining = heavy_need
        else:
            remaining = other_need

        if remaining:
            cmd = [str(PIP), "install", *remaining]
            if platform.system().lower() in {"linux", "windows"} and TORCH_INDEX_URL:
                cmd.extend(["--extra-index-url", TORCH_INDEX_URL])
            try:
                subprocess.check_call(cmd, env=cuda_env)
            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call([str(PIP), "install", *remaining], env=cuda_env)
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
from huggingface_hub import snapshot_download

from dia.model import DEFAULT_SAMPLE_RATE, Dia

# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#

APP_START = time.time()

MODEL_ID = os.getenv("DIA_MODEL_ID", "nari-labs/Dia-1.6B-0626")
MODEL_CHOICES = [m.strip() for m in os.getenv("DIA_MODEL_CHOICES", MODEL_ID).split(",") if m.strip()]
ACTIVE_MODEL_ID = MODEL_CHOICES[0] if MODEL_CHOICES else MODEL_ID
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
CACHE_DIR = Path(os.getenv("DIA_CACHE_DIR", BASE / ".cache" / "hf")).expanduser()
os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # use rust downloader when available

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
SHUTDOWN_EVENT = threading.Event()
MODEL_STATUS: Dict[str, Any] = {"status": "idle", "model_id": ACTIVE_MODEL_ID, "error": None, "ts": time.time()}
MODEL_LOADER_THREAD: Optional[threading.Thread] = None

NVIDIA_SMI_QUERY = "index,name,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw"
NVIDIA_SMI_FIELDS = [
    "index",
    "name",
    "uuid",
    "gpu_util_percent",
    "mem_util_percent",
    "mem_used_mb",
    "mem_total_mb",
    "temperature_c",
    "power_draw_w",
]


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
            MODEL_STATUS.update({"status": "loading", "model_id": ACTIVE_MODEL_ID, "error": None, "ts": time.time()})
            _pre_download_model(ACTIVE_MODEL_ID)
            MODEL = Dia.from_pretrained(ACTIVE_MODEL_ID, compute_dtype=COMPUTE_DTYPE, device=DEVICE)
            MODEL_LOAD_ERROR = None
            MODEL_STATUS.update({"status": "ready", "model_id": ACTIVE_MODEL_ID, "error": None, "ts": time.time()})
            return MODEL
        except Exception as exc:  # noqa: BLE001
            MODEL_LOAD_ERROR = str(exc)
            MODEL_STATUS.update({"status": "error", "model_id": ACTIVE_MODEL_ID, "error": MODEL_LOAD_ERROR, "ts": time.time()})
            raise


def _merge_prompt_text(text: str, prompt_transcript: str) -> str:
    text = text.strip()
    prompt_transcript = prompt_transcript.strip()
    return f"{prompt_transcript}\n{text}" if prompt_transcript else text


def _pre_download_model(model_id: str):
    """Ensure model artifacts are locally cached before load (resumable)."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=CACHE_DIR / "models" / model_id.replace("/", "_"),
            allow_patterns=[
                "*.safetensors",
                "pytorch_model.bin",
                "config.json",
                "model.safetensors.index.json",
                "*.json",
            ],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Dia] Warning: pre-download failed ({exc}), will rely on lazy load.")


def _gpu_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {"ts": time.time(), "devices": [], "gpu_available": False}
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        stats["error"] = "nvidia-smi not found"
        return stats

    try:
        output = subprocess.check_output(
            [nvidia_smi, f"--query-gpu={NVIDIA_SMI_QUERY}", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=5,
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        stats["error"] = str(exc)
        return stats

    def _parse_float(val: Optional[str]) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(str(val).strip())
        except Exception:
            return None

    def _parse_int(val: Optional[str]) -> Optional[int]:
        parsed = _parse_float(val)
        return int(parsed) if parsed is not None else None

    try:
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if not row or len(row) < len(NVIDIA_SMI_FIELDS):
                continue
            stats["devices"].append(
                {
                    "index": _parse_int(row[0]),
                    "name": row[1].strip() if row[1] else "",
                    "uuid": row[2].strip() if row[2] else "",
                    "gpu_util_percent": _parse_int(row[3]),
                    "mem_util_percent": _parse_int(row[4]),
                    "mem_used_mb": _parse_int(row[5]),
                    "mem_total_mb": _parse_int(row[6]),
                    "temperature_c": _parse_float(row[7]),
                    "power_draw_w": _parse_float(row[8]),
                }
            )
        stats["gpu_available"] = bool(stats["devices"])
    except Exception as exc:  # noqa: BLE001
        stats["error"] = str(exc)
    return stats


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
        "model_id": ACTIVE_MODEL_ID,
        "available_models": MODEL_CHOICES,
        "device": str(DEVICE),
        "compute_dtype": COMPUTE_DTYPE,
        "max_tokens": _model_max_tokens(),
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "loaded": MODEL is not None and MODEL_LOAD_ERROR is None,
        "load_error": MODEL_LOAD_ERROR,
        "uptime_s": round(time.time() - APP_START, 3),
        "model_status": MODEL_STATUS,
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
        if MODEL_STATUS.get("status") not in {"ready", "idle"} and MODEL is None:
            return jsonify({"error": "Model not ready", "status": MODEL_STATUS}), 503
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


@app.route("/api/v1/gpu", methods=["GET"])
def gpu():
    return jsonify(_gpu_stats())


@app.route("/api/v1/models", methods=["GET"])
def list_models():
    return jsonify({"active": ACTIVE_MODEL_ID, "available": MODEL_CHOICES, "status": MODEL_STATUS})


def _load_model_background(model_id: str):
    global MODEL, ACTIVE_MODEL_ID, MODEL_LOAD_ERROR
    with MODEL_LOCK:
        MODEL = None
        MODEL_LOAD_ERROR = None
        ACTIVE_MODEL_ID = model_id
        MODEL_STATUS.update({"status": "loading", "model_id": model_id, "error": None, "ts": time.time()})
    try:
        _pre_download_model(model_id)
        loaded = Dia.from_pretrained(model_id, compute_dtype=COMPUTE_DTYPE, device=DEVICE)
        with MODEL_LOCK:
            MODEL = loaded
            MODEL_STATUS.update({"status": "ready", "model_id": model_id, "error": None, "ts": time.time()})
    except Exception as exc:  # noqa: BLE001
        with MODEL_LOCK:
            MODEL = None
            MODEL_LOAD_ERROR = str(exc)
            MODEL_STATUS.update({"status": "error", "model_id": model_id, "error": MODEL_LOAD_ERROR, "ts": time.time()})


@app.route("/api/v1/models/load", methods=["POST"])
def load_model():
    global MODEL_LOADER_THREAD
    payload = request.get_json(silent=True) or {}
    target = (payload.get("model_id") or "").strip()
    if not target:
        return jsonify({"error": "model_id is required"}), 400
    if target not in MODEL_CHOICES:
        return jsonify({"error": f"model_id must be one of {MODEL_CHOICES}"}), 400
    if MODEL_STATUS.get("status") == "loading":
        return jsonify({"status": MODEL_STATUS, "message": "load already in progress"}), 200
    if target == ACTIVE_MODEL_ID and MODEL_STATUS.get("status") == "ready" and MODEL is not None:
        return jsonify({"status": MODEL_STATUS, "message": "model already loaded"}), 200

    if MODEL_LOADER_THREAD and MODEL_LOADER_THREAD.is_alive():
        MODEL_LOADER_THREAD.join(timeout=0.1)

    MODEL_LOADER_THREAD = threading.Thread(target=_load_model_background, args=(target,), daemon=True)
    MODEL_LOADER_THREAD.start()
    return jsonify({"status": MODEL_STATUS, "message": "loading started"}), 202


@app.route("/api/v1/model_status", methods=["GET"])
def model_status():
    return jsonify({"active": ACTIVE_MODEL_ID, "status": MODEL_STATUS, "error": MODEL_STATUS.get("error")})


if __name__ == "__main__":
    def _handle_signal(signum, _frame):
        print(f"[Dia] Received signal {signum}, shutting down gracefully...")
        SHUTDOWN_EVENT.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    try:
        app.run(host=HOST, port=PORT)
    except KeyboardInterrupt:
        _handle_signal(signal.SIGINT, None)
