#!/usr/bin/env bash
# setup_training_env.sh - Create a dedicated uv-managed training venv for train_v7.py.
#
# Usage examples:
#   ./scripts/setup_training_env.sh --backend auto --recreate
#   ./scripts/setup_training_env.sh --backend cuda --python 3.11
#   ./scripts/setup_training_env.sh --backend cpu --venv .venv-train-cpu

set -euo pipefail

BACKEND="auto"
PYTHON_VERSION="3.11"
VENV_PATH=".venv-train"
RECREATE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_PATH="$SCRIPT_DIR/requirements_train_v7.txt"

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "Missing requirements file: $REQUIREMENTS_PATH" >&2
  exit 1
fi

if [[ "$VENV_PATH" = /* ]]; then
  VENV_FULL_PATH="$VENV_PATH"
else
  VENV_FULL_PATH="$REPO_ROOT/$VENV_PATH"
fi

if [[ ! -x "$(command -v uv)" ]]; then
  echo "uv is required but not found on PATH. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

resolve_backend() {
  local requested="$1"
  if [[ "$requested" != "auto" ]]; then
    echo "$requested"
    return
  fi

  case "$(uname -s)" in
    Darwin)
      echo "mps"
      ;;
    Linux)
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
      elif command -v rocminfo >/dev/null 2>&1; then
        echo "rocm"
      else
        echo "cpu"
      fi
      ;;
    MINGW*|MSYS*|CYGWIN*)
      if command -v nvidia-smi >/dev/null 2>&1; then
        echo "cuda"
      else
        echo "cpu"
      fi
      ;;
    *)
      echo "cpu"
      ;;
  esac
}

run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[DRYRUN] $*"
    return
  fi
  "$@"
}

RESOLVED_BACKEND="$(resolve_backend "$BACKEND")"
VENV_PYTHON="$VENV_FULL_PATH/bin/python"

echo "[INFO] Repo root: $REPO_ROOT"
echo "[INFO] Requested backend: $BACKEND"
echo "[INFO] Resolved backend: $RESOLVED_BACKEND"
echo "[INFO] Python version: $PYTHON_VERSION"
echo "[INFO] Training venv: $VENV_FULL_PATH"

if [[ $RECREATE -eq 1 && -d "$VENV_FULL_PATH" ]]; then
  echo "[INFO] Recreating existing venv: $VENV_FULL_PATH"
  if [[ $DRY_RUN -eq 0 ]]; then
    rm -rf "$VENV_FULL_PATH"
  fi
fi

run_cmd uv python install "$PYTHON_VERSION"
run_cmd uv venv "$VENV_FULL_PATH" --python "$PYTHON_VERSION"
run_cmd uv pip install --python "$VENV_PYTHON" -r "$REQUIREMENTS_PATH"

case "$RESOLVED_BACKEND" in
  cuda)
    run_cmd uv pip install --python "$VENV_PYTHON" --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
    ;;
  rocm)
    run_cmd uv pip install --python "$VENV_PYTHON" --index-url https://download.pytorch.org/whl/rocm6.2.4 torch torchvision torchaudio
    ;;
  cpu)
    run_cmd uv pip install --python "$VENV_PYTHON" --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
    ;;
  mps)
    run_cmd uv pip install --python "$VENV_PYTHON" torch torchvision torchaudio
    ;;
  *)
    echo "Unsupported backend: $RESOLVED_BACKEND" >&2
    exit 1
    ;;
esac

if [[ $DRY_RUN -eq 0 ]]; then
  "$VENV_PYTHON" - <<PY
import sys
import torch

backend = "${RESOLVED_BACKEND}"
print(f"PYTHON={sys.executable}")
print(f"TORCH={torch.__version__}")
print(f"TORCH_CUDA={torch.version.cuda}")
print(f"TORCH_HIP={getattr(torch.version, 'hip', None)}")
print(f"CUDA_AVAILABLE={torch.cuda.is_available()}")
print(f"MPS_AVAILABLE={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

if backend == "cuda" and not torch.cuda.is_available():
    raise SystemExit("Expected CUDA backend, but torch.cuda.is_available() is False.")
if backend == "rocm" and not bool(getattr(torch.version, "hip", None)):
    raise SystemExit("Expected ROCm backend, but torch.version.hip is not available.")
if backend == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
    raise SystemExit("Expected MPS backend, but torch.backends.mps.is_available() is False.")
PY
else
  echo "[INFO] Dry run complete. Skipped runtime validation."
fi

echo "[OK] Training environment is ready."
echo
echo "Run training with:"
echo "  $VENV_PYTHON $REPO_ROOT/src/WoWMapConverter/scripts/train_v7.py --profile development-map"
echo
echo "If you intentionally need CPU fallback for a debug run, add: --allow-cpu"
