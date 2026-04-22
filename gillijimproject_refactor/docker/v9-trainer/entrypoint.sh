#!/bin/sh
set -eu

log() {
  printf '%s\n' "[v9-entrypoint] $*"
}

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

V9_REPO_ROOT="${V9_REPO_ROOT:-/opt/parp-tools/gillijimproject_refactor}"
V9_SCRIPT_DIR="${V9_SCRIPT_DIR:-$V9_REPO_ROOT/src/WoWMapConverter/scripts}"
V9_TRAINER_SCRIPT="${V9_TRAINER_SCRIPT:-$V9_SCRIPT_DIR/train_v9_optimized.py}"
V9_BUNDLE_ROOT="${V9_BUNDLE_ROOT:-/workspace/data/v9_bundle}"
V9_RUN_NAME="${V9_RUN_NAME:-v9_runpod_run}"
V9_OUTPUT_ROOT="${V9_OUTPUT_ROOT:-/workspace/runs}"
V9_OUTPUT_DIR="${V9_OUTPUT_DIR:-$V9_OUTPUT_ROOT/$V9_RUN_NAME}"
V9_TRAIN_MANIFEST="${V9_TRAIN_MANIFEST:-$V9_BUNDLE_ROOT/manifests/train_manifest.json}"
V9_DEV_EVAL_MANIFEST="${V9_DEV_EVAL_MANIFEST:-$V9_BUNDLE_ROOT/manifests/dev_holdout_manifest.json}"

mkdir -p "$V9_OUTPUT_DIR" "$(dirname "$V9_BUNDLE_ROOT")"

if [ ! -f "$V9_TRAIN_MANIFEST" ] && [ -n "${V9_BUNDLE_DOWNLOAD_URL:-}" ]; then
  log "training manifest not found at $V9_TRAIN_MANIFEST; downloading bundle archive"
  python - <<'PY'
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse


def detect_archive_name(url: str) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    return candidate or "v9_bundle_archive"


def archive_mode(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".tar") or lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        return "tar"
    raise SystemExit(f"Unsupported bundle archive format: {path}")


def extract_archive(archive_path: Path, destination: Path) -> None:
    mode = archive_mode(archive_path)
    destination.mkdir(parents=True, exist_ok=True)
    if mode == "zip":
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
        return
    with tarfile.open(archive_path, mode="r:*") as archive:
        archive.extractall(destination)


def find_bundle_root(extracted_root: Path) -> Path:
    direct = extracted_root / "manifests"
    if direct.exists():
        return extracted_root

    children = [child for child in extracted_root.iterdir() if child.is_dir()]
    if len(children) == 1 and (children[0] / "manifests").exists():
        return children[0]

    raise SystemExit(
        f"Could not locate a bundle root under {extracted_root}. Expected a manifests/ directory."
    )


bundle_root = Path(os.environ["V9_BUNDLE_ROOT"])
download_url = os.environ["V9_BUNDLE_DOWNLOAD_URL"]
header_values = json.loads(os.environ.get("V9_BUNDLE_HEADERS_JSON", "[]"))
auth_header = os.environ.get("V9_BUNDLE_AUTH_HEADER", "").strip()

header_items = {}
for header in header_values:
    if ":" not in header:
        raise SystemExit(f"Invalid V9_BUNDLE_HEADERS_JSON header entry: {header}")
    key, value = header.split(":", 1)
    header_items[key.strip()] = value.strip()
if auth_header:
    if ":" not in auth_header:
        raise SystemExit("V9_BUNDLE_AUTH_HEADER must be formatted as 'Header-Name: value'.")
    key, value = auth_header.split(":", 1)
    header_items[key.strip()] = value.strip()

download_name = detect_archive_name(download_url)
with tempfile.TemporaryDirectory(prefix="v9_bundle_") as temp_dir_text:
    temp_dir = Path(temp_dir_text)
    archive_path = temp_dir / download_name
    extract_root = temp_dir / "extract"

    request = urllib.request.Request(download_url, headers=header_items)
    with urllib.request.urlopen(request) as response, archive_path.open("wb") as output:
        shutil.copyfileobj(response, output)

    extract_archive(archive_path, extract_root)
    extracted_bundle_root = find_bundle_root(extract_root)
    shutil.copytree(extracted_bundle_root, bundle_root, dirs_exist_ok=True)
PY
fi

if [ ! -f "$V9_TRAIN_MANIFEST" ]; then
  log "missing training manifest: $V9_TRAIN_MANIFEST"
  exit 1
fi

if [ ! -f "$V9_TRAINER_SCRIPT" ]; then
  log "missing trainer script: $V9_TRAINER_SCRIPT"
  exit 1
fi

log "starting v9 training run '$V9_RUN_NAME'"
log "training manifest: $V9_TRAIN_MANIFEST"
log "output dir: $V9_OUTPUT_DIR"

python - <<'PY'
import json
import os
import sys
from pathlib import Path

trainer_script = Path(os.environ["V9_TRAINER_SCRIPT"])
train_manifest = os.environ["V9_TRAIN_MANIFEST"]
output_dir = os.environ["V9_OUTPUT_DIR"]
dev_manifest = os.environ.get("V9_DEV_EVAL_MANIFEST", "")
extra_args = json.loads(os.environ.get("V9_TRAINER_ARGS_JSON", "[]"))

if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
    raise SystemExit("V9_TRAINER_ARGS_JSON must be a JSON array of strings.")

command = [
    sys.executable,
    str(trainer_script),
    train_manifest,
    "--output-dir",
    output_dir,
]
if dev_manifest and Path(dev_manifest).exists():
    command.extend(["--dev-eval-cache-manifest", dev_manifest])
command.extend(extra_args)

os.chdir(trainer_script.parent)
os.execv(sys.executable, command)
PY
