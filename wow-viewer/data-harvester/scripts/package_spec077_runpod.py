"""Build a RunPod-ready Spec 077 training bundle.

The bundle intentionally copies only derived training artifacts:

* teacher-prior Zarr stores
* V18 tensor-pack Zarr stores needed for height/normal/weight targets
* albedo sidecar Zarr stores
* visibility-audit curation manifest
* the Python training code needed to run ``train_height_only_prior.py``

It does not copy staged game clients, MPQs, CASC data, or asset trees.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_BUILDS = ("0_5_3_3368", "3_3_5_12340")

REQUIRED_ARRAYS = {
    "teacher-prior": (
        "processed_minimap_prior_256",
        "teacher_object_mask_256",
        "teacher_object_confidence_256",
    ),
    "v18": (
        "height_257",
        "object_precise_mask",
        "object_filtered_mask",
        "normal_xyz",
        "normal_mask",
    ),
    "albedo": ("albedo_rgb_256",),
}

OPTIONAL_ARRAYS = {
    "teacher-prior": ("raw_minimap_rgb_256",),
    "v18": ("alpha_256", "mcly_texture_ids", "mcly_layer_mask"),
    "albedo": (),
}

CODE_DIRS = ("src",)
CODE_FILES = ("pyproject.toml", "uv.lock", "README.md")
SCRIPT_FILES = (
    "train_height_only_prior.py",
    "train_height_coarse_prior.py",
    "train_height_residual_prior.py",
    "build_albedo_dataset.py",
    "package_spec077_runpod.py",
    "setup_spec077_runpod.py",
)
TEST_FILES = (
    "test_height_only_prior.py",
    "test_height_residual_chain.py",
    "test_terrain_augment.py",
    "test_height_to_normal.py",
    "test_package_spec077_runpod.py",
)

RUNPOD_REQUIREMENTS = """# Runtime dependencies for the Spec 077 cloud bundle.
# Use a RunPod PyTorch/CUDA template so torch is already installed with CUDA.
numpy>=2.0
pyarrow>=24.0.0
zarr>=3.0
numcodecs>=0.13
pillow>=10.0
scipy>=1.12
"""


@dataclass(frozen=True)
class StoreCopy:
    kind: str
    build: str
    source: Path
    relative_dest: Path
    required_arrays: tuple[str, ...]
    optional_arrays: tuple[str, ...]


def _default_wow_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dir_size_bytes(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def _copy_tree(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def _copy_zarr_subset(source: Path, dest: Path, arrays: tuple[str, ...], files: tuple[str, ...] = ()) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for metadata_file in ("zarr.json", ".zgroup", ".zattrs"):
        _copy_file_if_exists(source / metadata_file, dest / metadata_file)
    for array_name in arrays:
        _copy_tree(source / array_name, dest / array_name)
    for filename in files:
        _copy_file_if_exists(source / filename, dest / filename)


def _copy_file_if_exists(source: Path, dest: Path) -> bool:
    if not source.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return True


def _validate_store(store: StoreCopy) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    if not store.source.exists() or not store.source.is_dir():
        errors.append(f"Missing {store.kind} store for build {store.build}: {store.source}")
    else:
        for array_name in store.required_arrays:
            if not (store.source / array_name).exists():
                errors.append(
                    f"Missing required array {array_name} in {store.kind} store for {store.build}: {store.source}"
                )
        for array_name in store.optional_arrays:
            if not (store.source / array_name).exists():
                warnings.append(
                    f"Optional array {array_name} not found in {store.kind} store for {store.build}: {store.source}"
                )
        if store.kind in {"teacher-prior", "albedo"} and not (store.source / "tiles.parquet").exists():
            warnings.append(f"No tiles.parquet found under {store.source}")
    return {
        "kind": store.kind,
        "build": store.build,
        "source": str(store.source),
        "relative_dest": store.relative_dest.as_posix(),
        "bytes": _dir_size_bytes(store.source) if store.source.exists() else 0,
        "required_arrays": list(store.required_arrays),
        "optional_arrays": list(store.optional_arrays),
        "errors": errors,
        "warnings": warnings,
    }


def _store_copies(wow_root: Path, builds: list[str]) -> list[StoreCopy]:
    copies: list[StoreCopy] = []
    for build in builds:
        copies.extend(
            [
                StoreCopy(
                    kind="teacher-prior",
                    build=build,
                    source=wow_root / "output" / "datasets" / "teacher-prior" / f"{build}.zarr",
                    relative_dest=Path("data") / "teacher-prior" / f"{build}.zarr",
                    required_arrays=REQUIRED_ARRAYS["teacher-prior"],
                    optional_arrays=OPTIONAL_ARRAYS["teacher-prior"],
                ),
                StoreCopy(
                    kind="v18",
                    build=build,
                    source=wow_root / "output" / "datasets" / "v18" / f"{build}.zarr",
                    relative_dest=Path("data") / "v18" / f"{build}.zarr",
                    required_arrays=REQUIRED_ARRAYS["v18"],
                    optional_arrays=OPTIONAL_ARRAYS["v18"],
                ),
                StoreCopy(
                    kind="albedo",
                    build=build,
                    source=wow_root / "output" / "datasets" / "albedo" / f"{build}.zarr",
                    relative_dest=Path("data") / "albedo" / f"{build}.zarr",
                    required_arrays=REQUIRED_ARRAYS["albedo"],
                    optional_arrays=OPTIONAL_ARRAYS["albedo"],
                ),
            ]
        )
    return copies


def _estimated_store_copy_bytes(store: StoreCopy, *, full_v18_stores: bool) -> int:
    if not store.source.exists():
        return 0
    if store.kind != "v18" or full_v18_stores:
        return _dir_size_bytes(store.source)
    total = 0
    for metadata_file in ("zarr.json", ".zgroup", ".zattrs"):
        path = store.source / metadata_file
        if path.exists():
            total += _dir_size_bytes(path)
    for array_name in store.required_arrays:
        total += _dir_size_bytes(store.source / array_name)
    return total


def _validate_inputs(
    wow_root: Path,
    builds: list[str],
    curation_manifest: Path,
    *,
    full_v18_stores: bool,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    stores = _store_copies(wow_root, builds)
    store_reports = [_validate_store(store) for store in stores]
    errors: list[str] = []
    warnings: list[str] = []
    for report in store_reports:
        errors.extend(str(item) for item in report["errors"])
        warnings.extend(str(item) for item in report["warnings"])

    if not curation_manifest.exists():
        errors.append(f"Missing curation manifest path: {curation_manifest}")
    elif curation_manifest.is_dir() and not (curation_manifest / "kept_tiles.parquet").exists():
        errors.append(f"Curation manifest directory lacks kept_tiles.parquet: {curation_manifest}")
    elif curation_manifest.is_file() and curation_manifest.name != "kept_tiles.parquet":
        warnings.append(f"Curation manifest is a file that is not named kept_tiles.parquet: {curation_manifest}")

    data_bytes = sum(_estimated_store_copy_bytes(store, full_v18_stores=full_v18_stores) for store in stores)
    if curation_manifest.exists():
        data_bytes += _dir_size_bytes(curation_manifest)
    estimate_kind = "full V18 stores" if full_v18_stores else "slim V18 stores"
    warnings.append(f"Estimated derived-data payload size ({estimate_kind}): {data_bytes / (1024 ** 3):.2f} GiB before archive container overhead")
    return store_reports, errors, warnings


def _copy_code(harvester_root: Path, bundle_root: Path, include_tests: bool) -> dict[str, object]:
    dest_root = bundle_root / "data-harvester"
    dest_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    missing: list[str] = []

    for dirname in CODE_DIRS:
        source = harvester_root / dirname
        if source.exists():
            _copy_tree(source, dest_root / dirname)
            copied.append(dirname + "/")
        else:
            missing.append(dirname + "/")

    scripts_dest = dest_root / "scripts"
    scripts_dest.mkdir(parents=True, exist_ok=True)
    for filename in SCRIPT_FILES:
        if _copy_file_if_exists(harvester_root / "scripts" / filename, scripts_dest / filename):
            copied.append(f"scripts/{filename}")
        else:
            missing.append(f"scripts/{filename}")

    if include_tests:
        tests_dest = dest_root / "tests"
        tests_dest.mkdir(parents=True, exist_ok=True)
        for filename in TEST_FILES:
            if _copy_file_if_exists(harvester_root / "tests" / filename, tests_dest / filename):
                copied.append(f"tests/{filename}")
            else:
                missing.append(f"tests/{filename}")

    for filename in CODE_FILES:
        if _copy_file_if_exists(harvester_root / filename, dest_root / filename):
            copied.append(filename)

    return {"copied": copied, "missing": missing, "include_tests": include_tests}


def _bash_array(values: list[str]) -> str:
    return "\n".join(f'  "{value}"' for value in values)


def _write_runpod_files(bundle_root: Path, builds: list[str], run_name: str, epochs: int, batch_size: int, target_vram_gb: float) -> None:
    runpod_dir = bundle_root / "runpod"
    runpod_dir.mkdir(parents=True, exist_ok=True)

    (bundle_root / "requirements-runpod.txt").write_text(RUNPOD_REQUIREMENTS, encoding="utf-8")

    prior_paths = [f"../data/teacher-prior/{build}.zarr" for build in builds]
    v18_paths = [f"../data/v18/{build}.zarr" for build in builds]
    albedo_paths = [f"../data/albedo/{build}.zarr" for build in builds]

    train_script = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "$ROOT_DIR/data-harvester"
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"

RUN_NAME="${{RUN_NAME:-{run_name}}}"
EPOCHS="${{EPOCHS:-{epochs}}}"
BATCH_SIZE="${{BATCH_SIZE:-{batch_size}}}"
TARGET_VRAM_GB="${{TARGET_VRAM_GB:-{target_vram_gb:g}}}"
NUM_WORKERS="${{NUM_WORKERS:-0}}"
OUTPUT_DIR="${{OUTPUT_DIR:-../models/spec077/height-only/${{RUN_NAME}}}}"

PRIOR_PATHS=(
{_bash_array(prior_paths)}
)
V18_PATHS=(
{_bash_array(v18_paths)}
)
ALBEDO_PATHS=(
{_bash_array(albedo_paths)}
)

python scripts/train_height_only_prior.py \
  --prior "${{PRIOR_PATHS[@]}}" \
  --v18 "${{V18_PATHS[@]}}" \
  --albedo-path "${{ALBEDO_PATHS[@]}}" \
  --curation-manifest "../data/visibility-audit/two_build" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --val-steps 0 \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --normal-guidance-weight 0.10 \
  --hard-error-weight 0.05 \
  --hard-error-power 1.0 \
  --hard-error-max-multiplier 4.0 \
  --albedo \
  --model-norm group \
  --decoder-upsample nearest \
  --autotune-batch-size \
  --target-vram-gb "$TARGET_VRAM_GB" \
  --num-workers "$NUM_WORKERS" \
  --no-persistent-workers \
  "$@"
"""
    (runpod_dir / "train_spec077.sh").write_text(train_script, encoding="utf-8", newline="\n")

    train_h0_script = f"""#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
cd "$ROOT_DIR/data-harvester"
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"

RUN_NAME="${{RUN_NAME:-h0_coarse_albedo_density}}"
EPOCHS="${{EPOCHS:-80}}"
BATCH_SIZE="${{BATCH_SIZE:-16}}"
OUTPUT_DIR="${{OUTPUT_DIR:-../models/spec077/height-residual-chain/h0_coarse}}"
NUM_WORKERS="${{NUM_WORKERS:-0}}"

PRIOR_PATHS=(
{_bash_array(prior_paths)}
)
V18_PATHS=(
{_bash_array(v18_paths)}
)
ALBEDO_PATHS=(
{_bash_array(albedo_paths)}
)

python scripts/train_height_coarse_prior.py \
  --prior "${{PRIOR_PATHS[@]}}" \
  --v18 "${{V18_PATHS[@]}}" \
  --albedo-path "${{ALBEDO_PATHS[@]}}" \
  --curation-manifest "../data/visibility-audit/two_build" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --val-steps 0 \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --albedo \
  --density \
  --model-norm group \
  --decoder-upsample nearest \
  --num-workers "$NUM_WORKERS" \
  --no-persistent-workers \
  "$@"
"""
    (runpod_dir / "train_spec077_h0.sh").write_text(train_h0_script, encoding="utf-8", newline="\n")

    train_h1_script = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/data-harvester"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-h1_residual_albedo_density}"
EPOCHS="${EPOCHS:-160}"
BATCH_SIZE="${BATCH_SIZE:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-../models/spec077/height-residual-chain/h1_residual}"
COARSE_CHECKPOINT="${COARSE_CHECKPOINT:-../models/spec077/height-residual-chain/h0_coarse/h0_coarse_albedo_density_h0_best.pt}"
NUM_WORKERS="${NUM_WORKERS:-0}"

PRIOR_PATHS=(
""" + _bash_array(prior_paths) + """
)
V18_PATHS=(
""" + _bash_array(v18_paths) + """
)
ALBEDO_PATHS=(
""" + _bash_array(albedo_paths) + """
)

python scripts/train_height_residual_prior.py \
  --coarse-checkpoint "$COARSE_CHECKPOINT" \
  --prior "${PRIOR_PATHS[@]}" \
  --v18 "${V18_PATHS[@]}" \
  --albedo-path "${ALBEDO_PATHS[@]}" \
  --curation-manifest "../data/visibility-audit/two_build" \
  --output-dir "$OUTPUT_DIR" \
  --run-name "$RUN_NAME" \
  --epochs "$EPOCHS" \
  --val-steps 0 \
  --batch-size "$BATCH_SIZE" \
  --device cuda \
  --albedo \
  --density \
  --model-norm group \
  --decoder-upsample nearest \
  --gradient-weight 0.05 \
  --normal-guidance-weight 0.10 \
  --hard-error-weight 0.05 \
  --delta-weight 0.25 \
  --num-workers "$NUM_WORKERS" \
  --no-persistent-workers \
  "$@"
"""
    (runpod_dir / "train_spec077_h1.sh").write_text(train_h1_script, encoding="utf-8", newline="\n")

    smoke_script = """#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-smoke_spec077}" \
EPOCHS="${EPOCHS:-1}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
TARGET_VRAM_GB="${TARGET_VRAM_GB:-4}" \
bash "$(dirname "${BASH_SOURCE[0]}")/train_spec077.sh" \
  --steps 2 \
  --val-steps 1 \
  --max-tiles 8 \
  --no-compile \
  --no-amp \
  "$@"
"""
    (runpod_dir / "smoke_spec077.sh").write_text(smoke_script, encoding="utf-8", newline="\n")

    install_script = """#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python -m pip install --upgrade pip
python - <<'PY'
import subprocess
import sys

def parse_version(text: str) -> tuple[int, int]:
    parts = text.split("+", 1)[0].split(".")
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return 0, 0

try:
    import torch
    current = parse_version(torch.__version__)
except Exception:
    current = (0, 0)

if current < (2, 5):
    print("Installing torch>=2.5 from the PyTorch CUDA 12.4 wheel index...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--index-url",
        "https://download.pytorch.org/whl/cu124",
        "torch>=2.5,<3",
    ])
PY
python -m pip install -r requirements-runpod.txt
python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Use a RunPod PyTorch CUDA template before training.")
PY
"""
    (runpod_dir / "install_deps.sh").write_text(install_script, encoding="utf-8", newline="\n")

    verify_py = f"""from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import zarr

ROOT = Path(__file__).resolve().parents[1]
BUILDS = {json.dumps(builds)}
EXPECTED = {{
    "teacher-prior": ["processed_minimap_prior_256", "teacher_object_mask_256", "teacher_object_confidence_256"],
    "v18": ["height_257", "object_filtered_mask", "normal_xyz", "normal_mask"],
    "albedo": ["albedo_rgb_256"],
}}

for kind, arrays in EXPECTED.items():
    for build in BUILDS:
        store_path = ROOT / "data" / kind / f"{{build}}.zarr"
        if not store_path.exists():
            raise SystemExit(f"missing store: {{store_path}}")
        root = zarr.open_group(str(store_path), mode="r")
        for array in arrays:
            if array not in root:
                raise SystemExit(f"missing {{kind}}/{{build}} array: {{array}}")
            print(f"{{kind}}/{{build}}/{{array}} shape={{root[array].shape}} dtype={{root[array].dtype}}")

kept = ROOT / "data" / "visibility-audit" / "two_build" / "kept_tiles.parquet"
if not kept.exists():
    raise SystemExit(f"missing curation manifest: {{kept}}")
table = pq.read_table(str(kept))
print(f"curation rows={{table.num_rows}} columns={{table.column_names}}")
print("bundle verification passed")
"""
    (runpod_dir / "verify_spec077_bundle.py").write_text(verify_py, encoding="utf-8", newline="\n")

    verify_sh = """#!/usr/bin/env bash
set -euo pipefail

python "$(dirname "${BASH_SOURCE[0]}")/verify_spec077_bundle.py"
"""
    (runpod_dir / "verify_bundle.sh").write_text(verify_sh, encoding="utf-8", newline="\n")


def _write_readme(bundle_root: Path, builds: list[str], run_name: str) -> None:
    builds_text = ", ".join(builds)
    readme = f"""# Spec 077 RunPod Training Bundle

This bundle contains the Python training code and derived datasets needed to train the Spec 077 height-only terrain model on RunPod.

It does not contain game clients, MPQs, CASC archives, or raw asset trees. Treat the bundled derived datasets as private BYOD training artifacts and do not redistribute them.

## Contents

- `data-harvester/`: minimal Python source and scripts for training.
- `data/teacher-prior/`: object-suppressed minimap-prior Zarr stores.
- `data/v18/`: V18 tensor-pack Zarr stores used for `height_257`, normals, and object loss gates. Loss gates prefer `object_precise_mask`.
- `data/albedo/`: precomputed `albedo_rgb_256` texture-identity sidecars.
- `data/visibility-audit/two_build/`: `kept_tiles.parquet` curation manifest.
- `runpod/`: setup, verification, smoke, and full-training shell scripts.
- `manifest.json`: source paths, copied bytes, and training command metadata.

Builds included: {builds_text}

## Recommended RunPod Setup

Use a RunPod Pod with a network volume. This is the simplest path for a long training job: the dataset and checkpoints live on persistent storage, and the training process runs like a normal shell command. The local helper `scripts/setup_spec077_runpod.py` creates a network-volume-backed Pod with `NVIDIA RTX 4000 Ada Generation` by default; alternative GPU fallbacks require explicit `--gpu-fallback` or `--gpu-types` opt-in. The helper can bootstrap transfer with `runpodctl send`/`receive`.

Use a PyTorch CUDA template. The generated installer keeps an existing `torch>=2.5` install, or upgrades Torch from the PyTorch CUDA 12.4 wheel index if the template image is older. The local repo `pyproject.toml` includes Windows-only training dependencies, so this bundle uses `requirements-runpod.txt` instead of `uv sync` on the pod.

If you did not use the setup helper's automatic bootstrap, upload and unpack the bundle on the pod, then run:

```bash
cd <unpacked-spec077-bundle-directory>
bash runpod/install_deps.sh
bash runpod/verify_bundle.sh
```

Run a tiny smoke pass first:

```bash
bash runpod/smoke_spec077.sh
```

Run the full current recommended base-model proof:

```bash
bash runpod/train_spec077.sh
```

The default full run is `{run_name}` and uses `--albedo --model-norm group --decoder-upsample nearest`.

If the direct height model plateaus with muddy previews, run the coarse-to-fine residual chain instead:

```bash
bash runpod/train_spec077_h0.sh
bash runpod/train_spec077_h1.sh
```

H0 writes `height_coarse_65` checkpoints under `../models/spec077/height-residual-chain/h0_coarse/`. H1 defaults to the H0 best checkpoint at `../models/spec077/height-residual-chain/h0_coarse/h0_coarse_albedo_density_h0_best.pt` and predicts only `height_delta_257`; override `COARSE_CHECKPOINT=...` if needed.

## Transfer Options

- Default helper path: Pod-side `runpodctl receive <code>` plus local `runpodctl send <archive>.tar --code <code>`.
- Manual large-package path: `rsync -avzP` over SSH into `/workspace/`.
- Manual single-archive path: upload the generated `.tar`, then run `tar -xf <archive>.tar`.
- Shared/repeated runs: keep this bundle and checkpoints on a RunPod network volume, then attach future Pods to that volume.

## Flash And Serverless Notes

RunPod Flash and Serverless can also attach network volumes. That is useful after this bundle is already on a volume, but it is extra machinery for the first training run. Flash runs local `@Endpoint` functions on Serverless workers and mounts network volumes at `/runpod-volume/`; Pods mount network volumes at `/workspace`. If you later want one-command cloud launches, wrap `bash runpod/train_spec077.sh` in a Flash endpoint that reads the unpacked bundle from `/runpod-volume/<bundle-directory>`.

Traditional Serverless workers require a Docker image or GitHub integration plus a handler. That is better for repeatable API-style inference/training jobs, not the first week of interactive model training and preview inspection.

## MCP Notes

RunPod provides an MCP server for managing Pods, templates, volumes, and endpoints through the RunPod API, plus a docs MCP server at `https://docs.runpod.io/mcp`. If you connect the API MCP server, keep the API key outside this bundle and do not commit it. The MCP route can create the Pod/network volume for you, but the training command remains the same: unpack this bundle, run `bash runpod/install_deps.sh`, verify, smoke, then train.

## Useful Overrides

```bash
EPOCHS=240 BATCH_SIZE=8 TARGET_VRAM_GB=24 RUN_NAME=runpod_a40_spec077 bash runpod/train_spec077.sh
```

Append extra trainer flags after the shell script, for example:

```bash
bash runpod/train_spec077.sh --preview-every-epochs 5
```

Outputs are written under `models/spec077/height-only/<run-name>/`. Download at least `*_metrics.json`, `*_best.pt`, `*_latest.pt`, `*_preview.png`, and the `*_validation_previews/` directory.
"""
    (bundle_root / "README_RunPod.md").write_text(readme, encoding="utf-8", newline="\n")


def _copy_data(
    wow_root: Path,
    bundle_root: Path,
    builds: list[str],
    curation_manifest: Path,
    *,
    full_v18_stores: bool,
) -> list[dict[str, object]]:
    copied: list[dict[str, object]] = []
    for store in _store_copies(wow_root, builds):
        dest = bundle_root / store.relative_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        slim_arrays: tuple[str, ...] | None = None
        copied_mode = "full"
        if store.kind == "v18" and not full_v18_stores:
            slim_arrays = store.required_arrays
            copied_mode = "slim-required-arrays"
        if slim_arrays is None:
            _copy_tree(store.source, dest)
        else:
            _copy_zarr_subset(store.source, dest, arrays=slim_arrays)
        copied.append(
            {
                "kind": store.kind,
                "build": store.build,
                "source": str(store.source),
                "dest": store.relative_dest.as_posix(),
                "bytes": _dir_size_bytes(dest),
                "copy_mode": copied_mode,
                "arrays_copied": list(slim_arrays) if slim_arrays is not None else "all",
            }
        )

    curation_dest = bundle_root / "data" / "visibility-audit" / "two_build"
    curation_dest.parent.mkdir(parents=True, exist_ok=True)
    if curation_manifest.is_dir():
        _copy_tree(curation_manifest, curation_dest)
    else:
        curation_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(curation_manifest, curation_dest / "kept_tiles.parquet")
    copied.append(
        {
            "kind": "curation-manifest",
            "build": "two_build",
            "source": str(curation_manifest),
            "dest": "data/visibility-audit/two_build",
            "bytes": _dir_size_bytes(curation_dest),
        }
    )
    return copied


def _write_manifest(
    bundle_root: Path,
    *,
    wow_root: Path,
    builds: list[str],
    code_report: dict[str, object],
    validation_reports: list[dict[str, object]],
    data_copies: list[dict[str, object]],
    warnings: list[str],
    run_name: str,
    epochs: int,
    batch_size: int,
    target_vram_gb: float,
    full_v18_stores: bool,
) -> None:
    payload = {
        "schema": "spec-077-runpod-training-package",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_name": bundle_root.name,
        "source_wow_root": str(wow_root),
        "contains_game_client_files": False,
        "game_client_policy": "Only derived Zarr/parquet training artifacts are copied. Staged clients and archive roots are excluded.",
        "loss_gate_policy": "Height trainers gate loss with object_precise_mask first, then object_filtered_mask, then object_mask.",
        "builds": builds,
        "code": code_report,
        "validated_inputs": validation_reports,
        "copied_data": data_copies,
        "full_v18_stores": bool(full_v18_stores),
        "warnings": warnings,
        "training": {
            "run_name": run_name,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "target_vram_gb": float(target_vram_gb),
            "entrypoint": "bash runpod/train_spec077.sh",
            "smoke_entrypoint": "bash runpod/smoke_spec077.sh",
            "verify_entrypoint": "bash runpod/verify_bundle.sh",
            "coarse_entrypoint": "bash runpod/train_spec077_h0.sh",
            "residual_entrypoint": "bash runpod/train_spec077_h1.sh",
            "model_flags": ["--albedo", "--model-norm", "group", "--decoder-upsample", "nearest"],
        },
    }
    (bundle_root / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_zip(bundle_root: Path) -> Path:
    zip_path = bundle_root.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(bundle_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_root.parent).as_posix())
    return zip_path


def _make_tar(bundle_root: Path) -> Path:
    tar_path = bundle_root.with_suffix(".tar")
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, mode="w") as archive:
        archive.add(bundle_root, arcname=bundle_root.name)
    return tar_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package Spec 077 height-only training for RunPod.")
    parser.add_argument("--wow-root", type=Path, default=_default_wow_root(), help="Path to wow-viewer root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory that will receive the staged bundle and archive.")
    parser.add_argument("--package-name", type=str, default=None, help="Bundle directory/archive name without extension.")
    parser.add_argument("--builds", nargs="+", default=list(DEFAULT_BUILDS), help="Build IDs to include.")
    parser.add_argument("--curation-manifest", type=Path, default=None, help="Path to kept_tiles.parquet or its parent directory.")
    parser.add_argument("--archive-format", choices=["zip", "tar", "none"], default="tar")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--validate-only", action="store_true", default=False,
                        help="Validate required local inputs and print payload estimate without copying data.")
    parser.add_argument("--full-v18-stores", action="store_true", default=False,
                        help="Copy entire V18 stores instead of the minimal arrays required for training.")
    parser.add_argument("--no-tests", action="store_true", default=False, help="Do not include focused pytest files in the bundle.")
    parser.add_argument("--run-name", type=str, default="cuda_albedo_group_nearest")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-vram-gb", type=float, default=12.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    wow_root = args.wow_root.resolve()
    harvester_root = wow_root / "data-harvester"
    if not harvester_root.exists():
        print(f"data-harvester root not found under --wow-root: {harvester_root}", file=sys.stderr)
        return 2

    builds = [str(build) for build in args.builds]
    output_root = (args.output_root or (wow_root / "output" / "cloud-packages")).resolve()
    package_name = args.package_name or f"spec077_runpod_bundle_{_utc_stamp()}"
    bundle_root = output_root / package_name
    curation_manifest = (
        args.curation_manifest.resolve()
        if args.curation_manifest is not None
        else (wow_root / "output" / "analysis" / "teacher-prior" / "visibility-audit" / "two_build").resolve()
    )

    validation_reports, errors, warnings = _validate_inputs(
        wow_root,
        builds,
        curation_manifest,
        full_v18_stores=bool(args.full_v18_stores),
    )
    if errors:
        print("Cannot build RunPod package; required inputs are missing:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    if args.validate_only:
        print("Spec 077 RunPod package inputs are present.")
        for warning in warnings:
            print(f"warning: {warning}")
        return 0

    if bundle_root.exists():
        if not args.overwrite:
            print(f"Bundle already exists: {bundle_root} (pass --overwrite to replace)", file=sys.stderr)
            return 2
        shutil.rmtree(bundle_root)
    output_root.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=False)

    try:
        code_report = _copy_code(harvester_root, bundle_root, include_tests=not args.no_tests)
        data_copies = _copy_data(
            wow_root,
            bundle_root,
            builds,
            curation_manifest,
            full_v18_stores=bool(args.full_v18_stores),
        )
        _write_runpod_files(bundle_root, builds, args.run_name, args.epochs, args.batch_size, args.target_vram_gb)
        _write_readme(bundle_root, builds, args.run_name)
        _write_manifest(
            bundle_root,
            wow_root=wow_root,
            builds=builds,
            code_report=code_report,
            validation_reports=validation_reports,
            data_copies=data_copies,
            warnings=warnings,
            run_name=args.run_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            target_vram_gb=args.target_vram_gb,
            full_v18_stores=bool(args.full_v18_stores),
        )
        archive_path = None
        if args.archive_format == "zip":
            archive_path = _make_zip(bundle_root)
        elif args.archive_format == "tar":
            archive_path = _make_tar(bundle_root)
    except Exception:
        if bundle_root.exists():
            shutil.rmtree(bundle_root, ignore_errors=True)
        raise

    print(f"Wrote RunPod bundle directory: {bundle_root}")
    if archive_path is not None:
        print(f"Wrote RunPod archive: {archive_path}")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
