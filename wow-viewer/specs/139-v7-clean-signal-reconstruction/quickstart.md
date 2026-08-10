# Quickstart: V7-Inspired Clean-Signal Terrain Reconstruction

All commands are PowerShell-ready. Training and corpus generation remain user-owned.

## 0. Current CPU contract proof

The implemented foundational slice can be checked without a corpus, client, CUDA device, or model
run:

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run --no-cache python -m pytest tests/v60/test_clean_signal_inputs.py tests/v60/test_clean_signal_targets.py tests/v60/test_clean_signal_contract.py -q --basetemp "I:/parp/parp-tools/output/tmp/pytest-v60-clean-signal-user"
```

The current proof is 15 passing tests. Use a fresh writable `--basetemp` path on Windows if a
previous pytest process still owns files in an older temp directory.

## 1. Validate the design inputs

```powershell
Set-Location "I:/parp/parp-tools/wow-viewer/data-harvester"
uv run --no-cache python scripts/v60_validate_clean_signal_corpus.py --corpus "../output/datasets/v60/v7-clean-signal-v1"
```

The validator must report finite four-channel observations, exact target hashes, complete required
families, and zero forbidden inference signals.

## 2. Print the synthetic training plan

```powershell
uv run --no-cache python scripts/v60_train_clean_signal.py --corpus "../output/datasets/v60/v7-clean-signal-v1" --output "../output/datasets/v60/v7-clean-signal-runs/pyramid-structural-v1" --architectures "pyramid_cnn,segformer_b0,unet_lite_v2" --loss-profiles "parity,v7_structural_v1" --split within_family --train-size 32 --epochs 80 --batch-size 8 --seed 7137
```

Without `--confirm-run`, this prints parameter counts, split identities, loss weights, and the
forbidden-signal audit, then exits without training.

## 3. User-owned training

After inspecting the dry run, add `--confirm-run`. Use a fresh output directory for every matrix
cell. The first meaningful matrix is the within-family learnability split. Only after a structural
loss/architecture cell clears that gate should the user run the complete-family transfer split.

## 4. Real transfer

```powershell
uv run --no-cache python scripts/v60_transfer_clean_signal.py --checkpoint "../output/datasets/v60/v7-clean-signal-runs/pyramid-structural-v1/pyramid_cnn/v7_structural_v1/checkpoint_best.pt" --normalized-corpus "../output/datasets/v60/albedo-accepted-0x-1x-v1" --output "../output/datasets/v60/v7-clean-signal-transfer-v1"
```

The transfer command must read only accepted normalized observation packages. It writes a forbidden-
signal audit and a `hold`/`diagnose`/`expand` decision; synthetic success alone never authorizes
broad real-data processing.
