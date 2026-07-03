# Contract: V23 Checkpoint

**Phase 1 contract. Companion to `plan.md`, `research.md`, and `data-model.md`.**

This contract defines the minimum metadata and tensor state a V23 checkpoint must persist.

## Shape

```text
V23Checkpoint
├── config
├── model_state
├── optimizer_state
├── epoch
├── commit_sha
├── environment
└── data_hashes
```

## Required Fields

### `config`

- `dataset_dir`
- `builds`
- `input_mode`
- `batch_size`
- `gpct_k`
- `gpct_weight`
- `sdc_weight`
- `bias_free_mask_ratio`
- `deterministic`
- `seed`

### `environment`

- `torch_version`
- `cuda_version`
- `device_name`
- `bf16_enabled`
- `bitsandbytes_version`
- `transformers_version`
- `peft_version`

### `data_hashes`

- V22 store hash or manifest hash
- tileset prune-table hash
- optional training/validation split hash

## Invariants

- A second run with the same `config`, `commit_sha`, `environment`, and `data_hashes` is the determinism proof target.
- The checkpoint must be loadable without opening any game client path.
- Checkpoints persist only V23 training state; they do not embed raw V22 arrays.

## Producer

- `harvester.v23.checkpoint.save_checkpoint`
- `scripts/train_v23_height.py`

## Consumer

- `scripts/train_v23_height.py --resume-checkpoint`
- `scripts/infer_v23_height.py`
