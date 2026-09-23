# Contract: Restoration Model

**Library**: `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` | **Script**: `scripts/train_v20_dxt1_restore.py`

## Purpose

Restore an authored tile toward its pre-compression appearance (FR-007..FR-012). A single residual
network predicting (pristine − encoded), trained on locally generated pairs.

## Model

- **Input**: encoded tile (after DXT1 cycle), 256×256×3.
- **Output**: residual (pristine − encoded), 256×256×3.
- **Architecture**: small residual network, one output, own checkpoint (constitution IV).
- **Training data**: locally generated pristine→encoded pairs; no authored reference required
  (FR-007).

## Inference / Verdict

```python
def restore(encoded: np.ndarray, model) -> np.ndarray:
    """Returns restored tile = encoded + model(encoded)."""

def verdict(pristine, encoded, restored) -> RestorationVerdict:
    """Improvement over encoded, re-encode agreement, hallucination fraction."""
```

## Gates (FR-008..FR-012)

- **FR-008**: evaluated on held-out pristine images whose originals are known; reports improvement
  over the un-restored encoded input.
- **FR-009**: verifiable against authored tiles by re-encoding output and measuring agreement with
  the authored source, within a stated tolerance.
- **FR-010**: reports a hallucination measure (unsupported detail); MUST NOT be promoted without it
  meeting a stated gate.
- **FR-011**: undamaged input returned substantially unchanged (<2% colour error, SC-006).
- **FR-012**: native-resolution restoration separable from any resolution increase; not evaluated by
  a shared metric.

## Success Criteria

- SC-004: ≥25% colour-error reduction vs encoded input on held-out pristine images; block-seam
  discontinuity within 15% of original.
- SC-005: re-encoding a restored authored tile reproduces the authored source within tolerance for
  ≥90% of tiles.
- SC-006: undamaged input changed by <2% colour error.
