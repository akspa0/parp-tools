# Contracts: Minimap DXT1 Artifact Inversion

**Phase 1 output** | **Date**: 2026-08-02 | **Spec**: [spec.md](./spec.md)

This directory holds the interface contracts for the feature. These are C# library contracts (not
REST endpoints — this is a CLI/library feature), expressed as signatures and CLI flags.

## Files

- [dxt1-tile-codec.md](./dxt1-tile-codec.md) — the DXT1 encode/decode cycle + round-trip check.
- [lighting-baseline.md](./lighting-baseline.md) — the per-map lighting-baseline survey.
- [encoding-survey.md](./encoding-survey.md) — the per-build/map encoding distribution survey.
- [cli-flags.md](./cli-flags.md) — the `synthetic-minimap` CLI surface.
- [restoration.md](./restoration.md) — the restoration model contract.
