# Requirements & Verification Checklist: Spec 263

**Feature**: [Spec 263: 1.60 MCCV Terrain Shadow Ground-Truth Validation & Bidirectional Renderer Synthesis](../spec.md)  
**Created**: 2026-10-07  

---

## 1. Specification Quality

- [x] **Clear Technical Contract**: Specifies 145-vertex MCNK MCCV chunk sub-chunk binary layout.
- [x] **No God-Class Growth**: Adheres strictly to `AGENTS.md` §10 (zero new members in `WorldScene.cs` or `ViewerApp.cs`).
- [x] **Core Library First**: Extractor and serializer implemented in `WowViewer.Core.IO`.
- [x] **Empirical Validation**: Quantifiable metrics (NCC $\ge 0.70$, Ridge Coincidence $\ge 75\%$).

---

## 2. Acceptance Criteria Verification Matrix

| AC ID | Description | Target Metric | Verification Method |
|---|---|---|---|
| **AC-001** | MCCV Extraction & 145-Vertex Grid | $\ge 99\%$ Coordinate Precision | `dotnet test --filter "MccvTerrainShadowServiceTests"` |
| **AC-002** | 1.12 vs 1.60 Shadow NCC Correlation | $\text{NCC} \ge 0.70$ on Tier 1 slopes (Westfall, Elwynn, STV) | `uv run python scripts/v60_compare_mccv_residuals.py --tile 27_49` |
| **AC-003** | Ridge Coincidence with Crease Minima | $\ge 75\%$ spatial overlap in Tier 1 zones | `uv run python scripts/v60_compare_mccv_residuals.py --tile 32_55` |
| **AC-004** | 1.60 MCCV Synthesis Roundtrip | Bit-exact 580 bytes BGRA per chunk | `dotnet test --filter "SynthesizeMccvChunks"` |
| **AC-005** | Terrain Shader Neutrality | Neutral gray (127) yields $1.0\times$ multiplier | `dotnet test --filter "TerrainShaderMccv"` |
| **AC-006** | Governance & Receipts | Complete receipt in `evidence/receipt-spec263.md` | Audit inspection per `AGENTS.md` §9.2 |

---

## 3. Geographic Sampling Stratification
- **Tier 1 (Trusted Baseline)**: Westfall (`27..30, 48..51`), Elwynn (`30..33, 47..50`), Stranglethorn Vale (`30..35, 52..58`). Pristine terrain lineage without structural overhauls.
- **Tier 2 (Excluded Mashup Zones)**: Capital Cities (Stormwind, Ironforge, Orgrimmar, Undercity), Wetlands, Badlands/Redridge, Thousand Needles. Excluded from baseline gate due to composite modern geometry revisions.
