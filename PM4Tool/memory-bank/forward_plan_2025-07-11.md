# Forward Plan – PM4Tool (2025-07-11)

> This document captures the **next actionable roadmap** so progress is not lost during long chat sessions. It complements `activeContext.md` and will be updated whenever priorities change.

---

## 1  OBJ Export Parity
1. Restore legacy MSPV/MSVT vertex-transform equations and bounding-box logic in `ObjExporter`.
2. Re-run SHA-based parity tests on a representative tile set.
3. Investigate any mismatches; iterate until **byte-level parity** achieved.
4. When parity is confirmed, mark OBJ exporter milestone complete in `progress.md`.

## 2  Link-Graph Analyzer Enhancements (Phase A-2)
1. 🔍 **Reference Frequency Summary** – emit `*_mslk_missing_refs_summary.csv` with columns `Reference,Flag,GroupId,Occurrences` (distinct per group).
2. 📊 **Flag Pivot Table** – tiny CSV showing total counts per `Unknown_0x00` flag to visualise spread (e.g. flag 2 vs flag 1/4/10).
3. 🗜️ **Distinct-Only Variant** – optional `--distinct` CLI flag to write only unique refs, reducing CSV size.
4. 🗺️ **Hotspot Mapping** – correlate high-frequency references to MSUR surface ranges & MPRL secondary links; emit diagnostics if a hotspot crosses tile boundaries.
5. 🧩 **Vertical Link Analysis** – examine flag 10 entries; hypothesise they correspond to vertical connectivity or portals, and cross-match with elevation data (MSUR normals / heights).
6. ⚙️ **CLI QoL** – add `--max-rows` & `--no-progress` options to aid scripting; update README.

## 3  Core.v2 Port & Refactor (Phase B)
1. Extract legacy PM4 export/loader logic into `WoWToolbox.Core.v2` ensuring chunk fidelity.
2. Add unit tests for each chunk read/write path.
3. Replace old Core.v1 dependencies in applications.

## 4  House-Keeping & Backlog
* Address package security warnings (low priority).
* Automate CI job to run analyzer nightly on `development_*` dataset and push CSVs to artefact store.
* Document newly confirmed chunk semantics in `mslk_linkage_reference.md`.

---

_Last updated: 2025-07-11 21:36_
