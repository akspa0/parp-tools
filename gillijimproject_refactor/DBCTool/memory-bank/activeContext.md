# Active Context

- Current Focus:
  - Area remapping pipeline: map early builds (0.5.3, 0.5.5) to 3.3.5 AreaIDs using a crosswalked map domain + robust name matching.
  - Deterministic remap flow: discover → export remap JSON → apply remap.
  - Fix MapId → Name reports to ensure parent map names are stable (Directory-first, then fallbacks).

- Recent Results (0.5.3 → 3.3.5):
  - Matching summary: `name=477, unmatched=1, ambiguous=0, skipped_dev=10`.
  - Exported `defs/053_to_335.remap.json` with aliases, explicit maps, ignore list, and options.
  - Alias/variant layer (examples):
    - Demonic Stronghold → Dreadmaul Hold
    - Dark Portal → The Dark Portal
    - Shadowfang → Shadowfang Keep
    - Lik’ash Tar Pits → Lakkari Tar Pits
    - Kargathia Outpost → Kargathia Keep
    - Toggle leading “The ” variants (e.g., The Wellspring River)
  - DO NOT USE targets excluded by default; can be overridden with `--allow-do-not-use`.

- Matching Strategy (summary):
  - Build map crosswalk by Directory (fallback to name) to get a 3.3.5 map bias.
  - Match Areas by name only (parent-agnostic), using:
    - Exact across variants (map-biased, then global)
    - Fuzzy within map (Levenshtein), then global fallback
  - Dev placeholders are filtered from unmatched stats: `***On Map Dungeon***`, Programmer Isle, Plains of Snow, Jeff Quadrant.

- Next Steps:
  1. Fix `MapId_to_Name_{0.5.3|0.5.5}.csv` generation so Directory/InternalName/Name fallbacks produce correct labels for all MapIDs.
  2. Run 0.5.5 → 3.3.5 mapping; export `defs/055_to_335.remap.json`.
  3. Re-check unmatched for 0.5.5 and 0.5.3 with corrected map-name reports; add aliases/explicit maps if still needed.
  4. Integrate patch CSV into ADT conversion (join `src_areaNumber` → write `tgt_areaID`).
