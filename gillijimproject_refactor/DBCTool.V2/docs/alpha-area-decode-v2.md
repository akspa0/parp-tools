# Alpha Area Decode V2 (DBCTool.V2)

- Output root: dbctool_outputs/<session>/...
- Audit CSVs:
  - dbctool_outputs/<session>/compare/alpha_areaid_decode_v2.csv
  - dbctool_outputs/<session>/compare/alpha_areaid_anomalies.csv
- No confidence tags are surfaced in CSVs.

Integration steps:
1) Decode hi16/lo16 halves from Alpha AreaNumber and validate ParentAreaNum == zoneBase.
2) Build per-continent ZoneIndex/SubIndex, resolve cross-continent ownership per zoneBase.
3) Build deterministic chains:
   - lo16 == 0 -> [zone]
   - lo16 > 0 with sub hit -> [zone, sub]
   - lo16 > 0 without sub hit -> [zone]
4) Map-locked exact matching against LK indices (no cross-map results).
