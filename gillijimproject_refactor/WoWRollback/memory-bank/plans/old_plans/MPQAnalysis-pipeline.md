Plan: MPQ → Temp Extraction (ADT/BLP) For Reliable Analysis
Goal: Ensure ADT placements and minimap processing by extracting needed assets from MPQs to a temporary folder, then running the validated on-disk tooling.
Scope:
Extract World/Maps/<map>/<map>.wdt and all ADTs (<map>_X_Y.adt, prefer <map>_X_Y_obj0.adt).
Optionally extract minimap BLPs under textures/Minimap/** (hashed/plain).
Algorithm:
Build PrioritizedArchiveSource(clientRoot, ArchiveLocator.LocateMpqs(clientRoot)).
Enumerate tiles:
Wildcard first if listfile present.
Else brute-force 64x64, prefer _obj0.adt, fallback to base.
Extract to temp: Path.GetTempPath()/wowrb/<map>/<timestamp>/World/Maps/<map>/.
WDT name tables:
Parse MDNM/MONM from <map>.wdt for model/WMO names if ADT-local MMDX/MMID or MWMO/MWID are empty.
Run placements analysis from disk:
Use existing file-based readers or a tolerant chunk walker to read MDDF/MODF and emit placements CSV at --out.
Why:
Avoid StormLib listfile and in-memory quirks.
Leverage proven on-disk readers to get immediate results.
Deliverables:
analysis_output/<map>_placements.csv (non-empty).
Optionally extracted minimaps for viewer.
Next Steps:
Implement ExtractToTempAndAnalyze(src, mapName, outPath) in WoWRollback.AnalysisModule.
Add a CLI switch to use temp extraction path by default for MPQ analysis.
Verify CSV contents; if empty, add targeted tile scanning + _obj1.adt probing.