using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.PM4;
using WoWRollback.PM4Module.Services;
using GillijimProject.WowFiles.Wl;

namespace WoWRollback.PM4Module
{
    public class PipelineService
    {
        private readonly Pm4ModfReconstructor _reconstructor;
        private readonly MuseumAdtPatcher _adtPatcher;
        
        public PipelineService()
        {
            _reconstructor = new Pm4ModfReconstructor();
            _adtPatcher = new MuseumAdtPatcher();
        }

        public void Execute(string gamePath, string listfilePath, string pm4Path, string splitAdtPath, string museumAdtPath, string outputRoot, string? wdlPath = null, string? wmoFilter = null, bool useFullMesh = false)
        {
            Console.WriteLine("=== Parsing Patch Pipeline ===\n");

            var dirs = new
            {
                Root = outputRoot,
                WmoLib = Path.Combine(outputRoot, "wmo_library"),
                ModfCsv = Path.Combine(outputRoot, "modf_csv"),
                MddfCsv = Path.Combine(outputRoot, "mddf_csv"),
                WdlUnpainted = Path.Combine(outputRoot, "WDL_to_ADT_unpainted"),
                WdlPainted = Path.Combine(outputRoot, "WDL_to_ADT_painted"),
                MuseumCopy = Path.Combine(outputRoot, "Museum_ADTs"),
                PatchedMuseum = Path.Combine(outputRoot, "Patched_Museum_ADTs"),
                PatchedWdlUnpainted = Path.Combine(outputRoot, "Patched_WDL_ADTS_unpainted"),
                PatchedWdlPainted = Path.Combine(outputRoot, "Patched_WDL_ADTS_painted"),
                FinalAssembly = Path.Combine(outputRoot, "World", "Maps", "development")
            };

            // Ensure directories exist
            Directory.CreateDirectory(dirs.WmoLib);
            Directory.CreateDirectory(dirs.ModfCsv);
            Directory.CreateDirectory(dirs.MddfCsv);
            Directory.CreateDirectory(dirs.WdlUnpainted);
            Directory.CreateDirectory(dirs.WdlPainted);
            Directory.CreateDirectory(dirs.MuseumCopy);
            Directory.CreateDirectory(dirs.PatchedMuseum);
            Directory.CreateDirectory(dirs.PatchedWdlUnpainted);
            Directory.CreateDirectory(dirs.PatchedWdlPainted);
            Directory.CreateDirectory(dirs.FinalAssembly);

            // Stage 1: WMO Extraction
            // User requested full rebuild support.
            // Stage 1 & 1.5: In-Memory WMO Processing & Conversion
            Console.WriteLine("\n[Stage 1 & 1.5] Processing WMOs (In-Memory)...");

            if (!string.IsNullOrEmpty(gamePath) && !string.IsNullOrEmpty(listfilePath))
            {
                var mpqFiles = Directory.GetFiles(Path.Combine(gamePath, "Data"), "*.MPQ", SearchOption.AllDirectories);
                using var archiveSource = new MpqArchiveSource(mpqFiles);
                var wmoConverter = new WmoWalkableSurfaceExtractor();
                
                Console.WriteLine("[INFO] Reading listfile...");
                var wmoEntries = File.ReadLines(listfilePath)
                    .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    // Filter out group files (e.g. "Name_000.wmo")
                    // Root names usually don't end in _###.wmo unless the name itself ends in digits which is rare but possible.
                    // Standard convention is strictly _###.wmo for groups.
                    .Where(l => !System.Text.RegularExpressions.Regex.IsMatch(l, @"_\d{3}\.wmo$"))
                    .Select(l => l.Replace('/', '\\'))
                    .ToList();
                
                Console.WriteLine($"[INFO] Found {wmoEntries.Count} Root WMOs to process.");
                
                int processed = 0;
                int skipped = 0;
                
                // Group loader lambda for In-Memory reading
                // variable capture of archiveSource is safe here as it's disposed after loop
                byte[]? GroupLoader(string path)
                {
                    if (archiveSource.FileExists(path))
                    {
                        using var s = archiveSource.OpenFile(path);
                        if (s == null) return null;
                        using var ms = new MemoryStream();
                        s.CopyTo(ms);
                        return ms.ToArray();
                    }
                    return null;
                }

                foreach (var wmoPath in wmoEntries)
                {
                    try
                    {
                        if (!archiveSource.FileExists(wmoPath)) continue;

                        // Check if OBJ output already exists - skip if so
                        var wmoFileName = Path.GetFileName(wmoPath);
                        var relativeDir = Path.GetDirectoryName(wmoPath);
                        var objOutputDir = Path.Combine(dirs.WmoLib, relativeDir ?? "", wmoFileName);
                        
                        if (Directory.Exists(objOutputDir) && Directory.GetFiles(objOutputDir, "*.obj").Length > 0)
                        {
                            skipped++;
                            continue; // Already extracted, skip
                        }

                        byte[]? rootData = null;
                        using (var s = archiveSource.OpenFile(wmoPath))
                        {
                             if (s == null) continue;
                             using (var ms = new MemoryStream()) { s.CopyTo(ms); rootData = ms.ToArray(); }
                        }
                        
                        var data = WmoWalkableSurfaceExtractor.ExtractFromBytes(rootData, wmoPath, GroupLoader);
                        
                        if (data.GroupCount > 0)
                        {
                            WmoWalkableSurfaceExtractor.ExportPerFlag(data, objOutputDir);
                            processed++;
                            
                            if (processed % 10 == 0) Console.Write($"\r[INFO] Processed {processed} WMOs (skipped {skipped} existing)...");
                        }
                    }
                    catch (Exception ex)
                    {
                        // Log verbose only on specific request to avoid spam
                        // Console.WriteLine($"[WARN] Failed {wmoPath}: {ex.Message}");
                    }
                }
                Console.WriteLine($"\n[INFO] WMO Processing Complete. Processed: {processed}, Skipped (already exist): {skipped}");
            }
            else
            {
                Console.WriteLine("[WARN] Game path or listfile missing, skipping WMO processing.");
            }

            // Stage 2: Direct PM4 Processing
            Console.WriteLine("\n[Stage 2] Direct PM4 Processing...");
            
            var modfCsvPath = Path.Combine(dirs.ModfCsv, "modf_entries.csv");
            var mwmoPath = Path.Combine(dirs.ModfCsv, "mwmo_names.csv");
            
            List<Pm4ModfReconstructor.ModfEntry> transformedEntries = new();
            List<string> wmoNames = new();
            
            // Check for existing MODF CSVs with actual data (more than just headers)
            bool hasExistingModf = File.Exists(modfCsvPath) && new FileInfo(modfCsvPath).Length > 200;
            bool hasExistingMwmo = File.Exists(mwmoPath) && new FileInfo(mwmoPath).Length > 50;
            
            if (hasExistingModf && hasExistingMwmo)
            {
                // REUSE existing MODF/MWMO data
                Console.WriteLine($"[INFO] Found existing MODF data, reusing: {modfCsvPath}");
                (transformedEntries, wmoNames) = LoadExistingModfCsv(modfCsvPath, mwmoPath);
                Console.WriteLine($"[INFO] Loaded {transformedEntries.Count} MODF entries, {wmoNames.Count} WMOs from cache");
            }
            else
            {
                // Full PM4 matching pipeline: parse PM4 → match WMO geometry → generate MODF
                Console.WriteLine("[INFO] Running PM4 → WMO matching pipeline...");
                
                // Step 1: Load PM4 objects directly from .pm4 files
                var pm4Objects = LoadPm4ObjectsFromFiles(pm4Path);
                
                if (pm4Objects.Count == 0)
                {
                    Console.WriteLine("[WARN] No PM4 objects found. Ensure pm4 path contains .pm4 files.");
                    Console.WriteLine("[WARN] Continuing without PM4 MODF data...");
                }
                else
                {
                    // Step 2: Build WMO reference library for matching
                    Console.WriteLine("[INFO] Building WMO reference library...");
                    if (!string.IsNullOrEmpty(wmoFilter))
                        Console.WriteLine($"[INFO] Filtering WMOs by path containing: {wmoFilter}");
                    if (useFullMesh)
                        Console.WriteLine("[INFO] Using full WMO mesh for matching (not just walkable surfaces)");
                    var wmoLibrary = _reconstructor.BuildWmoLibrary(gamePath, listfilePath, outputRoot, wmoFilter, useFullMesh);
                    
                    if (wmoLibrary.Count == 0)
                    {
                        Console.WriteLine("[WARN] WMO library is empty. Check game path and listfile.");
                        Console.WriteLine("[WARN] Continuing without PM4 MODF data...");
                    }
                    else
                    {
                        // Step 3: Match PM4 objects to WMOs and reconstruct MODF
                        Console.WriteLine("[INFO] Matching PM4 objects to WMOs...");
                        var result = _reconstructor.ReconstructModf(pm4Objects, wmoLibrary, 0.88f);
                        
                        // PM4 data is already transformed to ADT coords in LoadPm4ObjectsFromFiles
                        // so the result positions are already in correct coordinate space
                        transformedEntries = result.ModfEntries;
                        wmoNames = result.WmoNames;
                        
                        // Export CSVs for future cache
                        _reconstructor.ExportToCsv(result with { ModfEntries = transformedEntries }, modfCsvPath);
                        _reconstructor.ExportMwmoNames(result, mwmoPath);
                        
                        var candidatesPath = Path.Combine(dirs.ModfCsv, "match_candidates.csv");
                        _reconstructor.ExportCandidatesCsv(result, candidatesPath);
                        
                        Console.WriteLine($"[INFO] Matched {transformedEntries.Count} MODF entries, {wmoNames.Count} WMOs");
                        Console.WriteLine($"[INFO] Unmatched PM4 objects: {result.UnmatchedPm4Objects.Count}");
                        
                        // Export MPRL rotation investigation CSV
                        ExportMprlRotationData(pm4Path, dirs.ModfCsv);
                    }
                }
            }


            // Stage 3: WDL Generation
            Console.WriteLine("\n[Stage 3] Generating WDL ADTs...");
            var wdlService = new WdlService();
            string actualWdlFile = "";
            
            if (!string.IsNullOrEmpty(wdlPath) && File.Exists(wdlPath))
            {
                actualWdlFile = wdlPath;
            }
            else
            {
                // Fallback to searching splitAdtPath
                var wdlCandidates = Directory.GetFiles(splitAdtPath, "*.wdl");
                if (wdlCandidates.Length > 0)
                {
                    actualWdlFile = wdlCandidates[0];
                }
            }
            
            if (!string.IsNullOrEmpty(actualWdlFile) && File.Exists(actualWdlFile))
            {
                Console.WriteLine($"[INFO] Found WDL: {Path.GetFileName(actualWdlFile)}");
                
                int generated = 0;
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var tileData = wdlService.GetTileData(actualWdlFile, x, y);
                        if (tileData != null)
                        {
                            // Generate ADT bytes
                            var adtBytes = WdlToAdtGenerator.GenerateAdt(tileData, x, y);
                            
                            // Naming: MapName_X_Y.adt. Assume MapName from WDL filename.
                            string mapName = Path.GetFileNameWithoutExtension(actualWdlFile);
                            string adtName = $"{mapName}_{x}_{y}.adt";
                            
                            // Save unpainted
                            File.WriteAllBytes(Path.Combine(dirs.WdlUnpainted, adtName), adtBytes);
                            // Save painted (copy for now, theoretically would add textures)
                            File.WriteAllBytes(Path.Combine(dirs.WdlPainted, adtName), adtBytes);
                            
                            generated++;
                        }
                    }
                }
                Console.WriteLine($"[INFO] Generated {generated} ADTs from WDL.");
                
                // Copy WDL ADTs to Patched folders as base (to be overwritten/merged later)
                // If we want WDL-based ADTs to be the primary output for unvisited areas.
                // For now, let's keep them separate as per folder structure.
                foreach (var file in Directory.GetFiles(dirs.WdlUnpainted))
                    File.Copy(file, Path.Combine(dirs.PatchedWdlUnpainted, Path.GetFileName(file)), true);

                foreach (var file in Directory.GetFiles(dirs.WdlPainted))
                    File.Copy(file, Path.Combine(dirs.PatchedWdlPainted, Path.GetFileName(file)), true);

            }
            else
            {
                Console.WriteLine("[WARN] No WDL file found (via --wdl or in split-adt path). Skiping WDL generation.");
            }


            // Stage 4: Patching Museum ADTs
            Console.WriteLine("\n[Stage 4] Patching Museum ADTs...");
            // Copy Museum ADTs first
            foreach (var file in Directory.GetFiles(museumAdtPath, "*.adt"))
            {
                File.Copy(file, Path.Combine(dirs.MuseumCopy, Path.GetFileName(file)), true);
            }
            // Copy WDT files (Critical for map loading)
            foreach (var file in Directory.GetFiles(museumAdtPath, "*.wdt"))
            {
                File.Copy(file, Path.Combine(dirs.MuseumCopy, Path.GetFileName(file)), true);
                File.Copy(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), true);
            }

            // PRE-SCAN: Collect ALL existing UniqueIds AND positions from ALL museum ADTs
            // This enables: 1) UniqueId deduplication, 2) Proximity-based skip for existing placements
            var globalUsedUniqueIds = new HashSet<uint>();
            var existingPlacements = new List<System.Numerics.Vector3>(); // All existing MODF positions
            Console.WriteLine("[INFO] Pre-scanning museum ADTs for existing placements...");
            foreach (var file in Directory.GetFiles(museumAdtPath, "*.adt"))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                if (name.Contains("_obj") || name.Contains("_tex")) continue;
                
                try
                {
                    var bytes = File.ReadAllBytes(file);
                    var str = System.Text.Encoding.ASCII.GetString(bytes);
                    int modfIdx = str.IndexOf("FDOM"); // MODF reversed
                    if (modfIdx > 0 && modfIdx + 8 <= bytes.Length)
                    {
                        int modfSize = BitConverter.ToInt32(bytes, modfIdx + 4);
                        int entriesStart = modfIdx + 8;
                        int entryCount = modfSize / 64;
                        
                        for (int i = 0; i < entryCount; i++)
                        {
                            int entryOffset = entriesStart + i * 64;
                            if (entryOffset + 20 <= bytes.Length)
                            {
                                // UniqueId at byte 4
                                uint existingId = BitConverter.ToUInt32(bytes, entryOffset + 4);
                                globalUsedUniqueIds.Add(existingId);
                                
                                // Position at bytes 8-20 (3 floats, stored as X,Z,Y)
                                float posX = BitConverter.ToSingle(bytes, entryOffset + 8);
                                float posZ = BitConverter.ToSingle(bytes, entryOffset + 12); // Height
                                float posY = BitConverter.ToSingle(bytes, entryOffset + 16);
                                existingPlacements.Add(new System.Numerics.Vector3(posX, posY, posZ));
                            }
                        }
                    }
                }
                catch { /* ignore parse errors */ }
            }
            Console.WriteLine($"[INFO] Found {globalUsedUniqueIds.Count} existing UniqueIds, {existingPlacements.Count} placements in museum ADTs");

            // Inject Modf
            // Group by tiles - use TileX/TileY from PM4 filename (not calculated from position)
            // GLOBAL UniqueId tracking: IDs must be unique across ALL tiles, not just per-tile
            // Also track proximity to existing placements to avoid duplicates
            var modfByTile = new Dictionary<(int x, int y), List<AdtPatcher.ModfEntry>>();
            uint nextAvailableUniqueId = 100_000_000; // Start high to avoid conflicts with existing IDs
            int reassignedCount = 0;
            int proximitySkipCount = 0;
            const float PROXIMITY_THRESHOLD = 5.0f; // Skip PM4 entries within 5 units of existing
            
            foreach (var entry in transformedEntries)
            {
                // PROXIMITY CHECK: Skip if too close to an existing museum placement
                bool tooCloseToExisting = false;
                foreach (var existingPos in existingPlacements)
                {
                    float dx = entry.Position.X - existingPos.X;
                    float dy = entry.Position.Y - existingPos.Y;
                    float dz = entry.Position.Z - existingPos.Z;
                    float distSq = dx*dx + dy*dy + dz*dz;
                    if (distSq < PROXIMITY_THRESHOLD * PROXIMITY_THRESHOLD)
                    {
                        tooCloseToExisting = true;
                        break;
                    }
                }
                
                if (tooCloseToExisting)
                {
                    proximitySkipCount++;
                    continue; // Skip this PM4 entry - museum already has it
                }
                
                // Use tile from PM4 filename instead of calculating from position
                var (tx, ty) = (entry.TileX, entry.TileY);
                if (!modfByTile.TryGetValue((tx, ty), out var list))
                {
                    list = new List<AdtPatcher.ModfEntry>();
                    modfByTile[(tx, ty)] = list;
                }
                
                // Ensure UniqueId is globally unique - if already used, assign a new one
                uint uniqueId = entry.UniqueId;
                if (globalUsedUniqueIds.Contains(uniqueId))
                {
                    // Find next available unique ID
                    while (globalUsedUniqueIds.Contains(nextAvailableUniqueId))
                        nextAvailableUniqueId++;
                    uniqueId = nextAvailableUniqueId++;
                    reassignedCount++;
                }
                globalUsedUniqueIds.Add(uniqueId);
                
                // Map Core ModfEntry to AdtPatcher ModfEntry with guaranteed unique ID
                list.Add(new AdtPatcher.ModfEntry
                {
                    NameId = entry.NameId,
                    UniqueId = uniqueId,
                    Position = entry.Position,
                    Rotation = entry.Rotation,
                    BoundsMin = entry.BoundsMin,
                    BoundsMax = entry.BoundsMax,
                    Flags = entry.Flags,
                    DoodadSet = entry.DoodadSet,
                    NameSet = entry.NameSet,
                    Scale = entry.Scale
                });
            }
            
            if (proximitySkipCount > 0)
                Console.WriteLine($"[INFO] Skipped {proximitySkipCount} PM4 entries (within {PROXIMITY_THRESHOLD} units of existing museum placements)");
            
            if (reassignedCount > 0)
                Console.WriteLine($"[INFO] Reassigned {reassignedCount} duplicate UniqueIds to ensure global uniqueness");

            // Patch Museum ADTs - SKIP tiles that already have MODF entries (have existing objects)
            int patchedCount = 0;
            int emptyTileCount = 0; // Tiles without existing MODF - good for reference
            var patchedTiles = new List<(int x, int y, string wmoPath, int entryCount, bool hadExisting)>();
            
            foreach (var file in Directory.GetFiles(museumAdtPath, "*.adt"))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                if (name.Contains("_obj") || name.Contains("_tex")) continue;

                var match = System.Text.RegularExpressions.Regex.Match(name, @"(\d+)_(\d+)$");
                if (!match.Success) 
                {
                     // Provide default copy for non-tiled ADTs (root?)
                     File.Copy(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), true);
                     continue;
                }

                int tx = int.Parse(match.Groups[1].Value);
                int ty = int.Parse(match.Groups[2].Value);

                // Track if tile has existing objects (for stats - we patch anyway)
                bool hasExistingTileModf = false;
                try
                {
                    var bytes = File.ReadAllBytes(file);
                    var str = System.Text.Encoding.ASCII.GetString(bytes);
                    int modfIdx = str.IndexOf("FDOM"); // MODF reversed
                    if (modfIdx > 0)
                    {
                        int modfSize = BitConverter.ToInt32(bytes, modfIdx + 4);
                        if (modfSize > 0) hasExistingTileModf = true;
                    }
                }
                catch { /* ignore */ }

                // Use the wmoNames we loaded/generated in Stage 2
                var tileWmoNames = wmoNames;

                if (modfByTile.TryGetValue((tx, ty), out var tileEntries))
                {
                    _adtPatcher.PatchWmoPlacements(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), tileWmoNames, tileEntries);
                    patchedTiles.Add((tx, ty, Path.GetFileName(file), tileEntries.Count, hasExistingTileModf));
                    patchedCount++;
                    if (!hasExistingTileModf) emptyTileCount++;
                }
                else
                {
                    File.Copy(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), true);
                }
            }
            
            // Write patched tiles list for easy verification
            var patchedListPath = Path.Combine(dirs.Root, "patched_tiles.txt");
            using (var writer = new StreamWriter(patchedListPath))
            {
                writer.WriteLine($"# Patched Museum ADTs ({patchedCount} tiles, {emptyTileCount} were empty - good reference tiles)");
                writer.WriteLine($"# Empty tiles (no existing MODF - pure PM4 placements):");
                foreach (var (x, y, fileName, cnt, hadExisting) in patchedTiles.Where(t => !t.hadExisting).OrderBy(t => t.x).ThenBy(t => t.y))
                    writer.WriteLine($"#   {x},{y} -> {fileName} ({cnt} new entries)");
                writer.WriteLine($"# Format: tile_x, tile_y, filename, entry_count, had_existing");
                foreach (var (x, y, fileName, cnt, hadExisting) in patchedTiles.OrderBy(t => t.x).ThenBy(t => t.y))
                    writer.WriteLine($"{x},{y},{fileName},{cnt},{hadExisting}");
            }
            Console.WriteLine($"[INFO] Patched {patchedCount} Museum ADTs ({emptyTileCount} empty tiles = reference).");
            Console.WriteLine($"[INFO] Patched tile list: {patchedListPath}");

            // Stage 4b: Patch WDL-generated ADTs that have MODF data
            // This creates pure WDL+MODF ADTs for testing (separate from Museum ADTs)
            Console.WriteLine("\n[Stage 4b] Patching WDL-generated ADTs with MODF data...");

            int wdlPatchedCount = 0;
            foreach (var file in Directory.GetFiles(dirs.WdlPainted, "*.adt"))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                var match = System.Text.RegularExpressions.Regex.Match(name, @"(\d+)_(\d+)$");
                if (!match.Success) continue;

                int tx = int.Parse(match.Groups[1].Value);
                int ty = int.Parse(match.Groups[2].Value);

                // Patch all WDL tiles that have MODF data (no skipping Museum tiles)
                if (modfByTile.TryGetValue((tx, ty), out var tileEntries))
                {
                    var outputPath = Path.Combine(dirs.WdlPainted, Path.GetFileName(file) + ".patched");
                    _adtPatcher.PatchWmoPlacements(file, outputPath, wmoNames, tileEntries);
                    File.Move(outputPath, file, true); // Replace original
                    wdlPatchedCount++;
                }
            }
            Console.WriteLine($"[INFO] Patched {wdlPatchedCount} WDL-generated ADTs with MODF data.");

            // Stage 4c: WL* → MH2O Liquid Conversion
            // TEMPORARILY DISABLED: MH2O serialization has format issues causing Noggit crashes
            // TODO: Fix SMLiquidInstance structure to match wiki spec
            bool enableMh2oInjection = false;
            Console.WriteLine("\n[Stage 4c] Processing WL* files for liquid restoration...");
            if (!enableMh2oInjection)
            {
                Console.WriteLine("[WARN] MH2O injection is temporarily disabled due to format issues");
                Console.WriteLine("[INFO] WL* files will be scanned but not injected into ADTs");
            }
            
            // Check for WL* files in the split-adt path (same as pm4Path typically)
            var wlPath = splitAdtPath; // WL* files are usually alongside split ADTs
            var wlFiles = new[] { "wlw", "wlm", "wlq", "wll" }
                .SelectMany(ext => Directory.GetFiles(wlPath, $"*.{ext}", SearchOption.TopDirectoryOnly))
                .ToList();
            
            if (wlFiles.Count > 0)
            {
                Console.WriteLine($"[INFO] Found {wlFiles.Count} WL* files");
                
                var wlConverter = new WlToMh2oConverter();
                var mh2oByTile = new Dictionary<(int x, int y), WlToMh2oConverter.Mh2oTileData>();
                int blocksProcessed = 0;
                
                foreach (var wlFilePath in wlFiles)
                {
                    try
                    {
                        var wlFile = WlFile.Read(wlFilePath);
                        var result = wlConverter.Convert(wlFile, Path.GetFileName(wlFilePath));
                        
                        foreach (var (tile, data) in result.TileData)
                        {
                            if (!mh2oByTile.ContainsKey(tile))
                                mh2oByTile[tile] = data;
                            else
                            {
                                // Merge chunks from this file into existing tile data
                                var existing = mh2oByTile[tile];
                                for (int cx = 0; cx < 16; cx++)
                                for (int cy = 0; cy < 16; cy++)
                                {
                                    if (existing.Chunks[cx, cy] == null && data.Chunks[cx, cy] != null)
                                        existing.Chunks[cx, cy] = data.Chunks[cx, cy];
                                }
                            }
                            blocksProcessed++;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[WARN] Failed to process {Path.GetFileName(wlFilePath)}: {ex.Message}");
                    }
                }
                
                Console.WriteLine($"[INFO] Converted {blocksProcessed} liquid blocks across {mh2oByTile.Count} tiles");
                
                // Write liquid tiles list for verification (even if injection disabled)
                var liquidTilesList = new List<(int x, int y, string adtName, int chunkCount)>();
                foreach (var (tile, tileData) in mh2oByTile)
                {
                    var adtFileName = $"development_{tile.x:D2}_{tile.y:D2}.adt";
                    liquidTilesList.Add((tile.x, tile.y, adtFileName, tileData.ChunkCount));
                }
                
                var liquidListPath = Path.Combine(dirs.Root, "liquid_tiles.txt");
                using (var writer = new StreamWriter(liquidListPath))
                {
                    writer.WriteLine($"# Liquid tiles from WL* files ({mh2oByTile.Count} tiles)");
                    writer.WriteLine($"# Source WL* files: {wlFiles.Count}");
                    writer.WriteLine($"# MH2O Injection: {(enableMh2oInjection ? "ENABLED" : "DISABLED")}");
                    writer.WriteLine($"# Format: tile_x, tile_y, adt_filename, liquid_chunks");
                    foreach (var (x, y, adtName, chunkCount) in liquidTilesList.OrderBy(t => t.x).ThenBy(t => t.y))
                        writer.WriteLine($"{x},{y},{adtName},{chunkCount}");
                }
                Console.WriteLine($"[INFO] Liquid tile list: {liquidListPath}");
                
                // Only inject if enabled
                if (enableMh2oInjection)
                {
                    int liquidPatchedCount = 0;
                    foreach (var (tile, tileData) in mh2oByTile)
                    {
                        var adtFileName = $"development_{tile.x:D2}_{tile.y:D2}.adt";
                        var adtPath = Path.Combine(dirs.PatchedMuseum, adtFileName);
                        
                        if (!File.Exists(adtPath))
                        {
                            adtPath = Path.Combine(dirs.WdlPainted, adtFileName);
                            if (!File.Exists(adtPath)) continue;
                        }
                        
                        try
                        {
                            var adtBytes = File.ReadAllBytes(adtPath);
                            var mh2oData = WlToMh2oConverter.SerializeMh2oTile(tileData);
                            
                            var patchedBytes = InjectMh2oChunk(adtBytes, mh2oData);
                            if (patchedBytes != null)
                            {
                                File.WriteAllBytes(adtPath, patchedBytes);
                                liquidPatchedCount++;
                            }
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[WARN] Failed to inject MH2O into {adtFileName}: {ex.Message}");
                        }
                    }
                    Console.WriteLine($"[INFO] Injected MH2O liquid data into {liquidPatchedCount} ADTs");
                }
            }
            else
            {
                Console.WriteLine("[INFO] No WL* files found, skipping liquid restoration");
            }


            // Stage 5: Assembly & Noggit
            Console.WriteLine("\n[Stage 5] Assembling Project...");
            // Copy Patched Museum to Final (ADTs and WDTs)
            foreach (var file in Directory.GetFiles(dirs.PatchedMuseum))
            {
                File.Copy(file, Path.Combine(dirs.FinalAssembly, Path.GetFileName(file)), true);
            }

            // Generate Noggit Project
            var noggitProj = new
            {
                Project = new
                {
                    ProjectName = "PM4_Restoration_Project",
                    Client = new
                    {
                        ClientPath = gamePath.Replace('\\', '/'),
                        ClientVersion = "Wrath Of The Lich King"
                    },
                    PinnedMaps = new object[]
                    {
                        new { MapName = "development", MapId = 0 } // Assuming map ID 0 or similar for dev, user can adjust
                    },
                    Bookmarks = new object[] { }
                }
            };
            
            var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
            string projJson = JsonSerializer.Serialize(noggitProj, jsonOptions);
            File.WriteAllText(Path.Combine(outputRoot, "development.noggitproj"), projJson);

            Console.WriteLine("\n=== Pipeline Complete ===");
            Console.WriteLine($"Output: {outputRoot}");
            Console.WriteLine($"Noggit Project: {Path.Combine(outputRoot, "development.noggitproj")}");
        }

        /// <summary>
        /// Load existing MODF/MWMO CSV data from cache.
        /// </summary>
        private (List<Pm4ModfReconstructor.ModfEntry>, List<string>) LoadExistingModfCsv(string modfPath, string mwmoPath)
        {
            var entries = new List<Pm4ModfReconstructor.ModfEntry>();
            var wmoNames = new List<string>();

            // Parse MWMO names
            if (File.Exists(mwmoPath))
            {
                foreach (var line in File.ReadLines(mwmoPath).Skip(1)) // Skip header
                {
                    var parts = line.Split(',');
                    if (parts.Length >= 2)
                        wmoNames.Add(parts[1].Trim());
                }
            }

            // Parse MODF entries
            if (File.Exists(modfPath))
            {
                foreach (var line in File.ReadLines(modfPath).Skip(1)) // Skip header
                {
                    var parts = line.Split(',');
                    if (parts.Length < 12) continue;

                    try
                    {
                        entries.Add(new Pm4ModfReconstructor.ModfEntry(
                            NameId: uint.Parse(parts[2]),
                            UniqueId: uint.Parse(parts[3]),
                            Position: new System.Numerics.Vector3(
                                float.Parse(parts[4]), float.Parse(parts[5]), float.Parse(parts[6])),
                            Rotation: new System.Numerics.Vector3(
                                float.Parse(parts[7]), float.Parse(parts[8]), float.Parse(parts[9])),
                            BoundsMin: System.Numerics.Vector3.Zero,
                            BoundsMax: System.Numerics.Vector3.Zero,
                            Flags: 0,
                            DoodadSet: 0,
                            NameSet: 0,
                            Scale: (ushort)(float.Parse(parts[10]) * 1024),
                            WmoPath: parts[1],
                            Ck24: parts[0],
                            MatchConfidence: parts.Length > 11 ? float.Parse(parts[11]) : 1.0f,
                            TileX: parts.Length > 12 ? int.Parse(parts[12]) : 0,
                            TileY: parts.Length > 13 ? int.Parse(parts[13]) : 0
                        ));
                    }
                    catch { /* Skip malformed lines */ }
                }
            }

            return (entries, wmoNames);
        }

        /// <summary>
        /// Export MODF entries to CSV.
        /// </summary>
        private void ExportModfToCsv(List<Pm4ModfReconstructor.ModfEntry> entries, string path)
        {
            using var sw = new StreamWriter(path);
            sw.WriteLine("ck24,wmo_path,name_id,unique_id,pos_x,pos_y,pos_z,rot_x,rot_y,rot_z,scale,confidence,tile_x,tile_y");
            
            foreach (var e in entries)
            {
                var (tileX, tileY) = Pm4ModfReconstructor.GetTileForPosition(e.Position);
                sw.WriteLine(string.Join(",",
                    e.Ck24,
                    e.WmoPath,
                    e.NameId,
                    e.UniqueId,
                    e.Position.X.ToString("F2"),
                    e.Position.Y.ToString("F2"),
                    e.Position.Z.ToString("F2"),
                    e.Rotation.X.ToString("F2"),
                    e.Rotation.Y.ToString("F2"),
                    e.Rotation.Z.ToString("F2"),
                    (e.Scale / 1024f).ToString("F4"),
                    e.MatchConfidence.ToString("F3"),
                    tileX,
                    tileY
                ));
            }
        }

        /// <summary>
        /// Export WMO names to CSV.
        /// </summary>
        private void ExportMwmoToCsv(List<string> wmoNames, string path)
        {
            using var sw = new StreamWriter(path);
            sw.WriteLine("index,wmo_path");
            
            for (int i = 0; i < wmoNames.Count; i++)
            {
                sw.WriteLine($"{i},{wmoNames[i]}");
            }
        }

        /// <summary>
        /// Load PM4 objects directly from .pm4 files (no PM4FacesTool required).
        /// Parses PM4 files, groups surfaces by CK24, and computes geometry stats.
        /// </summary>
        private List<Pm4ModfReconstructor.Pm4Object> LoadPm4ObjectsFromFiles(string pm4Directory)
        {
            var objects = new List<Pm4ModfReconstructor.Pm4Object>();
            var matcher = new Pm4WmoGeometryMatcher();
            
            if (!Directory.Exists(pm4Directory))
            {
                Console.WriteLine($"[WARN] PM4 directory not found: {pm4Directory}");
                return objects;
            }
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            if (pm4Files.Length == 0)
            {
                Console.WriteLine($"[WARN] No .pm4 files found in {pm4Directory}");
                return objects;
            }
            
            Console.WriteLine($"[INFO] Found {pm4Files.Length} PM4 files to process...");
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    // Parse tile coordinates from filename (e.g., development_29_39.pm4)
                    var baseName = Path.GetFileNameWithoutExtension(pm4Path);
                    var match = System.Text.RegularExpressions.Regex.Match(baseName, @"(\d+)_(\d+)$");
                    if (!match.Success) continue;
                    
                    int tileX = int.Parse(match.Groups[1].Value);
                    int tileY = int.Parse(match.Groups[2].Value);
                    
                    // Parse PM4 file using local PM4File class
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    if (pm4.Surfaces.Count == 0 || pm4.MeshVertices.Count == 0)
                    {
                        continue;
                    }
                    
                    // Group surfaces by CK24 (WMO instance ID)
                    var surfacesByCk24 = pm4.Surfaces
                        .Where(s => !s.IsM2Bucket) // Skip M2/doodad surfaces
                        .GroupBy(s => s.CK24)
                        .Where(g => g.Key != 0) // Skip terrain (CK24=0)
                        .ToList();
                    
                    foreach (var group in surfacesByCk24)
                    {
                        string ck24Hex = $"0x{group.Key:X6}";
                        
                        // Collect all vertices for this CK24 group
                        var vertices = new List<System.Numerics.Vector3>();
                        foreach (var surface in group)
                        {
                            // Each surface references MSVI indices which point to MSVT vertices
                            for (int i = 0; i < surface.IndexCount && surface.MsviFirstIndex + i < pm4.MeshIndices.Count; i++)
                            {
                                uint vertIdx = pm4.MeshIndices[(int)surface.MsviFirstIndex + i];
                                if (vertIdx < pm4.MeshVertices.Count)
                                {
                                    vertices.Add(pm4.MeshVertices[(int)vertIdx]);
                                }
                            }
                        }
                        
                        if (vertices.Count < 10) continue; // Skip tiny objects
                        
                        // MSCN Enhancement: Add nearby exterior vertices to enrich geometry fingerprint
                        // MSCN lacks CK24, so we use proximity-based association (5-10 yard margin)
                        if (pm4.ExteriorVertices.Count > 0)
                        {
                            // Compute MSVT bounds first
                            var minBound = new System.Numerics.Vector3(float.MaxValue);
                            var maxBound = new System.Numerics.Vector3(float.MinValue);
                            foreach (var v in vertices)
                            {
                                minBound = System.Numerics.Vector3.Min(minBound, v);
                                maxBound = System.Numerics.Vector3.Max(maxBound, v);
                            }
                            
                            // Expand bounds by ~7 yards (WoW units are roughly yards)
                            const float MscnMargin = 7.0f;
                            minBound -= new System.Numerics.Vector3(MscnMargin);
                            maxBound += new System.Numerics.Vector3(MscnMargin);
                            
                            int mscnAdded = 0;
                            foreach (var mscnVert in pm4.ExteriorVertices)
                            {
                                // Apply MSCN transform: (Y, -X, Z) to match minimap orientation
                                var transformed = new System.Numerics.Vector3(mscnVert.Y, -mscnVert.X, mscnVert.Z);
                                
                                // Check if within expanded bounds
                                if (transformed.X >= minBound.X && transformed.X <= maxBound.X &&
                                    transformed.Y >= minBound.Y && transformed.Y <= maxBound.Y &&
                                    transformed.Z >= minBound.Z && transformed.Z <= maxBound.Z)
                                {
                                    vertices.Add(transformed);
                                    mscnAdded++;
                                }
                            }
                            
                            // Log significant MSCN additions
                            if (mscnAdded > 50)
                            {
                                Console.WriteLine($"[MSCN] {ck24Hex}: +{mscnAdded} exterior verts (total: {vertices.Count})");
                            }
                        }
                        
                        // Compute stats on combined MSVT + MSCN vertices
                        var stats = matcher.ComputeStats(vertices);
                        objects.Add(new Pm4ModfReconstructor.Pm4Object(ck24Hex, pm4Path, tileX, tileY, stats));
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to parse {pm4Path}: {ex.Message}");
                }
            }
            
            Console.WriteLine($"[INFO] Loaded {objects.Count} PM4 objects from {pm4Files.Length} files");
            return objects;
        }

        /// <summary>
        /// Injects MH2O chunk data into an ADT file.
        /// </summary>
        private byte[]? InjectMh2oChunk(byte[] adtBytes, byte[] mh2oData)
        {
            var str = System.Text.Encoding.ASCII.GetString(adtBytes);
            
            // Find MHDR chunk to update offset
            int mhdrIdx = str.IndexOf("RDHM"); // MHDR reversed
            if (mhdrIdx < 0) return null;
            
            // Find existing MH2O chunk
            int mh2oIdx = str.IndexOf("O2HM"); // MH2O reversed
            
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            
            if (mh2oIdx > 0)
            {
                // Replace existing MH2O chunk
                int existingSize = BitConverter.ToInt32(adtBytes, mh2oIdx + 4);
                int chunkEnd = mh2oIdx + 8 + existingSize;
                
                // Write everything up to MH2O
                ms.Write(adtBytes, 0, mh2oIdx);
                
                // Write new MH2O chunk
                bw.Write(System.Text.Encoding.ASCII.GetBytes("O2HM"));
                bw.Write(mh2oData.Length);
                bw.Write(mh2oData);
                
                // Write everything after old MH2O
                ms.Write(adtBytes, chunkEnd, adtBytes.Length - chunkEnd);
            }
            else
            {
                // Insert MH2O chunk after MHDR (before MCIN)
                int mcinIdx = str.IndexOf("NICM"); // MCIN reversed
                if (mcinIdx < 0)
                {
                    // Fallback: insert after MHDR chunk
                    int mhdrSize = BitConverter.ToInt32(adtBytes, mhdrIdx + 4);
                    mcinIdx = mhdrIdx + 8 + mhdrSize;
                }
                
                // Write everything up to insertion point
                ms.Write(adtBytes, 0, mcinIdx);
                
                // Write new MH2O chunk
                bw.Write(System.Text.Encoding.ASCII.GetBytes("O2HM"));
                bw.Write(mh2oData.Length);
                bw.Write(mh2oData);
                
                // Write remainder of file
                ms.Write(adtBytes, mcinIdx, adtBytes.Length - mcinIdx);
            }
            
            return ms.ToArray();
        }
        
        /// <summary>
        /// Export MPRL rotation candidates to CSV with MODF correlation.
        /// Correlates MPRL positions with matched MODF entries to discover rotation patterns.
        /// </summary>
        private void ExportMprlRotationData(string pm4Directory, string outputDir, 
            List<Pm4ModfReconstructor.ModfEntry>? modfEntries = null)
        {
            var mprlCsvPath = Path.Combine(outputDir, "mprl_rotation_analysis.csv");
            var correlationCsvPath = Path.Combine(outputDir, "mprl_modf_correlation.csv");
            var flagAnalysisPath = Path.Combine(outputDir, "mprl_flag_analysis.txt");
            
            // Collect all MPRL entries with their source files
            var allMprlEntries = new List<(int TileX, int TileY, MprlEntry Entry)>();
            int totalMprl = 0;
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Statistics for flag analysis
            var unk14Distribution = new Dictionary<int, int>();
            var unk16Distribution = new Dictionary<ushort, int>();
            var commandEntries = new List<(int TileX, int TileY, MprlEntry Entry)>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var baseName = Path.GetFileNameWithoutExtension(pm4Path);
                    var match = System.Text.RegularExpressions.Regex.Match(baseName, @"(\d+)_(\d+)$");
                    if (!match.Success) continue;
                    
                    int tileX = int.Parse(match.Groups[1].Value);
                    int tileY = int.Parse(match.Groups[2].Value);
                    
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    foreach (var mprl in pm4.PositionRefs)
                    {
                        allMprlEntries.Add((tileX, tileY, mprl));
                        totalMprl++;
                        
                        // Track distributions
                        if (!unk14Distribution.ContainsKey(mprl.Unknown0x14))
                            unk14Distribution[mprl.Unknown0x14] = 0;
                        unk14Distribution[mprl.Unknown0x14]++;
                        
                        if (!unk16Distribution.ContainsKey(mprl.Unknown0x16))
                            unk16Distribution[mprl.Unknown0x16] = 0;
                        unk16Distribution[mprl.Unknown0x16]++;
                        
                        // Track command/flag entries
                        if (mprl.Unknown0x16 == 0x3FFF || mprl.Unknown0x14 == -1)
                            commandEntries.Add((tileX, tileY, mprl));
                    }
                }
                catch { /* ignore parse errors */ }
            }
            
            // Write raw MPRL data
            using (var sw = new StreamWriter(mprlCsvPath))
            {
                sw.WriteLine("tile_x,tile_y,mprl_idx,pos_x,pos_y,pos_z,unk04_raw,unk04_degrees,unk14_raw,unk06_hex,unk16_hex,is_command");
                foreach (var (tileX, tileY, mprl) in allMprlEntries)
                {
                    sw.WriteLine(string.Join(",",
                        tileX, tileY, mprl.Index,
                        mprl.PositionX.ToString("F3"), mprl.PositionY.ToString("F3"), mprl.PositionZ.ToString("F3"),
                        mprl.Unknown0x04, mprl.HeadingDegrees.ToString("F2"), mprl.Unknown0x14,
                        $"0x{mprl.Unknown0x06:X4}", $"0x{mprl.Unknown0x16:X4}", mprl.IsCommandEntry));
                }
            }
            
            // Write flag analysis
            using (var sw = new StreamWriter(flagAnalysisPath))
            {
                sw.WriteLine("=== MPRL Flag Analysis ===");
                sw.WriteLine($"Total MPRL entries: {totalMprl}");
                sw.WriteLine($"Command/Flag entries (unk16=0x3FFF or unk14=-1): {commandEntries.Count}");
                sw.WriteLine();
                
                sw.WriteLine("=== Unknown0x14 Distribution ===");
                foreach (var kvp in unk14Distribution.OrderBy(x => x.Key))
                    sw.WriteLine($"  unk14={kvp.Key,4}: {kvp.Value,6} entries");
                
                sw.WriteLine();
                sw.WriteLine("=== Unknown0x16 Distribution ===");
                foreach (var kvp in unk16Distribution.OrderBy(x => x.Key))
                    sw.WriteLine($"  unk16=0x{kvp.Key:X4}: {kvp.Value,6} entries");
                
                sw.WriteLine();
                sw.WriteLine("=== Sample Command Entries (unk16=0x3FFF) ===");
                foreach (var (tx, ty, mprl) in commandEntries.Take(50))
                {
                    sw.WriteLine($"  Tile({tx},{ty}) Idx={mprl.Index}: pos=({mprl.PositionX:F1},{mprl.PositionY:F1},{mprl.PositionZ:F1}) unk04={mprl.Unknown0x04} unk14={mprl.Unknown0x14} unk16=0x{mprl.Unknown0x16:X4}");
                }
            }
            
            Console.WriteLine($"[INFO] MPRL Rotation Analysis: {mprlCsvPath} ({totalMprl} entries)");
            Console.WriteLine($"[INFO] MPRL Flag Analysis: {flagAnalysisPath}");
            Console.WriteLine($"[INFO] Command entries (0x3FFF flag): {commandEntries.Count}");
            Console.WriteLine($"[INFO] Unk14 unique values: {unk14Distribution.Count}");
            Console.WriteLine($"[INFO] Unk16 unique values: {unk16Distribution.Count}");
            
            // Deep MSLK-CK24 correlation analysis
            ExportMslkCk24Analysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Deep analysis of MSLK ↔ CK24 relationships to find sub-object segmentation patterns.
        /// Explores how MSLK TypeFlags, Subtype, GroupObjectId correlate with CK24 groups.
        /// </summary>
        private void ExportMslkCk24Analysis(string pm4Directory, string outputDir)
        {
            var mslkAnalysisPath = Path.Combine(outputDir, "mslk_ck24_analysis.txt");
            var mslkCsvPath = Path.Combine(outputDir, "mslk_detail.csv");
            
            // Track patterns across all PM4 files
            var typeDistribution = new Dictionary<byte, int>();
            var subtypeDistribution = new Dictionary<byte, int>();
            var groupIdToCk24s = new Dictionary<uint, HashSet<uint>>();  // GroupObjectId -> CK24s
            var ck24ToMslkCount = new Dictionary<uint, int>();          // CK24 -> MSLK entry count
            var typeFlagCombos = new Dictionary<(byte type, byte subtype), int>();
            
            int totalMslk = 0;
            int totalSurfaces = 0;
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Per-file detailed data
            var allMslkEntries = new List<(string File, int TileX, int TileY, MslkEntry Entry, List<uint> AssociatedCk24s)>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var baseName = Path.GetFileNameWithoutExtension(pm4Path);
                    var match = System.Text.RegularExpressions.Regex.Match(baseName, @"(\d+)_(\d+)$");
                    if (!match.Success) continue;
                    
                    int tileX = int.Parse(match.Groups[1].Value);
                    int tileY = int.Parse(match.Groups[2].Value);
                    
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    totalSurfaces += pm4.Surfaces.Count;
                    
                    // Build CK24 -> MSUR surface mapping
                    var ck24Surfaces = pm4.Surfaces
                        .Where(s => !s.IsM2Bucket && s.CK24 != 0)
                        .GroupBy(s => s.CK24)
                        .ToDictionary(g => g.Key, g => g.ToList());
                    
                    foreach (var mslk in pm4.LinkEntries)
                    {
                        totalMslk++;
                        
                        // Track distributions
                        if (!typeDistribution.ContainsKey(mslk.TypeFlags))
                            typeDistribution[mslk.TypeFlags] = 0;
                        typeDistribution[mslk.TypeFlags]++;
                        
                        if (!subtypeDistribution.ContainsKey(mslk.Subtype))
                            subtypeDistribution[mslk.Subtype] = 0;
                        subtypeDistribution[mslk.Subtype]++;
                        
                        var combo = (mslk.TypeFlags, mslk.Subtype);
                        if (!typeFlagCombos.ContainsKey(combo))
                            typeFlagCombos[combo] = 0;
                        typeFlagCombos[combo]++;
                        
                        // Find associated CK24s (surfaces that share GroupObjectId bits)
                        var associatedCk24s = new List<uint>();
                        
                        // Strategy 1: Check if GroupObjectId matches any CK24 directly
                        if (ck24Surfaces.ContainsKey(mslk.GroupObjectId))
                            associatedCk24s.Add(mslk.GroupObjectId);
                        
                        // Strategy 2: Check CK24 derived from GroupObjectId (high 24 bits)
                        uint derivedCk24 = (mslk.GroupObjectId & 0xFFFFFF00) >> 8;
                        if (derivedCk24 != 0 && ck24Surfaces.ContainsKey(derivedCk24))
                            associatedCk24s.Add(derivedCk24);
                        
                        // Track Group->CK24 relationships
                        if (!groupIdToCk24s.ContainsKey(mslk.GroupObjectId))
                            groupIdToCk24s[mslk.GroupObjectId] = new HashSet<uint>();
                        foreach (var ck24 in associatedCk24s)
                            groupIdToCk24s[mslk.GroupObjectId].Add(ck24);
                        
                        allMslkEntries.Add((baseName, tileX, tileY, mslk, associatedCk24s));
                    }
                }
                catch { /* ignore parse errors */ }
            }
            
            // Write detailed CSV
            using (var sw = new StreamWriter(mslkCsvPath))
            {
                sw.WriteLine("file,tile_x,tile_y,idx,type,subtype,group_id,group_id_hex,mspi_first,mspi_count,link_id_hex,ref_idx,sys_flag_hex,has_geo,ck24_matches");
                foreach (var (file, tx, ty, m, ck24s) in allMslkEntries)
                {
                    var idx = pm4Files.ToList().FindIndex(f => Path.GetFileNameWithoutExtension(f) == file);
                    sw.WriteLine(string.Join(",",
                        file, tx, ty, idx,
                        m.TypeFlags, m.Subtype,
                        m.GroupObjectId, $"0x{m.GroupObjectId:X8}",
                        m.MspiFirstIndex, m.MspiIndexCount,
                        $"0x{m.LinkId:X6}", m.RefIndex, $"0x{m.SystemFlag:X4}",
                        m.HasGeometry,
                        string.Join(";", ck24s.Select(c => $"0x{c:X6}"))));
                }
            }
            
            // Write analysis report
            using (var sw = new StreamWriter(mslkAnalysisPath))
            {
                sw.WriteLine("=== MSLK ↔ CK24 Deep Analysis ===");
                sw.WriteLine($"Total MSLK entries: {totalMslk}");
                sw.WriteLine($"Total MSUR surfaces: {totalSurfaces}");
                sw.WriteLine($"PM4 files analyzed: {pm4Files.Length}");
                sw.WriteLine();
                
                sw.WriteLine("=== TypeFlags Distribution (potential object type) ===");
                foreach (var kvp in typeDistribution.OrderBy(x => x.Key))
                    sw.WriteLine($"  Type {kvp.Key,2}: {kvp.Value,6} entries ({100.0 * kvp.Value / totalMslk:F1}%)");
                
                sw.WriteLine();
                sw.WriteLine("=== Subtype Distribution (potential sub-object layer?) ===");
                foreach (var kvp in subtypeDistribution.OrderBy(x => x.Key))
                    sw.WriteLine($"  Subtype {kvp.Key,3}: {kvp.Value,6} entries ({100.0 * kvp.Value / totalMslk:F1}%)");
                
                sw.WriteLine();
                sw.WriteLine("=== Type+Subtype Combinations (top 30) ===");
                foreach (var kvp in typeFlagCombos.OrderByDescending(x => x.Value).Take(30))
                    sw.WriteLine($"  Type={kvp.Key.type,2} Subtype={kvp.Key.subtype,3}: {kvp.Value,6} entries");
                
                sw.WriteLine();
                sw.WriteLine("=== GroupObjectId → CK24 Correlation ===");
                var multiCk24Groups = groupIdToCk24s.Where(g => g.Value.Count > 1).ToList();
                var singleCk24Groups = groupIdToCk24s.Where(g => g.Value.Count == 1).ToList();
                var noCk24Groups = groupIdToCk24s.Where(g => g.Value.Count == 0).ToList();
                sw.WriteLine($"  Groups with 1 CK24: {singleCk24Groups.Count}");
                sw.WriteLine($"  Groups with multiple CK24s: {multiCk24Groups.Count}");
                sw.WriteLine($"  Groups with no CK24 match: {noCk24Groups.Count}");
                
                if (multiCk24Groups.Count > 0)
                {
                    sw.WriteLine();
                    sw.WriteLine("=== Sample Multi-CK24 Groups (potential merged objects) ===");
                    foreach (var g in multiCk24Groups.Take(20))
                    {
                        sw.WriteLine($"  GroupId 0x{g.Key:X8}: CK24s = {string.Join(", ", g.Value.Select(c => $"0x{c:X6}"))}");
                    }
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Hypothesis: Sub-object Segmentation ===");
                sw.WriteLine("If TypeFlags or Subtype correlate with object boundaries:");
                sw.WriteLine("- Type values might distinguish WMO vs M2 vs terrain");
                sw.WriteLine("- Subtype might indicate floors/levels within a building");
                sw.WriteLine("- RefIndex might point to parent/child relationships");
                sw.WriteLine("- GroupObjectId high bits might encode object instance ID");
            }
            
            Console.WriteLine($"[INFO] MSLK-CK24 Analysis: {mslkAnalysisPath}");
            Console.WriteLine($"[INFO] MSLK Detail CSV: {mslkCsvPath}");
            Console.WriteLine($"[INFO] Type values: {typeDistribution.Count}, Subtype values: {subtypeDistribution.Count}");
            Console.WriteLine($"[INFO] Multi-CK24 groups (merged objects?): {groupIdToCk24s.Where(g => g.Value.Count > 1).Count()}");
        }
    }
}
