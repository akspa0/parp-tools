using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.PM4;
using WoWRollback.PM4Module.Services;

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

        public void Execute(string gamePath, string listfilePath, string pm4Path, string splitAdtPath, string museumAdtPath, string outputRoot, string? wdlPath = null)
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
                    var wmoLibrary = _reconstructor.BuildWmoLibrary(gamePath, listfilePath, outputRoot);
                    
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

            // Inject Modf
             // Group by tiles - use TileX/TileY from PM4 filename (not calculated from position)
            var modfByTile = new Dictionary<(int x, int y), List<AdtPatcher.ModfEntry>>();
            foreach (var entry in transformedEntries)
            {
                // Use tile from PM4 filename instead of calculating from position
                var (tx, ty) = (entry.TileX, entry.TileY);
                if (!modfByTile.TryGetValue((tx, ty), out var list))
                {
                    list = new List<AdtPatcher.ModfEntry>();
                    modfByTile[(tx, ty)] = list;
                }
                
                // Map Core ModfEntry to AdtPatcher ModfEntry
                list.Add(new AdtPatcher.ModfEntry
                {
                    NameId = entry.NameId,
                    UniqueId = entry.UniqueId,
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

            // Patch
            int patchedCount = 0;
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

                // Use the wmoNames we loaded/generated in Stage 2
                var tileWmoNames = wmoNames;

                if (modfByTile.TryGetValue((tx, ty), out var tileEntries))
                {
                    _adtPatcher.PatchWmoPlacements(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), tileWmoNames, tileEntries);
                    patchedCount++;
                }
                else
                {
                    File.Copy(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), true);
                }
            }
            Console.WriteLine($"[INFO] Patched {patchedCount} Museum ADTs.");

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
                        
                        // Per PM4FacesTool: PM4 MSVT vertices are already in global coordinates
                        // No transformation needed - use raw coordinates for geometry matching
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
    }
}
