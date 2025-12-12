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

                        byte[]? rootData = null;
                        using (var s = archiveSource.OpenFile(wmoPath))
                        {
                             if (s == null) continue;
                             using (var ms = new MemoryStream()) { s.CopyTo(ms); rootData = ms.ToArray(); }
                        }
                        
                        var data = WmoWalkableSurfaceExtractor.ExtractFromBytes(rootData, wmoPath, GroupLoader);
                        
                        if (data.GroupCount > 0)
                        {
                            var wmoFileName = Path.GetFileName(wmoPath); // e.g. "NSabbey.wmo"
                            var relativeDir = Path.GetDirectoryName(wmoPath);
                            // Structure: wmo_library/[OriginalPath]/[Name.wmo]/[Name]_flags_XX.obj
                            var objOutputDir = Path.Combine(dirs.WmoLib, relativeDir ?? "", wmoFileName);
                            
                            WmoWalkableSurfaceExtractor.ExportPerFlag(data, objOutputDir);
                            processed++;
                            
                            if (processed % 10 == 0) Console.Write($"\r[INFO] Processed {processed} WMOs...");
                        }
                    }
                    catch (Exception ex)
                    {
                        // Log verbose only on specific request to avoid spam
                        // Console.WriteLine($"[WARN] Failed {wmoPath}: {ex.Message}");
                    }
                }
                Console.WriteLine($"\n[INFO] WMO Processing Complete. Processed: {processed}");
            }
            else
            {
                Console.WriteLine("[WARN] Game path or listfile missing, skipping WMO processing.");
            }

            // Stage 2: PM4 Matching
            Console.WriteLine("\n[Stage 2] PM4 Matching...");
            var pm4Objects = _reconstructor.LoadPm4Objects(pm4Path);
            
            // Load path mappings
            var pathMap = _reconstructor.LoadWmoPathMapping(listfilePath);
            var wmoLibrary = _reconstructor.BuildWmoLibrary(gamePath, listfilePath, outputRoot);
            
            // Confidence raised to 88%
            var result = _reconstructor.ReconstructModf(pm4Objects, wmoLibrary, 0.88f); 

            var transformedEntries = result.ModfEntries
                .Select(e => e with { Position = PipelineCoordinateService.ServerToAdtPosition(e.Position) })
                .ToList();
            
            _reconstructor.ExportToCsv(result with { ModfEntries = transformedEntries }, Path.Combine(dirs.ModfCsv, "modf_entries.csv"));
            // Export Mwmo Names
            var mwmoPath = Path.Combine(dirs.ModfCsv, "mwmo_names.csv");
            _reconstructor.ExportMwmoNames(result, mwmoPath);
            
            // Export Candidate List
            var candidatesPath = Path.Combine(dirs.ModfCsv, "match_candidates.csv");
            _reconstructor.ExportCandidatesCsv(result, candidatesPath);


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
             // Group by tiles
            var modfByTile = new Dictionary<(int x, int y), List<AdtPatcher.ModfEntry>>();
            foreach (var entry in transformedEntries)
            {
                var (tx, ty) = Pm4ModfReconstructor.GetTileForPosition(entry.Position);
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

                List<string> wmoNames = File.ReadAllLines(mwmoPath).Skip(1).Select(l => l.Split(',')[1].Trim()).Take(result.WmoNames.Count).ToList();  

                if (modfByTile.TryGetValue((tx, ty), out var tileEntries))
                {
                    _adtPatcher.PatchWmoPlacements(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), wmoNames, tileEntries);
                    patchedCount++;
                }
                else
                {
                    File.Copy(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), true);
                }
            }
            Console.WriteLine($"[INFO] Patched {patchedCount} Museum ADTs.");


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
    }
}
