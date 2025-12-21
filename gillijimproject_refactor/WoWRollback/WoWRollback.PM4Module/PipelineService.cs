using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using WoWRollback.Core.Services.Archive;
using WoWRollback.Core.Services.PM4;
using WoWRollback.PM4Module.Services;
using GillijimProject.WowFiles.Wl;
using System.Numerics;

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

        public void Execute(string gamePath, string listfilePath, string pm4Path, string splitAdtPath, string museumAdtPath, string outputRoot, string? wdlPath = null, string? wmoFilter = null, string? m2Filter = null, bool useFullMesh = false, string? originalSplitPath = null)
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

            // M2 library - declared at method level so it's accessible in Stage 2b and 4e
            var m2LibraryCachePath = Path.Combine(dirs.Root, "m2_library_cache.json");
            Dictionary<string, Pm4ModfReconstructor.M2Reference> m2Library = new();

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
                        // Apply WMO path filter if specified
                        if (!string.IsNullOrEmpty(wmoFilter) && 
                            !wmoPath.Contains(wmoFilter, StringComparison.OrdinalIgnoreCase))
                        {
                            skipped++;
                            continue;
                        }
                        
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
                
                // Stage 1b: M2 Library Building (In-Memory from MPQ)
                Console.WriteLine("\n[Stage 1b] Building M2 Reference Library from MPQ...");
                
                if (File.Exists(m2LibraryCachePath))
                {
                    Console.WriteLine($"[INFO] Loading M2 library from cache: {m2LibraryCachePath}");
                    try
                    {
                        var json = File.ReadAllText(m2LibraryCachePath);
                        var list = System.Text.Json.JsonSerializer.Deserialize<List<Pm4ModfReconstructor.M2Reference>>(json);
                        if (list != null)
                            m2Library = list.ToDictionary(x => x.M2Path, x => x, StringComparer.OrdinalIgnoreCase);
                        Console.WriteLine($"[INFO] Loaded {m2Library.Count} M2s from cache");
                    }
                    catch { Console.WriteLine("[WARN] Failed to load M2 cache, rebuilding..."); }
                }
                
                if (m2Library.Count == 0)
                {
                    // Build M2 library in-memory from MPQ (just like WMOs)
                    var m2Entries = File.ReadLines(listfilePath)
                        .Where(l => l.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                        .Select(l => l.Replace('/', '\\'))
                        .ToList();
                    
                    Console.WriteLine($"[INFO] Found {m2Entries.Count} M2 files to process from listfile...");
                    
                    var m2Builder = new M2LibraryBuilder();
                    var matcher = new Pm4WmoGeometryMatcher();
                    int m2Processed = 0;
                    int m2Skipped = 0;
                    object m2Lock = new object();
                    
                    // Process M2s in parallel (like WMOs)
                    System.Threading.Tasks.Parallel.ForEach(m2Entries, m2Path =>
                    {
                        try
                        {
                            // Apply M2 path filter if specified
                            if (!string.IsNullOrEmpty(m2Filter) && 
                                !m2Path.Contains(m2Filter, StringComparison.OrdinalIgnoreCase))
                            {
                                return; // Skip M2s not matching filter
                            }
                            
                            if (!archiveSource.FileExists(m2Path)) return;
                            
                            byte[]? data = null;
                            using (var s = archiveSource.OpenFile(m2Path))
                            {
                                if (s == null) return;
                                using (var ms = new MemoryStream()) { s.CopyTo(ms); data = ms.ToArray(); }
                            }
                            
                            if (data == null || data.Length < 64) return;
                            
                            var m2File = new M2File(data);
                            if (m2File.Vertices.Count < 3) return;
                            
                            var stats = matcher.ComputeStats(m2File.Vertices);
                            var normalizedPath = m2Path.Replace('\\', '/');
                            var reference = new Pm4ModfReconstructor.M2Reference(0, normalizedPath, Path.GetFileName(normalizedPath), stats);
                            
                            lock (m2Lock)
                            {
                                m2Library[normalizedPath] = reference;
                                m2Processed++;
                                if (m2Processed % 500 == 0)
                                    Console.Write($"\r[INFO] Processed {m2Processed} M2s...");
                            }
                        }
                        catch { /* ignore invalid M2s */ }
                    });
                    
                    Console.WriteLine($"\n[INFO] M2 Processing Complete. Processed: {m2Processed}");
                    
                    // Cache the M2 library
                    if (m2Library.Count > 0)
                    {
                        try
                        {
                            var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
                            var json = System.Text.Json.JsonSerializer.Serialize(m2Library.Values.ToList(), options);
                            File.WriteAllText(m2LibraryCachePath, json);
                            Console.WriteLine($"[INFO] Saved M2 library cache: {m2LibraryCachePath}");
                        }
                        catch { }
                    }
                }
            }
            else
            {
                Console.WriteLine("[WARN] Game path or listfile missing, skipping WMO/M2 processing.");
            }

            // Stage 2: Direct PM4 Processing
            Console.WriteLine("\n[Stage 2] Direct PM4 Processing...");
            
            var modfCsvPath = Path.Combine(dirs.ModfCsv, "modf_entries.csv");
            var mwmoPath = Path.Combine(dirs.ModfCsv, "mwmo_names.csv");
            var mddfCsvPath = Path.Combine(dirs.MddfCsv, "mddf_entries.csv");
            var m2NamesPath = Path.Combine(dirs.MddfCsv, "m2_names.csv");
            
            List<Pm4ModfReconstructor.ModfEntry> transformedEntries = new();
            List<string> wmoNames = new();
            List<Pm4ModfReconstructor.MddfEntry> mddfEntries = new();
            List<string> m2Names = new();
            
            // Global UniqueID counter to prevent duplicates across WMOs and M2s
            uint globalNextUniqueId = 75_000_000; // Start at 75M to avoid conflicts with existing IDs
            
            // Check for existing MODF CSVs with actual data (more than just headers)
            bool hasExistingModf = File.Exists(modfCsvPath) && new FileInfo(modfCsvPath).Length > 200;
            bool hasExistingMwmo = File.Exists(mwmoPath) && new FileInfo(mwmoPath).Length > 50;
            bool hasExistingMddf = File.Exists(mddfCsvPath) && new FileInfo(mddfCsvPath).Length > 200;
            
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
                
                // Step 1b: Export PM4 candidates as OBJ for decoder verification
                var pm4ObjDebugDir = Path.Combine(outputRoot, "pm4_obj_debug");
                Console.WriteLine($"[INFO] Exporting PM4 candidates to OBJ: {pm4ObjDebugDir}");
                ExportPm4CandidatesToObj(pm4Path, pm4ObjDebugDir);
                
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
                        var result = _reconstructor.ReconstructModf(pm4Objects, wmoLibrary, 0.88f, globalNextUniqueId);
                        globalNextUniqueId += (uint)result.ModfEntries.Count; // Reserve IDs for next batch
                        
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
                    
                    // Step 4: M2 Matching (Stage 2b)
                    // M2 objects exist in any CK24 group, not just 0x000000
                    // Use all PM4 objects for M2 matching
                    var m2Candidates = pm4Objects.ToList();
                    Console.WriteLine($"\n[Stage 2b] Processing M2 candidates ({m2Candidates.Count} PM4 objects)...");
                    
                    // Load M2 library from cache if we didn't build it in Stage 1b
                    if (m2Library.Count == 0 && File.Exists(m2LibraryCachePath))
                    {
                        try
                        {
                            var json = File.ReadAllText(m2LibraryCachePath);
                            var list = System.Text.Json.JsonSerializer.Deserialize<List<Pm4ModfReconstructor.M2Reference>>(json);
                            if (list != null)
                                m2Library = list.ToDictionary(x => x.M2Path, x => x, StringComparer.OrdinalIgnoreCase);
                        }
                        catch { }
                    }
                    
                    if (m2Candidates.Count > 0 && m2Library.Count > 0)
                    {
                        Console.WriteLine($"[INFO] Matching {m2Candidates.Count} M2 candidates against {m2Library.Count} reference M2s...");
                        var mddfResult = _reconstructor.ReconstructMddf(m2Candidates, m2Library.Values.ToList(), 0.97f, globalNextUniqueId);
                        globalNextUniqueId += (uint)mddfResult.MddfEntries.Count; // Reserve IDs for M2s
                        mddfEntries = mddfResult.MddfEntries;
                        m2Names = mddfResult.M2Names;
                        
                        // Export MDDF CSV
                        _reconstructor.ExportMddfToCsv(mddfResult, mddfCsvPath);
                        
                        Console.WriteLine($"[INFO] Matched {mddfEntries.Count} MDDF entries, {m2Names.Count} unique M2s");
                    }
                    else if (m2Candidates.Count == 0)
                    {
                        Console.WriteLine("[INFO] No PM4 objects found for M2 matching");
                    }
                    else
                    {
                        Console.WriteLine("[WARN] M2 library empty, skipping M2 matching");
                    }
                }
            }


            // Stage 2b: Generate Debug M2s from CK24 Objects (for visual verification)
            // TEMPORARILY DISABLED: M2 format may be incorrect, causing Noggit crashes
            bool enableDebugM2s = false;
            Console.WriteLine("\n[Stage 2b] Generating debug M2s from CK24 objects...");
            var debugM2Entries = new List<(string path, Vector3 position, int tileX, int tileY)>();
            if (!enableDebugM2s)
            {
                Console.WriteLine("[WARN] Debug M2 generation is DISABLED - M2 format needs fixing");
            }
            else
            {
            var pm4M2Writer = new Analysis.Pm4M2Writer();
            string debugM2Dir = Path.Combine(outputRoot, "World", "m2", "pm4_debug");
            Directory.CreateDirectory(debugM2Dir);
            
            // Process each PM4 file and extract CK24 objects
            int debugM2Count = 0;
            foreach (var pm4File in Directory.GetFiles(pm4Path, "*.pm4", SearchOption.AllDirectories))
            {
                try
                {
                    var pm4 = PM4File.FromFile(pm4File);
                    if (pm4.Surfaces == null || pm4.Surfaces.Count == 0) continue;
                    
                    // Extract tile coordinates from PM4 filename (e.g., development_31_32.pm4 → 31,32)
                    var pm4FileName = Path.GetFileNameWithoutExtension(pm4File);
                    var tileMatch = System.Text.RegularExpressions.Regex.Match(pm4FileName, @"_(\d+)_(\d+)$");
                    int fileTileX = tileMatch.Success ? int.Parse(tileMatch.Groups[1].Value) : 0;
                    int fileTileY = tileMatch.Success ? int.Parse(tileMatch.Groups[2].Value) : 0;
                    
                    var groups = pm4.Surfaces
                        .GroupBy(s => s.CK24)
                        .Where(g => g.Key != 0) // Include ALL CK24 objects (not just 0x40 type)
                        .ToList();
                    
                    foreach (var group in groups)
                    {
                        uint ck24 = group.Key;
                        var surfaces = group.ToList();
                        
                        // Use proper indexed mesh extraction (like PM4FacesTool)
                        // Each CK24 object: collect all unique vertices and build triangle list
                        var vertexMap = new Dictionary<int, int>(); // MSVT index -> local vertex index
                        var vertices = new List<Vector3>();
                        var triangles = new List<int>(); // Triangle indices (3 per triangle)
                        
                        foreach (var surf in surfaces)
                        {
                            if (surf.GroupKey == 0) continue;
                            
                            int first = (int)surf.MsviFirstIndex;
                            int count = surf.IndexCount;
                            if (first < 0 || count < 3) continue;
                            if (first + count > pm4.MeshIndices.Count) continue;
                            
                            // Build polygon vertex list for this surface
                            int[] poly = new int[count];
                            bool valid = true;
                            for (int k = 0; k < count && valid; k++)
                            {
                                uint meshIdx = pm4.MeshIndices[first + k];
                                if (meshIdx >= pm4.MeshVertices.Count) { valid = false; break; }
                                
                                // Map MSVT index to local vertex list
                                int msvtIdx = (int)meshIdx;
                                if (!vertexMap.TryGetValue(msvtIdx, out int localIdx))
                                {
                                    localIdx = vertices.Count;
                                    vertexMap[msvtIdx] = localIdx;
                                    vertices.Add(pm4.MeshVertices[msvtIdx]);
                                }
                                poly[k] = localIdx;
                            }
                            if (!valid) continue;
                            
                            // Triangulate polygon as fan (PM4FacesTool pattern):
                            // tri: 0-1-2
                            // quad: 0-1-2, 0-2-3
                            // n-gon: 0-1-2, 0-2-3, 0-3-4, ...
                            if (count == 3)
                            {
                                triangles.Add(poly[0]); triangles.Add(poly[1]); triangles.Add(poly[2]);
                            }
                            else if (count == 4)
                            {
                                triangles.Add(poly[0]); triangles.Add(poly[1]); triangles.Add(poly[2]);
                                triangles.Add(poly[0]); triangles.Add(poly[2]); triangles.Add(poly[3]);
                            }
                            else
                            {
                                // N-gon fan triangulation
                                for (int i = 1; i + 1 < count; i++)
                                {
                                    triangles.Add(poly[0]); triangles.Add(poly[i]); triangles.Add(poly[i + 1]);
                                }
                            }
                        }
                        
                        if (vertices.Count >= 3 && triangles.Count >= 3)
                        {
                            string m2Name = $"ck24_{ck24:X6}";
                            Vector3 centroid = pm4M2Writer.WriteM2(debugM2Dir, m2Name, vertices, triangles);
                            // WoW expects forward slashes and lowercase paths, .m2 extension
                            string m2GamePath = $"world\\m2\\pm4_debug\\{m2Name}.m2";
                            // Store tile from PM4 filename with the entry
                            debugM2Entries.Add((m2GamePath, centroid, fileTileX, fileTileY));
                            debugM2Count++;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to process {Path.GetFileName(pm4File)}: {ex.Message}");
                }
            }
            Console.WriteLine($"[INFO] Generated {debugM2Count} debug M2s from CK24 objects");
            } // End if (enableDebugM2s)
            
            // Add debug M2 paths to m2Names and create MDDF entries for injection
            // These will be automatically added to mddfEntries and patched via Stage 4e
            int debugM2NameIdStart = m2Names.Count; // NameIds for debug M2s start after existing ones
            foreach (var (m2Path, _, _, _) in debugM2Entries)
            {
                m2Names.Add(m2Path);
            }
            
            // Create MDDF entries for debug M2s (will be injected via Stage 4e)
            uint debugM2UniqueIdBase = 9_000_000; // Different range from matched M2s
            for (int i = 0; i < debugM2Entries.Count; i++)
            {
                var (debugM2Path, centroid, tileX, tileY) = debugM2Entries[i];
                
                // Transform centroid from WoW world coords (already transformed by M2Writer) to ADT placement coords
                // ADT MDDF placement: same as MODF - PlacementX = 32*TILESIZE - WorldY, PlacementZ = 32*TILESIZE - WorldX
                const float HalfMap = 533.33333f * 32f;
                var adtPosition = new Vector3(
                    HalfMap - centroid.Y, // Placement X
                    centroid.Z,            // Placement Y (Height)
                    HalfMap - centroid.X   // Placement Z
                );
                
                // Add to mddfEntries - these get injected in Stage 4e
                mddfEntries.Add(new Pm4ModfReconstructor.MddfEntry(
                    NameId: (uint)(debugM2NameIdStart + i),
                    UniqueId: debugM2UniqueIdBase + (uint)i,
                    Position: adtPosition,
                    Rotation: Vector3.Zero,
                    Scale: 1024,
                    Flags: 0,
                    M2Path: debugM2Path,
                    Ck24: $"debug_{i:X6}",
                    MatchConfidence: 1.0f,  // Debug M2s are 100% "matched" (we made them)
                    TileX: tileY,  // SWAP: M2 bucket tiles appear swapped in PM4 vs WMOs
                    TileY: tileX   // SWAP: M2 bucket tiles appear swapped in PM4 vs WMOs
                ));
            }
            Console.WriteLine($"[INFO] Added {debugM2Entries.Count} debug M2 placements to MDDF injection queue");
            
            // Export debug M2 placements CSV for verification
            var debugM2CsvPath = Path.Combine(outputRoot, "debug_m2_placements.csv");
            using (var sw = new StreamWriter(debugM2CsvPath))
            {
                sw.WriteLine("M2Path,TileX,TileY,PosX,PosY,PosZ,NameId,UniqueId");
                foreach (var (m2Path, centroid, tileX, tileY) in debugM2Entries)
                {
                    const float HalfMap = 533.33333f * 32f;
                    sw.WriteLine($"{m2Path},{tileY},{tileX},{HalfMap - centroid.Y:F2},{centroid.Z:F2},{HalfMap - centroid.X:F2},{debugM2NameIdStart + debugM2Entries.IndexOf((m2Path, centroid, tileX, tileY))},{debugM2UniqueIdBase + (uint)debugM2Entries.IndexOf((m2Path, centroid, tileX, tileY))}");  // SWAP tileX/tileY in CSV output
                }
            }
            Console.WriteLine($"[INFO] Debug M2 placements exported to: {debugM2CsvPath}");
            
            // Log tile distribution
            var m2TileDistribution = debugM2Entries.GroupBy(e => (e.tileX, e.tileY)).OrderBy(g => g.Key.tileX).ThenBy(g => g.Key.tileY).Take(20);
            Console.WriteLine($"[INFO] Debug M2 tile distribution (first 20): {string.Join(", ", m2TileDistribution.Select(g => $"({g.Key.tileX},{g.Key.tileY})={g.Count()}"))}");

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
                string mapName = Path.GetFileNameWithoutExtension(actualWdlFile);
                
                int generated = 0;
                int textured = 0;
                for (int y = 0; y < 64; y++)
                {
                    for (int x = 0; x < 64; x++)
                    {
                        var tileData = wdlService.GetTileData(actualWdlFile, x, y);
                        if (tileData != null)
                        {
                            // Generate unpainted ADT (from WDL, no textures)
                            var adtBytesUnpainted = WdlToAdtGenerator.GenerateAdt(tileData, x, y);
                            string adtName = $"{mapName}_{x}_{y}.adt";
                            
                            // Save unpainted (WDL-based)
                            File.WriteAllBytes(Path.Combine(dirs.WdlUnpainted, adtName), adtBytesUnpainted);
                            
                            // For painted: if museum ADT exists, just copy it directly
                            // This preserves the original terrain + textures intact
                            string museumAdtFile = Path.Combine(museumAdtPath, adtName);
                            if (File.Exists(museumAdtFile))
                            {
                                // Copy museum ADT directly for painted version
                                File.Copy(museumAdtFile, Path.Combine(dirs.WdlPainted, adtName), true);
                                textured++;
                            }
                            else
                            {
                                // No museum ADT, use WDL-generated unpainted
                                File.WriteAllBytes(Path.Combine(dirs.WdlPainted, adtName), adtBytesUnpainted);
                            }
                            
                            generated++;
                        }
                    }
                }
                Console.WriteLine($"[INFO] Generated {generated} ADTs from WDL ({textured} copied from museum).");
                
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

            // STAGE 0: Restore Original UniqueIDs from Original Split ADTs (if provided)
            // This step reads the ORIGINAL development split ADTs (e.g., from WoW Client data)
            // and extracts ALL UniqueIDs. These are used to:
            // 1. Match/restore IDs in Museum ADTs (based on position + name)
            // 2. Reserve these IDs so new PM4 placements don't conflict
            var originalIdsByTile = new Dictionary<(int x, int y), List<(uint uniqueId, Vector3 position, uint nameId)>>();
            if (!string.IsNullOrEmpty(originalSplitPath) && Directory.Exists(originalSplitPath))
            {
                Console.WriteLine("\n[Stage 0] Scanning Original Split ADTs for UniqueID restoration...");
                var obj0Files = Directory.GetFiles(originalSplitPath, "*_obj0.adt", SearchOption.AllDirectories);
                int totalOriginalIds = 0;
                
                foreach (var obj0File in obj0Files)
                {
                    var name = Path.GetFileNameWithoutExtension(obj0File).Replace("_obj0", "");
                    var match = System.Text.RegularExpressions.Regex.Match(name, @"(\d+)_(\d+)$");
                    if (!match.Success) continue;
                    
                    int tx = int.Parse(match.Groups[1].Value);
                    int ty = int.Parse(match.Groups[2].Value);
                    
                    try
                    {
                        var bytes = File.ReadAllBytes(obj0File);
                        var parsed = new AdtPatcher().ParseAdt(bytes);
                        
                        // Extract MODF entries
                        var modfChunk = parsed.FindChunk("MODF");
                        if (modfChunk?.Data != null && modfChunk.Data.Length >= 64)
                        {
                            var tileEntries = new List<(uint uniqueId, Vector3 position, uint nameId)>();
                            int entryCount = modfChunk.Data.Length / 64;
                            
                            for (int i = 0; i < entryCount; i++)
                            {
                                int offset = i * 64;
                                uint nameId = BitConverter.ToUInt32(modfChunk.Data, offset);
                                uint uniqueId = BitConverter.ToUInt32(modfChunk.Data, offset + 4);
                                float posX = BitConverter.ToSingle(modfChunk.Data, offset + 8);
                                float posZ = BitConverter.ToSingle(modfChunk.Data, offset + 12);
                                float posY = BitConverter.ToSingle(modfChunk.Data, offset + 16);
                                
                                tileEntries.Add((uniqueId, new Vector3(posX, posY, posZ), nameId));
                                totalOriginalIds++;
                            }
                            
                            if (tileEntries.Count > 0)
                                originalIdsByTile[(tx, ty)] = tileEntries;
                        }
                        
                        // Extract MDDF entries
                        var mddfChunk = parsed.FindChunk("MDDF");
                        if (mddfChunk?.Data != null && mddfChunk.Data.Length >= 36)
                        {
                            if (!originalIdsByTile.TryGetValue((tx, ty), out var existingList))
                            {
                                existingList = new List<(uint, Vector3, uint)>();
                                originalIdsByTile[(tx, ty)] = existingList;
                            }
                            
                            int entryCount = mddfChunk.Data.Length / 36;
                            for (int i = 0; i < entryCount; i++)
                            {
                                int offset = i * 36;
                                uint nameId = BitConverter.ToUInt32(mddfChunk.Data, offset);
                                uint uniqueId = BitConverter.ToUInt32(mddfChunk.Data, offset + 4);
                                float posX = BitConverter.ToSingle(mddfChunk.Data, offset + 8);
                                float posZ = BitConverter.ToSingle(mddfChunk.Data, offset + 12);
                                float posY = BitConverter.ToSingle(mddfChunk.Data, offset + 16);
                                
                                existingList.Add((uniqueId, new Vector3(posX, posY, posZ), nameId));
                                totalOriginalIds++;
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[WARN] Failed to parse {obj0File}: {ex.Message}");
                    }
                }
                
                Console.WriteLine($"[INFO] Loaded {totalOriginalIds} original UniqueIDs from {originalIdsByTile.Count} tiles.");
            }
            else
            {
                Console.WriteLine("[INFO] No --original-split provided. Skipping Stage 0 (OriginalID restoration).");
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

            // Add all original IDs (from Stage 0) to the global set to prevent conflicts
            foreach (var tileData in originalIdsByTile.Values)
            {
                foreach (var (uniqueId, _, _) in tileData)
                {
                    globalUsedUniqueIds.Add(uniqueId);
                }
            }
            Console.WriteLine($"[INFO] After Stage 0 integration: {globalUsedUniqueIds.Count} total reserved UniqueIds");

            // Inject Modf
            // Group by tiles - use TileX/TileY from PM4 filename (not calculated from position)
            // GLOBAL UniqueId tracking: IDs must be unique across ALL tiles, not just per-tile
            // Also track proximity to existing placements to avoid duplicates
            var modfByTile = new Dictionary<(int x, int y), List<AdtPatcher.ModfEntry>>();
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
                    while (globalUsedUniqueIds.Contains(globalNextUniqueId))
                        globalNextUniqueId++;
                    uniqueId = globalNextUniqueId++;
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
                    _adtPatcher.PatchWmoPlacements(file, Path.Combine(dirs.PatchedMuseum, Path.GetFileName(file)), tileWmoNames, tileEntries, ref globalNextUniqueId);
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
                    _adtPatcher.PatchWmoPlacements(file, outputPath, wmoNames, tileEntries, ref globalNextUniqueId);
                    File.Move(outputPath, file, true); // Replace original
                    wdlPatchedCount++;
                }
            }
            Console.WriteLine($"[INFO] Patched {wdlPatchedCount} WDL-generated ADTs with MODF data.");

            // Stage 4e: MDDF Injection (M2/Doodad placements)
            Console.WriteLine("\n[Stage 4e] Injecting MDDF (M2/Doodad) placements...");
            // Group MDDF entries by tile - use TileX/TileY from the entry
            var mddfByTile = new Dictionary<(int x, int y), List<AdtPatcher.MddfEntry>>();
            int mddfReassignedCount = 0;
            
            foreach (var entry in mddfEntries)
            {
                // NOTE: TileX/TileY are correct from PM4 parsing (Group 1 is X, Group 2 is Y)
                // NO SWAP needed.
                var (tx, ty) = (entry.TileX, entry.TileY);
                if (!mddfByTile.TryGetValue((tx, ty), out var list))
                {
                    list = new List<AdtPatcher.MddfEntry>();
                    mddfByTile[(tx, ty)] = list;
                }
                
                // Ensure UniqueId is globally unique (across both MODF AND MDDF)
                uint uniqueId = entry.UniqueId;
                if (globalUsedUniqueIds.Contains(uniqueId))
                {
                    while (globalUsedUniqueIds.Contains(globalNextUniqueId))
                        globalNextUniqueId++;
                    uniqueId = globalNextUniqueId++;
                    mddfReassignedCount++;
                }
                globalUsedUniqueIds.Add(uniqueId);
                
                // Map Core MddfEntry to AdtPatcher MddfEntry
                // Per ADT v18 spec, MDDF position encoding is:
                //   mddf.position.X = 32 * TILESIZE - world.X
                //   mddf.position.Y = world.Y (height unchanged)
                //   mddf.position.Z = 32 * TILESIZE - world.Z
                // This is same formula as MODF but note axis layout is XYZ where Y is height
                const float TILESIZE = 533.33333f;
                const float MAP_CENTER = 32.0f * TILESIZE; // 17066.666...
                var mddfPosition = new Vector3(
                    MAP_CENTER - entry.Position.X,  // X transform
                    entry.Position.Y,                // Y (height) unchanged
                    MAP_CENTER - entry.Position.Z   // Z transform
                );
                
                list.Add(new AdtPatcher.MddfEntry
                {
                    NameId = entry.NameId,
                    UniqueId = uniqueId,
                    Position = mddfPosition,
                    Rotation = entry.Rotation,
                    // Clamp scale to reasonable range: 512-2048 (0.5x - 2.0x)
                    // Computed scales can be insane (100x+) due to geometry matching issues
                    Scale = Math.Clamp(entry.Scale, (ushort)512, (ushort)2048),
                    Flags = entry.Flags
                });
            }
            
            if (mddfReassignedCount > 0)
                Console.WriteLine($"[INFO] Reassigned {mddfReassignedCount} MDDF UniqueIds for global uniqueness");
            
            // Patch MDDF into Museum ADTs (already created by MODF patching)
            int mddfPatchedCount = 0;
            foreach (var file in Directory.GetFiles(dirs.PatchedMuseum, "*.adt"))
            {
                var name = Path.GetFileNameWithoutExtension(file);
                if (name.Contains("_obj") || name.Contains("_tex")) continue;

                var match = System.Text.RegularExpressions.Regex.Match(name, @"(\d+)_(\d+)$");
                if (!match.Success) continue;

                int tx = int.Parse(match.Groups[1].Value);
                int ty = int.Parse(match.Groups[2].Value);

                if (mddfByTile.TryGetValue((tx, ty), out var tileEntries) && tileEntries.Count > 0)
                {
                    // Patch MDDF into the already-patched ADT (in place)
                    _adtPatcher.PatchDoodadPlacements(file, file, m2Names, tileEntries, ref globalNextUniqueId);
                    mddfPatchedCount++;
                }
            }
            
            Console.WriteLine($"[INFO] Patched {mddfPatchedCount} ADTs with MDDF data ({mddfEntries.Count} total M2 placements)");


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

            // Stage 4f: Global UniqueID Deduplication
            // TEMPORARILY DISABLED: Corrupting MDDF data by finding false chunk matches
            Console.WriteLine("\n[Stage 4f] Reassigning UniqueIds for global uniqueness...");
            Console.WriteLine("[WARN] UniqueID reassignment is DISABLED - caused MDDF corruption");
            // TODO: Fix FindChunkPosition to not find false matches within strings
            if (false)  // DISABLED
            {
                var globalNextId = 1u; // Start from 1
                
                // Get ADT files from PatchedMuseum
                var adtFiles = Directory.GetFiles(dirs.PatchedMuseum, "*.adt")
                    .Where(f => !Path.GetFileName(f).Contains("_obj") && !Path.GetFileName(f).Contains("_tex"))
                    .OrderBy(f => f)
                    .ToList();
                
                Console.WriteLine($"[DEBUG] Found {adtFiles.Count} ADT files in {dirs.PatchedMuseum}");
                
                int totalModfReassigned = 0;
                int totalMddfReassigned = 0;
                
                foreach (var adtPath in adtFiles)
                {
                    try
                    {
                        var bytes = File.ReadAllBytes(adtPath);
                        bool modified = false;
                        
                        // Scan for MODF chunk and reassign UniqueIds (bytes 4-7 of each 64-byte entry)
                        int modfPos = FindChunkPosition(bytes, "MODF");
                        
                        // Debug: log first file's chunk findings
                        if (adtPath == adtFiles[0])
                        {
                            Console.WriteLine($"[DEBUG] First file: {Path.GetFileName(adtPath)}, size: {bytes.Length} bytes");
                            Console.WriteLine($"[DEBUG] MODF pos: {modfPos}, MDDF pos: {FindChunkPosition(bytes, "MDDF")}");
                        }
                        
                        if (modfPos >= 0)
                        {
                            int modfSize = BitConverter.ToInt32(bytes, modfPos + 4);
                            int modfDataStart = modfPos + 8;
                            int entryCount = modfSize / 64;
                            for (int i = 0; i < entryCount; i++)
                            {
                                int offset = modfDataStart + i * 64 + 4; // UniqueId at byte 4
                                if (offset + 4 <= bytes.Length)
                                {
                                    var newIdBytes = BitConverter.GetBytes(globalNextId++);
                                    Buffer.BlockCopy(newIdBytes, 0, bytes, offset, 4);
                                    totalModfReassigned++;
                                    modified = true;
                                }
                            }
                        }
                        
                        // Scan for MDDF chunk and reassign UniqueIds (bytes 4-7 of each 36-byte entry)
                        int mddfPos = FindChunkPosition(bytes, "MDDF");
                        if (mddfPos >= 0)
                        {
                            int mddfSize = BitConverter.ToInt32(bytes, mddfPos + 4);
                            int mddfDataStart = mddfPos + 8;
                            int entryCount = mddfSize / 36;
                            for (int i = 0; i < entryCount; i++)
                            {
                                int offset = mddfDataStart + i * 36 + 4; // UniqueId at byte 4
                                if (offset + 4 <= bytes.Length)
                                {
                                    var newIdBytes = BitConverter.GetBytes(globalNextId++);
                                    Buffer.BlockCopy(newIdBytes, 0, bytes, offset, 4);
                                    totalMddfReassigned++;
                                    modified = true;
                                }
                            }
                        }
                        
                        if (modified)
                            File.WriteAllBytes(adtPath, bytes);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[WARN] Failed to reassign UniqueIds in {Path.GetFileName(adtPath)}: {ex.Message}");
                    }
                }
                
                Console.WriteLine($"[INFO] Reassigned {totalModfReassigned} MODF + {totalMddfReassigned} MDDF UniqueIds (total: {globalNextId - 1})");
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
        /// Export PM4 candidates as OBJ files for decoder verification.
        /// Uses Pm4Decoder + Pm4ObjectBuilder to extract geometry.
        /// </summary>
        private void ExportPm4CandidatesToObj(string pm4Directory, string outputDir)
        {
            try
            {
                var extractor = new Pipeline.Pm4ObjectExtractor();
                var candidates = extractor.ExtractAllWmoCandidates(pm4Directory).ToList();
                Pipeline.Pm4ObjectExtractor.ExportCandidatesToObj(candidates, outputDir);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to export PM4 OBJs: {ex.Message}");
            }
        }

        /// <summary>
        /// Load PM4 objects directly from .pm4 files using new Pm4Decoder + Pm4ObjectBuilder.
        /// Uses MdosIndex-based MSCN linking (not bounding-box heuristics) for accurate geometry.
        /// </summary>
        public List<Pm4ModfReconstructor.Pm4Object> LoadPm4ObjectsFromFiles(string pm4Directory)
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
            
            Console.WriteLine($"[INFO] Found {pm4Files.Length} PM4 files to process (using Pm4Decoder)...");
            
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
                    
                    // Use new Pm4Decoder (with correct coordinate handling)
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var decoded = Decoding.Pm4Decoder.Decode(pm4Data);
                    
                    if (decoded.Surfaces.Count == 0 || decoded.MeshVertices.Count == 0)
                        continue;
                    
                    // Use Pm4ObjectBuilder (with MdosIndex-based MSCN linking)
                    var candidates = Decoding.Pm4ObjectBuilder.BuildCandidates(decoded, tileX, tileY);
                    
                    foreach (var candidate in candidates)
                    {
                        if (candidate.DebugGeometry == null || candidate.DebugGeometry.Count < 10)
                            continue;
                        
                        string ck24Hex = $"0x{candidate.CK24:X6}";
                        var stats = matcher.ComputeStats(candidate.DebugGeometry);
                        
                        // Include MSCN points, MPRL rotation and position
                        objects.Add(new Pm4ModfReconstructor.Pm4Object(
                            ck24Hex, pm4Path, tileX, tileY, stats, 
                            candidate.DebugMscnVertices,
                            candidate.MprlRotationDegrees,
                            candidate.MprlPosition));  // Pass MPRL position
                    }
                    
                    // Also extract M2 candidates from IsM2Bucket surfaces (GroupKey == 0)
                    var m2Surfaces = decoded.Surfaces.Where(s => s.GroupKey == 0).ToList();
                    if (m2Surfaces.Count > 0)
                    {
                        // Use existing PM4File for M2 clustering (reuse old logic for now)
                        var pm4Legacy = new PM4File(pm4Data);
                        var clusters = Pm4GeometryClusterer.ClusterSurfaces(
                            pm4Legacy.Surfaces.Where(s => s.IsM2Bucket).ToList(),
                            pm4Legacy.MeshIndices,
                            pm4Legacy.MeshVertices);
                        
                        int m2Count = 0;
                        foreach (var cluster in clusters)
                        {
                            if (cluster.TriangleCount < 10 || cluster.TriangleCount > 5000) continue;
                            
                            string candidateId = $"M2_{tileX}_{tileY}_{m2Count++}";
                            var stats = matcher.ComputeStats(cluster.Vertices);
                            objects.Add(new Pm4ModfReconstructor.Pm4Object(candidateId, pm4Path, tileX, tileY, stats));
                        }
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
        public void ExportMprlRotationData(string pm4Directory, string outputDir, 
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
        public void ExportMslkCk24Analysis(string pm4Directory, string outputDir)
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
            
            // Cross-chunk correlation deep dive
            ExportCrossChunkCorrelation(pm4Directory, outputDir);
            
            // Chunk inventory to find missing data
            ExportChunkInventory(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Deep correlation analysis between MSLK, MPRL, and MPRR to find object linkage patterns.
        /// Goal: Find the key that uniquely identifies each object instance.
        /// </summary>
        public void ExportCrossChunkCorrelation(string pm4Directory, string outputDir)
        {
            var correlationPath = Path.Combine(outputDir, "cross_chunk_correlation.txt");
            
            // Collect data across all files
            var refIndexToMprlMatch = new Dictionary<string, int>(); // Does RefIndex < MPRL count?
            var refIndexDistribution = new Dictionary<ushort, int>();
            var groupIdDistribution = new Dictionary<uint, int>();
            var mslkWithGeometry = 0;
            var mslkWithoutGeometry = 0;
            var totalMslk = 0;
            var totalMprl = 0;
            
            // Sample entries for inspection
            var sampleMatches = new List<(string File, int MslkIdx, MslkEntry Mslk, MprlEntry? Mprl)>();
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    totalMprl += pm4.PositionRefs.Count;
                    
                    for (int i = 0; i < pm4.LinkEntries.Count; i++)
                    {
                        var mslk = pm4.LinkEntries[i];
                        totalMslk++;
                        
                        if (mslk.HasGeometry)
                            mslkWithGeometry++;
                        else
                            mslkWithoutGeometry++;
                        
                        // Track RefIndex distribution
                        if (!refIndexDistribution.ContainsKey(mslk.RefIndex))
                            refIndexDistribution[mslk.RefIndex] = 0;
                        refIndexDistribution[mslk.RefIndex]++;
                        
                        // Track GroupObjectId distribution
                        if (!groupIdDistribution.ContainsKey(mslk.GroupObjectId))
                            groupIdDistribution[mslk.GroupObjectId] = 0;
                        groupIdDistribution[mslk.GroupObjectId]++;
                        
                        // Check if RefIndex is a valid MPRL index
                        bool refInMprl = mslk.RefIndex < pm4.PositionRefs.Count;
                        var key = refInMprl ? "RefIndex_valid" : "RefIndex_invalid";
                        if (!refIndexToMprlMatch.ContainsKey(key))
                            refIndexToMprlMatch[key] = 0;
                        refIndexToMprlMatch[key]++;
                        
                        // Sample some entries with valid RefIndex -> MPRL link
                        if (sampleMatches.Count < 50 && refInMprl && mslk.HasGeometry)
                        {
                            var mprl = pm4.PositionRefs[mslk.RefIndex];
                            sampleMatches.Add((fileName, i, mslk, mprl));
                        }
                    }
                }
                catch { /* ignore */ }
            }
            
            using (var sw = new StreamWriter(correlationPath))
            {
                sw.WriteLine("=== Cross-Chunk Correlation Analysis ===");
                sw.WriteLine($"Total MSLK entries: {totalMslk:N0}");
                sw.WriteLine($"Total MPRL entries: {totalMprl:N0}");
                sw.WriteLine();
                
                sw.WriteLine("=== MSLK Geometry Status ===");
                sw.WriteLine($"  With geometry (MspiFirstIndex >= 0): {mslkWithGeometry:N0} ({100.0 * mslkWithGeometry / totalMslk:F1}%)");
                sw.WriteLine($"  Without geometry (MspiFirstIndex = -1): {mslkWithoutGeometry:N0}");
                sw.WriteLine();
                
                sw.WriteLine("=== MSLK.RefIndex → MPRL Correlation ===");
                foreach (var (key, count) in refIndexToMprlMatch)
                {
                    sw.WriteLine($"  {key}: {count:N0} ({100.0 * count / totalMslk:F1}%)");
                }
                sw.WriteLine();
                
                sw.WriteLine("=== RefIndex Distribution (top 30) ===");
                foreach (var (val, count) in refIndexDistribution.OrderByDescending(x => x.Value).Take(30))
                {
                    sw.WriteLine($"  RefIndex={val,5}: {count,7} MSLK entries");
                }
                sw.WriteLine($"  Unique RefIndex values: {refIndexDistribution.Count}");
                sw.WriteLine();
                
                sw.WriteLine("=== GroupObjectId Distribution ===");
                sw.WriteLine($"  Unique GroupObjectId values: {groupIdDistribution.Count}");
                var topGroups = groupIdDistribution.OrderByDescending(x => x.Value).Take(10).ToList();
                sw.WriteLine("  Top 10 most common:");
                foreach (var (id, count) in topGroups)
                {
                    sw.WriteLine($"    GroupId=0x{id:X8}: {count,6} MSLK entries");
                }
                sw.WriteLine();
                
                sw.WriteLine("=== Sample MSLK → MPRL Links (RefIndex as index) ===");
                foreach (var (file, idx, mslk, mprl) in sampleMatches)
                {
                    if (mprl != null)
                    {
                        sw.WriteLine($"  {file} MSLK[{idx}]:");
                        sw.WriteLine($"    MSLK: Type={mslk.TypeFlags} Subtype={mslk.Subtype} GroupId=0x{mslk.GroupObjectId:X8} RefIdx={mslk.RefIndex}");
                        sw.WriteLine($"    MPRL[{mslk.RefIndex}]: pos=({mprl.PositionX:F1},{mprl.PositionY:F1},{mprl.PositionZ:F1}) unk04={mprl.Unknown0x04} unk14={mprl.Unknown0x14}");
                        sw.WriteLine();
                    }
                }
                
                sw.WriteLine("=== Hypothesis ===");
                sw.WriteLine("If RefIndex_valid is high, MSLK.RefIndex IS an index into MPRL!");
                sw.WriteLine("This would link object catalog entries to world positions.");
                sw.WriteLine("Combined with CK24/GroupObjectId, this could enable proper object grouping.");
            }
            
            Console.WriteLine($"[INFO] Cross-Chunk Correlation: {correlationPath}");
            Console.WriteLine($"[INFO] MSLK->MPRL valid links: {refIndexToMprlMatch.GetValueOrDefault("RefIndex_valid", 0):N0} / {totalMslk:N0}");
            
            // Deep dive on invalid RefIndex - are they cross-tile?
            ExportInvalidRefIndexAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Analyze invalid RefIndex values to understand if they're cross-tile references.
        /// </summary>
        public void ExportInvalidRefIndexAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "refindex_invalid_analysis.txt");
            
            // Build global MPRL index across all tiles
            var globalMprl = new List<(string File, int LocalIdx, MprlEntry Entry)>();
            var perFileMprlCounts = new Dictionary<string, int>();
            var invalidRefStats = new Dictionary<string, int>(); // category -> count
            var linkIdPatterns = new Dictionary<uint, int>();
            var sampleInvalid = new List<(string File, int MslkIdx, MslkEntry Mslk)>();
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories)
                .OrderBy(f => f).ToList();
            
            // First pass: build global MPRL list
            int globalOffset = 0;
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    perFileMprlCounts[fileName] = pm4.PositionRefs.Count;
                    
                    for (int i = 0; i < pm4.PositionRefs.Count; i++)
                    {
                        globalMprl.Add((fileName, i, pm4.PositionRefs[i]));
                    }
                    globalOffset += pm4.PositionRefs.Count;
                }
                catch { }
            }
            
            int totalGlobalMprl = globalMprl.Count;
            
            // Second pass: analyze invalid RefIndex
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    int localMprlCount = pm4.PositionRefs.Count;
                    
                    for (int i = 0; i < pm4.LinkEntries.Count; i++)
                    {
                        var mslk = pm4.LinkEntries[i];
                        
                        // Track LinkId patterns
                        if (!linkIdPatterns.ContainsKey(mslk.LinkId))
                            linkIdPatterns[mslk.LinkId] = 0;
                        linkIdPatterns[mslk.LinkId]++;
                        
                        if (mslk.RefIndex >= localMprlCount)
                        {
                            // Invalid local reference - what is it?
                            string category;
                            if (mslk.RefIndex < totalGlobalMprl)
                                category = "could_be_global";
                            else if (mslk.RefIndex == 0xFFFF)
                                category = "is_sentinel";
                            else if (mslk.RefIndex > 50000)
                                category = "very_large";
                            else
                                category = "medium_range";
                            
                            if (!invalidRefStats.ContainsKey(category))
                                invalidRefStats[category] = 0;
                            invalidRefStats[category]++;
                            
                            // Sample some invalid entries
                            if (sampleInvalid.Count < 30 && mslk.RefIndex != 0xFFFF)
                            {
                                sampleInvalid.Add((fileName, i, mslk));
                            }
                        }
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("=== Invalid RefIndex Deep Analysis ===");
                sw.WriteLine($"Total MPRL entries (global): {totalGlobalMprl:N0}");
                sw.WriteLine($"PM4 files analyzed: {pm4Files.Count}");
                sw.WriteLine();
                
                sw.WriteLine("=== Invalid RefIndex Categories ===");
                foreach (var (cat, count) in invalidRefStats.OrderByDescending(x => x.Value))
                {
                    sw.WriteLine($"  {cat}: {count:N0}");
                }
                sw.WriteLine();
                
                sw.WriteLine("=== LinkId Pattern Analysis ===");
                sw.WriteLine($"  Unique LinkId values: {linkIdPatterns.Count}");
                sw.WriteLine("  Top 20 most common:");
                foreach (var (id, count) in linkIdPatterns.OrderByDescending(x => x.Value).Take(20))
                {
                    // Parse LinkId: 0xFFFFYYXX format for cross-tile
                    byte xx = (byte)(id & 0xFF);
                    byte yy = (byte)((id >> 8) & 0xFF);
                    uint highWord = (id >> 16);
                    string interpretation = highWord == 0xFFFF ? $"cross-tile to ({xx},{yy})" : 
                                           id == 0 ? "local (no cross-tile)" : 
                                           $"unknown pattern";
                    sw.WriteLine($"    LinkId=0x{id:X8}: {count,6} ({interpretation})");
                }
                sw.WriteLine();
                
                sw.WriteLine("=== Sample Invalid RefIndex Entries ===");
                foreach (var (file, idx, mslk) in sampleInvalid)
                {
                    sw.WriteLine($"  {file} MSLK[{idx}]:");
                    sw.WriteLine($"    Type={mslk.TypeFlags} Subtype={mslk.Subtype} GroupId=0x{mslk.GroupObjectId:X8}");
                    sw.WriteLine($"    RefIdx={mslk.RefIndex} (invalid) LinkId=0x{mslk.LinkId:X8}");
                    sw.WriteLine();
                }
                
                sw.WriteLine("=== Key Findings ===");
                sw.WriteLine("If 'could_be_global' is high → RefIndex might be a global/cumulative index");
                sw.WriteLine("If LinkId has cross-tile patterns (0xFFFFYYXX) → Object spans multiple tiles");
                sw.WriteLine("If 'is_sentinel' is high → 0xFFFF means 'no reference'");
            }
            
            Console.WriteLine($"[INFO] Invalid RefIndex Analysis: {analysisPath}");
            Console.WriteLine($"[INFO] Global MPRL pool: {totalGlobalMprl:N0} entries across {pm4Files.Count} files");
            
            // Cross-tile MPRL resolution test
            ExportCrossTileMprlResolution(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Test if RefIndex references MPRL in the target tile specified by LinkId.
        /// LinkId format: 0x00FFXXYY where XX=tile X, YY=tile Y (or reversed).
        /// </summary>
        public void ExportCrossTileMprlResolution(string pm4Directory, string outputDir)
        {
            var resolutionPath = Path.Combine(outputDir, "cross_tile_mprl_resolution.txt");
            
            // Build per-tile MPRL lookup
            var tileMprl = new Dictionary<(int X, int Y), List<MprlEntry>>();
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Parse tile coords from filename (e.g., development_14_37.pm4)
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileNameWithoutExtension(pm4Path);
                    var parts = fileName.Split('_');
                    if (parts.Length >= 3 && int.TryParse(parts[^2], out int tileX) && int.TryParse(parts[^1], out int tileY))
                    {
                        var pm4Data = File.ReadAllBytes(pm4Path);
                        var pm4 = new PM4File(pm4Data);
                        tileMprl[(tileX, tileY)] = pm4.PositionRefs.ToList();
                    }
                }
                catch { }
            }
            
            // Test RefIndex resolution via LinkId target tile
            int resolved = 0;
            int unresolved = 0;
            var sampleResolved = new List<(string SrcFile, int MslkIdx, MslkEntry Mslk, string TgtTile, MprlEntry Mprl)>();
            var sampleUnresolved = new List<(string SrcFile, int MslkIdx, MslkEntry Mslk, string TargetTile, string Reason)>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var srcFileName = Path.GetFileName(pm4Path);
                    int localMprlCount = pm4.PositionRefs.Count;
                    
                    for (int i = 0; i < pm4.LinkEntries.Count; i++)
                    {
                        var mslk = pm4.LinkEntries[i];
                        
                        // Only process entries with invalid local RefIndex
                        if (mslk.RefIndex >= localMprlCount && mslk.RefIndex != 0xFFFF)
                        {
                            // Parse target tile from LinkId (0x00FFXXYY format)
                            // Try both XY and YX interpretation
                            byte linkByte0 = (byte)(mslk.LinkId & 0xFF);
                            byte linkByte1 = (byte)((mslk.LinkId >> 8) & 0xFF);
                            
                            // Interpretation 1: byte0=X, byte1=Y
                            var tgtTileA = (linkByte0, linkByte1);
                            // Interpretation 2: byte0=Y, byte1=X  
                            var tgtTileB = (linkByte1, linkByte0);
                            
                            MprlEntry? resolvedMprl = null;
                            string targetTileStr = "";
                            
                            if (tileMprl.TryGetValue(tgtTileA, out var mprlListA) && mslk.RefIndex < mprlListA.Count)
                            {
                                resolvedMprl = mprlListA[mslk.RefIndex];
                                targetTileStr = $"({tgtTileA.Item1},{tgtTileA.Item2})";
                            }
                            else if (tileMprl.TryGetValue(tgtTileB, out var mprlListB) && mslk.RefIndex < mprlListB.Count)
                            {
                                resolvedMprl = mprlListB[mslk.RefIndex];
                                targetTileStr = $"({tgtTileB.Item1},{tgtTileB.Item2})";
                            }
                            
                            if (resolvedMprl != null)
                            {
                                resolved++;
                                if (sampleResolved.Count < 20)
                                {
                                    sampleResolved.Add((srcFileName, i, mslk, targetTileStr, resolvedMprl));
                                }
                            }
                            else
                            {
                                unresolved++;
                                if (sampleUnresolved.Count < 10)
                                {
                                    string reason = tileMprl.ContainsKey(tgtTileA) ? 
                                        $"RefIdx {mslk.RefIndex} >= {tileMprl[tgtTileA].Count}" :
                                        $"tile ({linkByte0},{linkByte1}) not found";
                                    sampleUnresolved.Add((srcFileName, i, mslk, $"({linkByte0},{linkByte1})", reason));
                                }
                            }
                        }
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(resolutionPath))
            {
                sw.WriteLine("=== Cross-Tile MPRL Resolution Test ===");
                sw.WriteLine($"Tiles loaded: {tileMprl.Count}");
                sw.WriteLine();
                
                sw.WriteLine("=== Resolution Results ===");
                sw.WriteLine($"  Resolved via target tile: {resolved:N0}");
                sw.WriteLine($"  Unresolved: {unresolved:N0}");
                sw.WriteLine($"  Resolution rate: {100.0 * resolved / Math.Max(1, resolved + unresolved):F1}%");
                sw.WriteLine();
                
                sw.WriteLine("=== Sample Resolved Entries ===");
                foreach (var (src, idx, mslk, tgt, mprl) in sampleResolved)
                {
                    sw.WriteLine($"  {src} MSLK[{idx}] → target tile {tgt}:");
                    sw.WriteLine($"    MSLK: GroupId=0x{mslk.GroupObjectId:X8} RefIdx={mslk.RefIndex} LinkId=0x{mslk.LinkId:X8}");
                    sw.WriteLine($"    MPRL: pos=({mprl.PositionX:F1},{mprl.PositionY:F1},{mprl.PositionZ:F1}) unk04={mprl.Unknown0x04}");
                    sw.WriteLine();
                }
                
                if (sampleUnresolved.Count > 0)
                {
                    sw.WriteLine("=== Sample Unresolved Entries ===");
                    foreach (var (src, idx, mslk, tgt, reason) in sampleUnresolved)
                    {
                        sw.WriteLine($"  {src} MSLK[{idx}] → {tgt}: {reason}");
                    }
                }
            }
            
            Console.WriteLine($"[INFO] Cross-Tile MPRL Resolution: {resolutionPath}");
            Console.WriteLine($"[INFO] Resolved: {resolved:N0} / {resolved + unresolved:N0} ({100.0 * resolved / Math.Max(1, resolved + unresolved):F1}%)");
            
            // MSHD header analysis
            ExportMshdAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Dump MSHD header values to understand their meaning.
        /// </summary>
        public void ExportMshdAnalysis(string pm4Directory, string outputDir)
        {
            var mshdPath = Path.Combine(outputDir, "mshd_header_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories)
                .OrderBy(f => f).ToList();
            
            // Track value distributions
            var fieldDistributions = new Dictionary<string, Dictionary<uint, int>>();
            for (int i = 0; i < 8; i++)
                fieldDistributions[$"Unk{i * 4:X2}"] = new Dictionary<uint, int>();
            
            var sampleHeaders = new List<(string File, PM4Header Header, int MprlCount, int MslkCount)>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    if (pm4.Header != null)
                    {
                        // Track field distributions
                        var values = new[]
                        {
                            pm4.Header.Unk00, pm4.Header.Unk04, pm4.Header.Unk08, pm4.Header.Unk0C,
                            pm4.Header.Unk10, pm4.Header.Unk14, pm4.Header.Unk18, pm4.Header.Unk1C
                        };
                        
                        for (int i = 0; i < 8; i++)
                        {
                            var field = $"Unk{i * 4:X2}";
                            if (!fieldDistributions[field].ContainsKey(values[i]))
                                fieldDistributions[field][values[i]] = 0;
                            fieldDistributions[field][values[i]]++;
                        }
                        
                        // Sample some headers
                        if (sampleHeaders.Count < 30)
                        {
                            sampleHeaders.Add((fileName, pm4.Header, pm4.PositionRefs.Count, pm4.LinkEntries.Count));
                        }
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(mshdPath))
            {
                sw.WriteLine("=== MSHD Header Analysis ===");
                sw.WriteLine($"Files analyzed: {pm4Files.Count}");
                sw.WriteLine();
                
                sw.WriteLine("=== Field Value Distributions ===");
                foreach (var (field, dist) in fieldDistributions)
                {
                    sw.WriteLine($"\n{field}:");
                    foreach (var (val, count) in dist.OrderByDescending(x => x.Value).Take(10))
                    {
                        sw.WriteLine($"  0x{val:X8} ({val,10}): {count,4} files");
                    }
                    if (dist.Count > 10)
                        sw.WriteLine($"  ... {dist.Count - 10} more unique values");
                }
                
                sw.WriteLine("\n\n=== Sample Headers (with chunk counts) ===");
                sw.WriteLine("File                          | Unk00    Unk04    Unk08    Unk0C    | MPRL   MSLK");
                sw.WriteLine("------------------------------|----------------------------------------|---------------");
                foreach (var (file, h, mprl, mslk) in sampleHeaders)
                {
                    sw.WriteLine($"{file,-30}| {h.Unk00,8} {h.Unk04,8} {h.Unk08,8} {h.Unk0C,8} | {mprl,6} {mslk,6}");
                }
                
                sw.WriteLine("\n\n=== Hypothesis ===");
                sw.WriteLine("Look for:");
                sw.WriteLine("  - Values that match chunk counts (MPRL, MSLK, MSUR)");
                sw.WriteLine("  - Values that could be cumulative offsets");
                sw.WriteLine("  - Values that correlate with tile coordinates");
            }
            
            Console.WriteLine($"[INFO] MSHD Header Analysis: {mshdPath}");
            
            // Comprehensive relationship analysis
            ExportComprehensiveRelationshipAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Comprehensive analysis of PM4 chunk relationships including CK24 decomposition.
        /// Treats PM4 as a database with chunks as tables and indexes as foreign keys.
        /// </summary>
        public void ExportComprehensiveRelationshipAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "pm4_relationship_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // CK24 component analysis
            var ck24Byte0 = new Dictionary<byte, int>(); // Low byte
            var ck24Byte1 = new Dictionary<byte, int>(); // Mid byte  
            var ck24Byte2 = new Dictionary<byte, int>(); // High byte
            var ck24ByZ = new Dictionary<int, List<(uint CK24, float AvgZ)>>();
            
            // Index field ranges  
            int maxMspiFirst = 0, maxMsviFirst = 0, maxRefIndex = 0;
            var allMspiCounts = new List<byte>();
            var allIndexCounts = new List<byte>();
            
            // Relationship samples
            var mslkToMspv = new Dictionary<int, List<int>>(); // MSLK -> MSPV via MSPI
            var msurToMsvt = new Dictionary<int, List<int>>(); // MSUR -> MSVT via MSVI
            var sampleChains = new List<string>();
            
            foreach (var pm4Path in pm4Files.Take(20)) // Sample first 20 files
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    // CK24 decomposition
                    foreach (var surf in pm4.Surfaces)
                    {
                        uint ck24 = surf.CK24;
                        byte b0 = (byte)(ck24 & 0xFF);
                        byte b1 = (byte)((ck24 >> 8) & 0xFF);
                        byte b2 = (byte)((ck24 >> 16) & 0xFF);
                        
                        if (!ck24Byte0.ContainsKey(b0)) ck24Byte0[b0] = 0;
                        if (!ck24Byte1.ContainsKey(b1)) ck24Byte1[b1] = 0;
                        if (!ck24Byte2.ContainsKey(b2)) ck24Byte2[b2] = 0;
                        ck24Byte0[b0]++;
                        ck24Byte1[b1]++;
                        ck24Byte2[b2]++;
                    }
                    
                    // Track index ranges
                    foreach (var mslk in pm4.LinkEntries)
                    {
                        if (mslk.MspiFirstIndex > maxMspiFirst) maxMspiFirst = mslk.MspiFirstIndex;
                        if (mslk.RefIndex > maxRefIndex && mslk.RefIndex != 0xFFFF) maxRefIndex = mslk.RefIndex;
                        allMspiCounts.Add(mslk.MspiIndexCount);
                    }
                    
                    foreach (var surf in pm4.Surfaces)
                    {
                        if (surf.MsviFirstIndex > maxMsviFirst) maxMsviFirst = (int)surf.MsviFirstIndex;
                        allIndexCounts.Add(surf.IndexCount);
                    }
                    
                    // Sample a few relationship chains
                    if (sampleChains.Count < 10 && pm4.LinkEntries.Count > 0 && pm4.PathIndices.Count > 0)
                    {
                        var mslk = pm4.LinkEntries[0];
                        if (mslk.MspiFirstIndex >= 0 && mslk.MspiFirstIndex + mslk.MspiIndexCount <= pm4.PathIndices.Count)
                        {
                            var pathIndices = new List<uint>();
                            for (int i = 0; i < mslk.MspiIndexCount; i++)
                                pathIndices.Add(pm4.PathIndices[mslk.MspiFirstIndex + i]);
                            
                            sampleChains.Add($"{fileName} MSLK[0] -> MSPI[{mslk.MspiFirstIndex}:{mslk.MspiIndexCount}] -> paths:{string.Join(",", pathIndices.Take(5))}");
                        }
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("╔════════════════════════════════════════════════════════════════╗");
                sw.WriteLine("║        PM4 COMPREHENSIVE RELATIONSHIP ANALYSIS                 ║");
                sw.WriteLine("║  Treating PM4 as a database with chunks as tables              ║");
                sw.WriteLine("╚════════════════════════════════════════════════════════════════╝");
                sw.WriteLine();
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("CHUNK RELATIONSHIP MAP (Database Schema)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
    ┌──────────────────────────────────────────────────────────────┐
    │                      PM4 SCENE GRAPH                         │
    └──────────────────────────────────────────────────────────────┘
    
    MSHD (Header)
      │
      ├─► MSLK[n] ──┬──► MSPI[first:count] ──► MSPV (Path vertices)
      │   │         │     (navigation paths)
      │   │         │
      │   │         └──► RefIndex ──► MPRL? (position reference?)
      │   │
      │   ├── TypeFlags (object type: 1=walkable, 2=wall, etc.)
      │   ├── Subtype (floor level 0-18)
      │   ├── GroupObjectId (local grouping)
      │   └── LinkId (cross-tile reference: 0x00FFXXYY)
      │
      ├─► MSUR[n] ──┬──► MSVI[first:count] ──► MSVT (Mesh vertices)
      │   │         │     (renderable surfaces)
      │   │         │
      │   │         └──► CK24 (object grouping key from PackedParams)
      │   │
      │   ├── Normal (surface orientation)
      │   ├── Height (Z level)
      │   └── AttributeMask (bit7 = liquid?)
      │
      ├─► MPRL[n] (Position references)
      │   ├── Position (X, Y, Z)
      │   ├── Unk14 (floor level, -1 = command)
      │   ├── Unk16 (0x3FFF = terminator)
      │   └── Unk04 (NOT rotation - varies at same position)
      │
      ├─► MPRR[n] (Object boundaries)  
      │   ├── Value1 (0xFFFF = SENTINEL)
      │   └── Value2 (component type)
      │
      └─► MSCN[n] (Exterior vertices for collision hull)
");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("CK24 BYTE DECOMPOSITION (Testing Z-layer hypothesis)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("CK24 = ((PackedParams >> 8) & 0xFFFFFF)");
                sw.WriteLine("Decomposed: [Byte2][Byte1][Byte0]");
                sw.WriteLine();
                
                sw.WriteLine("Byte 0 (Low byte) distribution:");
                foreach (var (val, count) in ck24Byte0.OrderByDescending(x => x.Value).Take(15))
                    sw.WriteLine($"  0x{val:X2}: {count,6}");
                sw.WriteLine($"  Unique values: {ck24Byte0.Count}");
                
                sw.WriteLine("\nByte 1 (Mid byte) distribution:");
                foreach (var (val, count) in ck24Byte1.OrderByDescending(x => x.Value).Take(15))
                    sw.WriteLine($"  0x{val:X2}: {count,6}");
                sw.WriteLine($"  Unique values: {ck24Byte1.Count}");
                
                sw.WriteLine("\nByte 2 (High byte) distribution:");
                foreach (var (val, count) in ck24Byte2.OrderByDescending(x => x.Value).Take(15))
                    sw.WriteLine($"  0x{val:X2}: {count,6}");
                sw.WriteLine($"  Unique values: {ck24Byte2.Count}");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("INDEX FIELD ANALYSIS");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine($"MSLK.MspiFirstIndex max: {maxMspiFirst}");
                sw.WriteLine($"MSLK.MspiIndexCount distribution: min={allMspiCounts.DefaultIfEmpty((byte)0).Min()}, max={allMspiCounts.DefaultIfEmpty((byte)0).Max()}, avg={allMspiCounts.Select(x => (double)x).DefaultIfEmpty(0).Average():F1}");
                sw.WriteLine($"MSLK.RefIndex max (excl 0xFFFF): {maxRefIndex}");
                sw.WriteLine($"MSUR.MsviFirstIndex max: {maxMsviFirst}");
                sw.WriteLine($"MSUR.IndexCount distribution: min={allIndexCounts.DefaultIfEmpty((byte)0).Min()}, max={allIndexCounts.DefaultIfEmpty((byte)0).Max()}, avg={allIndexCounts.Select(x => (double)x).DefaultIfEmpty(0).Average():F1}");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("SAMPLE RELATIONSHIP CHAINS");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                foreach (var chain in sampleChains)
                    sw.WriteLine($"  {chain}");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("UNKNOWN/UNMAPPED RELATIONSHIPS");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
  ❓ MSLK.RefIndex → What does it reference when > local MPRL count?
     - NOT cross-tile via LinkId (0% resolution rate)
     - NOT global cumulative index
     - Could be: object definition ID? WMO file hash? External reference?
  
  ❓ MSLK.GroupObjectId → Relationship to CK24?
     - No direct correlation found
     - Could be: local scene graph node ID
  
  ❓ MPRR Value1/Value2 → What geometry do they reference?
     - Value1=0xFFFF marks boundaries
     - Value2 meaning unknown (component type?)
  
  ❓ MPRL.Unk04 → Multiple values at same position
     - NOT rotation angle
     - Could be: LOD level? Animation state? Event trigger?
  
  ❓ CK24 byte components → Z-layer separation?
     - User hypothesis: different bytes = different Z levels
     - Need to test: group surfaces by CK24 byte, check Z ranges
  
  ❓ MSHD fields → Index offsets into global data?
     - Unk00, Unk04, Unk08 have non-zero values
     - Could be: cumulative offsets, tile metadata, grid dimensions
");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("NEXT INVESTIGATION TARGETS");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
  1. CK24 Z-Layer Test: Group surfaces by CK24 byte, compute Z ranges
  2. MPRR-MSUR Correlation: Do MPRR entries count match MSUR surfaces?
  3. GroupObjectId-MPRL: Test if GroupObjectId indexes into MPRL
  4. RefIndex Pattern: Analyze RefIndex values as possible file hashes
  5. MSHD Cumulative: Test if MSHD values are cumulative across tiles
");
            }
            
            Console.WriteLine($"[INFO] PM4 Relationship Analysis: {analysisPath}");
            
            // CK24 Z-layer correlation test
            ExportCk24ZLayerAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Test if CK24 Byte2 (type flags) correlates with Z-levels.
        /// Hypothesis: Different Byte2 values = different floor/height layers.
        /// </summary>
        public void ExportCk24ZLayerAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "ck24_z_layer_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Track Z ranges by CK24 Byte2
            var zByByte2 = new Dictionary<byte, List<float>>();
            // Track full CK24 with Z values
            var ck24ZSamples = new Dictionary<uint, (float MinZ, float MaxZ, int Count)>();
            // Track Byte0+Byte1 combinations for same Byte2
            var lowByteCombos = new Dictionary<byte, HashSet<ushort>>();
            
            int totalSurfaces = 0;
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    // Get Z values from MSUR height/normal
                    foreach (var surf in pm4.Surfaces)
                    {
                        uint ck24 = surf.CK24;
                        byte b0 = (byte)(ck24 & 0xFF);
                        byte b1 = (byte)((ck24 >> 8) & 0xFF);
                        byte b2 = (byte)((ck24 >> 16) & 0xFF);
                        ushort lowBytes = (ushort)(ck24 & 0xFFFF); // Byte0 + Byte1
                        
                        float z = surf.Height; // Height field as Z
                        
                        // Track Z by Byte2 type
                        if (!zByByte2.ContainsKey(b2))
                            zByByte2[b2] = new List<float>();
                        zByByte2[b2].Add(z);
                        
                        // Track low byte combinations per Byte2
                        if (!lowByteCombos.ContainsKey(b2))
                            lowByteCombos[b2] = new HashSet<ushort>();
                        lowByteCombos[b2].Add(lowBytes);
                        
                        // Track per-CK24 Z ranges
                        if (!ck24ZSamples.ContainsKey(ck24))
                            ck24ZSamples[ck24] = (z, z, 0);
                        var (minZ, maxZ, count) = ck24ZSamples[ck24];
                        ck24ZSamples[ck24] = (Math.Min(minZ, z), Math.Max(maxZ, z), count + 1);
                        
                        totalSurfaces++;
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("╔════════════════════════════════════════════════════════════════╗");
                sw.WriteLine("║         CK24 Z-LAYER CORRELATION ANALYSIS                      ║");
                sw.WriteLine("║  Testing: Does Byte2 (type flag) correlate with Z height?     ║");
                sw.WriteLine("╚════════════════════════════════════════════════════════════════╝");
                sw.WriteLine();
                sw.WriteLine($"Total surfaces analyzed: {totalSurfaces:N0}");
                sw.WriteLine();
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("Z RANGES BY BYTE2 (Type Flag)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("Byte2  |  Count   |   Min Z   |   Max Z   |  Z Range  | LowByte Combos");
                sw.WriteLine("-------|----------|-----------|-----------|-----------|---------------");
                
                foreach (var (b2, zList) in zByByte2.OrderByDescending(x => x.Value.Count))
                {
                    if (zList.Count == 0) continue;
                    float minZ = zList.Min();
                    float maxZ = zList.Max();
                    float range = maxZ - minZ;
                    int combos = lowByteCombos.GetValueOrDefault(b2, new HashSet<ushort>()).Count;
                    
                    sw.WriteLine($"0x{b2:X2}   | {zList.Count,8} | {minZ,9:F1} | {maxZ,9:F1} | {range,9:F1} | {combos,5}");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("INTERPRETATION");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
If Z ranges for different Byte2 values are NON-OVERLAPPING:
  → Byte2 encodes FLOOR/LAYER index (Z-layer hypothesis CONFIRMED)
  
If Z ranges for different Byte2 values OVERLAP significantly:
  → Byte2 encodes OBJECT TYPE (geometry classification flag)
  → Byte0+Byte1 = object instance identifier
");
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("TOP CK24 VALUES BY SURFACE COUNT");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("CK24       | Count | Min Z   | Max Z   | Z Span | Byte2 | Byte1 | Byte0");
                sw.WriteLine("-----------|-------|---------|---------|--------|-------|-------|------");
                
                foreach (var (ck24, stats) in ck24ZSamples.OrderByDescending(x => x.Value.Count).Take(30))
                {
                    byte b0 = (byte)(ck24 & 0xFF);
                    byte b1 = (byte)((ck24 >> 8) & 0xFF);
                    byte b2 = (byte)((ck24 >> 16) & 0xFF);
                    float span = stats.MaxZ - stats.MinZ;
                    
                    sw.WriteLine($"0x{ck24:X6} | {stats.Count,5} | {stats.MinZ,7:F1} | {stats.MaxZ,7:F1} | {span,6:F1} | 0x{b2:X2}  | 0x{b1:X2}  | 0x{b0:X2}");
                }
                
                sw.WriteLine("\n════════════════════════════════════════════════════════════════");
                sw.WriteLine("KEY QUESTIONS ANSWERED");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                // Calculate overlap
                var byte2Ranges = zByByte2.Where(x => x.Value.Count > 100)
                    .ToDictionary(x => x.Key, x => (Min: x.Value.Min(), Max: x.Value.Max()));
                
                bool hasSignificantOverlap = false;
                var overlaps = new List<string>();
                
                var sortedTypes = byte2Ranges.OrderBy(x => x.Value.Min).ToList();
                for (int i = 0; i < sortedTypes.Count - 1; i++)
                {
                    for (int j = i + 1; j < sortedTypes.Count; j++)
                    {
                        var a = sortedTypes[i];
                        var b = sortedTypes[j];
                        // Check overlap
                        if (a.Value.Max > b.Value.Min && b.Value.Max > a.Value.Min)
                        {
                            float overlap = Math.Min(a.Value.Max, b.Value.Max) - Math.Max(a.Value.Min, b.Value.Min);
                            if (overlap > 10) // Significant overlap (> 10 units)
                            {
                                hasSignificantOverlap = true;
                                overlaps.Add($"0x{a.Key:X2} ↔ 0x{b.Key:X2}: {overlap:F1} units overlap");
                            }
                        }
                    }
                }
                
                if (hasSignificantOverlap)
                {
                    sw.WriteLine("❌ Z-Layer Hypothesis: REJECTED (significant overlaps found)");
                    sw.WriteLine("   Byte2 appears to encode OBJECT TYPE, not floor level.");
                    sw.WriteLine("\n   Overlapping types:");
                    foreach (var o in overlaps.Take(10))
                        sw.WriteLine($"     {o}");
                }
                else
                {
                    sw.WriteLine("✓ Z-Layer Hypothesis: POSSIBLE (no significant overlaps)");
                    sw.WriteLine("   Byte2 may encode floor/layer index!");
                }
            }
            
            Console.WriteLine($"[INFO] CK24 Z-Layer Analysis: {analysisPath}");
            
            // CK24 ObjectID grouping analysis
            ExportCk24ObjectIdAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Test if CK24 Byte0+Byte1 (ObjectID) can identify distinct geometry groups.
        /// Hypothesis: Same ObjectID = same building/object with compact bounding box.
        /// </summary>
        public void ExportCk24ObjectIdAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "ck24_objectid_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Track per-ObjectID (Byte0+Byte1) geometry stats
            // Key: (Byte2, ObjectID) to keep type+id separate
            var objectGroups = new Dictionary<(byte Type, ushort ObjectId), ObjectStats>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    foreach (var surf in pm4.Surfaces)
                    {
                        uint ck24 = surf.CK24;
                        byte b2 = (byte)((ck24 >> 16) & 0xFF);  // Type
                        ushort objId = (ushort)(ck24 & 0xFFFF); // ObjectId
                        
                        var key = (b2, objId);
                        if (!objectGroups.ContainsKey(key))
                            objectGroups[key] = new ObjectStats();
                        
                        var stats = objectGroups[key];
                        stats.SurfaceCount++;
                        stats.IndexCount += surf.IndexCount;
                        
                        // Track bounding box from height
                        float z = surf.Height;
                        if (stats.MinZ > z) stats.MinZ = z;
                        if (stats.MaxZ < z) stats.MaxZ = z;
                        
                        // Track file distribution
                        if (!stats.Files.Contains(fileName))
                            stats.Files.Add(fileName);
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("╔════════════════════════════════════════════════════════════════╗");
                sw.WriteLine("║       CK24 OBJECTID GROUPING ANALYSIS                          ║");
                sw.WriteLine("║  Testing: Does Byte0+Byte1 identify distinct objects?          ║");
                sw.WriteLine("╚════════════════════════════════════════════════════════════════╝");
                sw.WriteLine();
                sw.WriteLine($"Total object groups (Type+ObjectID): {objectGroups.Count:N0}");
                sw.WriteLine();
                
                // Group by type
                var byType = objectGroups.GroupBy(x => x.Key.Type)
                    .OrderByDescending(g => g.Sum(x => x.Value.SurfaceCount));
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("SUMMARY BY TYPE (Byte2)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("Type   | Objects | Total Surfs | Avg Surfs/Obj | Multi-File Objs");
                sw.WriteLine("-------|---------|-------------|---------------|----------------");
                
                foreach (var typeGroup in byType)
                {
                    int objCount = typeGroup.Count();
                    int totalSurfs = typeGroup.Sum(x => x.Value.SurfaceCount);
                    float avgSurfs = objCount > 0 ? (float)totalSurfs / objCount : 0;
                    int multiFile = typeGroup.Count(x => x.Value.Files.Count > 1);
                    
                    sw.WriteLine($"0x{typeGroup.Key:X2}   | {objCount,7} | {totalSurfs,11} | {avgSurfs,13:F1} | {multiFile,14}");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("OBJECT SIZE DISTRIBUTION");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                var sizeGroups = objectGroups.Values
                    .GroupBy(o => o.SurfaceCount switch { < 10 => "1-9", < 50 => "10-49", < 100 => "50-99", < 500 => "100-499", _ => "500+" })
                    .OrderBy(g => g.Key);
                
                foreach (var sg in sizeGroups)
                {
                    sw.WriteLine($"  {sg.Key} surfaces: {sg.Count()} objects");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("TOP 30 LARGEST OBJECTS (by surface count)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("Type+ObjID  | Surfaces | Indices | Z Span   | Files | Could be");
                sw.WriteLine("------------|----------|---------|----------|-------|----------");
                
                foreach (var (key, stats) in objectGroups.OrderByDescending(x => x.Value.SurfaceCount).Take(30))
                {
                    float zSpan = stats.MaxZ - stats.MinZ;
                    string guess = GuessObjectType(key.Type, stats);
                    sw.WriteLine($"0x{key.Type:X2}:{key.ObjectId:X4}  | {stats.SurfaceCount,8} | {stats.IndexCount,7} | {zSpan,8:F0} | {stats.Files.Count,5} | {guess}");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("CROSS-TILE OBJECTS (span multiple files)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                var crossTile = objectGroups.Where(x => x.Value.Files.Count > 1)
                    .OrderByDescending(x => x.Value.Files.Count).Take(15);
                
                foreach (var (key, stats) in crossTile)
                {
                    sw.WriteLine($"  0x{key.Type:X2}:{key.ObjectId:X4}: {stats.SurfaceCount} surfaces across {stats.Files.Count} tiles");
                    sw.WriteLine($"    Files: {string.Join(", ", stats.Files.Take(5))}{(stats.Files.Count > 5 ? "..." : "")}");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("INTERPRETATION");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
CK24 Structure (CONFIRMED):
  ┌────────────────────────────────────────┐
  │ Byte2  │  Byte1  │  Byte0             │
  │ (Type) │  (ObjectID high)  (low)      │
  └────────────────────────────────────────┘

Type flags (Byte2):
  - 0x40 bit = has pathfinding mesh
  - 0x80 bit = exterior/outdoor
  - 0x00 = terrain/default (no object ID)

ObjectID (Byte0+Byte1):
  - 16-bit identifier for unique object
  - Same ID across tiles = same building spanning tiles
  - Use with Type for unique key
");
            }
            
            Console.WriteLine($"[INFO] CK24 ObjectID Analysis: {analysisPath}");
            
            // RefIndex alternative hypothesis analysis
            ExportRefIndexAlternativeAnalysis(pm4Directory, outputDir);
        }
        
        private string GuessObjectType(byte type, ObjectStats stats)
        {
            if (type == 0x00) return "terrain";
            if (stats.Files.Count > 10) return "large building";
            if (stats.SurfaceCount > 1000) return "major structure";
            if (stats.SurfaceCount > 100) return "building";
            if ((type & 0x80) != 0) return "exterior obj";
            return "interior obj";
        }
        
        private class ObjectStats
        {
            public int SurfaceCount;
            public int IndexCount;
            public float MinZ = float.MaxValue;
            public float MaxZ = float.MinValue;
            public List<string> Files = new List<string>();
        }
        
        /// <summary>
        /// Test alternative hypotheses for what MSLK.RefIndex references.
        /// </summary>
        public void ExportRefIndexAlternativeAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "refindex_alternative_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // Hypothesis tracking
            int refMatchesMsur = 0, refMatchesMsvt = 0, refMatchesMsvi = 0;
            int refMatchesGroupId = 0, refInvalidButPatternedCount = 0;
            int totalInvalid = 0, totalValid = 0;
            
            // Track RefIndex value patterns
            var refIndexBitPatterns = new Dictionary<string, int>();
            var refIndexByHighByte = new Dictionary<byte, int>();
            var samplePatterns = new List<(string File, int MslkIdx, MslkEntry Mslk, int LocalMprl, int Msur, int Msvt)>();
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    var fileName = Path.GetFileName(pm4Path);
                    
                    int mprlCount = pm4.PositionRefs.Count;
                    int msurCount = pm4.Surfaces.Count;
                    int msvtCount = pm4.MeshVertices.Count;
                    int msviCount = pm4.MeshIndices.Count;
                    
                    for (int i = 0; i < pm4.LinkEntries.Count; i++)
                    {
                        var mslk = pm4.LinkEntries[i];
                        ushort refIdx = mslk.RefIndex;
                        
                        if (refIdx == 0xFFFF) continue; // Skip sentinel
                        
                        // Valid local MPRL?
                        if (refIdx < mprlCount)
                        {
                            totalValid++;
                        }
                        else
                        {
                            totalInvalid++;
                            
                            // Test alternative targets
                            if (refIdx < msurCount) refMatchesMsur++;
                            if (refIdx < msvtCount) refMatchesMsvt++;
                            if (refIdx < msviCount) refMatchesMsvi++;
                            
                            // Does it match GroupObjectId?
                            if (refIdx == (ushort)(mslk.GroupObjectId & 0xFFFF)) refMatchesGroupId++;
                            
                            // Analyze bit patterns
                            byte highByte = (byte)((refIdx >> 8) & 0xFF);
                            if (!refIndexByHighByte.ContainsKey(highByte))
                                refIndexByHighByte[highByte] = 0;
                            refIndexByHighByte[highByte]++;
                            
                            // Sample for analysis
                            if (samplePatterns.Count < 20)
                            {
                                samplePatterns.Add((fileName, i, mslk, mprlCount, msurCount, msvtCount));
                            }
                        }
                    }
                }
                catch { }
            }
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("╔════════════════════════════════════════════════════════════════╗");
                sw.WriteLine("║       REFINDEX ALTERNATIVE HYPOTHESIS ANALYSIS                 ║");
                sw.WriteLine("║  What does RefIndex reference when > local MPRL count?         ║");
                sw.WriteLine("╚════════════════════════════════════════════════════════════════╝");
                sw.WriteLine();
                sw.WriteLine($"Valid (< MPRL count): {totalValid:N0}");
                sw.WriteLine($"Invalid (>= MPRL count): {totalInvalid:N0}");
                sw.WriteLine();
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("ALTERNATIVE TARGET TESTS");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine($"  RefIndex < MSUR count: {refMatchesMsur:N0} ({100.0 * refMatchesMsur / Math.Max(1, totalInvalid):F1}%)");
                sw.WriteLine($"  RefIndex < MSVT count: {refMatchesMsvt:N0} ({100.0 * refMatchesMsvt / Math.Max(1, totalInvalid):F1}%)");
                sw.WriteLine($"  RefIndex < MSVI count: {refMatchesMsvi:N0} ({100.0 * refMatchesMsvi / Math.Max(1, totalInvalid):F1}%)");
                sw.WriteLine($"  RefIndex == GroupObjectId low word: {refMatchesGroupId:N0} ({100.0 * refMatchesGroupId / Math.Max(1, totalInvalid):F1}%)");
                sw.WriteLine();
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("REFINDEX HIGH BYTE DISTRIBUTION (for invalid refs)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("High Byte | Count    | Could mean");
                sw.WriteLine("----------|----------|-------------------------------------------");
                
                foreach (var (hb, count) in refIndexByHighByte.OrderByDescending(x => x.Value).Take(15))
                {
                    string interpretation = hb switch
                    {
                        0x00 => "low values (10xx - 40xx range)",
                        0x01 => "values 256-511",
                        0x02 => "values 512-767",
                        0x0F => "values 3840-4095",
                        0x10 => "values 4096-4351",
                        >= 0x80 => "high bit set - could be flag",
                        _ => ""
                    };
                    sw.WriteLine($"0x{hb:X2}      | {count,8} | {interpretation}");
                }
                
                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("SAMPLE INVALID REFINDEX WITH CONTEXT");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                foreach (var (file, idx, mslk, mprl, msur, msvt) in samplePatterns)
                {
                    sw.WriteLine($"  {file} MSLK[{idx}]:");
                    sw.WriteLine($"    RefIdx={mslk.RefIndex} (0x{mslk.RefIndex:X4})");
                    sw.WriteLine($"    Local counts: MPRL={mprl}, MSUR={msur}, MSVT={msvt}");
                    sw.WriteLine($"    GroupObjId=0x{mslk.GroupObjectId:X8} Type={mslk.TypeFlags} Subtype={mslk.Subtype}");
                    sw.WriteLine();
                }
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("HYPOTHESES");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine(@"
1. RefIndex is a GLOBAL index:
   - Into a master MPRL table combining all tiles
   - Requires knowing tile ordering/offsets

2. RefIndex is a PACKED value:
   - High byte = type/flag, Low byte = index
   - Or: X/Y grid coordinate packed

3. RefIndex references DIFFERENT chunk:
   - Test: does it fit MSUR, MSVT, MSVI counts?

4. RefIndex is EXTERNAL reference:
   - Points to WMO internal data
   - Or: PD4 file reference

5. RefIndex has SPECIAL meaning for certain TypeFlags:
   - Maybe only valid for certain MSLK types
");
            }
            
            Console.WriteLine($"[INFO] RefIndex Alternative Analysis: {analysisPath}");
        }
        
        /// <summary>
        /// Inventory all PM4 chunks to discover unparsed/unknown data.
        /// </summary>
        private void ExportChunkInventory(string pm4Directory, string outputDir)
        {
            var inventoryPath = Path.Combine(outputDir, "pm4_chunk_inventory.txt");
            
            var chunkTotals = new Dictionary<string, long>();
            var unparsedChunks = new Dictionary<string, int>();
            var mprrStats = new List<(string File, int Count, int Sequences)>();
            int totalFiles = 0;
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    totalFiles++;
                    
                    // Aggregate chunk sizes
                    foreach (var (chunk, size) in pm4.ChunkSizes)
                    {
                        if (!chunkTotals.ContainsKey(chunk))
                            chunkTotals[chunk] = 0;
                        chunkTotals[chunk] += size;
                    }
                    
                    // Track unparsed chunks
                    foreach (var unparsed in pm4.UnparsedChunks)
                    {
                        if (!unparsedChunks.ContainsKey(unparsed.Split(':')[0]))
                            unparsedChunks[unparsed.Split(':')[0]] = 0;
                        unparsedChunks[unparsed.Split(':')[0]]++;
                    }
                    
                    // MPRR statistics
                    if (pm4.MprrEntries.Count > 0)
                    {
                        int sentinels = pm4.MprrEntries.Count(e => e.IsSentinel);
                        mprrStats.Add((Path.GetFileName(pm4Path), pm4.MprrEntries.Count, sentinels));
                    }
                }
                catch { /* ignore */ }
            }
            
            using (var sw = new StreamWriter(inventoryPath))
            {
                sw.WriteLine("=== PM4 Chunk Inventory ===");
                sw.WriteLine($"Files analyzed: {totalFiles}");
                sw.WriteLine();
                
                sw.WriteLine("=== Chunk Sizes (Total across all files) ===");
                foreach (var (chunk, totalSize) in chunkTotals.OrderByDescending(x => x.Value))
                {
                    var sizeKb = totalSize / 1024.0;
                    sw.WriteLine($"  {chunk}: {totalSize:N0} bytes ({sizeKb:F1} KB)");
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Unparsed Chunks (potential missing data!) ===");
                if (unparsedChunks.Count == 0)
                {
                    sw.WriteLine("  None - all chunks are being parsed!");
                }
                else
                {
                    foreach (var (chunk, count) in unparsedChunks.OrderByDescending(x => x.Value))
                    {
                        sw.WriteLine($"  {chunk}: found in {count} files (NOT PARSED!)");
                    }
                }
                
                sw.WriteLine();
                sw.WriteLine("=== MPRR Statistics ===");
                sw.WriteLine($"  Files with MPRR: {mprrStats.Count}");
                if (mprrStats.Count > 0)
                {
                    var totalMprr = mprrStats.Sum(x => x.Count);
                    var totalSeq = mprrStats.Sum(x => x.Sequences);
                    sw.WriteLine($"  Total MPRR ushorts: {totalMprr:N0}");
                    sw.WriteLine($"  Total sequences (terminated by 0xFFFF): {totalSeq:N0}");
                    sw.WriteLine($"  Average entries per sequence: {(double)totalMprr / Math.Max(1, totalSeq):F1}");
                }
            }
            
            Console.WriteLine($"[INFO] Chunk Inventory: {inventoryPath}");
            Console.WriteLine($"[INFO] Unparsed chunks: {unparsedChunks.Count}");
            
            // Deep MPRR analysis
            ExportMprrAnalysis(pm4Directory, outputDir);
        }
        
        /// <summary>
        /// Deep analysis of MPRR entries to understand object grouping.
        /// MPRR entries are 4-byte (Value1, Value2) pairs where Value1=0xFFFF marks object boundaries.
        /// </summary>
        private void ExportMprrAnalysis(string pm4Directory, string outputDir)
        {
            var mprrAnalysisPath = Path.Combine(outputDir, "mprr_deep_analysis.txt");
            
            // Track patterns
            var value1Distribution = new Dictionary<ushort, int>();
            var value2Distribution = new Dictionary<ushort, int>();
            var objectSizes = new List<int>();  // Entries per object (between sentinels)
            var sampleObjects = new List<(string File, int ObjIdx, List<MprrEntry> Entries)>();
            int totalSentinels = 0;
            int totalEntries = 0;
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    if (pm4.MprrEntries.Count == 0) continue;
                    totalEntries += pm4.MprrEntries.Count;
                    
                    // Track value distributions
                    foreach (var entry in pm4.MprrEntries)
                    {
                        if (!value1Distribution.ContainsKey(entry.Value1))
                            value1Distribution[entry.Value1] = 0;
                        value1Distribution[entry.Value1]++;
                        
                        if (!value2Distribution.ContainsKey(entry.Value2))
                            value2Distribution[entry.Value2] = 0;
                        value2Distribution[entry.Value2]++;
                    }
                    
                    // Parse objects (separated by sentinels where Value1=0xFFFF)
                    var currentObject = new List<MprrEntry>();
                    int objIdx = 0;
                    
                    foreach (var entry in pm4.MprrEntries)
                    {
                        if (entry.IsSentinel)
                        {
                            // Sentinel marks object boundary
                            if (currentObject.Count > 0)
                            {
                                objectSizes.Add(currentObject.Count);
                                
                                // Sample first few objects
                                if (sampleObjects.Count < 50 && objIdx < 3)
                                {
                                    sampleObjects.Add((Path.GetFileName(pm4Path), objIdx, new List<MprrEntry>(currentObject)));
                                }
                                objIdx++;
                            }
                            totalSentinels++;
                            currentObject.Clear();
                        }
                        else
                        {
                            currentObject.Add(entry);
                        }
                    }
                }
                catch { /* ignore */ }
            }
            
            using (var sw = new StreamWriter(mprrAnalysisPath))
            {
                sw.WriteLine("=== MPRR Deep Analysis (CORRECTED) ===");
                sw.WriteLine($"Total entries: {totalEntries:N0}");
                sw.WriteLine($"Total sentinels (object boundaries): {totalSentinels:N0}");
                sw.WriteLine($"Estimated objects: {objectSizes.Count:N0}");
                sw.WriteLine();
                
                sw.WriteLine("=== Structure ===");
                sw.WriteLine("Each MPRR entry is 4 bytes: (ushort Value1, ushort Value2)");
                sw.WriteLine("Value1=0xFFFF (65535) = SENTINEL marking object boundary");
                sw.WriteLine("Between sentinels = entries for one object");
                sw.WriteLine();
                
                sw.WriteLine("=== Object Size Distribution ===");
                if (objectSizes.Count > 0)
                {
                    sw.WriteLine($"  Min entries/object: {objectSizes.Min()}");
                    sw.WriteLine($"  Max entries/object: {objectSizes.Max()}");
                    sw.WriteLine($"  Avg entries/object: {objectSizes.Average():F1}");
                    
                    var sizeGroups = objectSizes.GroupBy(s => s / 10 * 10)
                        .OrderBy(g => g.Key).Take(20);
                    sw.WriteLine("  Size distribution (by 10s):");
                    foreach (var g in sizeGroups)
                        sw.WriteLine($"    {g.Key,4}-{g.Key + 9,4}: {g.Count(),6} objects");
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Value1 Frequency (top 20) ===");
                foreach (var (val, count) in value1Distribution.OrderByDescending(x => x.Value).Take(20))
                {
                    var marker = val == 0xFFFF ? " ← SENTINEL" : "";
                    sw.WriteLine($"  {val,5} (0x{val:X4}): {count,8} entries{marker}");
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Value2 Frequency (top 20) ===");
                foreach (var (val, count) in value2Distribution.OrderByDescending(x => x.Value).Take(20))
                {
                    sw.WriteLine($"  {val,5} (0x{val:X4}): {count,8} entries");
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Sample Objects ===");
                foreach (var (file, objI, entries) in sampleObjects.Take(15))
                {
                    var entryStr = string.Join(", ", entries.Take(8).Select(e => $"({e.Value1},{e.Value2})"));
                    if (entries.Count > 8) entryStr += $" ... ({entries.Count} total)";
                    sw.WriteLine($"  {file} obj[{objI}]: [{entryStr}]");
                }
                
                sw.WriteLine();
                sw.WriteLine("=== Key Discovery ===");
                sw.WriteLine("MPRR contains object boundaries using sentinel values (Value1=65535).");
                sw.WriteLine("This is the most effective method for grouping PM4 geometry into buildings.");
            }
            
            Console.WriteLine($"[INFO] MPRR Deep Analysis: {mprrAnalysisPath}");
            Console.WriteLine($"[INFO] MPRR: {totalEntries:N0} entries, {totalSentinels:N0} sentinels (objects)");
            
            // Per-file correlation with chunk counts
            var correlationPath = Path.Combine(outputDir, "mprr_correlation.csv");
            using (var sw = new StreamWriter(correlationPath))
            {
                sw.WriteLine("file,mprl_count,mslk_count,msur_count,mprr_count,sentinel_count,max_value1,max_value2");
                
                foreach (var pm4Path in pm4Files)
                {
                    try
                    {
                        var pm4Data = File.ReadAllBytes(pm4Path);
                        var pm4 = new PM4File(pm4Data);
                        
                        if (pm4.MprrEntries.Count == 0) continue;
                        
                        int mprlCount = pm4.PositionRefs.Count;
                        int mslkCount = pm4.LinkEntries.Count;
                        int msurCount = pm4.Surfaces.Count;
                        
                        int sentinels = pm4.MprrEntries.Count(e => e.IsSentinel);
                        ushort maxV1 = pm4.MprrEntries.Where(e => !e.IsSentinel).Select(e => e.Value1).DefaultIfEmpty((ushort)0).Max();
                        ushort maxV2 = pm4.MprrEntries.Select(e => e.Value2).DefaultIfEmpty((ushort)0).Max();
                        
                        sw.WriteLine(string.Join(",",
                            Path.GetFileName(pm4Path),
                            mprlCount, mslkCount, msurCount,
                            pm4.MprrEntries.Count, sentinels,
                            maxV1, maxV2));
                    }
                    catch { /* ignore */ }
                }
            }
            
            Console.WriteLine($"[INFO] MPRR Correlation: {correlationPath}");
        }

        /// <summary>
        /// Analyze geometric properties of MSLK and MSUR types to determine their physical meaning.
        /// Correlates Types with Normal Vectors (Flat vs Vertical).
        /// </summary>
        public void ExportGeometricAnalysis(string pm4Directory, string outputDir)
        {
            var analysisPath = Path.Combine(outputDir, "geometric_type_analysis.txt");
            
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);
            
            // MSLK Type analysis
            var mslkStats = new Dictionary<byte, (long Count, double SumNormalZ, double SumArea, double MaxZ, double MinZ)>();
            
            // MSUR CK24 Type analysis
            var ck24Stats = new Dictionary<byte, (long Count, double SumNormalZ, double SumArea)>();
            
            int filesAnalyzed = 0;
            
            // Debug counters
            long totalMslkSkippedNoGeom = 0;
            long totalMslkSkippedLowCount = 0;
            long totalMslkSkippedOutOfBounds = 0;
            long totalMslkProcessed = 0;
            
            var failureSamples = new List<string>();

            foreach (var pm4Path in pm4Files)
            {
                try
                {
                    var pm4Data = File.ReadAllBytes(pm4Path);
                    var pm4 = new PM4File(pm4Data);
                    
                    // --- MSLK Analysis ---
                    // MSLK -> MSPI -> MSPV (triangles)
                    // We need to parse MSPI and MSPV which aren't fully exposed in PM4File public model details
                    // Assuming PM4File populates PathVertices (MSPV) and PathIndices (MSPI)
                    
                    if (pm4.LinkEntries != null && pm4.PathIndices != null && pm4.PathVertices != null)
                    {
                        foreach (var mslk in pm4.LinkEntries)
                        {
                            if (mslk.MspiFirstIndex < 0) { totalMslkSkippedNoGeom++; continue; }
                            if (mslk.MspiIndexCount < 3) { totalMslkSkippedLowCount++; continue; }
                            if (mslk.MspiFirstIndex + mslk.MspiIndexCount > pm4.PathIndices.Count) 
                            { 
                                totalMslkSkippedOutOfBounds++; 
                                if (failureSamples.Count < 10)
                                {
                                    failureSamples.Add($"File: {Path.GetFileName(pm4Path)} | First: {mslk.MspiFirstIndex} + Count: {mslk.MspiIndexCount} > MSPI: {pm4.PathIndices.Count}");
                                }
                                continue; 
                            }
                            
                            totalMslkProcessed++;

                            // Process triangles
                            for (int i = 0; i < mslk.MspiIndexCount - 2; i += 3) // Assuming triangle list
                            {
                                // Indices into MSPV
                                var idx0 = pm4.PathIndices[mslk.MspiFirstIndex + i];
                                var idx1 = pm4.PathIndices[mslk.MspiFirstIndex + i + 1];
                                var idx2 = pm4.PathIndices[mslk.MspiFirstIndex + i + 2];
                                
                                if (idx0 >= pm4.PathVertices.Count || idx1 >= pm4.PathVertices.Count || idx2 >= pm4.PathVertices.Count) continue;
                                
                                var v0 = pm4.PathVertices[(int)idx0];
                                var v1 = pm4.PathVertices[(int)idx1];
                                var v2 = pm4.PathVertices[(int)idx2];
                                
                                // Calc normal
                                var edge1 = v1 - v0;
                                var edge2 = v2 - v0;
                                var normal = System.Numerics.Vector3.Cross(edge1, edge2);
                                float area = normal.Length() * 0.5f;
                                if (area > 0) normal = System.Numerics.Vector3.Normalize(normal);
                                
                                // Update stats
                                if (!mslkStats.ContainsKey(mslk.TypeFlags))
                                    mslkStats[mslk.TypeFlags] = (0, 0, 0, double.MinValue, double.MaxValue);
                                    
                                var s = mslkStats[mslk.TypeFlags];
                                mslkStats[mslk.TypeFlags] = (s.Count + 1, s.SumNormalZ + Math.Abs(normal.Z), s.SumArea + area, Math.Max(s.MaxZ, v0.Z), Math.Min(s.MinZ, v0.Z)); // Z is Up standard
                            }
                        }
                    }
                    
                    // --- CK24 Analysis ---
                    // MSUR -> Normal (pre-calculated)
                    foreach (var surf in pm4.Surfaces)
                    {
                         uint ck24 = surf.CK24;
                         byte type = (byte)((ck24 >> 16) & 0xFF);
                         
                         // Surface normal is stored directly
                         // Y is Up in MSUR? Let's check spec again.
                         // Spec says: MSUR has normal_x, normal_y, normal_z
                         // Assuming Z is up for MSUR based on 'Height' field being separate
                         
                         float normalZ = Math.Abs(surf.Normal.Z); // Assuming Z is up
                         
                         if (!ck24Stats.ContainsKey(type))
                             ck24Stats[type] = (0, 0, 0);
                             
                         var s = ck24Stats[type];
                         ck24Stats[type] = (s.Count + 1, s.SumNormalZ + normalZ, s.SumArea + 0); // No area readily available without indices
                    }
                    
                    filesAnalyzed++;
                    if (filesAnalyzed % 50 == 0) Console.Write(".");
                }
                catch {}
            }
            Console.WriteLine();
            
            using (var sw = new StreamWriter(analysisPath))
            {
                sw.WriteLine("╔════════════════════════════════════════════════════════════════╗");
                sw.WriteLine("║        GEOMETRIC TYPE CORRELATION ANALYSIS                     ║");
                sw.WriteLine("║  Validating 'Type' meaning via Physical Properties             ║");
                sw.WriteLine("╚════════════════════════════════════════════════════════════════╝");
                sw.WriteLine();
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("MSLK TYPE (Pathfinding Mesh)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("MSLK TYPE (Pathfinding Mesh)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                
                // Debug info
                sw.WriteLine($"[DEBUG] Total MSLK analyzed: {filesAnalyzed}");
                sw.WriteLine($"[DEBUG] Skipped (No Geom/FirstIndex<0): {totalMslkSkippedNoGeom}");
                sw.WriteLine($"[DEBUG] Skipped (Low Count < 3): {totalMslkSkippedLowCount}");
                sw.WriteLine($"[DEBUG] Skipped (Out of Bounds): {totalMslkSkippedOutOfBounds}");
                sw.WriteLine($"[DEBUG] Processed Successfully: {totalMslkProcessed}");

                if (totalMslkSkippedOutOfBounds > 0)
                {
                    sw.WriteLine();
                    sw.WriteLine("[DEBUG] OUT OF BOUNDS DIAGNOSIS");
                    sw.WriteLine("Sample failures (FirstIndex + Count > MSPI Count):");
                    foreach (var s in failureSamples)
                        sw.WriteLine("  " + s);
                }
                
                sw.WriteLine("Type | Triangles | Avg Normal Z (Up) | Avg Area | Z Range | Prediction");
                sw.WriteLine("-----|-----------|-------------------|----------|---------|-----------");
                
                foreach (var kvp in mslkStats.OrderByDescending(x => x.Value.Count))
                {
                    double avgNorm = kvp.Value.SumNormalZ / Math.Max(1, kvp.Value.Count);
                    double avgArea = kvp.Value.SumArea / Math.Max(1, kvp.Value.Count);
                    double zRange = kvp.Value.MaxZ - kvp.Value.MinZ;
                    
                    string prediction = avgNorm > 0.8 ? "FLATS (Floor)" : (avgNorm < 0.2 ? "WALLS (Vertical)" : "SLOPES");
                    
                    sw.WriteLine($"{kvp.Key,4} | {kvp.Value.Count,9} | {avgNorm,17:F3} | {avgArea,8:F2} | {zRange,7:F0} | {prediction}");
                }




                sw.WriteLine();
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("CK24 TYPE (Surface Grouping)");
                sw.WriteLine("════════════════════════════════════════════════════════════════");
                sw.WriteLine("Type | Surfaces | Avg Normal Z (Up) | Prediction");
                sw.WriteLine("-----|----------|-------------------|-----------");
                
                foreach (var kvp in ck24Stats.OrderByDescending(x => x.Value.Count))
                {
                    double avgNorm = kvp.Value.SumNormalZ / Math.Max(1, kvp.Value.Count);
                    string prediction = avgNorm > 0.8 ? "FLATS" : (avgNorm < 0.2 ? "WALLS" : "COMPLEX");
                    
                    sw.WriteLine($"0x{kvp.Key:X2} | {kvp.Value.Count,8} | {avgNorm,17:F3} | {prediction}");
                }
            }
            
            Console.WriteLine($"[INFO] Geometric Analysis: {analysisPath}");
        }




        public void AnalyzeRotationCandidates(string pm4Directory, string wmoLibraryPath, string outPath, string? mpqPath = null, string? listfilePath = null)
        {
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);

            // Output writer
            using var sw = new StreamWriter(outPath);
            sw.WriteLine("PM4 Rotation Analysis");
            sw.WriteLine("=====================");

            // 1. Calculate PM4 Dominant Wall Angles per Object (Grouped by Type/Id)
            var pm4Objects = new Dictionary<uint, List<Vector3>>(); // CompositeKey -> Wall Normals

            foreach (var pm4Path in pm4Files)
            {
                // Load PM4
                var pm4 = PM4File.FromFile(pm4Path);

            for (int i = 0; i < pm4.LinkEntries.Count; i++)
            {
                var entry = pm4.LinkEntries[i];
                // Only look at "Wall" types (Type 2, 4, 10, 12 from previous analysis)
                // Or generically: any surface with normal.Z ~ 0
                if (!entry.HasGeometry) continue;

                uint key = ((uint)entry.TypeFlags << 24) | (entry.GroupObjectId & 0xFFFFFF);

                if (!pm4Objects.ContainsKey(key))
                    pm4Objects[key] = new List<Vector3>();

                int start = entry.MspiFirstIndex;
                int count = entry.MspiIndexCount;

                if (start < 0 || start + count > pm4.PathIndices.Count) continue;

                for (int j = 0; j < count; j += 3)
                {
                     if (start + j + 2 >= pm4.PathIndices.Count) break;

                    int i0 = (int)pm4.PathIndices[start + j];
                    int i1 = (int)pm4.PathIndices[start + j + 1];
                    int i2 = (int)pm4.PathIndices[start + j + 2];

                    if (i0 >= pm4.PathVertices.Count || i1 >= pm4.PathVertices.Count || i2 >= pm4.PathVertices.Count) continue;

                    var v0 = pm4.PathVertices[i0];
                    var v1 = pm4.PathVertices[i1];
                    var v2 = pm4.PathVertices[i2];

                    var edge1 = v1 - v0;
                    var edge2 = v2 - v0;
                    var normal = Vector3.Normalize(Vector3.Cross(edge1, edge2));

                    // Check if Wall (Standard Z-up: Abs(Z) < 0.5)
                    if (Math.Abs(normal.Z) < 0.5f)
                    {
                        pm4Objects[key].Add(normal);
                    }
                }
            }
            }

            // 2. Compute Dominant Angle for each PM4 Object
            sw.WriteLine("Object ID | Type | Wall Tris | Dominant Angle | Confidence");
            sw.WriteLine("----------|------|-----------|----------------|-----------");

            foreach (var kvp in pm4Objects)
            {
                 if (kvp.Value.Count < 10) continue; // Skip noise

                 // Histogram 5-deg bins
                 float[] bins = new float[72];
                 foreach(var norm in kvp.Value)
                 {
                     float angle = (float)Math.Atan2(norm.Y, norm.X) * (180f / (float)Math.PI);
                     if (angle < 0) angle += 360f;
                     int bin = (int)(angle / 5) % 72;
                     bins[bin] += 1.0f; // Weight by count for now (could do area)
                 }

                 int bestBin = -1;
                 float maxVal = 0;
                 for(int i=0; i<72; i++)
                 {
                     if(bins[i] > maxVal) { maxVal = bins[i]; bestBin = i; }
                 }

                 float domAngle = bestBin * 5f + 2.5f;
                 float confidence = maxVal / kvp.Value.Count;

                 uint type = kvp.Key >> 24;
                 uint id = kvp.Key & 0xFFFFFF;

                sw.WriteLine($"{id,9} | {type,4} | {kvp.Value.Count,9} | {domAngle,14:F1} | {confidence,9:F2}");
            }

            // 3. WMO Analysis (If path provided)
            if (!string.IsNullOrEmpty(wmoLibraryPath) && Directory.Exists(wmoLibraryPath))
            {
                sw.WriteLine();
                sw.WriteLine("WMO Rotation Analysis");
                sw.WriteLine("=====================");
                sw.WriteLine("WMO Name                                   | Wall Tris | Dominant Angle | Size");
                sw.WriteLine("-------------------------------------------|-----------|----------------|------");

                var wmoFiles = Directory.GetFiles(wmoLibraryPath, "*.wmo", SearchOption.AllDirectories)
                    .Where(f => !f.EndsWith("_000.wmo") && !f.Contains("_00")) // Filter group files if possible, keep root
                    .ToList();


                sw.WriteLine($"Found {wmoFiles.Count} WMO files in library.");

                foreach (var wmoPath in wmoFiles)
                {
                    try
                    {
                        // We need to use WmoPathfindingExtractor directly or via service
                        // Since WmoExtractorService might not expose PathfindingExtractor directly, let's instantiate it.
                        var pfExtractor = new WmoPathfindingExtractor();
                        var pfData = pfExtractor.ExtractFromWmo(wmoPath);

                        if (pfData.Aggregate != null && pfData.Aggregate.WallSurfaces.Count > 0)
                        {
                            string name = Path.GetFileName(wmoPath);
                            // Truncate name for display
                            if (name.Length > 42) name = name.Substring(0, 39) + "...";

                            sw.WriteLine($"{name,-42} | {pfData.Aggregate.WallCount,9} | {pfData.Aggregate.DominantWallAngle,14:F1} | {pfData.Aggregate.Size.X:F0}x{pfData.Aggregate.Size.Y:F0}x{pfData.Aggregate.Size.Z:F0}");
                        }
                    }
                    catch (Exception ex)
                    {
                        sw.WriteLine($"[ERROR] {Path.GetFileName(wmoPath)}: {ex.Message}");
                    }
                }
            }
            
            // 4. MPQ Analysis
            if (!string.IsNullOrEmpty(mpqPath) && File.Exists(mpqPath) && 
                !string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
            {
                sw.WriteLine();
                sw.WriteLine("WMO Rotation Analysis (MPQ Source)");
                sw.WriteLine("=====================");
                sw.WriteLine("Archive: " + Path.GetFileName(mpqPath));
                sw.WriteLine("WMO Name                                   | Wall Tris | Dominant Angle | Size");
                sw.WriteLine("-------------------------------------------|-----------|----------------|------");
                
                using var archive = new WoWRollback.Core.Services.Archive.MpqArchiveSource(new[] { mpqPath });
                
                var wmoFiles = File.ReadLines(listfilePath)
                    .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    .Where(l => !l.Contains("_00")) // Exclude group files
                    .ToList();
                    
                sw.WriteLine($"Found {wmoFiles.Count} WMO candidates in listfile.");
                
                int processed = 0;
                foreach (var wmoFile in wmoFiles)
                {
                    // Normalize path
                    string wmoPath = wmoFile.Replace('/', '\\');
                    
                    if (!archive.FileExists(wmoPath)) continue;
                    
                    try
                    {
                        var pfExtractor = new WmoPathfindingExtractor();
                        var structure = pfExtractor.ExtractStructureFromMpq(archive, wmoPath);

                        if (structure.Aggregate != null && structure.Aggregate.WallSurfaces.Count > 0)
                        {
                            string name = Path.GetFileName(wmoPath);
                            if (name.Length > 42) name = name.Substring(0, 39) + "...";
                            sw.WriteLine($"{name,-42} | {structure.Aggregate.WallCount,9} | {structure.Aggregate.DominantWallAngle,14:F1} | {structure.Aggregate.Size.X:F0}x{structure.Aggregate.Size.Y:F0}x{structure.Aggregate.Size.Z:F0}");
                            processed++;
                        }
                    }
                    catch (Exception ex)
                    {
                        // sw.WriteLine($"[ERROR] {Path.GetFileName(wmoPath)}: {ex.Message}");
                    }
                    
                    if (processed % 100 == 0 && processed > 0) Console.Write(".");
                }
            }
            
            Console.WriteLine($"[INFO] Rotation Analysis: {outPath}");
        }
        public void AnalyzeRotationCandidatesV2(string pm4Directory, string wmoLibraryPath, string outPath, string typeOutPath, string? gamePath = null, string? listfilePath = null)
        {
            var pm4Files = Directory.GetFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories);

            // Output writer
            using var sw = new StreamWriter(outPath);
            sw.WriteLine("Geometric Rotation & Matching Analysis");
            sw.WriteLine("====================================");
            sw.WriteLine($"Generated: {DateTime.Now}");
            sw.WriteLine();

            // ---------------------------------------------------------
            // 1. Calculate PM4 Object Fingerprints (Wall Angle + bounds)
            // ---------------------------------------------------------
            Console.WriteLine("[INFO] Fingerprinting PM4 Objects...");
            
            var pm4Candidates = new Dictionary<uint, Pm4Candidate>(); 

            foreach (var pm4Path in pm4Files)
            {
                var pm4 = PM4File.FromFile(pm4Path);

                for (int i = 0; i < pm4.LinkEntries.Count; i++)
                {
                    var entry = pm4.LinkEntries[i];
                    if (!entry.HasGeometry) continue;

                    uint key = ((uint)entry.TypeFlags << 24) | (entry.GroupObjectId & 0xFFFFFF);

                    if (!pm4Candidates.TryGetValue(key, out var candidate))
                    {
                        candidate = new Pm4Candidate { Id = entry.GroupObjectId & 0xFFFFFF, Type = (uint)entry.TypeFlags };
                        pm4Candidates[key] = candidate;
                    }

                    int start = entry.MspiFirstIndex;
                    int count = entry.MspiIndexCount;

                    if (start < 0 || start + count > pm4.PathIndices.Count) continue;

                    for (int j = 0; j < count; j += 3)
                    {
                        if (start + j + 2 >= pm4.PathIndices.Count) break;

                        int i0 = (int)pm4.PathIndices[start + j];
                        int i1 = (int)pm4.PathIndices[start + j + 1];
                        int i2 = (int)pm4.PathIndices[start + j + 2];

                        if (i0 >= pm4.PathVertices.Count || i1 >= pm4.PathVertices.Count || i2 >= pm4.PathVertices.Count) continue;

                        var v0 = pm4.PathVertices[i0];
                        var v1 = pm4.PathVertices[i1];
                        var v2 = pm4.PathVertices[i2];
                        
                        // Update Bounds
                        candidate.Min = Vector3.Min(candidate.Min, v0);
                        candidate.Min = Vector3.Min(candidate.Min, v1);
                        candidate.Min = Vector3.Min(candidate.Min, v2);
                        candidate.Max = Vector3.Max(candidate.Max, v0);
                        candidate.Max = Vector3.Max(candidate.Max, v1);
                        candidate.Max = Vector3.Max(candidate.Max, v2);

                        // Calculate Normal
                        var edge1 = v1 - v0;
                        var edge2 = v2 - v0;
                        var normal = Vector3.Normalize(Vector3.Cross(edge1, edge2));

                        if (Math.Abs(normal.Z) < 0.5f)
                        {
                            candidate.WallNormals.Add(normal);
                        }
                        else if (normal.Z > 0.5f)
                        {
                            candidate.FloorNormals.Add(normal);
                        }
                    }
                }
            }
            
            // Process PM4 Candidates (calc angles)
            var validPm4 = new List<Pm4Candidate>();
            foreach (var cand in pm4Candidates.Values)
            {
                if (cand.WallNormals.Count < 5) continue; // Filter noise

                float[] bins = new float[72];
                foreach (var norm in cand.WallNormals)
                {
                    float angle = (float)Math.Atan2(norm.Y, norm.X) * (180f / (float)Math.PI);
                    if (angle < 0) angle += 360f;
                    int bin = (int)(angle / 5) % 72;
                    bins[bin] += 1.0f;
                }

                int bestBin = -1;
                float maxVal = 0;
                for (int i = 0; i < 72; i++)
                {
                    if (bins[i] > maxVal) { maxVal = bins[i]; bestBin = i; }
                }

                cand.DominantAngle = bestBin * 5f + 2.5f;
                validPm4.Add(cand);
            }
            
            sw.WriteLine($"Found {validPm4.Count} valid PM4 Objects (with walls).");
            Console.WriteLine($"[INFO] Found {validPm4.Count} valid PM4 Objects.");

            // ---------------------------------------------------------
            // 2. Extract WMO Fingerprints (from MPQ)
            // ---------------------------------------------------------
            var validWmos = new List<WoWRollback.PM4Module.WmoPathfindingData>();

            if (!string.IsNullOrEmpty(gamePath) && Directory.Exists(gamePath) && 
                !string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
            {
                Console.WriteLine("[INFO] extracting WMO fingerprints from MPQ...");
                
                var mpqFiles = Directory.GetFiles(Path.Combine(gamePath, "Data"), "*.MPQ", SearchOption.AllDirectories);
                using var archive = new WoWRollback.Core.Services.Archive.MpqArchiveSource(mpqFiles);
                
                var wmoEntries = File.ReadLines(listfilePath)
                    .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    .Where(l => !l.Contains("_00")) // Skip sub-groups
                    .ToList();
                
                int processed = 0;
                
                // Single thread for safety.
                foreach (var wmoFile in wmoEntries)
                {
                    string wmoPath = wmoFile.Replace('/', '\\');
                    if (!archive.FileExists(wmoPath)) continue;

                    try
                    {
                        var pfExtractor = new WmoPathfindingExtractor();
                        var structure = pfExtractor.ExtractStructureFromMpq(archive, wmoPath);
                        
                        // Add Groups
                        foreach (var g in structure.Groups) 
                        {
                            if (g.WallCount > 0) validWmos.Add(g);
                        }
                        
                        // Add Aggregate
                        if (structure.Aggregate != null && structure.Aggregate.WallCount > 0)
                        {
                            validWmos.Add(structure.Aggregate);
                        }
                    }
                    catch { /* ignore */ }
                    
                    processed++;
                    if (processed % 200 == 0) Console.Write(".");
                }
                Console.WriteLine();
            }
            
            sw.WriteLine($"Found {validWmos.Count} valid WMO candidates (with walls).");
            Console.WriteLine($"[INFO] Found {validWmos.Count} valid WMO candidates.");

            // ---------------------------------------------------------
            // 3. Perform Matching
            // ---------------------------------------------------------
            sw.WriteLine();
            sw.WriteLine("Matching Analysis (Size Tolerance: 15%)");
            sw.WriteLine("PM4 ID    | Type | Size (WxDxH)   | WMO Name                                 | WMO Size       | Rot Delta | Conf");
            sw.WriteLine("----------|------|----------------|------------------------------------------|----------------|-----------|-----");
            
            foreach (var pm4 in validPm4.OrderBy(p => p.Type).ThenBy(p => p.Id))
            {
                // Find candidates
                var matches = new List<(WmoPathfindingData wmo, float rot, float sizeDiff)>();
                
                foreach (var wmo in validWmos)
                {
                    // Check Height match (important barrier for false positives)
                    if (Math.Abs(pm4.Size.Z - wmo.Size.Z) > pm4.Size.Z * 0.25f + 5.0f && pm4.Size.Z > 10) continue; 

                    bool matchRaw = IsSizeMatch(pm4.Size.X, pm4.Size.Y, wmo.Size.X, wmo.Size.Y);
                    bool matchRot = IsSizeMatch(pm4.Size.X, pm4.Size.Y, wmo.Size.Y, wmo.Size.X); // 90 deg rotated
                    
                    if (matchRaw || matchRot)
                    {
                        // Calculate Rotation Delta
                        // RotDelta = PM4 - WMO
                        float delta = pm4.DominantAngle - wmo.DominantWallAngle;
                        
                        // Normalize delta to 0..360
                        while (delta < 0) delta += 360f;
                        while (delta >= 360) delta -= 360f;
                        
                        // Check if alignment is near 0, 90, 180, 270
                        if (IsCardinal(delta, out float cardinal))
                        {
                            matches.Add((wmo, cardinal, Vector3.Distance(pm4.Size, wmo.Size)));
                        }
                    }
                }
                
                if (matches.Count > 0)
                {
                    // Sort by size difference
                    var best = matches.OrderBy(m => m.sizeDiff).Take(5);
                    
                    foreach (var m in best)
                    {
                        string wmoName = Path.GetFileName(m.wmo.WmoPath);
                        if (wmoName.Length > 40) wmoName = wmoName.Substring(0, 37) + "...";
                        
                        sw.WriteLine($"{pm4.Id,9} | {pm4.Type,4} | {pm4.Size.X,4:F0}x{pm4.Size.Y,4:F0}x{pm4.Size.Z,4:F0} | {wmoName,-40} | {m.wmo.Size.X,4:F0}x{m.wmo.Size.Y,4:F0}x{m.wmo.Size.Z,4:F0} | {m.rot,9:F1} | HIGH");
                    }
                }
            }

            // Export Type Correlation Analysis
            string matchCsvPath = Path.Combine(Path.GetDirectoryName(outPath), "matches.csv");
            File.WriteAllText(matchCsvPath, "PM4_ID,WMO_Name,PosX,PosY,PosZ,RotBox_X,RotBox_Y,RotBox_Z\n");

            using (var typeSw = new StreamWriter(typeOutPath))
            {
                typeSw.WriteLine("PM4_Type,WMO_Name,WMO_Flags_Hex,Start_Indoor,Start_Outdoor,WMO_Size,Dominant_MOPY_Byte");
                
                foreach (var pm4 in validPm4.OrderBy(p => p.Type).ThenBy(p => p.Id))
                {
                    // Reuse match logic logic (simplified for correlation)
                     foreach (var wmo in validWmos)
                    {
                        if (Math.Abs(pm4.Size.Z - wmo.Size.Z) > pm4.Size.Z * 0.25f + 5.0f && pm4.Size.Z > 10) continue;
                        
                        bool matchRaw = IsSizeMatch(pm4.Size.X, pm4.Size.Y, wmo.Size.X, wmo.Size.Y);
                        bool matchRot = IsSizeMatch(pm4.Size.X, pm4.Size.Y, wmo.Size.Y, wmo.Size.X);
                        
                        // Strict sizing for correlation to avoid noise
                        if (matchRaw || matchRot)
                        {
                            // Check rotation alignment
                            float delta = pm4.DominantAngle - wmo.DominantWallAngle;
                            while (delta < 0) delta += 360f;
                            while (delta >= 360) delta -= 360f;
                            
                            if (IsCardinal(delta, out _))
                            {
                                // Found a high-confidence match
                                uint combinedFlags = 0;
                                foreach (var f in wmo.GroupFlags) combinedFlags |= f;
                                
                                bool startIndoor = (combinedFlags & 0x2000) != 0;
                                bool startOutdoor = (combinedFlags & 0x8) != 0;
                                
                                string wmoName = Path.GetFileName(wmo.WmoPath);
                                typeSw.WriteLine($"{pm4.Type},{wmoName},0x{combinedFlags:X},{startIndoor},{startOutdoor},{wmo.Size.X:F0}x{wmo.Size.Y:F0}x{wmo.Size.Z:F0},0x{wmo.DominantMopyFlag:X}");
                                
                                // Calculate TILT (Pitch/Roll) from Floor Normals
                                Vector3 avgFloor = Vector3.UnitZ;
                                if (pm4.FloorNormals.Count > 0)
                                {
                                    Vector3 sum = Vector3.Zero;
                                    foreach (var n in pm4.FloorNormals) sum += n;
                                    avgFloor = Vector3.Normalize(sum);
                                }
                                
                                // Eulers
                                // Eulers
                                // 1. Yaw (Heading) from Cardinal Match
                                float yaw = delta; // This is the delta we calculated
                                
                                // 2. Pitch/Roll from Floor Normal
                                // Simple approach: Tilt around axis perpendicular to Z and FloorNormal
                                // But combining with Yaw is tricky.
                                // Let's simplify: WMO is defined in local space.
                                // Position: Center of Bounds? Or (0,0,0) of PM4? 
                                // PM4 coords are absolute.
                                // WMO coords are relative.
                                // We place WMO at Center(PM4).
                                
                                // For CSV, just output Raw Eulers if possible, or components.
                                // Let's output the Floor Normal and the Yaw. 
                                // The Patcher can do the Quaternion math.
                                
                                // But user asked to ENCODE it.
                                // Let's estimate Pitch/Roll.
                                float pitch = -(float)Math.Asin(avgFloor.Y); // Approx
                                float roll = -(float)Math.Asin(avgFloor.X);  // Approx
                                // Accurate conversion requires Quaternion.
                                
                                // matches.csv content
                                // ID, WMO, PosX, PosY, PosZ, RotX, RotY, RotZ
                                Vector3 center = (pm4.Min + pm4.Max) * 0.5f;
                                // Sanitize -0
                                if (pitch == -0.0f) pitch = 0.0f;
                                if (yaw == -0.0f) yaw = 0.0f;
                                if (roll == -0.0f) roll = 0.0f;

                                File.AppendAllText(Path.Combine(Path.GetDirectoryName(outPath), "matches.csv"), 
                                    $"{pm4.Id},{wmoName},{center.X},{center.Y},{center.Z},{pitch},{yaw},{roll}\n");

                                // One high-confidence match per PM4 is enough for correlation stats
                                break; 
                            }
                        }
                    }
                }
            }

            Console.WriteLine($"[INFO] Analysis Complete. Saved to {outPath}");
        }

        private class Pm4Candidate {
            public uint Id;
            public uint Type;
            public List<Vector3> WallNormals = new();
            public List<Vector3> FloorNormals = new();
            public Vector3 Min = new Vector3(float.MaxValue);
            public Vector3 Max = new Vector3(float.MinValue);
            public float DominantAngle;
            public Vector3 Size => Max - Min;
        }

        private bool IsSizeMatch(float w1, float d1, float w2, float d2)
        {
             // 15% tolerance
             float tol = 0.15f;
             if (Math.Abs(w1 - w2) / (w2 + 0.1f) > tol) return false;
             if (Math.Abs(d1 - d2) / (d2 + 0.1f) > tol) return false;
             return true;
        }

        private bool IsCardinal(float angle, out float cleanAngle)
        {
            float tol = 10f; // 10 degree slop
            cleanAngle = 0;
            
            if (Math.Abs(angle - 0) < tol || Math.Abs(angle - 360) < tol) { cleanAngle = 0; return true; }
            if (Math.Abs(angle - 90) < tol) { cleanAngle = 90; return true; }
            if (Math.Abs(angle - 180) < tol) { cleanAngle = 180; return true; }
            if (Math.Abs(angle - 270) < tol) { cleanAngle = 270; return true; }
            return false;
    }
    
    /// <summary>
    /// Find the position of a chunk by its 4-character signature in raw bytes.
    /// Returns -1 if not found.
    /// </summary>
    private static int FindChunkPosition(byte[] data, string signature)
    {
        if (data.Length < 8 || signature.Length != 4)
            return -1;
            
        // ADT chunks store signatures in reverse byte order (little-endian)
        // So "MODF" is stored as "FDOM" (bytes reversed)
        byte[] sig = System.Text.Encoding.ASCII.GetBytes(signature);
        Array.Reverse(sig);  // Reverse for little-endian ADT format
        
        for (int i = 0; i <= data.Length - 8; i++)
        {
            if (data[i] == sig[0] && data[i + 1] == sig[1] && 
                data[i + 2] == sig[2] && data[i + 3] == sig[3])
            {
                return i;
            }
        }
        return -1;
    }
}
}
