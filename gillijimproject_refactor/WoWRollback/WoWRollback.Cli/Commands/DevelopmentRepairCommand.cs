using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using WoWRollback.Core.Services.PM4;
using WoWRollback.Core.Services.Archive;
using WoWRollback.PM4Module;

namespace WoWRollback.Cli.Commands;

/// <summary>
/// A one-stop-shop command to repair the development map.
/// Automates: WMO extraction -> PM4 matching -> ADT creation.
/// </summary>
public static class DevelopmentRepairCommand
{
    public static int Execute(Dictionary<string, string> opts)
    {
        var pm4Dir = opts.GetValueOrDefault("pm4-dir");
        var sourceAdtDir = opts.GetValueOrDefault("source-adt");
        var clientPath = opts.GetValueOrDefault("client-path");
        var outDir = opts.GetValueOrDefault("out");
        var mapName = opts.GetValueOrDefault("map", "development");
        var listfilePath = opts.GetValueOrDefault("listfile");
        var dumpObjs = opts.ContainsKey("dump-objs");
        var reuseCsvPath = opts.GetValueOrDefault("reuse-csv");
        var cacheDir = opts.GetValueOrDefault("cache-dir");

        if (string.IsNullOrEmpty(pm4Dir) || string.IsNullOrEmpty(sourceAdtDir) || string.IsNullOrEmpty(clientPath) || string.IsNullOrEmpty(outDir))
        {
            Console.WriteLine("Usage: development-repair --pm4-dir <dir> --source-adt <dir> --client-path <dir> --out <dir> [--cache-dir <dir>] [--map <name>] [--listfile <path>] [--dump-objs] [--reuse-csv <path>]");
            Console.WriteLine();
            Console.WriteLine("Automates the full repair pipeline for the development map (In-Memory Fast Mode).");
            return 1;
        }

        // Default cache dir to out dir if not specified
        var effectiveCacheDir = !string.IsNullOrEmpty(cacheDir) ? cacheDir : outDir;
        Directory.CreateDirectory(effectiveCacheDir);

        Console.WriteLine("=== Development Map Repair Pipeline (In-Memory) ===");
        Console.WriteLine($"PM4 Dir:     {pm4Dir}");
        Console.WriteLine($"Source ADT:  {sourceAdtDir}");
        Console.WriteLine($"Client:      {clientPath}");
        Console.WriteLine($"Output:      {outDir}");
        Console.WriteLine($"Cache Dir:   {effectiveCacheDir}"); // Explicit feedback
        if (!string.IsNullOrEmpty(listfilePath)) Console.WriteLine($"Listfile:    {listfilePath}");
        Console.WriteLine();

        Directory.CreateDirectory(outDir);
        var reconstructionDir = Path.Combine(outDir, "modf_reconstruction");
        var adtOutDir = Path.Combine(outDir, "adt_335");
        Directory.CreateDirectory(reconstructionDir);
        Directory.CreateDirectory(adtOutDir);

        List<Pm4ModfReconstructor.ModfEntry> transformedEntries;

        if (!string.IsNullOrEmpty(reuseCsvPath) && File.Exists(reuseCsvPath))
        {
            Console.WriteLine($"[INFO] Skipping extraction/matching. Reusing CSV: {reuseCsvPath}");
            transformedEntries = LoadEntriesFromCsv(reuseCsvPath);
            Console.WriteLine($"[INFO] Loaded {transformedEntries.Count} placements from CSV.");
        }
        else
        {
            // 1. Build WMO Library (In-Memory)
            Console.WriteLine("[Step 1/3] building WMO geometry library from client...");
            
            IEnumerable<string> wmoFiles;
            if (!string.IsNullOrEmpty(listfilePath) && File.Exists(listfilePath))
            {
                Console.WriteLine($"  Loading WMO list from {listfilePath}...");
                wmoFiles = File.ReadLines(listfilePath)
                    .Where(l => l.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
                    .Where(l => l.StartsWith("World\\wmo", StringComparison.OrdinalIgnoreCase))
                    .Where(l => !Regex.IsMatch(l, @"_\d{3}\.wmo$", RegexOptions.IgnoreCase)) // Exclude group files
                    .ToList();
                Console.WriteLine($"  Found {wmoFiles.Count()} Root WMOs in listfile");
            }
            else
            {
                Console.WriteLine("  No listfile provided, using fallback list...");
                wmoFiles = GetFallbackWmoList();
            }

            // Pass effectiveCacheDir instead of outDir
            var wmoLibrary = BuildWmoLibraryInMemory(clientPath, wmoFiles, dumpObjs ? Path.Combine(outDir, "wmo_dump") : null, effectiveCacheDir);

            if (wmoLibrary.Count == 0)
            {
                Console.Error.WriteLine("[ERROR] Failed to build WMO library (no geometry found)");
                return 1;
            }

            // 2. Reconstruct Placements
            Console.WriteLine("[Step 2/3] Reconstructing placements...");
            var reconstructor = new Pm4ModfReconstructor();
            
            // Load PM4 objects first so we can include them in verification
            var pm4Objects = reconstructor.LoadPm4Objects(pm4Dir);
            
            // Use 0.5f confidence
            var result = reconstructor.ReconstructModf(pm4Objects, wmoLibrary, 0.5f);

            // Apply Coordinate Transform
            Console.WriteLine("[INFO] Applying PM4->ADT coordinate transform...");
            transformedEntries = result.ModfEntries.Select(e => e with 
            { 
                Position = AdtModfInjector.ServerToAdtPosition(e.Position),
                // WmoPath is already correct because we built the library with real paths!
            }).ToList();
            
            var transformedResult = result with { ModfEntries = transformedEntries };
            var modfCsvPath = Path.Combine(reconstructionDir, "modf_entries.csv");
            
            reconstructor.ExportToCsv(transformedResult, modfCsvPath);
            reconstructor.ExportMwmo(transformedResult, Path.Combine(reconstructionDir, "mwmo_names.csv"));
            
            // *** VERIFICATION JSON - PROOF OF DATA GENERATION ***
            var verificationJsonPath = Path.Combine(reconstructionDir, "verification_report.json");
            reconstructor.ExportVerificationJson(transformedResult, pm4Objects, wmoLibrary, verificationJsonPath);
        }

        // 3. Create ADTs
        Console.WriteLine("[Step 3/3] Creating patched ADTs...");
        var merger = new AdtPatcher();
        
        var entriesByTile = transformedEntries
            .GroupBy(e => GetTileForPosition(e.Position))
            .ToDictionary(g => g.Key, g => g.ToList());

        Console.WriteLine($"[INFO] Found placements for {entriesByTile.Count} tiles");

        var adtFiles = Directory.GetFiles(sourceAdtDir, "*_*.adt");
        int processedCount = 0;

        foreach (var adtPath in adtFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(adtPath);
            if (fileName.EndsWith("_obj0") || fileName.EndsWith("_tex0")) continue;

            var parts = fileName.Split('_');
            if (parts.Length < 3) continue;
            
            if (!int.TryParse(parts[parts.Length - 2], out int tileX) || 
                !int.TryParse(parts[parts.Length - 1], out int tileY)) continue;

            // Determine paths for split files
            var baseName = fileName;
            var obj0Path = Path.Combine(sourceAdtDir, $"{baseName}_obj0.adt");
            var tex0Path = Path.Combine(sourceAdtDir, $"{baseName}_tex0.adt");

            // 1. Merge Split ADT first (to get textures + existing props)
            // Even if we don't have new placements, we want to fix the ADT format
            byte[] mergedAdt;
            try
            {
                mergedAdt = merger.MergeSplitAdt(adtPath, 
                    File.Exists(obj0Path) ? obj0Path : null, 
                    File.Exists(tex0Path) ? tex0Path : null)!;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to merge {fileName}: {ex.Message}");
                continue;
            }

            // 2. Inject Placements (if any)
            byte[] finalAdt = mergedAdt;
            if (entriesByTile.TryGetValue((tileX, tileY), out var tileEntries))
            {
                Console.WriteLine($"  Patching tile {tileX}_{tileY} ({tileEntries.Count} placements)...");

                var modfList = tileEntries.Select(e => new AdtModfInjector.ModfEntry335
                {
                    NameId = 0,
                    UniqueId = e.UniqueId,
                    Position = e.Position,
                    Rotation = e.Rotation,
                    BoundsMin = e.BoundsMin,
                    BoundsMax = e.BoundsMax,
                    Flags = e.Flags,
                    DoodadSet = e.DoodadSet,
                    NameSet = e.NameSet,
                    Scale = e.Scale
                }).ToList();

                var wmoNames = tileEntries.Select(e => e.WmoPath).Distinct().ToList();

                var injector = new AdtModfInjector();
                finalAdt = injector.InjectModfIntoAdt(mergedAdt, wmoNames, modfList);
            }
            else
            {
                // Just write the merged ADT (fixes texturing even if no objects)
                // Console.WriteLine($"  Merging tile {tileX}_{tileY} (no new placements)...");
            }

            // 3. Write Output
            var outPath = Path.Combine(adtOutDir, Path.GetFileName(adtPath));
            File.WriteAllBytes(outPath, finalAdt);
            
            // 4. Verify & Dump
            VerifyAndDumpAdt(outPath, Path.Combine(outDir, "chunk_dump"));

            processedCount++;
        }

        Console.WriteLine($"[COMPLETE] Patched {processedCount} tiles. Output in {adtOutDir}");
        return 0;
    }

    private static void VerifyAndDumpAdt(string path, string dumpRootDir)
    {
        try 
        {
            var data = File.ReadAllBytes(path);
            var fileName = Path.GetFileNameWithoutExtension(path);
            var dumpDir = Path.Combine(dumpRootDir, fileName);
            
            bool hasMver = false, hasMcnk = false, hasMwmo = false, hasModf = false, hasMtex = false;
            int mwmoSize = 0, modfSize = 0, mtexSize = 0;
            int mcnkCount = 0;

            int pos = 0;
            while (pos < data.Length - 8)
            {
                var sigBytes = new byte[4];
                Array.Copy(data, pos, sigBytes, 0, 4);
                // Standard ADT chunks are usually reversed "REVM" for "MVER" in byte stream if expecting string match?
                // But typically we look for "MVER" string.
                // Let's rely on string matching.
                var sig = System.Text.Encoding.ASCII.GetString(data, pos, 4);
                // Handle reversed signatures (e.g. "REVM" -> "MVER")
                var rSig = new string(sig.Reverse().ToArray());
                
                // Identify normalized signature
                string normSig = sig;
                if (rSig == "MVER" || rSig == "MHDR" || rSig == "MCNK" || rSig == "MTEX" || 
                    rSig == "MWMO" || rSig == "MODF" || rSig == "MDDF" || rSig == "MMDX" || rSig == "MMID" || rSig == "MWID")
                {
                    normSig = rSig;
                }

                var size = BitConverter.ToInt32(data, pos + 4);
                
                // Track presence
                if (normSig == "MVER") hasMver = true;
                if (normSig == "MCNK") 
                { 
                    hasMcnk = true; 
                    mcnkCount++; 
                }
                if (normSig == "MWMO") { hasMwmo = true; mwmoSize = size; }
                if (normSig == "MODF") { hasModf = true; modfSize = size; }
                if (normSig == "MTEX") { hasMtex = true; mtexSize = size; }

                // DUMP CHUNKS (Global metadata + First MCNK)
                if (dumpRootDir != null)
                {
                    bool shouldDump = normSig == "MHDR" || normSig == "MTEX" || normSig == "MWMO" || 
                                      normSig == "MODF" || normSig == "MDDF" || 
                                      (normSig == "MCNK" && mcnkCount == 1); // Only dump first MCNK to save space

                    if (shouldDump)
                    {
                        Directory.CreateDirectory(dumpDir);
                        var chunkData = new byte[size + 8]; // Include header in dump for context
                        if (pos + 8 + size <= data.Length)
                        {
                            Array.Copy(data, pos, chunkData, 0, size + 8);
                            var dumpFile = Path.Combine(dumpDir, $"{normSig}_{pos:X}.bin");
                            File.WriteAllBytes(dumpFile, chunkData);
                        }
                    }
                }

                if (size < 0 || pos + 8 + size > data.Length) break;
                pos += 8 + size;
            }

            if (mtexSize == 0)
            {
                Console.WriteLine($"    [WARN] {fileName} has EMPTY TEXTURES (MTEX size 0)!");
            }
            if (hasModf && modfSize == 0)
            {
                Console.WriteLine($"    [WARN] {fileName} has EMPTY MODF chunk!");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"    [WARN] Verification failed for {path}: {ex.Message}");
        }
    }

    private static List<Pm4ModfReconstructor.ModfEntry> LoadEntriesFromCsv(string csvPath)
    {
        var entries = new List<Pm4ModfReconstructor.ModfEntry>();
        var lines = File.ReadAllLines(csvPath);
        
        // Skip header
        var dataLines = lines.Where(l => !l.StartsWith("ck24")).ToList();
        
        foreach (var line in dataLines)
        {
            try
            {
                // Simple CSV split (assuming no commas in fields)
                var cols = line.Split(',');
                if (cols.Length < 12) continue;

                var ck24 = cols[0];
                var wmoPath = cols[1];
                var nameId = uint.Parse(cols[2]);
                var uniqueId = uint.Parse(cols[3]);
                var posX = float.Parse(cols[4]);
                var posY = float.Parse(cols[5]);
                var posZ = float.Parse(cols[6]);
                var rotX = float.Parse(cols[7]);
                var rotY = float.Parse(cols[8]);
                var rotZ = float.Parse(cols[9]);
                var scale = float.Parse(cols[10]); // This is 1024 or 1.0? 
                // The reconstruction uses 1024 as internal ushort, but CSV export probably writes float?
                // Let's check Pm4ModfReconstructor.ExportToCsv.
                // It writes `e.Scale / 1024.0f`.
                // But AdtModfInjector uses ushort scale (1024 = 1.0).
                // So we need to parse float and convert back to ushort? 
                // Wait, ModfEntry record uses `ushort Scale`.
                // So I should parse float and multiply by 1024.
                
                var confidence = float.Parse(cols[11]);

                entries.Add(new Pm4ModfReconstructor.ModfEntry(
                    nameId, 
                    uniqueId,
                    new System.Numerics.Vector3(posX, posY, posZ),
                    new System.Numerics.Vector3(rotX, rotY, rotZ),
                    System.Numerics.Vector3.Zero, // Bounds not in CSV?
                    System.Numerics.Vector3.Zero,
                    0, // Flags not in CSV?
                    0, // DoodadSet
                    0, // NameSet
                    (ushort)(scale * 1024.0f),
                    wmoPath,
                    ck24,
                    confidence,
                    0, // TileX (not loaded from minimal CSV)
                    0  // TileY
                ));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Error parsing CSV line: {ex.Message}");
            }
        }
        return entries;
    }

    private static (int X, int Y) GetTileForPosition(System.Numerics.Vector3 pos)
    {
        const float TileSize = 533.33333f;
        int x = (int)(32 - (pos.X / TileSize));
        int y = (int)(32 - (pos.Y / TileSize));
        return (x, y);
    }

    private static List<Pm4ModfReconstructor.WmoReference> BuildWmoLibraryInMemory(string clientPath, IEnumerable<string> wmoFiles, string? dumpDir, string outDir)
    {
        var cachePath = Path.Combine(outDir, "wmo_library.json");
        var jsonOptions = new JsonSerializerOptions 
        { 
            WriteIndented = true,
            IncludeFields = true // CRITICAL: Required for Vector3/Matrix4x4 fields
        };
        
        // Try Load Cache
        // If dumping OBJs, we force re-extraction to actually have the geometry data to write.
        if (dumpDir == null && File.Exists(cachePath))
        {
            try 
            {
                Console.WriteLine($"[INFO] Loading WMO library from cache: {cachePath}");
                var json = File.ReadAllText(cachePath);
                var cachedLib = JsonSerializer.Deserialize<List<Pm4ModfReconstructor.WmoReference>>(json, jsonOptions);
                
                // Validate cache integrity
                if (cachedLib != null && cachedLib.Count > 0)
                {
                    // Check if we have valid stats (fix for previous empty JSON issue)
                    var firstValid = cachedLib.FirstOrDefault(x => x.Stats.VertexCount > 0);
                    if (firstValid != null && firstValid.Stats.Dimensions == System.Numerics.Vector3.Zero)
                    {
                        Console.WriteLine("[WARN] Cache detected but appears corrupted (empty stats). Forcing re-extraction.");
                    }
                    else
                    {
                        Console.WriteLine($"[INFO] Cache hit! Loaded {cachedLib.Count} WMOs.");
                        return cachedLib;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] Failed to load cache (will re-extract): {ex.Message}");
            }
        }

        var library = new List<Pm4ModfReconstructor.WmoReference>();
        // WmoWalkableSurfaceExtractor methods are static
        var matcher = new Pm4WmoGeometryMatcher(); // Helper to compute stats

        // Setup archive
        var mpqs = ArchiveLocator.LocateMpqs(clientPath);
        using var src = new PrioritizedArchiveSource(clientPath, mpqs);

        if (dumpDir != null) Directory.CreateDirectory(dumpDir);

        int count = 0;
        int processed = 0;

        foreach (var wmo in wmoFiles)
        {
            count++;
            if (count % 100 == 0) Console.Write($".");
            if (count % 1000 == 0) Console.WriteLine($" ({count} processed)");

            try
            {
                if (!src.FileExists(wmo)) continue;

                // Extract
                byte[] rootData;
                using (var stream = src.OpenFile(wmo))
                using (var ms = new MemoryStream())
                {
                    stream.CopyTo(ms);
                    rootData = ms.ToArray();
                }

                byte[]? LoadGroup(string p) => src.FileExists(p) ? ReadAllBytes(src, p) : null;
                var data = WmoWalkableSurfaceExtractor.ExtractFromBytes(rootData, wmo, LoadGroup);

                // We want to match against substantial geometry
                if (data.WalkableVertices.Count < 3) continue;

                // Compute Stats In-Memory
                var stats = matcher.ComputeStats(data.WalkableVertices);
                
                // Add to library with REAL PATH
                // Normalize path to backslashes for client consistency
                var realPath = wmo.Replace('/', '\\');
                library.Add(new Pm4ModfReconstructor.WmoReference(realPath, stats));

                processed++;

                // Optional Dump
                if (dumpDir != null)
                {
                    var safeName = wmo.Replace('\\', '_').Replace('/', '_') + "_collision.obj";
                    WmoWalkableSurfaceExtractor.ExportToObj(data, Path.Combine(dumpDir, safeName));
                }
            }
            catch {}
        }
        Console.WriteLine();
        Console.WriteLine($"[INFO] Built library with {processed} WMOs in-memory.");

        // Save Cache
        try 
        {
            var json = JsonSerializer.Serialize(library, jsonOptions);
            File.WriteAllText(cachePath, json);
            Console.WriteLine($"[INFO] Saved library cache to {cachePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to save cache: {ex.Message}");
        }

        return library;
    }

    private static IEnumerable<string> GetFallbackWmoList()
    {
        return new[]
        {
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind.wmo",
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_Cathedral.wmo",
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_Keep.wmo",
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_Bank.wmo",
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_MageTower.wmo",
            @"World\wmo\Azeroth\Buildings\Stormwind\Stormwind_TradeDistrict.wmo",
            @"World\wmo\Kalimdor\Buildings\Ogrimmar\Ogrimmar.wmo"
        };
    }

    private static byte[] ReadAllBytes(IArchiveSource src, string path)
    {
        using var s = src.OpenFile(path);
        using var ms = new MemoryStream();
        s.CopyTo(ms);
        return ms.ToArray();
    }
}
