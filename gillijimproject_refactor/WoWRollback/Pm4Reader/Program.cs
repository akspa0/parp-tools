using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace Pm4Reader;

/// <summary>
/// Standalone PM4 Reader - Clean, single-path implementation
/// Goal: Decode PM4 files ONE way, and investigate CK24=0x000000 objects
/// </summary>
class Program
{
    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: Pm4Reader <pm4_file_or_directory>");
            Console.WriteLine("       Pm4Reader --ck24-analysis <directory>");
            Console.WriteLine("       Pm4Reader --compare <directory>");
            return 1;
        }

        if (args[0] == "--ck24-analysis" && args.Length > 1)
        {
            return RunCk24Analysis(args[1]);
        }
        
        if (args[0] == "--compare" && args.Length > 1)
        {
            return RunTileComparison(args[1]);
        }
        
        if (args[0] == "--trace" && args.Length > 1)
        {
            return RunSceneGraphTrace(args[1]);
        }
        
        if (args[0] == "--export-obj" && args.Length > 1)
        {
            return ExportObjPerCk24(args[1], args.Length > 2 ? args[2] : null);
        }
        
        if (args[0] == "--multi-tile" && args.Length > 1)
        {
            return RunMultiTileAnalysis(args[1], args.Length > 2 ? args[2] : null);
        }
        
        if (args[0] == "--export-mscn" && args.Length > 1)
        {
            return ExportMscnPoints(args[1], args.Length > 2 ? args[2] : null);
        }
        
        if (args[0] == "--mslk-trace" && args.Length > 1)
        {
            return RunMslkTrace(args[1]);
        }
        
        if (args[0] == "--mscn-analysis" && args.Length > 1)
        {
            return RunMscnAnalysis(args[1]);
        }
        
        if (args[0] == "--raw-dump" && args.Length > 1)
        {
            return RunRawDump(args[1]);
        }
        
        if (args[0] == "--cross-tile" && args.Length > 1)
        {
            return RunCrossTileAnalysis(args[1]);
        }
        
        if (args[0] == "--mdsf-analysis" && args.Length > 1)
        {
            return RunMdsfAnalysis(args[1]);
        }
        
        if (args[0] == "--object-complete" && args.Length > 1)
        {
            return RunObjectCompleteAnalysis(args[1], args.Length > 2 ? args[2] : null);
        }
        
        if (args[0] == "--node-graph" && args.Length > 1)
        {
            return RunNodeGraphAnalysis(args[1]);
        }
        
        if (args[0] == "--mprl-graph" && args.Length > 1)
        {
            return RunMprlGraphAnalysis(args[1]);
        }
        
        if (args[0] == "--full-map" && args.Length > 1)
        {
            return RunFullStructureMap(args[1]);
        }
        
        if (args[0] == "--mprl-footprint" && args.Length > 1)
        {
            return RunMprlFootprintAnalysis(args[1]);
        }
        
        if (args[0] == "--mprl-to-adt" && args.Length > 1)
        {
            return RunMprlToAdtTerrainExtraction(args[1]);
        }

        string path = args[0];
        
        if (Directory.Exists(path))
        {
            // Process all PM4 files in directory
            var files = Directory.GetFiles(path, "*.pm4");
            Console.WriteLine($"Found {files.Length} PM4 files in {path}\n");
            
            var allStats = new List<Pm4Stats>();
            foreach (var file in files)
            {
                var stats = ProcessFile(file);
                if (stats != null) allStats.Add(stats);
            }
            
            // Summary
            Console.WriteLine("\n=== SUMMARY ===");
            Console.WriteLine($"Total files: {allStats.Count}");
            Console.WriteLine($"Total CK24=0 surfaces: {allStats.Sum(s => s.Ck24ZeroSurfaceCount):N0}");
            Console.WriteLine($"Total CK24!=0 surfaces: {allStats.Sum(s => s.Ck24NonZeroSurfaceCount):N0}");
        }
        else if (File.Exists(path))
        {
            ProcessFile(path, verbose: true);
        }
        else
        {
            Console.Error.WriteLine($"Path not found: {path}");
            return 1;
        }
        
        return 0;
    }

    static int RunCk24Analysis(string directory)
    {
        Console.WriteLine($"=== CK24=0x000000 Analysis ===\n");
        Console.WriteLine($"Scanning: {directory}\n");
        
        var files = Directory.GetFiles(directory, "*.pm4");
        
        // Track global CK24 statistics
        var ck24Counts = new Dictionary<uint, int>();
        var ck24Tiles = new Dictionary<uint, HashSet<string>>();
        int totalCk24ZeroObjects = 0;
        
        foreach (var file in files)
        {
            var pm4 = Pm4File.Parse(File.ReadAllBytes(file));
            var tileName = Path.GetFileNameWithoutExtension(file);
            
            // Group surfaces by CK24
            var surfacesByCk24 = pm4.Surfaces.GroupBy(s => s.CK24).ToDictionary(g => g.Key, g => g.ToList());
            
            foreach (var (ck24, surfaces) in surfacesByCk24)
            {
                if (!ck24Counts.ContainsKey(ck24))
                {
                    ck24Counts[ck24] = 0;
                    ck24Tiles[ck24] = new HashSet<string>();
                }
                ck24Counts[ck24] += surfaces.Count;
                ck24Tiles[ck24].Add(tileName);
            }
            
            // Count CK24=0 objects
            if (surfacesByCk24.TryGetValue(0, out var zeroSurfaces))
            {
                totalCk24ZeroObjects++;
                Console.WriteLine($"{tileName}: CK24=0 has {zeroSurfaces.Count} surfaces");
            }
        }
        
        Console.WriteLine($"\n=== CK24 Summary ===");
        Console.WriteLine($"Unique CK24 values: {ck24Counts.Count}");
        Console.WriteLine($"Tiles with CK24=0: {ck24Tiles.GetValueOrDefault(0u)?.Count ?? 0}");
        Console.WriteLine($"Total CK24=0 surfaces across all tiles: {ck24Counts.GetValueOrDefault(0u):N0}");
        
        // Top 10 CK24 values by surface count
        Console.WriteLine($"\nTop 10 CK24 values by surface count:");
        foreach (var (ck24, count) in ck24Counts.OrderByDescending(kv => kv.Value).Take(10))
        {
            Console.WriteLine($"  CK24 0x{ck24:X6}: {count:N0} surfaces across {ck24Tiles[ck24].Count} tiles");
        }
        
        return 0;
    }

    static int RunTileComparison(string directory)
    {
        Console.WriteLine($"=== PM4 Tile Comparison Analysis ===\n");
        Console.WriteLine($"Scanning: {directory}\n");
        
        var files = Directory.GetFiles(directory, "*.pm4");
        Console.WriteLine($"Found {files.Length} PM4 files\n");
        
        // Global statistics
        var globalCk24Surfaces = new Dictionary<uint, int>();
        var globalCk24Tiles = new Dictionary<uint, HashSet<string>>();
        int totalSurfaces = 0, totalMslk = 0, totalMprl = 0, totalMprr = 0;
        int totalMslkToMprl = 0, totalMslkToMsvt = 0;
        int totalMprrToMprl = 0, totalMprrToMsvt = 0;
        
        // Per-tile data for comparison
        var tileData = new List<(string Name, int Surfaces, int Ck24Count, int MprlCount, int MslkMprlRefs)>();
        
        foreach (var file in files)
        {
            var pm4 = Pm4File.Parse(File.ReadAllBytes(file));
            var name = Path.GetFileNameWithoutExtension(file);
            
            if (pm4.Surfaces.Count == 0) continue; // Skip empty tiles
            
            // CK24 tracking
            var surfacesByCk24 = pm4.Surfaces.GroupBy(s => s.CK24).ToDictionary(g => g.Key, g => g.ToList());
            foreach (var (ck24, surfaces) in surfacesByCk24)
            {
                if (!globalCk24Surfaces.ContainsKey(ck24))
                {
                    globalCk24Surfaces[ck24] = 0;
                    globalCk24Tiles[ck24] = new HashSet<string>();
                }
                globalCk24Surfaces[ck24] += surfaces.Count;
                globalCk24Tiles[ck24].Add(name);
            }
            
            totalSurfaces += pm4.Surfaces.Count;
            totalMslk += pm4.LinkEntries.Count;
            totalMprl += pm4.PositionRefs.Count;
            totalMprr += pm4.MprrEntries.Count;
            
            // MSLK RefIndex analysis
            int mslkToMprl = pm4.LinkEntries.Count(e => e.RefIndex < pm4.PositionRefs.Count);
            int mslkToMsvt = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.PositionRefs.Count);
            totalMslkToMprl += mslkToMprl;
            totalMslkToMsvt += mslkToMsvt;
            
            // MPRR analysis
            var nonSentinel = pm4.MprrEntries.Where(e => e.Value1 != 0xFFFF).ToList();
            int mprrToMprl = nonSentinel.Count(e => e.Value1 < pm4.PositionRefs.Count);
            int mprrToMsvt = nonSentinel.Count(e => e.Value1 >= pm4.PositionRefs.Count);
            totalMprrToMprl += mprrToMprl;
            totalMprrToMsvt += mprrToMsvt;
            
            tileData.Add((name, pm4.Surfaces.Count, surfacesByCk24.Count, pm4.PositionRefs.Count, mslkToMprl));
        }
        
        // Global summary
        Console.WriteLine("=== Global Statistics ===");
        Console.WriteLine($"  Total tiles analyzed: {tileData.Count}");
        Console.WriteLine($"  Total surfaces: {totalSurfaces:N0}");
        Console.WriteLine($"  Total MSLK entries: {totalMslk:N0}");
        Console.WriteLine($"  Total MPRL entries: {totalMprl:N0}");
        Console.WriteLine($"  Total MPRR entries: {totalMprr:N0}");
        
        Console.WriteLine("\n=== Cross-Tile CK24 Analysis ===");
        Console.WriteLine($"  Unique CK24 values (global): {globalCk24Surfaces.Count}");
        
        // Multi-tile CK24s
        var multiTileCk24 = globalCk24Tiles.Where(kv => kv.Value.Count > 1).OrderByDescending(kv => kv.Value.Count);
        Console.WriteLine($"  CK24s spanning multiple tiles: {multiTileCk24.Count()}");
        Console.WriteLine("\n  Top 10 Multi-Tile CK24s:");
        foreach (var (ck24, tiles) in multiTileCk24.Take(10))
        {
            Console.WriteLine($"    CK24 0x{ck24:X6}: {globalCk24Surfaces[ck24]:N0} surfaces across {tiles.Count} tiles");
        }
        
        Console.WriteLine("\n=== MSLK RefIndex Summary ===");
        Console.WriteLine($"  Total MSLK → MPRL: {totalMslkToMprl:N0} ({100.0 * totalMslkToMprl / Math.Max(1, totalMslk):F1}%)");
        Console.WriteLine($"  Total MSLK → MSVT: {totalMslkToMsvt:N0} ({100.0 * totalMslkToMsvt / Math.Max(1, totalMslk):F1}%)");
        
        Console.WriteLine("\n=== MPRR Value1 Summary ===");
        int totalNonSentinel = totalMprrToMprl + totalMprrToMsvt;
        Console.WriteLine($"  Total MPRR → MPRL: {totalMprrToMprl:N0} ({100.0 * totalMprrToMprl / Math.Max(1, totalNonSentinel):F1}%)");
        Console.WriteLine($"  Total MPRR → MSVT: {totalMprrToMsvt:N0} ({100.0 * totalMprrToMsvt / Math.Max(1, totalNonSentinel):F1}%)");
        
        // Check for potential map-level patterns
        Console.WriteLine("\n=== Map-Level Patterns ===");
        var totalMprlSum = tileData.Sum(t => t.MprlCount);
        Console.WriteLine($"  Sum of all MPRL counts: {totalMprlSum}");
        Console.WriteLine($"  Sum of all MSLK→MPRL refs: {totalMslkToMprl}");
        
        // Tiles with most surfaces
        Console.WriteLine("\n=== Top 10 Tiles by Surface Count ===");
        foreach (var t in tileData.OrderByDescending(t => t.Surfaces).Take(10))
        {
            Console.WriteLine($"  {t.Name}: {t.Surfaces:N0} surfaces, {t.Ck24Count} CK24s, {t.MprlCount} MPRL, {t.MslkMprlRefs} MSLK→MPRL");
        }
        
        // MSHD Header Analysis Across Tiles
        Console.WriteLine("\n=== MSHD Header Analysis (Cross-Tile) ===");
        var headerData = new List<(string Name, int X, int Y, uint F0, uint F4, uint F8)>();
        
        foreach (var file in files)
        {
            var pm4 = Pm4File.Parse(File.ReadAllBytes(file));
            if (pm4.Header == null) continue;
            
            var name = Path.GetFileNameWithoutExtension(file);
            // Try to extract tile coords from name (e.g., development_22_18)
            var parts = name.Split('_');
            int x = 0, y = 0;
            if (parts.Length >= 2)
            {
                int.TryParse(parts[^2], out x);
                int.TryParse(parts[^1], out y);
            }
            
            headerData.Add((name, x, y, pm4.Header.Field00, pm4.Header.Field04, pm4.Header.Field08));
        }
        
        Console.WriteLine($"  Tiles with MSHD headers: {headerData.Count}");
        
        // Check for patterns
        var uniqueF0 = headerData.Select(h => h.F0).Distinct().Count();
        var uniqueF4 = headerData.Select(h => h.F4).Distinct().Count();
        var uniqueF8 = headerData.Select(h => h.F8).Distinct().Count();
        Console.WriteLine($"  Unique Field00 values: {uniqueF0}");
        Console.WriteLine($"  Unique Field04 values: {uniqueF4}");
        Console.WriteLine($"  Unique Field08 values: {uniqueF8}");
        
        // Check if Field00 == Field08 (like tile 22_18)
        int matchingF0F8 = headerData.Count(h => h.F0 == h.F8);
        Console.WriteLine($"  Tiles where Field00 == Field08: {matchingF0F8} ({100.0 * matchingF0F8 / Math.Max(1, headerData.Count):F1}%)");
        
        // Sample headers
        Console.WriteLine("\n  Sample MSHD headers (showing first 10):");
        foreach (var h in headerData.Take(10))
        {
            Console.WriteLine($"    {h.Name}: F0=0x{h.F0:X4} ({h.F0}), F4=0x{h.F4:X4} ({h.F4}), F8=0x{h.F8:X4} ({h.F8})");
        }
        
        // Check for tile coordinate correlation
        Console.WriteLine("\n  Tile coordinate correlation check:");
        foreach (var h in headerData.Where(h => h.X > 0 || h.Y > 0).Take(5))
        {
            Console.WriteLine($"    {h.Name} (tile {h.X},{h.Y}): F0={h.F0}, F4={h.F4}, F0-F4={h.F0 - h.F4}");
        }
        
        return 0;
    }

    static int RunSceneGraphTrace(string filePath)
    {
        Console.WriteLine("=== PM4 Scene Graph Trace ===\n");
        
        if (!File.Exists(filePath))
        {
            Console.Error.WriteLine($"File not found: {filePath}");
            return 1;
        }
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(filePath));
        Console.WriteLine($"File: {Path.GetFileName(filePath)}");
        Console.WriteLine($"Chunks: MSUR={pm4.Surfaces.Count}, MSLK={pm4.LinkEntries.Count}, MPRL={pm4.PositionRefs.Count}, MSVT={pm4.MeshVertices.Count}, MSCN={pm4.SceneNodes.Count}\n");
        
        // Get unique CK24 objects (excluding 0)
        var ck24Objects = pm4.Surfaces
            .Where(s => s.CK24 != 0)
            .GroupBy(s => s.CK24)
            .OrderByDescending(g => g.Count())
            .Take(5)
            .ToList();
        
        Console.WriteLine($"Tracing top 5 CK24 objects:\n");
        
        foreach (var ck24Group in ck24Objects)
        {
            uint ck24 = ck24Group.Key;
            var surfaces = ck24Group.ToList();
            
            Console.WriteLine($"┌─ CK24 0x{ck24:X6} ───────────────────────────────────────");
            Console.WriteLine($"│  Surfaces: {surfaces.Count}");
            
            // Collect all MSVT vertices used by this CK24
            var usedMsvtIndices = new HashSet<uint>();
            foreach (var surf in surfaces)
            {
                int start = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;
                if (start >= 0 && start + count <= pm4.MeshIndices.Count)
                {
                    for (int i = 0; i < count; i++)
                    {
                        usedMsvtIndices.Add(pm4.MeshIndices[start + i]);
                    }
                }
            }
            Console.WriteLine($"│  └→ MSVI→MSVT vertices: {usedMsvtIndices.Count} unique");
            
            // Find MSLK entries whose RefIndex falls within our MSVT range
            var relatedMslk = pm4.LinkEntries
                .Where(l => l.RefIndex >= pm4.PositionRefs.Count && usedMsvtIndices.Contains((uint)l.RefIndex))
                .Take(10)
                .ToList();
            
            Console.WriteLine($"│");
            Console.WriteLine($"├─ MSLK entries referencing our MSVT: {relatedMslk.Count}");
            
            foreach (var mslk in relatedMslk.Take(3))
            {
                Console.WriteLine($"│  ├─ MSLK: Type={mslk.TypeFlags}, Floor={mslk.Subtype}, GroupId=0x{mslk.GroupObjectId:X8}");
                
                // Follow MSPI→MSPV chain
                if (mslk.MspiFirstIndex >= 0 && mslk.MspiFirstIndex < pm4.PathIndices.Count)
                {
                    var pathVerts = new List<string>();
                    for (int i = 0; i < Math.Min((int)mslk.MspiIndexCount, 4); i++)
                    {
                        int mspiIdx = mslk.MspiFirstIndex + i;
                        if (mspiIdx < pm4.PathIndices.Count)
                        {
                            uint mspvIdx = pm4.PathIndices[mspiIdx];
                            if (mspvIdx < pm4.PathVertices.Count)
                            {
                                var pv = pm4.PathVertices[(int)mspvIdx];
                                pathVerts.Add($"({pv.X:F0},{pv.Y:F0},{pv.Z:F0})");
                            }
                        }
                    }
                    Console.WriteLine($"│  │  └→ MSPI[{mslk.MspiFirstIndex}..{mslk.MspiFirstIndex + mslk.MspiIndexCount - 1}]→MSPV: {string.Join(" ", pathVerts)}");
                }
                
                // Show RefIndex target
                if (mslk.RefIndex < pm4.PositionRefs.Count)
                {
                    var mprl = pm4.PositionRefs[mslk.RefIndex];
                    float angle = 360.0f * mprl.Unknown0x04 / 65536.0f;
                    Console.WriteLine($"│  │  └→ RefIndex {mslk.RefIndex}→MPRL: ({mprl.PositionX:F0},{mprl.PositionY:F0},{mprl.PositionZ:F0}) Rot={angle:F0}° Floor={mprl.Unknown0x14}");
                }
                else
                {
                    var mv = pm4.MeshVertices[mslk.RefIndex];
                    Console.WriteLine($"│  │  └→ RefIndex {mslk.RefIndex}→MSVT: ({mv.X:F0},{mv.Y:F0},{mv.Z:F0})");
                }
            }
            
            // Find MPRL entries that might be related (nearby positions)
            // MPRL is stored as Y,Z,X - need to swap to match MSVT's X,Y,Z
            var surfBounds = ComputeSurfaceBounds(pm4, surfaces);
            var nearbyMprl = pm4.PositionRefs
                .Where(p => p.Unknown0x16 == 0)
                .Select(p => new { 
                    Raw = p, 
                    // Convert YZX → XYZ
                    X = p.PositionZ,  // stored Z is real X
                    Y = p.PositionX,  // stored X is real Y  
                    Z = p.PositionY   // stored Y is real Z
                })
                .Where(p => p.X >= surfBounds.min.X - 20 && p.X <= surfBounds.max.X + 20 &&
                           p.Y >= surfBounds.min.Y - 20 && p.Y <= surfBounds.max.Y + 20)
                .ToList();
            
            Console.WriteLine($"│");
            Console.WriteLine($"├─ MPRL entries near object bounds (YZX→XYZ): {nearbyMprl.Count}");
            foreach (var mprl in nearbyMprl.Take(3))
            {
                float angle = 360.0f * mprl.Raw.Unknown0x04 / 65536.0f;
                Console.WriteLine($"│  └─ XYZ=({mprl.X:F0},{mprl.Y:F0},{mprl.Z:F0}) Rot={angle:F0}° Floor={mprl.Raw.Unknown0x14}");
            }
            
            // Check MSCN collision points near object
            var nearbyMscn = pm4.SceneNodes
                .Where(n => n.X >= surfBounds.min.X - 5 && n.X <= surfBounds.max.X + 5 &&
                           n.Y >= surfBounds.min.Y - 5 && n.Y <= surfBounds.max.Y + 5 &&
                           n.Z >= surfBounds.min.Z - 5 && n.Z <= surfBounds.max.Z + 5)
                .ToList();
            
            Console.WriteLine($"│");
            Console.WriteLine($"└─ MSCN collision points inside bounds: {nearbyMscn.Count}");
            
            Console.WriteLine();
        }
        
        // Show connection summary
        Console.WriteLine("=== Scene Graph Connection Summary ===");
        Console.WriteLine("CK24 → MSUR (surfaces) → MSVI (indices) → MSVT (mesh vertices)");
        Console.WriteLine("MSLK → MSPI (path indices) → MSPV (path vertices)");
        Console.WriteLine("MSLK → RefIndex → [MPRL (position+rotation) | MSVT (anchor)]");
        Console.WriteLine("MPRR → [MPRL (with edge flags) | MSVT (with Value2=0)]");
        Console.WriteLine("MSCN → Collision geometry points (Z-clustered)");
        
        return 0;
    }
    
    static int RunMultiTileAnalysis(string directory, string? outputDir)
    {
        var pm4Files = Directory.GetFiles(directory, "*.pm4", SearchOption.AllDirectories);
        Console.WriteLine($"=== Multi-Tile PM4 Analysis ===");
        Console.WriteLine($"Found {pm4Files.Length} PM4 files\n");
        
        outputDir ??= Path.Combine(Path.GetTempPath(), "pm4_multi_tile");
        Directory.CreateDirectory(outputDir);
        
        // Global data structures
        var globalCK24 = new Dictionary<uint, List<(string tile, int surfaceCount, Vector3 minBound, Vector3 maxBound)>>();
        var allTileSurfaces = new Dictionary<string, List<MsurEntry>>();
        var allTileVertices = new Dictionary<string, List<Vector3>>();
        var allTileIndices = new Dictionary<string, List<uint>>();
        var allTileSceneNodes = new Dictionary<string, List<Vector3>>(); // MSCN cache
        
        // Parse all tiles
        foreach (var file in pm4Files)
        {
            try
            {
                var pm4 = Pm4File.Parse(File.ReadAllBytes(file));
                string tileName = Path.GetFileNameWithoutExtension(file);
                
                allTileSurfaces[tileName] = pm4.Surfaces;
                allTileVertices[tileName] = pm4.MeshVertices;
                allTileIndices[tileName] = pm4.MeshIndices;
                allTileSceneNodes[tileName] = pm4.SceneNodes; // Cache MSCN
                
                // Group surfaces by CK24
                var byCk24 = pm4.Surfaces.GroupBy(s => s.CK24);
                foreach (var grp in byCk24)
                {
                    if (!globalCK24.ContainsKey(grp.Key))
                        globalCK24[grp.Key] = new List<(string, int, Vector3, Vector3)>();
                    
                    // Compute bounds for this tile's CK24 group
                    var verts = new List<Vector3>();
                    foreach (var surf in grp)
                    {
                        for (int i = 0; i < surf.IndexCount; i++)
                        {
                            uint msviIdx = surf.MsviFirstIndex + (uint)i;
                            if (msviIdx < pm4.MeshIndices.Count)
                            {
                                int msvtIdx = (int)pm4.MeshIndices[(int)msviIdx];
                                if (msvtIdx < pm4.MeshVertices.Count)
                                    verts.Add(pm4.MeshVertices[msvtIdx]);
                            }
                        }
                    }
                    
                    if (verts.Count > 0)
                    {
                        var bounds = ComputeBounds(verts);
                        globalCK24[grp.Key].Add((tileName, grp.Count(), bounds.min, bounds.max));
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error parsing {file}: {ex.Message}");
            }
        }
        
        // Report cross-tile CK24s
        Console.WriteLine("=== Cross-Tile CK24 Analysis ===");
        Console.WriteLine($"Total unique CK24 values: {globalCK24.Count}\n");
        
        var multiTileCK24 = globalCK24.Where(kv => kv.Value.Count > 1).OrderByDescending(kv => kv.Value.Count).Take(20).ToList();
        Console.WriteLine($"CK24 values spanning multiple tiles: {multiTileCK24.Count}\n");
        
        foreach (var (ck24, tiles) in multiTileCK24)
        {
            int totalSurfaces = tiles.Sum(t => t.surfaceCount);
            Console.WriteLine($"  CK24 0x{ck24:X6}: {tiles.Count} tiles, {totalSurfaces} total surfaces");
            foreach (var (tile, count, min, max) in tiles.Take(3))
            {
                Console.WriteLine($"    {tile}: {count} surfaces at ({min.X:F0}, {min.Y:F0})");
            }
            if (tiles.Count > 3)
                Console.WriteLine($"    ... and {tiles.Count - 3} more tiles");
        }
        
        // Export combined OBJs for ALL multi-tile CK24s, separated by type
        Console.WriteLine("\n=== Exporting ALL Multi-Tile CK24 Objects ===");
        
        // Create subdirs for WMO vs Other
        var wmoDir = Path.Combine(outputDir, "WMO");
        var otherDir = Path.Combine(outputDir, "Other");
        Directory.CreateDirectory(wmoDir);
        Directory.CreateDirectory(otherDir);
        
        int exported = 0;
        // Process ALL CK24s that span multiple tiles (skip CK24=0 nav mesh)
        var allMultiTileCK24 = globalCK24.Where(kv => kv.Value.Count > 1 && kv.Key != 0)
            .OrderByDescending(kv => kv.Value.Sum(t => t.surfaceCount));
        
        Console.WriteLine($"  Processing {allMultiTileCK24.Count()} multi-tile objects...\n");
        
        foreach (var (ck24, tiles) in allMultiTileCK24)
        {
            var allVerts = new List<Vector3>();
            var allFaces = new List<int[]>();
            var vertexMap = new Dictionary<(string tile, int idx), int>();
            
            // New: Collect MSCN points for vertical geometry
            var allMscnPoints = new List<Vector3>();
            var faceToMscnMap = new Dictionary<int, int>(); // Face Index -> MSCN Point Index
            
            foreach (var (tileName, _, _, _) in tiles)
            {
                if (!allTileSurfaces.ContainsKey(tileName)) continue;
                var surfs = allTileSurfaces[tileName].Where(s => s.CK24 == ck24);
                var indices = allTileIndices[tileName];
                var verts = allTileVertices[tileName];
                var sceneNodes = allTileSceneNodes[tileName];
                
                foreach (var surf in surfs)
                {
                    var faceIndices = new List<int>();
                    for (int i = 0; i < surf.IndexCount; i++)
                    {
                        uint msviIdx = surf.MsviFirstIndex + (uint)i;
                        if (msviIdx < indices.Count)
                        {
                            int msvtIdx = (int)indices[(int)msviIdx];
                            var key = (tileName, msvtIdx);
                            if (!vertexMap.ContainsKey(key))
                            {
                                vertexMap[key] = allVerts.Count + 1;
                                if (msvtIdx < verts.Count)
                                    allVerts.Add(verts[msvtIdx]);
                            }
                            faceIndices.Add(vertexMap[key]);
                        }
                    }
                    if (faceIndices.Count >= 3)
                    {
                        allFaces.Add(faceIndices.ToArray());
                        int faceIndex = allFaces.Count - 1;
                        
                        // Collect linked MSCN node (vertical geometry/path node)
                        if (surf.MdosIndex < sceneNodes.Count)
                        {
                            var mscn = sceneNodes[(int)surf.MdosIndex];
                            allMscnPoints.Add(mscn);
                            faceToMscnMap[faceIndex] = allMscnPoints.Count - 1; // 0-based index for internal storage
                        }
                    }
                }
            }
            
            if (allVerts.Count > 0)
            {
                // Determine type from high byte of CK24
                int typeByte = (int)((ck24 >> 16) & 0xFF);
                bool isWmo = typeByte == 0x42 || typeByte == 0x43;
                var targetDir = isWmo ? wmoDir : otherDir;
                string typeLabel = isWmo ? "WMO" : "Other";
                
                // === VERTEX CONNECTIVITY CLUSTERING: Proper instance separation ===
                // Faces sharing vertices belong to same instance (Union-Find)
                int[] faceParent = new int[allFaces.Count];
                for (int i = 0; i < faceParent.Length; i++) faceParent[i] = i;
                
                int FindFace(int x) {
                    if (faceParent[x] != x) faceParent[x] = FindFace(faceParent[x]);
                    return faceParent[x];
                }
                void UnionFace(int a, int b) {
                    int ra = FindFace(a), rb = FindFace(b);
                    if (ra != rb) faceParent[ra] = rb;
                }
                
                // Map vertices to faces that use them
                var vertexToFaces = new Dictionary<int, List<int>>();
                for (int fi = 0; fi < allFaces.Count; fi++)
                {
                    foreach (var vi in allFaces[fi])
                    {
                        if (!vertexToFaces.ContainsKey(vi))
                            vertexToFaces[vi] = new List<int>();
                        vertexToFaces[vi].Add(fi);
                    }
                }
                
                // Union faces that share vertices
                foreach (var (_, faces) in vertexToFaces)
                {
                    for (int i = 1; i < faces.Count; i++)
                        UnionFace(faces[0], faces[i]);
                }
                
                // Group faces by root
                var instanceGroups = new Dictionary<int, List<int>>();
                for (int fi = 0; fi < allFaces.Count; fi++)
                {
                    int root = FindFace(fi);
                    if (!instanceGroups.ContainsKey(root))
                        instanceGroups[root] = new List<int>();
                    instanceGroups[root].Add(fi);
                }
                
                int numInstances = instanceGroups.Count;
                
                // Write OBJ with groups for each instance
                var objPath = Path.Combine(targetDir, $"CK24_{ck24:X6}_{tiles.Count}tiles_{numInstances}inst.obj");
                using var sw = new StreamWriter(objPath);
                sw.WriteLine($"# Multi-tile CK24 0x{ck24:X6} ({typeLabel})");
                sw.WriteLine($"# Type byte: 0x{typeByte:X2}");
                sw.WriteLine($"# Tiles: {tiles.Count}");
                sw.WriteLine($"# Vertices: {allVerts.Count} (Mesh) + {allMscnPoints.Count} (MSCN)");
                sw.WriteLine($"# Faces: {allFaces.Count}");
                sw.WriteLine($"# Instances (spatial clusters): {numInstances}");
                sw.WriteLine();
                
                // Write Mesh Vertices
                foreach (var v in allVerts)
                    sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                    
                // Write MSCN Vertices (appended to vertex list)
                foreach (var v in allMscnPoints)
                    sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4} # MSCN");
                
                int mscnBaseIndex = allVerts.Count + 1; // OBJ indices are 1-based
                
                sw.WriteLine();
                
                // Write items grouped by instance
                int grpNum = 0;
                foreach (var grp in instanceGroups.Values.OrderByDescending(g => g.Count))
                {
                    sw.WriteLine($"g instance_{grpNum++}");
                    
                    // Faces
                    foreach (var fi in grp)
                        sw.WriteLine($"f {string.Join(" ", allFaces[fi])}");
                        
                    // MSCN Points
                    var grpPoints = new HashSet<int>();
                    foreach(var fi in grp)
                    {
                        if(faceToMscnMap.TryGetValue(fi, out int mscnLocalIdx))
                            grpPoints.Add(mscnLocalIdx);
                    }
                    
                    if (grpPoints.Count > 0)
                    {
                        sw.WriteLine($"# MSCN Points: {grpPoints.Count}");
                        foreach(var pi in grpPoints)
                            sw.WriteLine($"p {mscnBaseIndex + pi}");
                    }
                }
                
                Console.WriteLine($"  [{typeLabel}] CK24_{ck24:X6} ({allVerts.Count} verts, {numInstances} instances)");
                exported++;
            }
        }
        
        Console.WriteLine($"\nExported {exported} multi-tile OBJ files to {outputDir}");
        return 0;
    }
    
    static int ExportMscnPoints(string pm4Path, string? outputDir)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        outputDir ??= Path.Combine(Path.GetTempPath(), "pm4_mscn");
        Directory.CreateDirectory(outputDir);
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MSCN Export from {baseName} ===");
        Console.WriteLine($"MSCN points: {pm4.SceneNodes.Count}");
        Console.WriteLine($"MSVT vertices: {pm4.MeshVertices.Count}");
        
        // Build MSVT lookup set
        var msvtSet = new HashSet<(float, float, float)>();
        foreach (var v in pm4.MeshVertices)
            msvtSet.Add((MathF.Round(v.X, 1), MathF.Round(v.Y, 1), MathF.Round(v.Z, 1)));
        
        // Separate MSCN into unique and shared
        var uniqueMscn = new List<Vector3>();
        var sharedMscn = new List<Vector3>();
        
        foreach (var n in pm4.SceneNodes)
        {
            if (msvtSet.Contains((MathF.Round(n.X, 1), MathF.Round(n.Y, 1), MathF.Round(n.Z, 1))))
                sharedMscn.Add(n);
            else
                uniqueMscn.Add(n);
        }
        
        Console.WriteLine($"\nUnique MSCN (not in MSVT): {uniqueMscn.Count}");
        Console.WriteLine($"Shared MSCN (also in MSVT): {sharedMscn.Count}");
        
        // Export unique MSCN as point cloud
        var objPath = Path.Combine(outputDir, $"{baseName}_MSCN_unique.obj");
        using (var sw = new StreamWriter(objPath))
        {
            sw.WriteLine($"# MSCN unique points (not in MSVT)");
            sw.WriteLine($"# Count: {uniqueMscn.Count}");
            sw.WriteLine();
            foreach (var v in uniqueMscn)
                sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
            
            // Create point elements (p command for point cloud)
            for (int i = 1; i <= uniqueMscn.Count; i++)
                sw.WriteLine($"p {i}");
        }
        Console.WriteLine($"\nExported: {objPath}");
        
        // Export full MSCN
        var allPath = Path.Combine(outputDir, $"{baseName}_MSCN_all.obj");
        using (var sw = new StreamWriter(allPath))
        {
            sw.WriteLine($"# All MSCN points");
            sw.WriteLine($"# Count: {pm4.SceneNodes.Count}");
            sw.WriteLine();
            foreach (var v in pm4.SceneNodes)
                sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
        }
        Console.WriteLine($"Exported: {allPath}");
        
        // Export MSVT for comparison
        var msvtPath = Path.Combine(outputDir, $"{baseName}_MSVT_all.obj");
        using (var sw = new StreamWriter(msvtPath))
        {
            sw.WriteLine($"# All MSVT vertices");
            sw.WriteLine($"# Count: {pm4.MeshVertices.Count}");
            sw.WriteLine();
            foreach (var v in pm4.MeshVertices)
                sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
        }
        Console.WriteLine($"Exported: {msvtPath}");
        
        Console.WriteLine($"\nOutput directory: {outputDir}");
        return 0;
    }
    
    /// <summary>
    /// Trace MSLK → MSUR linkage to understand how navigation nodes connect to surfaces.
    /// Key question: Does MSLK.RefIndex (msur_index) point to MSUR entries?
    /// </summary>
    static int RunMslkTrace(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MSLK → MSUR Trace: {baseName} ===\n");
        Console.WriteLine($"MSLK entries: {pm4.LinkEntries.Count}");
        Console.WriteLine($"MSUR entries: {pm4.Surfaces.Count}");
        Console.WriteLine($"MPRL entries: {pm4.PositionRefs.Count}");
        Console.WriteLine($"MSVT entries: {pm4.MeshVertices.Count}");
        Console.WriteLine($"MSCN entries: {pm4.SceneNodes.Count}");
        
        // Analyze RefIndex distribution
        var validMsurRefs = pm4.LinkEntries.Count(e => e.RefIndex < pm4.Surfaces.Count);
        var outOfRangeMsurRefs = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.Surfaces.Count);
        
        Console.WriteLine($"\n=== RefIndex Analysis ===");
        Console.WriteLine($"  RefIndex < MSUR.Count ({pm4.Surfaces.Count}): {validMsurRefs}");
        Console.WriteLine($"  RefIndex >= MSUR.Count: {outOfRangeMsurRefs}");
        
        // Test hypothesis: RefIndex directly indexes into MSUR
        Console.WriteLine($"\n=== Testing: RefIndex is MSUR index ===");
        int validLinks = 0;
        var linkedCk24s = new Dictionary<uint, int>();
        
        foreach (var mslk in pm4.LinkEntries.Take(100)) // Sample first 100
        {
            if (mslk.RefIndex < pm4.Surfaces.Count)
            {
                var msur = pm4.Surfaces[mslk.RefIndex];
                validLinks++;
                if (!linkedCk24s.ContainsKey(msur.CK24))
                    linkedCk24s[msur.CK24] = 0;
                linkedCk24s[msur.CK24]++;
            }
        }
        Console.WriteLine($"  Valid MSLK→MSUR links (first 100): {validLinks}/100");
        Console.WriteLine($"  Unique CK24s through these links: {linkedCk24s.Count}");
        
        // Show CK24 distribution via MSLK
        Console.WriteLine($"\n=== CK24 values reached via MSLK.RefIndex (sample) ===");
        foreach (var (ck24, count) in linkedCk24s.OrderByDescending(kv => kv.Value).Take(10))
        {
            Console.WriteLine($"    CK24 0x{ck24:X6}: {count} MSLK entries");
        }
        
        // Alternative hypothesis: RefIndex >= MPRL.Count means MSVT index
        Console.WriteLine($"\n=== Testing: Dual-index pattern (like MPRR) ===");
        Console.WriteLine($"  If RefIndex < MPRL.Count ({pm4.PositionRefs.Count}) → MPRL");
        Console.WriteLine($"  If RefIndex >= MPRL.Count → MSVT");
        
        int toMprl = pm4.LinkEntries.Count(e => e.RefIndex < pm4.PositionRefs.Count);
        int toMsvt = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.PositionRefs.Count && 
                                                  e.RefIndex < pm4.PositionRefs.Count + pm4.MeshVertices.Count);
        int outOfAll = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.PositionRefs.Count + pm4.MeshVertices.Count);
        
        Console.WriteLine($"  RefIndex → MPRL: {toMprl} ({100.0 * toMprl / pm4.LinkEntries.Count:F1}%)");
        Console.WriteLine($"  RefIndex → MSVT: {toMsvt} ({100.0 * toMsvt / pm4.LinkEntries.Count:F1}%)");
        Console.WriteLine($"  Out of range: {outOfAll}");
        
        // MSLK GroupObjectId analysis - is this the object grouping key?
        Console.WriteLine($"\n=== MSLK.GroupObjectId → Object Grouping? ===");
        var uniqueGroupIds = pm4.LinkEntries.Select(e => e.GroupObjectId).Distinct().Count();
        Console.WriteLine($"  Unique GroupObjectId values: {uniqueGroupIds}");
        
        // Check if GroupObjectId correlates with CK24
        var groupToCk24 = new Dictionary<uint, HashSet<uint>>();
        foreach (var mslk in pm4.LinkEntries)
        {
            if (mslk.RefIndex < pm4.Surfaces.Count)
            {
                var ck24 = pm4.Surfaces[mslk.RefIndex].CK24;
                if (!groupToCk24.ContainsKey(mslk.GroupObjectId))
                    groupToCk24[mslk.GroupObjectId] = new HashSet<uint>();
                groupToCk24[mslk.GroupObjectId].Add(ck24);
            }
        }
        
        var singleCk24Groups = groupToCk24.Count(kv => kv.Value.Count == 1);
        var multiCk24Groups = groupToCk24.Count(kv => kv.Value.Count > 1);
        Console.WriteLine($"  GroupObjectIds mapping to 1 CK24: {singleCk24Groups}");
        Console.WriteLine($"  GroupObjectIds mapping to multiple CK24s: {multiCk24Groups}");
        
        // Show sample linkages
        Console.WriteLine($"\n=== Sample MSLK → MSUR → CK24 Chains ===");
        foreach (var mslk in pm4.LinkEntries.Where(e => e.RefIndex < pm4.Surfaces.Count).Take(5))
        {
            var msur = pm4.Surfaces[mslk.RefIndex];
            Console.WriteLine($"  MSLK[Type={mslk.TypeFlags}, Subtype={mslk.Subtype}, GroupId=0x{mslk.GroupObjectId:X8}]");
            Console.WriteLine($"    → RefIndex={mslk.RefIndex} → MSUR[CK24=0x{msur.CK24:X6}, GroupKey={msur.GroupKey}]");
            Console.WriteLine($"    → MspiFirst={mslk.MspiFirstIndex}, Count={mslk.MspiIndexCount}");
        }
        
        return 0;
    }
    
    /// <summary>
    /// Analyze MSCN data to understand how it relates to objects.
    /// Key question: How do we group MSCN vertices by object?
    /// </summary>
    static int RunMscnAnalysis(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MSCN Object Grouping Analysis: {baseName} ===\n");
        Console.WriteLine($"MSCN entries: {pm4.SceneNodes.Count}");
        Console.WriteLine($"MSUR entries: {pm4.Surfaces.Count}");
        Console.WriteLine($"Unique MdosIndex values in MSUR: {pm4.Surfaces.Select(s => s.MdosIndex).Distinct().Count()}");
        
        // Key hypothesis: MdosIndex indexes into MSCN
        Console.WriteLine($"\n=== Testing: MSUR.MdosIndex → MSCN index? ===");
        var mdosRange = pm4.Surfaces.Where(s => s.MdosIndex != 0)
            .Select(s => s.MdosIndex)
            .DefaultIfEmpty()
            .ToList();
        
        if (mdosRange.Any())
        {
            var minMdos = mdosRange.Min();
            var maxMdos = mdosRange.Max();
            Console.WriteLine($"  MdosIndex range: {minMdos} to {maxMdos}");
            Console.WriteLine($"  MSCN count: {pm4.SceneNodes.Count}");
            Console.WriteLine($"  Max MdosIndex < MSCN.Count? {maxMdos < pm4.SceneNodes.Count}");
            
            // If MdosIndex < MSCN.Count, test the linkage
            if (maxMdos < pm4.SceneNodes.Count)
            {
                Console.WriteLine($"\n  ✓ MdosIndex could be MSCN index!");
                
                // Group MSCN by CK24 via MdosIndex linkage
                var ck24ToMscn = new Dictionary<uint, List<Vector3>>();
                foreach (var surf in pm4.Surfaces.Where(s => s.MdosIndex < pm4.SceneNodes.Count))
                {
                    var mscnVert = pm4.SceneNodes[(int)surf.MdosIndex];
                    if (!ck24ToMscn.ContainsKey(surf.CK24))
                        ck24ToMscn[surf.CK24] = new List<Vector3>();
                    ck24ToMscn[surf.CK24].Add(mscnVert);
                }
                
                Console.WriteLine($"\n=== MSCN grouped by CK24 via MdosIndex ===");
                foreach (var (ck24, verts) in ck24ToMscn.OrderByDescending(kv => kv.Value.Count).Take(10))
                {
                    var bounds = ComputeBounds(verts);
                    var size = bounds.max - bounds.min;
                    Console.WriteLine($"  CK24 0x{ck24:X6}: {verts.Count} MSCN verts, Size: {size.X:F0}x{size.Y:F0}x{size.Z:F0}");
                }
            }
            else
            {
                Console.WriteLine($"\n  ✗ MdosIndex exceeds MSCN count - different interpretation needed");
            }
        }
        
        // Look for MDSF chunk in unparsed chunks (links MSUR to MDOS)
        Console.WriteLine($"\n=== MDSF Chunk (MSUR→MDOS linkage) ===");
        // MDSF is not currently parsed - let's check if it exists in the file
        Console.WriteLine($"  (MDSF parsing not yet implemented - would link msur_index → mdos_index)");
        
        // Alternative: Check if MSCN vertices spatially correlate with CK24 objects
        Console.WriteLine($"\n=== MSCN Spatial Correlation with CK24 Objects ===");
        
        // Get bounds of each CK24 object from MSVT
        var ck24Bounds = new Dictionary<uint, (Vector3 min, Vector3 max)>();
        foreach (var group in pm4.Surfaces.GroupBy(s => s.CK24))
        {
            var allVerts = new List<Vector3>();
            foreach (var surf in group)
            {
                var verts = GetSurfaceVertices(pm4, surf);
                allVerts.AddRange(verts);
            }
            if (allVerts.Count > 0)
            {
                ck24Bounds[group.Key] = ComputeBounds(allVerts);
            }
        }
        
        // For each MSCN vertex, find which CK24 bounding box it falls into
        var mscnToCk24 = new Dictionary<uint, int>();
        int mscnWithoutCk24 = 0;
        
        foreach (var mscnVert in pm4.SceneNodes)
        {
            uint matchedCk24 = 0;
            foreach (var (ck24, bounds) in ck24Bounds)
            {
                if (ck24 == 0) continue; // Skip nav mesh
                if (mscnVert.X >= bounds.min.X && mscnVert.X <= bounds.max.X &&
                    mscnVert.Y >= bounds.min.Y && mscnVert.Y <= bounds.max.Y &&
                    mscnVert.Z >= bounds.min.Z - 5 && mscnVert.Z <= bounds.max.Z + 5) // Z tolerance
                {
                    matchedCk24 = ck24;
                    break;
                }
            }
            
            if (matchedCk24 != 0)
            {
                if (!mscnToCk24.ContainsKey(matchedCk24))
                    mscnToCk24[matchedCk24] = 0;
                mscnToCk24[matchedCk24]++;
            }
            else
            {
                mscnWithoutCk24++;
            }
        }
        
        Console.WriteLine($"  MSCN verts inside CK24 bounds: {pm4.SceneNodes.Count - mscnWithoutCk24}");
        Console.WriteLine($"  MSCN verts outside all CK24 bounds: {mscnWithoutCk24}");
        Console.WriteLine($"\n  Top CK24s by MSCN containment:");
        foreach (var (ck24, count) in mscnToCk24.OrderByDescending(kv => kv.Value).Take(10))
        {
            Console.WriteLine($"    CK24 0x{ck24:X6}: {count} MSCN verts");
        }
        
        return 0;
    }
    
    /// <summary>
    /// Fresh raw dump of PM4 file - examining structure without old assumptions.
    /// Shows raw chunk data, entry counts at different sizes, cross-references.
    /// </summary>
    static int RunRawDump(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var data = File.ReadAllBytes(pm4Path);
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== FRESH PM4 RAW ANALYSIS: {baseName} ===");
        Console.WriteLine($"File size: {data.Length:N0} bytes\n");
        
        // Parse chunks raw
        var chunks = new List<(string sig, uint size, long offset, byte[] data)>();
        using (var ms = new MemoryStream(data))
        using (var br = new BinaryReader(ms))
        {
            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                long chunkStart = br.BaseStream.Position;
                var sigBytes = br.ReadBytes(4);
                Array.Reverse(sigBytes); // Reversed on disk
                string sig = Encoding.ASCII.GetString(sigBytes);
                uint size = br.ReadUInt32();
                byte[] chunkData = br.ReadBytes((int)size);
                chunks.Add((sig, size, chunkStart, chunkData));
            }
        }
        
        Console.WriteLine("=== CHUNK INVENTORY ===");
        Console.WriteLine($"Total chunks: {chunks.Count}\n");
        
        foreach (var (sig, size, offset, chunkData) in chunks)
        {
            Console.WriteLine($"  {sig}: {size:N0} bytes @ offset 0x{offset:X8}");
            
            // Try common entry sizes
            if (size >= 4)
            {
                var possibleSizes = new[] { 4, 8, 12, 16, 20, 24, 32 };
                var validSizes = possibleSizes.Where(s => size % s == 0).ToList();
                if (validSizes.Count > 0)
                {
                    Console.Write($"        Possible entries: ");
                    Console.WriteLine(string.Join(", ", validSizes.Select(s => $"{size/s} @ {s}B")));
                }
                
                // Show first few bytes as hex
                int previewLen = Math.Min(32, chunkData.Length);
                Console.WriteLine($"        First {previewLen} bytes: {BitConverter.ToString(chunkData, 0, previewLen).Replace("-", " ")}");
            }
        }
        
        // Deep analysis of key chunks with fresh eyes
        Console.WriteLine("\n=== MSLK DEEP ANALYSIS (20 bytes/entry - per wowdev wiki) ===");
        var mslkChunk = chunks.FirstOrDefault(c => c.sig == "MSLK");
        if (mslkChunk.data != null && mslkChunk.data.Length >= 20)
        {
            int entryCount = mslkChunk.data.Length / 20;
            Console.WriteLine($"Entry count: {entryCount}");
            
            // Analyze each field position
            using var br = new BinaryReader(new MemoryStream(mslkChunk.data));
            
            // Collect stats on each field
            var field0x00 = new List<byte>();   // uint8
            var field0x01 = new List<byte>();   // uint8  
            var field0x02 = new List<ushort>(); // uint16 (padding per wiki)
            var field0x04 = new List<uint>();   // uint32 (index somewhere)
            var field0x08 = new List<int>();    // int24 (MSPI first)
            var field0x0b = new List<byte>();   // uint8 (MSPI count)
            var field0x0c = new List<uint>();   // uint32 (0xFFFFFFFF per wiki)
            var field0x10 = new List<ushort>(); // uint16 (msur_index per wiki!)
            var field0x12 = new List<ushort>(); // uint16 (0x8000 per wiki)
            
            for (int i = 0; i < entryCount; i++)
            {
                field0x00.Add(br.ReadByte());
                field0x01.Add(br.ReadByte());
                field0x02.Add(br.ReadUInt16());
                field0x04.Add(br.ReadUInt32());
                
                // Read int24
                byte[] b = br.ReadBytes(3);
                int val = b[0] | (b[1] << 8) | (b[2] << 16);
                if ((val & 0x800000) != 0) val |= unchecked((int)0xFF000000);
                field0x08.Add(val);
                
                field0x0b.Add(br.ReadByte());
                field0x0c.Add(br.ReadUInt32());
                field0x10.Add(br.ReadUInt16());
                field0x12.Add(br.ReadUInt16());
            }
            
            Console.WriteLine("\nField statistics:");
            Console.WriteLine($"  [0x00] byte:   unique={field0x00.Distinct().Count()}, range={field0x00.Min()}-{field0x00.Max()}");
            Console.WriteLine($"  [0x01] byte:   unique={field0x01.Distinct().Count()}, range={field0x01.Min()}-{field0x01.Max()}");
            Console.WriteLine($"  [0x02] ushort: unique={field0x02.Distinct().Count()}, range={field0x02.Min()}-{field0x02.Max()}");
            Console.WriteLine($"  [0x04] uint:   unique={field0x04.Distinct().Count()}, range={field0x04.Min()}-{field0x04.Max()}");
            Console.WriteLine($"  [0x08] int24:  unique={field0x08.Distinct().Count()}, range={field0x08.Min()}-{field0x08.Max()}");
            Console.WriteLine($"  [0x0B] byte:   unique={field0x0b.Distinct().Count()}, range={field0x0b.Min()}-{field0x0b.Max()}");
            Console.WriteLine($"  [0x0C] uint:   unique={field0x0c.Distinct().Count()}, values: {string.Join(",", field0x0c.Distinct().Take(5).Select(v => $"0x{v:X8}"))}");
            Console.WriteLine($"  [0x10] ushort: unique={field0x10.Distinct().Count()}, range={field0x10.Min()}-{field0x10.Max()}");
            Console.WriteLine($"  [0x12] ushort: unique={field0x12.Distinct().Count()}, values: {string.Join(",", field0x12.Distinct().Take(5).Select(v => $"0x{v:X4}"))}");
            
            // Cross-reference analysis
            var msurChunk = chunks.FirstOrDefault(c => c.sig == "MSUR");
            var mscnChunk = chunks.FirstOrDefault(c => c.sig == "MSCN");
            var mprlChunk = chunks.FirstOrDefault(c => c.sig == "MPRL");
            var mspiChunk = chunks.FirstOrDefault(c => c.sig == "MSPI");
            
            int msurCount = msurChunk.data?.Length / 32 ?? 0;
            int mscnCount = mscnChunk.data?.Length / 12 ?? 0;
            int mprlCount = mprlChunk.data?.Length / 24 ?? 0;
            int mspiCount = mspiChunk.data?.Length / 4 ?? 0;
            
            Console.WriteLine("\nCross-reference validation:");
            Console.WriteLine($"  MSUR count: {msurCount}");
            Console.WriteLine($"  MSCN count: {mscnCount}");
            Console.WriteLine($"  MPRL count: {mprlCount}");
            Console.WriteLine($"  MSPI count: {mspiCount}");
            
            int refInMsur = field0x10.Count(v => v < msurCount);
            int refInMscn = field0x10.Count(v => v < mscnCount);
            int refInMprl = field0x10.Count(v => v < mprlCount);
            
            Console.WriteLine($"\n  [0x10] as MSUR index: {refInMsur}/{entryCount} valid ({100.0*refInMsur/entryCount:F1}%)");
            Console.WriteLine($"  [0x10] as MSCN index: {refInMscn}/{entryCount} valid ({100.0*refInMscn/entryCount:F1}%)");
            Console.WriteLine($"  [0x10] as MPRL index: {refInMprl}/{entryCount} valid ({100.0*refInMprl/entryCount:F1}%)");
            
            // MSPI validation
            int mspiValid = field0x08.Count(v => v >= 0 && v < mspiCount);
            Console.WriteLine($"\n  [0x08] as MSPI index: {mspiValid}/{entryCount} valid ({100.0*mspiValid/entryCount:F1}%)");
            
            // Show first 5 entries raw
            Console.WriteLine("\nFirst 5 entries (raw hex per field):");
            br.BaseStream.Position = 0;
            for (int i = 0; i < Math.Min(5, entryCount); i++)
            {
                var bytes = br.ReadBytes(20);
                Console.WriteLine($"  [{i}] {BitConverter.ToString(bytes).Replace("-", " ")}");
            }
        }
        
        // MSUR analysis
        Console.WriteLine("\n=== MSUR DEEP ANALYSIS (32 bytes/entry) ===");
        var msurData = chunks.FirstOrDefault(c => c.sig == "MSUR");
        if (msurData.data != null && msurData.data.Length >= 32)
        {
            int entryCount = msurData.data.Length / 32;
            Console.WriteLine($"Entry count: {entryCount}");
            
            using var br = new BinaryReader(new MemoryStream(msurData.data));
            
            // Fields per current understanding
            var groupKeys = new List<byte>();
            var indexCounts = new List<byte>();
            var attrMasks = new List<byte>();
            var msviFirsts = new List<uint>();
            var mdosIndices = new List<uint>();
            var packedParams = new List<uint>();
            
            for (int i = 0; i < entryCount; i++)
            {
                groupKeys.Add(br.ReadByte());      // 0x00
                indexCounts.Add(br.ReadByte());    // 0x01
                attrMasks.Add(br.ReadByte());      // 0x02
                br.ReadByte();                      // 0x03 padding
                br.ReadSingle(); br.ReadSingle(); br.ReadSingle(); // normals 0x04-0x0F
                br.ReadSingle();                    // height 0x10
                msviFirsts.Add(br.ReadUInt32());   // 0x14
                mdosIndices.Add(br.ReadUInt32());  // 0x18
                packedParams.Add(br.ReadUInt32()); // 0x1C
            }
            
            Console.WriteLine("\nField statistics:");
            Console.WriteLine($"  [0x00] GroupKey:   unique={groupKeys.Distinct().Count()}, values: {string.Join(",", groupKeys.Distinct().OrderBy(x=>x).Take(10))}");
            Console.WriteLine($"  [0x01] IndexCount: unique={indexCounts.Distinct().Count()}, range={indexCounts.Min()}-{indexCounts.Max()}");
            Console.WriteLine($"  [0x14] MsviFirst:  unique={msviFirsts.Distinct().Count()}, range={msviFirsts.Min()}-{msviFirsts.Max()}");
            Console.WriteLine($"  [0x18] MdosIndex:  unique={mdosIndices.Distinct().Count()}, range={mdosIndices.Min()}-{mdosIndices.Max()}");
            
            var ck24s = packedParams.Select(p => (p & 0xFFFFFF00) >> 8).Distinct().ToList();
            Console.WriteLine($"  [0x1C] CK24:       unique={ck24s.Count}");
            Console.WriteLine($"         Top CK24s: {string.Join(", ", ck24s.OrderByDescending(x => packedParams.Count(p => ((p & 0xFFFFFF00) >> 8) == x)).Take(5).Select(x => $"0x{x:X6}"))}");
            
            // MSCN cross-ref
            var mscnCount = chunks.FirstOrDefault(c => c.sig == "MSCN").data?.Length / 12 ?? 0;
            int mdosInMscn = mdosIndices.Count(v => v < mscnCount);
            Console.WriteLine($"\n  MdosIndex < MSCN.Count ({mscnCount}): {mdosInMscn}/{entryCount} ({100.0*mdosInMscn/entryCount:F1}%)");
        }
        
        return 0;
    }
    
    /// <summary>
    /// Analyze cross-tile linking in PM4 files.
    /// Goal: Understand how data is distributed across tiles and how tiles reference each other.
    /// </summary>
    static int RunCrossTileAnalysis(string directory)
    {
        if (!Directory.Exists(directory))
        {
            Console.WriteLine($"Directory not found: {directory}");
            return 1;
        }
        
        var pm4Files = Directory.GetFiles(directory, "*.pm4");
        Console.WriteLine($"=== CROSS-TILE PM4 ANALYSIS ===");
        Console.WriteLine($"Found {pm4Files.Length} PM4 files in {Path.GetFileName(directory)}\n");
        
        // Parse tile coords from filename and load each file
        var tiles = new Dictionary<(int x, int y), Pm4File>();
        var tileStats = new List<(int x, int y, Pm4File pm4)>();
        
        foreach (var path in pm4Files)
        {
            var match = System.Text.RegularExpressions.Regex.Match(
                Path.GetFileName(path), @"(\d+)_(\d+)\.pm4$");
            if (!match.Success) continue;
            
            int x = int.Parse(match.Groups[1].Value);
            int y = int.Parse(match.Groups[2].Value);
            
            try
            {
                var pm4 = Pm4File.Parse(File.ReadAllBytes(path));
                tiles[(x, y)] = pm4;
                tileStats.Add((x, y, pm4));
            }
            catch { /* skip unreadable files */ }
        }
        
        Console.WriteLine($"Successfully loaded {tiles.Count} tiles\n");
        
        // Analyze cross-tile references via LinkId
        Console.WriteLine("=== MSLK CROSS-TILE REFERENCES ===");
        var allCrossRefs = new List<(int srcX, int srcY, int destX, int destY, int count)>();
        int totalLocal = 0, totalCross = 0;
        
        foreach (var (x, y, pm4) in tileStats.Take(20)) // Sample first 20 tiles
        {
            if (pm4.LinkEntries.Count == 0) continue;
            
            // Parse LinkId to extract tile coordinates
            var localRefs = 0;
            var crossRefs = new Dictionary<(int, int), int>();
            
            foreach (var lnk in pm4.LinkEntries)
            {
                // LinkId format: 0xFFFFYYXX (little endian)
                uint linkId = lnk.LinkId;
                int linkX = (int)(linkId & 0xFF);
                int linkY = (int)((linkId >> 8) & 0xFF);
                
                if (linkX == x && linkY == y)
                    localRefs++;
                else
                {
                    var key = (linkX, linkY);
                    crossRefs[key] = crossRefs.GetValueOrDefault(key) + 1;
                }
            }
            
            totalLocal += localRefs;
            totalCross += crossRefs.Values.Sum();
            
            if (crossRefs.Count > 0)
            {
                Console.WriteLine($"  Tile {x}_{y}: {localRefs} local, {crossRefs.Values.Sum()} cross-tile");
                foreach (var (dest, cnt) in crossRefs.OrderByDescending(kv => kv.Value))
                {
                    Console.WriteLine($"    → Tile {dest.Item1}_{dest.Item2}: {cnt} refs");
                    allCrossRefs.Add((x, y, dest.Item1, dest.Item2, cnt));
                }
            }
        }
        
        Console.WriteLine($"\nTotal: {totalLocal} local refs, {totalCross} cross-tile refs");
        Console.WriteLine($"Cross-tile percentage: {100.0 * totalCross / (totalLocal + totalCross):F2}%");
        
        // Analyze CK24 distribution across tiles
        Console.WriteLine("\n=== CK24 DISTRIBUTION ACROSS TILES ===");
        var ck24ToTiles = new Dictionary<uint, HashSet<(int, int)>>();
        var ck24ToSurfaces = new Dictionary<uint, int>();
        
        foreach (var (x, y, pm4) in tileStats)
        {
            foreach (var surf in pm4.Surfaces)
            {
                if (surf.CK24 == 0) continue; // Skip nav mesh
                
                if (!ck24ToTiles.ContainsKey(surf.CK24))
                {
                    ck24ToTiles[surf.CK24] = new HashSet<(int, int)>();
                    ck24ToSurfaces[surf.CK24] = 0;
                }
                ck24ToTiles[surf.CK24].Add((x, y));
                ck24ToSurfaces[surf.CK24]++;
            }
        }
        
        var multiTileCk24s = ck24ToTiles.Where(kv => kv.Value.Count > 1)
            .OrderByDescending(kv => kv.Value.Count)
            .ToList();
        
        Console.WriteLine($"Total unique CK24s: {ck24ToTiles.Count}");
        Console.WriteLine($"CK24s spanning 1 tile: {ck24ToTiles.Count(kv => kv.Value.Count == 1)}");
        Console.WriteLine($"CK24s spanning 2+ tiles: {multiTileCk24s.Count}");
        
        Console.WriteLine("\nLargest multi-tile objects:");
        foreach (var (ck24, tileset) in multiTileCk24s.Take(10))
        {
            Console.WriteLine($"  CK24 0x{ck24:X6}: {tileset.Count} tiles, {ck24ToSurfaces[ck24]} total surfaces");
            Console.WriteLine($"    Tiles: {string.Join(", ", tileset.OrderBy(t => t.Item1).ThenBy(t => t.Item2).Select(t => $"{t.Item1}_{t.Item2}"))}");
        }
        
        // Analyze index spaces - do they overlap or segment?
        Console.WriteLine("\n=== INDEX SPACE ANALYSIS ===");
        Console.WriteLine("Checking if MSVI/MSVT indices are local or global...\n");
        
        foreach (var (x, y, pm4) in tileStats.Take(5))
        {
            if (pm4.Surfaces.Count == 0) continue;
            
            var msviMin = pm4.Surfaces.Min(s => s.MsviFirstIndex);
            var msviMax = pm4.Surfaces.Max(s => s.MsviFirstIndex + (uint)s.IndexCount);
            var msvtCount = pm4.MeshVertices.Count;
            
            Console.WriteLine($"Tile {x}_{y}:");
            Console.WriteLine($"  MSVI range used: {msviMin} - {msviMax}");
            Console.WriteLine($"  MSVT count: {msvtCount}");
            Console.WriteLine($"  Max MSVI index < MSVI.Count: {msviMax <= pm4.MeshIndices.Count} (local={pm4.MeshIndices.Count})");
            
            // Check if any surfaces reference beyond local MSVI
            var outOfRange = pm4.Surfaces.Count(s => s.MsviFirstIndex + s.IndexCount > pm4.MeshIndices.Count);
            if (outOfRange > 0)
                Console.WriteLine($"  ⚠️ {outOfRange} surfaces reference MSVI beyond local count!");
            else
                Console.WriteLine($"  ✓ All surfaces reference local MSVI");
        }
        
        // Hypothesis test: Do MSLK cross-tile refs point to indices in the other tile?
        Console.WriteLine("\n=== CROSS-TILE INDEX RESOLUTION ===");
        Console.WriteLine("Testing if cross-tile MSLK entries reference data in the target tile...\n");
        
        int resolved = 0, unresolved = 0;
        
        foreach (var (srcX, srcY, destX, destY, count) in allCrossRefs.Take(10))
        {
            if (!tiles.ContainsKey((srcX, srcY)) || !tiles.ContainsKey((destX, destY))) continue;
            
            var srcPm4 = tiles[(srcX, srcY)];
            var destPm4 = tiles[(destX, destY)];
            
            // Find MSLK entries that reference the dest tile
            var crossEntries = srcPm4.LinkEntries
                .Where(lnk => (lnk.LinkId & 0xFF) == destX && ((lnk.LinkId >> 8) & 0xFF) == destY)
                .Take(5);
            
            foreach (var lnk in crossEntries)
            {
                // Does RefIndex make sense in DEST tile?
                bool validInDest = lnk.RefIndex < destPm4.Surfaces.Count;
                bool validInSrc = lnk.RefIndex < srcPm4.Surfaces.Count;
                
                if (validInDest)
                {
                    resolved++;
                    var destSurf = destPm4.Surfaces[lnk.RefIndex];
                    Console.WriteLine($"  Cross-ref {srcX}_{srcY}→{destX}_{destY}: RefIndex={lnk.RefIndex} → DEST.CK24=0x{destSurf.CK24:X6}");
                }
                else if (validInSrc)
                {
                    unresolved++;
                    var srcSurf = srcPm4.Surfaces[lnk.RefIndex];
                    Console.WriteLine($"  Cross-ref {srcX}_{srcY}→{destX}_{destY}: RefIndex={lnk.RefIndex} → SRC.CK24=0x{srcSurf.CK24:X6} (local!)");
                }
                else
                {
                    Console.WriteLine($"  Cross-ref {srcX}_{srcY}→{destX}_{destY}: RefIndex={lnk.RefIndex} INVALID in both!");
                }
            }
        }
        
        Console.WriteLine($"\nResolved in DEST tile: {resolved}");
        Console.WriteLine($"Resolved in SRC (local): {unresolved}");
        
        return 0;
    }
    
    /// <summary>
    /// Analyze MDSF chunk to understand MSUR→MDOS linkage and investigate MdosIndex.
    /// MDSF per wowdev wiki: struct { uint32_t msur_index; uint32_t mdos_index; }
    /// </summary>
    static int RunMdsfAnalysis(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var data = File.ReadAllBytes(pm4Path);
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MDSF/MDOS ANALYSIS: {baseName} ===\n");
        
        // Parse chunks raw to get MDSF, MDOS, MDBH, MDBI, MDBF
        var chunks = new Dictionary<string, List<byte[]>>();
        using (var ms = new MemoryStream(data))
        using (var br = new BinaryReader(ms))
        {
            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                var sigBytes = br.ReadBytes(4);
                Array.Reverse(sigBytes);
                string sig = Encoding.ASCII.GetString(sigBytes);
                uint size = br.ReadUInt32();
                byte[] chunkData = br.ReadBytes((int)size);
                
                if (!chunks.ContainsKey(sig))
                    chunks[sig] = new List<byte[]>();
                chunks[sig].Add(chunkData);
            }
        }
        
        // Parse standard PM4 for cross-reference
        var pm4 = Pm4File.Parse(data);
        
        Console.WriteLine("=== DESTRUCTIBLE BUILDING CHUNKS ===");
        
        // MDBH: count of destructible buildings
        if (chunks.TryGetValue("MDBH", out var mdbhList) && mdbhList.Count > 0)
        {
            uint buildingCount = BitConverter.ToUInt32(mdbhList[0], 0);
            Console.WriteLine($"MDBH: {buildingCount} destructible buildings");
        }
        
        // MDBI: building indices (file IDs)
        if (chunks.TryGetValue("MDBI", out var mdbiList))
        {
            Console.WriteLine($"MDBI: {mdbiList.Count} entries (building file IDs)");
            foreach (var mdbi in mdbiList.Take(5))
            {
                uint fileId = BitConverter.ToUInt32(mdbi, 0);
                Console.WriteLine($"  FileID: 0x{fileId:X8} ({fileId})");
            }
        }
        
        // MDBF: building filenames (null-terminated strings)
        if (chunks.TryGetValue("MDBF", out var mdbfList))
        {
            var nonEmpty = mdbfList.Where(b => b.Length > 0).ToList();
            Console.WriteLine($"MDBF: {mdbfList.Count} entries ({nonEmpty.Count} with data)");
            foreach (var mdbf in nonEmpty.Take(3))
            {
                string path = Encoding.UTF8.GetString(mdbf).TrimEnd('\0');
                Console.WriteLine($"  Path: {path}");
            }
        }
        
        // MDOS: destructible object states
        if (chunks.TryGetValue("MDOS", out var mdosList) && mdosList.Count > 0)
        {
            var mdosData = mdosList[0];
            int mdosCount = mdosData.Length / 8; // 8 bytes per entry
            Console.WriteLine($"\nMDOS: {mdosCount} entries (8 bytes each)");
            
            using var br = new BinaryReader(new MemoryStream(mdosData));
            for (int i = 0; i < Math.Min(10, mdosCount); i++)
            {
                uint field0 = br.ReadUInt32(); // building_index?
                uint field1 = br.ReadUInt32(); // destruction_state?
                Console.WriteLine($"  [{i}] Field0: 0x{field0:X8} ({field0}), Field1: 0x{field1:X8} ({field1})");
            }
        }
        
        // MDSF: THE KEY CHUNK - links MSUR to MDOS!
        Console.WriteLine("\n=== MDSF CHUNK (MSUR → MDOS LINKAGE) ===");
        if (chunks.TryGetValue("MDSF", out var mdsfList) && mdsfList.Count > 0)
        {
            var mdsfData = mdsfList[0];
            int mdsfCount = mdsfData.Length / 8; // struct { uint32 msur_index; uint32 mdos_index; }
            Console.WriteLine($"MDSF: {mdsfCount} entries");
            
            var msurToMdos = new Dictionary<uint, uint>();
            using var br = new BinaryReader(new MemoryStream(mdsfData));
            
            for (int i = 0; i < mdsfCount; i++)
            {
                uint msurIdx = br.ReadUInt32();
                uint mdosIdx = br.ReadUInt32();
                msurToMdos[msurIdx] = mdosIdx;
            }
            
            // Show first entries
            Console.WriteLine("\nFirst 10 mappings:");
            br.BaseStream.Position = 0;
            for (int i = 0; i < Math.Min(10, mdsfCount); i++)
            {
                uint msurIdx = br.ReadUInt32();
                uint mdosIdx = br.ReadUInt32();
                
                // Cross-reference with MSUR
                string ck24Info = "";
                if (msurIdx < pm4.Surfaces.Count)
                {
                    var surf = pm4.Surfaces[(int)msurIdx];
                    ck24Info = $", CK24=0x{surf.CK24:X6}, MdosIndex={surf.MdosIndex}";
                }
                Console.WriteLine($"  MSUR[{msurIdx}] → MDOS[{mdosIdx}]{ck24Info}");
            }
            
            // Analyze MDSF→MdosIndex correlation
            Console.WriteLine("\n=== MDSF vs MSUR.MdosIndex Correlation ===");
            int matches = 0, mismatches = 0;
            
            foreach (var (msurIdx, mdosIdx) in msurToMdos.Take(100))
            {
                if (msurIdx < pm4.Surfaces.Count)
                {
                    var surf = pm4.Surfaces[(int)msurIdx];
                    if (surf.MdosIndex == mdosIdx)
                        matches++;
                    else
                        mismatches++;
                }
            }
            Console.WriteLine($"MDSF.mdos_index == MSUR.MdosIndex: {matches}/{matches+mismatches} ({100.0*matches/(matches+mismatches):F1}%)");
            
            // Check if MDSF covers all surfaces with non-zero MdosIndex
            var surfacesWithMdos = pm4.Surfaces.Where(s => s.MdosIndex > 0).Count();
            Console.WriteLine($"\nSurfaces with MdosIndex > 0: {surfacesWithMdos}");
            Console.WriteLine($"MDSF entries: {mdsfCount}");
            Console.WriteLine($"Coverage: {100.0*mdsfCount/Math.Max(1,surfacesWithMdos):F1}%");
        }
        else
        {
            Console.WriteLine("MDSF chunk not found in this file!");
        }
        
        // Investigate MdosIndex → MSCN relationship
        Console.WriteLine("\n=== MSUR.MdosIndex → MSCN ANALYSIS ===");
        
        var mdosValues = pm4.Surfaces.Select(s => s.MdosIndex).ToList();
        var uniqueMdos = mdosValues.Distinct().Count();
        var mdosMax = mdosValues.Max();
        var mdosMin = mdosValues.Min();
        var mscnCount = pm4.SceneNodes.Count;
        
        Console.WriteLine($"MdosIndex range: {mdosMin} - {mdosMax}");
        Console.WriteLine($"Unique MdosIndex values: {uniqueMdos}");
        Console.WriteLine($"MSCN count: {mscnCount}");
        
        int validAsMscn = mdosValues.Count(v => v < mscnCount);
        Console.WriteLine($"MdosIndex < MSCN.Count: {validAsMscn}/{pm4.Surfaces.Count} ({100.0*validAsMscn/pm4.Surfaces.Count:F1}%)");
        
        // Group surfaces by MdosIndex and see if they cluster spatially
        Console.WriteLine("\n=== MdosIndex Clustering Check ===");
        var byMdos = pm4.Surfaces
            .Where(s => s.MdosIndex < mscnCount && s.MdosIndex > 0)
            .GroupBy(s => s.MdosIndex)
            .OrderByDescending(g => g.Count())
            .Take(5);
        
        foreach (var group in byMdos)
        {
            var mscnVert = pm4.SceneNodes[(int)group.Key];
            var surfCk24s = group.Select(s => s.CK24).Distinct().ToList();
            Console.WriteLine($"  MdosIndex={group.Key}: {group.Count()} surfaces, CK24s: {string.Join(",", surfCk24s.Take(3).Select(c => $"0x{c:X6}"))}");
            Console.WriteLine($"    → MSCN vertex: ({mscnVert.X:F1}, {mscnVert.Y:F1}, {mscnVert.Z:F1})");
        }
        
        // NEW: Investigate MdosIndex as range start into MSCN
        Console.WriteLine("\n=== MdosIndex STRIDE PATTERN ANALYSIS ===");
        
        // Sort by MdosIndex and look for stride pattern
        var sortedByMdos = pm4.Surfaces
            .Where(s => s.MdosIndex < mscnCount)
            .OrderBy(s => s.MdosIndex)
            .ToList();
        
        if (sortedByMdos.Count > 10)
        {
            Console.WriteLine("First 10 MdosIndex values (sorted):");
            var strides = new List<int>();
            for (int i = 0; i < 10; i++)
            {
                var surf = sortedByMdos[i];
                int stride = i > 0 ? (int)(surf.MdosIndex - sortedByMdos[i-1].MdosIndex) : 0;
                strides.Add(stride);
                Console.WriteLine($"  [{i}] MdosIndex={surf.MdosIndex}, CK24=0x{surf.CK24:X6}, Stride={stride}");
            }
            
            // Compute common stride
            var nonZeroStrides = strides.Where(s => s > 0).ToList();
            if (nonZeroStrides.Count > 0)
            {
                var avgStride = nonZeroStrides.Average();
                var commonStride = nonZeroStrides.GroupBy(s => s).OrderByDescending(g => g.Count()).First().Key;
                Console.WriteLine($"\nAverage stride: {avgStride:F1}");
                Console.WriteLine($"Most common stride: {commonStride}");
            }
        }
        
        // Check if MdosIndex is a boundary marker - does it mark where MSCN vertices for an object START?
        Console.WriteLine("\n=== MdosIndex as MSCN RANGE START ===");
        var surfacesByMdos = pm4.Surfaces
            .Where(s => s.MdosIndex < mscnCount)
            .GroupBy(s => s.MdosIndex)
            .OrderBy(g => g.Key)
            .ToList();
        
        if (surfacesByMdos.Count > 3)
        {
            Console.WriteLine("Testing if MdosIndex marks start of MSCN range for each surface:");
            for (int i = 0; i < Math.Min(5, surfacesByMdos.Count - 1); i++)
            {
                var current = surfacesByMdos[i];
                var next = surfacesByMdos[i + 1];
                
                int rangeStart = (int)current.Key;
                int rangeEnd = (int)next.Key;
                int rangeSize = rangeEnd - rangeStart;
                
                // Get MSCN vertices in this range
                var mscnInRange = new List<Vector3>();
                for (int m = rangeStart; m < rangeEnd && m < mscnCount; m++)
                    mscnInRange.Add(pm4.SceneNodes[m]);
                
                // Get surface geometry bounds
                var surfVerts = new List<Vector3>();
                foreach (var surf in current)
                    surfVerts.AddRange(GetSurfaceVertices(pm4, surf));
                
                if (surfVerts.Count > 0 && mscnInRange.Count > 0)
                {
                    var surfBounds = ComputeBounds(surfVerts);
                    var mscnBounds = ComputeBounds(mscnInRange);
                    
                    // Check overlap
                    bool overlaps = 
                        surfBounds.min.X <= mscnBounds.max.X && surfBounds.max.X >= mscnBounds.min.X &&
                        surfBounds.min.Y <= mscnBounds.max.Y && surfBounds.max.Y >= mscnBounds.min.Y;
                    
                    Console.WriteLine($"  MdosIndex {rangeStart}-{rangeEnd} ({rangeSize} MSCN verts):");
                    Console.WriteLine($"    MSCN bounds: ({mscnBounds.min.X:F0},{mscnBounds.min.Y:F0}) to ({mscnBounds.max.X:F0},{mscnBounds.max.Y:F0})");
                    Console.WriteLine($"    Surf bounds: ({surfBounds.min.X:F0},{surfBounds.min.Y:F0}) to ({surfBounds.max.X:F0},{surfBounds.max.Y:F0})");
                    Console.WriteLine($"    Overlaps: {overlaps}");
                }
            }
        }
        
        return 0;
    }
    
    /// <summary>
    /// Analyze whether we can isolate MSCN per object and pair with MSVT geometry.
    /// Goal: Create "complete" objects with both mesh vertices AND scene nodes.
    /// </summary>
    static int RunObjectCompleteAnalysis(string pm4Path, string? outputDir)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        outputDir ??= Path.Combine(Path.GetTempPath(), "pm4_complete_objects");
        Directory.CreateDirectory(outputDir);
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== COMPLETE OBJECT ANALYSIS: {baseName} ===\n");
        Console.WriteLine($"MSUR surfaces: {pm4.Surfaces.Count}");
        Console.WriteLine($"MSVT vertices: {pm4.MeshVertices.Count}");
        Console.WriteLine($"MSCN vertices: {pm4.SceneNodes.Count}");
        
        // Group surfaces by CK24
        var surfacesByCk24 = pm4.Surfaces
            .Where(s => s.CK24 != 0) // Skip nav mesh
            .GroupBy(s => s.CK24)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        Console.WriteLine($"Unique CK24s (non-zero): {surfacesByCk24.Count}\n");
        
        // For each CK24, gather:
        // 1. MSVT vertices (via MSUR→MSVI→MSVT)
        // 2. MSCN vertices (via MSUR.MdosIndex→MSCN)
        Console.WriteLine("=== MSCN ISOLATION PER CK24 ===");
        Console.WriteLine("Testing if MdosIndex groups MSCN by object...\n");
        
        int exported = 0;
        var results = new List<(uint ck24, int msvtCount, int mscnCount, bool overlaps)>();
        
        foreach (var (ck24, surfaces) in surfacesByCk24.OrderByDescending(kv => kv.Value.Count).Take(10))
        {
            // Get all MSVT vertices for this CK24
            var msvtVerts = new HashSet<Vector3>();
            foreach (var surf in surfaces)
            {
                var verts = GetSurfaceVertices(pm4, surf);
                foreach (var v in verts) msvtVerts.Add(v);
            }
            
            // Get all unique MdosIndex values for this CK24
            var mdosIndices = surfaces
                .Select(s => s.MdosIndex)
                .Where(m => m < pm4.SceneNodes.Count)
                .Distinct()
                .ToList();
            
            // Get MSCN vertices referenced by these surfaces
            var mscnVerts = new HashSet<Vector3>();
            foreach (var idx in mdosIndices)
            {
                mscnVerts.Add(pm4.SceneNodes[(int)idx]);
            }
            
            // Check spatial overlap
            bool overlaps = false;
            if (msvtVerts.Count > 0 && mscnVerts.Count > 0)
            {
                var msvtBounds = ComputeBounds(msvtVerts.ToList());
                var mscnBounds = ComputeBounds(mscnVerts.ToList());
                
                overlaps = 
                    msvtBounds.min.X <= mscnBounds.max.X + 10 && msvtBounds.max.X >= mscnBounds.min.X - 10 &&
                    msvtBounds.min.Y <= mscnBounds.max.Y + 10 && msvtBounds.max.Y >= mscnBounds.min.Y - 10;
                
                Console.WriteLine($"CK24 0x{ck24:X6}:");
                Console.WriteLine($"  Surfaces: {surfaces.Count}");
                Console.WriteLine($"  MSVT vertices: {msvtVerts.Count}");
                Console.WriteLine($"  MSCN vertices (via MdosIndex): {mscnVerts.Count}");
                Console.WriteLine($"  MdosIndex range: {mdosIndices.Min()}-{mdosIndices.Max()}");
                Console.WriteLine($"  MSVT bounds: ({msvtBounds.min.X:F0},{msvtBounds.min.Y:F0}) to ({msvtBounds.max.X:F0},{msvtBounds.max.Y:F0})");
                Console.WriteLine($"  MSCN bounds: ({mscnBounds.min.X:F0},{mscnBounds.min.Y:F0}) to ({mscnBounds.max.X:F0},{mscnBounds.max.Y:F0})");
                Console.WriteLine($"  Spatial overlap: {(overlaps ? "✓ YES" : "✗ NO")}\n");
            }
            
            results.Add((ck24, msvtVerts.Count, mscnVerts.Count, overlaps));
            
            // Export complete object (MSVT + MSCN)
            if (msvtVerts.Count > 0 && mscnVerts.Count > 0 && exported < 5)
            {
                var objPath = Path.Combine(outputDir, $"{baseName}_CK24_{ck24:X6}_complete.obj");
                using (var sw = new StreamWriter(objPath))
                {
                    sw.WriteLine($"# Complete object: CK24 0x{ck24:X6}");
                    sw.WriteLine($"# MSVT vertices: {msvtVerts.Count}");
                    sw.WriteLine($"# MSCN vertices: {mscnVerts.Count}");
                    sw.WriteLine();
                    
                    // Write MSVT vertices
                    sw.WriteLine("# === MSVT MESH VERTICES ===");
                    foreach (var v in msvtVerts)
                        sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                    
                    int msvtEnd = msvtVerts.Count;
                    
                    // Write MSCN vertices (as different color/group)
                    sw.WriteLine();
                    sw.WriteLine("# === MSCN SCENE NODES ===");
                    sw.WriteLine("g mscn_points");
                    foreach (var v in mscnVerts)
                        sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                    
                    // Create point cloud for MSCN
                    for (int i = msvtEnd + 1; i <= msvtEnd + mscnVerts.Count; i++)
                        sw.WriteLine($"p {i}");
                }
                Console.WriteLine($"  Exported: {objPath}");
                exported++;
            }
        }
        
        // Summary
        Console.WriteLine("\n=== SUMMARY ===");
        int overlapping = results.Count(r => r.overlaps);
        Console.WriteLine($"Objects with spatial MSVT/MSCN overlap: {overlapping}/{results.Count}");
        Console.WriteLine($"Total MSCN isolated via MdosIndex: {results.Sum(r => r.mscnCount)}");
        
        // Now try alternative: Use spatial proximity to assign MSCN to CK24
        Console.WriteLine("\n=== ALTERNATIVE: SPATIAL ASSIGNMENT ===");
        Console.WriteLine("Assigning each MSCN vertex to nearest CK24 bounding box...\n");
        
        // Build CK24 bounding boxes
        var ck24Bounds = new Dictionary<uint, (Vector3 min, Vector3 max)>();
        foreach (var (ck24, surfaces) in surfacesByCk24)
        {
            var allVerts = new List<Vector3>();
            foreach (var surf in surfaces)
                allVerts.AddRange(GetSurfaceVertices(pm4, surf));
            if (allVerts.Count > 0)
                ck24Bounds[ck24] = ComputeBounds(allVerts);
        }
        
        // Assign each MSCN to a CK24 based on containment
        var mscnToCk24 = new Dictionary<uint, List<int>>(); // CK24 → list of MSCN indices
        int unassigned = 0;
        
        for (int i = 0; i < pm4.SceneNodes.Count; i++)
        {
            var mscn = pm4.SceneNodes[i];
            uint matched = 0;
            
            foreach (var (ck24, bounds) in ck24Bounds)
            {
                // Expand bounds slightly for tolerance
                if (mscn.X >= bounds.min.X - 5 && mscn.X <= bounds.max.X + 5 &&
                    mscn.Y >= bounds.min.Y - 5 && mscn.Y <= bounds.max.Y + 5 &&
                    mscn.Z >= bounds.min.Z - 10 && mscn.Z <= bounds.max.Z + 10)
                {
                    matched = ck24;
                    break;
                }
            }
            
            if (matched != 0)
            {
                if (!mscnToCk24.ContainsKey(matched))
                    mscnToCk24[matched] = new List<int>();
                mscnToCk24[matched].Add(i);
            }
            else
            {
                unassigned++;
            }
        }
        
        Console.WriteLine($"MSCN assigned to CK24 objects: {pm4.SceneNodes.Count - unassigned}");
        Console.WriteLine($"MSCN unassigned (terrain/nav?): {unassigned}");
        Console.WriteLine($"CK24s with MSCN data: {mscnToCk24.Count}");
        
        Console.WriteLine("\nTop CK24s by MSCN count:");
        foreach (var (ck24, indices) in mscnToCk24.OrderByDescending(kv => kv.Value.Count).Take(5))
        {
            Console.WriteLine($"  CK24 0x{ck24:X6}: {indices.Count} MSCN vertices");
        }
        
        Console.WriteLine($"\nOutput directory: {outputDir}");
        return 0;
    }
    
    /// <summary>
    /// Analyze PM4 as a node graph structure.
    /// Hypothesis: MSLK entries are edges connecting navigation nodes.
    /// GroupObjectId = edge ID, TypeFlags = edge type, surfaces linked via RefIndex.
    /// </summary>
    static int RunNodeGraphAnalysis(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== PM4 NODE GRAPH ANALYSIS: {baseName} ===\n");
        Console.WriteLine($"MSLK entries (graph edges): {pm4.LinkEntries.Count}");
        Console.WriteLine($"MSUR surfaces (face data): {pm4.Surfaces.Count}");
        Console.WriteLine($"MSPV path vertices: {pm4.PathVertices.Count}");
        Console.WriteLine($"MSCN scene nodes: {pm4.SceneNodes.Count}");
        
        // === MSLK as edges analysis ===
        Console.WriteLine("\n=== GRAPH STRUCTURE ANALYSIS ===");
        
        // Group MSLK by GroupObjectId (these are the "edges")
        var edgeGroups = pm4.LinkEntries
            .GroupBy(e => e.GroupObjectId)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        Console.WriteLine($"Unique edge IDs (GroupObjectId): {edgeGroups.Count}");
        
        // Analyze edge cardinality (how many MSLK entries per edge?)
        var edgeSizes = edgeGroups.Values.Select(v => v.Count).ToList();
        Console.WriteLine($"Edge cardinality distribution:");
        foreach (var group in edgeSizes.GroupBy(s => s).OrderBy(g => g.Key))
        {
            Console.WriteLine($"  {group.Key} entries/edge: {group.Count()} edges ({100.0*group.Count()/edgeGroups.Count:F1}%)");
        }
        
        // === TypeFlags as edge types ===
        Console.WriteLine("\n=== EDGE TYPES (TypeFlags) ===");
        var typeDistribution = pm4.LinkEntries
            .GroupBy(e => e.TypeFlags)
            .OrderByDescending(g => g.Count());
        
        foreach (var group in typeDistribution)
        {
            int withGeom = group.Count(e => e.MspiIndexCount > 0);
            int withoutGeom = group.Count(e => e.MspiIndexCount == 0);
            Console.WriteLine($"  Type {group.Key,2}: {group.Count(),6} entries ({100.0*group.Count()/pm4.LinkEntries.Count:F1}%) - Geometry: {withGeom} yes, {withoutGeom} no");
        }
        
        // === Subtype as floor levels ===
        Console.WriteLine("\n=== SUBTYPE (FLOOR LEVELS) ===");
        var subtypeDistribution = pm4.LinkEntries
            .GroupBy(e => e.Subtype)
            .OrderBy(g => g.Key);
        
        foreach (var group in subtypeDistribution)
        {
            Console.WriteLine($"  Subtype {group.Key,2}: {group.Count(),6} entries");
        }
        
        // === Try to build actual graph connectivity ===
        Console.WriteLine("\n=== GRAPH CONNECTIVITY ===");
        Console.WriteLine("Analyzing how edges connect via shared surfaces (RefIndex → MSUR)...\n");
        
        // Build: MSUR index → list of edge IDs that reference it
        var surfaceToEdges = new Dictionary<int, List<uint>>();
        foreach (var (edgeId, entries) in edgeGroups)
        {
            foreach (var entry in entries)
            {
                if (entry.RefIndex < pm4.Surfaces.Count)
                {
                    int idx = entry.RefIndex;
                    if (!surfaceToEdges.ContainsKey(idx))
                        surfaceToEdges[idx] = new List<uint>();
                    if (!surfaceToEdges[idx].Contains(edgeId))
                        surfaceToEdges[idx].Add(edgeId);
                }
            }
        }
        
        // Count how many surfaces are shared by multiple edges
        var sharedSurfaces = surfaceToEdges.Where(kv => kv.Value.Count > 1).ToList();
        Console.WriteLine($"Total surfaces referenced: {surfaceToEdges.Count}");
        Console.WriteLine($"Surfaces shared by 2+ edges: {sharedSurfaces.Count}");
        
        // Build edge adjacency graph
        var edgeAdjacency = new Dictionary<uint, HashSet<uint>>();
        foreach (var (surfIdx, edgeIds) in sharedSurfaces)
        {
            // All edges sharing this surface are connected
            foreach (var e1 in edgeIds)
            {
                if (!edgeAdjacency.ContainsKey(e1))
                    edgeAdjacency[e1] = new HashSet<uint>();
                foreach (var e2 in edgeIds)
                {
                    if (e1 != e2)
                        edgeAdjacency[e1].Add(e2);
                }
            }
        }
        
        Console.WriteLine($"Edges with connections: {edgeAdjacency.Count}");
        
        // Find connected components (subgraphs = isolated objects?)
        Console.WriteLine("\n=== CONNECTED COMPONENTS (SUBGRAPHS) ===");
        var visited = new HashSet<uint>();
        var components = new List<HashSet<uint>>();
        
        foreach (var edgeId in edgeGroups.Keys)
        {
            if (!visited.Contains(edgeId))
            {
                var component = new HashSet<uint>();
                var queue = new Queue<uint>();
                queue.Enqueue(edgeId);
                
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (visited.Contains(current)) continue;
                    visited.Add(current);
                    component.Add(current);
                    
                    if (edgeAdjacency.TryGetValue(current, out var neighbors))
                    {
                        foreach (var n in neighbors)
                            if (!visited.Contains(n))
                                queue.Enqueue(n);
                    }
                }
                
                if (component.Count > 0)
                    components.Add(component);
            }
        }
        
        Console.WriteLine($"Total connected components: {components.Count}");
        var sortedComponents = components.OrderByDescending(c => c.Count).ToList();
        
        Console.WriteLine("\nLargest components:");
        for (int i = 0; i < Math.Min(10, sortedComponents.Count); i++)
        {
            var comp = sortedComponents[i];
            
            // Get CK24s in this component
            var ck24s = new HashSet<uint>();
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.RefIndex < pm4.Surfaces.Count)
                        ck24s.Add(pm4.Surfaces[entry.RefIndex].CK24);
                }
            }
            
            Console.WriteLine($"  Component {i+1}: {comp.Count} edges, CK24s: {string.Join(",", ck24s.Take(3).Select(c => $"0x{c:X6}"))}");
        }
        
        // === Analyze MSLK MSPI geometry as actual path connections ===
        Console.WriteLine("\n=== MSPI PATH GEOMETRY ===");
        Console.WriteLine("Analyzing edges with geometry (MspiIndexCount > 0)...\n");
        
        var edgesWithGeom = pm4.LinkEntries.Where(e => e.MspiIndexCount > 0).ToList();
        Console.WriteLine($"Edges with path geometry: {edgesWithGeom.Count}/{pm4.LinkEntries.Count}");
        
        // Look at MSPV (path vertices) referenced by MSPI
        var mspiMax = edgesWithGeom.Max(e => e.MspiFirstIndex + e.MspiIndexCount);
        Console.WriteLine($"MSPI index range used: 0 to {mspiMax}");
        Console.WriteLine($"MSPI total count: {pm4.PathIndices.Count}");
        Console.WriteLine($"MSPV total count: {pm4.PathVertices.Count}");
        
        // Sample a few path geometries
        Console.WriteLine("\nSample path geometries:");
        foreach (var entry in edgesWithGeom.Take(5))
        {
            Console.WriteLine($"  Edge GroupId={entry.GroupObjectId}, Type={entry.TypeFlags}, Subtype={entry.Subtype}:");
            Console.WriteLine($"    MSPI[{entry.MspiFirstIndex}..{entry.MspiFirstIndex + entry.MspiIndexCount - 1}] ({entry.MspiIndexCount} indices)");
            
            // Get actual vertices
            if (entry.MspiFirstIndex >= 0 && entry.MspiFirstIndex + entry.MspiIndexCount <= pm4.PathIndices.Count)
            {
                var pathVerts = new List<Vector3>();
                for (int i = 0; i < entry.MspiIndexCount; i++)
                {
                    int mspiIdx = entry.MspiFirstIndex + i;
                    if (mspiIdx < pm4.PathIndices.Count)
                    {
                        int mspvIdx = (int)pm4.PathIndices[mspiIdx];
                        if (mspvIdx < pm4.PathVertices.Count)
                            pathVerts.Add(pm4.PathVertices[mspvIdx]);
                    }
                }
                
                if (pathVerts.Count > 0)
                {
                    var bounds = ComputeBounds(pathVerts);
                    Console.WriteLine($"    Path vertices: {pathVerts.Count}");
                    Console.WriteLine($"    Bounds: ({bounds.min.X:F0},{bounds.min.Y:F0},{bounds.min.Z:F0}) to ({bounds.max.X:F0},{bounds.max.Y:F0},{bounds.max.Z:F0})");
                }
            }
        }
        
        // === Isolate polygons via MSPI ===
        Console.WriteLine("\n=== POLYGON ISOLATION VIA MSPI ===");
        Console.WriteLine("Each MSLK entry with geometry defines a polygon via MSPI→MSPV...\n");
        
        int polygonCount = edgesWithGeom.Count;
        var polygonSizes = edgesWithGeom.Select(e => e.MspiIndexCount).GroupBy(s => s);
        Console.WriteLine($"Total polygons: {polygonCount}");
        Console.WriteLine("Polygon sizes (vertex count):");
        foreach (var group in polygonSizes.OrderBy(g => g.Key))
        {
            Console.WriteLine($"  {group.Key} vertices: {group.Count()} polygons");
        }
        
        // === Component to CK24 correlation ===
        Console.WriteLine("\n=== COMPONENT ↔ CK24 CORRELATION ===");
        Console.WriteLine("Testing if connected components match individual CK24 objects...\n");
        
        // For each component, count how many CK24s it contains
        var singleCk24Components = 0;
        var multiCk24Components = 0;
        var componentCk24Map = new List<(int componentIdx, HashSet<uint> ck24s, int edgeCount)>();
        
        for (int i = 0; i < sortedComponents.Count; i++)
        {
            var comp = sortedComponents[i];
            var ck24s = new HashSet<uint>();
            
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.RefIndex < pm4.Surfaces.Count)
                        ck24s.Add(pm4.Surfaces[entry.RefIndex].CK24);
                }
            }
            
            componentCk24Map.Add((i + 1, ck24s, comp.Count));
            
            if (ck24s.Count == 1)
                singleCk24Components++;
            else
                multiCk24Components++;
        }
        
        Console.WriteLine($"Components with SINGLE CK24: {singleCk24Components} ({100.0*singleCk24Components/components.Count:F1}%)");
        Console.WriteLine($"Components with MULTIPLE CK24s: {multiCk24Components} ({100.0*multiCk24Components/components.Count:F1}%)");
        
        // Show some multi-CK24 components
        Console.WriteLine("\nComponents spanning multiple CK24s:");
        foreach (var (idx, ck24s, edgeCount) in componentCk24Map.Where(c => c.ck24s.Count > 1).Take(5))
        {
            Console.WriteLine($"  Component {idx}: {edgeCount} edges, {ck24s.Count} CK24s: {string.Join(",", ck24s.Take(5).Select(c => $"0x{c:X6}"))}");
        }
        
        // === Analyze Type 1 vs other types in components ===
        Console.WriteLine("\n=== TYPE DISTRIBUTION IN COMPONENTS ===");
        Console.WriteLine("Type 1 = anchor nodes (no geometry), Types 2/4/10/12 = connection nodes...\n");
        
        for (int i = 0; i < Math.Min(5, sortedComponents.Count); i++)
        {
            var comp = sortedComponents[i];
            int type1Count = 0, otherTypeCount = 0;
            
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.TypeFlags == 1)
                        type1Count++;
                    else
                        otherTypeCount++;
                }
            }
            
            Console.WriteLine($"  Component {i+1}: Type1={type1Count}, Other={otherTypeCount} (ratio: {(otherTypeCount > 0 ? 100.0*type1Count/otherTypeCount : 0):F0}%)");
        }
        
        // === Export navigation mesh for one component ===
        Console.WriteLine("\n=== EXPORTING COMPONENT NAV MESH ===");
        var exportComp = sortedComponents.FirstOrDefault(c => c.Count > 50 && c.Count < 200);
        if (exportComp != null && exportComp.Count > 0)
        {
            var outputDir = Path.Combine(Path.GetTempPath(), "pm4_nav_components");
            Directory.CreateDirectory(outputDir);
            
            // Collect all path geometry from this component
            var navVerts = new List<Vector3>();
            var navQuads = new List<(int a, int b, int c, int d)>();
            var vertexMap = new Dictionary<Vector3, int>();
            
            foreach (var edgeId in exportComp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.MspiIndexCount == 4 && entry.MspiFirstIndex >= 0)
                    {
                        var quadVerts = new List<int>();
                        for (int j = 0; j < 4; j++)
                        {
                            int mspiIdx = entry.MspiFirstIndex + j;
                            if (mspiIdx < pm4.PathIndices.Count)
                            {
                                int mspvIdx = (int)pm4.PathIndices[mspiIdx];
                                if (mspvIdx < pm4.PathVertices.Count)
                                {
                                    var v = pm4.PathVertices[mspvIdx];
                                    if (!vertexMap.ContainsKey(v))
                                    {
                                        vertexMap[v] = navVerts.Count + 1; // OBJ is 1-indexed
                                        navVerts.Add(v);
                                    }
                                    quadVerts.Add(vertexMap[v]);
                                }
                            }
                        }
                        if (quadVerts.Count == 4)
                            navQuads.Add((quadVerts[0], quadVerts[1], quadVerts[2], quadVerts[3]));
                    }
                }
            }
            
            // Get CK24s for this component
            var compCk24s = new HashSet<uint>();
            foreach (var edgeId in exportComp)
                foreach (var entry in edgeGroups[edgeId])
                    if (entry.RefIndex < pm4.Surfaces.Count)
                        compCk24s.Add(pm4.Surfaces[entry.RefIndex].CK24);
            
            var ck24Str = string.Join("_", compCk24s.Take(2).Select(c => $"{c:X6}"));
            var objPath = Path.Combine(outputDir, $"{baseName}_component_{exportComp.Count}edges_{ck24Str}.obj");
            
            using (var sw = new StreamWriter(objPath))
            {
                sw.WriteLine($"# Navigation mesh component");
                sw.WriteLine($"# Edges: {exportComp.Count}");
                sw.WriteLine($"# Vertices: {navVerts.Count}");
                sw.WriteLine($"# Quads: {navQuads.Count}");
                sw.WriteLine($"# CK24s: {string.Join(",", compCk24s.Select(c => $"0x{c:X6}"))}");
                sw.WriteLine();
                
                foreach (var v in navVerts)
                    sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                
                sw.WriteLine();
                foreach (var (a, b, c, d) in navQuads)
                    sw.WriteLine($"f {a} {b} {c} {d}");
            }
            
            Console.WriteLine($"Exported {navQuads.Count} quads from component with {exportComp.Count} edges");
            Console.WriteLine($"Output: {objPath}");
        }
        
        // === MSCN relationship to components ===
        Console.WriteLine("\n=== MSCN ↔ COMPONENT RELATIONSHIP ===");
        Console.WriteLine("Testing how MSCN vertices relate to navigation graph components...\n");
        
        // For each component, gather MSCN via MdosIndex from the surfaces
        var componentMscn = new Dictionary<int, HashSet<int>>(); // componentIdx → MSCN indices
        
        for (int i = 0; i < sortedComponents.Count; i++)
        {
            var comp = sortedComponents[i];
            var mscnIndices = new HashSet<int>();
            
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.RefIndex < pm4.Surfaces.Count)
                    {
                        var surf = pm4.Surfaces[entry.RefIndex];
                        if (surf.MdosIndex < pm4.SceneNodes.Count)
                            mscnIndices.Add((int)surf.MdosIndex);
                    }
                }
            }
            
            if (mscnIndices.Count > 0)
                componentMscn[i] = mscnIndices;
        }
        
        int componentsWithMscn = componentMscn.Count;
        int totalMscnInComponents = componentMscn.Values.Sum(s => s.Count);
        
        Console.WriteLine($"Components with MSCN data: {componentsWithMscn}/{components.Count} ({100.0*componentsWithMscn/components.Count:F1}%)");
        Console.WriteLine($"Total MSCN referenced by components: {totalMscnInComponents}");
        Console.WriteLine($"MSCN coverage: {100.0*totalMscnInComponents/pm4.SceneNodes.Count:F1}% of {pm4.SceneNodes.Count}");
        
        // Check if nav mesh quads spatially overlap with their component's MSCN
        Console.WriteLine("\n=== NAV MESH ↔ MSCN SPATIAL CORRELATION ===");
        Console.WriteLine("Testing if component's nav quads overlap spatially with component's MSCN...\n");
        
        int overlappingComponents = 0;
        int nonOverlappingComponents = 0;
        
        for (int i = 0; i < Math.Min(20, sortedComponents.Count); i++)
        {
            var comp = sortedComponents[i];
            if (!componentMscn.ContainsKey(i)) continue;
            
            // Get nav mesh bounds for this component
            var navVerts = new List<Vector3>();
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.MspiIndexCount > 0 && entry.MspiFirstIndex >= 0)
                    {
                        for (int j = 0; j < entry.MspiIndexCount; j++)
                        {
                            int mspiIdx = entry.MspiFirstIndex + j;
                            if (mspiIdx < pm4.PathIndices.Count)
                            {
                                int mspvIdx = (int)pm4.PathIndices[mspiIdx];
                                if (mspvIdx < pm4.PathVertices.Count)
                                    navVerts.Add(pm4.PathVertices[mspvIdx]);
                            }
                        }
                    }
                }
            }
            
            // Get MSCN bounds for this component
            var mscnVerts = componentMscn[i].Select(idx => pm4.SceneNodes[idx]).ToList();
            
            if (navVerts.Count > 0 && mscnVerts.Count > 0)
            {
                var navBounds = ComputeBounds(navVerts);
                var mscnBounds = ComputeBounds(mscnVerts);
                
                bool overlaps = 
                    navBounds.min.X <= mscnBounds.max.X + 20 && navBounds.max.X >= mscnBounds.min.X - 20 &&
                    navBounds.min.Y <= mscnBounds.max.Y + 20 && navBounds.max.Y >= mscnBounds.min.Y - 20;
                
                if (overlaps) overlappingComponents++;
                else nonOverlappingComponents++;
                
                if (i < 5)
                {
                    Console.WriteLine($"  Component {i+1}: {comp.Count} edges, {mscnVerts.Count} MSCN, {navVerts.Count} nav verts");
                    Console.WriteLine($"    Nav bounds: ({navBounds.min.X:F0},{navBounds.min.Y:F0}) to ({navBounds.max.X:F0},{navBounds.max.Y:F0})");
                    Console.WriteLine($"    MSCN bounds: ({mscnBounds.min.X:F0},{mscnBounds.min.Y:F0}) to ({mscnBounds.max.X:F0},{mscnBounds.max.Y:F0})");
                    Console.WriteLine($"    Overlaps: {(overlaps ? "✓" : "✗")}");
                }
            }
        }
        
        Console.WriteLine($"\nSpatial overlap (first 20 components): {overlappingComponents} overlapping, {nonOverlappingComponents} non-overlapping");
        
        // === Export complete navigable object with Nav + MSCN + MSVT ===
        Console.WriteLine("\n=== EXPORTING COMPLETE NAVIGABLE OBJECT ===");
        var exportIdx = sortedComponents.FindIndex(c => c.Count > 80 && c.Count < 150);
        if (exportIdx >= 0 && componentMscn.ContainsKey(exportIdx))
        {
            var outputDir = Path.Combine(Path.GetTempPath(), "pm4_complete_nav_objects");
            Directory.CreateDirectory(outputDir);
            
            var comp = sortedComponents[exportIdx];
            var compMscnIndices = componentMscn[exportIdx];
            
            // Collect all data for this component
            var allVerts = new List<Vector3>();
            var navQuads = new List<(int, int, int, int)>();
            var msvtVerts = new List<Vector3>();
            var mscnPoints = new List<Vector3>();
            var vertexMap = new Dictionary<Vector3, int>();
            
            // Nav mesh vertices (from MSPV via MSPI)
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.MspiIndexCount == 4 && entry.MspiFirstIndex >= 0)
                    {
                        var quadVerts = new List<int>();
                        for (int j = 0; j < 4; j++)
                        {
                            int mspiIdx = entry.MspiFirstIndex + j;
                            if (mspiIdx < pm4.PathIndices.Count)
                            {
                                int mspvIdx = (int)pm4.PathIndices[mspiIdx];
                                if (mspvIdx < pm4.PathVertices.Count)
                                {
                                    var v = pm4.PathVertices[mspvIdx];
                                    if (!vertexMap.ContainsKey(v))
                                    {
                                        vertexMap[v] = allVerts.Count + 1;
                                        allVerts.Add(v);
                                    }
                                    quadVerts.Add(vertexMap[v]);
                                }
                            }
                        }
                        if (quadVerts.Count == 4)
                            navQuads.Add((quadVerts[0], quadVerts[1], quadVerts[2], quadVerts[3]));
                    }
                }
            }
            int navVertEnd = allVerts.Count;
            
            // MSVT vertices (object geometry via MSUR→MSVI→MSVT)
            var compSurfaces = new HashSet<int>();
            foreach (var edgeId in comp)
                foreach (var entry in edgeGroups[edgeId])
                    if (entry.RefIndex < pm4.Surfaces.Count)
                        compSurfaces.Add(entry.RefIndex);
            
            foreach (var surfIdx in compSurfaces)
            {
                var surf = pm4.Surfaces[surfIdx];
                var verts = GetSurfaceVertices(pm4, surf);
                foreach (var v in verts)
                {
                    if (!vertexMap.ContainsKey(v))
                    {
                        vertexMap[v] = allVerts.Count + 1;
                        allVerts.Add(v);
                        msvtVerts.Add(v);
                    }
                }
            }
            int msvtVertEnd = allVerts.Count;
            
            // MSCN points
            foreach (var mscnIdx in compMscnIndices)
            {
                var v = pm4.SceneNodes[mscnIdx];
                if (!vertexMap.ContainsKey(v))
                {
                    vertexMap[v] = allVerts.Count + 1;
                    allVerts.Add(v);
                }
                mscnPoints.Add(v);
            }
            
            // Get CK24s
            var compCk24s = compSurfaces.Select(i => pm4.Surfaces[i].CK24).Distinct().ToList();
            var ck24Str = string.Join("_", compCk24s.Take(2).Select(c => $"{c:X6}"));
            var objPath = Path.Combine(outputDir, $"{baseName}_complete_{comp.Count}edges_{ck24Str}.obj");
            
            using (var sw = new StreamWriter(objPath))
            {
                sw.WriteLine($"# Complete Navigable Object");
                sw.WriteLine($"# Component: {comp.Count} edges");
                sw.WriteLine($"# Nav vertices: {navVertEnd}");
                sw.WriteLine($"# Nav quads: {navQuads.Count}");
                sw.WriteLine($"# MSVT vertices: {msvtVerts.Count}");
                sw.WriteLine($"# MSCN points: {mscnPoints.Count}");
                sw.WriteLine($"# CK24s: {string.Join(",", compCk24s.Select(c => $"0x{c:X6}"))}");
                sw.WriteLine();
                
                // Write all vertices
                sw.WriteLine("# All vertices (nav + msvt + mscn)");
                foreach (var v in allVerts)
                    sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                
                // Nav mesh faces
                sw.WriteLine();
                sw.WriteLine("g navigation_mesh");
                foreach (var (a, b, c, d) in navQuads)
                    sw.WriteLine($"f {a} {b} {c} {d}");
                
                // MSCN as point cloud
                sw.WriteLine();
                sw.WriteLine("g mscn_collision_points");
                foreach (var v in mscnPoints)
                {
                    if (vertexMap.TryGetValue(v, out var idx))
                        sw.WriteLine($"p {idx}");
                }
            }
            
            Console.WriteLine($"Exported complete navigable object:");
            Console.WriteLine($"  Nav mesh: {navQuads.Count} quads ({navVertEnd} vertices)");
            Console.WriteLine($"  MSVT geometry: {msvtVerts.Count} vertices");
            Console.WriteLine($"  MSCN collision: {mscnPoints.Count} points");
            Console.WriteLine($"  Output: {objPath}");
        }
        
        return 0;
    }
    
    /// <summary>
    /// Analyze how MPRL (object placements) relate to the navigation graph.
    /// MPRL contains position/rotation for placed objects, MPRR may link back.
    /// </summary>
    static int RunMprlGraphAnalysis(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MPRL ↔ GRAPH CORRELATION: {baseName} ===\n");
        Console.WriteLine($"MPRL entries (placements): {pm4.PositionRefs.Count}");
        Console.WriteLine($"MPRR entries (references): {pm4.MprrEntries.Count}");
        Console.WriteLine($"MSLK entries (edges): {pm4.LinkEntries.Count}");
        Console.WriteLine($"MSUR surfaces: {pm4.Surfaces.Count}");
        
        // === MSLK RefIndex → MPRL analysis ===
        Console.WriteLine("\n=== MSLK RefIndex → MPRL/MSVT ===");
        
        int mprlRefs = pm4.LinkEntries.Count(e => e.RefIndex < pm4.PositionRefs.Count);
        int msvtRefs = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.PositionRefs.Count);
        
        Console.WriteLine($"RefIndex < MPRL.Count ({pm4.PositionRefs.Count}): {mprlRefs} ({100.0*mprlRefs/pm4.LinkEntries.Count:F1}%)");
        Console.WriteLine($"RefIndex >= MPRL.Count (→MSVT?): {msvtRefs} ({100.0*msvtRefs/pm4.LinkEntries.Count:F1}%)");
        
        // Wait - we confirmed RefIndex → MSUR, not MPRL! Let me check...
        Console.WriteLine("\nClarification: We CONFIRMED RefIndex → MSUR (100% valid)");
        Console.WriteLine("So RefIndex IS an MSUR index, not MPRL index.\n");
        
        // === How does MPRL relate to graph components? ===
        Console.WriteLine("=== MPRL SPATIAL CORRELATION WITH GRAPH ===");
        
        // For each MPRL, find which graph component(s) are spatially nearby
        // First build graph components (reusing logic from node-graph)
        var edgeGroups = pm4.LinkEntries
            .GroupBy(e => e.GroupObjectId)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        // Build surface-to-edge adjacency for component finding
        var surfaceToEdges = new Dictionary<int, List<uint>>();
        foreach (var (edgeId, entries) in edgeGroups)
        {
            foreach (var entry in entries)
            {
                if (entry.RefIndex < pm4.Surfaces.Count)
                {
                    int idx = entry.RefIndex;
                    if (!surfaceToEdges.ContainsKey(idx))
                        surfaceToEdges[idx] = new List<uint>();
                    if (!surfaceToEdges[idx].Contains(edgeId))
                        surfaceToEdges[idx].Add(edgeId);
                }
            }
        }
        
        // Build edge adjacency
        var edgeAdjacency = new Dictionary<uint, HashSet<uint>>();
        foreach (var (surfIdx, edgeIds) in surfaceToEdges.Where(kv => kv.Value.Count > 1))
        {
            foreach (var e1 in edgeIds)
            {
                if (!edgeAdjacency.ContainsKey(e1))
                    edgeAdjacency[e1] = new HashSet<uint>();
                foreach (var e2 in edgeIds)
                    if (e1 != e2)
                        edgeAdjacency[e1].Add(e2);
            }
        }
        
        // Find connected components
        var visited = new HashSet<uint>();
        var components = new List<HashSet<uint>>();
        
        foreach (var edgeId in edgeGroups.Keys)
        {
            if (!visited.Contains(edgeId))
            {
                var component = new HashSet<uint>();
                var queue = new Queue<uint>();
                queue.Enqueue(edgeId);
                
                while (queue.Count > 0)
                {
                    var current = queue.Dequeue();
                    if (visited.Contains(current)) continue;
                    visited.Add(current);
                    component.Add(current);
                    
                    if (edgeAdjacency.TryGetValue(current, out var neighbors))
                        foreach (var n in neighbors)
                            if (!visited.Contains(n))
                                queue.Enqueue(n);
                }
                if (component.Count > 0)
                    components.Add(component);
            }
        }
        
        Console.WriteLine($"Graph has {components.Count} components");
        
        // Build bounding boxes for each component
        var componentBounds = new Dictionary<int, (Vector3 min, Vector3 max)>();
        for (int i = 0; i < components.Count; i++)
        {
            var comp = components[i];
            var allVerts = new List<Vector3>();
            foreach (var edgeId in comp)
            {
                foreach (var entry in edgeGroups[edgeId])
                {
                    if (entry.RefIndex < pm4.Surfaces.Count)
                    {
                        var verts = GetSurfaceVertices(pm4, pm4.Surfaces[entry.RefIndex]);
                        allVerts.AddRange(verts);
                    }
                }
            }
            if (allVerts.Count > 0)
                componentBounds[i] = ComputeBounds(allVerts);
        }
        
        // For each MPRL, find which component it's inside
        Console.WriteLine("\nAssigning MPRL entries to graph components...\n");
        var mprlToComponent = new Dictionary<int, int>(); // MPRL index → component index
        int unassigned = 0;
        
        for (int m = 0; m < pm4.PositionRefs.Count; m++)
        {
            var mprl = pm4.PositionRefs[m];
            // MPRL position stored as Y, Z, X (needs swap)
            float px = mprl.PositionZ;
            float py = mprl.PositionX;
            float pz = mprl.PositionY;
            
            int matchedComponent = -1;
            foreach (var (idx, bounds) in componentBounds)
            {
                if (px >= bounds.min.X - 10 && px <= bounds.max.X + 10 &&
                    py >= bounds.min.Y - 10 && py <= bounds.max.Y + 10 &&
                    pz >= bounds.min.Z - 20 && pz <= bounds.max.Z + 20)
                {
                    matchedComponent = idx;
                    break;
                }
            }
            
            if (matchedComponent >= 0)
                mprlToComponent[m] = matchedComponent;
            else
                unassigned++;
        }
        
        Console.WriteLine($"MPRL assigned to components: {mprlToComponent.Count}");
        Console.WriteLine($"MPRL unassigned: {unassigned}");
        
        // Show components with most MPRL placements
        var componentMprlCount = mprlToComponent.GroupBy(kv => kv.Value)
            .OrderByDescending(g => g.Count())
            .Take(10);
        
        Console.WriteLine("\nComponents with most MPRL placements:");
        foreach (var grp in componentMprlCount)
        {
            var comp = components[grp.Key];
            // Get CK24s for this component
            var ck24s = new HashSet<uint>();
            foreach (var edgeId in comp)
                foreach (var entry in edgeGroups[edgeId])
                    if (entry.RefIndex < pm4.Surfaces.Count)
                        ck24s.Add(pm4.Surfaces[entry.RefIndex].CK24);
            
            Console.WriteLine($"  Component {grp.Key}: {grp.Count()} MPRL, {comp.Count} edges, CK24s: {string.Join(",", ck24s.Take(3).Select(c => $"0x{c:X6}"))}");
        }
        
        // === MPRR Analysis ===
        Console.WriteLine("\n=== MPRR STRUCTURE ANALYSIS ===");
        Console.WriteLine($"Total MPRR entries: {pm4.MprrEntries.Count}");
        
        // Non-sentinel MPRR
        var nonSentinel = pm4.MprrEntries.Where(e => e.Value1 != 0xFFFF).ToList();
        Console.WriteLine($"Non-sentinel MPRR: {nonSentinel.Count}");
        
        if (nonSentinel.Count > 0)
        {
            var v1Min = nonSentinel.Min(e => e.Value1);
            var v1Max = nonSentinel.Max(e => e.Value1);
            Console.WriteLine($"Value1 range: {v1Min} - {v1Max}");
            Console.WriteLine($"Value1 < MPRL.Count: {nonSentinel.Count(e => e.Value1 < pm4.PositionRefs.Count)}");
            Console.WriteLine($"Value1 < MSVT.Count: {nonSentinel.Count(e => e.Value1 < pm4.MeshVertices.Count)}");
            
            // Value2 patterns
            var v2Patterns = nonSentinel.GroupBy(e => e.Value2 >> 8) // High byte
                .OrderByDescending(g => g.Count())
                .Take(5);
            
            Console.WriteLine("\nValue2 high-byte patterns:");
            foreach (var p in v2Patterns)
            {
                Console.WriteLine($"  0x{p.Key:X2}xx: {p.Count()} entries");
            }
        }
        
        return 0;
    }
    
    /// <summary>
    /// Complete PM4 structure mapping - analyze ALL chunks comprehensively.
    /// </summary>
    static int RunFullStructureMap(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var data = File.ReadAllBytes(pm4Path);
        var pm4 = Pm4File.Parse(data);
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== COMPLETE PM4 STRUCTURE MAP: {baseName} ===");
        Console.WriteLine($"File size: {data.Length:N0} bytes\n");
        
        // === Parse all chunks raw ===
        var chunks = new List<(string sig, uint size, long offset, byte[] data)>();
        using (var ms = new MemoryStream(data))
        using (var br = new BinaryReader(ms))
        {
            while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
            {
                long startPos = br.BaseStream.Position;
                var sigBytes = br.ReadBytes(4);
                Array.Reverse(sigBytes);
                string sig = Encoding.ASCII.GetString(sigBytes);
                uint size = br.ReadUInt32();
                byte[] chunkData = br.ReadBytes((int)size);
                chunks.Add((sig, size, startPos, chunkData));
            }
        }
        
        Console.WriteLine("=== CHUNK INVENTORY ===");
        Console.WriteLine($"Total chunks: {chunks.Count}\n");
        
        foreach (var c in chunks.GroupBy(x => x.sig))
        {
            var totalSize = c.Sum(x => (long)x.size);
            int entrySize4 = (int)(totalSize / Math.Max(1, c.Sum(x => x.size / 4)));
            Console.WriteLine($"  {c.Key}: {c.Count()} chunk(s), {totalSize:N0} bytes total");
        }
        
        // === MSHD Header Analysis ===
        Console.WriteLine("\n=== MSHD HEADER ANALYSIS ===");
        if (pm4.Header != null)
        {
            var h = pm4.Header;
            Console.WriteLine($"  Field00 (MSUR-related?): {h.Field00}");
            Console.WriteLine($"  Field04 (count?): {h.Field04}");
            Console.WriteLine($"  Field08 (count?): {h.Field08}");
            Console.WriteLine($"  Field0C-1C: {h.Field0C}, {h.Field10}, {h.Field14}, {h.Field18}, {h.Field1C}");
            
            // Correlate with chunk counts
            Console.WriteLine("\n  Correlation check:");
            Console.WriteLine($"    Field00={h.Field00} vs MSUR.Count={pm4.Surfaces.Count} (match: {h.Field00 == pm4.Surfaces.Count})");
            Console.WriteLine($"    Field04={h.Field04} vs MPRL.Count={pm4.PositionRefs.Count} (match: {h.Field04 == pm4.PositionRefs.Count})");
            Console.WriteLine($"    Field08={h.Field08} vs MSLK.Count={pm4.LinkEntries.Count} (match: {h.Field08 == pm4.LinkEntries.Count})");
        }
        else
        {
            Console.WriteLine("  No MSHD header found!");
        }
        
        // === MPRR Deep Analysis ===
        Console.WriteLine("\n=== MPRR DEEP ANALYSIS ===");
        Console.WriteLine($"Total entries: {pm4.MprrEntries.Count}");
        
        var nonSentinel = pm4.MprrEntries.Where(e => e.Value1 != 0xFFFF).ToList();
        var sentinel = pm4.MprrEntries.Where(e => e.Value1 == 0xFFFF).ToList();
        Console.WriteLine($"Non-sentinel: {nonSentinel.Count}");
        Console.WriteLine($"Sentinel (0xFFFF): {sentinel.Count}");
        
        if (nonSentinel.Count > 0)
        {
            // Value1 analysis
            Console.WriteLine("\n  Value1 (reference index):");
            Console.WriteLine($"    Range: 0 - {nonSentinel.Max(e => e.Value1)}");
            Console.WriteLine($"    < MPRL.Count ({pm4.PositionRefs.Count}): {nonSentinel.Count(e => e.Value1 < pm4.PositionRefs.Count)}");
            Console.WriteLine($"    >= MPRL.Count (MSVT?): {nonSentinel.Count(e => e.Value1 >= pm4.PositionRefs.Count)}");
            
            // Value2 analysis - decode the pattern
            Console.WriteLine("\n  Value2 (flags/type):");
            var v2Groups = nonSentinel.GroupBy(e => e.Value2).OrderByDescending(g => g.Count()).Take(10);
            foreach (var g in v2Groups)
            {
                byte hiByte = (byte)(g.Key >> 8);
                byte loByte = (byte)(g.Key & 0xFF);
                Console.WriteLine($"    0x{g.Key:X4} (hi=0x{hiByte:X2}, lo=0x{loByte:X2}): {g.Count()} entries");
            }
            
            // Hypothesis: High byte = floor/level, Low byte = edge flags?
            Console.WriteLine("\n  Value2 interpretation hypothesis:");
            var byHiByte = nonSentinel.GroupBy(e => e.Value2 >> 8).OrderBy(g => g.Key);
            Console.WriteLine("    High byte (floor level?):");
            foreach (var g in byHiByte.Take(5))
            {
                Console.WriteLine($"      0x{g.Key:X2}: {g.Count()} entries");
            }
        }
        
        // === MSVI → MSVT Triangle Assembly ===
        Console.WriteLine("\n=== MSVI → MSVT TRIANGLE ASSEMBLY ===");
        Console.WriteLine($"MSVI indices: {pm4.MeshIndices.Count}");
        Console.WriteLine($"MSVT vertices: {pm4.MeshVertices.Count}");
        
        // Check if MSVI values are valid MSVT indices
        if (pm4.MeshIndices.Count > 0)
        {
            var maxMsvi = pm4.MeshIndices.Max();
            var minMsvi = pm4.MeshIndices.Min();
            Console.WriteLine($"  MSVI value range: {minMsvi} - {maxMsvi}");
            Console.WriteLine($"  All valid MSVT indices: {maxMsvi < pm4.MeshVertices.Count}");
            
            // How does MSUR use MSVI?
            if (pm4.Surfaces.Count > 0)
            {
                var surf = pm4.Surfaces[0];
                Console.WriteLine($"\n  Sample surface (MSUR[0]):");
                Console.WriteLine($"    MsviFirstIndex: {surf.MsviFirstIndex}");
                Console.WriteLine($"    IndexCount: {surf.IndexCount}");
                Console.WriteLine($"    Triangle count: {surf.IndexCount / 3}");
                
                // Get the actual indices and vertices
                Console.WriteLine($"    First 6 MSVI values:");
                for (int i = 0; i < Math.Min(6, (int)surf.IndexCount); i++)
                {
                    int msviIdx = (int)surf.MsviFirstIndex + i;
                    if (msviIdx < pm4.MeshIndices.Count)
                    {
                        uint msvtIdx = pm4.MeshIndices[msviIdx];
                        if (msvtIdx < pm4.MeshVertices.Count)
                        {
                            var v = pm4.MeshVertices[(int)msvtIdx];
                            Console.WriteLine($"      MSVI[{msviIdx}]={msvtIdx} → MSVT: ({v.X:F1}, {v.Y:F1}, {v.Z:F1})");
                        }
                    }
                }
            }
        }
        
        // === MSPV → MSPI Path Geometry ===
        Console.WriteLine("\n=== MSPV/MSPI PATH GEOMETRY ===");
        Console.WriteLine($"MSPI indices: {pm4.PathIndices.Count}");
        Console.WriteLine($"MSPV vertices: {pm4.PathVertices.Count}");
        
        if (pm4.PathIndices.Count > 0)
        {
            var maxMspi = pm4.PathIndices.Max();
            Console.WriteLine($"MSPI value range: 0 - {maxMspi}");
            Console.WriteLine($"All valid MSPV indices: {maxMspi < pm4.PathVertices.Count}");
        }
        
        // === Complete Structure Summary ===
        Console.WriteLine("\n=== COMPLETE PM4 STRUCTURE ===");
        Console.WriteLine(@"
PM4 = Phased Model 4 (Server-side Navigation/Collision)

HIERARCHY:
┌──────────────────────────────────────────────────────────────┐
│ MSHD: File Header                                             │
│   ├── Field00 = MSUR count                                   │
│   ├── Field04 = MPRL count                                   │
│   └── Field08 = ? (to investigate)                           │
├──────────────────────────────────────────────────────────────┤
│ NAVIGATION GRAPH (walkable surfaces)                          │
│   MSLK → Edges (27,087 unique GroupObjectId)                 │
│     ├── TypeFlags: 1=anchor, 2/4/10/12=walkable              │
│     ├── Subtype: Floor level (0-8)                           │
│     ├── RefIndex → MSUR (surface)                            │
│     ├── MSPI → MSPV (4-vertex nav quads)                     │
│     └── LinkId: Tile coords for cross-tile refs              │
├──────────────────────────────────────────────────────────────┤
│ COLLISION GEOMETRY                                            │
│   MSUR → Surface definitions                                  │
│     ├── GroupKey: 3=terrain, 18/19=WMO                       │
│     ├── CK24: Object ID (WMO/M2 unique key)                  │
│     ├── MdosIndex → MSCN (scene node reference)              │
│     └── MSVI → MSVT (triangle mesh vertices)                 │
├──────────────────────────────────────────────────────────────┤
│ OBJECT PLACEMENT                                              │
│   MPRL → Position + Rotation for placed objects               │
│   MPRR → Reference records (links to MPRL or MSVT)           │
│     ├── Value1: Index (MPRL if < MPRL.Count, else MSVT)      │
│     └── Value2: Flags (hi-byte=floor?, lo-byte=edge type?)   │
├──────────────────────────────────────────────────────────────┤
│ SCENE DATA                                                    │
│   MSCN → Scene/collision nodes (point cloud)                  │
│          (Referenced by MSUR.MdosIndex)                       │
├──────────────────────────────────────────────────────────────┤
│ DESTRUCTIBLE (Wintergrasp only)                               │
│   MDBH/MDBI/MDBF/MDOS/MDSF → Server-side building states     │
└──────────────────────────────────────────────────────────────┘

DATA FLOW:
  Object Instance → CK24 → MSUR surfaces → MSVT geometry
                                        → MSCN collision points
  Navigation      → MSLK edges → Components → Nav mesh quads
  Placement       → MPRL position/rotation → MPRR links
");
        
        // === Cross-reference verification ===
        Console.WriteLine("=== CROSS-REFERENCE VERIFICATION ===");
        
        // MSLK → MSUR
        int validMslkToMsur = pm4.LinkEntries.Count(e => e.RefIndex < pm4.Surfaces.Count);
        Console.WriteLine($"MSLK.RefIndex → MSUR: {validMslkToMsur}/{pm4.LinkEntries.Count} ({100.0*validMslkToMsur/pm4.LinkEntries.Count:F1}%)");
        
        // MSUR.MdosIndex → MSCN
        int validMsurToMscn = pm4.Surfaces.Count(s => s.MdosIndex < pm4.SceneNodes.Count);
        Console.WriteLine($"MSUR.MdosIndex → MSCN: {validMsurToMscn}/{pm4.Surfaces.Count} ({100.0*validMsurToMscn/pm4.Surfaces.Count:F1}%)");
        
        // MSUR → MSVI → MSVT
        int validMsurToMsvt = pm4.Surfaces.Count(s => 
            s.MsviFirstIndex + s.IndexCount <= pm4.MeshIndices.Count);
        Console.WriteLine($"MSUR → MSVI (in range): {validMsurToMsvt}/{pm4.Surfaces.Count} ({100.0*validMsurToMsvt/pm4.Surfaces.Count:F1}%)");
        
        return 0;
    }
    
    /// <summary>
    /// Deep MPRL footprint analysis - investigate if MPRL is building footprints/terrain barriers.
    /// User hypothesis: MPRL contains many vertices around building bases, not single placement points.
    /// </summary>
    static int RunMprlFootprintAnalysis(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var data = File.ReadAllBytes(pm4Path);
        var pm4 = Pm4File.Parse(data);
        string tileName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== MPRL FOOTPRINT ANALYSIS: {tileName} ===");
        Console.WriteLine($"Testing hypothesis: MPRL = building footprints/terrain barriers\n");
        
        // Basic MPRL stats
        int totalMprl = pm4.PositionRefs.Count;
        int normalEntries = pm4.PositionRefs.Count(p => p.Unknown0x16 == 0);  // Type 0 = normal
        int terminatorEntries = pm4.PositionRefs.Count(p => p.Unknown0x16 != 0);  // Type 0x3FFF = terminator
        
        Console.WriteLine($"=== MPRL ENTRY DISTRIBUTION ===");
        Console.WriteLine($"Total MPRL entries: {totalMprl}");
        Console.WriteLine($"  Normal entries (type=0): {normalEntries}");
        Console.WriteLine($"  Terminator entries: {terminatorEntries}");
        
        // Get normal entries only for spatial analysis
        var normalMprl = pm4.PositionRefs
            .Select((p, i) => new {
                Index = i,
                Entry = p,
                // MPRL to MSVT coordinate mapping (based on observed ranges):
                // MPRL field | Observed range | Maps to MSVT axis
                // PositionX  | 11749-12263    | Y (MSVT Y: 11733-12267)
                // PositionY  | 40-184         | Z height (MSVT Z: -12 to 390)
                // PositionZ  | 9605-10130     | X (MSVT X: 9600-10133)
                X = p.PositionZ,   // MPRL Z -> MSVT X
                Y = p.PositionX,   // MPRL X -> MSVT Y
                Z = p.PositionY,   // MPRL Y -> MSVT Z (height)
                Floor = p.Unknown0x14,
                Rotation = 360.0f * p.Unknown0x04 / 65536.0f
            })
            .Where(p => p.Entry.Unknown0x16 == 0)
            .ToList();
        
        if (normalMprl.Count == 0)
        {
            Console.WriteLine("No normal MPRL entries found!");
            return 0;
        }
        
        // Compute MPRL bounding box
        float minX = normalMprl.Min(p => p.X);
        float maxX = normalMprl.Max(p => p.X);
        float minY = normalMprl.Min(p => p.Y);
        float maxY = normalMprl.Max(p => p.Y);
        float minZ = normalMprl.Min(p => p.Z);
        float maxZ = normalMprl.Max(p => p.Z);
        
        Console.WriteLine($"\n=== MPRL SPATIAL BOUNDS ===");
        Console.WriteLine($"X range: {minX:F1} to {maxX:F1} (span: {maxX-minX:F1})");
        Console.WriteLine($"Y range: {minY:F1} to {maxY:F1} (span: {maxY-minY:F1})");
        Console.WriteLine($"Z range: {minZ:F1} to {maxZ:F1} (span: {maxZ-minZ:F1})");
        
        // Compare with MSVT geometry bounds
        if (pm4.MeshVertices.Count > 0)
        {
            float msvtMinX = pm4.MeshVertices.Min(v => v.X);
            float msvtMaxX = pm4.MeshVertices.Max(v => v.X);
            float msvtMinY = pm4.MeshVertices.Min(v => v.Y);
            float msvtMaxY = pm4.MeshVertices.Max(v => v.Y);
            float msvtMinZ = pm4.MeshVertices.Min(v => v.Z);
            float msvtMaxZ = pm4.MeshVertices.Max(v => v.Z);
            
            Console.WriteLine($"\n=== MSVT (Geometry) BOUNDS for comparison ===");
            Console.WriteLine($"MSVT X: {msvtMinX:F1} to {msvtMaxX:F1} (span: {msvtMaxX-msvtMinX:F1})");
            Console.WriteLine($"MSVT Y: {msvtMinY:F1} to {msvtMaxY:F1} (span: {msvtMaxY-msvtMinY:F1})");
            Console.WriteLine($"MSVT Z: {msvtMinZ:F1} to {msvtMaxZ:F1} (span: {msvtMaxZ-msvtMinZ:F1})");
            
            Console.WriteLine($"\nOverlap analysis:");
            Console.WriteLine($"  MPRL within MSVT X: {(minX >= msvtMinX && maxX <= msvtMaxX)}");
            Console.WriteLine($"  MPRL within MSVT Y: {(minY >= msvtMinY && maxY <= msvtMaxY)}");
        }
        
        // === FLOOR LEVEL ANALYSIS ===
        Console.WriteLine($"\n=== MPRL FLOOR LEVEL DISTRIBUTION ===");
        var floorGroups = normalMprl.GroupBy(p => p.Floor).OrderBy(g => g.Key);
        foreach (var fg in floorGroups.Take(10))
        {
            float fMinZ = fg.Min(p => p.Z);
            float fMaxZ = fg.Max(p => p.Z);
            Console.WriteLine($"  Floor {fg.Key}: {fg.Count()} entries, Z range: {fMinZ:F1} - {fMaxZ:F1}");
        }
        
        // === ROTATION ANALYSIS ===
        Console.WriteLine($"\n=== MPRL ROTATION ANALYSIS ===");
        var rotationGroups = normalMprl.GroupBy(p => Math.Round(p.Rotation / 5.0) * 5)  // Group by 5-degree bins
            .OrderByDescending(g => g.Count())
            .Take(10);
        Console.WriteLine("Top 10 rotation values (5° bins):");
        foreach (var rg in rotationGroups)
        {
            Console.WriteLine($"  ~{rg.Key:F0}°: {rg.Count()} entries");
        }
        
        // === SPATIAL CLUSTERING (are MPRL points clustered around objects?) ===
        Console.WriteLine($"\n=== SPATIAL CLUSTERING ANALYSIS ===");
        
        // Group by CK24 and check if MPRL points cluster near those objects
        var surfacesByCk24 = pm4.Surfaces.GroupBy(s => s.CK24).Where(g => g.Key != 0).ToList();
        Console.WriteLine($"Unique CK24 objects (excluding 0): {surfacesByCk24.Count}");
        
        // For each CK24, compute its bounding box from MSVT vertices
        var ck24Bounds = new Dictionary<uint, (Vector3 min, Vector3 max, int surfCount)>();
        foreach (var ck24Grp in surfacesByCk24)
        {
            var vertices = new List<Vector3>();
            foreach (var surf in ck24Grp)
            {
                for (int i = 0; i < (int)surf.IndexCount && surf.MsviFirstIndex + i < pm4.MeshIndices.Count; i++)
                {
                    uint vsIdx = pm4.MeshIndices[(int)surf.MsviFirstIndex + i];
                    if (vsIdx < pm4.MeshVertices.Count)
                    {
                        var v = pm4.MeshVertices[(int)vsIdx];
                        vertices.Add(new Vector3(v.X, v.Y, v.Z));
                    }
                }
            }
            
            if (vertices.Count > 0)
            {
                var min = new Vector3(vertices.Min(v => v.X), vertices.Min(v => v.Y), vertices.Min(v => v.Z));
                var max = new Vector3(vertices.Max(v => v.X), vertices.Max(v => v.Y), vertices.Max(v => v.Z));
                ck24Bounds[ck24Grp.Key] = (min, max, ck24Grp.Count());
            }
        }
        
        // Count MPRL points within each CK24's bounding box (foot-print test)
        Console.WriteLine($"\nMPRL points near CK24 object bounds (within 5 units):");
        var mprlNearCk24 = new Dictionary<uint, int>();
        float margin = 5.0f;  // Search margin around bounding box
        
        foreach (var (ck24, bounds) in ck24Bounds.OrderByDescending(kv => kv.Value.surfCount).Take(20))
        {
            int nearbyCount = normalMprl.Count(p => 
                p.X >= bounds.min.X - margin && p.X <= bounds.max.X + margin &&
                p.Y >= bounds.min.Y - margin && p.Y <= bounds.max.Y + margin);
            
            if (nearbyCount > 0)
            {
                mprlNearCk24[ck24] = nearbyCount;
                float sizeX = bounds.max.X - bounds.min.X;
                float sizeY = bounds.max.Y - bounds.min.Y;
                Console.WriteLine($"  CK24 0x{ck24:X6}: {nearbyCount} MPRL, box {sizeX:F0}x{sizeY:F0}, {bounds.surfCount} surfaces");
            }
        }
        
        int totalNearObjects = mprlNearCk24.Values.Sum();
        Console.WriteLine($"\nTotal MPRL within object bounds: {totalNearObjects}/{normalMprl.Count} ({100.0*totalNearObjects/normalMprl.Count:F1}%)");
        
        // === FOOTPRINT PERIMETER TEST ===
        // If MPRL is a footprint, points should be at the EDGES of bounding boxes (perimeter)
        Console.WriteLine($"\n=== FOOTPRINT PERIMETER TEST ===");
        Console.WriteLine("Testing if MPRL points cluster at object EDGES (footprint perimeter)");
        
        int perimeterPoints = 0;
        int interiorPoints = 0;
        float edgeMargin = 3.0f;  // Consider points within 3 units of edge as "on edge"
        
        foreach (var (ck24, bounds) in ck24Bounds)
        {
            foreach (var mprl in normalMprl)
            {
                // Check if within bounding box (with margin)
                if (mprl.X >= bounds.min.X - margin && mprl.X <= bounds.max.X + margin &&
                    mprl.Y >= bounds.min.Y - margin && mprl.Y <= bounds.max.Y + margin)
                {
                    // Check if on edge (near boundary) or in interior
                    bool nearXMin = Math.Abs(mprl.X - bounds.min.X) < edgeMargin;
                    bool nearXMax = Math.Abs(mprl.X - bounds.max.X) < edgeMargin;
                    bool nearYMin = Math.Abs(mprl.Y - bounds.min.Y) < edgeMargin;
                    bool nearYMax = Math.Abs(mprl.Y - bounds.max.Y) < edgeMargin;
                    
                    if (nearXMin || nearXMax || nearYMin || nearYMax)
                        perimeterPoints++;
                    else
                        interiorPoints++;
                }
            }
        }
        
        Console.WriteLine($"  Perimeter points (near edges): {perimeterPoints}");
        Console.WriteLine($"  Interior points: {interiorPoints}");
        if (perimeterPoints + interiorPoints > 0)
        {
            Console.WriteLine($"  Perimeter ratio: {100.0*perimeterPoints/(perimeterPoints+interiorPoints):F1}%");
            if (perimeterPoints > interiorPoints)
                Console.WriteLine($"  RESULT: MPRL favors EDGES → supports FOOTPRINT hypothesis!");
            else
                Console.WriteLine($"  RESULT: MPRL in interiors → does NOT support footprint hypothesis");
        }
        
        // === DENSITY ANALYSIS (many points per object = footprint, few = placement) ===
        Console.WriteLine($"\n=== MPRL DENSITY PER OBJECT ===");
        var mprlPerCk24 = mprlNearCk24.OrderByDescending(kv => kv.Value).Take(10);
        Console.WriteLine("Objects with most MPRL points:");
        foreach (var (ck24, count) in mprlPerCk24)
        {
            var bounds = ck24Bounds[ck24];
            float area = (bounds.max.X - bounds.min.X) * (bounds.max.Y - bounds.min.Y);
            float density = area > 0 ? count / area : 0;
            Console.WriteLine($"  CK24 0x{ck24:X6}: {count} MPRL, area: {area:F0} sq, density: {density:F3}/sq");
        }
        
        // === EXPORT OBJs FOR VISUAL VERIFICATION ===
        // Output to test_output folder (not temp - user clears that)
        string baseDir = Path.GetDirectoryName(pm4Path) ?? ".";
        string outputDir = Path.Combine(baseDir, "..", "..", "..", "..", "test_output", "mprl_terrain_points");
        Directory.CreateDirectory(outputDir);
        
        // Export 1: MPRL points only
        string mprlObjPath = Path.Combine(outputDir, $"{tileName}_mprl_points.obj");
        using (var sw = new StreamWriter(mprlObjPath))
        {
            sw.WriteLine($"# MPRL Points from {tileName}");
            sw.WriteLine($"# Total points: {normalMprl.Count}");
            sw.WriteLine($"# Showing MPRL distribution - load alongside geometry to verify footprint hypothesis");
            sw.WriteLine();
            
            foreach (var p in normalMprl)
            {
                sw.WriteLine($"v {p.X:F3} {p.Y:F3} {p.Z:F3}");
            }
        }
        Console.WriteLine($"\n=== EXPORTED OBJs FOR VISUAL VERIFICATION ===");
        Console.WriteLine($"MPRL points: {mprlObjPath}");
        
        // Export 2: For each CK24 with MPRL points, export geometry + MPRL together
        var topCk24s = mprlNearCk24.OrderByDescending(kv => kv.Value).Take(5);
        foreach (var (ck24, mprlCount) in topCk24s)
        {
            string ck24ObjPath = Path.Combine(outputDir, $"{tileName}_ck24_{ck24:X6}_with_mprl.obj");
            
            using (var sw = new StreamWriter(ck24ObjPath))
            {
                sw.WriteLine($"# CK24 0x{ck24:X6} geometry + MPRL points");
                sw.WriteLine($"# This shows MPRL distribution relative to object geometry");
                sw.WriteLine();
                
                // Collect geometry for this CK24
                var ck24Surfaces = pm4.Surfaces.Where(s => s.CK24 == ck24).ToList();
                var vertices = new List<Vector3>();
                var faces = new List<(int, int, int)>();
                
                foreach (var surf in ck24Surfaces)
                {
                    if (surf.IndexCount < 3) continue;
                    
                    // Get triangle indices
                    for (int t = 0; t < (int)surf.IndexCount - 2; t += 3)
                    {
                        int idx0 = (int)surf.MsviFirstIndex + t;
                        int idx1 = (int)surf.MsviFirstIndex + t + 1;
                        int idx2 = (int)surf.MsviFirstIndex + t + 2;
                        
                        if (idx2 >= pm4.MeshIndices.Count) continue;
                        
                        uint vi0 = pm4.MeshIndices[idx0];
                        uint vi1 = pm4.MeshIndices[idx1];
                        uint vi2 = pm4.MeshIndices[idx2];
                        
                        if (vi0 >= pm4.MeshVertices.Count || vi1 >= pm4.MeshVertices.Count || vi2 >= pm4.MeshVertices.Count) continue;
                        
                        var v0 = pm4.MeshVertices[(int)vi0];
                        var v1 = pm4.MeshVertices[(int)vi1];
                        var v2 = pm4.MeshVertices[(int)vi2];
                        
                        int baseIdx = vertices.Count;
                        vertices.Add(new Vector3(v0.X, v0.Y, v0.Z));
                        vertices.Add(new Vector3(v1.X, v1.Y, v1.Z));
                        vertices.Add(new Vector3(v2.X, v2.Y, v2.Z));
                        faces.Add((baseIdx + 1, baseIdx + 2, baseIdx + 3));  // OBJ is 1-indexed
                    }
                }
                
                // Write geometry vertices
                sw.WriteLine($"# CK24 geometry: {vertices.Count} vertices, {faces.Count} triangles");
                foreach (var v in vertices)
                {
                    sw.WriteLine($"v {v.X:F3} {v.Y:F3} {v.Z:F3}");
                }
                
                // Write faces
                foreach (var (a, b, c) in faces)
                {
                    sw.WriteLine($"f {a} {b} {c}");
                }
                
                // Now add MPRL points as additional vertices (for visual overlay)
                var bounds = ck24Bounds[ck24];
                // margin already defined in outer scope
                var nearbyMprlPts = normalMprl.Where(p => 
                    p.X >= bounds.min.X - margin && p.X <= bounds.max.X + margin &&
                    p.Y >= bounds.min.Y - margin && p.Y <= bounds.max.Y + margin).ToList();
                
                sw.WriteLine();
                sw.WriteLine($"# MPRL points near this object: {nearbyMprlPts.Count}");
                sw.WriteLine("# (These are additional vertices - not connected, just point markers)");
                foreach (var mp in nearbyMprlPts)
                {
                    // Output MPRL at same Z as geometry floor for visibility
                    sw.WriteLine($"v {mp.X:F3} {mp.Y:F3} {mp.Z:F3}");
                }
            }
            
            Console.WriteLine($"CK24 0x{ck24:X6} + MPRL: {ck24ObjPath}");
        }
        
        // Export 3: All geometry bounding boxes as wireframe + MPRL
        string combinedPath = Path.Combine(outputDir, $"{tileName}_all_bounds_with_mprl.obj");
        using (var sw = new StreamWriter(combinedPath))
        {
            sw.WriteLine($"# All CK24 bounding boxes (wireframe) + MPRL points");
            sw.WriteLine($"# This shows spatial relationship between MPRL and object footprints");
            sw.WriteLine();
            
            int vertIdx = 1;
            
            // Draw bounding boxes as wireframe cubes
            foreach (var (ck24, bounds) in ck24Bounds)
            {
                // 8 corners of bounding box
                var corners = new[] {
                    new Vector3(bounds.min.X, bounds.min.Y, bounds.min.Z),  // 0: min corner
                    new Vector3(bounds.max.X, bounds.min.Y, bounds.min.Z),  // 1
                    new Vector3(bounds.max.X, bounds.max.Y, bounds.min.Z),  // 2
                    new Vector3(bounds.min.X, bounds.max.Y, bounds.min.Z),  // 3
                    new Vector3(bounds.min.X, bounds.min.Y, bounds.max.Z),  // 4: max Z level
                    new Vector3(bounds.max.X, bounds.min.Y, bounds.max.Z),  // 5
                    new Vector3(bounds.max.X, bounds.max.Y, bounds.max.Z),  // 6: max corner
                    new Vector3(bounds.min.X, bounds.max.Y, bounds.max.Z),  // 7
                };
                
                sw.WriteLine($"# CK24 0x{ck24:X6} bounding box");
                foreach (var c in corners)
                {
                    sw.WriteLine($"v {c.X:F3} {c.Y:F3} {c.Z:F3}");
                }
                
                // Bottom face, top face, vertical edges (as lines - using 'l')
                sw.WriteLine($"l {vertIdx} {vertIdx+1}");
                sw.WriteLine($"l {vertIdx+1} {vertIdx+2}");
                sw.WriteLine($"l {vertIdx+2} {vertIdx+3}");
                sw.WriteLine($"l {vertIdx+3} {vertIdx}");
                sw.WriteLine($"l {vertIdx+4} {vertIdx+5}");
                sw.WriteLine($"l {vertIdx+5} {vertIdx+6}");
                sw.WriteLine($"l {vertIdx+6} {vertIdx+7}");
                sw.WriteLine($"l {vertIdx+7} {vertIdx+4}");
                sw.WriteLine($"l {vertIdx} {vertIdx+4}");
                sw.WriteLine($"l {vertIdx+1} {vertIdx+5}");
                sw.WriteLine($"l {vertIdx+2} {vertIdx+6}");
                sw.WriteLine($"l {vertIdx+3} {vertIdx+7}");
                
                vertIdx += 8;
            }
            
            // Add all MPRL points 
            sw.WriteLine();
            sw.WriteLine($"# MPRL points: {normalMprl.Count}");
            foreach (var p in normalMprl)
            {
                sw.WriteLine($"v {p.X:F3} {p.Y:F3} {p.Z:F3}");
            }
        }
        Console.WriteLine($"All bounds + MPRL: {combinedPath}");
        Console.WriteLine($"\nOpen these OBJs in a 3D viewer to verify MPRL distribution!");
        
        // === HYPOTHESIS CONCLUSION ===
        Console.WriteLine($"\n=== HYPOTHESIS EVALUATION ===");
        Console.WriteLine("Testing: MPRL = building footprints/terrain barriers (many perimeter vertices)");
        
        // Evidence summary
        int evidence = 0;
        if (normalMprl.Count > 100)
        {
            Console.WriteLine("✓ High vertex count suggests footprints (not simple placement points)");
            evidence++;
        }
        if (perimeterPoints > interiorPoints)
        {
            Console.WriteLine("✓ More perimeter points than interior → supports footprint");
            evidence++;
        }
        else
        {
            Console.WriteLine("✗ Interior points dominate → may NOT be footprints");
        }
        if (mprlNearCk24.Count > surfacesByCk24.Count / 2)
        {
            Console.WriteLine("✓ Most CK24 objects have associated MPRL → supports per-object footprints");
            evidence++;
        }
        
        Console.WriteLine($"\nEvidence score: {evidence}/3");
        if (evidence >= 2)
            Console.WriteLine("CONCLUSION: MPRL likely IS building footprints/terrain barriers!");
        else
            Console.WriteLine("CONCLUSION: MPRL may have different purpose (needs more investigation)");
        
        return 0;
    }
    
    /// <summary>
    /// Extract terrain height data from MPRL terrain intersection points.
    /// Maps MPRL Z values to ADT MCNK chunk positions for terrain reconstruction.
    /// </summary>
    static int RunMprlToAdtTerrainExtraction(string pm4Path)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        var data = File.ReadAllBytes(pm4Path);
        var pm4 = Pm4File.Parse(data);
        string tileName = Path.GetFileNameWithoutExtension(pm4Path);
        
        // Extract tile coordinates from filename (development_XX_YY.pm4)
        int tileX = 0, tileY = 0;
        var match = System.Text.RegularExpressions.Regex.Match(tileName, @"_(\d+)_(\d+)$");
        if (match.Success)
        {
            tileX = int.Parse(match.Groups[1].Value);
            tileY = int.Parse(match.Groups[2].Value);
        }
        
        Console.WriteLine($"=== MPRL TO ADT TERRAIN EXTRACTION: {tileName} ===");
        Console.WriteLine($"Tile coordinates: ({tileX}, {tileY})");
        Console.WriteLine($"Extracting terrain heights from MPRL intersection points\n");
        
        // ADT constants
        const float ADT_SIZE = 533.333f;  // ADT tile size in yards
        const int CHUNKS_PER_SIDE = 16;   // 16x16 MCNK chunks per ADT
        const float MCNK_SIZE = ADT_SIZE / CHUNKS_PER_SIDE;  // ~33.33 yards per chunk
        const int VERTS_PER_CHUNK = 9;    // 9x9 heightmap vertices per chunk (edges shared)
        const float VERT_SPACING = MCNK_SIZE / 8;  // spacing between vertices
        
        // Get normal MPRL entries (type=0)
        var normalMprl = pm4.PositionRefs
            .Where(p => p.Unknown0x16 == 0)
            .Select(p => new {
                // MPRL to MSVT coordinate mapping
                X = p.PositionZ,   // MPRL Z -> MSVT X
                Y = p.PositionX,   // MPRL X -> MSVT Y
                Z = p.PositionY,   // MPRL Y -> MSVT Z (height)
                Floor = p.Unknown0x14
            })
            .ToList();
        
        Console.WriteLine($"MPRL entries: {pm4.PositionRefs.Count} total, {normalMprl.Count} normal entries");
        
        if (normalMprl.Count == 0)
        {
            Console.WriteLine("No MPRL data to extract!");
            return 0;
        }
        
        // Compute MPRL bounds
        float minX = normalMprl.Min(p => p.X);
        float maxX = normalMprl.Max(p => p.X);
        float minY = normalMprl.Min(p => p.Y);
        float maxY = normalMprl.Max(p => p.Y);
        float minZ = normalMprl.Min(p => p.Z);
        float maxZ = normalMprl.Max(p => p.Z);
        
        Console.WriteLine($"\nMPRL terrain intersection bounds:");
        Console.WriteLine($"  X: {minX:F1} to {maxX:F1} (span: {maxX-minX:F1})");
        Console.WriteLine($"  Y: {minY:F1} to {maxY:F1} (span: {maxY-minY:F1})");
        Console.WriteLine($"  Z (height): {minZ:F1} to {maxZ:F1} (span: {maxZ-minZ:F1})");
        
        // Create 16x16 grid of MCNK chunks
        var chunkHeights = new List<float>[16, 16];
        for (int cx = 0; cx < 16; cx++)
            for (int cy = 0; cy < 16; cy++)
                chunkHeights[cx, cy] = new List<float>();
        
        // Map MPRL points to chunks
        // Assume tile starts at minX, minY and covers ADT_SIZE
        float tileMinX = minX;
        float tileMinY = minY;
        
        int mappedPoints = 0;
        foreach (var pt in normalMprl)
        {
            // Calculate chunk indices
            int cx = (int)((pt.X - tileMinX) / MCNK_SIZE);
            int cy = (int)((pt.Y - tileMinY) / MCNK_SIZE);
            
            // Clamp to valid range
            cx = Math.Clamp(cx, 0, 15);
            cy = Math.Clamp(cy, 0, 15);
            
            // Add height to chunk's height collection
            chunkHeights[cx, cy].Add(pt.Z);
            mappedPoints++;
        }
        
        Console.WriteLine($"\nMapped {mappedPoints} MPRL points to MCNK chunks");
        
        // Analyze chunk coverage
        int chunksWithData = 0;
        int totalHeightPoints = 0;
        float[,] avgChunkHeights = new float[16, 16];
        
        Console.WriteLine("\n=== MCNK CHUNK HEIGHT DATA ===");
        for (int cy = 0; cy < 16; cy++)
        {
            var row = new List<string>();
            for (int cx = 0; cx < 16; cx++)
            {
                var heights = chunkHeights[cx, cy];
                if (heights.Count > 0)
                {
                    chunksWithData++;
                    totalHeightPoints += heights.Count;
                    avgChunkHeights[cx, cy] = heights.Average();
                    row.Add($"{heights.Count,3}");
                }
                else
                {
                    avgChunkHeights[cx, cy] = float.NaN;
                    row.Add("  -");
                }
            }
            Console.WriteLine($"Row {cy,2}: {string.Join(" ", row)}");
        }
        
        Console.WriteLine($"\nChunks with MPRL data: {chunksWithData}/256 ({100.0*chunksWithData/256:F1}%)");
        Console.WriteLine($"Total height points: {totalHeightPoints}");
        
        // === Export terrain heightmap as OBJ ===
        string baseDir = Path.GetDirectoryName(pm4Path) ?? ".";
        string outputDir = Path.Combine(baseDir, "..", "..", "..", "..", "test_output", "mprl_terrain");
        Directory.CreateDirectory(outputDir);
        
        // Export 1: Raw MPRL points as terrain mesh
        string terrainObjPath = Path.Combine(outputDir, $"{tileName}_terrain_points.obj");
        using (var sw = new StreamWriter(terrainObjPath))
        {
            sw.WriteLine($"# MPRL Terrain Intersection Points - {tileName}");
            sw.WriteLine($"# {normalMprl.Count} points representing terrain heights where objects touch ground");
            sw.WriteLine($"# Z range: {minZ:F1} to {maxZ:F1}");
            sw.WriteLine();
            
            foreach (var pt in normalMprl)
            {
                sw.WriteLine($"v {pt.X:F3} {pt.Y:F3} {pt.Z:F3}");
            }
        }
        Console.WriteLine($"\n=== EXPORTED ===");
        Console.WriteLine($"Terrain points: {terrainObjPath}");
        
        // Export 2: Chunk average heights as grid
        string gridObjPath = Path.Combine(outputDir, $"{tileName}_terrain_grid.obj");
        using (var sw = new StreamWriter(gridObjPath))
        {
            sw.WriteLine($"# MPRL Chunk Average Heights - {tileName}");
            sw.WriteLine($"# 16x16 grid of average terrain heights from MPRL data");
            sw.WriteLine();
            
            int vertIdx = 1;
            
            // Create grid vertices
            for (int cy = 0; cy < 16; cy++)
            {
                for (int cx = 0; cx < 16; cx++)
                {
                    float x = tileMinX + cx * MCNK_SIZE + MCNK_SIZE / 2;
                    float y = tileMinY + cy * MCNK_SIZE + MCNK_SIZE / 2;
                    float z = float.IsNaN(avgChunkHeights[cx, cy]) ? minZ : avgChunkHeights[cx, cy];
                    sw.WriteLine($"v {x:F3} {y:F3} {z:F3}");
                }
            }
            
            // Create quad faces for the grid
            for (int cy = 0; cy < 15; cy++)
            {
                for (int cx = 0; cx < 15; cx++)
                {
                    int v0 = cy * 16 + cx + 1;        // bottom-left
                    int v1 = cy * 16 + cx + 2;        // bottom-right
                    int v2 = (cy + 1) * 16 + cx + 2;  // top-right
                    int v3 = (cy + 1) * 16 + cx + 1;  // top-left
                    sw.WriteLine($"f {v0} {v1} {v2} {v3}");
                }
            }
        }
        Console.WriteLine($"Terrain grid: {gridObjPath}");
        
        // Export 3: CSV with chunk height data for ADT injection
        string csvPath = Path.Combine(outputDir, $"{tileName}_terrain_heights.csv");
        using (var sw = new StreamWriter(csvPath))
        {
            sw.WriteLine("ChunkX,ChunkY,AvgHeight,MinHeight,MaxHeight,PointCount");
            for (int cy = 0; cy < 16; cy++)
            {
                for (int cx = 0; cx < 16; cx++)
                {
                    var heights = chunkHeights[cx, cy];
                    if (heights.Count > 0)
                    {
                        sw.WriteLine($"{cx},{cy},{heights.Average():F3},{heights.Min():F3},{heights.Max():F3},{heights.Count}");
                    }
                }
            }
        }
        Console.WriteLine($"Height CSV: {csvPath}");
        
        Console.WriteLine($"\nReady for ADT terrain injection!");
        Console.WriteLine($"Use CSV data to update MCNK heightmap (MCVT) in ADT files.");
        
        return 0;
    }
    
    static int ExportObjPerCk24(string pm4Path, string? outputDir)
    {
        if (!File.Exists(pm4Path))
        {
            Console.WriteLine($"File not found: {pm4Path}");
            return 1;
        }
        
        // NEVER write to source data folders - use temp or explicit path
        outputDir ??= Path.Combine(Path.GetTempPath(), "pm4_obj_export");
        Directory.CreateDirectory(outputDir);
        
        var pm4 = Pm4File.Parse(File.ReadAllBytes(pm4Path));
        var baseName = Path.GetFileNameWithoutExtension(pm4Path);
        
        Console.WriteLine($"=== Exporting OBJ files from {baseName} ===");
        Console.WriteLine($"Output directory: {outputDir}");
        
        int totalExported = 0;
        
        // Export 1: Full tile
        Console.WriteLine("\n--- Full Tile ---");
        ExportSurfacesToObj(pm4, pm4.Surfaces.ToList(), 
            Path.Combine(outputDir, $"{baseName}_FULL.obj"), "Full Tile");
        totalExported++;
        
        // === VERTEX CONNECTIVITY GROUPING ===
        Console.WriteLine("\n--- Vertex Connectivity Grouping (Union-Find) ---");
        
        // For largest CK24, group surfaces by shared vertices
        var largestCk24 = pm4.Surfaces.Where(s => s.CK24 != 0)
            .GroupBy(s => s.CK24).OrderByDescending(g => g.Count()).First();
        var ck24Surfaces = largestCk24.ToList();
        uint ck24Val = largestCk24.Key;
        
        Console.WriteLine($"  Analyzing CK24 0x{ck24Val:X6}: {ck24Surfaces.Count} surfaces");
        
        // Build vertex -> surface mappings
        var vertexToSurfaces = new Dictionary<int, List<int>>(); // msvtIdx -> surface indices
        for (int si = 0; si < ck24Surfaces.Count; si++)
        {
            var surf = ck24Surfaces[si];
            for (int i = 0; i < surf.IndexCount; i++)
            {
                uint msviIdx = surf.MsviFirstIndex + (uint)i;
                if (msviIdx < pm4.MeshIndices.Count)
                {
                    int msvtIdx = (int)pm4.MeshIndices[(int)msviIdx];
                    if (!vertexToSurfaces.ContainsKey(msvtIdx))
                        vertexToSurfaces[msvtIdx] = new List<int>();
                    vertexToSurfaces[msvtIdx].Add(si);
                }
            }
        }
        
        // Union-Find: group surfaces that share vertices
        int[] parent = new int[ck24Surfaces.Count];
        for (int i = 0; i < parent.Length; i++) parent[i] = i;
        
        int Find(int x) {
            if (parent[x] != x) parent[x] = Find(parent[x]);
            return parent[x];
        }
        void Union(int a, int b) {
            int ra = Find(a), rb = Find(b);
            if (ra != rb) parent[ra] = rb;
        }
        
        // Union surfaces that share any vertex
        foreach (var (_, surfaceIndices) in vertexToSurfaces)
        {
            for (int i = 1; i < surfaceIndices.Count; i++)
            {
                Union(surfaceIndices[0], surfaceIndices[i]);
            }
        }
        
        // Group surfaces by their root
        var groups = new Dictionary<int, List<MsurEntry>>();
        for (int i = 0; i < ck24Surfaces.Count; i++)
        {
            int root = Find(i);
            if (!groups.ContainsKey(root))
                groups[root] = new List<MsurEntry>();
            groups[root].Add(ck24Surfaces[i]);
        }
        
        Console.WriteLine($"  Found {groups.Count} connected components (objects)");
        
        // Export top 20 largest groups
        int objNum = 0;
        foreach (var grp in groups.Values.OrderByDescending(g => g.Count).Take(20))
        {
            ExportSurfacesToObj(pm4, grp, 
                Path.Combine(outputDir, $"{baseName}_CK24_{ck24Val:X6}_connected_{objNum:D2}.obj"), 
                $"Connected component {objNum}");
            objNum++;
            totalExported++;
        }
        
        Console.WriteLine($"\nExported {totalExported} OBJ files to {outputDir}");
        return 0;
    }
    
    static void ExportSurfacesToObj(Pm4File pm4, List<MsurEntry> surfaces, string objPath, string description)
    {
        var usedVertices = new Dictionary<int, int>();
        var outputVerts = new List<Vector3>();
        var faces = new List<int[]>();
        
        foreach (var surf in surfaces)
        {
            var faceIndices = new List<int>();
            for (int i = 0; i < surf.IndexCount; i++)
            {
                uint msviIdx = surf.MsviFirstIndex + (uint)i;
                if (msviIdx < pm4.MeshIndices.Count)
                {
                    int msvtIdx = (int)pm4.MeshIndices[(int)msviIdx];
                    if (msvtIdx < pm4.MeshVertices.Count)
                    {
                        if (!usedVertices.TryGetValue(msvtIdx, out int newIdx))
                        {
                            newIdx = outputVerts.Count + 1;
                            usedVertices[msvtIdx] = newIdx;
                            outputVerts.Add(pm4.MeshVertices[msvtIdx]);
                        }
                        faceIndices.Add(newIdx);
                    }
                }
            }
            if (faceIndices.Count >= 3)
            {
                faces.Add(faceIndices.ToArray());
            }
        }
        
        if (outputVerts.Count == 0) 
        {
            Console.WriteLine($"  SKIP: {description} (no vertices)");
            return;
        }
        
        var bounds = ComputeBounds(outputVerts);
        
        using var sw = new StreamWriter(objPath);
        sw.WriteLine($"# PM4 Export: {description}");
        sw.WriteLine($"# Surfaces: {surfaces.Count}");
        sw.WriteLine($"# Vertices: {outputVerts.Count}");
        sw.WriteLine($"# Faces: {faces.Count}");
        sw.WriteLine($"# Bounds: ({bounds.min.X:F1}, {bounds.min.Y:F1}, {bounds.min.Z:F1}) to ({bounds.max.X:F1}, {bounds.max.Y:F1}, {bounds.max.Z:F1})");
        sw.WriteLine();
        
        foreach (var v in outputVerts)
            sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
        sw.WriteLine();
        
        foreach (var f in faces)
            sw.WriteLine($"f {string.Join(" ", f)}");
        
        Console.WriteLine($"  Exported: {Path.GetFileName(objPath)} ({outputVerts.Count} verts, {faces.Count} faces)");
    }
    
    static (Vector3 min, Vector3 max) ComputeSurfaceBounds(Pm4File pm4, List<MsurEntry> surfaces)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        
        foreach (var surf in surfaces)
        {
            var verts = GetSurfaceVertices(pm4, surf);
            foreach (var v in verts)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
        }
        return (min, max);
    }
    static Pm4Stats? ProcessFile(string path, bool verbose = false)
    {
        try
        {
            var data = File.ReadAllBytes(path);
            var pm4 = Pm4File.Parse(data);
            var stats = new Pm4Stats { FileName = Path.GetFileName(path) };
            
            if (verbose)
            {
                Console.WriteLine($"=== {stats.FileName} ===\n");
                Console.WriteLine("Chunks:");
                foreach (var (sig, size) in pm4.ChunkSizes)
                {
                    int count = GetEntryCount(sig, size);
                    Console.WriteLine($"  {sig}: {size:N0} bytes ({count} entries)");
                }
                Console.WriteLine();
            }
            
            // Analyze surfaces by CK24
            var surfacesByCk24 = pm4.Surfaces.GroupBy(s => s.CK24).ToDictionary(g => g.Key, g => g.ToList());
            
            stats.Ck24ZeroSurfaceCount = surfacesByCk24.GetValueOrDefault(0u)?.Count ?? 0;
            stats.Ck24NonZeroSurfaceCount = pm4.Surfaces.Count - stats.Ck24ZeroSurfaceCount;
            stats.UniqueCk24Count = surfacesByCk24.Count;
            
            if (verbose)
            {
                Console.WriteLine($"Surfaces: {pm4.Surfaces.Count:N0}");
                Console.WriteLine($"  CK24=0x000000: {stats.Ck24ZeroSurfaceCount:N0} ({100.0 * stats.Ck24ZeroSurfaceCount / pm4.Surfaces.Count:F1}%)");
                Console.WriteLine($"  CK24!=0: {stats.Ck24NonZeroSurfaceCount:N0}");
                Console.WriteLine($"  Unique CK24 values: {stats.UniqueCk24Count}");
                
                // MSHD Header Analysis
                if (pm4.Header != null)
                {
                    Console.WriteLine("\n=== MSHD Header Analysis (32 bytes) ===");
                    var h = pm4.Header;
                    Console.WriteLine($"  Field00: 0x{h.Field00:X8} ({h.Field00})");
                    Console.WriteLine($"  Field04: 0x{h.Field04:X8} ({h.Field04})");
                    Console.WriteLine($"  Field08: 0x{h.Field08:X8} ({h.Field08})");
                    Console.WriteLine($"  Field0C: 0x{h.Field0C:X8} ({h.Field0C})");
                    Console.WriteLine($"  Field10: 0x{h.Field10:X8} ({h.Field10})");
                    Console.WriteLine($"  Field14: 0x{h.Field14:X8} ({h.Field14})");
                    Console.WriteLine($"  Field18: 0x{h.Field18:X8} ({h.Field18})");
                    Console.WriteLine($"  Field1C: 0x{h.Field1C:X8} ({h.Field1C})");
                    
                    // Check for correlations
                    Console.WriteLine("\n  Potential correlations:");
                    Console.WriteLine($"    Field00 == MSLK.Count? {h.Field00 == pm4.LinkEntries.Count} (MSLK={pm4.LinkEntries.Count})");
                    Console.WriteLine($"    Field04 == MPRL.Count? {h.Field04 == pm4.PositionRefs.Count} (MPRL={pm4.PositionRefs.Count})");
                    Console.WriteLine($"    Field08 == MSVT.Count? {h.Field08 == pm4.MeshVertices.Count} (MSVT={pm4.MeshVertices.Count})");
                    Console.WriteLine($"    Field0C == MSUR.Count? {h.Field0C == pm4.Surfaces.Count} (MSUR={pm4.Surfaces.Count})");
                    Console.WriteLine($"    Field10 == MSPV.Count? {h.Field10 == pm4.PathVertices.Count} (MSPV={pm4.PathVertices.Count})");
                    Console.WriteLine($"    Field14 == MSCN.Count? {h.Field14 == pm4.SceneNodes.Count} (MSCN={pm4.SceneNodes.Count})");
                    
                    // More correlations to try
                    int nonTermMprl = pm4.PositionRefs.Count(p => p.Unknown0x16 == 0);
                    int uniqueCk24 = pm4.Surfaces.Select(s => s.CK24).Distinct().Count();
                    int mslkWithGeom = pm4.LinkEntries.Count(l => l.MspiFirstIndex >= 0);
                    int mslkNoGeom = pm4.LinkEntries.Count(l => l.MspiFirstIndex < 0);
                    int maxFloor = pm4.LinkEntries.Max(l => l.Subtype);
                    
                    Console.WriteLine("\n  Extended correlations:");
                    Console.WriteLine($"    Field00 == Non-terminator MPRLs? {h.Field00 == nonTermMprl} (count={nonTermMprl})");
                    Console.WriteLine($"    Field04 == Unique CK24s? {h.Field04 == uniqueCk24} (count={uniqueCk24})");
                    Console.WriteLine($"    Field00 == MSLK with geom? {h.Field00 == mslkWithGeom} (count={mslkWithGeom})");
                    Console.WriteLine($"    Field04 == MSLK no geom? {h.Field04 == mslkNoGeom} (count={mslkNoGeom})");
                    Console.WriteLine($"    Field00 - Field04 = {h.Field00 - h.Field04} (difference)");
                    Console.WriteLine($"    Max floor level: {maxFloor}");
                    
                    // Check if related to MPRR segments
                    int mprrSentinels = pm4.MprrEntries.Count(r => r.Value1 == 0xFFFF);
                    Console.WriteLine($"    Field00 == MPRR sentinels? {h.Field00 == mprrSentinels} (count={mprrSentinels})");
                    
                    // Tile coordinate analysis
                    var match = System.Text.RegularExpressions.Regex.Match(path, @"(\d+)_(\d+)\.pm4$");
                    if (match.Success)
                    {
                        int tileX = int.Parse(match.Groups[1].Value);
                        int tileY = int.Parse(match.Groups[2].Value);
                        Console.WriteLine($"\n  Tile coordinate analysis (tile {tileX}_{tileY}):");
                        Console.WriteLine($"    Field00 (534) vs tileX*24 ({tileX * 24}): {h.Field00 == tileX * 24}");
                        Console.WriteLine($"    Field04 (525) vs tileY*30 ({tileY * 30}): {h.Field04 == tileY * 30}");
                        Console.WriteLine($"    Field00 / 2 = {h.Field00 / 2}, Field04 / 2 = {h.Field04 / 2}");
                        Console.WriteLine($"    sqrt(Field00) = {Math.Sqrt(h.Field00):F1}, sqrt(Field04) = {Math.Sqrt(h.Field04):F1}");
                        
                        // Check if related to grid subdivisions
                        long gridCells = ((long)h.Field00 + 1) * (h.Field04 + 1);
                        Console.WriteLine($"    Grid cells if (F00+1)*(F04+1) = {gridCells}");
                    }
                    
                    Console.WriteLine($"    Raw hex: {BitConverter.ToString(h.RawBytes).Replace("-", " ")}");
                }
                
                // Analyze CK24 bit structure
                Console.WriteLine("\n=== CK24 Bit Structure Analysis ===");
                Console.WriteLine("  CK24 = [Type:8bit][ObjectID:16bit]");
                Console.WriteLine("  Type = (CK24 >> 16) & 0xFF");
                Console.WriteLine("  ObjectID = CK24 & 0xFFFF");
                
                // Group by Type byte (top 8 bits of CK24)
                var byType = pm4.Surfaces
                    .Where(s => s.CK24 != 0)
                    .GroupBy(s => s.CK24Type)
                    .OrderByDescending(g => g.Count());
                
                Console.WriteLine("\n  By Type Byte (top 8 bits):");
                foreach (var g in byType.Take(10))
                {
                    var uniqueObjects = g.Select(s => s.CK24ObjectId).Distinct().Count();
                    Console.WriteLine($"    Type 0x{g.Key:X2}: {g.Count(),5} surfaces, {uniqueObjects} unique ObjectIDs");
                }
                
                // Group by ObjectID (bottom 16 bits of CK24)
                var byObjectId = pm4.Surfaces
                    .Where(s => s.CK24 != 0)
                    .GroupBy(s => s.CK24ObjectId)
                    .OrderByDescending(g => g.Count());
                
                Console.WriteLine($"\n  Unique ObjectIDs (bottom 16 bits): {byObjectId.Count()}");
                Console.WriteLine("  Top 5 ObjectIDs by surface count:");
                foreach (var g in byObjectId.Take(5))
                {
                    var types = g.Select(s => s.CK24Type).Distinct().ToList();
                    Console.WriteLine($"    ObjectID 0x{g.Key:X4}: {g.Count(),5} surfaces, Types: [{string.Join(", ", types.Select(t => $"0x{t:X2}"))}]");
                }
                
                // Cross-reference: GroupKey vs CK24
                Console.WriteLine("\n  GroupKey vs CK24 cross-reference:");
                var groupKeyCk24 = pm4.Surfaces
                    .GroupBy(s => s.GroupKey)
                    .OrderBy(g => g.Key);
                
                foreach (var gk in groupKeyCk24)
                {
                    var uniqueCk24 = gk.Select(s => s.CK24).Distinct().Count();
                    var ck24Zero = gk.Count(s => s.CK24 == 0);
                    Console.WriteLine($"    GroupKey={gk.Key}: {gk.Count(),5} surfaces, {uniqueCk24} unique CK24s, CK24=0: {ck24Zero}");
                }
                
                // === GroupKey vs Type Byte analysis (M2 vs WMO?) ===
                Console.WriteLine("\n=== GroupKey vs CK24 Type Byte (M2/WMO indicator?) ===");
                var typeByteGroups = pm4.Surfaces.Where(s => s.CK24 != 0)
                    .GroupBy(s => (s.CK24 >> 16) & 0xFF)
                    .OrderByDescending(g => g.Count());
                
                foreach (var tbg in typeByteGroups.Take(10))
                {
                    var groupKeyDist = tbg.GroupBy(s => s.GroupKey)
                        .Select(g => $"GK{g.Key}={g.Count()}")
                        .Take(3);
                    Console.WriteLine($"  Type 0x{tbg.Key:X2}: {tbg.Count()} surfaces, GroupKeys: {string.Join(", ", groupKeyDist)}");
                }
                
                // === DEEP CK24 BIT STRUCTURE ANALYSIS ===
                Console.WriteLine("\n=== CK24 Deep Bit Structure Analysis ===");
                
                // Get all non-zero CK24 values
                var allCk24 = pm4.Surfaces.Where(s => s.CK24 != 0).Select(s => s.CK24).Distinct().OrderBy(x => x).ToList();
                Console.WriteLine($"  Non-zero CK24 values: {allCk24.Count}");
                
                // Try different bit interpretations
                Console.WriteLine("\n  Interpretation 1: [Type:8][ObjectID:16]");
                var byType8 = allCk24.GroupBy(ck => (ck >> 16) & 0xFF);
                foreach (var g in byType8.OrderByDescending(g => g.Count()))
                {
                    Console.WriteLine($"    Type 0x{g.Key:X2}: {g.Count()} unique ObjectIDs");
                }
                
                Console.WriteLine("\n  Interpretation 2: [Type:8][GroupA:8][GroupB:8]");
                var parsed = allCk24.Select(ck => new {
                    CK24 = ck,
                    Byte2 = (ck >> 16) & 0xFF,
                    Byte1 = (ck >> 8) & 0xFF,
                    Byte0 = ck & 0xFF
                }).ToList();
                
                var uniqueByte2 = parsed.Select(p => p.Byte2).Distinct().Count();
                var uniqueByte1 = parsed.Select(p => p.Byte1).Distinct().Count();
                var uniqueByte0 = parsed.Select(p => p.Byte0).Distinct().Count();
                Console.WriteLine($"    Unique Byte2 (top): {uniqueByte2}");
                Console.WriteLine($"    Unique Byte1 (mid): {uniqueByte1}");
                Console.WriteLine($"    Unique Byte0 (low): {uniqueByte0}");
                
                // Check if lower bytes are sequential or follow patterns
                Console.WriteLine("\n  Byte distribution within largest CK24 type (0x43):");
                var type43 = parsed.Where(p => p.Byte2 == 0x43).ToList();
                if (type43.Count > 0)
                {
                    var byte1Dist = type43.GroupBy(p => p.Byte1).OrderByDescending(g => g.Count()).Take(5);
                    Console.WriteLine($"    Byte1 distribution:");
                    foreach (var g in byte1Dist)
                        Console.WriteLine($"      0x{g.Key:X2}: {g.Count()} CK24 values");
                    
                    var byte0Dist = type43.GroupBy(p => p.Byte0).OrderByDescending(g => g.Count()).Take(5);
                    Console.WriteLine($"    Byte0 distribution:");
                    foreach (var g in byte0Dist)
                        Console.WriteLine($"      0x{g.Key:X2}: {g.Count()} CK24 values");
                }
                
                // Check surface distribution WITHIN a single CK24
                Console.WriteLine("\n  Surface count per CK24 (top 10):");
                var surfPerCk24 = pm4.Surfaces.Where(s => s.CK24 != 0).GroupBy(s => s.CK24).OrderByDescending(g => g.Count()).Take(10);
                foreach (var g in surfPerCk24)
                {
                    uint ck24 = g.Key;
                    Console.WriteLine($"    CK24 0x{ck24:X6}: {g.Count()} surfaces, Byte2=0x{(ck24>>16)&0xFF:X2}, Byte1=0x{(ck24>>8)&0xFF:X2}, Byte0=0x{ck24&0xFF:X2}");
                }
                
                // Check: Does MsviFirstIndex group surfaces?
                Console.WriteLine("\n  === MSVI Grouping Within CK24 ===");
                var largestCk24Surfaces = pm4.Surfaces.Where(s => s.CK24 != 0).GroupBy(s => s.CK24).OrderByDescending(g => g.Count()).First();
                Console.WriteLine($"  Analyzing largest CK24: 0x{largestCk24Surfaces.Key:X6} ({largestCk24Surfaces.Count()} surfaces)");
                
                // Check if MsviFirstIndex is contiguous or has gaps
                var msviIndices = largestCk24Surfaces.Select(s => (int)s.MsviFirstIndex).OrderBy(x => x).ToList();
                int msviGaps = 0;
                int msviMaxGap = 0;
                for (int i = 1; i < msviIndices.Count; i++)
                {
                    int gap = msviIndices[i] - msviIndices[i-1];
                    if (gap > 10) 
                    {
                        msviGaps++;
                        msviMaxGap = Math.Max(msviMaxGap, gap);
                    }
                }
                Console.WriteLine($"    MSVI index range: {msviIndices.First()} - {msviIndices.Last()}");
                Console.WriteLine($"    Large gaps (>10 indices): {msviGaps}");
                Console.WriteLine($"    Max gap: {msviMaxGap}");
                Console.WriteLine($"    (Large gaps could indicate object boundaries!)");
                
                // MSLK Analysis - The connector chunk!
                Console.WriteLine("\n=== MSLK Analysis (The Connector Chunk!) ===");
                Console.WriteLine($"  Total entries: {pm4.LinkEntries.Count}");
                
                // TypeFlags distribution
                var byTypeFlags = pm4.LinkEntries.GroupBy(e => e.TypeFlags).OrderByDescending(g => g.Count());
                Console.WriteLine("\n  TypeFlags distribution:");
                foreach (var g in byTypeFlags.Take(8))
                {
                    Console.WriteLine($"    Type {g.Key,2}: {g.Count(),5} entries ({100.0 * g.Count() / pm4.LinkEntries.Count:F1}%)");
                }
                
                // Subtype distribution (floor level?)
                var bySubtype = pm4.LinkEntries.GroupBy(e => e.Subtype).OrderBy(g => g.Key);
                Console.WriteLine("\n  Subtype distribution (floor level hypothesis):");
                foreach (var g in bySubtype.Take(10))
                {
                    Console.WriteLine($"    Subtype {g.Key,2}: {g.Count(),5} entries");
                }
                
                // MSPI index analysis
                int hasGeometry = pm4.LinkEntries.Count(e => e.MspiFirstIndex >= 0);
                int noGeometry = pm4.LinkEntries.Count(e => e.MspiFirstIndex < 0);
                Console.WriteLine($"\n  Geometry linkage (MSPI):");
                Console.WriteLine($"    Has geometry (MspiFirst >= 0): {hasGeometry} ({100.0 * hasGeometry / pm4.LinkEntries.Count:F1}%)");
                Console.WriteLine($"    No geometry (MspiFirst = -1): {noGeometry}");
                
                // RefIndex analysis - same dual-index pattern as MPRR!
                Console.WriteLine($"\n  RefIndex analysis (dual-index like MPRR!):");
                int refMprl = pm4.LinkEntries.Count(e => e.RefIndex < pm4.PositionRefs.Count);
                int refMsvt = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.PositionRefs.Count && e.RefIndex < pm4.MeshVertices.Count);
                int refOutOfRange = pm4.LinkEntries.Count(e => e.RefIndex >= pm4.MeshVertices.Count);
                Console.WriteLine($"    RefIndex → MPRL (< {pm4.PositionRefs.Count}): {refMprl}");
                Console.WriteLine($"    RefIndex → MSVT ({pm4.PositionRefs.Count}-{pm4.MeshVertices.Count}): {refMsvt}");
                Console.WriteLine($"    RefIndex out of range: {refOutOfRange}");
                
                // GroupObjectId analysis
                var uniqueGroupIds = pm4.LinkEntries.Select(e => e.GroupObjectId).Distinct().Count();
                Console.WriteLine($"\n  GroupObjectId:");
                Console.WriteLine($"    Unique values: {uniqueGroupIds}");
                var topGroupIds = pm4.LinkEntries.GroupBy(e => e.GroupObjectId).OrderByDescending(g => g.Count()).Take(5);
                Console.WriteLine("    Top 5 by count:");
                foreach (var g in topGroupIds)
                {
                    Console.WriteLine($"      GroupId 0x{g.Key:X8}: {g.Count()} entries");
                }
                
                // GroupObjectId pattern analysis
                Console.WriteLine("\n  === GroupObjectId Graph Analysis ===");
                var sortedGroupIds = pm4.LinkEntries.Select(e => e.GroupObjectId).Distinct().OrderBy(x => x).ToList();
                Console.WriteLine($"    Total unique: {sortedGroupIds.Count}");
                Console.WriteLine($"    Min GroupId: 0x{sortedGroupIds.First():X8}");
                Console.WriteLine($"    Max GroupId: 0x{sortedGroupIds.Last():X8}");
                
                // Check if sequential
                int gaps = 0;
                for (int i = 1; i < sortedGroupIds.Count; i++)
                {
                    if (sortedGroupIds[i] - sortedGroupIds[i-1] > 1) gaps++;
                }
                Console.WriteLine($"    Sequential gaps: {gaps} (0 = perfectly sequential)");
                
                // Check edge-like structure (pairs of entries per GroupId)
                var groupSizes = pm4.LinkEntries.GroupBy(e => e.GroupObjectId).Select(g => g.Count()).ToList();
                var avgSize = groupSizes.Average();
                var sizes = groupSizes.GroupBy(s => s).OrderByDescending(g => g.Count()).Take(5);
                Console.WriteLine($"    Avg entries per GroupId: {avgSize:F1}");
                Console.Write("    Size distribution: ");
                Console.WriteLine(string.Join(", ", sizes.Select(s => $"{s.Key}x={s.Count()}")));
                
                // Deep TypeFlags analysis
                Console.WriteLine("\n  === MSLK TypeFlags Deep Analysis ===");
                foreach (var typeGroup in byTypeFlags.Take(5))
                {
                    Console.WriteLine($"\n    Type {typeGroup.Key}:");
                    
                    // Has geometry?
                    int withGeom = typeGroup.Count(e => e.MspiFirstIndex >= 0);
                    Console.WriteLine($"      Has geometry: {withGeom} / {typeGroup.Count()} ({100.0 * withGeom / typeGroup.Count():F0}%)");
                    
                    // RefIndex target
                    int toMprl = typeGroup.Count(e => e.RefIndex < pm4.PositionRefs.Count);
                    int toMsvt = typeGroup.Count(e => e.RefIndex >= pm4.PositionRefs.Count);
                    Console.WriteLine($"      RefIndex→MPRL: {toMprl} ({100.0 * toMprl / typeGroup.Count():F0}%)");
                    Console.WriteLine($"      RefIndex→MSVT: {toMsvt} ({100.0 * toMsvt / typeGroup.Count():F0}%)");
                    
                    // Floor levels for this type
                    var floors = typeGroup.GroupBy(e => e.Subtype).OrderBy(g => g.Key);
                    Console.Write("      Floor distribution: ");
                    Console.WriteLine(string.Join(", ", floors.Select(f => $"L{f.Key}={f.Count()}")));
                    
                    // MSPI count (path vertices)
                    var avgMspi = typeGroup.Where(e => e.MspiFirstIndex >= 0).Select(e => (int)e.MspiIndexCount).DefaultIfEmpty().Average();
                    Console.WriteLine($"      Avg MSPI count: {avgMspi:F1}");
                }
                
                // Sample MSLK entries showing linkages
                Console.WriteLine("\n  Sample MSLK entries (showing full linkage):");
                foreach (var entry in pm4.LinkEntries.Where(e => e.MspiFirstIndex >= 0).Take(3))
                {
                    Console.WriteLine($"    MSLK: Type={entry.TypeFlags}, Subtype={entry.Subtype}, GroupId=0x{entry.GroupObjectId:X8}");
                    Console.WriteLine($"      → MSPI[{entry.MspiFirstIndex}..{entry.MspiFirstIndex + entry.MspiIndexCount - 1}] ({entry.MspiIndexCount} entries)");
                    
                    // Follow MSPI → MSPV chain
                    if (entry.MspiFirstIndex >= 0 && entry.MspiFirstIndex < pm4.PathIndices.Count)
                    {
                        var firstMspv = pm4.PathIndices[entry.MspiFirstIndex];
                        if (firstMspv < pm4.PathVertices.Count)
                        {
                            var pv = pm4.PathVertices[(int)firstMspv];
                            Console.WriteLine($"      → MSPI[{entry.MspiFirstIndex}] = {firstMspv} → MSPV vertex ({pv.X:F1}, {pv.Y:F1}, {pv.Z:F1})");
                        }
                    }
                    
                    // Follow RefIndex
                    if (entry.RefIndex < pm4.PositionRefs.Count)
                    {
                        var mprl = pm4.PositionRefs[entry.RefIndex];
                        Console.WriteLine($"      → RefIndex {entry.RefIndex} → MPRL pos ({mprl.PositionX:F1}, {mprl.PositionY:F1}, {mprl.PositionZ:F1})");
                    }
                    else if (entry.RefIndex < pm4.MeshVertices.Count)
                    {
                        var mv = pm4.MeshVertices[entry.RefIndex];
                        Console.WriteLine($"      → RefIndex {entry.RefIndex} → MSVT vertex ({mv.X:F1}, {mv.Y:F1}, {mv.Z:F1})");
                    }
                }
                Console.WriteLine();
                
                // MSPV Portal Analysis
                Console.WriteLine("  === MSPV Path Vertex Analysis ===");
                var mspvQuads = new List<(Vector3[] verts, float width, float height)>();
                
                foreach (var entry in pm4.LinkEntries.Where(e => e.MspiFirstIndex >= 0 && e.MspiIndexCount == 4).Take(100))
                {
                    var verts = new List<Vector3>();
                    for (int i = 0; i < 4; i++)
                    {
                        int mspiIdx = entry.MspiFirstIndex + i;
                        if (mspiIdx < pm4.PathIndices.Count)
                        {
                            uint mspvIdx = pm4.PathIndices[mspiIdx];
                            if (mspvIdx < pm4.PathVertices.Count)
                            {
                                verts.Add(pm4.PathVertices[(int)mspvIdx]);
                            }
                        }
                    }
                    if (verts.Count == 4)
                    {
                        // Calculate quad dimensions
                        float minX = verts.Min(v => v.X), maxX = verts.Max(v => v.X);
                        float minY = verts.Min(v => v.Y), maxY = verts.Max(v => v.Y);
                        float minZ = verts.Min(v => v.Z), maxZ = verts.Max(v => v.Z);
                        float width = maxX - minX;
                        float height = maxZ - minZ;
                        float depth = maxY - minY;
                        mspvQuads.Add((verts.ToArray(), Math.Max(width, depth), height));
                    }
                }
                
                if (mspvQuads.Count > 0)
                {
                    var avgWidth = mspvQuads.Average(q => q.width);
                    var avgHeight = mspvQuads.Average(q => q.height);
                    var zeroWidth = mspvQuads.Count(q => q.width < 0.1f);
                    var zeroHeight = mspvQuads.Count(q => q.height < 0.1f);
                    
                    Console.WriteLine($"    Analyzed {mspvQuads.Count} quads (MSLK with 4 MSPI entries)");
                    Console.WriteLine($"    Average width (XY): {avgWidth:F1} units");
                    Console.WriteLine($"    Average height (Z): {avgHeight:F1} units");
                    Console.WriteLine($"    Flat quads (height≈0): {zeroHeight} ({100.0 * zeroHeight / mspvQuads.Count:F0}%)");
                    Console.WriteLine($"    Line quads (width≈0): {zeroWidth} ({100.0 * zeroWidth / mspvQuads.Count:F0}%)");
                    
                    // Sample some quads
                    Console.WriteLine("\n    Sample MSPV quads:");
                    foreach (var (verts, w, h) in mspvQuads.Take(3))
                    {
                        Console.WriteLine($"      Quad: w={w:F1} h={h:F1}");
                        foreach (var v in verts)
                            Console.WriteLine($"        ({v.X:F1}, {v.Y:F1}, {v.Z:F1})");
                    }
                }
                Console.WriteLine();
                
                // Analyze CK24=0 surfaces
                if (surfacesByCk24.TryGetValue(0, out var zeroSurfaces))
                {
                    Console.WriteLine("=== CK24=0x000000 Analysis ===");
                    
                    // Group by GroupKey
                    var byGroupKey = zeroSurfaces.GroupBy(s => s.GroupKey).OrderBy(g => g.Key);
                    foreach (var group in byGroupKey)
                    {
                        Console.WriteLine($"  GroupKey={group.Key}: {group.Count()} surfaces");
                    }
                    
                    // Check geometry - are these portals?
                    var indexCounts = zeroSurfaces.GroupBy(s => s.IndexCount).OrderBy(g => g.Key);
                    Console.WriteLine("\n  Index counts (triangles per surface):");
                    foreach (var ic in indexCounts)
                    {
                        Console.WriteLine($"    {ic.Key} indices: {ic.Count()} surfaces");
                    }
                    
                    // Compute total area of CK24=0 surfaces
                    float totalArea = 0f;
                    foreach (var surf in zeroSurfaces)
                    {
                        var verts = GetSurfaceVertices(pm4, surf);
                        if (verts.Count >= 3)
                        {
                            var bounds = ComputeBounds(verts);
                            totalArea += (bounds.max.X - bounds.min.X) * (bounds.max.Y - bounds.min.Y);
                        }
                    }
                    Console.WriteLine($"\n  Approximate total area covered: {totalArea:N0} sq units");
                    
                    // Sample some vertices from CK24=0
                    Console.WriteLine("\n  Sample geometry (first 5 surfaces):");
                    foreach (var surf in zeroSurfaces.Take(5))
                    {
                        var verts = GetSurfaceVertices(pm4, surf);
                        if (verts.Count > 0)
                        {
                            var bounds = ComputeBounds(verts);
                            Console.WriteLine($"    Surface: {surf.IndexCount} indices, Normal=({surf.NormalX:F2}, {surf.NormalY:F2}, {surf.NormalZ:F2}), Height={surf.Height:F1}");
                            Console.WriteLine($"      Bounds: ({bounds.min.X:F1}, {bounds.min.Y:F1}, {bounds.min.Z:F1}) to ({bounds.max.X:F1}, {bounds.max.Y:F1}, {bounds.max.Z:F1})");
                            Console.WriteLine($"      Size: {bounds.max.X - bounds.min.X:F1} x {bounds.max.Y - bounds.min.Y:F1} x {bounds.max.Z - bounds.min.Z:F1}");
                        }
                    }
                    
                    // Check: Are CK24=0 surfaces part of the navigation mesh or are they distinct objects?
                    Console.WriteLine("\n  CK24=0 Size Distribution:");
                    int tiny = 0, small = 0, medium = 0, large = 0;
                    float maxSizeX = 0, maxSizeY = 0;
                    MsurEntry? largestSurf = null;
                    
                    foreach (var surf in zeroSurfaces)
                    {
                        var verts = GetSurfaceVertices(pm4, surf);
                        if (verts.Count >= 3)
                        {
                            var bounds = ComputeBounds(verts);
                            float sizeX = bounds.max.X - bounds.min.X;
                            float sizeY = bounds.max.Y - bounds.min.Y;
                            float size = Math.Max(sizeX, sizeY);
                            
                            if (size < 5) tiny++;
                            else if (size < 20) small++;
                            else if (size < 50) medium++;
                            else large++;
                            
                            if (sizeX > maxSizeX || sizeY > maxSizeY)
                            {
                                maxSizeX = Math.Max(maxSizeX, sizeX);
                                maxSizeY = Math.Max(maxSizeY, sizeY);
                                largestSurf = surf;
                            }
                        }
                    }
                    
                    Console.WriteLine($"    Tiny (<5 units): {tiny}");
                    Console.WriteLine($"    Small (5-20 units): {small}");
                    Console.WriteLine($"    Medium (20-50 units): {medium}");
                    Console.WriteLine($"    Large (>50 units): {large}");
                    Console.WriteLine($"    Max size seen: {maxSizeX:F1} x {maxSizeY:F1}");
                    
                    if (largestSurf != null)
                    {
                        var lv = GetSurfaceVertices(pm4, largestSurf);
                        var lb = ComputeBounds(lv);
                        Console.WriteLine($"    Largest surface: Normal=({largestSurf.NormalX:F2}, {largestSurf.NormalY:F2}, {largestSurf.NormalZ:F2})");
                        Console.WriteLine($"      At position: ({lb.min.X:F1}, {lb.min.Y:F1}, {lb.min.Z:F1})");
                    }
                    
                    // Check for non-horizontal surfaces (walls)
                    var verticalSurfs = zeroSurfaces.Where(s => Math.Abs(s.NormalZ) < 0.5f).ToList();
                    if (verticalSurfs.Count > 0)
                    {
                        Console.WriteLine($"\n  ⚠️ Found {verticalSurfs.Count} vertical/wall surfaces in CK24=0!");
                        Console.WriteLine("    These could be actual WMO geometry being lost.");
                        foreach (var vs in verticalSurfs.Take(3))
                        {
                            Console.WriteLine($"      Normal=({vs.NormalX:F2}, {vs.NormalY:F2}, {vs.NormalZ:F2}), GroupKey={vs.GroupKey}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("\n    All CK24=0 surfaces are horizontal - likely navigation mesh only.");
                    }
                }
                
                // List CK24!=0 objects (actual WMOs)
                Console.WriteLine("\n=== CK24!=0 Objects (WMOs) ===");
                var nonZeroCk24 = surfacesByCk24.Where(kv => kv.Key != 0)
                    .OrderByDescending(kv => kv.Value.Count)
                    .Take(15);
                
                foreach (var (ck24, surfaces) in nonZeroCk24)
                {
                    var allVerts = new List<Vector3>();
                    foreach (var s in surfaces)
                    {
                        allVerts.AddRange(GetSurfaceVertices(pm4, s));
                    }
                    
                    if (allVerts.Count > 0)
                    {
                        var bounds = ComputeBounds(allVerts);
                        float sizeX = bounds.max.X - bounds.min.X;
                        float sizeY = bounds.max.Y - bounds.min.Y;
                        float sizeZ = bounds.max.Z - bounds.min.Z;
                        Console.WriteLine($"  CK24 0x{ck24:X6}: {surfaces.Count,5} surfaces, Size: {sizeX:F0}x{sizeY:F0}x{sizeZ:F0}, Pos: ({bounds.min.X:F0}, {bounds.min.Y:F0}, {bounds.min.Z:F0})");
                    }
                    else
                    {
                        Console.WriteLine($"  CK24 0x{ck24:X6}: {surfaces.Count,5} surfaces (no vertices)");
                    }
                }
                
                if (surfacesByCk24.Count > 16)
                {
                    Console.WriteLine($"  ... and {surfacesByCk24.Count - 16} more objects");
                }
                
                // === INSTANCE DETECTION WITHIN CK24 ===
                Console.WriteLine("\n=== Instance Detection Analysis ===");
                Console.WriteLine("  Checking if CK24 contains multiple spatially-separated instances...\n");
                
                // Pick the largest non-zero CK24 and check for spatial clusters
                var largestCk24 = surfacesByCk24.Where(kv => kv.Key != 0).OrderByDescending(kv => kv.Value.Count).FirstOrDefault();
                if (largestCk24.Value != null && largestCk24.Value.Count > 0)
                {
                    Console.WriteLine($"  Analyzing CK24 0x{largestCk24.Key:X6} ({largestCk24.Value.Count} surfaces):");
                    
                    // Get centroid of each surface
                    var surfaceCentroids = new List<(MsurEntry surf, Vector3 centroid)>();
                    foreach (var s in largestCk24.Value)
                    {
                        var verts = GetSurfaceVertices(pm4, s);
                        if (verts.Count > 0)
                        {
                            var centroid = new Vector3(
                                verts.Average(v => v.X),
                                verts.Average(v => v.Y),
                                verts.Average(v => v.Z)
                            );
                            surfaceCentroids.Add((s, centroid));
                        }
                    }
                    
                    // Simple clustering: find distinct XY position groups (>20 units apart)
                    var clusters = new List<List<(MsurEntry surf, Vector3 centroid)>>();
                    float clusterDistance = 20f;
                    
                    foreach (var sc in surfaceCentroids)
                    {
                        var nearestCluster = clusters.FirstOrDefault(c => 
                            c.Any(cs => Vector2.Distance(new Vector2(cs.centroid.X, cs.centroid.Y), 
                                                          new Vector2(sc.centroid.X, sc.centroid.Y)) < clusterDistance));
                        if (nearestCluster != null)
                        {
                            nearestCluster.Add(sc);
                        }
                        else
                        {
                            clusters.Add(new List<(MsurEntry, Vector3)> { sc });
                        }
                    }
                    
                    Console.WriteLine($"    Found {clusters.Count} spatial clusters (separation: {clusterDistance} units)");
                    
                    // Show cluster info
                    foreach (var (cluster, idx) in clusters.OrderByDescending(c => c.Count).Take(5).Select((c, i) => (c, i)))
                    {
                        var clusterBounds = ComputeBounds(cluster.Select(c => c.centroid).ToList());
                        Console.WriteLine($"    Cluster {idx + 1}: {cluster.Count} surfaces at ({clusterBounds.min.X:F0}, {clusterBounds.min.Y:F0})");
                    }
                    
                    if (clusters.Count > 5)
                        Console.WriteLine($"    ... and {clusters.Count - 5} more clusters");
                    
                    Console.WriteLine($"\n    ⚡ If clusters > 1, CK24 contains MULTIPLE object instances!");
                }
                
                // === MSUR.MdosIndex INVESTIGATION (potential instance ID?) ===
                Console.WriteLine("\n=== MSUR.MdosIndex Investigation ===");
                var mdosValues = pm4.Surfaces.GroupBy(s => s.MdosIndex).OrderByDescending(g => g.Count());
                Console.WriteLine($"  Unique MdosIndex values: {mdosValues.Count()}");
                Console.WriteLine("  Top 10 MdosIndex values:");
                foreach (var g in mdosValues.Take(10))
                {
                    var ck24s = g.Select(s => s.CK24).Distinct().Count();
                    Console.WriteLine($"    MdosIndex 0x{g.Key:X8}: {g.Count()} surfaces, {ck24s} unique CK24s");
                }
                
                // Check if MdosIndex could be instance ID
                var nonZeroMdos = pm4.Surfaces.Where(s => s.MdosIndex != 0).ToList();
                Console.WriteLine($"\n  Non-zero MdosIndex: {nonZeroMdos.Count} / {pm4.Surfaces.Count}");
                
                if (nonZeroMdos.Count > 0)
                {
                    // Check if MdosIndex subdivides within CK24
                    var largestCk24Surfs = surfacesByCk24.Where(kv => kv.Key != 0).OrderByDescending(kv => kv.Value.Count).First();
                    var mdosWithinCk24 = largestCk24Surfs.Value.GroupBy(s => s.MdosIndex).Count();
                    Console.WriteLine($"  MdosIndex unique values within largest CK24: {mdosWithinCk24}");
                    Console.WriteLine($"    (If > 1, MdosIndex might separate instances within CK24!)");
                }
                
                // === MPRR GROUP → GEOMETRY TRACING ===
                Console.WriteLine("\n=== MPRR Group → Geometry Tracing ===");
                
                // === MSLK → MSVT → MSUR Connection Analysis ===
                Console.WriteLine("\n=== MSLK → MSVT → MSUR Connection Analysis ===");
                Console.WriteLine("  Attempting to link MSLK.GroupObjectId to MSUR surfaces via shared MSVT vertices...\n");
                
                // Build a map: MSVT index → which GroupObjectIds reference it
                var msvtToGroupIds = new Dictionary<int, HashSet<uint>>();
                foreach (var lnk in pm4.LinkEntries)
                {
                    // RefIndex >= MPRL.Count means it points to MSVT directly!
                    // The value IS the MSVT index (no offset subtraction needed)
                    if (lnk.RefIndex >= pm4.PositionRefs.Count && lnk.RefIndex < pm4.PositionRefs.Count + pm4.MeshVertices.Count)
                    {
                        int msvtIdx = lnk.RefIndex;  // Try direct index first
                        if (msvtIdx < pm4.MeshVertices.Count)
                        {
                            if (!msvtToGroupIds.ContainsKey(msvtIdx))
                                msvtToGroupIds[msvtIdx] = new HashSet<uint>();
                            msvtToGroupIds[msvtIdx].Add(lnk.GroupObjectId);
                        }
                    }
                }
                Console.WriteLine($"  MSVT vertices referenced by MSLK: {msvtToGroupIds.Count} / {pm4.MeshVertices.Count}");
                
                // Build map: MSUR → which MSVT indices it uses
                var msurToMsvt = new Dictionary<MsurEntry, HashSet<int>>();
                foreach (var surf in pm4.Surfaces)
                {
                    var msvtIndices = new HashSet<int>();
                    for (int i = 0; i < surf.IndexCount; i++)
                    {
                        uint msviIdx = surf.MsviFirstIndex + (uint)i;
                        if (msviIdx < pm4.MeshIndices.Count)
                        {
                            msvtIndices.Add((int)pm4.MeshIndices[(int)msviIdx]);
                        }
                    }
                    msurToMsvt[surf] = msvtIndices;
                }
                
                // For each MSUR, find which GroupObjectIds its vertices belong to
                var msurToGroupIds = new Dictionary<MsurEntry, HashSet<uint>>();
                int surfacesWithGroupId = 0;
                foreach (var (surf, msvtIndices) in msurToMsvt)
                {
                    var groupIds = new HashSet<uint>();
                    foreach (var msvtIdx in msvtIndices)
                    {
                        if (msvtToGroupIds.TryGetValue(msvtIdx, out var gids))
                            groupIds.UnionWith(gids);
                    }
                    if (groupIds.Count > 0)
                    {
                        msurToGroupIds[surf] = groupIds;
                        surfacesWithGroupId++;
                    }
                }
                
                Console.WriteLine($"  MSUR surfaces linked to GroupObjectId: {surfacesWithGroupId} / {pm4.Surfaces.Count} ({100.0 * surfacesWithGroupId / pm4.Surfaces.Count:F1}%)");
                
                // Check within largest CK24 - how many unique GroupObjectIds?
                var largestCk24Surfs2 = pm4.Surfaces.Where(s => s.CK24 != 0).GroupBy(s => s.CK24).OrderByDescending(g => g.Count()).First();
                var groupIdsInLargestCk24 = new HashSet<uint>();
                foreach (var surf in largestCk24Surfs2)
                {
                    if (msurToGroupIds.TryGetValue(surf, out var gids))
                        groupIdsInLargestCk24.UnionWith(gids);
                }
                Console.WriteLine($"\n  Within largest CK24 (0x{largestCk24Surfs2.Key:X6}):");
                Console.WriteLine($"    Surfaces: {largestCk24Surfs2.Count()}");
                Console.WriteLine($"    Unique GroupObjectIds linked: {groupIdsInLargestCk24.Count}");
                Console.WriteLine($"    ⚡ GroupObjectId could be the object instance separator!");
                // Parse MPRR into groups
                var mprrGroups = new List<List<MprrEntry>>();
                var currentGroup = new List<MprrEntry>();
                foreach (var e in pm4.MprrEntries)
                {
                    if (e.Value1 == 0xFFFF)
                    {
                        if (currentGroup.Count > 0) mprrGroups.Add(currentGroup);
                        currentGroup = new List<MprrEntry>();
                    }
                    else
                    {
                        currentGroup.Add(e);
                    }
                }
                if (currentGroup.Count > 0) mprrGroups.Add(currentGroup);
                
                Console.WriteLine($"  Total MPRR groups: {mprrGroups.Count}");
                
                // Sample a few groups
                Console.WriteLine("\n  Sample MPRR groups (first 5):");
                foreach (var (grp, idx) in mprrGroups.Take(5).Select((g, i) => (g, i)))
                {
                    var grpMprlRefs = grp.Count(e => e.Value1 < pm4.PositionRefs.Count);
                    var grpMsvtRefs = grp.Count(e => e.Value1 >= pm4.PositionRefs.Count);
                    Console.WriteLine($"    Group {idx}: {grp.Count} entries ({grpMprlRefs} MPRL, {grpMsvtRefs} MSVT refs)");
                    
                    // If has MPRL refs, show position
                    var firstMprl = grp.FirstOrDefault(e => e.Value1 < pm4.PositionRefs.Count);
                    if (firstMprl != null)
                    {
                        var mprl = pm4.PositionRefs[firstMprl.Value1];
                        float rot = 360f * mprl.Unknown0x04 / 65536f;
                        Console.WriteLine($"      → MPRL pos: ({mprl.PositionX:F0}, {mprl.PositionY:F0}, {mprl.PositionZ:F0}), Rot: {rot:F0}°");
                    }
                }
                
                // Analyze MPRR sentinels
                Console.WriteLine("\n=== MPRR Analysis ===");
                int sentinelCount = pm4.MprrEntries.Count(e => e.Value1 == 0xFFFF);
                Console.WriteLine($"  Total entries: {pm4.MprrEntries.Count}");
                Console.WriteLine($"  Sentinel (0xFFFF) count: {sentinelCount}");
                Console.WriteLine($"  Objects (between sentinels): ~{sentinelCount}");
                
                // Analyze MPRR object sizes (entries between sentinels)
                Console.WriteLine("\n  MPRR Object Sizes (entries between sentinels):");
                var objectSizes = new List<int>();
                int currentSize = 0;
                foreach (var entry in pm4.MprrEntries)
                {
                    if (entry.Value1 == 0xFFFF)
                    {
                        if (currentSize > 0) objectSizes.Add(currentSize);
                        currentSize = 0;
                    }
                    else
                    {
                        currentSize++;
                    }
                }
                if (currentSize > 0) objectSizes.Add(currentSize);
                
                var sizeGroups = objectSizes.GroupBy(s => s switch {
                    1 => "1 entry",
                    2 => "2 entries", 
                    <= 5 => "3-5 entries",
                    <= 10 => "6-10 entries",
                    <= 50 => "11-50 entries",
                    _ => ">50 entries"
                });
                
                foreach (var g in sizeGroups.OrderBy(g => g.First()))
                {
                    Console.WriteLine($"    {g.Key}: {g.Count()} objects");
                }
                
                if (objectSizes.Count > 0)
                {
                    Console.WriteLine($"  Average object size: {objectSizes.Average():F1} entries");
                    Console.WriteLine($"  Max object size: {objectSizes.Max()} entries");
                }
                
                // Compare CK24 count vs MPRR count
                Console.WriteLine("\n  ⚠️ CK24 vs MPRR Comparison:");
                Console.WriteLine($"    CK24 unique objects: {surfacesByCk24.Count}");
                Console.WriteLine($"    MPRR sentinel objects: {sentinelCount}");
                Console.WriteLine($"    Ratio: {(float)sentinelCount / surfacesByCk24.Count:F1}x more MPRR objects!");
                
                // Investigate: Is MPRR a range-record into MPRL?
                Console.WriteLine("\n=== MPRR → MPRL Correlation Analysis ===");
                Console.WriteLine($"  MPRL entries: {pm4.PositionRefs.Count}");
                Console.WriteLine($"  MPRR entries (non-sentinel): {pm4.MprrEntries.Count - sentinelCount}");
                
                // Check if MPRR values look like MPRL indices
                var nonSentinelMprr = pm4.MprrEntries.Where(e => e.Value1 != 0xFFFF).ToList();
                int maxValue1 = nonSentinelMprr.Count > 0 ? nonSentinelMprr.Max(e => e.Value1) : 0;
                int maxValue2 = nonSentinelMprr.Count > 0 ? nonSentinelMprr.Max(e => e.Value2) : 0;
                
                Console.WriteLine($"\n  MPRR Value ranges:");
                Console.WriteLine($"    Value1: 0 - {maxValue1} (MPRL count: {pm4.PositionRefs.Count})");
                Console.WriteLine($"    Value2: 0 - {maxValue2}");
                
                // Check if Value1 could be MPRL index
                int validMprlRefs = nonSentinelMprr.Count(e => e.Value1 < pm4.PositionRefs.Count);
                Console.WriteLine($"\n  Value1 as MPRL index:");
                Console.WriteLine($"    Valid MPRL refs (Value1 < MPRL.Count): {validMprlRefs} / {nonSentinelMprr.Count} ({100.0 * validMprlRefs / Math.Max(1, nonSentinelMprr.Count):F1}%)");
                
                // Check common Value2 values
                var value2Groups = nonSentinelMprr.GroupBy(e => e.Value2).OrderByDescending(g => g.Count()).Take(8);
                Console.WriteLine($"\n  Most common Value2 values:");
                foreach (var g in value2Groups)
                {
                    // Decode bit pattern
                    ushort v = g.Key;
                    Console.WriteLine($"    0x{v:X4} ({v,5}): {g.Count(),5} occurrences  Binary: {Convert.ToString(v, 2).PadLeft(16, '0')}");
                }
                
                // Deep Value2 bit analysis
                Console.WriteLine("\n  === MPRR Value2 Bit Pattern Analysis ===");
                
                // Check for common bit patterns
                var allValue2 = nonSentinelMprr.Select(e => e.Value2).ToList();
                int hasLowByte = allValue2.Count(v => (v & 0xFF) != 0);
                int hasHighByte = allValue2.Count(v => (v >> 8) != 0);
                Console.WriteLine($"    Entries with low byte (bits 0-7) set: {hasLowByte}");
                Console.WriteLine($"    Entries with high byte (bits 8-15) set: {hasHighByte}");
                
                // Try to decode as [HighByte][LowByte]
                var byHighByte = allValue2.GroupBy(v => v >> 8).OrderByDescending(g => g.Count());
                Console.WriteLine("\n    High byte (bits 8-15) distribution:");
                foreach (var g in byHighByte.Take(5))
                {
                    Console.WriteLine($"      HiByte 0x{g.Key:X2}: {g.Count()} entries");
                }
                
                var byLowByte = allValue2.GroupBy(v => v & 0xFF).OrderByDescending(g => g.Count());
                Console.WriteLine("\n    Low byte (bits 0-7) distribution:");
                foreach (var g in byLowByte.Take(5))
                {
                    Console.WriteLine($"      LoByte 0x{g.Key:X2}: {g.Count()} entries");
                }
                
                // Check if Value2 correlates with anything
                Console.WriteLine("\n    Correlating Value2 with Value1 type:");
                var mprlRefs = nonSentinelMprr.Where(e => e.Value1 < pm4.PositionRefs.Count).ToList();
                var msvtRefs = nonSentinelMprr.Where(e => e.Value1 >= pm4.PositionRefs.Count).ToList();
                
                Console.WriteLine($"      When Value1 → MPRL: Value2 != 0 count: {mprlRefs.Count(e => e.Value2 != 0)} / {mprlRefs.Count}");
                Console.WriteLine($"      When Value1 → MSVT: Value2 != 0 count: {msvtRefs.Count(e => e.Value2 != 0)} / {msvtRefs.Count}");
                
                // Decode Value2 high byte - check if it correlates with floor level
                Console.WriteLine("\n    === Value2 High Byte vs MPRL Floor Level ===");
                var v2FloorCorr = mprlRefs
                    .Where(e => e.Value1 < pm4.PositionRefs.Count)
                    .GroupBy(e => (HiByte: (int)(e.Value2 >> 8), Floor: pm4.PositionRefs[e.Value1].Unknown0x14))
                    .OrderByDescending(g => g.Count())
                    .Take(10);
                
                foreach (var g in v2FloorCorr)
                {
                    Console.WriteLine($"      V2 HiByte=0x{g.Key.HiByte:X2}, Floor={g.Key.Floor}: {g.Count()} entries");
                }
                
                // Sample entries with common Value2 patterns
                Console.WriteLine("\n    Sample entries by Value2 type:");
                var sampleV2_0 = nonSentinelMprr.FirstOrDefault(e => e.Value2 == 0);
                var sampleV2_0x0300 = nonSentinelMprr.FirstOrDefault(e => e.Value2 == 0x0300);
                var sampleV2_0x1100 = nonSentinelMprr.FirstOrDefault(e => e.Value2 == 0x1100);
                
                if (sampleV2_0 != null && sampleV2_0.Value1 < pm4.MeshVertices.Count && sampleV2_0.Value2 == 0)
                {
                    var pos = sampleV2_0.Value1 < pm4.PositionRefs.Count 
                        ? $"MPRL ({pm4.PositionRefs[sampleV2_0.Value1].PositionX:F0}, {pm4.PositionRefs[sampleV2_0.Value1].PositionY:F0})"
                        : $"MSVT idx {sampleV2_0.Value1}";
                    Console.WriteLine($"      Value2=0x0000: Value1={sampleV2_0.Value1} → {pos}");
                }
                if (sampleV2_0x0300 != null && sampleV2_0x0300.Value2 == 0x0300 && sampleV2_0x0300.Value1 < pm4.PositionRefs.Count)
                {
                    var mprl = pm4.PositionRefs[sampleV2_0x0300.Value1];
                    Console.WriteLine($"      Value2=0x0300: Value1={sampleV2_0x0300.Value1} → MPRL ({mprl.PositionX:F0}, {mprl.PositionY:F0}), Floor={mprl.Unknown0x14}");
                }
                if (sampleV2_0x1100 != null && sampleV2_0x1100.Value2 == 0x1100 && sampleV2_0x1100.Value1 < pm4.PositionRefs.Count)
                {
                    var mprl = pm4.PositionRefs[sampleV2_0x1100.Value1];
                    Console.WriteLine($"      Value2=0x1100: Value1={sampleV2_0x1100.Value1} → MPRL ({mprl.PositionX:F0}, {mprl.PositionY:F0}), Floor={mprl.Unknown0x14}");
                }
                
                // Sample some MPRR→MPRL lookups
                Console.WriteLine($"\n  Sample MPRR→MPRL lookups (if Value1 = MPRL index):");
                foreach (var entry in nonSentinelMprr.Take(5))
                {
                    if (entry.Value1 < pm4.PositionRefs.Count)
                    {
                        var mprl = pm4.PositionRefs[entry.Value1];
                        Console.WriteLine($"    MPRR[{entry.Value1}, {entry.Value2}] → MPRL pos=({mprl.PositionX:F1}, {mprl.PositionY:F1}, {mprl.PositionZ:F1})");
                    }
                    else
                    {
                        Console.WriteLine($"    MPRR[{entry.Value1}, {entry.Value2}] → (out of range)");
                    }
                }
                
                // Investigate: What are the high Value1 entries referencing?
                var highValue1 = nonSentinelMprr.Where(e => e.Value1 >= pm4.PositionRefs.Count).ToList();
                if (highValue1.Count > 0)
                {
                    Console.WriteLine($"\n  ⚡ High Value1 Investigation (Value1 >= MPRL.Count):");
                    Console.WriteLine($"    Count: {highValue1.Count} entries ({100.0 * highValue1.Count / nonSentinelMprr.Count:F1}%)");
                    Console.WriteLine($"    Value1 range: {highValue1.Min(e => e.Value1)} - {highValue1.Max(e => e.Value1)}");
                    
                    // Check if they could be MSVT indices
                    int validMsvt = highValue1.Count(e => e.Value1 < pm4.MeshVertices.Count);
                    Console.WriteLine($"\n    Could be MSVT indices (Value1 < {pm4.MeshVertices.Count}): {validMsvt} ({100.0 * validMsvt / highValue1.Count:F1}%)");
                    
                    // Check if they could be MSVI indices  
                    int validMsvi = highValue1.Count(e => e.Value1 < pm4.MeshIndices.Count);
                    Console.WriteLine($"    Could be MSVI indices (Value1 < {pm4.MeshIndices.Count}): {validMsvi} ({100.0 * validMsvi / highValue1.Count:F1}%)");
                    
                    // Check if they could be MSLK indices
                    int validMslk = highValue1.Count(e => e.Value1 < pm4.LinkEntries.Count);
                    Console.WriteLine($"    Could be MSLK indices (Value1 < {pm4.LinkEntries.Count}): {validMslk} ({100.0 * validMslk / highValue1.Count:F1}%)");
                    
                    // Check Value2 distribution for high Value1 entries
                    var highV1By2 = highValue1.GroupBy(e => e.Value2).OrderByDescending(g => g.Count()).Take(5);
                    Console.WriteLine($"\n    Value2 distribution for high Value1:");
                    foreach (var g in highV1By2)
                    {
                        Console.WriteLine($"      Value2={g.Key} (0x{g.Key:X4}): {g.Count()}");
                    }
                    
                    // Sample some high Value1 lookups as MSVT
                    Console.WriteLine($"\n    Sample high Value1 as MSVT vertex lookup:");
                    foreach (var e in highValue1.Take(5))
                    {
                        if (e.Value1 < pm4.MeshVertices.Count)
                        {
                            var v = pm4.MeshVertices[e.Value1];
                            Console.WriteLine($"      MPRR[{e.Value1}, {e.Value2}] → MSVT vertex ({v.X:F1}, {v.Y:F1}, {v.Z:F1})");
                        }
                        else
                        {
                            Console.WriteLine($"      MPRR[{e.Value1}, {e.Value2}] → (exceeds MSVT count too)");
                        }
                    }
                }
                
                // MPRL Unknown Fields Investigation
                Console.WriteLine("\n=== MPRL Unknown Fields Investigation ===");
                Console.WriteLine($"  Total MPRL entries: {pm4.PositionRefs.Count}");
                
                // Unknown0x04 - Rotation candidate?
                var unk04Values = pm4.PositionRefs.GroupBy(e => e.Unknown0x04).OrderByDescending(g => g.Count());
                Console.WriteLine("\n  Unknown0x04 (Rotation candidate?):");
                Console.WriteLine($"    Unique values: {unk04Values.Count()}");
                Console.WriteLine("    Top 5:");
                foreach (var g in unk04Values.Take(5))
                {
                    // If it's rotation, might be 0-65535 representing 0-360 degrees
                    float angle = 360.0f * g.Key / 65536.0f;
                    Console.WriteLine($"      0x{g.Key:X4} ({g.Key}) = {angle:F1}°: {g.Count()} entries");
                }
                
                // Unknown0x14 - Floor level
                var unk14Values = pm4.PositionRefs.GroupBy(e => e.Unknown0x14).OrderBy(g => g.Key);
                Console.WriteLine("\n  Unknown0x14 (Floor level):");
                foreach (var g in unk14Values.Take(10))
                {
                    Console.WriteLine($"    Level {g.Key,3}: {g.Count()} entries");
                }
                
                // Unknown0x16 - Entry type
                var unk16Values = pm4.PositionRefs.GroupBy(e => e.Unknown0x16).OrderByDescending(g => g.Count());
                Console.WriteLine("\n  Unknown0x16 (Entry type):");
                foreach (var g in unk16Values)
                {
                    Console.WriteLine($"    0x{g.Key:X4}: {g.Count()} entries ({(g.Key == 0x3FFF ? "TERMINATOR" : "Normal")})");
                }
                
                // Sample some non-terminator MPRL entries
                Console.WriteLine("\n  Sample MPRL entries (non-terminator):");
                foreach (var e in pm4.PositionRefs.Where(p => p.Unknown0x16 != 0x3FFF).Take(5))
                {
                    float angle = 360.0f * e.Unknown0x04 / 65536.0f;
                    Console.WriteLine($"    Pos=({e.PositionX:F1}, {e.PositionY:F1}, {e.PositionZ:F1}), Unk04={e.Unknown0x04} ({angle:F1}°), Floor={e.Unknown0x14}");
                }
                
                // === 0x8000 FLAG INVESTIGATION ===
                Console.WriteLine("\n=== 0x8000 Flag Investigation ===");
                
                // MPRL Unknown0x06
                var mprlUnk06 = pm4.PositionRefs.GroupBy(e => e.Unknown0x06).OrderByDescending(g => g.Count());
                Console.WriteLine("  MPRL.Unknown0x06 distribution:");
                foreach (var g in mprlUnk06)
                {
                    Console.WriteLine($"    0x{g.Key:X4}: {g.Count()} entries ({100.0 * g.Count() / pm4.PositionRefs.Count:F1}%)");
                }
                
                // MSLK SystemFlag
                var mslkSysFlag = pm4.LinkEntries.GroupBy(e => e.SystemFlag).OrderByDescending(g => g.Count());
                Console.WriteLine("\n  MSLK.SystemFlag distribution:");
                foreach (var g in mslkSysFlag)
                {
                    Console.WriteLine($"    0x{g.Key:X4}: {g.Count()} entries ({100.0 * g.Count() / pm4.LinkEntries.Count:F1}%)");
                }
                
                // Are they always the same?
                bool allMprl8000 = pm4.PositionRefs.All(e => e.Unknown0x06 == 0x8000);
                bool allMslk8000 = pm4.LinkEntries.All(e => e.SystemFlag == 0x8000);
                Console.WriteLine($"\n  All MPRL Unknown0x06 == 0x8000? {allMprl8000}");
                Console.WriteLine($"  All MSLK SystemFlag == 0x8000? {allMslk8000}");
                
                // === MSLK LINK_ID BYTES INVESTIGATION ===
                Console.WriteLine("\n=== MSLK LinkId[4] Bytes Investigation ===");
                
                // Check if link_id is all zeros
                int allZeros = pm4.LinkEntries.Count(e => e.LinkId == 0);
                Console.WriteLine($"  All zeros: {allZeros} / {pm4.LinkEntries.Count} ({100.0 * allZeros / pm4.LinkEntries.Count:F1}%)");
                
                // Distribution of non-zero patterns
                var linkIdPatterns = pm4.LinkEntries
                    .GroupBy(e => e.LinkId)
                    .OrderByDescending(g => g.Count())
                    .Take(10);
                Console.WriteLine("\n  Top LinkId patterns:");
                foreach (var g in linkIdPatterns)
                {
                    byte[] bytes = BitConverter.GetBytes(g.Key);
                    Console.WriteLine($"    0x{g.Key:X8} [{bytes[0]:X2} {bytes[1]:X2} {bytes[2]:X2} {bytes[3]:X2}]: {g.Count()} entries");
                }
                
                // Check byte-by-byte
                Console.WriteLine("\n  Byte analysis:");
                for (int b = 0; b < 4; b++)
                {
                    int shift = b * 8;
                    var byteVals = pm4.LinkEntries.Select(e => (byte)((e.LinkId >> shift) & 0xFF)).Distinct().Count();
                    int nonZero = pm4.LinkEntries.Count(e => ((e.LinkId >> shift) & 0xFF) != 0);
                    Console.WriteLine($"    Byte[{b}]: {byteVals} unique values, {nonZero} non-zero ({100.0 * nonZero / pm4.LinkEntries.Count:F1}%)");
                }
                
                // Sample non-zero link_id entries
                var nonZeroLinkId = pm4.LinkEntries.Where(e => e.LinkId != 0).Take(5);
                Console.WriteLine("\n  Sample non-zero LinkId entries:");
                foreach (var e in nonZeroLinkId)
                {
                    byte[] bytes = BitConverter.GetBytes(e.LinkId);
                    Console.WriteLine($"    Type={e.TypeFlags}, Floor={e.Subtype}, GroupId=0x{e.GroupObjectId:X}, LinkId=0x{e.LinkId:X8} [{bytes[0]:X2} {bytes[1]:X2} {bytes[2]:X2} {bytes[3]:X2}]");
                }
                
                // === MDBH/MDOS Analysis (potential MSCN index?) ===
                Console.WriteLine("\n=== MDBH/MDOS Analysis ===");
                if (pm4.MdbhRaw != null)
                {
                    Console.WriteLine($"  MDBH size: {pm4.MdbhRaw.Length} bytes");
                    if (pm4.MdbhRaw.Length >= 4)
                    {
                        int mdbhValue = BitConverter.ToInt32(pm4.MdbhRaw, 0);
                        Console.WriteLine($"  MDBH[0-3] as int32: {mdbhValue}");
                        Console.WriteLine($"  == MSCN.Count? {mdbhValue == pm4.SceneNodes.Count}");
                        Console.WriteLine($"  == MDOS entries? maybe...");
                    }
                    Console.WriteLine($"  Raw hex: {BitConverter.ToString(pm4.MdbhRaw.Take(Math.Min(16, pm4.MdbhRaw.Length)).ToArray())}");
                }
                else
                    Console.WriteLine("  MDBH: NOT PRESENT");
                
                if (pm4.MdosRaw != null)
                {
                    Console.WriteLine($"  MDOS size: {pm4.MdosRaw.Length} bytes");
                    int mdosEntrySize = 8; // Hypothesis: 8 bytes per entry
                    int mdosCount = pm4.MdosRaw.Length / mdosEntrySize;
                    Console.WriteLine($"  Potential entry count (÷8): {mdosCount}");
                    
                    // Check if any field correlates with MSCN
                    Console.WriteLine($"  Sample entries:");
                    for (int i = 0; i < Math.Min(5, mdosCount); i++)
                    {
                        int offset = i * mdosEntrySize;
                        uint val0 = BitConverter.ToUInt32(pm4.MdosRaw, offset);
                        uint val1 = BitConverter.ToUInt32(pm4.MdosRaw, offset + 4);
                        Console.WriteLine($"    [{i}] 0x{val0:X8} 0x{val1:X8}");
                    }
                }
                else
                    Console.WriteLine("  MDOS: NOT PRESENT");
                
                // MSCN Scene Nodes Investigation
                Console.WriteLine("\n=== MSCN Scene Nodes Investigation ===");
                Console.WriteLine($"  Total MSCN nodes: {pm4.SceneNodes.Count}");
                
                if (pm4.SceneNodes.Count > 0)
                {
                    // Bounds
                    var minX = pm4.SceneNodes.Min(n => n.X);
                    var maxX = pm4.SceneNodes.Max(n => n.X);
                    var minY = pm4.SceneNodes.Min(n => n.Y);
                    var maxY = pm4.SceneNodes.Max(n => n.Y);
                    var minZ = pm4.SceneNodes.Min(n => n.Z);
                    var maxZ = pm4.SceneNodes.Max(n => n.Z);
                    Console.WriteLine($"\n  Bounds:");
                    Console.WriteLine($"    X: {minX:F1} to {maxX:F1} (range: {maxX - minX:F1})");
                    Console.WriteLine($"    Y: {minY:F1} to {maxY:F1} (range: {maxY - minY:F1})");
                    Console.WriteLine($"    Z: {minZ:F1} to {maxZ:F1} (range: {maxZ - minZ:F1})");
                    
                    // Clustering analysis
                    var uniqueZ = pm4.SceneNodes.Select(n => Math.Round(n.Z, 0)).Distinct().Count();
                    Console.WriteLine($"\n  Z-level clustering:");
                    Console.WriteLine($"    Unique Z values (rounded): {uniqueZ}");
                    
                    // Sample nodes
                    Console.WriteLine("\n  Sample MSCN nodes (first 5):");
                    foreach (var n in pm4.SceneNodes.Take(5))
                    {
                        Console.WriteLine($"    ({n.X:F1}, {n.Y:F1}, {n.Z:F1})");
                    }
                    
                    // Check for tile-edge nodes
                    var tileSize = 533.33333f;
                    int edgeNodes = pm4.SceneNodes.Count(n => 
                        (n.X % tileSize < 1 || n.X % tileSize > tileSize - 1) ||
                        (n.Y % tileSize < 1 || n.Y % tileSize > tileSize - 1));
                    Console.WriteLine($"\n  Tile-edge nodes: {edgeNodes} ({100.0 * edgeNodes / pm4.SceneNodes.Count:F1}%)");
                    
                    // === MSCN INDEX CORRELATION CHECK ===
                    Console.WriteLine("\n  === MSCN Index Correlation Check ===");
                    Console.WriteLine($"  MSCN count: {pm4.SceneNodes.Count}");
                    Console.WriteLine($"  MSVT count: {pm4.MeshVertices.Count}");
                    Console.WriteLine($"  MSVI count: {pm4.MeshIndices.Count}");
                    Console.WriteLine($"  MSPV count: {pm4.PathVertices.Count}");
                    Console.WriteLine($"  MSPI count: {pm4.PathIndices.Count}");
                    
                    // Check if any MSUR field could index MSCN
                    var maxMsviFirst = pm4.Surfaces.Max(s => s.MsviFirstIndex);
                    var maxMdosIndex = pm4.Surfaces.Max(s => s.MdosIndex);
                    Console.WriteLine($"\n  MSUR max MsviFirstIndex: {maxMsviFirst} (MSCN={pm4.SceneNodes.Count})");
                    Console.WriteLine($"  MSUR max MdosIndex: {maxMdosIndex}");
                    
                    // Check if MdosIndex could be MSCN index
                    int mdosInMscnRange = pm4.Surfaces.Count(s => s.MdosIndex < pm4.SceneNodes.Count);
                    Console.WriteLine($"  Surfaces with MdosIndex < MSCN.Count: {mdosInMscnRange} / {pm4.Surfaces.Count}");
                    
                    // Check MSLK RefIndex range vs MSCN
                    var mslkMaxRef = pm4.LinkEntries.Max(e => e.RefIndex);
                    Console.WriteLine($"\n  MSLK max RefIndex: {mslkMaxRef}");
                    Console.WriteLine($"  Could RefIndex index MSCN? {mslkMaxRef < pm4.SceneNodes.Count}");
                    
                    // Vertical vs horizontal MSCN points
                    // Compare MSCN to MSVT - are they different point sets?
                    var msvtSet = new HashSet<(float, float, float)>();
                    foreach (var v in pm4.MeshVertices)
                        msvtSet.Add((MathF.Round(v.X, 1), MathF.Round(v.Y, 1), MathF.Round(v.Z, 1)));
                    
                    int mscnInMsvt = 0;
                    int mscnNotInMsvt = 0;
                    foreach (var n in pm4.SceneNodes)
                    {
                        if (msvtSet.Contains((MathF.Round(n.X, 1), MathF.Round(n.Y, 1), MathF.Round(n.Z, 1))))
                            mscnInMsvt++;
                        else
                            mscnNotInMsvt++;
                    }
                    Console.WriteLine($"\n  MSCN vs MSVT comparison:");
                    Console.WriteLine($"    MSCN points found in MSVT: {mscnInMsvt} ({100.0*mscnInMsvt/pm4.SceneNodes.Count:F1}%)");
                    Console.WriteLine($"    MSCN points NOT in MSVT: {mscnNotInMsvt} ({100.0*mscnNotInMsvt/pm4.SceneNodes.Count:F1}%)");
                    Console.WriteLine($"    ⚡ If NOT in MSVT > 0, MSCN has unique geometry!");
                }
                
                // Detailed Link Analysis
                AnalyzeMdosMscnLink(pm4, Path.Combine(Path.GetDirectoryName(path), "pm4_mscn_correlation"));
            }
            
            return stats;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error reading {path}: {ex.Message}");
            return null;
        }
    }
    
    static int GetEntryCount(string sig, uint size) => sig switch
    {
        "MSLK" => (int)(size / 20),
        "MSUR" => (int)(size / 32),
        "MSVT" or "MSPV" or "MSCN" => (int)(size / 12),
        "MSVI" or "MSPI" => (int)(size / 4),
        "MPRL" => (int)(size / 24),
        "MPRR" => (int)(size / 4),
        "MSHD" => 1,
        "MVER" => 1,
        _ => 0
    };
    
    static List<Vector3> GetSurfaceVertices(Pm4File pm4, MsurEntry surf)
    {
        var verts = new List<Vector3>();
        int start = (int)surf.MsviFirstIndex;
        int count = surf.IndexCount;
        
        if (start < 0 || start + count > pm4.MeshIndices.Count) return verts;
        
        for (int i = 0; i < count; i++)
        {
            uint vidx = pm4.MeshIndices[start + i];
            if (vidx < pm4.MeshVertices.Count)
            {
                verts.Add(pm4.MeshVertices[(int)vidx]);
            }
        }
        return verts;
    }
    
    static void AnalyzeMdosMscnLink(Pm4File pm4, string? outputDir = null)
    {
        Console.WriteLine("\n=== MSCN Linkage Analysis (Field14/MdosIndex) ===");
        
        if (outputDir != null) Directory.CreateDirectory(outputDir);
        
        // ... (existing uniqueness check) ...
        var uniqueMdos = pm4.Surfaces.Select(s => s.MdosIndex).Distinct().Count();
        Console.WriteLine($"Total Surfaces: {pm4.Surfaces.Count}");
        Console.WriteLine($"Unique MdosIndices: {uniqueMdos}");
        
        // Specific check for CK24 groups
        var groups = pm4.Surfaces
            .Where(s => s.CK24 != 0)
            .GroupBy(s => s.CK24)
            .OrderByDescending(g => g.Count())
            .Take(10); // Take top 10

        foreach (var g in groups)
        {
            var minMdos = g.Min(s => s.MdosIndex);
            var maxMdos = g.Max(s => s.MdosIndex);
            var countMdos = g.Select(s => s.MdosIndex).Distinct().Count();
            
            Console.WriteLine($"\nCK24 0x{g.Key:X6}:");
            Console.WriteLine($"  Surfaces: {g.Count()}");
            Console.WriteLine($"  MdosIndex Range: {minMdos} to {maxMdos} (Span: {maxMdos - minMdos})");
            Console.WriteLine($"  Unique Mdos: {countMdos}");
            
            if (minMdos < pm4.SceneNodes.Count && maxMdos < pm4.SceneNodes.Count)
            {
                var points = new List<Vector3>();
                foreach(var s in g)
                    if (s.MdosIndex < pm4.SceneNodes.Count)
                        points.Add(pm4.SceneNodes[(int)s.MdosIndex]);
                
                // Bounds check
                var boundMin = new Vector3(float.MaxValue);
                var boundMax = new Vector3(float.MinValue);
                foreach(var p in points) {
                    boundMin = Vector3.Min(boundMin, p);
                    boundMax = Vector3.Max(boundMax, p);
                }
                Console.WriteLine($"  Referenced MSCN Bounds: {boundMin} to {boundMax}");
                
                // EXPORT OBJ
                if (outputDir != null)
                {
                    string filename = Path.Combine(outputDir, $"CK24_{g.Key:X6}_{g.Count()}surf_MSCN.obj");
                    using (var sw = new StreamWriter(filename))
                    {
                        sw.WriteLine($"# CK24 0x{g.Key:X6} MSCN points via MdosIndex");
                        sw.WriteLine($"# Count: {points.Count}");
                        foreach(var p in points)
                            sw.WriteLine($"v {p.X:F4} {p.Y:F4} {p.Z:F4}");
                        for(int i=1; i<=points.Count; i++)
                            sw.WriteLine($"p {i}"); // Point cloud
                    }
                    Console.WriteLine($"  Exported: {filename}");
                }
            }
        }
    }
        


    
    static (Vector3 min, Vector3 max) ComputeBounds(List<Vector3> verts)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        foreach (var v in verts)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        return (min, max);
    }
}

class Pm4Stats
{
    public string FileName { get; set; } = "";
    public int Ck24ZeroSurfaceCount { get; set; }
    public int Ck24NonZeroSurfaceCount { get; set; }
    public int UniqueCk24Count { get; set; }
}

// =====================================================
// CLEAN PM4 FILE PARSER - One way, no duplicates
// =====================================================

public class Pm4File
{
    public uint Version { get; private set; }
    public MshdHeader? Header { get; private set; }         // MSHD header!
    public List<MslkEntry> LinkEntries { get; } = new();
    public List<Vector3> PathVertices { get; } = new();     // MSPV
    public List<uint> PathIndices { get; } = new();         // MSPI
    public List<Vector3> MeshVertices { get; } = new();     // MSVT
    public List<uint> MeshIndices { get; } = new();         // MSVI
    public List<MsurEntry> Surfaces { get; } = new();       // MSUR
    public List<Vector3> SceneNodes { get; } = new();       // MSCN
    public List<MprlEntry> PositionRefs { get; } = new();   // MPRL
    public List<MprrEntry> MprrEntries { get; } = new();    // MPRR
    
    public Dictionary<string, uint> ChunkSizes { get; } = new();
    public byte[]? MdbhRaw { get; set; }   // MDBH - Destructible Building Header
    public byte[]? MdosRaw { get; set; }   // MDOS - Destructible Object State
    
    public static Pm4File Parse(byte[] data)
    {
        var pm4 = new Pm4File();
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        
        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            // Read chunk signature (reversed on disk)
            var sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes);
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = br.BaseStream.Position;
            
            pm4.ChunkSizes[sig] = size;
            
            switch (sig)
            {
                case "MVER":
                    pm4.Version = br.ReadUInt32();
                    break;
                
                case "MSHD":
                    pm4.Header = MshdHeader.Parse(br, size);
                    break;
                    
                case "MSLK":
                    ReadMslk(br, size, pm4.LinkEntries);
                    break;
                    
                case "MSPV":
                    ReadVectors(br, size, pm4.PathVertices);
                    break;
                    
                case "MSPI":
                    ReadUints(br, size, pm4.PathIndices);
                    break;
                    
                case "MSVT":
                    ReadVectors(br, size, pm4.MeshVertices);
                    break;
                    
                case "MSVI":
                    ReadUints(br, size, pm4.MeshIndices);
                    break;
                    
                case "MSUR":
                    ReadMsur(br, size, pm4.Surfaces);
                    break;
                    
                case "MSCN":
                    ReadVectors(br, size, pm4.SceneNodes);
                    break;
                    
                case "MPRL":
                    ReadMprl(br, size, pm4.PositionRefs);
                    break;
                    
                case "MPRR":
                    ReadMprr(br, size, pm4.MprrEntries);
                    break;
                    
                case "MDBH":
                    // Destructible Building Header - raw capture for analysis
                    pm4.MdbhRaw = br.ReadBytes((int)size);
                    break;
                    
                case "MDOS":
                    // Destructible Object State - raw capture for analysis
                    pm4.MdosRaw = br.ReadBytes((int)size);
                    break;
            }
            
            br.BaseStream.Position = dataStart + size;
        }
        
        return pm4;
    }
    
    static void ReadMslk(BinaryReader br, uint size, List<MslkEntry> list)
    {
        int count = (int)(size / 20);
        for (int i = 0; i < count; i++)
        {
            var entry = new MslkEntry
            {
                TypeFlags = br.ReadByte(),
                Subtype = br.ReadByte(),
                Padding = br.ReadUInt16(),
                GroupObjectId = br.ReadUInt32()
            };
            
            // MSPI first index is 24-bit (3 bytes)
            byte[] b = br.ReadBytes(3);
            int mspiFirst = b[0] | (b[1] << 8) | (b[2] << 16);
            if ((mspiFirst & 0x800000) != 0) mspiFirst |= unchecked((int)0xFF000000);
            entry.MspiFirstIndex = mspiFirst;
            
            entry.MspiIndexCount = br.ReadByte();
            entry.LinkId = br.ReadUInt32();
            entry.RefIndex = br.ReadUInt16();
            entry.SystemFlag = br.ReadUInt16();
            
            list.Add(entry);
        }
    }
    
    static void ReadMsur(BinaryReader br, uint size, List<MsurEntry> list)
    {
        int count = (int)(size / 32);
        for (int i = 0; i < count; i++)
        {
            list.Add(new MsurEntry
            {
                GroupKey = br.ReadByte(),
                IndexCount = br.ReadByte(),
                AttributeMask = br.ReadByte(),
                Padding = br.ReadByte(),
                NormalX = br.ReadSingle(),
                NormalY = br.ReadSingle(),
                NormalZ = br.ReadSingle(),
                Height = br.ReadSingle(),
                MsviFirstIndex = br.ReadUInt32(),
                MdosIndex = br.ReadUInt32(),
                PackedParams = br.ReadUInt32()
            });
        }
    }
    
    static void ReadMprl(BinaryReader br, uint size, List<MprlEntry> list)
    {
        int count = (int)(size / 24);
        for (int i = 0; i < count; i++)
        {
            list.Add(new MprlEntry
            {
                Index = i,
                Unknown0x00 = br.ReadUInt16(),
                Unknown0x02 = br.ReadInt16(),
                Unknown0x04 = br.ReadUInt16(),
                Unknown0x06 = br.ReadUInt16(),
                PositionX = br.ReadSingle(),
                PositionY = br.ReadSingle(),
                PositionZ = br.ReadSingle(),
                Unknown0x14 = br.ReadInt16(),
                Unknown0x16 = br.ReadUInt16()
            });
        }
    }
    
    static void ReadMprr(BinaryReader br, uint size, List<MprrEntry> list)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            list.Add(new MprrEntry(br.ReadUInt16(), br.ReadUInt16()));
        }
    }
    
    static void ReadVectors(BinaryReader br, uint size, List<Vector3> list)
    {
        int count = (int)(size / 12);
        for (int i = 0; i < count; i++)
        {
            float x = br.ReadSingle();
            float y = br.ReadSingle();
            float z = br.ReadSingle();
            list.Add(new Vector3(x, y, z));
        }
    }
    
    static void ReadUints(BinaryReader br, uint size, List<uint> list)
    {
        int count = (int)(size / 4);
        for (int i = 0; i < count; i++)
        {
            list.Add(br.ReadUInt32());
        }
    }
}

// Entry types

public class MslkEntry
{
    public byte TypeFlags { get; set; }
    public byte Subtype { get; set; }
    public ushort Padding { get; set; }
    public uint GroupObjectId { get; set; }
    public int MspiFirstIndex { get; set; }
    public byte MspiIndexCount { get; set; }
    public uint LinkId { get; set; }
    public ushort RefIndex { get; set; }
    public ushort SystemFlag { get; set; }
}

public class MsurEntry
{
    public byte GroupKey { get; set; }
    public byte IndexCount { get; set; }
    public byte AttributeMask { get; set; }
    public byte Padding { get; set; }
    public float NormalX { get; set; }
    public float NormalY { get; set; }
    public float NormalZ { get; set; }
    public float Height { get; set; }
    public uint MsviFirstIndex { get; set; }
    public uint MdosIndex { get; set; }
    public uint PackedParams { get; set; }
    
    public uint CK24 => (PackedParams >> 8) & 0xFFFFFF;
    public byte CK24Type => (byte)(CK24 >> 16);
    public ushort CK24ObjectId => (ushort)(CK24 & 0xFFFF);
}

public class MprlEntry
{
    public int Index { get; set; }
    public ushort Unknown0x00 { get; set; }
    public short Unknown0x02 { get; set; }
    public ushort Unknown0x04 { get; set; }
    public ushort Unknown0x06 { get; set; }
    public float PositionX { get; set; }
    public float PositionY { get; set; }
    public float PositionZ { get; set; }
    public short Unknown0x14 { get; set; }
    public ushort Unknown0x16 { get; set; }
    
    public Vector3 Position => new(PositionX, PositionY, PositionZ);
}

public record MprrEntry(ushort Value1, ushort Value2);

// MSHD Header - 32 bytes
public class MshdHeader
{
    public byte[] RawBytes { get; set; } = new byte[32];
    
    // Interpretations (hypotheses to test)
    public uint Field00 { get; set; }   // 0x00-0x03
    public uint Field04 { get; set; }   // 0x04-0x07
    public uint Field08 { get; set; }   // 0x08-0x0B
    public uint Field0C { get; set; }   // 0x0C-0x0F
    public uint Field10 { get; set; }   // 0x10-0x13
    public uint Field14 { get; set; }   // 0x14-0x17
    public uint Field18 { get; set; }   // 0x18-0x1B
    public uint Field1C { get; set; }   // 0x1C-0x1F
    
    public static MshdHeader Parse(BinaryReader br, uint size)
    {
        var header = new MshdHeader();
        int toRead = Math.Min(32, (int)size);
        header.RawBytes = br.ReadBytes(toRead);
        
        // Parse as 8 uint32s
        using var ms = new MemoryStream(header.RawBytes);
        using var reader = new BinaryReader(ms);
        if (header.RawBytes.Length >= 4) header.Field00 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 8) header.Field04 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 12) header.Field08 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 16) header.Field0C = reader.ReadUInt32();
        if (header.RawBytes.Length >= 20) header.Field10 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 24) header.Field14 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 28) header.Field18 = reader.ReadUInt32();
        if (header.RawBytes.Length >= 32) header.Field1C = reader.ReadUInt32();
        
        return header;
    }
}
