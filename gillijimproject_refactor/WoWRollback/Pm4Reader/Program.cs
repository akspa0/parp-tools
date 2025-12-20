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
            
            foreach (var (tileName, _, _, _) in tiles)
            {
                if (!allTileSurfaces.ContainsKey(tileName)) continue;
                var surfs = allTileSurfaces[tileName].Where(s => s.CK24 == ck24);
                var indices = allTileIndices[tileName];
                var verts = allTileVertices[tileName];
                
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
                        allFaces.Add(faceIndices.ToArray());
                }
            }
            
            if (allVerts.Count > 0)
            {
                // Determine type from high byte of CK24
                int typeByte = (int)((ck24 >> 16) & 0xFF);
                bool isWmo = typeByte == 0x42 || typeByte == 0x43;
                var targetDir = isWmo ? wmoDir : otherDir;
                string typeLabel = isWmo ? "WMO" : "Other";
                
                var objPath = Path.Combine(targetDir, $"CK24_{ck24:X6}_{tiles.Count}tiles.obj");
                using var sw = new StreamWriter(objPath);
                sw.WriteLine($"# Multi-tile CK24 0x{ck24:X6} ({typeLabel})");
                sw.WriteLine($"# Type byte: 0x{typeByte:X2}");
                sw.WriteLine($"# Tiles: {tiles.Count}");
                sw.WriteLine($"# Vertices: {allVerts.Count}");
                sw.WriteLine($"# Faces: {allFaces.Count}");
                sw.WriteLine();
                foreach (var v in allVerts)
                    sw.WriteLine($"v {v.X:F4} {v.Y:F4} {v.Z:F4}");
                sw.WriteLine();
                foreach (var f in allFaces)
                    sw.WriteLine($"f {string.Join(" ", f)}");
                
                Console.WriteLine($"  [{typeLabel}] CK24_{ck24:X6}_{tiles.Count}tiles.obj ({allVerts.Count} verts)");
                exported++;
            }
        }
        
        Console.WriteLine($"\nExported {exported} multi-tile OBJ files to {outputDir}");
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
                }
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
