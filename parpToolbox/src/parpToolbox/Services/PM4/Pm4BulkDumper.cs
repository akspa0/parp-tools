namespace ParpToolbox.Services.PM4;

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;

/// <summary>
/// Writes a comprehensive dump of a <see cref="Pm4Scene"/>: CSV plus OBJ exports for several grouping strategies.
/// Triggered by the CLI flag <c>--bulk-dump</c>.
/// </summary>
internal static class Pm4BulkDumper
{
    public static void Dump(Pm4Scene scene, string outputRoot, bool exportFaces, byte[]? rawMsvtData = null)
    {
        DumpPm4Scene(scene, outputRoot, exportFaces, rawMsvtData).GetAwaiter().GetResult();
    }
    
    public static async Task DumpPm4Scene(Pm4Scene scene, string outputRoot, bool exportFaces, byte[]? rawMsvtData = null)
    {
        Directory.CreateDirectory(outputRoot);
        
        // 1. Write CSV of raw MSUR fields.
        var csvPath = Path.Combine(outputRoot, "msur_dump.csv");
        using (var writer = new StreamWriter(csvPath))
        {
            writer.WriteLine("Idx,SurfaceKey,SurfaceGroupKey,Flags0x00,AttributeMask,MsviFirstIndex,IndexCount,IsM2Bucket");
            int idx = 0;
            foreach (var s in scene.Surfaces)
            {
                writer.WriteLine(string.Join(',',
                    idx++,
                    $"0x{s.SurfaceKey:X8}",
                    s.SurfaceGroupKey,
                    s.FlagsOrUnknown_0x00,
                    $"0x{s.SurfaceAttributeMask:X4}",
                    s.MsviFirstIndex,
                    s.IndexCount,
                    s.IsM2Bucket));
            }
        }
        
        // 2. Write CSV of MSLK fields (focus on unknown fields)
        var mslkCsvPath = Path.Combine(outputRoot, "mslk_dump.csv");
        using (var writer = new StreamWriter(mslkCsvPath))
        {
            writer.WriteLine("Idx,Unknown0x00,Unknown0x01,Unknown0x02,ParentIndex,MspiFirstIndex,MspiIndexCount,LinkIdPadding,LinkIdTileY,LinkIdTileX,TileCoordinate,ReferenceIndex,Unknown0x12,HasGeometry");
            int idx = 0;
            foreach (var link in scene.Links)
            {
                writer.WriteLine(string.Join(',',
                    idx++,
                    $"0x{link.Unknown_0x00:X2}",
                    $"0x{link.Unknown_0x01:X2}",
                    $"0x{link.Unknown_0x02:X4}",
                    $"0x{link.Unknown_0x04:X8}",
                    link.MspiFirstIndex,
                    link.MspiIndexCount,
                    $"0x{link.LinkIdPadding:X4}",
                    link.LinkIdTileY,
                    link.LinkIdTileX,
                    $"0x{link.TileCoordinate:X4}",
                    $"0x{link.Unknown_0x10:X4}",
                    $"0x{link.Unknown_0x12:X4}",
                    link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0));
            }
        }
        
        // 3. Write MPRR properties dump
        var mprrCsvPath = Path.Combine(outputRoot, "mprr_dump.csv");
        using (var writer = new StreamWriter(mprrCsvPath))
        {
            writer.WriteLine("Idx,Value1,Value2");
            int idx = 0;
            foreach (var prop in scene.Properties)
            {
                writer.WriteLine($"{idx++},{prop.Value1},{prop.Value2}");
            }
        }
        
        // 3b. Write MPRL placements dump
        var mprlCsvPath = Path.Combine(outputRoot, "mprl_dump.csv");
        using (var writer = new StreamWriter(mprlCsvPath))
        {
            writer.WriteLine("Idx,Unknown0,Unknown2,Unknown4,Unknown6,PosX,PosY,PosZ,Unknown14,Unknown16");
            int idx = 0;
            foreach (var placement in scene.Placements)
            {
                writer.WriteLine($"{idx++},{placement.Unknown0},{placement.Unknown2},{placement.Unknown4},{placement.Unknown6},{placement.Position.X},{placement.Position.Y},{placement.Position.Z},{placement.Unknown14},{placement.Unknown16}");
            }
        }
        
        // 4. Write MSVT raw chunk analysis (if available)
        if (rawMsvtData != null && rawMsvtData.Length > 0)
        {
            var msvtRawPath = Path.Combine(outputRoot, "msvt_raw_analysis.csv");
            using (var writer = new StreamWriter(msvtRawPath))
            {
                writer.WriteLine("ByteOffset,Value,AsFloat,AsInt,AsUInt,Interpretation");
                
                // Analyze chunk in 4-byte increments
                for (int i = 0; i < rawMsvtData.Length - 3; i += 4)
                {
                    var floatVal = BitConverter.ToSingle(rawMsvtData, i);
                    var intVal = BitConverter.ToInt32(rawMsvtData, i);
                    var uintVal = BitConverter.ToUInt32(rawMsvtData, i);
                    
                    string interpretation = "";
                    if (i % 12 == 0) interpretation += "[X/Y-coord?] ";
                    else if (i % 12 == 4) interpretation += "[Y/X-coord?] ";
                    else if (i % 12 == 8) interpretation += "[Z-coord?] ";
                    
                    if (i % 24 == 12) interpretation += "[Unknown1?] ";
                    else if (i % 24 == 16) interpretation += "[Unknown2?] ";
                    else if (i % 24 == 20) interpretation += "[Unknown3?] ";
                    
                    writer.WriteLine($"0x{i:X4},{rawMsvtData[i]:X2}{rawMsvtData[i+1]:X2}{rawMsvtData[i+2]:X2}{rawMsvtData[i+3]:X2},{floatVal:F6},{intVal},{uintVal},{interpretation}");
                }
                
                // Summary analysis
                writer.WriteLine();
                writer.WriteLine($"# Total bytes: {rawMsvtData.Length}");
                writer.WriteLine($"# Divisible by 12: {rawMsvtData.Length % 12 == 0}");
                writer.WriteLine($"# Divisible by 24: {rawMsvtData.Length % 24 == 0}");
                writer.WriteLine($"# Vertex count (12-byte): {rawMsvtData.Length / 12}");
                writer.WriteLine($"# Vertex count (24-byte): {rawMsvtData.Length / 24}");
                writer.WriteLine($"# Parsed vertex count: {scene.Vertices.Count}");
            }
        }
        
        // Export comprehensive chunk analysis for hierarchical pattern discovery
        await ExportComprehensiveChunkAnalysis(scene, outputRoot);
        
        // 5. Write MSVT vertex analysis
        var msvtCsvPath = Path.Combine(outputRoot, "msvt_vertices.csv");
        using (var writer = new StreamWriter(msvtCsvPath))
        {
            writer.WriteLine("Idx,X,Y,Z,DistanceFromOrigin,IsNearZero");
            for (int i = 0; i < scene.Vertices.Count; i++)
            {
                var vertex = scene.Vertices[i];
                var distance = Math.Sqrt(vertex.X * vertex.X + vertex.Y * vertex.Y + vertex.Z * vertex.Z);
                var isNearZero = Math.Abs(vertex.X) < 0.001f && Math.Abs(vertex.Y) < 0.001f && Math.Abs(vertex.Z) < 0.001f;
                writer.WriteLine($"{i},{vertex.X:F6},{vertex.Y:F6},{vertex.Z:F6},{distance:F6},{isNearZero}");
            }
        }
        
        // 6. Write vertex stats and bounds analysis
        var vertexStatsPath = Path.Combine(outputRoot, "vertex_stats.csv");
        using (var writer = new StreamWriter(vertexStatsPath))
        {
            writer.WriteLine("TotalVertices,TotalIndices,TotalTriangles,MinX,MaxX,MinY,MaxY,MinZ,MaxZ,BoundsWidth,BoundsHeight,BoundsDepth");
            
            if (scene.Vertices.Count > 0)
            {
                var minX = scene.Vertices.Min(v => v.X);
                var maxX = scene.Vertices.Max(v => v.X);
                var minY = scene.Vertices.Min(v => v.Y);
                var maxY = scene.Vertices.Max(v => v.Y);
                var minZ = scene.Vertices.Min(v => v.Z);
                var maxZ = scene.Vertices.Max(v => v.Z);
                
                writer.WriteLine($"{scene.Vertices.Count},{scene.Indices.Count},{scene.Indices.Count / 3},{minX:F6},{maxX:F6},{minY:F6},{maxY:F6},{minZ:F6},{maxZ:F6},{maxX - minX:F6},{maxY - minY:F6},{maxZ - minZ:F6}");
            }
            else
            {
                writer.WriteLine($"{scene.Vertices.Count},{scene.Indices.Count},{scene.Indices.Count / 3},0,0,0,0,0,0,0,0,0");
            }
        }



        // Export OBJ files if requested
        if (exportFaces)
        {
            Console.WriteLine("Exporting OBJ files...");
            
            // Export complete scene as unified building interior
            Pm4SceneExporter.ExportCompleteScene(scene, outputRoot);
            
            // Export objects grouped by MSUR SurfaceKey (component-level)
            var msurObjects = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
            Pm4MsurObjectAssembler.ExportMsurObjects(msurObjects, scene, outputRoot);
            
            // Export objects using MPRR hierarchical grouping (building-level)
            var hierarchicalObjects = Pm4HierarchicalObjectAssembler.AssembleHierarchicalObjects(scene);
            Pm4HierarchicalObjectAssembler.ExportHierarchicalObjects(hierarchicalObjects, scene, outputRoot);
            
            Console.WriteLine("OBJ export completed.");
        }
        
        // Summary of object assembly results
        if (scene.Links.Count > 0 && scene.Surfaces.Count > 0)
        {
            Console.WriteLine($"  Exported MSVT analysis to {outputRoot}");
        }
        
        // 6. Legacy object assembly (for comparison/debugging)
        if (scene.Links.Count > 0 && scene.Placements.Count > 0)
        {
            Console.WriteLine("Legacy assembly for comparison (likely produces fragments)...");
            var assembledObjects = Pm4ObjectAssembler.AssembleObjects(scene);
            Console.WriteLine($"Legacy method found {assembledObjects.Count} object fragments");
        }
        
        // 5. Analyze chunk relationships
        AnalyzeChunkRelationships(scene, outputRoot);
    }

    private static void WriteCsv(string path, IEnumerable<object> items, Func<object, object[]> selector)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine(string.Join(',', selector(items.First()).Select(x => x.ToString())));

        foreach (var item in items)
        {
            writer.WriteLine(string.Join(',', selector(item).Select(x => x.ToString())));
        }
    }

    private static void WriteVertexStats(string path, Pm4Scene scene)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("TotalVertices,TotalIndices,TotalTriangles");
        writer.WriteLine($"{scene.Vertices.Count},{scene.Indices.Count},{scene.Indices.Count / 3}");
    }

    private static Dictionary<uint, List<(int A, int B, int C)>> BuildFaceGroups(Pm4Scene scene, Func<MsurChunk.Entry, uint> keySelector, uint? filterKey = null)
    {
        var groups = new Dictionary<uint, List<(int A, int B, int C)>>();
        var msviIndices = CollectMsviIndices(scene);

        foreach (var surf in scene.Surfaces)
        {
            uint key = keySelector(surf);
            if (filterKey.HasValue && key != filterKey.Value) continue;

            if (!groups.TryGetValue(key, out var faces))
            {
                faces = new List<(int A, int B, int C)>();
                groups[key] = faces;
            }

            int first = (int)surf.MsviFirstIndex;
            int count = surf.IndexCount;
            if (first < 0 || count < 3 || first + count > msviIndices.Count) continue;

            for (int i = 0; i < count; i += 3)
            {
                if (i + 2 < count)
                {
                    int idxA = msviIndices[first + i];
                    int idxB = msviIndices[first + i + 1];
                    int idxC = msviIndices[first + i + 2];

                    if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                    {
                        faces.Add((idxA, idxB, idxC));
                    }
                }
            }
        }

        return groups;
    }

    private static List<int> CollectMsviIndices(Pm4Scene scene)
    {
        // But we also need raw index list for face remap. We can flatten.
        var list = new List<int>();
        foreach (var tri in scene.Triangles)
        {
            list.Add(tri.A);
            list.Add(tri.B);
            list.Add(tri.C);
        }
        return list;
    }

    private static Dictionary<uint, List<(int A, int B, int C)>> BuildParentGroups(Pm4Scene scene)
    {
        var groups = new Dictionary<uint, List<(int A, int B, int C)>>();
        
        Console.WriteLine($"Building parent groups from {scene.Links.Count} MSLK entries...");
        
        // Group MSLK entries by Unknown_0x04 (ParentIndex)
        foreach (var link in scene.Links)
        {
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount <= 0) continue; // Skip entries without geometry
            
            uint parentIndex = link.Unknown_0x04;
            if (!groups.TryGetValue(parentIndex, out var faces))
            {
                faces = new List<(int A, int B, int C)>();
                groups[parentIndex] = faces;
            }
            
            // Extract triangles from the MSPI indices for this link
            int startIndex = link.MspiFirstIndex;
            int count = link.MspiIndexCount;
            
            Console.WriteLine($"  MSLK ParentIndex=0x{parentIndex:X8}: MspiFirstIndex={startIndex}, MspiIndexCount={count}");
            
            if (startIndex + count <= scene.Indices.Count)
            {
                for (int i = 0; i < count; i += 3)
                {
                    if (i + 2 < count)
                    {
                        int idxA = scene.Indices[startIndex + i];
                        int idxB = scene.Indices[startIndex + i + 1];
                        int idxC = scene.Indices[startIndex + i + 2];
                        
                        Console.WriteLine($"    Triangle: {idxA}, {idxB}, {idxC}");
                        
                        if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                        {
                            faces.Add((idxA, idxB, idxC));
                        }
                    }
                }
            }
            else
            {
                Console.WriteLine($"    WARNING: Index range {startIndex}+{count} exceeds scene.Indices.Count={scene.Indices.Count}");
            }
        }
        
        Console.WriteLine($"Exported {groups.Count} parent groups");
        return groups;
    }
    
    private static Dictionary<uint, List<(int A, int B, int C)>> BuildReferenceGroups(Pm4Scene scene)
    {
        var groups = new Dictionary<uint, List<(int A, int B, int C)>>();
        
        // Group MSLK entries by Unknown_0x10 (ReferenceIndex)
        foreach (var link in scene.Links)
        {
            if (link.MspiFirstIndex < 0 || link.MspiIndexCount <= 0) continue; // Skip entries without geometry
            
            uint referenceIndex = link.Unknown_0x10;
            if (!groups.TryGetValue(referenceIndex, out var faces))
            {
                faces = new List<(int A, int B, int C)>();
                groups[referenceIndex] = faces;
            }
            
            // Extract triangles from the MSPI indices for this link
            int startIndex = link.MspiFirstIndex;
            int count = link.MspiIndexCount;
            
            if (startIndex + count <= scene.Indices.Count)
            {
                for (int i = 0; i < count; i += 3)
                {
                    if (i + 2 < count)
                    {
                        int idxA = scene.Indices[startIndex + i];
                        int idxB = scene.Indices[startIndex + i + 1];
                        int idxC = scene.Indices[startIndex + i + 2];
                        
                        if (idxA < scene.Vertices.Count && idxB < scene.Vertices.Count && idxC < scene.Vertices.Count)
                        {
                            faces.Add((idxA, idxB, idxC));
                        }
                    }
                }
            }
        }
        
        return groups;
    }

    private static void WriteObj(string path, IReadOnlyList<Vector3> verts, List<(int A, int B, int C)> faces, bool includeFaces)
    {
        using var sw = new StreamWriter(path);
        sw.WriteLine("# Auto-generated by parpToolbox bulk-dump");
        sw.WriteLine($"# {faces.Count} triangles");

        // 1) Collect vertex indices actually referenced by this OBJ
        var used = new HashSet<int>();
        if (includeFaces)
        {
            foreach (var (A, B, C) in faces)
            {
                if (A < verts.Count) used.Add(A);
                if (B < verts.Count) used.Add(B);
                if (C < verts.Count) used.Add(C);
            }
        }
        else
        {
            // point cloud: fall back to *all* vertices so the OBJ can still be opened
            for (int i = 0; i < verts.Count; i++)
                used.Add(i);
        }

        Console.WriteLine($"Writing OBJ {Path.GetFileName(path)}: {used.Count} vertices, {faces.Count} triangles");

        // 2) Remap to sequential indices starting at 1
        var remap = new Dictionary<int, int>(used.Count);
        int next = 1;
        foreach (var idx in used)
        {
            remap[idx] = next++;
            var v = verts[idx];
            sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
        }

        sw.WriteLine("usemtl default");

        // 3) Faces or points
        if (includeFaces)
        {
            foreach (var (A, B, C) in faces)
            {
                if (remap.TryGetValue(A, out var ra) && remap.TryGetValue(B, out var rb) && remap.TryGetValue(C, out var rc))
                    sw.WriteLine($"f {ra} {rb} {rc}");
            }
        }
        else
        {
            foreach (var pair in remap)
                sw.WriteLine($"p {pair.Value}");
        }
    }

    /// <summary>
    /// Analyzes relationships between MPRL, MPRR, and MSLK chunks to understand PM4 object assembly.
    /// </summary>
    private static void AnalyzeChunkRelationships(Pm4Scene scene, string outputRoot)
    {
        var analysisPath = Path.Combine(outputRoot, "chunk_relationships.csv");
        using var writer = new StreamWriter(analysisPath);
        
        writer.WriteLine("Analysis,Count,Details");
        
        // Basic counts
        writer.WriteLine($"MPRL_Entries,{scene.Placements.Count},Total placement records");
        writer.WriteLine($"MPRR_Entries,{scene.Properties.Count},Total property pairs");
        writer.WriteLine($"MSLK_Entries,{scene.Links.Count},Total link records");
        writer.WriteLine($"MSUR_Entries,{scene.Surfaces.Count},Total surface records");
        
        // MPRL analysis
        if (scene.Placements.Count > 0)
        {
            var mprlUnknown4Values = scene.Placements.Select(p => p.Unknown4).Distinct().OrderBy(x => x).ToList();
            writer.WriteLine($"MPRL_Unknown4_Unique,{mprlUnknown4Values.Count},\"[{string.Join(", ", mprlUnknown4Values.Take(20))}]\"");
            
            var mprlUnknown6Values = scene.Placements.Select(p => p.Unknown6).Distinct().OrderBy(x => x).ToList();
            writer.WriteLine($"MPRL_Unknown6_Unique,{mprlUnknown6Values.Count},\"[{string.Join(", ", mprlUnknown6Values)}]\"");
        }
        
        // MPRR analysis
        if (scene.Properties.Count > 0)
        {
            var mprrValue2Values = scene.Properties.Select(p => p.Value2).Distinct().OrderBy(x => x).ToList();
            writer.WriteLine($"MPRR_Value2_Unique,{mprrValue2Values.Count},\"[{string.Join(", ", mprrValue2Values.Take(20))}]\"");
            
            var sentinelCount = scene.Properties.Count(p => p.Value1 == 65535);
            writer.WriteLine($"MPRR_Sentinel_Count,{sentinelCount},Value1 == 65535 (likely separators)");
        }
        
        // MSLK analysis
        if (scene.Links.Count > 0)
        {
            var parentIndices = scene.Links.Select(l => l.ParentIndex).Distinct().OrderBy(x => x).ToList();
            writer.WriteLine($"MSLK_ParentIndex_Unique,{parentIndices.Count},\"[{string.Join(", ", parentIndices.Take(20))}]\"");
            
            var referenceIndices = scene.Links.Select(l => l.ReferenceIndex).Distinct().OrderBy(x => x).ToList();
            writer.WriteLine($"MSLK_ReferenceIndex_Unique,{referenceIndices.Count},\"[{string.Join(", ", referenceIndices.Take(20))}]\"");
        }
        
        // Cross-chunk relationship analysis
        if (scene.Placements.Count > 0 && scene.Links.Count > 0)
        {
            // Check if MPRL Unknown4 values match MSLK ParentIndex values (cast to uint for comparison)
            var mprlUnknown4Set = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
            var mslkParentSet = scene.Links.Select(l => l.ParentIndex).ToHashSet();
            var intersection = mprlUnknown4Set.Intersect(mslkParentSet).Count();
            writer.WriteLine($"MPRL_MSLK_ParentMatch,{intersection},MPRL.Unknown4 matches MSLK.ParentIndex");
        }
        
        if (scene.Properties.Count > 0 && scene.Placements.Count > 0)
        {
            // Check if MPRR Value1 could be indices into MPRL
            var maxMprrValue1 = scene.Properties.Where(p => p.Value1 != 65535).Max(p => p.Value1);
            var mprlMaxIndex = scene.Placements.Count - 1;
            writer.WriteLine($"MPRR_MPRL_IndexCheck,{maxMprrValue1 <= mprlMaxIndex},MPRR.Value1 max ({maxMprrValue1}) <= MPRL count ({scene.Placements.Count})");
        }
        
        Console.WriteLine($"Chunk relationship analysis written to: {analysisPath}");
    }
    
    /// <summary>
    /// Exports comprehensive analysis of all PM4 chunks to discover hierarchical patterns and cross-references.
    /// </summary>
    private static async Task ExportComprehensiveChunkAnalysis(Pm4Scene scene, string outputRoot)
    {
        Console.WriteLine("  Exporting comprehensive chunk analysis for pattern discovery...");
        
        var analysisDir = Path.Combine(outputRoot, "chunk_analysis");
        Directory.CreateDirectory(analysisDir);
        
        // Export MSLK with all fields for hierarchy analysis
        await ExportMslkDetailedAnalysis(scene, analysisDir);
        
        // Export MPRL with cross-reference analysis
        await ExportMprlDetailedAnalysis(scene, analysisDir);
        
        // Export MPRR with pattern analysis
        await ExportMprrDetailedAnalysis(scene, analysisDir);
        
        // Export MSUR with grouping analysis
        await ExportMsurDetailedAnalysis(scene, analysisDir);
        
        // Export MSCN collision vertex analysis
        await ExportMscnDetailedAnalysis(scene, analysisDir);
        
        // Export cross-reference matrix
        await ExportCrossReferenceMatrix(scene, analysisDir);
        
        Console.WriteLine($"  Exported comprehensive chunk analysis to {analysisDir}");
    }
    
    /// <summary>
    /// Exports detailed MSLK analysis with hierarchy discovery focus.
    /// </summary>
    private static Task ExportMslkDetailedAnalysis(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "mslk_detailed.csv");
        using var writer = new StreamWriter(csvPath);
        
        // Header with all MSLK fields
        writer.WriteLine("Index,Unknown_0x00,Unknown_0x01,Unknown_0x02,ParentIndex,MspiFirstIndex,MspiIndexCount,LinkIdRaw,ReferenceIndex,Unknown_0x12," +
                        "LinkIdPadding,ReferenceIndexHigh,ReferenceIndexLow,HasGeometry,IsContainer");
        
        for (int i = 0; i < scene.Links.Count; i++)
        {
            var link = scene.Links[i];
            bool hasGeometry = link.MspiFirstIndex != -1;
            bool isContainer = link.MspiFirstIndex == -1;
            
            writer.WriteLine($"{i},{link.Unknown_0x00},{link.Unknown_0x01},{link.Unknown_0x02},{link.ParentIndex}," +
                           $"{link.MspiFirstIndex},{link.MspiIndexCount},{link.LinkIdRaw},{link.ReferenceIndex},{link.Unknown_0x12}," +
                           $"{link.LinkIdPadding},{link.ReferenceIndexHigh},{link.ReferenceIndexLow},{hasGeometry},{isContainer}");
        }
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Exports detailed MPRL analysis with placement and linkage focus.
    /// </summary>
    private static Task ExportMprlDetailedAnalysis(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "mprl_detailed.csv");
        using var writer = new StreamWriter(csvPath);
        
        // Header with all MPRL fields
        writer.WriteLine("Index,Unknown0,Unknown2,Unknown4,Unknown6,Position_X,Position_Y,Position_Z,Unknown14,Unknown16," +
                        "LinksToMSLK,MSLKMatches");
        
        for (int i = 0; i < scene.Placements.Count; i++)
        {
            var placement = scene.Placements[i];
            
            // Check for MSLK linkage
            bool linksToMslk = scene.Links.Any(link => link.ParentIndex == placement.Unknown4);
            int mslkMatches = scene.Links.Count(link => link.ParentIndex == placement.Unknown4);
            
            writer.WriteLine($"{i},{placement.Unknown0},{placement.Unknown2},{placement.Unknown4},{placement.Unknown6}," +
                           $"{placement.Position.X},{placement.Position.Y},{placement.Position.Z},{placement.Unknown14},{placement.Unknown16}," +
                           $"{linksToMslk},{mslkMatches}");
        }
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Exports detailed MPRR analysis with property pattern focus.
    /// </summary>
    private static Task ExportMprrDetailedAnalysis(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "mprr_detailed.csv");
        using var writer = new StreamWriter(csvPath);
        
        // Header with pattern analysis
        writer.WriteLine("Index,Value1,Value2,IsSeparator,SegmentIndex,DistanceFromLastSeparator");
        
        int segmentIndex = 0;
        int lastSeparatorIndex = -1;
        
        for (int i = 0; i < scene.Properties.Count; i++)
        {
            var prop = scene.Properties[i];
            bool isSeparator = prop.Value1 == 65535;
            
            if (isSeparator)
            {
                segmentIndex++;
                lastSeparatorIndex = i;
            }
            
            int distanceFromSeparator = lastSeparatorIndex == -1 ? i : i - lastSeparatorIndex;
            
            writer.WriteLine($"{i},{prop.Value1},{prop.Value2},{isSeparator},{segmentIndex},{distanceFromSeparator}");
        }
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Exports detailed MSUR analysis with surface grouping focus.
    /// </summary>
    private static Task ExportMsurDetailedAnalysis(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "msur_detailed.csv");
        using var writer = new StreamWriter(csvPath);
        
        // Header with grouping analysis
        writer.WriteLine("Index,SurfaceGroupKey,IndexCount,Unknown_0x02,Nx,Ny,Nz,Height,MsviFirstIndex,MdosIndex,SurfaceKey," +
                        "SurfaceKeyHigh16,SurfaceKeyLow16,IsM2Bucket,IsLiquidCandidate,HasGeometry");
        
        for (int i = 0; i < scene.Surfaces.Count; i++)
        {
            var surface = scene.Surfaces[i];
            bool hasGeometry = surface.IndexCount > 0;
            
            writer.WriteLine($"{i},{surface.SurfaceGroupKey},{surface.IndexCount},{surface.Unknown_0x02}," +
                           $"{surface.Nx},{surface.Ny},{surface.Nz},{surface.Height},{surface.MsviFirstIndex},{surface.MdosIndex}," +
                           $"{surface.SurfaceKey},{surface.SurfaceKeyHigh16},{surface.SurfaceKeyLow16}," +
                           $"{surface.IsM2Bucket},{surface.IsLiquidCandidate},{hasGeometry}");
        }
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Exports cross-reference matrix to discover chunk relationships.
    /// </summary>
    private static Task ExportCrossReferenceMatrix(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "cross_references.csv");
        using var writer = new StreamWriter(csvPath);
        
        writer.WriteLine("Analysis: Cross-Reference Patterns in PM4 Chunks");
        writer.WriteLine();
        
        // MPRL.Unknown4 -> MSLK.ParentIndex analysis
        writer.WriteLine("MPRL.Unknown4 -> MSLK.ParentIndex Matches:");
        writer.WriteLine("MPRL_Index,Unknown4,MSLK_Matches,MSLK_Indices");
        
        for (int i = 0; i < scene.Placements.Count; i++)
        {
            var placement = scene.Placements[i];
            var matchingLinks = scene.Links
                .Select((link, index) => new { Link = link, Index = index })
                .Where(x => x.Link.ParentIndex == placement.Unknown4)
                .ToList();
            
            if (matchingLinks.Any())
            {
                var indices = string.Join(";", matchingLinks.Select(x => x.Index));
                writer.WriteLine($"{i},{placement.Unknown4},{matchingLinks.Count},\"{indices}\"");
            }
        }
        
        writer.WriteLine();
        
        // MSLK hierarchy analysis
        writer.WriteLine("MSLK Hierarchy Analysis:");
        writer.WriteLine("ParentIndex,ChildCount,ContainerCount,GeometryCount");
        
        var parentGroups = scene.Links.GroupBy(link => link.ParentIndex);
        foreach (var group in parentGroups.OrderBy(g => g.Key))
        {
            int childCount = group.Count();
            int containerCount = group.Count(link => link.MspiFirstIndex == -1);
            int geometryCount = group.Count(link => link.MspiFirstIndex != -1);
            
            writer.WriteLine($"{group.Key},{childCount},{containerCount},{geometryCount}");
        }
        
        writer.WriteLine();
        
        // Surface grouping analysis
        writer.WriteLine("Surface Grouping Analysis:");
        writer.WriteLine("SurfaceKey,SurfaceCount,TotalTriangles,TotalVertices");
        
        var surfaceGroups = scene.Surfaces.GroupBy(s => s.SurfaceKey);
        foreach (var group in surfaceGroups.OrderBy(g => g.Key))
        {
            int surfaceCount = group.Count();
            int totalTriangles = group.Sum(s => (int)s.IndexCount / 3);
            int totalVertices = group.Sum(s => (int)s.IndexCount);
            
            writer.WriteLine($"0x{group.Key:X8},{surfaceCount},{totalTriangles},{totalVertices}");
        }
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Exports detailed MSCN collision vertex analysis with spatial relationships to MSVT geometry.
    /// </summary>
    private static Task ExportMscnDetailedAnalysis(Pm4Scene scene, string analysisDir)
    {
        var csvPath = Path.Combine(analysisDir, "mscn_detailed.csv");
        using var writer = new StreamWriter(csvPath);
        
        // Header for MSCN collision vertex analysis
        writer.WriteLine("Index,Position_X,Position_Y,Position_Z,ClosestMSVTDistance,ClosestMSVTIndex,IsNearGeometry," +
                        "DistanceToOrigin,IsAtOrigin,ClusterGroup");
        
        // Note: MSCN data needs to be passed directly to this method
        // For now, create empty analysis until we refactor to pass MSCN chunk
        var collisionVertices = new List<Vector3>(); // TODO: Pass MSCN chunk data
        var geometryVertices = scene.Vertices;
        
        // Spatial clustering analysis - group nearby collision vertices
        var clusters = new Dictionary<int, int>(); // vertex index -> cluster ID
        var nextClusterId = 0;
        const float clusterThreshold = 1.0f;
        
        for (int i = 0; i < collisionVertices.Count; i++)
        {
            var collisionVert = collisionVertices[i];
            
            // Find closest MSVT geometry vertex
            var closestDistance = double.MaxValue;
            var closestIndex = -1;
            
            for (int j = 0; j < geometryVertices.Count; j++)
            {
                var distance = Vector3.Distance(collisionVert, geometryVertices[j]);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestIndex = j;
                }
            }
            
            var isNearGeometry = closestDistance < 0.1; // Within 0.1 units
            var distanceToOrigin = Vector3.Distance(collisionVert, Vector3.Zero);
            var isAtOrigin = distanceToOrigin < 0.001f;
            
            // Assign to cluster (simple spatial grouping)
            var clusterId = -1;
            for (int k = 0; k < i; k++)
            {
                if (Vector3.Distance(collisionVert, collisionVertices[k]) < clusterThreshold)
                {
                    if (clusters.TryGetValue(k, out var existingCluster))
                    {
                        clusterId = existingCluster;
                        break;
                    }
                }
            }
            
            if (clusterId == -1)
            {
                clusterId = nextClusterId++;
            }
            clusters[i] = clusterId;
            
            writer.WriteLine($"{i},{collisionVert.X:F6},{collisionVert.Y:F6},{collisionVert.Z:F6}," +
                           $"{closestDistance:F6},{closestIndex},{isNearGeometry}," +
                           $"{distanceToOrigin:F6},{isAtOrigin},{clusterId}");
        }
        
        return Task.CompletedTask;
    }
    
    /// <summary>
    /// Dumps detailed MSCN analysis to understand collision vertex relationships
    /// </summary>
    private static async Task DumpMscnDetailedAsync(StreamWriter writer, MscnChunk mscn, MsvtChunk msvt)
    {
        await writer.WriteLineAsync($"MSCN Collision Vertices: {mscn.Vertices.Count}");
        await writer.WriteLineAsync($"MSVT Geometry Vertices: {msvt.Vertices.Count}");
        await writer.WriteLineAsync();
        
        // Analyze spatial relationships
        var spatialMatches = 0;
        var averageDistance = 0.0;
        var minDistance = double.MaxValue;
        var maxDistance = 0.0;
        
        for (int i = 0; i < Math.Min(mscn.Vertices.Count, 100); i++) // Sample first 100
        {
            var collisionVert = mscn.Vertices[i];
            var closestGeomDistance = double.MaxValue;
            
            foreach (var geomVert in msvt.Vertices)
            {
                var distance = Vector3.Distance(collisionVert, geomVert);
                if (distance < closestGeomDistance)
                    closestGeomDistance = distance;
            }
            
            if (closestGeomDistance < 0.1) // Within 0.1 units
                spatialMatches++;
                
            averageDistance += closestGeomDistance;
            minDistance = Math.Min(minDistance, closestGeomDistance);
            maxDistance = Math.Max(maxDistance, closestGeomDistance);
        }
        
        averageDistance /= Math.Min(mscn.Vertices.Count, 100);
        
        await writer.WriteLineAsync($"Spatial Analysis (first 100 MSCN vertices):");
        await writer.WriteLineAsync($"  Close matches to MSVT: {spatialMatches}/100");
        await writer.WriteLineAsync($"  Average distance to closest MSVT: {averageDistance:F6}");
        await writer.WriteLineAsync($"  Min distance: {minDistance:F6}");
        await writer.WriteLineAsync($"  Max distance: {maxDistance:F6}");
        await writer.WriteLineAsync();
        
        // Show first 20 collision vertices with analysis
        await writer.WriteLineAsync("First 20 MSCN collision vertices:");
        for (int i = 0; i < Math.Min(mscn.Vertices.Count, 20); i++)
        {
            var vert = mscn.Vertices[i];
            await writer.WriteLineAsync($"  [{i:D3}] ({vert.X:F6}, {vert.Y:F6}, {vert.Z:F6})");
        }
    }
    
    /// <summary>
    /// Exports detailed MSCN collision vertex data to CSV
    /// </summary>
    private static async Task ExportMscnDetailedCsv(string filePath, MscnChunk mscn, MsvtChunk msvt)
    {
        using var writer = new StreamWriter(filePath);
        
        // CSV header
        await writer.WriteLineAsync("Index,Position_X,Position_Y,Position_Z,ClosestMSVTDistance,ClosestMSVTIndex,IsNearGeometry");
        
        // Export each collision vertex with spatial analysis
        for (int i = 0; i < mscn.Vertices.Count; i++)
        {
            var collisionVert = mscn.Vertices[i];
            
            // Find closest MSVT vertex
            var closestDistance = double.MaxValue;
            var closestIndex = -1;
            
            for (int j = 0; j < msvt.Vertices.Count; j++)
            {
                var distance = Vector3.Distance(collisionVert, msvt.Vertices[j]);
                if (distance < closestDistance)
                {
                    closestDistance = distance;
                    closestIndex = j;
                }
            }
            
            var isNearGeometry = closestDistance < 0.1; // Within 0.1 units
            
            await writer.WriteLineAsync($"{i},{collisionVert.X},{collisionVert.Y},{collisionVert.Z}," +
                                      $"{closestDistance},{closestIndex},{isNearGeometry}");
        }
    }
}
