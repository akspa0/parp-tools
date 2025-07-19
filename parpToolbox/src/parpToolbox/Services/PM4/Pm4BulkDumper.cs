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
    public static void Dump(Pm4Scene scene, string outputRoot, bool exportFaces)
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
        
        // 4. Write vertex stats
        var vertexStatsPath = Path.Combine(outputRoot, "vertex_stats.csv");
        using (var writer = new StreamWriter(vertexStatsPath))
        {
            writer.WriteLine("TotalVertices,TotalIndices,TotalTriangles");
            writer.WriteLine($"{scene.Vertices.Count},{scene.Indices.Count},{scene.Indices.Count / 3}");
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
            Console.WriteLine("PM4 object assembly analysis completed.");
            Console.WriteLine($"  MPRL placements: {scene.Placements.Count}");
            Console.WriteLine($"  MSLK links: {scene.Links.Count}");
            Console.WriteLine($"  MSUR surfaces: {scene.Surfaces.Count}");
            Console.WriteLine($"  MPRR properties: {scene.Properties.Count}");
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
        
        Console.WriteLine($"Created {groups.Count} parent groups:");
        foreach (var group in groups)
        {
            Console.WriteLine($"  Group 0x{group.Key:X8}: {group.Value.Count} triangles");
        }
        
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
}
