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

        // Helper to export groups
        void ExportGroups(Dictionary<uint, List<(int A, int B, int C)>> gmap, string subFolder)
        {
            var dir = Path.Combine(outputRoot, subFolder);
            Directory.CreateDirectory(dir);
            foreach (var (key, faces) in gmap)
            {
                if (faces.Count == 0) continue;
                var objPath = Path.Combine(dir, $"G{key:X8}.obj");
                WriteObj(objPath, scene.Vertices, faces, exportFaces);
            }
        }

        // 2a. Group by SurfaceKey (Unknown_0x1C)
        ExportGroups(BuildFaceGroups(scene, s => s.SurfaceKey), "by_surfacekey");

        // 2b. For surfaces with key==0, further group by (SurfaceGroupKey<<8)|Flags0x00
        ExportGroups(BuildFaceGroups(scene, s => s.SurfaceKey == 0 ? (uint)((s.SurfaceGroupKey << 8) | s.FlagsOrUnknown_0x00) : 0xFFFFFFFFu, filterKey: 0xFFFFFFFFu), "by_ground_subbucket");

        // 2c. Group by SurfaceKey low 16 bits (sub-group id)
        ExportGroups(BuildFaceGroups(scene, s => s.SurfaceKeyLow16), "by_surfacekey_lo16");

        // 2d. Group by SurfaceKey high 16 bits (root group id)
        ExportGroups(BuildFaceGroups(scene, s => s.SurfaceKeyHigh16), "by_surfacekey_hi16");

        // 2e. Composite: (SK_lo16 << 8) | SurfaceGroupKey (experimental finer bucket)
        ExportGroups(BuildFaceGroups(scene, s => (uint)((s.SurfaceKeyLow16 << 8) | s.SurfaceGroupKey)), "by_sklo16_group");

        // 2f. Group by MSLK parent index (object assembly)
        if (scene.Links.Count > 0)
        {
            ExportGroups(BuildParentGroups(scene), "by_parent_index");
            ExportGroups(BuildReferenceGroups(scene), "by_reference_index");
        }
    }

    private static Dictionary<uint, List<(int A, int B, int C)>> BuildFaceGroups(Pm4Scene scene, Func<MsurChunk.Entry, uint> keySelector, uint? filterKey = null)
    {
        var groups = new Dictionary<uint, List<(int,int,int)>>();
        var msviIndices = CollectMsviIndices(scene);
        foreach (var surf in scene.Surfaces)
        {
            if (surf.IsM2Bucket) continue; // skip overlay model bucket
            if (filterKey.HasValue && keySelector(surf) == filterKey.Value) continue;

            uint key = keySelector(surf);
            if (!groups.TryGetValue(key, out var faces))
            {
                faces = new();
                groups[key] = faces;
            }
            int first = (int)surf.MsviFirstIndex;
            int count = surf.IndexCount;
            if (first < 0 || count < 3 || first + count > msviIndices.Count) continue;
            for (int i = 0; i < count; i += 3)
            {
                faces.Add((msviIndices[first + i], msviIndices[first + i + 1], msviIndices[first + i + 2]));
            }
        }
        return groups;
    }

    // Build groups by MSLK.ParentIndex (assembly/group root)
    private static Dictionary<uint, List<(int A, int B, int C)>> BuildParentGroups(Pm4Scene scene)
    {
        var mapSkloToParent = new Dictionary<ushort, uint>();
        foreach (var link in scene.Links)
        {
            // Prefer first seen parent if duplicates
            ushort key = link.LinkSubKey;
            if (!mapSkloToParent.ContainsKey(key))
                mapSkloToParent[key] = link.ParentIndex;
        }

        var groups = new Dictionary<uint, List<(int,int,int)>>();
        var msviIndices = CollectMsviIndices(scene);

        foreach (var surf in scene.Surfaces)
        {
            if (surf.IsM2Bucket) continue;
            if (!mapSkloToParent.TryGetValue(surf.SurfaceKeyLow16, out uint parent))
                continue; // skip if no parent link

            if (!groups.TryGetValue(parent, out var faces))
            {
                faces = new();
                groups[parent] = faces;
            }

            int first = (int)surf.MsviFirstIndex;
            int count = surf.IndexCount;
            if (first < 0 || count < 3 || first + count > msviIndices.Count) continue;
            for (int i = 0; i < count; i += 3)
            {
                faces.Add((msviIndices[first + i], msviIndices[first + i + 1], msviIndices[first + i + 2]));
            }
        }
        return groups;
    }

    private static Dictionary<uint, List<(int A, int B, int C)>> BuildReferenceGroups(Pm4Scene scene)
    {
        var mapSkloToRef = new Dictionary<ushort, uint>();
        foreach (var link in scene.Links)
        {
            ushort key = link.LinkSubKey;
            if (!mapSkloToRef.ContainsKey(key))
                mapSkloToRef[key] = link.ReferenceIndex;
        }

        var groups = new Dictionary<uint, List<(int,int,int)>>();
        var msviIndices = CollectMsviIndices(scene);
        foreach (var surf in scene.Surfaces)
        {
            if (surf.IsM2Bucket) continue;
            if (!mapSkloToRef.TryGetValue(surf.SurfaceKeyLow16, out uint refIdx))
                continue;
            if (!groups.TryGetValue(refIdx, out var faces))
            {
                faces = new();
                groups[refIdx] = faces;
            }
            int first = (int)surf.MsviFirstIndex;
            int count = surf.IndexCount;
            if (first < 0 || count < 3 || first + count > msviIndices.Count) continue;
            for (int i = 0; i < count; i += 3)
                faces.Add((msviIndices[first + i], msviIndices[first + i + 1], msviIndices[first + i + 2]));
        }
        return groups;
    }

    private static List<int> CollectMsviIndices(Pm4Scene scene)
    {
        // The adapter already triangulated msvi -> scene.Triangles when present
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

    private static void WriteObj(string path, IReadOnlyList<Vector3> verts, List<(int A, int B, int C)> faces, bool includeFaces)
    {
        using var sw = new StreamWriter(path);
        sw.WriteLine("# Auto-generated by parpToolbox bulk-dump");

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
}
