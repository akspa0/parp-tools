using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using PM4NextExporter.Services;
using PM4NextExporter.Model;

namespace PM4NextExporter.Services
{
    public static class DiagnosticsService
    {
        /// <summary>
        /// Writes flattened MSUR surface data to <paramref name="outDir"/>/surfaces.csv for offline analysis.
        /// </summary>
        internal static void WriteSurfaceCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;

            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "surfaces.csv");

            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            // Header
            writer.WriteLine("index,compositeKey,byteAA,byteBB,byteCC,byteDD,surfaceKeyHigh16,msviFirstIndex,indexCount,groupKey");

            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var s = scene.Surfaces[i];
                uint key = s.CompositeKey;
                byte aa = (byte)(key >> 24);
                byte bb = (byte)(key >> 16);
                byte cc = (byte)(key >> 8);
                byte dd = (byte)(key);
                writer.WriteLine(string.Join(',',
                    i,
                    $"0x{key:X8}",
                    aa,
                    bb,
                    cc,
                    dd,
                    $"0x{s.SurfaceKeyHigh16:X4}",
                    s.MsviFirstIndex,
                    s.IndexCount,
                    s.SurfaceGroupKey));
            }
        }

        /// <summary>
        /// Writes per-assembled-object coverage metrics to <paramref name="outDir"/>/assembly_coverage.csv.
        /// Columns: index,name,vertexCount,triangleCount
        /// </summary>
        internal static void WriteAssemblyCoverageCsv(string outDir, List<AssembledObject> assembled)
        {
            if (assembled == null || assembled.Count == 0)
                return;
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "assembly_coverage.csv");
            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("index,name,vertexCount,triangleCount");
            for (int i = 0; i < assembled.Count; i++)
            {
                var o = assembled[i];
                var name = o?.Name ?? string.Empty;
                // Basic CSV escaping for commas/quotes
                if (name.Contains('"')) name = name.Replace("\"", "\"\"");
                if (name.Contains(',') || name.Contains('"')) name = "\"" + name + "\"";
                writer.WriteLine(string.Join(',', i, name, o?.VertexCount ?? 0, o?.TriangleCount ?? 0));
            }
        }

        /// <summary>
        /// Writes per-surface MSLK parent hit counts to surface_parent_hits.csv.
        /// Columns: surfaceIndex,compositeKey,ck24,groupKey,msviFirstIndex,indexCount,parentHitCount
        /// </summary>
        internal static void WriteSurfaceParentHitsCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "surface_parent_hits.csv");
            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("surfaceIndex,compositeKey,ck24,groupKey,msviFirstIndex,indexCount,parentHitCount");

            // Build lookup MsviFirstIndex(global) -> surface index
            var surfaceByFirstIndex = new Dictionary<int, int>();
            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var s = scene.Surfaces[i];
                if (s.MsviFirstIndex <= int.MaxValue)
                {
                    int fi = unchecked((int)s.MsviFirstIndex);
                    if (!surfaceByFirstIndex.ContainsKey(fi)) surfaceByFirstIndex[fi] = i;
                }
            }

            var parentHits = new Dictionary<int, HashSet<uint>>();
            if (scene.Links != null && scene.Links.Count > 0 && scene.TileIndexOffsetByTileId != null)
            {
                foreach (var link in scene.Links)
                {
                    if (!link.HasGeometry) continue;
                    if (!link.TryDecodeTileCoordinates(out int tileX, out int tileY)) continue;
                    int tileId = tileY * 64 + tileX;
                    if (!scene.TileIndexOffsetByTileId.TryGetValue(tileId, out int baseIdx)) continue;

                    int globalFirst = baseIdx + link.MspiFirstIndex;
                    if (surfaceByFirstIndex.TryGetValue(globalFirst, out int sidx))
                    {
                        if (!parentHits.TryGetValue(sidx, out var set)) { set = new HashSet<uint>(); parentHits[sidx] = set; }
                        set.Add(link.ParentId);
                    }
                }
            }

            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var s = scene.Surfaces[i];
                uint key = s.CompositeKey;
                uint ck24 = (key & 0xFFFFFF00u) >> 8;
                int count = parentHits.TryGetValue(i, out var set2) ? set2.Count : 0;
                writer.WriteLine(string.Join(',', i, $"0x{key:X8}", $"0x{ck24:X6}", s.SurfaceGroupKey, s.MsviFirstIndex, s.IndexCount, count));
            }
        }

        // Legacy stub for snapshot; kept for compatibility
        internal static void WriteSnapshotCsv(string outDir, string name)
        {
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, $"{name}.csv");
            File.WriteAllText(path, "metric,value\nstub,1\n");
        }
        /// <summary>
        /// Writes aggregated statistics per CompositeKey to <paramref name="outDir"/>/surface_summary.csv.
        /// No data is filtered – every MSUR entry contributes to its key’s totals.
        /// Columns: compositeKey,byteAA,byteBB,byteCC,byteDD,rowCount,totalIndexCount,uniqueGroupKeys
        /// </summary>
        internal static void WriteCompositeSummaryCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return;

            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "surface_summary.csv");

            // Aggregate
            var stats = new Dictionary<uint, (int rowCount, int indexSum, HashSet<byte> groups)>();
            foreach (var s in scene.Surfaces)
            {
                if (!stats.TryGetValue(s.CompositeKey, out var tuple))
                    tuple = (0, 0, new HashSet<byte>());
                tuple.rowCount++;
                tuple.indexSum += s.IndexCount;
                tuple.groups.Add(s.SurfaceGroupKey);
                stats[s.CompositeKey] = tuple;
            }

            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("compositeKey,byteAA,byteBB,byteCC,byteDD,rowCount,totalIndexCount,uniqueGroupKeys");
            foreach (var kvp in stats)
            {
                uint key = kvp.Key;
                byte aa = (byte)(key >> 24);
                byte bb = (byte)(key >> 16);
                byte cc = (byte)(key >> 8);
                byte dd = (byte)(key);
                var (cnt, sum, groups) = kvp.Value;
                writer.WriteLine(string.Join(',',
                    $"0x{key:X8}", aa, bb, cc, dd, cnt, sum, groups.Count));
            }
        }

        /// <summary>
        /// Dumps all MSCN exterior vertices to <paramref name="outDir"/>/mscn_vertices.csv for analysis.
        /// Columns: index,x,y,z,xCanonical,yCanonical,zCanonical,tileId,worldA_X,worldA_Y,blockA_X,blockA_Y,worldB_X,worldB_Y,blockB_X,blockB_Y
        /// where worldA/B are server->world candidate transforms to help correlate spaces.
        /// </summary>
        internal static void WriteMscnCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.MscnVertices == null || scene.MscnVertices.Count == 0)
                return;
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "mscn_vertices.csv");
            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("index,x,y,z,xCanonical,yCanonical,zCanonical,tileId,tileX,tileY,worldA_X,worldA_Y,blockA_X,blockA_Y,worldB_X,worldB_Y,blockB_X,blockB_Y");

            // Determine per-vertex tileId mapping if available; otherwise attempt best-effort fallback for single-tile loads
            var tileIds = (scene.MscnTileIds != null && scene.MscnTileIds.Count == scene.MscnVertices.Count)
                ? scene.MscnTileIds
                : null;

            int fallbackTileId = -1;
            if (tileIds == null)
            {
                try
                {
                    // If SourcePath points to a tile file, parse its coordinates
                    if (!string.IsNullOrWhiteSpace(scene.SourcePath) && File.Exists(scene.SourcePath))
                    {
                        var coord = ParpToolbox.Services.PM4.Pm4GlobalTileLoader.TileCoordinate.FromFileName(Path.GetFileName(scene.SourcePath));
                        fallbackTileId = coord.ToLinearIndex();
                    }
                }
                catch { /* keep fallback -1 */ }
            }
            for (int i = 0; i < scene.MscnVertices.Count; i++)
            {
                var v = scene.MscnVertices[i];
                var world = new System.Numerics.Vector3(v.Y, -v.X, v.Z);
                int tileId = tileIds != null ? tileIds[i] : fallbackTileId;
                int tileXOut = tileId >= 0 ? (tileId % 64) : -1;
                int tileYOut = tileId >= 0 ? (tileId / 64) : -1;
                // Server->world candidate transforms on raw server axes (per ADT_v18 docs)
                var (worldAX, worldAY) = CoordinateUtils.ServerAxisToWorldCandidates(v.X);
                var tmp = CoordinateUtils.ServerAxisToWorldCandidates(v.Y);
                // Note: compute per-axis separately
                double wAx = worldAX; // from X
                double wAy = tmp.worldA; // from Y
                double wBx = CoordinateUtils.ServerAxisToWorldCandidates(v.X).worldB;
                double wBy = CoordinateUtils.ServerAxisToWorldCandidates(v.Y).worldB;
                int bAx = CoordinateUtils.ClampBlock(CoordinateUtils.WorldAxisToBlockIndex(wAx));
                int bAy = CoordinateUtils.ClampBlock(CoordinateUtils.WorldAxisToBlockIndex(wAy));
                int bBx = CoordinateUtils.ClampBlock(CoordinateUtils.WorldAxisToBlockIndex(wBx));
                int bBy = CoordinateUtils.ClampBlock(CoordinateUtils.WorldAxisToBlockIndex(wBy));
                writer.WriteLine(string.Join(',', i, v.X, v.Y, v.Z, world.X, world.Y, world.Z, tileId, tileXOut, tileYOut, wAx, wAy, bAx, bAy, wBx, wBy, bBx, bBy));
            }
        }

        /// <summary>
        /// Writes MSLK links to <paramref name="outDir"/>/mslk_links.csv for diagnostics.
        /// Columns: index,parentId,type,flags,sortKey,surfaceRefIndex,mspiFirstIndex,mspiIndexCount,hasGeom,tileX,tileY,tileId,msurCompositeKey,msurGroupKey,msurMsviFirstIndex,msurIndexCount
        /// </summary>
        internal static void WriteMslkLinksCsv(string outDir, PM4NextExporter.Model.Scene scene)
        {
            if (scene == null || scene.Links == null || scene.Links.Count == 0)
                return;
            Directory.CreateDirectory(outDir);
            var path = Path.Combine(outDir, "mslk_links.csv");
            using var writer = new StreamWriter(path, false, System.Text.Encoding.UTF8);
            writer.WriteLine("index,parentId,type,flags,sortKey,surfaceRefIndex,mspiFirstIndex,mspiIndexCount,hasGeom,tileX,tileY,tileId,msurCompositeKey,msurGroupKey,msurMsviFirstIndex,msurIndexCount");

            for (int i = 0; i < scene.Links.Count; i++)
            {
                var l = scene.Links[i];
                int tileX = -1, tileY = -1, tileId = -1;
                if (l.TryDecodeTileCoordinates(out var tx, out var ty))
                {
                    tileX = tx; tileY = ty; tileId = ty * 64 + tx;
                }

                // Join with MSUR entry if available
                uint msurKey = 0; byte msurGroup = 0; uint msurFirst = 0; int msurCount = 0;
                if (scene.Surfaces != null && scene.Surfaces.Count > 0)
                {
                    int sidx = l.SurfaceRefIndex;
                    if (sidx >= 0 && sidx < scene.Surfaces.Count)
                    {
                        var s = scene.Surfaces[sidx];
                        msurKey = s.CompositeKey;
                        msurGroup = s.SurfaceGroupKey;
                        msurFirst = s.MsviFirstIndex;
                        msurCount = s.IndexCount;
                    }
                }

                writer.WriteLine(string.Join(',',
                    i,
                    l.ParentId,
                    l.Type_0x01,
                    l.Flags_0x00,
                    l.SortKey_0x02,
                    l.SurfaceRefIndex,
                    l.MspiFirstIndex,
                    l.MspiIndexCount,
                    l.HasGeometry ? 1 : 0,
                    tileX,
                    tileY,
                    tileId,
                    $"0x{msurKey:X8}",
                    msurGroup,
                    msurFirst,
                    msurCount));
            }
        }
    }
}
