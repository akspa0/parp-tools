using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Assembles geometry per MSLK parent (ParentId), resolving links to MSUR surfaces by
    /// rebased MSVI first-index: GlobalFirstIndex = TileIndexOffset(tileId) + MspiFirstIndex.
    /// Falls back to scanning all tile index offsets when MSLK tile coordinates are invalid.
    /// </summary>
    internal sealed class MslkParentAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Vertices.Count == 0 || scene.Indices.Count == 0 || scene.Surfaces.Count == 0 || scene.Links.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;
            var surfaces = scene.Surfaces;
            var links = scene.Links;

            // Fast lookup: MsviFirstIndex -> surface index
            var surfaceByFirstIndex = new Dictionary<int, int>();
            for (int i = 0; i < surfaces.Count; i++)
            {
                var s = surfaces[i];
                if (s.MsviFirstIndex <= int.MaxValue)
                {
                    var fi = unchecked((int)s.MsviFirstIndex);
                    if (!surfaceByFirstIndex.ContainsKey(fi))
                        surfaceByFirstIndex[fi] = i;
                }
            }

            // Precompute list of tile index offsets
            var tileOffsets = scene.TileIndexOffsetByTileId?.ToArray() ?? Array.Empty<KeyValuePair<int,int>>();

            // Group links by ParentId (authoritative instance identifier)
            var groups = links.GroupBy(l => l.ParentId).OrderBy(g => g.Key);

            foreach (var grp in groups)
            {
                var surfaceIdxSet = new HashSet<int>();
                int resolved = 0, unresolved = 0;

                foreach (var link in grp)
                {
                    if (!link.HasGeometry) { continue; }
                    int mspiFirst = link.MspiFirstIndex;
                    int mspiCount = link.MspiIndexCount;

                    // Try use tile coords if valid
                    bool found = false;
                    if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                    {
                        int tileId = tileY * 64 + tileX; // matches ToLinearIndex() encoding
                        if (scene.TileIndexOffsetByTileId != null && scene.TileIndexOffsetByTileId.TryGetValue(tileId, out int baseIdx))
                        {
                            int globalFirst = baseIdx + mspiFirst;
                            if (surfaceByFirstIndex.TryGetValue(globalFirst, out int sidx))
                            {
                                // Strengthen match: require index-count consistency (indices vs triangles semantics)
                                var surf = surfaces[sidx];
                                int surfCount = surf.IndexCount;
                                int linkCount = mspiCount;
                                bool countOk = linkCount > 0 && (
                                    surfCount == linkCount ||
                                    surfCount == linkCount * 3 ||
                                    (surfCount % 3 == 0 && (surfCount / 3) == linkCount)
                                );
                                if (countOk)
                                {
                                    surfaceIdxSet.Add(sidx);
                                    found = true;
                                }
                            }
                        }
                    }

                    // Fallback: scan all tile index offsets (optional; may over-match)
                    if (!found && options.MslkParentAllowFallbackScan && tileOffsets.Length > 0)
                    {
                        for (int t = 0; t < tileOffsets.Length; t++)
                        {
                            int baseIdx = tileOffsets[t].Value;
                            int globalFirst = baseIdx + mspiFirst;
                            if (surfaceByFirstIndex.TryGetValue(globalFirst, out int sidx))
                            {
                                var surf = surfaces[sidx];
                                int surfCount = surf.IndexCount;
                                int linkCount = mspiCount;
                                bool countOk = linkCount > 0 && (
                                    surfCount == linkCount ||
                                    surfCount == linkCount * 3 ||
                                    (surfCount % 3 == 0 && (surfCount / 3) == linkCount)
                                );
                                if (countOk)
                                {
                                    surfaceIdxSet.Add(sidx);
                                    found = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (found) resolved++; else unresolved++;
                }

                if (surfaceIdxSet.Count == 0)
                    continue; // container-only instance or unresolved links

                var name = $"Parent_{grp.Key:X8}";
                AssembleFromSurfaceSet(name, surfaceIdxSet, verts, indices, surfaces, result, options.MslkParentMinTriangles);
            }

            return result;
        }

        private static void AssembleFromSurfaceSet(
            string name,
            IEnumerable<int> surfaceIndices,
            List<Vector3> gVerts,
            List<int> gIndices,
            List<MsurChunk.Entry> gSurfaces,
            List<AssembledObject> output,
            int minTriangles)
        {
            var localVerts = new List<Vector3>();
            var localTris = new List<(int A,int B,int C)>();
            var map = new Dictionary<int,int>();

            foreach (var si in surfaceIndices)
            {
                var surf = gSurfaces[si];
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                int count = surf.IndexCount;
                if (first < 0 || count < 3 || first + count > gIndices.Count) continue;

                int triCount = count / 3;
                for (int t = 0; t < triCount; t++)
                {
                    int baseIdx = first + t*3;
                    int a = gIndices[baseIdx];
                    int b = gIndices[baseIdx+1];
                    int c = gIndices[baseIdx+2];
                    if (a<0||b<0||c<0||a>=gVerts.Count||b>=gVerts.Count||c>=gVerts.Count)
                        continue;
                    int la = GetLocal(a, gVerts, localVerts, map);
                    int lb = GetLocal(b, gVerts, localVerts, map);
                    int lc = GetLocal(c, gVerts, localVerts, map);
                    localTris.Add((la,lb,lc));
                }
            }

            if (localTris.Count >= Math.Max(0, minTriangles))
            {
                output.Add(new AssembledObject(name, localVerts, localTris));
            }
        }

        private static int GetLocal(int gIdx, List<Vector3> gVerts, List<Vector3> lVerts, Dictionary<int,int> map)
        {
            if (map.TryGetValue(gIdx, out var li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIdx]);
            map[gIdx] = li;
            return li;
        }
    }
}
