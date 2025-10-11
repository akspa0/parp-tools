using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Partitions each CK24 (CompositeKey top 24 bits) group into connectivity-based instances.
    /// Optionally applies soft MSLK gating to drop obviously unrelated surfaces.
    /// </summary>
    internal sealed class CompositeHierarchyInstanceAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var output = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0)
                return output;

            var gVerts = scene.Vertices;
            var gIndices = scene.Indices;
            var gSurfaces = scene.Surfaces;

            // Precompute surface lookup by MsviFirstIndex (global)
            var surfaceByFirstIndex = new Dictionary<int, int>();
            for (int i = 0; i < gSurfaces.Count; i++)
            {
                var s = gSurfaces[i];
                if (s.MsviFirstIndex <= int.MaxValue)
                {
                    var fi = unchecked((int)s.MsviFirstIndex);
                    if (!surfaceByFirstIndex.ContainsKey(fi))
                        surfaceByFirstIndex[fi] = i;
                }
            }

            // Group by CK24
            static uint Top24(uint key) => key & 0xFFFFFF00u;
            var groups = gSurfaces
                .Select((s, idx) => (s, idx))
                .GroupBy(x => Top24(x.s.CompositeKey))
                .OrderBy(g => g.Key);

            // Use MSLK gates?
            bool useMslk = options.CkInstanceUseMslk;
            var tileOffsets = scene.TileIndexOffsetByTileId?.ToArray() ?? Array.Empty<KeyValuePair<int, int>>();

            // Build quick per-surface parent hit sets if requested
            Dictionary<int, HashSet<uint>>? surfaceParentHits = null;
            if (useMslk && scene.Links.Count > 0)
            {
                surfaceParentHits = new Dictionary<int, HashSet<uint>>();
                foreach (var link in scene.Links)
                {
                    if (!link.HasGeometry) continue;
                    int mspiFirst = link.MspiFirstIndex;
                    int mspiCount = link.MspiIndexCount;
                    bool found = false;
                    if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                    {
                        int tileId = tileY * 64 + tileX;
                        if (scene.TileIndexOffsetByTileId != null && scene.TileIndexOffsetByTileId.TryGetValue(tileId, out int baseIdx))
                        {
                            int globalFirst = baseIdx + mspiFirst;
                            if (surfaceByFirstIndex.TryGetValue(globalFirst, out int sidx))
                            {
                                var surf = gSurfaces[sidx];
                                int surfCount = surf.IndexCount;
                                int linkCount = mspiCount;
                                bool countOk = linkCount > 0 && (
                                    surfCount == linkCount ||
                                    surfCount == linkCount * 3 ||
                                    (surfCount % 3 == 0 && (surfCount / 3) == linkCount)
                                );
                                if (countOk)
                                {
                                    if (!surfaceParentHits.TryGetValue(sidx, out var set))
                                    {
                                        set = new HashSet<uint>();
                                        surfaceParentHits[sidx] = set;
                                    }
                                    set.Add(link.ParentId);
                                    found = true;
                                }
                            }
                        }
                    }
                    // no global fallback scan here by default to avoid over-matching
                }
            }

            foreach (var grp in groups)
            {
                var ck24 = grp.Key >> 8; // 24-bit value for naming
                var groupSurfaceIdx = grp.Select(x => x.idx).ToList();
                int n = groupSurfaceIdx.Count;
                if (n == 0) continue;

                // Map scene-surface-index -> local-group-index
                var localIndexBySurface = new Dictionary<int, int>(n);
                for (int i = 0; i < n; i++) localIndexBySurface[groupSurfaceIdx[i]] = i;

                // Union-Find for connectivity via shared global vertices
                var dsu = new DSU(n);

                // vertex -> list of local surface indices
                var vertToSurfs = new Dictionary<int, List<int>>();

                for (int gi = 0; gi < n; gi++)
                {
                    int sIdx = groupSurfaceIdx[gi];
                    var surf = gSurfaces[sIdx];
                    if (surf.MsviFirstIndex > int.MaxValue) continue;
                    int first = unchecked((int)surf.MsviFirstIndex);
                    int count = surf.IndexCount;
                    if (first < 0 || count < 3 || first + count > gIndices.Count) continue;

                    // Dedup per-surface indices to keep memory reasonable
                    var used = new HashSet<int>();
                    for (int k = 0; k < count; k++)
                    {
                        int v = gIndices[first + k];
                        if (v < 0 || v >= gVerts.Count) continue;
                        if (!used.Add(v)) continue;
                        if (!vertToSurfs.TryGetValue(v, out var list))
                        {
                            list = new List<int>();
                            vertToSurfs[v] = list;
                        }
                        list.Add(gi);
                    }
                }

                // Union all surfaces that share at least one vertex
                foreach (var kv in vertToSurfs)
                {
                    var list = kv.Value;
                    if (list.Count <= 1) continue;
                    int root = list[0];
                    for (int i = 1; i < list.Count; i++) dsu.Union(root, list[i]);
                }

                // Gather components
                var compToSurfaces = new Dictionary<int, List<int>>();
                for (int gi = 0; gi < n; gi++)
                {
                    int r = dsu.Find(gi);
                    if (!compToSurfaces.TryGetValue(r, out var lst)) { lst = new List<int>(); compToSurfaces[r] = lst; }
                    lst.Add(groupSurfaceIdx[gi]); // store scene-surface-index
                }

                // Assemble each component as an instance
                int compIndex = 0;
                foreach (var comp in compToSurfaces.Values)
                {
                    var compList = comp.ToList();
                    IEnumerable<int> surfacesForBuild = compList;
                    int preGatingCount = compList.Count;
                    int withHits = 0;
                    int withoutHits = 0;

                    if (useMslk && surfaceParentHits != null)
                    {
                        // Soft gating: prefer surfaces that have any parent hits; if many have none, allow up to ratio
                        var hits = compList.Select(s => surfaceParentHits.ContainsKey(s)).ToList();
                        withHits = hits.Count(b => b);
                        withoutHits = hits.Count - withHits;
                        double ratioWithout = (double)withoutHits / Math.Max(1, hits.Count);
                        if (ratioWithout > options.CkInstanceAllowUnlinkedRatio && withHits > 0)
                        {
                            surfacesForBuild = compList.Where(s => surfaceParentHits.ContainsKey(s));
                            // If we filtered everything out due to threshold, revert
                            if (!surfacesForBuild.Any()) surfacesForBuild = comp;
                        }
                    }

                    var name = $"CK_{ck24:X6}_inst_{compIndex:D5}";
                    var assembled = AssembleFromSurfaceSet(name, surfacesForBuild, gVerts, gIndices, gSurfaces, options.CkInstanceMinTriangles);
                    if (assembled != null)
                    {
                        assembled.Meta["source"] = "composite-hierarchy-instance";
                        assembled.Meta["ck24"] = $"0x{ck24:X6}";
                        assembled.Meta["component_index"] = compIndex.ToString();
                        assembled.Meta["component_surface_count_pre"] = preGatingCount.ToString();
                        assembled.Meta["component_surface_count"] = surfacesForBuild.Count().ToString();
                        if (useMslk && surfaceParentHits != null)
                        {
                            assembled.Meta["mslk_hits"] = withHits.ToString();
                            assembled.Meta["mslk_unlinked"] = withoutHits.ToString();
                            double ratio = (double)withoutHits / Math.Max(1, withHits + withoutHits);
                            assembled.Meta["unlinked_ratio"] = ratio.ToString("0.###");
                        }
                        output.Add(assembled);
                    }
                    compIndex++;
                }
            }

            return output;
        }

        private static AssembledObject? AssembleFromSurfaceSet(
            string name,
            IEnumerable<int> surfaceIndices,
            List<Vector3> gVerts,
            List<int> gIndices,
            List<MsurChunk.Entry> gSurfaces,
            int minTriangles)
        {
            var localVerts = new List<Vector3>();
            var localTris = new List<(int A, int B, int C)>();
            var map = new Dictionary<int, int>();

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
                    int baseIdx = first + t * 3;
                    int a = gIndices[baseIdx];
                    int b = gIndices[baseIdx + 1];
                    int c = gIndices[baseIdx + 2];
                    if (a < 0 || b < 0 || c < 0 || a >= gVerts.Count || b >= gVerts.Count || c >= gVerts.Count)
                        continue;
                    int la = GetLocal(a, gVerts, localVerts, map);
                    int lb = GetLocal(b, gVerts, localVerts, map);
                    int lc = GetLocal(c, gVerts, localVerts, map);
                    localTris.Add((la, lb, lc));
                }
            }

            if (localTris.Count >= Math.Max(0, minTriangles))
                return new AssembledObject(name, localVerts, localTris);
            return null;
        }

        private static int GetLocal(int gIdx, List<Vector3> gVerts, List<Vector3> lVerts, Dictionary<int, int> map)
        {
            if (map.TryGetValue(gIdx, out var li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIdx]);
            map[gIdx] = li;
            return li;
        }

        private sealed class DSU
        {
            private readonly int[] parent;
            private readonly byte[] rank;
            public DSU(int n)
            {
                parent = new int[n];
                rank = new byte[n];
                for (int i = 0; i < n; i++) parent[i] = i;
            }
            public int Find(int x)
            {
                if (parent[x] != x) parent[x] = Find(parent[x]);
                return parent[x];
            }
            public void Union(int a, int b)
            {
                a = Find(a); b = Find(b);
                if (a == b) return;
                if (rank[a] < rank[b]) { parent[a] = b; }
                else if (rank[a] > rank[b]) { parent[b] = a; }
                else { parent[b] = a; rank[a]++; }
            }
        }
    }
}
