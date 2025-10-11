using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Assembles objects per MSLK instance (ParentId). Two modes:
    /// - pure instance: one OBJ per ParentId grouping all referenced surfaces
    /// - instance+ck24: split each ParentId by MSUR.CompositeKey top-24 bits
    /// </summary>
    internal sealed class MslkInstanceAssembler : IAssembler
    {
        private readonly bool _splitByCk24;
        public MslkInstanceAssembler(bool splitByCk24) => _splitByCk24 = splitByCk24;

        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Vertices.Count == 0 || scene.Indices.Count == 0 || scene.Surfaces.Count == 0 || scene.Links.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;
            var surfaces = scene.Surfaces;
            var links = scene.Links;

            // Group links by ParentId (authoritative instance identifier)
            var groups = links.GroupBy(l => l.ParentId).OrderBy(g => g.Key);

            foreach (var grp in groups)
            {
                // Collect unique surfaces referenced by this instance
                var surfaceIdxSet = new HashSet<int>();
                foreach (var link in grp)
                {
                    int sidx = link.SurfaceRefIndex;
                    if (sidx >= 0 && sidx < surfaces.Count)
                        surfaceIdxSet.Add(sidx);
                }
                if (surfaceIdxSet.Count == 0)
                    continue; // container-only instance

                if (_splitByCk24)
                {
                    // Further split by top-24 bits of CompositeKey
                    static uint Top24(uint key) => key & 0xFFFFFF00u;
                    var perKey = surfaceIdxSet
                        .Select(i => (i, key: Top24(surfaces[i].CompositeKey)))
                        .GroupBy(t => t.key)
                        .OrderBy(g => g.Key);

                    foreach (var keyGroup in perKey)
                    {
                        var name = $"Instance_{grp.Key}_CK_{keyGroup.Key >> 8:X6}";
                        AssembleFromSurfaceSet(name, keyGroup.Select(t => t.i), verts, indices, surfaces, result);
                    }
                }
                else
                {
                    var name = $"Instance_{grp.Key}";
                    AssembleFromSurfaceSet(name, surfaceIdxSet, verts, indices, surfaces, result);
                }
            }

            return result;
        }

        private static void AssembleFromSurfaceSet(
            string name,
            IEnumerable<int> surfaceIndices,
            List<Vector3> gVerts,
            List<int> gIndices,
            List<MsurChunk.Entry> gSurfaces,
            List<AssembledObject> output)
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

            if (localTris.Count > 0)
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
