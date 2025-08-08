using System.Collections.Generic;
using PM4NextExporter.Model;
using System.Linq;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    internal sealed class MsurIndexCountAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();

            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0)
                return result;

            var indices = scene.Indices ?? new List<int>();
            var verts = scene.Vertices ?? new List<System.Numerics.Vector3>();
            if (indices.Count == 0 || verts.Count == 0)
                return result;

            // Group by MSUR.IndexCount as per discovery
            var groups = scene.Surfaces
                .Where(s => s.IndexCount >= 3 && !s.IsM2Bucket)
                .GroupBy(s => s.IndexCount)
                .OrderBy(g => g.Key);

            foreach (var g in groups)
            {
                var name = $"MSUR_IndexCount_{g.Key}";

                // Collect triangle indices from all surfaces in this group
                var localVerts = new List<System.Numerics.Vector3>();
                var localTris = new List<(int A, int B, int C)>();
                var map = new Dictionary<int, int>(); // globalIndex -> localIndex

                foreach (var surf in g)
                {
                    if (surf.MsviFirstIndex > int.MaxValue)
                        continue;
                    int first = unchecked((int)surf.MsviFirstIndex);
                    int count = surf.IndexCount;

                    // Validate slice inside indices buffer
                    if (first < 0 || count < 3 || first + count > indices.Count)
                        continue;

                    int triCount = count / 3;
                    for (int i = 0; i < triCount; i++)
                    {
                        int baseIdx = first + i * 3;
                        int ga = indices[baseIdx];
                        int gb = indices[baseIdx + 1];
                        int gc = indices[baseIdx + 2];

                        // Validate vertex bounds (skip invalid to avoid (0,0,0) artifacts)
                        if (ga < 0 || gb < 0 || gc < 0 || ga >= verts.Count || gb >= verts.Count || gc >= verts.Count)
                            continue;

                        // Map to local vertex buffer
                        int la = GetOrAddLocal(ga, verts, localVerts, map);
                        int lb = GetOrAddLocal(gb, verts, localVerts, map);
                        int lc = GetOrAddLocal(gc, verts, localVerts, map);

                        localTris.Add((la, lb, lc));
                    }
                }

                if (localTris.Count > 0 && localVerts.Count > 0)
                {
                    result.Add(new AssembledObject(name, localVerts, localTris));
                }
            }

            return result;
        }

        private static int GetOrAddLocal(int gIndex, List<System.Numerics.Vector3> gVerts,
            List<System.Numerics.Vector3> lVerts, Dictionary<int, int> map)
        {
            if (map.TryGetValue(gIndex, out int li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIndex]);
            map[gIndex] = li;
            return li;
        }
    }
}
