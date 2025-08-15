using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Groups geometry by MSUR surface key (high 16 bits of the CompositeKey / SurfaceKey field).
    /// This mirrors the grouping used in legacy DirectPm4Exporter to yield per-building objects.
    /// </summary>
    internal sealed class SurfaceKeyAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0 || scene.Indices.Count == 0 || scene.Vertices.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;

            // group by SurfaceKeyHigh16 (uint16) – this tends to correspond to individual game objects
            var groups = scene.Surfaces.GroupBy(s => s.SurfaceKeyHigh16).OrderBy(g => g.Key);

            foreach (var grp in groups)
            {
                var name = $"SurfaceKey_{grp.Key:X4}";
                var localVerts = new List<System.Numerics.Vector3>();
                var localTris = new List<(int A,int B,int C)>();
                var map = new Dictionary<int,int>();

                foreach (var surf in grp)
                {
                    if (surf.MsviFirstIndex > int.MaxValue) continue;
                    int first = unchecked((int)surf.MsviFirstIndex);
                    int count = surf.IndexCount;
                    if (first < 0 || count < 3 || first + count > indices.Count) continue;

                    int triCount = count / 3;
                    for (int t = 0; t < triCount; t++)
                    {
                        int baseIdx = first + t*3;
                        int a = indices[baseIdx];
                        int b = indices[baseIdx+1];
                        int c = indices[baseIdx+2];
                        if (a<0||b<0||c<0||a>=verts.Count||b>=verts.Count||c>=verts.Count)
                            continue;
                        int la = GetLocal(a, verts, localVerts, map);
                        int lb = GetLocal(b, verts, localVerts, map);
                        int lc = GetLocal(c, verts, localVerts, map);
                        localTris.Add((la,lb,lc));
                    }
                }
                if (localTris.Count>0)
                    result.Add(new AssembledObject(name, localVerts, localTris));
            }
            return result;
        }

        private static int GetLocal(int gIdx, List<System.Numerics.Vector3> gVerts, List<System.Numerics.Vector3> lVerts, Dictionary<int,int> map)
        {
            if (map.TryGetValue(gIdx, out var li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIdx]);
            map[gIdx]=li;
            return li;
        }
    }
}
