using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Experimental assembler that groups surfaces by the upper 24-bits of MSUR.CompositeKey.
    /// Idea: CompositeKey (SurfaceKey) encodes a 4-byte hierarchy.  Masking off the lowest byte
    /// tends to merge sub-components belonging to a single placed object while still separating
    /// neighbouring objects.  This is a heuristic until we wire in full MSLK container logic.
    /// </summary>
    internal sealed class CompositeHierarchyAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;

            // Group key: top 24 bits (mask low 8)
            static uint Top24(uint key) => key & 0xFFFFFF00u;

            var groups = scene.Surfaces.GroupBy(s => Top24(s.CompositeKey)).OrderBy(g => g.Key);

            foreach (var grp in groups)
            {
                var name = $"CK_{grp.Key >> 8:X6}";
                var localVerts = new List<System.Numerics.Vector3>();
                var localTris = new List<(int,int,int)>();
                var map = new Dictionary<int,int>();

                foreach (var surf in grp)
                {
                    if (surf.MsviFirstIndex > int.MaxValue)
                        continue;
                    int first = unchecked((int)surf.MsviFirstIndex);
                    int count = surf.IndexCount;
                    if (first < 0 || count < 3 || first + count > indices.Count)
                        continue;
                    int triCnt = count / 3;
                    for (int t = 0; t < triCnt; t++)
                    {
                        int baseIdx = first + t*3;
                        int a = indices[baseIdx];
                        int b = indices[baseIdx+1];
                        int c = indices[baseIdx+2];
                        if (a<0||b<0||c<0||a>=verts.Count||b>=verts.Count||c>=verts.Count)
                            continue;
                        int la = Map(a, verts, localVerts, map);
                        int lb = Map(b, verts, localVerts, map);
                        int lc = Map(c, verts, localVerts, map);
                        localTris.Add((la,lb,lc));
                    }
                }
                if (localTris.Count>0)
                    result.Add(new AssembledObject(name, localVerts, localTris));
            }
            return result;
        }

        private static int Map(int gIdx, List<System.Numerics.Vector3> gVerts, List<System.Numerics.Vector3> lVerts, Dictionary<int,int> map)
        {
            if (map.TryGetValue(gIdx, out var li)) return li;
            li = lVerts.Count;
            lVerts.Add(gVerts[gIdx]);
            map[gIdx] = li;
            return li;
        }
    }
}
