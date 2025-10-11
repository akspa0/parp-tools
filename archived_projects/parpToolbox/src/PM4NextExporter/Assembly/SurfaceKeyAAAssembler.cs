using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Groups geometry by the top byte (AA) of the 32-bit CompositeKey / SurfaceKey (0xAABBCCDD).
    /// Produces one OBJ per AA value: surfacekey_AA.obj.
    /// </summary>
    internal sealed class SurfaceKeyAAAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces == null || scene.Surfaces.Count == 0 || scene.Indices == null || scene.Vertices == null)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;
            var surfaces = scene.Surfaces;

            // Only include leaves (entries with valid geometry slice)
            var leaves = surfaces.Where(s => !(s.MsviFirstIndex == uint.MaxValue || (int)s.MsviFirstIndex == -1));

            // Group by AA (top 8 bits)
            var groups = leaves.GroupBy(s => (byte)(s.CompositeKey >> 24)).OrderBy(g => g.Key);

            foreach (var grp in groups)
            {
                byte aa = grp.Key;
                var name = $"surfacekey_{aa:X2}";
                var localVerts = new List<System.Numerics.Vector3>();
                var localTris = new List<(int A, int B, int C)>();
                var map = new Dictionary<int, int>(); // globalIndex -> localIndex

                foreach (var surf in grp)
                {
                    if (surf.MsviFirstIndex > int.MaxValue)
                        continue;
                    int first = unchecked((int)surf.MsviFirstIndex);
                    int count = surf.IndexCount;
                    if (first < 0 || count < 3 || first + count > indices.Count)
                        continue;

                    int triCount = count / 3;
                    for (int t = 0; t < triCount; t++)
                    {
                        int baseIdx = first + t * 3;
                        int ga = indices[baseIdx];
                        int gb = indices[baseIdx + 1];
                        int gc = indices[baseIdx + 2];
                        if (ga < 0 || gb < 0 || gc < 0 || ga >= verts.Count || gb >= verts.Count || gc >= verts.Count)
                            continue;

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
