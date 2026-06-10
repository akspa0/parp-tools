using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Groups surfaces by significant byte-pair hierarchy of CompositeKey (0xAABBCCDD).
    /// If DD != 0 -> group by full key.
    /// Else if CC != 0 -> group by AABBCC00.
    /// Else if BB != 0 -> group by AABB0000.
    /// Else -> AA000000 (skips 00000000 root).
    /// </summary>
    internal sealed class CompositeBytePairAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var outList = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0)
                return outList;

            var verts = scene.Vertices;
            var indices = scene.Indices;
            var surf = scene.Surfaces;

            // Container detection (no geometry) to skip/identify grouping-only nodes
            var containers = new HashSet<uint>(surf.Where(s => s.MsviFirstIndex == uint.MaxValue || (int)s.MsviFirstIndex == -1)
                                                   .Select(s => s.CompositeKey));

            uint KeyFor(uint compKey)
            {
                if (compKey == 0) return 0;
                // Unpack bytes
                byte aa = (byte)(compKey >> 24);
                byte bb = (byte)(compKey >> 16);
                byte cc = (byte)(compKey >> 8);
                byte dd = (byte)(compKey);

                if (dd != 0) return compKey;
                if (cc != 0) return compKey & 0xFFFFFF00u; // zero DD
                if (bb != 0) return compKey & 0xFFFF0000u; // zero CC DD
                if (aa != 0) return compKey & 0xFF000000u; // zero BB CC DD
                return compKey; // fallthrough
            }

            var leaves = surf.Where(s => !(s.MsviFirstIndex == uint.MaxValue || (int)s.MsviFirstIndex == -1));
            var grouped = leaves.GroupBy(s => KeyFor(s.CompositeKey));

            foreach (var g in grouped)
            {
                uint gKey = g.Key;
                if (gKey == 0) continue; // skip world root
                var name = $"CBP_{gKey:X8}";
                var localVerts = new List<System.Numerics.Vector3>();
                var localTris = new List<(int,int,int)>();
                var map = new Dictionary<int,int>();

                foreach (var s in g)
                {
                    if (s.MsviFirstIndex > int.MaxValue) continue;
                    int first = unchecked((int)s.MsviFirstIndex);
                    int count = s.IndexCount;
                    if (first < 0 || count < 3 || first + count > indices.Count) continue;
                    int triCnt = count / 3;
                    for (int t = 0; t < triCnt; t++)
                    {
                        int baseIdx = first + t*3;
                        int a = indices[baseIdx];
                        int b = indices[baseIdx+1];
                        int c = indices[baseIdx+2];
                        if (a<0||b<0||c<0||a>=verts.Count||b>=verts.Count||c>=verts.Count) continue;
                        int la = Map(a, verts, localVerts, map);
                        int lb = Map(b, verts, localVerts, map);
                        int lc = Map(c, verts, localVerts, map);
                        localTris.Add((la,lb,lc));
                    }
                }
                if (localTris.Count > 0)
                    outList.Add(new AssembledObject(name, localVerts, localTris));
            }
            return outList;
        }

        private static int Map(int globalIdx, List<System.Numerics.Vector3> gVerts, List<System.Numerics.Vector3> lVerts, Dictionary<int,int> map)
        {
            if (map.TryGetValue(globalIdx, out int local)) return local;
            local = lVerts.Count;
            lVerts.Add(gVerts[globalIdx]);
            map[globalIdx] = local;
            return local;
        }
    }
}
