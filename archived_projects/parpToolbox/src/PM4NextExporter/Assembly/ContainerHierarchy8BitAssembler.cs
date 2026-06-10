using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Groups leaf surfaces under the nearest container key in the 8-bit CompositeKey hierarchy.
    /// A container surface is identified by MspiFirstIndex == -1.
    /// The hierarchy is encoded big-endian: A.B.C.D (MSB..LSB).  We strip the lowest non-zero byte
    /// successively until we hit a key that has at least one container surface.
    /// If none found, we fall back to the original key.
    /// </summary>
    internal sealed class ContainerHierarchy8BitAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;
            var surfaces = scene.Surfaces;

            // Build set of explicit container keys (geometry-less nodes)
            var containerKeys = new HashSet<uint>(surfaces.Where(s => s.MsviFirstIndex == uint.MaxValue || (int)s.MsviFirstIndex == -1)
                                                          .Select(s => s.CompositeKey));

            // Helper to resolve leaf's grouping key
            uint Resolve(uint key)
            {
                uint k = key;
                for (int shift = 0; shift < 4; shift++)
                {
                    if (containerKeys.Contains(k))
                        return k;
                    // zero out lowest byte
                    k &= 0xFFFFFF00u;
                    if (containerKeys.Contains(k))
                        return k;
                    k &= 0xFFFF0000u;
                    if (containerKeys.Contains(k))
                        return k;
                    k &= 0xFF000000u;
                    if (containerKeys.Contains(k))
                        return k;
                    // if we reached here, break
                    break;
                }
                return key; // fallback
            }

            // Group leaves by resolved key
            var leaves = surfaces.Where(s => !(s.MsviFirstIndex == uint.MaxValue || (int)s.MsviFirstIndex == -1));
            var grouped = leaves.GroupBy(s => Resolve(s.CompositeKey));

            foreach (var grp in grouped)
            {
                var key = grp.Key;
                var name = $"CH8_{key:X8}";
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
