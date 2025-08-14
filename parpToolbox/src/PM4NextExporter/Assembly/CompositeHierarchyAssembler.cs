using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;
using ParpToolbox.Formats.P4.Chunks.Common;

namespace PM4NextExporter.Assembly
{
    /// <summary>
    /// Assembler that groups surfaces by MSUR.CompositeKey with selectable split modes.
    /// Default groups by the full 32-bit CompositeKey. Alternate modes:
    ///  - Hi24: group by upper 24 bits
    ///  - Low8: group by low 8 bits
    ///  - Hi24ThenLow8: first by upper 24, then split by low 8
    /// </summary>
    internal sealed class CompositeHierarchyAssembler : IAssembler
    {
        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0) return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;

            // 1) Partition by tile to prevent cross-tile merging
            var byTile = scene.Surfaces
                .GroupBy(s =>
                {
                    if (s.MsviFirstIndex > int.MaxValue) return (int?)null;
                    int first = unchecked((int)s.MsviFirstIndex);
                    return GetTileIdForIndex(scene.TileIndexOffsetByTileId, scene.TileIndexCountByTileId, first);
                })
                .OrderBy(g => g.Key.HasValue ? 0 : 1)
                .ThenBy(g => g.Key);

            foreach (var tileGroup in byTile)
            {
                int? groupTileId = tileGroup.Key;
                // Precompute tile prefix once per tile group using original coords if available
                string tilePrefix = string.Empty;
                if (options.NameObjectsWithTile && groupTileId.HasValue)
                {
                    if (scene.TileCoordByTileId != null && scene.TileCoordByTileId.TryGetValue(groupTileId.Value, out var coord))
                    {
                        tilePrefix = $"X{coord.X:00}_Y{coord.Y:00}_";
                    }
                }

                // 2) Within tile, group according to options.CkSplit
                switch (options.CkSplit)
                {
                    case CkSplitMode.Full:
                    {
                        var groups = tileGroup.GroupBy(s => s.CompositeKey).OrderBy(g => g.Key);
                        foreach (var grp in groups)
                        {
                            var rep = grp.First();
                            uint key = rep.CompositeKey;
                            var name = $"{tilePrefix}CK_{key:X8}";
                            AssembleGroup(grp, name, Low8(key), groupTileId, scene, verts, indices, result);
                        }
                        break;
                    }
                    case CkSplitMode.Hi24:
                    {
                        var groups = tileGroup.GroupBy(s => Top24(s.CompositeKey)).OrderBy(g => g.Key);
                        foreach (var grp in groups)
                        {
                            var rep = grp.First();
                            uint key = rep.CompositeKey;
                            uint hi24 = Top24(key) >> 8;
                            var name = $"{tilePrefix}CK24_{hi24:X6}";
                            AssembleGroup(grp, name, Low8(key), groupTileId, scene, verts, indices, result);
                        }
                        break;
                    }
                    case CkSplitMode.Low8:
                    {
                        var groups = tileGroup.GroupBy(s => Low8(s.CompositeKey)).OrderBy(g => g.Key);
                        foreach (var grp in groups)
                        {
                            var rep = grp.First();
                            uint key = rep.CompositeKey;
                            var name = $"{tilePrefix}O{grp.Key:X2}";
                            AssembleGroup(grp, name, grp.Key, groupTileId, scene, verts, indices, result);
                        }
                        break;
                    }
                    case CkSplitMode.Hi24ThenLow8:
                    default:
                    {
                        var hi24Groups = tileGroup.GroupBy(s => Top24(s.CompositeKey)).OrderBy(g => g.Key);
                        foreach (var grp in hi24Groups)
                        {
                            var orientGroups = grp.GroupBy(s => Low8(s.CompositeKey)).OrderBy(g => g.Key);
                            foreach (var orient in orientGroups)
                            {
                                var rep = orient.First();
                                uint key = rep.CompositeKey;
                                int hi12 = (int)((key >> 12) & 0xFFF);
                                int lo12 = (int)(key & 0xFFF);
                                var name = $"{tilePrefix}CK12_{hi12:03X}_{lo12:03X}_O{orient.Key:X2}";
                                AssembleGroup(orient, name, orient.Key, groupTileId, scene, verts, indices, result);
                            }
                        }
                        break;
                    }
                }
            }

            return result;
        }

        private static uint Top24(uint key) => key & 0xFFFFFF00u;
        private static byte Low8(uint key) => (byte)(key & 0xFF);
        private static int? GetTileIdForIndex(Dictionary<int,int> indexOffsetByTileId, Dictionary<int,int> indexCountByTileId, int absoluteIndex)
        {
            if (indexOffsetByTileId == null || indexCountByTileId == null) return null;
            foreach (var kv in indexOffsetByTileId)
            {
                var tileId = kv.Key;
                var offset = kv.Value;
                if (!indexCountByTileId.TryGetValue(tileId, out var count)) continue;
                if (absoluteIndex >= offset && absoluteIndex < offset + count) return tileId;
            }
            return null;
        }

        private static void AssembleGroup(
            IEnumerable<MsurChunk.Entry> group,
            string name,
            byte superId,
            int? groupTileId,
            Scene scene,
            List<System.Numerics.Vector3> verts,
            List<int> indices,
            List<AssembledObject> result)
        {
            var localVerts = new List<System.Numerics.Vector3>();
            var localTris = new List<(int,int,int)>();
            var map = new Dictionary<int,int>();

            foreach (var surf in group)
            {
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                int count = surf.IndexCount;
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
            {
                var obj = new AssembledObject(name, localVerts, localTris);
                obj.Meta["superObjectId"] = $"O{superId:X2}";
                if (groupTileId.HasValue)
                {
                    obj.Meta["tileId"] = groupTileId.Value.ToString();
                    if (scene.TileCoordByTileId != null && scene.TileCoordByTileId.TryGetValue(groupTileId.Value, out var coord))
                    {
                        obj.Meta["tileX"] = coord.X.ToString();
                        obj.Meta["tileY"] = coord.Y.ToString();
                    }
                }
                result.Add(obj);
            }
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
