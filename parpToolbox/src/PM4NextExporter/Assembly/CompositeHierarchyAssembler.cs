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
        private const int TileGridSize = 64;

        private static int? GetTileIdForIndex(Dictionary<int,int> tileIndexOffsetsByTileId, int firstIndex)
        {
            if (tileIndexOffsetsByTileId == null || tileIndexOffsetsByTileId.Count == 0) return null;
            int bestTile = -1;
            int bestOffset = int.MinValue;
            foreach (var kvp in tileIndexOffsetsByTileId)
            {
                int tileId = kvp.Key;
                int off = kvp.Value;
                if (off <= firstIndex && off > bestOffset)
                {
                    bestOffset = off;
                    bestTile = tileId;
                }
            }
            return bestTile >= 0 ? bestTile : (int?)null;
        }

        private static string TileSuffixForSurfaces(IEnumerable<MsurChunk.Entry> surfaces, Scene scene)
        {
            if (scene == null || scene.TileIndexOffsetByTileId == null || scene.TileIndexOffsetByTileId.Count == 0)
                return string.Empty;
            var counts = new Dictionary<int,int>();
            foreach (var surf in surfaces)
            {
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                var tileId = GetTileIdForIndex(scene.TileIndexOffsetByTileId, first);
                if (tileId.HasValue)
                {
                    counts[tileId.Value] = counts.TryGetValue(tileId.Value, out var c) ? c + 1 : 1;
                }
            }
            if (counts.Count == 0) return string.Empty;
            int dominant = counts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First().Key;
            int x = dominant % TileGridSize;
            int y = dominant / TileGridSize;
            return $"_X{x:00}_Y{y:00}";
        }

        private static int? DominantTileIdForSurfaces(IEnumerable<MsurChunk.Entry> surfaces, Scene scene)
        {
            if (scene == null || scene.TileIndexOffsetByTileId == null || scene.TileIndexOffsetByTileId.Count == 0)
                return null;
            var counts = new Dictionary<int,int>();
            foreach (var surf in surfaces)
            {
                if (surf.MsviFirstIndex > int.MaxValue) continue;
                int first = unchecked((int)surf.MsviFirstIndex);
                var tileId = GetTileIdForIndex(scene.TileIndexOffsetByTileId, first);
                if (tileId.HasValue)
                {
                    counts[tileId.Value] = counts.TryGetValue(tileId.Value, out var c) ? c + 1 : 1;
                }
            }
            if (counts.Count == 0) return null;
            int dominant = counts.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key).First().Key;
            return dominant;
        }

        public IEnumerable<AssembledObject> Assemble(Scene scene, Options options)
        {
            var result = new List<AssembledObject>();
            if (scene == null || scene.Surfaces.Count == 0)
                return result;

            var verts = scene.Vertices;
            var indices = scene.Indices;

            // Group key: top 24 bits (mask low 8)
            static uint Top24(uint key) => key & 0xFFFFFF00u;
            static byte Low8(uint key) => (byte)(key & 0xFF);

            // Optional: build dominant MSLK type per surface, using SurfaceRefIndex -> Type_0x01 votes
            // Default tag 0xFF means "untyped/unknown".
            var surfaceIndexByRef = new Dictionary<MsurChunk.Entry, int>();
            for (int i = 0; i < scene.Surfaces.Count; i++) surfaceIndexByRef[scene.Surfaces[i]] = i;

            var surfaceTypeTag = new byte[scene.Surfaces.Count];
            for (int i = 0; i < surfaceTypeTag.Length; i++) surfaceTypeTag[i] = 0xFF;
            if (options.CkSplitByType && scene.Links != null && scene.Links.Count > 0)
            {
                var typeCounts = new Dictionary<int, Dictionary<byte, int>>();
                foreach (var link in scene.Links)
                {
                    int sidx = link.SurfaceRefIndex;
                    if (sidx >= 0 && sidx < scene.Surfaces.Count)
                    {
                        byte t = link.Type_0x01;
                        if (!typeCounts.TryGetValue(sidx, out var tc)) { tc = new Dictionary<byte, int>(); typeCounts[sidx] = tc; }
                        tc[t] = tc.TryGetValue(t, out var c) ? c + 1 : 1;
                    }
                }
                foreach (var kvp in typeCounts)
                {
                    byte bestType = 0xFF;
                    int bestCount = -1;
                    foreach (var tkv in kvp.Value)
                    {
                        if (tkv.Value > bestCount || (tkv.Value == bestCount && tkv.Key < bestType))
                        {
                            bestType = tkv.Key;
                            bestCount = tkv.Value;
                        }
                    }
                    surfaceTypeTag[kvp.Key] = bestType;
                }
            }

            // 1) Partition by tile to prevent cross-tile merging (which created long horizontal slices)
            var byTile = scene.Surfaces.GroupBy(s =>
            {
                if (s.MsviFirstIndex > int.MaxValue) return (int?)null;
                int first = unchecked((int)s.MsviFirstIndex);
                return GetTileIdForIndex(scene.TileIndexOffsetByTileId, first);
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
                    // else: leave empty to avoid guessing
                }

                // 2) Within tile, group by top 24-bit CompositeKey
                var groups = tileGroup.GroupBy(s => Top24(s.CompositeKey)).OrderBy(g => g.Key);

                foreach (var grp in groups)
                {
                    var baseName = $"CK_{grp.Key >> 8:X6}"; // retained but not used in final name; kept for context
                if (options.CkSplitByType)
                {
                    // Partition this CK24 group by per-surface type tag
                    var subGroups = grp.GroupBy(surf =>
                    {
                        int idx = surfaceIndexByRef.TryGetValue(surf, out var ii) ? ii : -1;
                        return (idx >= 0 && idx < surfaceTypeTag.Length) ? surfaceTypeTag[idx] : (byte)0xFF;
                    }).OrderBy(g => g.Key);

                    foreach (var sub in subGroups)
                    {
                        // Further split by low 8 bits (orientation variant) and suffix _Oxx
                        var orientGroups = sub.GroupBy(s => Low8(s.CompositeKey)).OrderBy(g => g.Key);
                        foreach (var orient in orientGroups)
                        {
                            // Derive 12+12 view from a representative surface in this orient variant
                            var rep = orient.First();
                            uint key = rep.CompositeKey;
                            int hi12 = (int)((key >> 12) & 0xFFF);
                            int lo12 = (int)(key & 0xFFF);
                            // Use the tile group's prefix (authoritative tile)
                            var name = $"{tilePrefix}CK12_{hi12:03X}_{lo12:03X}_T{sub.Key:X2}_O{orient.Key:X2}";
                            var localVerts = new List<System.Numerics.Vector3>();
                            var localTris = new List<(int,int,int)>();
                            var map = new Dictionary<int,int>();

                            foreach (var surf in orient)
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
                            {
                                var obj = new AssembledObject(name, localVerts, localTris);
                                // super-object membership from orientation key
                                obj.Meta["superObjectId"] = $"O{orient.Key:X2}";
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
                    }
                }
                else
                {
                    // Split by low 8 bits (orientation variant) and suffix _Oxx
                    var orientGroups = grp.GroupBy(s => Low8(s.CompositeKey)).OrderBy(g => g.Key);
                    foreach (var orient in orientGroups)
                    {
                        var rep = orient.First();
                        uint key = rep.CompositeKey;
                        int hi12 = (int)((key >> 12) & 0xFFF);
                        int lo12 = (int)(key & 0xFFF);
                        // Use the tile group's prefix (authoritative tile)
                        var name = $"{tilePrefix}CK12_{hi12:03X}_{lo12:03X}_O{orient.Key:X2}";
                        var localVerts = new List<System.Numerics.Vector3>();
                        var localTris = new List<(int,int,int)>();
                        var map = new Dictionary<int,int>();

                        foreach (var surf in orient)
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
                        {
                            var obj = new AssembledObject(name, localVerts, localTris);
                            // super-object membership from orientation key
                            obj.Meta["superObjectId"] = $"O{orient.Key:X2}";
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
                }
                }
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
