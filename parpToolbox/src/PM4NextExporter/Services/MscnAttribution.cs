using System;
using System.Collections.Generic;
using System.Linq;
using PM4NextExporter.Model;

namespace PM4NextExporter.Services
{
    internal static class MscnAttribution
    {
        /// <summary>
        /// Attribute MSCN vertices to assembled objects by scanning the original surface index slices
        /// for each object's MSUR.IndexCount group and mapping indices that reference the MSCN pool.
        /// </summary>
        /// <param name="scene">Loaded scene, with MSCN aggregation and tile maps populated for region loads.</param>
        /// <param name="objects">Assembled objects (expected to have meta key 'msur.indexcount').</param>
        /// <param name="remapApplied">True if MSCN remap was applied during load (typical for region loads).</param>
        /// <returns>Map of object -> set of MSCN vertex indices attributed to that object.</returns>
        public static Dictionary<AssembledObject, HashSet<int>> Attribute(Scene scene, IEnumerable<AssembledObject> objects, bool remapApplied)
        {
            var result = new Dictionary<AssembledObject, HashSet<int>>();

            if (scene == null || scene.MscnVertices == null || scene.MscnVertices.Count == 0)
                return result;

            var indices = scene.Indices ?? new List<int>();
            var surfaces = scene.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
            if (indices.Count == 0 || surfaces.Count == 0)
                return result;

            // Precompute tile index ranges for quick lookup
            var tileIndexRanges = new List<(int tileId, int start, int endExclusive)>();
            foreach (var kvp in scene.TileIndexOffsetByTileId)
            {
                var tileId = kvp.Key;
                var start = kvp.Value;
                if (!scene.TileIndexCountByTileId.TryGetValue(tileId, out var count)) continue;
                tileIndexRanges.Add((tileId, start, start + count));
            }

            // Sort ranges by start for faster linear scan
            tileIndexRanges.Sort((a, b) => a.start.CompareTo(b.start));

            int mscnStart = remapApplied ? (scene.Vertices.Count - scene.MscnVertices.Count) : -1;
            int totalMscn = scene.MscnVertices.Count;

            foreach (var obj in objects ?? Enumerable.Empty<AssembledObject>())
            {
                if (obj == null) continue;
                var set = new HashSet<int>();

                // Prefer composite-key attribution when available
                if (obj.Meta.TryGetValue("msur.ck", out var ckStr) && uint.TryParse(ckStr, out var ck))
                {
                    foreach (var surf in surfaces.Where(s => s.CompositeKey == ck))
                    {
                        if (surf.MsviFirstIndex > int.MaxValue) continue;
                        int first = unchecked((int)surf.MsviFirstIndex);
                        int count = surf.IndexCount;
                        if (first < 0 || count <= 0 || first + count > indices.Count)
                            continue;

                        int tileId = FindTileIdForIndex(first, tileIndexRanges);

                        // Tile vertex info (for overflow detection when remap is not applied)
                        int tileVertStart = 0, tileVertCount = 0;
                        if (tileId >= 0)
                        {
                            scene.TileVertexOffsetByTileId.TryGetValue(tileId, out tileVertStart);
                            scene.TileVertexCountByTileId.TryGetValue(tileId, out tileVertCount);
                        }

                        int end = first + count;
                        for (int i = first; i < end; i++)
                        {
                            int gi = indices[i];
                            if (remapApplied)
                            {
                                if (gi >= mscnStart)
                                {
                                    int idx = gi - mscnStart;
                                    if (idx >= 0 && idx < totalMscn)
                                        set.Add(idx);
                                }
                            }
                            else if (tileId >= 0)
                            {
                                int localIdx = gi - tileVertStart;
                                if (tileVertCount > 0 && localIdx >= tileVertCount)
                                {
                                    int offset = localIdx - tileVertCount;
                                    // Map into the aggregated MSCN array (fallback mapping consistent with remapper)
                                    int idx = offset % totalMscn;
                                    if (idx < 0) idx += totalMscn;
                                    set.Add(idx);
                                }
                            }
                        }
                    }
                }
                else if (obj.Meta.TryGetValue("msur.indexcount", out var keyStr) && int.TryParse(keyStr, out var key) && key > 0)
                {
                    // Fallback path: original IndexCount grouping
                    foreach (var surf in surfaces.Where(s => s.IndexCount == key))
                    {
                        if (surf.MsviFirstIndex > int.MaxValue) continue;
                        int first = unchecked((int)surf.MsviFirstIndex);
                        int count = surf.IndexCount;
                        if (first < 0 || count <= 0 || first + count > indices.Count)
                            continue;

                        int tileId = FindTileIdForIndex(first, tileIndexRanges);

                        int tileVertStart = 0, tileVertCount = 0;
                        if (tileId >= 0)
                        {
                            scene.TileVertexOffsetByTileId.TryGetValue(tileId, out tileVertStart);
                            scene.TileVertexCountByTileId.TryGetValue(tileId, out tileVertCount);
                        }

                        int end = first + count;
                        for (int i = first; i < end; i++)
                        {
                            int gi = indices[i];
                            if (remapApplied)
                            {
                                if (gi >= mscnStart)
                                {
                                    int idx = gi - mscnStart;
                                    if (idx >= 0 && idx < totalMscn)
                                        set.Add(idx);
                                }
                            }
                            else if (tileId >= 0)
                            {
                                int localIdx = gi - tileVertStart;
                                if (tileVertCount > 0 && localIdx >= tileVertCount)
                                {
                                    int offset = localIdx - tileVertCount;
                                    int idx = offset % totalMscn;
                                    if (idx < 0) idx += totalMscn;
                                    set.Add(idx);
                                }
                            }
                        }
                    }
                }

                obj.Meta["mscn.count"] = set.Count.ToString();
                result[obj] = set;
            }

            return result;
        }

        private static int FindTileIdForIndex(int globalFirstIndex, List<(int tileId, int start, int endExclusive)> ranges)
        {
            // Linear scan is usually fine for up to 4096 tiles; could be optimized to binary search if needed
            for (int i = 0; i < ranges.Count; i++)
            {
                var (tileId, start, end) = ranges[i];
                if (globalFirstIndex >= start && globalFirstIndex < end)
                    return tileId;
            }
            return -1;
        }
    }
}
