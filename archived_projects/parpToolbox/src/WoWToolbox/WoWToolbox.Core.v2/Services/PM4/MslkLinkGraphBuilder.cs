using System;
using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Utilities;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Builds cross-tile linkage graphs from MSLK chunks.
    /// Groups entries that share the same <see cref="MSLKEntry.LinkIdRaw"/> across many PM4 tiles.
    /// </summary>
    public static class MslkLinkGraphBuilder
    {
        /// <summary>
        /// Scans the provided PM4 tiles and groups their <see cref="MSLKEntry"/> objects by <c>LinkIdRaw</c>.
        /// </summary>
        /// <param name="tiles">Sequence of (PM4File, tileName) tuples. Tile name is typically "development_00_01" etc.</param>
        /// <returns>A dictionary keyed by LinkIdRaw where each value is a list of linked entries spanning tiles.</returns>
        public static Dictionary<uint, List<MslkLinkedEntry>> Build(IEnumerable<(PM4File file, string tileName)> tiles)
        {
            if (tiles == null) throw new ArgumentNullException(nameof(tiles));

            var map = new Dictionary<uint, List<MslkLinkedEntry>>();

            foreach (var (file, tileName) in tiles)
            {
                if (file.MSLK == null || file.MSLK.Entries.Count==0) continue;

                // derive tile coords from filename if possible (xx_yy) else use LinkIdDecoder fallback
                var tileCoords = LinkIdDecoder.TryExtractTileCoords(tileName);
                foreach (var entry in file.MSLK.Entries)
                {
                    uint linkId = entry.LinkIdRaw;
                    if (linkId == 0 || linkId == 0xFFFFFFFF) continue; // skip empty / padding IDs

                    if (!map.TryGetValue(linkId, out var list))
                    {
                        list = new List<MslkLinkedEntry>();
                        map[linkId] = list;
                    }

                    list.Add(new MslkLinkedEntry(file, entry, tileCoords?.x, tileCoords?.y));
                }
            }

            return map;
        }

        /// <summary>
        /// Convenience wrapper for a single PM4 file.
        /// </summary>
        public static Dictionary<uint, List<MslkLinkedEntry>> Build(PM4File file, string tileName)
            => Build(new[] { (file, tileName) });
    }

    /// <summary>
    /// Represents an MSLK entry along with the tile it originates from.
    /// </summary>
    public sealed record MslkLinkedEntry(PM4File SourceFile, MSLKEntry Entry, int? TileX, int? TileY);
}
