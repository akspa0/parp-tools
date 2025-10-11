using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Produces high-level summary statistics for each MSLK LinkId group across many tiles.
    /// Each row represents one logical object (LinkIdHex).
    /// </summary>
    public static class MslkPatternAnalyzer
    {
        /// <summary>
        /// A summary row for a single LinkId group.
        /// </summary>
        public sealed record Row(
            uint LinkIdHex,
            int EntryCount,
            int TileCount,
            int GeometryEntryCount,
            byte MinFlags,
            byte MaxFlags,
            byte MinSeqPos,
            byte MaxSeqPos,
            string TileList);

        /// <summary>
        /// Builds summary rows from a collection of PM4 files.
        /// </summary>
        public static IEnumerable<Row> Build(IEnumerable<(PM4File file, string tileName)> pm4s)
        {
            // Flatten all entries with parent context info
            var flat = pm4s.SelectMany(t => t.file.MSLK?.Entries
                                    .Select(e => (Entry: e, Tile: t.tileName)) ?? Enumerable.Empty<(Foundation.PM4.Chunks.MSLKEntry, string)>());

            var groups = flat.GroupBy(x => x.Entry.LinkIdRaw);

            foreach (var g in groups)
            {
                uint link = g.Key;
                int entryCount = g.Count();
                var tiles = g.Select(x => x.Tile).Distinct().ToList();
                int tileCount = tiles.Count;
                int geomCount = g.Count(x => x.Entry.MspiFirstIndex >= 0);
                byte minFlags = g.Min(x => x.Entry.Unknown_0x00);
                byte maxFlags = g.Max(x => x.Entry.Unknown_0x00);
                byte minSeq = g.Min(x => x.Entry.Unknown_0x01);
                byte maxSeq = g.Max(x => x.Entry.Unknown_0x01);
                string tileList = string.Join(";", tiles);
                yield return new Row(link, entryCount, tileCount, geomCount, minFlags, maxFlags, minSeq, maxSeq, tileList);
            }
        }

        /// <summary>
        /// Writes summary rows to a CSV file.
        /// </summary>
        public static void WriteCsv(IEnumerable<Row> rows, string csvPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("LinkIdHex,EntryCount,TileCount,GeometryCount,MinFlags,MaxFlags,MinSeqPos,MaxSeqPos,TileList");
            foreach (var r in rows.OrderBy(r => r.LinkIdHex))
            {
                sw.WriteLine(string.Join(',', new object[]
                {
                    $"0x{r.LinkIdHex:X8}",
                    r.EntryCount,
                    r.TileCount,
                    r.GeometryEntryCount,
                    r.MinFlags,
                    r.MaxFlags,
                    r.MinSeqPos,
                    r.MaxSeqPos,
                    r.TileList
                }));
            }
        }
    }
}
