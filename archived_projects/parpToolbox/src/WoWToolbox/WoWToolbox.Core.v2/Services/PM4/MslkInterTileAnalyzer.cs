using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Generates a CSV describing every MSLK entry and whether it links to the same tile or a different tile.
    /// Helps reveal cross-tile object relationships.
    /// </summary>
    public static class MslkInterTileAnalyzer
    {
        public sealed record Row(
            string TileName,
            int SourceTileX,
            int SourceTileY,
            uint GroupId,
            uint LinkIdHex,
            int TargetTileX,
            int TargetTileY,
            bool IsSameTile,
            bool HasGeometry,
            byte Flags,
            byte SequencePosition,
            int MspiFirstIndex,
            byte MspiIndexCount,
            ushort MsurIndex);

        private static (int x, int y) ParseTileName(string tileName)
        {
            // Expecting   something_YY_XX  (matching legacy development_13_36)
            // Split by '_' and take last two parts as ints
            var parts = tileName.Split('_');
            if (parts.Length < 3)
                return (-1, -1);
            if (int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int y) &&
                int.TryParse(parts[^2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int x))
            {
                return (x, y);
            }
            return (-1, -1);
        }

        private static (int x, int y) DecodeLinkId(uint linkId)
        {
            // 0xFFFFYYXX, little-endian order for low word: low byte = XX, next byte = YY
            int xx = (int)(linkId & 0xFF);
            int yy = (int)((linkId >> 8) & 0xFF);
            return (xx, yy);
        }

        /// <summary>
        /// Builds inter-tile rows from the supplied PM4 files.
        /// </summary>
        public static IEnumerable<Row> Build(IEnumerable<(PM4File file, string tileName)> pm4s)
        {
            foreach (var (pm4, name) in pm4s)
            {
                var (tileX, tileY) = ParseTileName(name);
                if (pm4.MSLK?.Entries == null)
                    continue;
                for (int i = 0; i < pm4.MSLK.Entries.Count; i++)
                {
                    var e = pm4.MSLK.Entries[i];
                    var (destX, destY) = DecodeLinkId(e.LinkIdRaw);
                    bool sameTile = tileX == destX && tileY == destY;
                    bool hasGeom = e.MspiFirstIndex >= 0;
                    yield return new Row(name, tileX, tileY, e.GroupObjectId, e.LinkIdRaw, destX, destY, sameTile, hasGeom,
                        e.Unknown_0x00, e.Unknown_0x01, e.MspiFirstIndex, e.MspiIndexCount, e.Unknown_0x10);
                }
            }
        }

        /// <summary>
        /// Writes the inter-tile rows to CSV at the given path.
        /// </summary>
        public static void WriteCsv(IEnumerable<Row> rows, string csvPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("Tile,SrcX,SrcY,GroupId,LinkIdHex,DestX,DestY,IsSameTile,HasGeometry,Flags,SeqPos,MspiFirstIndex,MspiCount,MsurIndex");
            foreach (var r in rows)
            {
                sw.WriteLine(string.Join(',', new object[]
                {
                    r.TileName,
                    r.SourceTileX,
                    r.SourceTileY,
                    r.GroupId,
                    $"0x{r.LinkIdHex:X8}",
                    r.TargetTileX,
                    r.TargetTileY,
                    r.IsSameTile ? 1 : 0,
                    r.HasGeometry ? 1 : 0,
                    r.Flags,
                    r.SequencePosition,
                    r.MspiFirstIndex,
                    r.MspiIndexCount,
                    r.MsurIndex
                }));
            }
        }
    }
}
