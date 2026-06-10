using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    /// <summary>
    /// Exports one 64×64 tile quilt OBJ for each selected PM4 chunk type.
    /// Useful for visualising spatial alignment and scale issues per chunk.
    /// </summary>
    public static class TileQuiltChunkExporter
    {
        private const float AdtTileSize = 533.3333f; // WoW server-space tile size
        private static readonly Regex TileRegex = new(@"_(?<x>\d{2})_(?<y>\d{2})\.pm4$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private enum ChunkKind { MSVT, MSPV, MSCN, MPRL }

        /// <summary>
        /// Generates OBJ quilts in <paramref name="outputDir"/> named <c>tile_quilt_CHUNK.obj</c>.
        /// Currently supports MSVT, MSPV, MSCN, and MPRL.
        /// </summary>
        public static void ExportChunkQuilts(string pm4RootDir, string outputDir)
        {
            if (!Directory.Exists(pm4RootDir)) throw new DirectoryNotFoundException(pm4RootDir);
            Directory.CreateDirectory(outputDir);

            // Index file paths by tile coordinate for quick access
            var fileByTile = new Dictionary<(int x, int y), string>();
            foreach (var path in Directory.GetFiles(pm4RootDir, "*.pm4", SearchOption.AllDirectories))
            {
                var m = TileRegex.Match(Path.GetFileName(path));
                if (!m.Success) continue;
                int tx = int.Parse(m.Groups["x"].Value, CultureInfo.InvariantCulture);
                int ty = int.Parse(m.Groups["y"].Value, CultureInfo.InvariantCulture);
                fileByTile[(tx, ty)] = path;
            }

            foreach (ChunkKind kind in Enum.GetValues(typeof(ChunkKind)))
            {
                string outPath = Path.Combine(outputDir, $"tile_quilt_{kind}.obj");
                using var writer = new StreamWriter(outPath);
                writer.WriteLine($"# Quilt OBJ for chunk {kind} – generated {DateTime.Now:O}");

                int vertexIndex = 1;
                for (int tileY = 0; tileY < 64; tileY++)
                {
                    for (int tileX = 0; tileX < 64; tileX++)
                    {
                        float offsetX = tileX * AdtTileSize;
                        float offsetY = tileY * AdtTileSize;
                        writer.WriteLine($"g {kind}_Tile_{tileX}_{tileY}");

                        if (fileByTile.TryGetValue((tileX, tileY), out var filePath))
                        {
                            PM4File pm4;
                            try { pm4 = PM4File.FromFile(filePath); }
                            catch { goto placeholder; }

                            switch (kind)
                            {
                                case ChunkKind.MSVT when pm4.MSVT?.Vertices != null:
                                    foreach (var v in pm4.MSVT.Vertices)
                                        WriteVert(writer, Pm4CoordinateTransforms.FromMsvtVertexSimple(v), offsetX, offsetY);
                                    break;
                                case ChunkKind.MSPV when pm4.MSPV?.Vertices != null:
                                    foreach (var v in pm4.MSPV.Vertices)
                                        WriteVert(writer, Pm4CoordinateTransforms.FromMspvVertex(v), offsetX, offsetY);
                                    break;
                                case ChunkKind.MSCN when pm4.MSCN?.ExteriorVertices != null:
                                    foreach (var v in pm4.MSCN.ExteriorVertices)
                                        WriteVert(writer, Pm4CoordinateTransforms.FromMscnVertex(v), offsetX, offsetY);
                                    break;
                                case ChunkKind.MPRL when pm4.MPRL?.Entries != null:
                                    foreach (var e in pm4.MPRL.Entries)
                                        WriteVert(writer, Pm4CoordinateTransforms.FromMprlEntry(e), offsetX, offsetY);
                                    break;
                            }
                            continue;
                        }

                        placeholder:
                        // If no data, write one placeholder vertex at tile centre
                        var centre = new Vector3(offsetX + AdtTileSize * 0.5f, offsetY + AdtTileSize * 0.5f, 0);
                        writer.WriteLine($"v {centre.X.ToString(CultureInfo.InvariantCulture)} {centre.Y.ToString(CultureInfo.InvariantCulture)} {centre.Z.ToString(CultureInfo.InvariantCulture)}");
                        vertexIndex++;
                    }
                }
                Console.WriteLine($"✅ Wrote {outPath}");
            }
        }

        private static void WriteVert(StreamWriter w, Vector3 v, float offsetX, float offsetY)
        {
            var world = new Vector3(v.X + offsetX, v.Y + offsetY, v.Z);
            w.WriteLine($"v {world.X.ToString(CultureInfo.InvariantCulture)} {world.Y.ToString(CultureInfo.InvariantCulture)} {world.Z.ToString(CultureInfo.InvariantCulture)}");
        }
    }
}
