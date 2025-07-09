using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.RegularExpressions;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Writes a 64×64 tile quilt OBJ for each specified PM4 chunk type.
    /// Output file names: <c>tile_quilt_CHUNK.obj</c>.
    /// </summary>
    public static class TileQuiltChunkExporter
    {
        private const float AdtTileSize = 533.3333f; // size of one ADT tile
        private static readonly Regex TileRegex = new(@"_(?<x>\d{2})_(?<y>\d{2})\.pm4$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private enum ChunkKind { MSVT, MSPV, MSCN, MPRL }

        /// <summary>
        /// Generates OBJ quilts for each supported chunk kind.
        /// </summary>
        /// <param name="pm4RootDir">Directory containing *.pm4 files arranged in any structure.</param>
        /// <param name="outputDir">Destination directory where OBJ files will be written.</param>
        public static void ExportChunkQuilts(string pm4RootDir, string outputDir)
        {
            if (!Directory.Exists(pm4RootDir)) throw new DirectoryNotFoundException(pm4RootDir);
            Directory.CreateDirectory(outputDir);

            // Map each tile coordinate to its pm4 path (first found wins)
            var fileByTile = new Dictionary<(int x, int y), string>();
            foreach (var path in Directory.EnumerateFiles(pm4RootDir, "*.pm4", SearchOption.AllDirectories))
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
                writer.WriteLine($"# Quilt OBJ for chunk {kind} – {DateTime.Now:O}");
                int vertexCount = 0;

                for (int tileY = 0; tileY < 64; tileY++)
                {
                    for (int tileX = 0; tileX < 64; tileX++)
                    {
                        float offsetX = tileX * AdtTileSize;
                        float offsetY = tileY * AdtTileSize;
                        writer.WriteLine($"g {kind}_Tile_{tileX}_{tileY}");

                        if (fileByTile.TryGetValue((tileX, tileY), out var path))
                        {
                            PM4File pm4;
                            try { pm4 = PM4File.FromFile(path); }
                            catch { goto Placeholder; }

                            WriteChunk(kind, pm4, writer, offsetX, offsetY, ref vertexCount);
                            continue;
                        }

                        Placeholder:
                        // placeholder vertex so tile still visible
                        WriteVertex(writer, new Vector3(offsetX + AdtTileSize * 0.5f, offsetY + AdtTileSize * 0.5f, 0), ref vertexCount);
                    }
                }
                Console.WriteLine($"  • Wrote {outPath} ({vertexCount} verts)");
            }
        }

        private static void WriteChunk(ChunkKind kind, PM4File pm4, StreamWriter w, float ox, float oy, ref int count)
        {
            switch (kind)
            {
                case ChunkKind.MSVT when pm4.MSVT?.Vertices != null:
                    foreach (var v in pm4.MSVT.Vertices)
                        WriteVertex(w, Offset(Pm4CoordinateTransforms.FromMsvtVertexSimple(v), ox, oy), ref count);
                    break;
                case ChunkKind.MSPV when pm4.MSPV?.Vertices != null:
                    foreach (var v in pm4.MSPV.Vertices)
                        WriteVertex(w, Offset(Pm4CoordinateTransforms.FromMspvVertex(v), ox, oy), ref count);
                    break;
                case ChunkKind.MSCN when pm4.MSCN?.ExteriorVertices != null:
                    foreach (var v in pm4.MSCN.ExteriorVertices)
                        WriteVertex(w, Offset(Pm4CoordinateTransforms.FromMscnVertex(v), ox, oy), ref count);
                    break;
                case ChunkKind.MPRL when pm4.MPRL?.Entries != null:
                    foreach (var e in pm4.MPRL.Entries)
                        WriteVertex(w, Offset(Pm4CoordinateTransforms.FromMprlEntry(e), ox, oy), ref count);
                    break;
            }
        }

        private static Vector3 Offset(Vector3 v, float ox, float oy) => new(v.X + ox, v.Y + oy, v.Z);

        private static void WriteVertex(StreamWriter w, Vector3 v, ref int index)
        {
            w.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            index++;
        }
    }
}
