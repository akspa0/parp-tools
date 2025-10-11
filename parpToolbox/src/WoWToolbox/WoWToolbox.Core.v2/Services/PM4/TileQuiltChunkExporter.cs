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
                        float offsetY = (63 - tileY) * AdtTileSize;

                        if (fileByTile.TryGetValue((tileX, tileY), out var path))
                        {
                            try
                            {
                                var pm4 = PM4File.FromFile(path);
                                writer.WriteLine($"g {kind}_Tile_{tileX}_{tileY}");
                                WriteChunk(kind, pm4, writer, offsetX, offsetY, ref vertexCount);
                            }
                            catch { /* skip file on parse error */ }
                        }
                    }
                }
                Console.WriteLine($"  • Wrote {outPath} ({vertexCount} verts)");
            }
        }

        private static void WriteChunk(ChunkKind kind, PM4File pm4, StreamWriter w, float ox, float oy, ref int count)
        {
            int before = count;
            switch (kind)
            {
                case ChunkKind.MSVT when pm4.MSVT?.Vertices != null:
                    foreach (var v in pm4.MSVT.Vertices)
                    {
                        var s = Pm4CoordinateTransforms.FromMsvtVertexSimple(v);
                        // Write ground plate duplicate vertex (z = 0)
                        var ground = new Vector3(s.X, s.Y, 0f);
                        WriteVertex(w, Offset(ground, ox, oy), ref count);
                        // Write original (embossed) vertex
                        WriteVertex(w, Offset(s, ox, oy), ref count);
                    }
                    break;
                case ChunkKind.MSPV when pm4.MSPV?.Vertices != null:
                    foreach (var v in pm4.MSPV.Vertices)
                    {
                        var p = Pm4CoordinateTransforms.FromMspvVertex(v);
                        var ground = new Vector3(p.X, p.Y, 0f);
                        WriteVertex(w, Offset(ground, ox, oy), ref count);
                        WriteVertex(w, Offset(p, ox, oy), ref count);
                    }
                    break;
                case ChunkKind.MSCN when pm4.MSCN?.ExteriorVertices != null:
                    foreach (var vRaw in pm4.MSCN.ExteriorVertices)
                    {
                        var vTop = Pm4CoordinateTransforms.FromMscnVertex(vRaw);
                        var vGround = new Vector3(vTop.X, vTop.Y, 0f);
                        // Ground plate vertex (z = 0)
                        WriteVertex(w, Offset(vGround, ox, oy), ref count);
                        // Embossed collision vertex
                        WriteVertex(w, Offset(vTop, ox, oy), ref count);
                    }
                    break;
                case ChunkKind.MPRL when pm4.MPRL?.Entries != null:
                    foreach (var e in pm4.MPRL.Entries)
                    {
                        var p = Pm4CoordinateTransforms.FromMprlEntry(e);
                        var ground = new Vector3(p.X, p.Y, 0f);
                        WriteVertex(w, Offset(ground, ox, oy), ref count);
                        WriteVertex(w, Offset(p, ox, oy), ref count);
                    }
                    break;
            }

            // Ensure a base ground plate if no vertices were emitted for this tile
            if (count == before)
            {
                var v1 = new Vector3(ox, oy, 0f);
                var v2 = new Vector3(ox + AdtTileSize, oy, 0f);
                var v3 = new Vector3(ox + AdtTileSize, oy + AdtTileSize, 0f);
                var v4 = new Vector3(ox, oy + AdtTileSize, 0f);
                WriteVertex(w, v1, ref count);
                WriteVertex(w, v2, ref count);
                WriteVertex(w, v3, ref count);
                WriteVertex(w, v4, ref count);
                int s = count - 3; // index of v1
                w.WriteLine($"f {s} {s + 1} {s + 2}");
                w.WriteLine($"f {s} {s + 2} {s + 3}");
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
