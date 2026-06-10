using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Text.RegularExpressions;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Writes a single master OBJ containing geometry from every supported PM4 chunk type across the full 64×64 tile quilt.
    /// Each tile/chunk pair is emitted as its own object (<c>o CHUNK_tileX_tileY</c>) so it can be isolated in viewers.
    /// </summary>
    public static class UnifiedQuiltExporter
    {
        private const float AdtTileSize = 533.3333f; // Same constant used elsewhere
        private static readonly Regex TileRegex = new("_(?<x>\\d{2})_(?<y>\\d{2})\\.pm4$", RegexOptions.IgnoreCase | RegexOptions.Compiled);

        private enum ChunkKind { MSVT, MSPV, MSCN, MPRL }

        /// <summary>
        /// Generates a master quilt OBJ with all chunk data.
        /// </summary>
        /// <param name="pm4RootDir">Directory (or single file path) containing *.pm4 files.</param>
        /// <param name="outputObjPath">Destination OBJ file path.</param>
        public static void Export(string pm4RootDir, string outputObjPath)
        {
            string searchRoot = pm4RootDir;
            if (File.Exists(pm4RootDir))
                searchRoot = Path.GetDirectoryName(pm4RootDir)!;

            // Build mapping of tile -> file path (first file wins)
            var fileByTile = new Dictionary<(int x, int y), string>();
            foreach (var path in Directory.EnumerateFiles(searchRoot, "*.pm4", SearchOption.AllDirectories))
            {
                var m = TileRegex.Match(Path.GetFileName(path));
                if (!m.Success) continue;
                int tx = int.Parse(m.Groups["x"].Value, CultureInfo.InvariantCulture);
                int ty = int.Parse(m.Groups["y"].Value, CultureInfo.InvariantCulture);
                if (!fileByTile.ContainsKey((tx, ty)))
                    fileByTile[(tx, ty)] = path;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputObjPath)!);
            using var writer = new StreamWriter(outputObjPath);
            writer.WriteLine("# Master Quilt OBJ – all chunk geometry");
            writer.WriteLine($"# Generated {DateTime.Now:O}");

            int vertexCount = 0; // running vertex index for face offset (not used yet)

            for (int tileY = 0; tileY < 64; tileY++)
            {
                for (int tileX = 0; tileX < 64; tileX++)
                {
                    float offsetX = tileX * AdtTileSize;
                    float offsetY = (63 - tileY) * AdtTileSize; // north-up

                    if (!fileByTile.TryGetValue((tileX, tileY), out var pm4Path))
                        continue; // non-existent tile, skip

                    PM4File? pm4;
                    try { pm4 = PM4File.FromFile(pm4Path); }
                    catch { continue; }

                    WriteChunk(writer, "MSPV", pm4.MSPV?.Vertices, v => Pm4CoordinateTransforms.FromMspvVertex(v), offsetX, offsetY, ref vertexCount, tileX, tileY);
                    WriteChunk(writer, "MSVT", pm4.MSVT?.Vertices, v => Pm4CoordinateTransforms.FromMsvtVertexSimple(v), offsetX, offsetY, ref vertexCount, tileX, tileY);
                    WriteChunk(writer, "MSCN", pm4.MSCN?.ExteriorVertices, v => Pm4CoordinateTransforms.FromMscnVertex(v), offsetX, offsetY, ref vertexCount, tileX, tileY);
                }
            }
        }

        private static void WriteChunk<T>(StreamWriter w, string chunkName, IReadOnlyList<T>? data, Func<T, Vector3> transform, float ox, float oy, ref int index, int tileX, int tileY)
        {
            if (data == null || data.Count == 0) return;
            w.WriteLine($"o {chunkName}_{tileX}_{tileY}");
            bool applyOffset = chunkName is "MSPV" or "MSVT"; // only these are tile-relative
            foreach (var item in data)
            {
                var v = transform(item);
                float x = applyOffset ? v.X + ox : v.X;
                float y = applyOffset ? v.Y + oy : v.Y;
                w.WriteLine($"v {x.ToString(CultureInfo.InvariantCulture)} {y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                index++;
            }
            w.WriteLine();
        }
    }
}
