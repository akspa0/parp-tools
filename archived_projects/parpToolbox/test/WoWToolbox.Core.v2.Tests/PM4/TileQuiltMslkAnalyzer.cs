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
    /// Utility helper that aggregates MSLK anchor points from <c>.pm4</c> files
    /// into a single OBJ so we can visually inspect spatial patterns across a
    /// full 64×64 ADT tile quilt.
    /// 
    /// THIS IS A MANUAL TOOL – call <see cref="ExportTileQuilt"/> from a test or
    /// console harness with appropriate paths. It is placed in the test project
    /// so it compiles with the rest of Core.v2 while avoiding multiple entry
    /// points.
    /// </summary>
    public static class TileQuiltMslkAnalyzer
    {
        private static readonly Regex TileRegex = new(@"_(?<x>\d{2})_(?<y>\d{2})\.pm4$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
        private const float AdtTileSize = 533.3333f; // WoW map tile edge length (server-space units).

        /// <summary>
        /// Scans <paramref name="pm4RootDir"/> for all <c>.pm4</c> files, resolves tile
        /// coordinates from their filenames (format *_xx_yy.pm4) and exports an
        /// OBJ containing the anchor position for every MSLK entry. Tiles that are
        /// missing any PM4 file are still represented by a single dummy vertex so
        /// the resulting quilt preserves the full 64×64 layout and does not appear
        /// skewed along a diagonal. Each tile is
        /// placed in world-space by applying <c>(x * 533.3333, y * 533.3333)</c>
        /// translation.
        /// </summary>
        /// <param name="pm4RootDir">Directory containing tile directories/files.</param>
        /// <param name="outputObjPath">Destination OBJ path.</param>
        /// <remarks>
        /// Only minimal error checking is performed – intended for offline
        /// analysis, not production pipelines.
        /// </remarks>
        public static void ExportTileQuilt(string pm4RootDir, string outputObjPath)
        {
            if (!Directory.Exists(pm4RootDir))
                throw new DirectoryNotFoundException(pm4RootDir);

            var pm4Files = Directory.GetFiles(pm4RootDir, "*.pm4", SearchOption.AllDirectories)
                                     .OrderBy(f => f) // deterministic order
                                     .ToArray();

            // Index by tile coordinate for quick lookup.
            var fileByTile = new Dictionary<(int x,int y), string>();
            foreach (var path in pm4Files)
            {
                var m = TileRegex.Match(Path.GetFileName(path));
                if (m.Success)
                {
                    int tx = int.Parse(m.Groups["x"].Value, CultureInfo.InvariantCulture);
                    int ty = int.Parse(m.Groups["y"].Value, CultureInfo.InvariantCulture);
                    fileByTile[(tx, ty)] = path;
                }
            }

            if (pm4Files.Length == 0)
                throw new InvalidOperationException($"No .pm4 files found under {pm4RootDir}");

            Console.WriteLine($"🔍 Aggregating {pm4Files.Length} PM4 files into quilt OBJ …");
            Directory.CreateDirectory(Path.GetDirectoryName(outputObjPath)!);

            using var writer = new StreamWriter(outputObjPath);
            writer.WriteLine($"# PM4 Tile Quilt MSLK Anchor Export – {DateTime.Now:O}");
            writer.WriteLine("o MSLK_Anchors_AllTiles");

            int globalVertexIndex = 1;
            // Traverse the entire 64×64 grid to ensure every tile appears even if no PM4 exists.
            for (int tileY = 0; tileY < 64; tileY++)
            {
                for (int tileX = 0; tileX < 64; tileX++)
                {
                    string? filePath = fileByTile.TryGetValue((tileX,tileY), out var v) ? v : null;

                    float offsetX = tileX * AdtTileSize; // east-west (positive X)
                    float offsetY = (63 - tileY) * AdtTileSize; // north-south – invert so Y=0 is north edge

                    writer.WriteLine($"g Tile_{tileX}_{tileY}");

                    if (filePath != null)
                    {
                        PM4File pm4;
                        try
                        {
                            pm4 = PM4File.FromFile(filePath);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"❌ Failed to parse {filePath}: {ex.Message}");
                            // fall through to placeholder vertex
                            pm4 = null;
                        }

                        if (pm4?.MSLK?.Entries != null && pm4.MSLK.Entries.Count > 0)
                        {
                            foreach (var entry in pm4.MSLK.Entries)
                            {
                                var anchor = ResolveMslkAnchor(entry, pm4);
                                if (anchor == null) continue;

                                // Apply correct axis offsets (X ⇄ Y swap)
                                var world = new Vector3(
                                    anchor.Value.X + offsetX, // east-west offset on X (tileX)
                                    anchor.Value.Y + offsetY, // north-south offset on Y (tileY)
                                    anchor.Value.Z);

                                writer.WriteLine($"v {world.X.ToString(CultureInfo.InvariantCulture)} {world.Y.ToString(CultureInfo.InvariantCulture)} {world.Z.ToString(CultureInfo.InvariantCulture)}");
                            }

                            globalVertexIndex += pm4.MSLK.Entries.Count;
                            continue;
                        }
                    }

                    // --- Placeholder for empty/missing tile ---
                    var placeholder = new Vector3(offsetX + AdtTileSize * 0.5f, offsetY + AdtTileSize * 0.5f, 0);
                    writer.WriteLine($"v {placeholder.X.ToString(CultureInfo.InvariantCulture)} {placeholder.Y.ToString(CultureInfo.InvariantCulture)} {placeholder.Z.ToString(CultureInfo.InvariantCulture)}");
                    globalVertexIndex++;
                }
            }

            Console.WriteLine($"✅ Quilt OBJ written → {outputObjPath}");
        }

        /// <summary>
        /// Attempts to obtain a representative position for an MSLK entry.
        /// For geometry nodes we sample the centroid of referenced MSPV vertices; for
        /// doodad nodes we use the referenced MSVT vertex via MSVI.
        /// </summary>
        private static Vector3? ResolveMslkAnchor(MSLKEntry entry, PM4File pm4)
        {
            // Doodad / navigation node – use Unknown_0x10 → MSVI → MSVT.
            if (entry.MspiFirstIndex == -1 && pm4.MSVI != null && pm4.MSVT != null)
            {
                if (entry.Unknown_0x10 >= pm4.MSVI.Indices.Count)
                    return null;
                var msviIndex = pm4.MSVI.Indices[entry.Unknown_0x10];
                if (msviIndex >= pm4.MSVT.Vertices.Count) return null;
                var v = pm4.MSVT.Vertices[(int)msviIndex];
                return Pm4CoordinateTransforms.FromMsvtVertexSimple(v);
            }

            // Geometry node – average MSPV vertices referenced by MSPI indices.
            if (entry.MspiFirstIndex >= 0 && pm4.MSPI != null && pm4.MSPV != null)
            {
                int start = entry.MspiFirstIndex;
                int count = entry.MspiIndexCount;
                if (start + count > pm4.MSPI.Indices.Count) return null;
                var indices = pm4.MSPI.Indices.Skip(start).Take(count).ToArray();
                if (indices.Length == 0) return null;

                Vector3 sum = Vector3.Zero;
                int valid = 0;
                foreach (var idx in indices)
                {
                    if (idx >= pm4.MSPV.Vertices.Count) continue;
                    var v = pm4.MSPV.Vertices[(int)idx];
                    sum += Pm4CoordinateTransforms.FromMspvVertex(v);
                    valid++;
                }
                if (valid == 0) return null;
                return sum / valid;
            }

            return null;
        }
    }
}
