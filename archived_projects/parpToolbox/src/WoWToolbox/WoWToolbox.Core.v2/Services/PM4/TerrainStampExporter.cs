using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.Transforms;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Services.ADT;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Exports a continuous flat base plate at z = 0 and stamps MSCN exterior geometry
    /// into it by creating vertical side faces between the plate and the MSCN vertices.
    /// The result is a watertight terrain mesh resembling an embossed clay stamp.
    /// </summary>
    public static class TerrainStampExporter
    {
        private const float AdtTileSize = 533.3333f; // WoW ADT tile size (in world units)
        private const int MapTiles = 64; // 64x64 tiles per map

        public static void Export(string pm4Directory, string outputObjPath, int? tileXFilter = null, int? tileYFilter = null, bool includePlate = true, bool includeStamp = true)
        {
            var fileCandidates = new Dictionary<(int x, int y), List<string>>();
            foreach (var path in Directory.EnumerateFiles(pm4Directory, "*.pm4", SearchOption.AllDirectories))
            {
                var name = Path.GetFileNameWithoutExtension(path);
                // Expecting something like Tile_32_15_MSCN.pm4 -> grab 32,15
                var parts = name.Split('_');
                if (parts.Length < 3) continue; // allow e.g. development_22_17.pm4
                if (int.TryParse(parts[^2], out int tx) && int.TryParse(parts[^1], out int ty))
                {
                    if (!fileCandidates.TryGetValue((tx,ty), out var list))
                        fileCandidates[(tx,ty)] = list = new List<string>();
                    list.Add(path);
                }
            }

            using var writer = new StreamWriter(outputObjPath);
            writer.WriteLine("# Terrain stamp OBJ generated " + DateTime.Now.ToString("O"));
            int vertIndex = 0;

            var plateVerts = new List<Vector3>();
            var plateTris  = new List<(int a,int b,int c)>();

            // --- Build plates ONLY for tiles that actually have MSCN data (and pass filter) ---
            if (includePlate)
            {
                // Build plates ONLY for tiles with MSCN data (and matching filter if provided)
                foreach (var tile in fileCandidates.Keys)
                {
                    int tx = tile.x;
                    int ty = tile.y;
                    if (tileXFilter.HasValue && tx != tileXFilter.Value) continue;
                    if (tileYFilter.HasValue && ty != tileYFilter.Value) continue;
                    AdtFlatPlateBuilder.BuildTile(tx, ty, plateVerts, plateTris);
                }

                writer.WriteLine("g white_plate");
                foreach (var v in plateVerts)
                    WriteVertex(writer, v, ref vertIndex);
                foreach (var (a, b, c) in plateTris)
                    writer.WriteLine($"f {a + 1} {b + 1} {c + 1}"); // OBJ is 1-based
            }

            if (!includeStamp) return; // nothing else to do

            // --- Stamp MSCN geometry ---
            foreach (var kv in fileCandidates)
            {
                int tileX = kv.Key.x;
                int tileY = kv.Key.y;
                foreach (var path in kv.Value)
                {
                try
                {
                    var pm4 = PM4File.FromFile(path);
                    if (pm4.MSCN?.ExteriorVertices == null) continue;
                        // Sort vertices clockwise around centroid for robust triangulation
                        var sortedVerts = SortPolygonClockwise(pm4.MSCN.ExteriorVertices);


                    writer.WriteLine($"g MSCN_Tile_{tileX}_{tileY}");

                    float offsetX = 0f; // MSCN vertices are already in world coordinates
                    float offsetY = 0f;
                    foreach (var vRaw in sortedVerts)
                    {
                        var local = Pm4CoordinateTransforms.FromMscnVertex(vRaw);
                        var top = new Vector3(local.X + offsetX, local.Y + offsetY, local.Z);
                        var bottom = new Vector3(local.X + offsetX, local.Y + offsetY, 0f);
                        WriteVertex(writer, bottom, ref vertIndex);
                        WriteVertex(writer, top, ref vertIndex);
                    }

                    int polyVertCount = sortedVerts.Count;
                    int start = vertIndex - polyVertCount * 2; // bottom/top pairs
                    // Triangulate top face (fan)
                    for (int i = 0; i < polyVertCount - 2; i++)
                    {
                        int a = start + 1 + i * 2;         // top vertices are odd indices
                        int b = start + 1 + (i + 1) * 2;
                        int c = start + 1 + (i + 2) * 2;
                        writer.WriteLine($"f {a} {b} {c}");
                    }
                    // Bottom face (reverse order for correct normal)
                    for (int i = 0; i < polyVertCount - 2; i++)
                    {
                        int a = start + (polyVertCount - 1 - i) * 2; // even indices are bottoms
                        int b = start + (polyVertCount - 2 - i) * 2;
                        int c = start + (polyVertCount - 3 - i) * 2;
                        writer.WriteLine($"f {a} {b} {c}");
                    }
                    // Side faces (quads split into two triangles)
                    for (int i = 0; i < polyVertCount; i++)
                    {
                        int next = (i + 1) % polyVertCount;
                        int bottomA = start + i * 2;
                        int topA = bottomA + 1;
                        int bottomB = start + next * 2;
                        int topB = bottomB + 1;
                        // First triangle
                        writer.WriteLine($"f {bottomA} {topA} {topB}");
                        // Second triangle
                        writer.WriteLine($"f {bottomA} {topB} {bottomB}");
                    }
                }
                catch
                {
                    // ignore parse errors and continue
                }
            }

            // ------- Local helpers -------
            static List<Vector3> SortPolygonClockwise(IReadOnlyList<Vector3> verts)
            {
                // project to XY plane (tiling plane)
                float cx = 0, cy = 0;
                foreach (var v in verts)
                {
                    cx += v.X;
                    cy += v.Y;
                }
                cx /= verts.Count;
                cy /= verts.Count;
                var ordered = verts
                    .Select(v => (v, angle: MathF.Atan2(v.Y - cy, v.X - cx)))
                    .OrderBy(t => t.angle)
                    .Select(t => t.v)
                    .ToList();
                return ordered;
            }
        }
    }

        private static void WriteBasePlate(StreamWriter w, ref int vertIndex)
        {
            w.WriteLine("g base_plate");
            float size = MapTiles * AdtTileSize;
            var v1 = new Vector3(0f, 0f, 0f);           // top-left (map origin)
            var v2 = new Vector3(size, 0f, 0f);         // top-right (east)
            var v3 = new Vector3(size, size, 0f);       // bottom-right (south-east)
            var v4 = new Vector3(0f, size, 0f);         // bottom-left (south)
            WriteVertex(w, v1, ref vertIndex);
            WriteVertex(w, v2, ref vertIndex);
            WriteVertex(w, v3, ref vertIndex);
            WriteVertex(w, v4, ref vertIndex);
            int s = vertIndex - 3; // index of v1
            w.WriteLine($"f {s} {s + 1} {s + 2}");
            w.WriteLine($"f {s} {s + 2} {s + 3}");
        }

        private static void WriteVertex(StreamWriter w, Vector3 v, ref int index)
        {
            w.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            index++;
        }
    }
}
