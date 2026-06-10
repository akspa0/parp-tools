using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;
using Warcraft.NET.Files.Structures; // for C3Vector

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Exports a watertight terrain mesh directly from raw PM4 tile data using MSPV/MSVT vertices and MSVI triangle indices.
    /// Vertices are assumed to already be in world coordinates – no per-tile offset is applied.
    /// If <paramref name="stitch"/> is true (default) vertices that share identical coordinates will be deduplicated so
    /// neighbouring tiles share edge vertices, yielding a single manifold mesh.
    /// </summary>
    public static class TerrainMeshExporter
    {
        private struct ExportOptions
        {
            public bool Stitch;
            public int? FilterTileX;
            public int? FilterTileY;
        }

        /// <summary>
        /// Entry point – gathers *.pm4 files under <paramref name="inputPath"/> and writes a single OBJ.
        /// </summary>
        public static void Export(string inputPath, string outputObjPath, bool stitch = true, int? tileX = null, int? tileY = null)
        {
            var opts = new ExportOptions { Stitch = stitch, FilterTileX = tileX, FilterTileY = tileY };

            // Collect PM4 paths
            var pm4Paths = Directory.EnumerateFiles(inputPath, "*.pm4", SearchOption.AllDirectories).ToList();
            if (pm4Paths.Count == 0)
                throw new IOException($"No PM4 files found in {inputPath}");

            // OBJ buffers
            var vertices = new List<Vector3>();
            var faces = new List<(int a, int b, int c)>();
            var vertLookup = new Dictionary<(float, float, float), int>(); // for dedup
            const float epsilon = 0.001f; // positional tolerance for dedup – PM4 vertices use 0.01 precision typically

            foreach (var path in pm4Paths)
            {
                if (!TryParseTileCoords(Path.GetFileNameWithoutExtension(path), out int tx, out int ty))
                    continue; // skip non-tile names
                if (opts.FilterTileX.HasValue && opts.FilterTileX.Value != tx) continue;
                if (opts.FilterTileY.HasValue && opts.FilterTileY.Value != ty) continue;

                PM4File pm4;
                try
                {
                    pm4 = PM4File.FromFile(path);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Warning: failed loading {Path.GetFileName(path)} – {ex.Message}");
                    continue;
                }

                // Gather tile vertices into a common Vector3 list regardless of chunk type
                List<Vector3> tileVerts = new();
                if (pm4.MSPV?.Vertices is { Count: > 0 } pvList)
                {
                    foreach (var v in pvList)
                        tileVerts.Add(new Vector3(v.X, v.Y, v.Z));
                }
                else if (pm4.MSVT?.Vertices is { Count: > 0 } vtList)
                {
                    foreach (var mv in vtList)
                        tileVerts.Add(new Vector3(mv.X, mv.Y, mv.Z));
                }
                if (tileVerts.Count == 0)
                    continue; // nothing to export

                // Build per-tile vertex index mapping → global
                var localToGlobal = new int[tileVerts.Count];
                for (int i = 0; i < tileVerts.Count; i++)
                {
                    Vector3 vec = tileVerts[i];
                    int globalIdx;
                    if (opts.Stitch)
                    {
                        var key = (MathF.Round(vec.X / epsilon) * epsilon,
                                   MathF.Round(vec.Y / epsilon) * epsilon,
                                   MathF.Round(vec.Z / epsilon) * epsilon);
                        if (!vertLookup.TryGetValue(key, out globalIdx))
                        {
                            globalIdx = vertices.Count;
                            vertices.Add(vec);
                            vertLookup[key] = globalIdx;
                        }
                    }
                    else
                    {
                        globalIdx = vertices.Count;
                        vertices.Add(vec);
                    }
                    localToGlobal[i] = globalIdx;
                }

                // Faces via MSVI
                var idxList = pm4.MSVI?.Indices;
                if (idxList != null && idxList.Count >= 3)
                {
                    for (int i = 0; i + 2 < idxList.Count; i += 3)
                    {
                        int ia = (int)idxList[i];
                        int ib = (int)idxList[i + 1];
                        int ic = (int)idxList[i + 2];
                        if (ia >= localToGlobal.Length || ib >= localToGlobal.Length || ic >= localToGlobal.Length)
                            continue; // skip invalid triplet
                        faces.Add((localToGlobal[ia] + 1, localToGlobal[ib] + 1, localToGlobal[ic] + 1)); // OBJ is 1-based
                    }
                }

                // ---------------- Collision ribbon based on real exterior verts ----------------
                if (pm4.MSCN?.ExteriorVertices is { Count: >= 3 } ext)
                {
                    // MSCN vertices are assumed to already be in world coords
                    List<int> ribbonIndices = new();
                    foreach (var raw in ext)
                    {
                        var v = new Vector3(raw.X, raw.Y - 0.02f, raw.Z); // slight offset
                        int idx;
                        if (opts.Stitch)
                        {
                            var key = (MathF.Round(v.X / epsilon) * epsilon,
                                       MathF.Round(v.Y / epsilon) * epsilon,
                                       MathF.Round(v.Z / epsilon) * epsilon);
                            if (!vertLookup.TryGetValue(key, out idx))
                            {
                                idx = vertices.Count;
                                vertices.Add(v);
                                vertLookup[key] = idx;
                            }
                        }
                        else
                        {
                            idx = vertices.Count;
                            vertices.Add(v);
                        }
                        ribbonIndices.Add(idx);
                    }

                    // Fan triangulation from first vertex
                    for (int i = 1; i + 1 < ribbonIndices.Count; i++)
                    {
                        faces.Add((ribbonIndices[0] + 1, ribbonIndices[i] + 1, ribbonIndices[i + 1] + 1));
                    }
                }
            }

            // Write OBJ
            Directory.CreateDirectory(Path.GetDirectoryName(outputObjPath)!);
            using var writer = new StreamWriter(outputObjPath);
            writer.WriteLine($"# terrain mesh generated {DateTime.UtcNow:O}");
            writer.WriteLine($"# vertices {vertices.Count} faces {faces.Count}");

            foreach (var v in vertices)
                writer.WriteLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            writer.WriteLine("g terrain");
            foreach (var (a,b,c) in faces)
                writer.WriteLine($"f {a} {b} {c}");
        }

        // ---------------- Helpers ----------------
        private static bool TryParseTileCoords(string stem, out int tileX, out int tileY)
        {
            tileX = tileY = 0;
            var parts = stem.Split('_');
            if (parts.Length < 2) return false;
            return int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY);
        }
    }
}
