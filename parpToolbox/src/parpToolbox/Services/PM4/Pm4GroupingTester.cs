using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Quickly exports geometry grouped by MSUR.SurfaceGroupKey (flags byte 0x00).
    /// Helpful to visualise the hierarchy (e.g. subdivision 24 vs. higher-level groups).
    /// </summary>
    internal static class Pm4GroupingTester
    {
        /// <param name="scene">Loaded PM4 scene (global). Must have Vertices, Indices, Surfaces populated.</param>
        /// <param name="outputDir">Where to write OBJ files (created if needed).</param>
        /// <param name="writeFaces">True to export faces, false for point cloud.</param>
        /// <param name="minGroup">Only export groups whose byte value is >= minGroup when specified; otherwise export all.</param>
        public static void ExportBySurfaceGroupKey(Pm4Scene scene, string outputDir, bool writeFaces = true, byte? minGroup = null)
        {
            if (scene.Surfaces.Count == 0)
                throw new InvalidOperationException("Scene has no MSUR entries – cannot group by SurfaceGroupKey.");

            Directory.CreateDirectory(outputDir);

            // Build triangles list per group key.
            var groups = new Dictionary<byte, List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<byte, HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                byte gKey = surf.SurfaceGroupKey;
                if (minGroup.HasValue && gKey < minGroup.Value)
                    continue; // skip coarse groups if caller only wants fine ones

                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // invalid – skip

                if (scene.Indices.Count == 0)
                    continue; // no flat index buffer – skip face generation

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    if (baseIdx + 2 >= scene.Indices.Count)
                        continue; // out of range – skip

                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    if (!groups.TryGetValue(gKey, out var list))
                    {
                        list = new List<(int, int, int)>();
                        groups[gKey] = list;
                        usedVertsPerGroup[gKey] = new HashSet<int>();
                    }

                    list.Add((ia, ib, ic));
                    usedVertsPerGroup[gKey].Add(ia);
                    usedVertsPerGroup[gKey].Add(ib);
                    usedVertsPerGroup[gKey].Add(ic);
                }
            }

            Console.WriteLine($"[GroupingTester] Discovered {groups.Count} distinct group keys.");

            foreach (var kvp in groups.OrderBy(g => g.Key))
            {
                byte key = kvp.Key;
                var faces = kvp.Value;
                var used = usedVertsPerGroup[key];

                string baseName = $"group_{key}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                // Map original vertex index -> OBJ index (1-based)
                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-test-grouping OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                foreach (var vIdx in used)
                {
                    if (vIdx < 0 || vIdx >= scene.Vertices.Count)
                        continue; // skip invalid vertex refs

                    remap[vIdx] = nextIdx++;
                    Vector3 v = scene.Vertices[vIdx];
                    // Flip X for world to OBJ convention
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                if (writeFaces)
                {
                    foreach (var (A, B, C) in faces)
                    {
                        if (remap.TryGetValue(A, out var ra) &&
                            remap.TryGetValue(B, out var rb) &&
                            remap.TryGetValue(C, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in used)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {used.Count}, faces {faces.Count})");
            }
        }
        /// <summary>
        /// Exports geometry grouped by a composite key consisting of MSUR.SurfaceGroupKey and IndexCount.
        /// This is useful for testing the hypothesis that IndexCount further sub-divides SurfaceGroup containers
        /// into per-object meshes.
        /// </summary>
        public static void ExportByCompositeKey(Pm4Scene scene, string outputDir, bool writeFaces = true)
        {
            if (scene.Surfaces.Count == 0)
                throw new InvalidOperationException("Scene has no MSUR entries – cannot group by composite key.");

            Directory.CreateDirectory(outputDir);

            // Map composite key -> triangles and used vertices
            var groups = new Dictionary<(byte GroupKey, int IndexCount), List<(int A, int B, int C)>>();
            var usedVertsPerGroup = new Dictionary<(byte GroupKey, int IndexCount), HashSet<int>>();

            foreach (var surf in scene.Surfaces)
            {
                var compKey = (surf.SurfaceGroupKey, surf.IndexCount);

                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;

                if (first < 0 || first + count > scene.Indices.Count)
                    continue; // skip invalid range

                int triCount = count / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    int ia = scene.Indices[baseIdx];
                    int ib = scene.Indices[baseIdx + 1];
                    int ic = scene.Indices[baseIdx + 2];

                    if (!groups.TryGetValue(compKey, out var list))
                    {
                        list = new List<(int, int, int)>();
                        groups[compKey] = list;
                        usedVertsPerGroup[compKey] = new HashSet<int>();
                    }

                    list.Add((ia, ib, ic));
                    usedVertsPerGroup[compKey].Add(ia);
                    usedVertsPerGroup[compKey].Add(ib);
                    usedVertsPerGroup[compKey].Add(ic);
                }
            }

            Console.WriteLine($"[GroupingTester] Discovered {groups.Count} distinct composite keys.");

            foreach (var kvp in groups.OrderBy(k => k.Key.GroupKey).ThenBy(k => k.Key.IndexCount))
            {
                var key = kvp.Key;
                var faces = kvp.Value;
                var used = usedVertsPerGroup[key];

                string baseName = $"group_{key.GroupKey}_cnt_{key.IndexCount}";
                var objPath = Path.Combine(outputDir, baseName + ".obj");
                var mtlPath = Path.Combine(outputDir, baseName + ".mtl");

                var remap = new Dictionary<int, int>();
                int nextIdx = 1;

                using var sw = new StreamWriter(objPath);
                sw.WriteLine("# parpToolbox pm4-composite-grouping OBJ");
                sw.WriteLine($"mtllib {baseName}.mtl");
                sw.WriteLine($"g {baseName}");

                foreach (var vIdx in used)
                {
                    if (vIdx < 0 || vIdx >= scene.Vertices.Count)
                        continue;

                    remap[vIdx] = nextIdx++;
                    var v = scene.Vertices[vIdx];
                    sw.WriteLine(FormattableString.Invariant($"v {-v.X:F6} {v.Y:F6} {v.Z:F6}"));
                }

                sw.WriteLine("usemtl default");

                if (writeFaces)
                {
                    foreach (var (A, B, C) in faces)
                    {
                        if (remap.TryGetValue(A, out var ra) &&
                            remap.TryGetValue(B, out var rb) &&
                            remap.TryGetValue(C, out var rc))
                        {
                            sw.WriteLine($"f {ra} {rb} {rc}");
                        }
                    }
                }
                else
                {
                    foreach (var vIdx in used)
                    {
                        if (remap.TryGetValue(vIdx, out var rp))
                            sw.WriteLine($"p {rp}");
                    }
                }

                File.WriteAllText(mtlPath, "newmtl default\nKd 0.8 0.8 0.8\n");
                Console.WriteLine($"  Wrote {objPath} (verts {used.Count}, faces {faces.Count})");
            }
        }
    }
}
