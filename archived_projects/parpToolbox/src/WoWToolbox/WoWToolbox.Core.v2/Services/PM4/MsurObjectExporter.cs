using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using WoWToolbox.Core.v2.Foundation.Transforms;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Exports render-mesh objects grouped by the Unknown_0x1C field of MSUR surfaces.
    /// Each unique 32-bit value becomes one OBJ file so we can inspect what geometry
    /// that key represents.
    /// </summary>
    public static class MsurObjectExporter
    {
        public static void ExportBySurfaceKey(PM4File pm4File, string outputDir)
        {
            if (pm4File.MSUR?.Entries == null || pm4File.MSVI?.Indices == null || pm4File.MSVT?.Vertices == null)
            {
                Console.WriteLine("[MsurExporter] Required chunks (MSUR/MSVI/MSVT) are missing.");
                return;
            }

            Directory.CreateDirectory(outputDir);

            // Map key -> list of MSUR indices
            var keyMap = new Dictionary<uint, List<int>>();
            for (int i = 0; i < pm4File.MSUR.Entries.Count; i++)
            {
                uint key = pm4File.MSUR.Entries[i].Unknown_0x1C;
                if (!keyMap.TryAdd(key, new List<int> { i }))
                    keyMap[key].Add(i);
            }

            Console.WriteLine($"[MsurExporter] Found {keyMap.Count} unique Unknown_1C keys.");

            foreach (var kvp in keyMap)
            {
                uint key = kvp.Key;
                string fileName = $"msur_{key:X8}.obj";
                string outPath = Path.Combine(outputDir, fileName);
                ExportKey(pm4File, kvp.Value, outPath, key);
            }
        }

        private static void ExportKey(PM4File pm4File, List<int> surfaceIndices, string outPath, uint key)
        {
            var vertices = new List<Vector3>();
            var faces = new List<int>();

            foreach (int surfIdx in surfaceIndices)
            {
                var surf = pm4File.MSUR!.Entries[surfIdx];
                int first = (int)surf.MsviFirstIndex;
                int count = surf.IndexCount;
                if (count < 3 || count % 3 != 0) continue;
                int end = first + count;
                if (end > pm4File.MSVI!.Indices.Count) end = pm4File.MSVI.Indices.Count;
                // Determine best normal orientation once per surface
                Vector3 nSrc = new Vector3(surf.SurfaceNormalX, surf.SurfaceNormalY, surf.SurfaceNormalZ);
                bool usePlane = surf.FlagsOrUnknown_0x00 != 18 && nSrc.LengthSquared() > 1e-6f;
                Vector3 nFinal = nSrc;
                if (usePlane && ShouldSwapNormal(surf, pm4File))
                {
                    nFinal = new Vector3(nSrc.Y, nSrc.X, nSrc.Z);
#if DEBUG
                    Console.WriteLine($"[MsurExporter] Swapped XY normal for surface {surfIdx} (key 0x{key:X8})");
#endif
                }
                if (usePlane)
                    nFinal = Vector3.Normalize(nFinal);

                for (int m = first; m + 2 < end; m += 3)
                {
                    uint aIdx = pm4File.MSVI.Indices[m];
                    uint bIdx = pm4File.MSVI.Indices[m + 1];
                    uint cIdx = pm4File.MSVI.Indices[m + 2];

                    int AddWithPlane(uint idx)
                    {
                        var vRaw = pm4File.MSVT!.Vertices[(int)idx];
                        var v = Pm4CoordinateTransforms.FromMsvtVertex(vRaw);

                        if (usePlane)
                        {
                            float d = Vector3.Dot(nFinal, v) - surf.SurfaceHeight;
                            v -= nFinal * d;
                        }
                        int local = vertices.Count;
                        vertices.Add(v);
                        return local;
                    }

                    faces.Add(AddWithPlane(aIdx));
                    faces.Add(AddWithPlane(bIdx));
                    faces.Add(AddWithPlane(cIdx));
                }
            }

            if (vertices.Count == 0)
            {
                Console.WriteLine($"[MsurExporter] Key 0x{key:X8}: no geometry, skipping");
                return;
            }

            using var writer = new StreamWriter(outPath);
            writer.WriteLine($"# MSUR Export key 0x{key:X8} | {surfaceIndices.Count} surfaces");
            foreach (var v in vertices)
                writer.WriteLine(FormattableString.Invariant($"v {v.X:F6} {v.Y:F6} {v.Z:F6}"));
            // no normals/uv
            for (int f = 0; f < faces.Count; f += 3)
            {
                int a = faces[f] + 1;
                int b = faces[f + 1] + 1;
                int c = faces[f + 2] + 1;
                writer.WriteLine($"f {a} {b} {c}");
            }
            Console.WriteLine($"[MsurExporter] Wrote {outPath} ({vertices.Count} verts, {faces.Count/3} tris)");
        }

        /// <summary>
        /// Heuristically decides whether swapping X/Y in the stored MSUR normal better fits the sampled vertices.
        /// </summary>
        private static bool ShouldSwapNormal(MsurEntry surf, PM4File pm4File)
        {
            int first = (int)surf.MsviFirstIndex;
            int sampleCount = Math.Min(surf.IndexCount, (byte)9); // up to 3 triangles
            if (sampleCount == 0 || pm4File.MSVI == null || pm4File.MSVT == null) return false;

            Vector3 nA = new Vector3(surf.SurfaceNormalX, surf.SurfaceNormalY, surf.SurfaceNormalZ);
            if (nA.LengthSquared() < 1e-6f) return false;
            Vector3 nB = new Vector3(nA.Y, nA.X, nA.Z);
            nA = Vector3.Normalize(nA);
            nB = Vector3.Normalize(nB);

            float errA = 0, errB = 0;
            int samples = 0;
            for (int i = 0; i < sampleCount && (first + i) < pm4File.MSVI.Indices.Count; i++)
            {
                uint idx = pm4File.MSVI.Indices[first + i];
                if (idx >= pm4File.MSVT.Vertices.Count) continue;
                var vRaw = pm4File.MSVT.Vertices[(int)idx];
                Vector3 v = Pm4CoordinateTransforms.FromMsvtVertex(vRaw);
                errA += MathF.Abs(Vector3.Dot(nA, v) - surf.SurfaceHeight);
                errB += MathF.Abs(Vector3.Dot(nB, v) - surf.SurfaceHeight);
                samples++;
            }
            if (samples == 0) return false;
            return errB < errA;
        }
    }
}
