namespace ParpDataHarvester.Export.Geometry
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;
    using ParpDataHarvester.Export;
    using ParpToolbox.Formats.PM4;

    // Assembles raw geometry grouped by objects (MSLK.ParentIndex) or surfaces
    internal sealed class RawGeometryAssembler
    {
        internal sealed record AssembleResult(IReadOnlyList<Vector3> Vertices, List<GltfRawWriter.PrimitiveSpec> Primitives);

        public AssembleResult Assemble(Pm4Scene scene, string mode)
        {
            if (scene is null) throw new ArgumentNullException(nameof(scene));
            mode = (mode ?? "surfaces").Trim().ToLowerInvariant();

            // Positions are already X-flipped per Pm4Scene contract
            var vertices = scene.Vertices;

            // Build primitive list
            var primitives = new List<GltfRawWriter.PrimitiveSpec>(Math.Max(1, scene.Surfaces?.Count ?? 1));

            // Precompute surface ranges from MSUR
            var surfaces = scene.Surfaces ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MsurChunk.Entry>();
            var indices = scene.Indices ?? new List<int>();

            // Diagnostics counters
            int clampedSliceCount = 0;
            int droppedTriangleCount = 0;
            int emittedTriangleCount = 0;

            // Helper local function to slice indices safely
            List<uint> SliceIndices(uint first, int count, string label)
            {
                if (count <= 0) return new List<uint>();
                int start = Math.Max(0, (int)first);
                int end = Math.Min(indices.Count, start + count);
                int len = end - start;
                // keep triangle alignment
                int lenBeforeTriAlign = len;
                len -= len % 3;
                bool clamped = (start != (int)first) || (end != (int)first + count) || (len != lenBeforeTriAlign);
                if (clamped)
                {
                    clampedSliceCount++;
                    try
                    {
                        Console.WriteLine($"[assemble] clamp {label}: first={first} count={count} -> start={start} len={len} (indices={indices.Count})");
                    }
                    catch { /* best-effort logging */ }
                }
                if (len <= 0) return new List<uint>();

                var list = new List<uint>(len);
                int vCount = vertices.Count;
                for (int i = start; i < start + len; i += 3)
                {
                    int a = indices[i];
                    int b = indices[i + 1];
                    int c = indices[i + 2];
                    bool ok = a >= 0 && b >= 0 && c >= 0 && a < vCount && b < vCount && c < vCount;
                    if (!ok)
                    {
                        droppedTriangleCount++; // drop whole triangle if any vertex index is invalid
                        continue;
                    }
                    emittedTriangleCount++;
                    list.Add((uint)a);
                    list.Add((uint)b);
                    list.Add((uint)c);
                }
                return list;
            }

            // Build per-surface primitives first
            var surfacePrimInfo = new List<(int primIndex, int start, int end)>();
            foreach (var s in surfaces)
            {
                int count = s.IndexCount;
                // floor to full triangles
                count -= count % 3;
                if (count <= 0) continue;
                var prim = new GltfRawWriter.PrimitiveSpec
                {
                    Name = $"S_{s.CompositeKey:X8}",
                    Extras = new Dictionary<string, object>
                    {
                        ["surfaceKey"] = s.CompositeKey,
                        ["groupKey"] = s.GroupKey,
                        ["attrMask"] = s.AttributeMask,
                        ["firstIndex"] = s.MsviFirstIndex,
                        ["indexCount"] = s.IndexCount
                    }
                };
                prim.Indices.AddRange(SliceIndices(s.MsviFirstIndex, count, prim.Name));
                if (prim.Indices.Count == 0) continue; // skip empty after validation

                int primIdx = primitives.Count;
                primitives.Add(prim);
                int sStart = Math.Max(0, (int)s.MsviFirstIndex);
                int sEnd = Math.Min(indices.Count, sStart + count);
                surfacePrimInfo.Add((primIdx, sStart, sEnd));
            }

            if (mode == "objects")
            {
                // Build sorted link ranges
                var links = scene.Links ?? new List<ParpToolbox.Formats.P4.Chunks.Common.MslkEntry>();
                var linkRanges = new List<(int start, int end, uint parent)>();
                foreach (var link in links)
                {
                    if (!link.HasGeometry) continue; // container or no geom
                    int lFirst = link.MspiFirstIndex;
                    int lCount = link.MspiIndexCount;
                    if (lFirst < 0 || lCount <= 0) continue;
                    linkRanges.Add((lFirst, lFirst + lCount, link.ParentIndex));
                }
                linkRanges.Sort((a, b) => a.start.CompareTo(b.start));

                // Sort surfaces by start for sweep (keep original prim index to tag Extras)
                surfacePrimInfo.Sort((a, b) => a.start.CompareTo(b.start));

                int li = 0; // pointer into linkRanges
                foreach (var (primIndex, sStart, sEnd) in surfacePrimInfo)
                {
                    // advance li past links that end before this surface starts
                    while (li < linkRanges.Count && linkRanges[li].end <= sStart) li++;

                    // scan forward for overlaps
                    for (int lj = li; lj < linkRanges.Count; lj++)
                    {
                        var (lStart, lEnd, parent) = linkRanges[lj];
                        if (lStart >= sEnd) break; // no more possible overlaps
                        if (lEnd > sStart)
                        {
                            var extras = primitives[primIndex].Extras;
                            if (extras is null) break;
                            if (!extras.ContainsKey("parentIndex"))
                            {
                                extras["parentIndex"] = parent;
                            }
                            // we only need one parentIndex tag; break to avoid extra work
                            break;
                        }
                    }
                }

                // Merge by parentIndex: collapse many surface primitives into one per parent
                var grouped = new Dictionary<uint, List<uint>>();
                var orphans = new List<uint>();
                foreach (var prim in primitives)
                {
                    uint? parent = null;
                    if (prim.Extras != null && prim.Extras.TryGetValue("parentIndex", out var pObj) && pObj != null)
                    {
                        parent = pObj is uint u ? u : Convert.ToUInt32(pObj);
                    }
                    if (parent.HasValue)
                    {
                        if (!grouped.TryGetValue(parent.Value, out var list))
                        {
                            list = new List<uint>(prim.Indices.Count);
                            grouped[parent.Value] = list;
                        }
                        list.AddRange(prim.Indices);
                    }
                    else
                    {
                        orphans.AddRange(prim.Indices);
                    }
                }

                var merged = new List<GltfRawWriter.PrimitiveSpec>(grouped.Count + (orphans.Count > 0 ? 1 : 0));
                foreach (var kv in grouped)
                {
                    var m = new GltfRawWriter.PrimitiveSpec
                    {
                        Name = $"O_{kv.Key:X8}",
                        Extras = new Dictionary<string, object> { ["parentIndex"] = kv.Key }
                    };
                    m.Indices.AddRange(kv.Value);
                    if (m.Indices.Count > 0) merged.Add(m);
                }
                if (orphans.Count > 0)
                {
                    var m = new GltfRawWriter.PrimitiveSpec
                    {
                        Name = "O_ORPHANS",
                        Extras = new Dictionary<string, object>()
                    };
                    m.Indices.AddRange(orphans);
                    merged.Add(m);
                }

                primitives = merged;
            }

            // Emit summary diagnostics
            try
            {
                Console.WriteLine($"[assemble] diagnostics: clampedSlices={clampedSliceCount}, droppedTriangles={droppedTriangleCount}, emittedTriangles={emittedTriangleCount}");
            }
            catch { /* best-effort logging */ }

            return new AssembleResult(vertices, primitives);
        }
    }
}
