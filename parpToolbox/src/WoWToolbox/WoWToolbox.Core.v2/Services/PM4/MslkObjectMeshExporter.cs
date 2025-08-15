using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Infrastructure;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Placeholder stub for the legacy <c>MslkObjectMeshExporter</c>.
    /// The full port of geometry extraction logic is sizeable – this stub is added
    /// so Core.v2 compiles while we incrementally migrate features.
    /// 
    /// TODO (Phase C): Copy and adapt the complete exporter logic from
    /// WoWToolbox.Core.Navigation.PM4.MslkObjectMeshExporter into this class,
    /// replacing legacy models/utilities with Core.v2 equivalents.
    /// </summary>
    public class MslkObjectMeshExporter
    {
        private readonly CoordinateService _coord = new CoordinateService();
        private readonly MslkHierarchyAnalyzer _analyzer = new MslkHierarchyAnalyzer();

        // Ensures that the requested output directory resides under ProjectOutput.RunDirectory.
        // If null/empty or pointing outside, it is redirected to a safe subfolder.
        private static string EnsureSafeOutputDir(string? requested, string fallbackSubdir)
        {
            if (string.IsNullOrWhiteSpace(requested))
                return ProjectOutput.GetPath(fallbackSubdir);

            string full = Path.GetFullPath(requested);
            string root = Path.GetFullPath(ProjectOutput.RunDirectory);

            // Redirect if outside the project_output tree to prevent contaminating input folders.
            if (!full.StartsWith(root, StringComparison.OrdinalIgnoreCase))
            {
                string safe = ProjectOutput.GetPath(fallbackSubdir, Path.GetFileName(full));
                Console.WriteLine($"[MslkExporter] Redirecting output directory '{requested}' → '{safe}' (outside project_output).");
                return safe;
            }
            return full;
        }

        /// <summary>
        /// Very first slice of the exporter: write one OBJ per GroupId that contains geometry.
        /// Only vertices are dumped; no faces yet.
        /// </summary>
        /// <summary>
        /// Writes one OBJ per GroupId (Unknown_0x04) – kept for debugging/compat.
        /// </summary>
        public void ExportAllGroupsAsObj(PM4File pm4File, string outputDirectory)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, "mslk_groups");
            if (!System.IO.Directory.Exists(outputDirectory))
                System.IO.Directory.CreateDirectory(outputDirectory);

            var grouping = _analyzer.GroupGeometryNodeIndicesByGroupId(pm4File.MSLK);
            if (grouping.Count == 0)
            {
                Console.WriteLine("[MslkExporter] No geometry groups found.");
                return;
            }

            foreach (var kvp in grouping)
            {
                uint groupId = kvp.Key;
                var nodeIndices = kvp.Value;
                string objPath = System.IO.Path.Combine(outputDirectory, $"group_{groupId:X4}.obj");
                using var writer = new System.IO.StreamWriter(objPath);
                writer.WriteLine($"# Auto-generated OBJ for Group 0x{groupId:X4}");
                writer.WriteLine("g group_" + groupId.ToString("X4"));

                var vertices = new System.Collections.Generic.List<System.Numerics.Vector3>();
                var faceIndices = new System.Collections.Generic.List<int>();

                // Build faces directly from MSPI triangle list
                foreach (int nodeIndex in nodeIndices)
                {
                    var entry = pm4File.MSLK!.Entries[nodeIndex];
                    if (pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null) continue;
                    if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;

                    // Local mapping: MSPV index -> local vertex index within this OBJ (0-based)
                    var vertexLookup = new System.Collections.Generic.Dictionary<uint, int>();

                    int rangeEnd = entry.MspiFirstIndex + entry.MspiIndexCount;
                    if (rangeEnd > pm4File.MSPI.Indices.Count) rangeEnd = pm4File.MSPI.Indices.Count;

                    for (int mspiIdx = entry.MspiFirstIndex; mspiIdx + 2 < rangeEnd; mspiIdx += 3)
                    {
                        uint vIdxA = pm4File.MSPI.Indices[mspiIdx];
                        uint vIdxB = pm4File.MSPI.Indices[mspiIdx + 1];
                        uint vIdxC = pm4File.MSPI.Indices[mspiIdx + 2];

                        int AddVertex(uint idx)
                        {
                            if (!vertexLookup.TryGetValue(idx, out int local))
                            {
                                var vRaw = pm4File.MSPV!.Vertices[(int)idx];
                                var v = _coord.FromMspvVertex(vRaw);
                                local = vertices.Count;
                                vertices.Add(v);
                                vertexLookup[idx] = local;
                            }
                            return vertexLookup[idx];
                        }

                        int a = AddVertex(vIdxA);
                        int b = AddVertex(vIdxB);
                        int c = AddVertex(vIdxC);

                        faceIndices.Add(a);
                        faceIndices.Add(b);
                        faceIndices.Add(c);
                    }
                }

                // Write gathered data
                // 1) vertices
                foreach (var vtx in vertices)
                {
                    writer.WriteLine(FormattableString.Invariant($"v {vtx.X:F6} {vtx.Y:F6} {vtx.Z:F6}"));
                }

                // 2) normals
                var normals = _coord.ComputeVertexNormals(vertices, faceIndices);
                foreach (var n in normals)
                {
                    writer.WriteLine(FormattableString.Invariant($"vn {n.X:F6} {n.Y:F6} {n.Z:F6}"));
                }

                // 3) faces (vertex and normal index are same ordering)
                for (int f = 0; f < faceIndices.Count; f += 3)
                {
                    int a = faceIndices[f] + 1;
                    int b = faceIndices[f+1] + 1;
                    int c = faceIndices[f+2] + 1;
                    writer.WriteLine($"f {a}//{a} {b}//{b} {c}//{c}");
                }

                Console.WriteLine($"[MslkExporter] Wrote {vertices.Count} vertices and {faceIndices.Count/3} faces to {System.IO.Path.GetFileName(objPath)}");
            }
        }
        /// <summary>
        /// Writes one OBJ per logical object grouped by ReferenceIndex (Unknown_0x10).
        /// </summary>
        /// <summary>
        /// Legacy wrapper kept for CLI compatibility. Delegates to MSUR grouping export.
        /// </summary>
        public void ExportAllObjectsAsObj(PM4File pm4, string outputDir) => ExportAllObjectsByMsurGroup(pm4, outputDir, includeM2: false);

        public void ExportAllObjectsByMsurGroup(PM4File pm4File, string outputDirectory, bool includeM2 = false)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, "mslk_objects");
            if (!System.IO.Directory.Exists(outputDirectory))
                System.IO.Directory.CreateDirectory(outputDirectory);

            // Build grouping by MSUR SurfaceGroupKey (msur_by_key)
            var grouping = new Dictionary<byte, List<int>>();
            if (pm4File.MSUR?.Entries == null)
            {
                Console.WriteLine("[MslkExporter] No MSUR data – cannot group by MSUR key, falling back to ReferenceIndex.");
                grouping = _analyzer.GroupGeometryNodeIndicesByObjectId(pm4File.MSLK)
                           .ToDictionary(k => (byte)(k.Key & 0xFF), v => v.Value);
            }
            else
            {
                // Pre-populate dictionary with empty lists for each key we encounter
                foreach (var msur in pm4File.MSUR.Entries)
                {
                    if (!includeM2 && msur.IsM2Bucket) continue; // skip M2 bucket unless requested
                    byte key = msur.SurfaceGroupKey;
                    if (!grouping.ContainsKey(key)) grouping[key] = new List<int>();
                }
                // Map each geometry node to a key by intersecting MSUR index ranges
                if (pm4File.MSLK?.Entries != null && pm4File.MSPI?.Indices != null)
                {
                    for (int nodeIdx = 0; nodeIdx < pm4File.MSLK.Entries.Count; nodeIdx++)
                    {
                        var e = pm4File.MSLK.Entries[nodeIdx];
                        if (e.MspiFirstIndex < 0 || e.MspiIndexCount == 0) continue;
                        uint first = (uint)e.MspiFirstIndex;
                        uint last = first + (uint)e.MspiIndexCount - 1;
                        // find first MSUR whose index range overlaps
                        foreach (var msur in pm4File.MSUR.Entries)
                        {
                            if (!includeM2 && msur.IsM2Bucket) continue;
                            uint sFirst = msur.MsviFirstIndex;
                            uint sLast = sFirst + msur.IndexCount - 1;
                            if (first >= sFirst && last <= sLast)
                            {
                                byte key = msur.SurfaceGroupKey;
                                grouping[key].Add(nodeIdx);
                                break;
                            }
                        }
                    }
                }
            }
            if (grouping.Count == 0)
            {
                Console.WriteLine("[MslkExporter] No object groupings found.");
                return;
            }

            foreach (var kvp in grouping)
            {
                byte groupKey = kvp.Key;
                var nodeIndices = kvp.Value;

                string objPath = System.IO.Path.Combine(outputDirectory, $"group_{groupKey:X2}.obj");
                using var writer = new System.IO.StreamWriter(objPath);
                writer.WriteLine($"# Auto-generated OBJ for MSUR GroupKey 0x{groupKey:X2}");
                writer.WriteLine("g group_" + groupKey.ToString("X2"));

                var vertices = new System.Collections.Generic.List<System.Numerics.Vector3>();
                var faceIndices = new System.Collections.Generic.List<int>();

                foreach (int nodeIndex in nodeIndices)
                {
                    var entry = pm4File.MSLK!.Entries[nodeIndex];
                    if (pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null) continue;
                    if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;

                    var vertexLookup = new System.Collections.Generic.Dictionary<uint, int>();

                    int rangeEnd = entry.MspiFirstIndex + entry.MspiIndexCount;
                    if (rangeEnd > pm4File.MSPI.Indices.Count) rangeEnd = pm4File.MSPI.Indices.Count;

                    for (int mspiIdx = entry.MspiFirstIndex; mspiIdx + 2 < rangeEnd; mspiIdx += 3)
                    {
                        uint vIdxA = pm4File.MSPI.Indices[mspiIdx];
                        uint vIdxB = pm4File.MSPI.Indices[mspiIdx + 1];
                        uint vIdxC = pm4File.MSPI.Indices[mspiIdx + 2];

                        int AddVertex(uint idx)
                        {
                            if (!vertexLookup.TryGetValue(idx, out int local))
                            {
                                var vRaw = pm4File.MSPV!.Vertices[(int)idx];
                                var v = _coord.FromMspvVertex(vRaw);
                                local = vertices.Count;
                                vertices.Add(v);
                                vertexLookup[idx] = local;
                            }
                            return vertexLookup[idx];
                        }

                        int a = AddVertex(vIdxA);
                        int b = AddVertex(vIdxB);
                        int c = AddVertex(vIdxC);

                        faceIndices.Add(a);
                        faceIndices.Add(b);
                        faceIndices.Add(c);
                    }
                }

                // ---- Add render mesh via MSUR→MSVI→MSVT ----
                if (pm4File.MSUR?.Entries != null && pm4File.MSVI?.Indices != null && pm4File.MSVT?.Vertices != null)
                {
                    foreach (var surface in pm4File.MSUR.Entries)
                    {
                        if (!includeM2 && surface.IsM2Bucket) continue;
                        if (surface.SurfaceGroupKey != groupKey) continue;
                        uint start = surface.MsviFirstIndex;
                        uint count = surface.IndexCount;
                        var vertexLookupRM = new Dictionary<uint,int>();
                        for (uint i = 0; i + 2 < count; i += 3)
                        {
                            uint idxA = pm4File.MSVI.Indices[(int)(start + i)];
                            uint idxB = pm4File.MSVI.Indices[(int)(start + i + 1)];
                            uint idxC = pm4File.MSVI.Indices[(int)(start + i + 2)];
                            int AddRMVertex(uint gi)
                            {
                                if (!vertexLookupRM.TryGetValue(gi, out int local))
                                {
                                    if (gi >= pm4File.MSVT.Vertices.Count) return -1;
                                    var vRaw = pm4File.MSVT.Vertices[(int)gi];
                                    var v = _coord.FromMsvtVertexSimple(vRaw);
                                    local = vertices.Count;
                                    vertices.Add(v);
                                    vertexLookupRM[gi] = local;
                                }
                                return vertexLookupRM[gi];
                            }
                            int ra = AddRMVertex(idxA);
                            int rb = AddRMVertex(idxB);
                            int rc = AddRMVertex(idxC);
                            if (ra>=0 && rb>=0 && rc>=0)
                            {
                                faceIndices.Add(ra);
                                faceIndices.Add(rb);
                                faceIndices.Add(rc);
                            }
                        }
                    }
                }

                foreach (var vtx in vertices)
                {
                    writer.WriteLine(FormattableString.Invariant($"v {vtx.X:F6} {vtx.Y:F6} {vtx.Z:F6}"));
                }

                var normals = _coord.ComputeVertexNormals(vertices, faceIndices);
                foreach (var n in normals)
                {
                    writer.WriteLine(FormattableString.Invariant($"vn {n.X:F6} {n.Y:F6} {n.Z:F6}"));
                }

                for (int f = 0; f < faceIndices.Count; f += 3)
                {
                    int a = faceIndices[f] + 1;
                    int b = faceIndices[f + 1] + 1;
                    int c = faceIndices[f + 2] + 1;
                    writer.WriteLine($"f {a}//{a} {b}//{b} {c}//{c}");
                }

                Console.WriteLine($"[MslkExporter] Wrote {vertices.Count} vertices and {faceIndices.Count / 3} faces to {System.IO.Path.GetFileName(objPath)}");
            }
        }

        // --- Additional grouping modes ---
        public void ExportAllFlagsAsObj(PM4File pm4File, string outputDirectory)
        {
            ExportByByteKey(pm4File, outputDirectory, _analyzer.GroupGeometryNodeIndicesByFlag(pm4File.MSLK), key => $"flag_{key:X2}.obj", "Flag");
        }

        public void ExportAllSubtypesAsObj(PM4File pm4File, string outputDirectory)
        {
            ExportByByteKey(pm4File, outputDirectory, _analyzer.GroupGeometryNodeIndicesBySubtype(pm4File.MSLK), key => $"sub_{key:X2}.obj", "Subtype");
        }

        public void ExportAllContainersAsObj(PM4File pm4File, string outputDirectory)
        {
            var grouping = _analyzer.GroupGeometryNodeIndicesByContainer(pm4File.MSLK);
            ExportGeneric(pm4File, outputDirectory, grouping, key => $"container_{key:X4}.obj", "Container");
        }

        // --- Helper methods to minimize duplication ---
        private void ExportByByteKey(PM4File pm4File, string outputDirectory, System.Collections.Generic.Dictionary<byte, System.Collections.Generic.List<int>> grouping, System.Func<byte,string> filenameFunc, string label)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, $"mslk_{label.ToLower()}s");
            var converted = grouping.ToDictionary(k => (ushort)k.Key, v => v.Value);
            ExportGeneric(pm4File, outputDirectory, converted, key => filenameFunc((byte)key), label);
        }

        private void ExportGeneric(PM4File pm4File, string outputDirectory, System.Collections.Generic.Dictionary<ushort, System.Collections.Generic.List<int>> grouping, System.Func<ushort,string> filenameFunc, string label)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, $"mslk_{label.ToLower()}s");
            if (!System.IO.Directory.Exists(outputDirectory))
                System.IO.Directory.CreateDirectory(outputDirectory);
            if (grouping.Count == 0)
            {
                Console.WriteLine($"[MslkExporter] No {label} groupings found.");
                return;
            }
            foreach (var kvp in grouping)
            {
                ushort key = kvp.Key;
                var nodeIndices = kvp.Value;
                string objPath = System.IO.Path.Combine(outputDirectory, filenameFunc(key));
                using var writer = new System.IO.StreamWriter(objPath);
                writer.WriteLine($"# Auto-generated OBJ for {label} 0x{key:X4}");
                writer.WriteLine("g " + label.ToLower() + "_" + key.ToString("X4"));
                var vertices = new System.Collections.Generic.List<System.Numerics.Vector3>();
                var faceIndices = new System.Collections.Generic.List<int>();
                foreach (int nodeIndex in nodeIndices)
                {
                    var entry = pm4File.MSLK!.Entries[nodeIndex];
                    if (pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null) continue;
                    if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;
                    var vertexLookup = new System.Collections.Generic.Dictionary<uint, int>();
                    int rangeEnd = entry.MspiFirstIndex + entry.MspiIndexCount;
                    if (rangeEnd > pm4File.MSPI.Indices.Count) rangeEnd = pm4File.MSPI.Indices.Count;
                    for (int mspiIdx = entry.MspiFirstIndex; mspiIdx + 2 < rangeEnd; mspiIdx += 3)
                    {
                        uint aIdx = pm4File.MSPI.Indices[mspiIdx];
                        uint bIdx = pm4File.MSPI.Indices[mspiIdx + 1];
                        uint cIdx = pm4File.MSPI.Indices[mspiIdx + 2];
                        int Add(uint idx)
                        {
                            if (!vertexLookup.TryGetValue(idx, out int local))
                            {
                                var vRaw = pm4File.MSPV!.Vertices[(int)idx];
                                var v = _coord.FromMspvVertex(vRaw);
                                local = vertices.Count;
                                vertices.Add(v);
                                vertexLookup[idx] = local;
                            }
                            return vertexLookup[idx];
                        }
                        faceIndices.Add(Add(aIdx));
                        faceIndices.Add(Add(bIdx));
                        faceIndices.Add(Add(cIdx));
                    }
                }
                foreach (var vtx in vertices)
                    writer.WriteLine(FormattableString.Invariant($"v {vtx.X:F6} {vtx.Y:F6} {vtx.Z:F6}"));
                var normals = _coord.ComputeVertexNormals(vertices, faceIndices);
                foreach (var n in normals)
                    writer.WriteLine(FormattableString.Invariant($"vn {n.X:F6} {n.Y:F6} {n.Z:F6}"));
                for (int f = 0; f < faceIndices.Count; f += 3)
                {
                    int a = faceIndices[f] + 1;
                    int b = faceIndices[f + 1] + 1;
                    int c = faceIndices[f + 2] + 1;
                    writer.WriteLine($"f {a}//{a} {b}//{b} {c}//{c}");
                }
                Console.WriteLine($"[MslkExporter] {label} 0x{key:X4}: {vertices.Count} verts, {faceIndices.Count / 3} tris");
            }
        }

        // Cluster containers that are spatially close and have sequential highRef values
        public void ExportAllClustersAsObj(PM4File pm4File, string outputDirectory, float distanceThreshold = 5f)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, "mslk_clusters");
            var containers = _analyzer.GroupGeometryNodeIndicesByContainer(pm4File.MSLK);
            if (containers.Count == 0)
            {
                Console.WriteLine("[MslkExporter] No containers found for clustering.");
                return;
            }
            // Build quick centre lookup for each container
            var containerCentres = new System.Collections.Generic.Dictionary<ushort, System.Numerics.Vector3>();
            foreach (var kvp in containers)
            {
                var verts = new System.Collections.Generic.List<System.Numerics.Vector3>();
                foreach (int idx in kvp.Value)
                {
                    var entry = pm4File.MSLK!.Entries[idx];
                    if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;
                    int end = entry.MspiFirstIndex + entry.MspiIndexCount;
                    if (end > pm4File.MSPI!.Indices.Count) end = pm4File.MSPI.Indices.Count;
                    for (int i = entry.MspiFirstIndex; i < end; i++)
                    {
                        uint vIdx = pm4File.MSPI.Indices[i];
                        var vRaw = pm4File.MSPV!.Vertices[(int)vIdx];
                        verts.Add(_coord.FromMspvVertex(vRaw));
                    }
                }
                if (verts.Count == 0) { containerCentres[kvp.Key] = System.Numerics.Vector3.Zero; continue; }
                var centre = new System.Numerics.Vector3();
                foreach (var v in verts) centre += v;
                centre /= verts.Count;
                containerCentres[kvp.Key] = centre;
            }
            // cluster grouping
            var clusters = new System.Collections.Generic.List<System.Collections.Generic.List<ushort>>();
            var visited = new System.Collections.Generic.HashSet<ushort>();
            // group by flag first
            var ordered = containers.Keys.OrderBy(k => k).ToList();
            foreach (var key in ordered)
            {
                if (visited.Contains(key)) continue;
                var cluster = new System.Collections.Generic.List<ushort> { key };
                visited.Add(key);
                byte flag = (byte)(key >> 8);
                byte highRef = (byte)(key & 0xFF);
                var centre = containerCentres[key];
                // look forward for neighbours within 2 highRef steps and close distance
                foreach (var cand in ordered)
                {
                    if (visited.Contains(cand)) continue;
                    if (((cand >> 8) != flag)) continue; // same flag only
                    byte candRef = (byte)(cand & 0xFF);
                    if (candRef >= highRef - 2 && candRef <= highRef + 2)
                    {
                        var candCentre = containerCentres[cand];
                        if (System.Numerics.Vector3.Distance(centre, candCentre) <= distanceThreshold)
                        {
                            cluster.Add(cand);
                            visited.Add(cand);
                        }
                    }
                }
                clusters.Add(cluster);
            }
            int clusterIdx = 0;
            foreach (var cluster in clusters)
            {
                var combined = new System.Collections.Generic.Dictionary<ushort, System.Collections.Generic.List<int>>();
                foreach (var key in cluster)
                {
                    combined[key] = containers[key];
                }
                var mergeMap = combined.SelectMany(kvp => kvp.Value).ToList();
                ExportMergedNodes(pm4File, mergeMap, System.IO.Path.Combine(outputDirectory, $"cluster_{clusterIdx:D3}.obj"));
                clusterIdx++;
            }
        }

        // Experimental: merge clusters whose AABBs overlap into higher-level objects
        public void ExportAllObjectClustersAsObj(PM4File pm4File, string outputDirectory)
        {
            outputDirectory = EnsureSafeOutputDir(outputDirectory, "mslk_object_clusters");
            // First, build clusters as before
            var tempDir = System.IO.Path.Combine(outputDirectory, "_temp");
            ExportAllClustersAsObj(pm4File, tempDir); // populates individual clusters on disk but we reuse centre map below

            var containers = _analyzer.GroupGeometryNodeIndicesByContainer(pm4File.MSLK);
            if (containers.Count == 0) return;
            // cluster list identical to earlier call
            var containerToCluster = new System.Collections.Generic.Dictionary<ushort, int>();
            var clusterAabbs = new System.Collections.Generic.List<(System.Numerics.Vector3 min, System.Numerics.Vector3 max, System.Collections.Generic.List<ushort> containers)>();
            // Use the same clustering logic to populate clusterAabbs
            var clusters = new System.Collections.Generic.List<System.Collections.Generic.List<ushort>>();
            {
                var containerCentres = new System.Collections.Generic.Dictionary<ushort, System.Numerics.Vector3>();
                foreach (var kvp in containers)
                {
                    var verts = new System.Collections.Generic.List<System.Numerics.Vector3>();
                    foreach (int idx in kvp.Value)
                    {
                        var entry = pm4File.MSLK!.Entries[idx];
                        if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;
                        int end = entry.MspiFirstIndex + entry.MspiIndexCount;
                        if (end > pm4File.MSPI!.Indices.Count) end = pm4File.MSPI.Indices.Count;
                        for (int i = entry.MspiFirstIndex; i < end; i++)
                        {
                            uint vIdx = pm4File.MSPI.Indices[i];
                            var vRaw = pm4File.MSPV!.Vertices[(int)vIdx];
                            verts.Add(_coord.FromMspvVertex(vRaw));
                        }
                    }
                    var centre = System.Numerics.Vector3.Zero;
                    if (verts.Count > 0)
                    {
                        foreach (var v in verts) centre += v;
                        centre /= verts.Count;
                    }
                    containerCentres[kvp.Key] = centre;
                }
                var visited = new System.Collections.Generic.HashSet<ushort>();
                var ordered = containers.Keys.OrderBy(k => k).ToList();
                foreach (var key in ordered)
                {
                    if (visited.Contains(key)) continue;
                    var cluster = new System.Collections.Generic.List<ushort> { key };
                    visited.Add(key);
                    byte flag = (byte)(key >> 8);
                    byte highRef = (byte)(key & 0xFF);
                    var centre = containerCentres[key];
                    foreach (var cand in ordered)
                    {
                        if (visited.Contains(cand)) continue;
                        if ((cand >> 8) != flag) continue;
                        byte candRef = (byte)(cand & 0xFF);
                        if (Math.Abs(candRef - highRef) <= 2 && System.Numerics.Vector3.Distance(centre, containerCentres[cand]) <= 5f)
                        {
                            cluster.Add(cand);
                            visited.Add(cand);
                        }
                    }
                    clusters.Add(cluster);
                }
                int index = 0;
                foreach (var cl in clusters)
                {
                    foreach (var c in cl) containerToCluster[c] = index;
                    // compute AABB for cluster
                    var min = new System.Numerics.Vector3(float.MaxValue);
                    var max = new System.Numerics.Vector3(float.MinValue);
                    foreach (var contKey in cl)
                    {
                        foreach (int node in containers[contKey])
                        {
                            var entry = pm4File.MSLK!.Entries[node];
                            if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;
                            int end = entry.MspiFirstIndex + entry.MspiIndexCount;
                            if (end > pm4File.MSPI!.Indices.Count) end = pm4File.MSPI.Indices.Count;
                            for (int i = entry.MspiFirstIndex; i < end; i++)
                            {
                                uint vIdx = pm4File.MSPI.Indices[i];
                                var v = _coord.FromMspvVertex(pm4File.MSPV!.Vertices[(int)vIdx]);
                                min = System.Numerics.Vector3.Min(min, v);
                                max = System.Numerics.Vector3.Max(max, v);
                            }
                        }
                    }
                    clusterAabbs.Add((min, max, cl));
                    index++;
                }
            }
            // Build overlap graph between clusters
            int n = clusterAabbs.Count;
            var unionFind = new int[n];
            for (int i = 0; i < n; i++) unionFind[i] = i;
            int Find(int x) { return unionFind[x] == x ? x : unionFind[x] = Find(unionFind[x]); }
            void Union(int a, int b) { a = Find(a); b = Find(b); if (a != b) unionFind[b] = a; }
            bool Overlap((System.Numerics.Vector3 min, System.Numerics.Vector3 max, System.Collections.Generic.List<ushort> containers) a,
                         (System.Numerics.Vector3 min, System.Numerics.Vector3 max, System.Collections.Generic.List<ushort> containers) b)
            {
                return (a.min.X <= b.max.X && a.max.X >= b.min.X) &&
                       (a.min.Y <= b.max.Y && a.max.Y >= b.min.Y) &&
                       (a.min.Z <= b.max.Z && a.max.Z >= b.min.Z);
            }
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    if (Overlap(clusterAabbs[i], clusterAabbs[j])) Union(i, j);
                }
            }
            var rootToContainers = new System.Collections.Generic.Dictionary<int, System.Collections.Generic.List<int>>();
            for (int i = 0; i < n; i++)
            {
                int root = Find(i);
                if (!rootToContainers.ContainsKey(root)) rootToContainers[root] = new();
                foreach (ushort key in clusterAabbs[i].containers)
                {
                    rootToContainers[root].Add(key);
                }
            }
            int objIdx = 0;
            foreach (var comp in rootToContainers.Values)
            {
                var nodes = new System.Collections.Generic.List<int>();
                foreach (ushort contKey in comp)
                    nodes.AddRange(containers[contKey]);
                ExportMergedNodes(pm4File, nodes, System.IO.Path.Combine(outputDirectory, $"object_{objIdx:D3}.obj"));
                objIdx++;
            }
            // cleanup temp
            if (System.IO.Directory.Exists(tempDir)) System.IO.Directory.Delete(tempDir, true);
        }

        private void ExportMergedNodes(PM4File pm4File, System.Collections.Generic.List<int> nodeIndices, string objPath)
        {
            if (nodeIndices.Count == 0) return;
            var dir = System.IO.Path.GetDirectoryName(objPath);
            if (!System.IO.Directory.Exists(dir))
                System.IO.Directory.CreateDirectory(dir!);
            using var writer = new System.IO.StreamWriter(objPath);
            writer.WriteLine("# Auto-generated clustered OBJ");
            writer.WriteLine("g cluster");
            var vertices = new System.Collections.Generic.List<System.Numerics.Vector3>();
            var faceIndices = new System.Collections.Generic.List<int>();
            var vertexLookup = new System.Collections.Generic.Dictionary<uint, int>();
            foreach (int nodeIndex in nodeIndices)
            {
                var entry = pm4File.MSLK!.Entries[nodeIndex];
                if (entry.MspiFirstIndex < 0 || entry.MspiIndexCount < 3) continue;
                int end = entry.MspiFirstIndex + entry.MspiIndexCount;
                if (end > pm4File.MSPI!.Indices.Count) end = pm4File.MSPI.Indices.Count;
                for (int i = entry.MspiFirstIndex; i + 2 < end; i += 3)
                {
                    uint aIdx = pm4File.MSPI.Indices[i];
                    uint bIdx = pm4File.MSPI.Indices[i + 1];
                    uint cIdx = pm4File.MSPI.Indices[i + 2];
                    int Add(uint idx)
                    {
                        if (!vertexLookup.TryGetValue(idx, out int local))
                        {
                            var v = _coord.FromMspvVertex(pm4File.MSPV!.Vertices[(int)idx]);
                            local = vertices.Count;
                            vertices.Add(v);
                            vertexLookup[idx] = local;
                        }
                        return vertexLookup[idx];
                    }
                    faceIndices.Add(Add(aIdx));
                    faceIndices.Add(Add(bIdx));
                    faceIndices.Add(Add(cIdx));
                }
            }
            foreach (var v in vertices)
                writer.WriteLine(FormattableString.Invariant($"v {v.X:F6} {v.Y:F6} {v.Z:F6}"));
            var normals = _coord.ComputeVertexNormals(vertices, faceIndices);
            foreach (var n in normals)
                writer.WriteLine(FormattableString.Invariant($"vn {n.X:F6} {n.Y:F6} {n.Z:F6}"));
            for (int f = 0; f < faceIndices.Count; f += 3)
            {
                int a = faceIndices[f] + 1;
                int b = faceIndices[f + 1] + 1;
                int c = faceIndices[f + 2] + 1;
                writer.WriteLine($"f {a}//{a} {b}//{b} {c}//{c}");
            }
            Console.WriteLine($"[MslkExporter] Cluster OBJ written: {System.IO.Path.GetFileName(objPath)} (verts {vertices.Count}, tris {faceIndices.Count / 3})");
        }
    }
}
