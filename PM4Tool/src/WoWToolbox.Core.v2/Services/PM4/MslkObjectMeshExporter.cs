using System;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;

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

        /// <summary>
        /// Very first slice of the exporter: write one OBJ per GroupId that contains geometry.
        /// Only vertices are dumped; no faces yet.
        /// </summary>
        /// <summary>
        /// Writes one OBJ per GroupId (Unknown_0x04) – kept for debugging/compat.
        /// </summary>
        public void ExportAllGroupsAsObj(PM4File pm4File, string outputDirectory)
        
        {
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
        public void ExportAllObjectsAsObj(PM4File pm4File, string outputDirectory)
        
        {
            if (!System.IO.Directory.Exists(outputDirectory))
                System.IO.Directory.CreateDirectory(outputDirectory);

            var grouping = _analyzer.GroupGeometryNodeIndicesByObjectId(pm4File.MSLK);
            if (grouping.Count == 0)
            {
                Console.WriteLine("[MslkExporter] No object groupings found.");
                return;
            }

            foreach (var kvp in grouping)
            {
                ushort objectId = kvp.Key;
                var nodeIndices = kvp.Value;

                string objPath = System.IO.Path.Combine(outputDirectory, $"object_{objectId:X4}.obj");
                using var writer = new System.IO.StreamWriter(objPath);
                writer.WriteLine($"# Auto-generated OBJ for ObjectId 0x{objectId:X4}");
                writer.WriteLine("g object_" + objectId.ToString("X4"));

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
            => ExportGeneric(pm4File, outputDirectory, grouping.ToDictionary(k=> (ushort)k.Key, v=> v.Value), key => filenameFunc((byte)key), label);

        private void ExportGeneric(PM4File pm4File, string outputDirectory, System.Collections.Generic.Dictionary<ushort, System.Collections.Generic.List<int>> grouping, System.Func<ushort,string> filenameFunc, string label)
        {
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
