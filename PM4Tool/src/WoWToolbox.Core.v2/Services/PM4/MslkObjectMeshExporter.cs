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
    }
}
