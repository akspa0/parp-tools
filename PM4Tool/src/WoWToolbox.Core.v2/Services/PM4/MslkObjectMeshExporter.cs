using System;
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

                // Gather vertices first
                foreach (int nodeIndex in nodeIndices)
                {
                    var entry = pm4File.MSLK!.Entries[nodeIndex];
                    if (pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null) continue;
                    var localIndices = new System.Collections.Generic.List<int>();
                    for (int i = 0; i < entry.MspiIndexCount; i++)
                    {
                        int mspiIdx = entry.MspiFirstIndex + i;
                        if (mspiIdx < 0 || mspiIdx >= pm4File.MSPI.Indices.Count) continue;
                        uint mspvIdx = pm4File.MSPI.Indices[mspiIdx];
                        if (mspvIdx >= pm4File.MSPV.Vertices.Count) continue;
                        var vRaw = pm4File.MSPV.Vertices[(int)mspvIdx];
                        var v = _coord.FromMspvVertex(vRaw);
                        vertices.Add(v);
                        localIndices.Add(vertices.Count); // 1-based index relative to group
                    }
                    // simple face fan if polygon has 3+ vertices
                    if (localIndices.Count >= 3)
                    {
                        for (int k = 1; k < localIndices.Count - 1; k++)
                        {
                            faceIndices.Add(localIndices[0]-1);
                            faceIndices.Add(localIndices[k]-1);
                            faceIndices.Add(localIndices[k+1]-1);
                        }
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
    }
}
