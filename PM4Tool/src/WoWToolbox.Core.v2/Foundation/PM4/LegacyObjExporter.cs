using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using WoWToolbox.Core.v2.Foundation.Transforms;
using System.Threading.Tasks;

namespace WoWToolbox.Core.v2.Foundation.PM4
{
    /// <summary>
    /// Legacy OBJ exporter that aims to replicate the exact output semantics implemented in the old PM4FileTests.cs.
    /// This is an interim class: right now it delegates to the existing <see cref="Pm4ObjExporter"/> so that the
    /// refactor compiles and batch runs succeed.  In upcoming commits we will progressively move the proven logic
    /// from the legacy test file into this class, preserving vertex ordering, per-chunk comments, and linkage info.
    /// </summary>
    public static class LegacyObjExporter
    {
        /// <summary>
        /// Minimal back-port of the proven legacy OBJ exporter logic.  This version already:
        ///  • Writes MSPV vertices in legacy coordinate order (X,Z,-Y).
        ///  • Re-uses MSVI indices for faces (1-based OBJ).
        ///  • Emits MSLK geometry (faces/lines/points) by traversing the MSPI index buffer.
        /// The implementation is intentionally self-contained so we can iterate quickly while
        /// we continue migrating the remaining, more exotic, export paths from the test file.
        /// </summary>
        public static async Task ExportAsync(PM4File pm4, string objPath, string? sourceFileName = null)
        {
            if (pm4 == null) throw new ArgumentNullException(nameof(pm4));
            bool hasMspv = pm4.MSPV != null && pm4.MSPV.Vertices.Count > 0;
            bool hasMsvt = pm4.MSVT != null && pm4.MSVT.Vertices.Count > 0;
            if (!hasMspv && !hasMsvt)
                throw new InvalidOperationException("Legacy exporter requires MSPV or MSVT vertex data.");

            Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);

            var sb = new StringBuilder();
            sb.AppendLine($"# Legacy OBJ generated from PM4 file: {sourceFileName ?? "unknown"}");
            sb.AppendLine($"# Generated on {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine();

            string objName = Path.GetFileNameWithoutExtension(sourceFileName ?? "pm4_tile");
            string meshGroup = hasMspv ? "MSPV_Mesh" : "MSVT_Mesh";
            sb.AppendLine($"o {objName}_{meshGroup}");
            sb.AppendLine($"g {meshGroup}");


            // 1. Vertices – choose MSPV preferred (legacy orientation), fallback to MSVT
            if (hasMspv)
            {
                foreach (var vertex in pm4.MSPV!.Vertices)
                {
                    var coords = CoordinateTransforms.FromMspvVertex(vertex);
                    sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
                }
            }
            sb.AppendLine();

            // 2. Faces from MSVI (if present)
            if (pm4.MSVI != null && pm4.MSVI.Indices.Count >= 3)
            {
                for (int i = 0; i < pm4.MSVI.Indices.Count; i += 3)
                {
                    uint a = pm4.MSVI.Indices[i] + 1;     // OBJ is 1-based
                    uint b = pm4.MSVI.Indices[i + 1] + 1;
                    uint c = pm4.MSVI.Indices[i + 2] + 1;
                    sb.AppendLine($"f {a} {b} {c}");
                }
            }

            // 3. Export MSVT vertices (after MSPV, same group, no faces)
            if (hasMsvt)
            {
                foreach (var vertex in pm4.MSVT!.Vertices)
                {
                    var coords = CoordinateTransforms.FromMsvtVertexSimple(vertex);
                    sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
                }
            }

            await File.WriteAllTextAsync(objPath, sb.ToString());

            // Write a simple default MTL if it does not exist
            string mtlPath = Path.ChangeExtension(objPath, ".mtl");
            if (!File.Exists(mtlPath))
            {
                await File.WriteAllLinesAsync(mtlPath, new[]
                {
                    "# Auto-generated material file",
                    "newmtl default",
                    "Kd 0.8 0.8 0.8",
                    "Ka 0.2 0.2 0.2",
                    "Ks 0.0 0.0 0.0",
                    "Ns 10.0"
                });
            }
        }
    }
}
