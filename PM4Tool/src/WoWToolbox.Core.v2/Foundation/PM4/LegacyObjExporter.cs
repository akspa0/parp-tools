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

            // Optional terrain-tile comment (filename pattern *_x_y.pm4)
            var tileCoords = TryExtractTileCoords(sourceFileName);
            if (tileCoords.HasValue)
                sb.AppendLine($"# Terrain coordinates: {tileCoords.Value.x}, {tileCoords.Value.y}");

            sb.AppendLine();

            string objName = Path.GetFileNameWithoutExtension(sourceFileName ?? "pm4_tile");
            string meshGroup = hasMspv ? "MSPV_Mesh" : "MSVT_Mesh";
            string mtlFileName = Path.GetFileName(Path.ChangeExtension(objPath, ".mtl"));
            // reference default material library
            sb.AppendLine($"mtllib {mtlFileName}");
            sb.AppendLine($"o {objName}_{meshGroup}");
            sb.AppendLine($"g {meshGroup}");
            sb.AppendLine("usemtl default");

            // bounding box trackers
            float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;

            // 1. Vertices – choose MSPV preferred (legacy orientation), fallback to MSVT
            if (hasMspv)
            {
                foreach (var vertex in pm4.MSPV!.Vertices)
                {
                    var coords = new Vector3(vertex.X, vertex.Y, vertex.Z);
                    sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
                    if (coords.X < minX) minX = coords.X; if (coords.Y < minY) minY = coords.Y; if (coords.Z < minZ) minZ = coords.Z;
                    if (coords.X > maxX) maxX = coords.X; if (coords.Y > maxY) maxY = coords.Y; if (coords.Z > maxZ) maxZ = coords.Z;
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
                    // Spec orientation for MSVT: (Y, X, Z)
                                            var coords = new Vector3(vertex.Y, vertex.X, vertex.Z);
                    sb.AppendLine($"v {coords.X.ToString(CultureInfo.InvariantCulture)} {coords.Y.ToString(CultureInfo.InvariantCulture)} {coords.Z.ToString(CultureInfo.InvariantCulture)}");
                    if (coords.X < minX) minX = coords.X; if (coords.Y < minY) minY = coords.Y; if (coords.Z < minZ) minZ = coords.Z;
                    if (coords.X > maxX) maxX = coords.X; if (coords.Y > maxY) maxY = coords.Y; if (coords.Z > maxZ) maxZ = coords.Z;
                }
            }

            // Bounding-box summary lines
            sb.AppendLine();
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"# Bounding Box: Min({minX:F2},{minY:F2},{minZ:F2}) Max({maxX:F2},{maxY:F2},{maxZ:F2})"));
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"# Dimensions: W({(maxX - minX):F2}) H({(maxY - minY):F2}) D({(maxZ - minZ):F2})"));
            sb.AppendLine();

            // Optional MPRL position / command point export (separate objects)
            if (pm4.MPRL != null && pm4.MPRL.Entries.Count > 0)
            {
                int startingIndex = (hasMspv ? pm4.MSPV!.Vertices.Count : 0) + (hasMsvt ? pm4.MSVT!.Vertices.Count : 0) + 1; // OBJ 1-based
                var posEntries = pm4.MPRL.Entries.Where(e => e.Unknown_0x02 != -1).ToList();
                var cmdEntries = pm4.MPRL.Entries.Where(e => e.Unknown_0x02 == -1).ToList();

                if (posEntries.Count > 0)
                {
                    sb.AppendLine($"o {objName}_PositionData");
                    sb.AppendLine("usemtl positionData");
                    foreach (var p in posEntries)
                    {
                        sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"v {p.Position.X} {p.Position.Y} {p.Position.Z}"));
                    }
                    sb.Append("p");
                    for (int i = 0; i < posEntries.Count; i++) sb.Append($" {startingIndex + i}");
                    sb.AppendLine();
                    startingIndex += posEntries.Count;
                }

                if (cmdEntries.Count > 0)
                {
                    sb.AppendLine($"o {objName}_CommandData");
                    sb.AppendLine("usemtl commandData");
                    foreach (var c in cmdEntries)
                    {
                        sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"v {c.Position.Z} {c.Position.X} {c.Position.Y}"));
                    }
                    sb.Append("p");
                    for (int i = 0; i < cmdEntries.Count; i++) sb.Append($" {startingIndex + i}");
                    sb.AppendLine();
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
        private static (int x,int y)? TryExtractTileCoords(string? fileName)
        {
            if (string.IsNullOrEmpty(fileName)) return null;
            var parts = Path.GetFileNameWithoutExtension(fileName).Split('_');
            if (parts.Length>=2 && int.TryParse(parts[^2],out int x) && int.TryParse(parts[^1],out int y))
                return (x,y);
            return null;
        }
    }
}
