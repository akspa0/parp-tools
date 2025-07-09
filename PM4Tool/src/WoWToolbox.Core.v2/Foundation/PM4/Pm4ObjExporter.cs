using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Foundation.PM4
{
    /// <summary>
    /// Very lightweight OBJ exporter for PM4 geometry (vertices + triangles).
    /// Produces one OBJ file per PM4 tile. Intended for batch output parity with the
    /// original PM4BatchOutput prototype. Materials, UVs, etc. are ignored for now.
    /// </summary>
    public static class Pm4ObjExporter
    {
        /// <summary>
        /// Exports the geometry of a <see cref="PM4File"/> to a Wavefront OBJ file.
        /// </summary>
        /// <param name="pm4">Parsed PM4 file.</param>
        /// <param name="objPath">Destination path (will be overwritten).</param>
        /// <summary>
        /// Returns all render vertices in world (OBJ) coordinates, preferring MSVT (render) data and falling back to MSPV.
        /// </summary>
        private static List<Vector3> GetWorldVertices(PM4File pm4)
        {
            if (pm4.MSVT != null && pm4.MSVT.Vertices.Count > 0)
            {
                return pm4.MSVT.Vertices.Select(v => new Vector3(v.Y, v.Z, -v.X)).ToList();
            }
            else if (pm4.MSPV != null && pm4.MSPV.Vertices.Count > 0)
            {
                return pm4.MSPV.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
            }
            return new List<Vector3>();
        }

        public static async Task ExportAsync(PM4File pm4, string objPath, string? sourceFileName = null)
        {
            if (pm4 == null) throw new ArgumentNullException(nameof(pm4));
            if ((pm4.MSPV == null || pm4.MSPV.Vertices.Count == 0) &&
                (pm4.MSVT == null || pm4.MSVT.Vertices.Count == 0))
                throw new InvalidOperationException("PM4 file contains no vertex data (MSPV/MSVT both absent or empty).");
            // Face data (MSVI) is optional; if absent we'll export vertices only.

            Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);

            var sb = new StringBuilder();
            var verts = GetWorldVertices(pm4);
            // header comments
            sb.AppendLine($"# OBJ generated from PM4 file: {sourceFileName ?? "unknown"}");
            sb.AppendLine($"# Generated on {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            var coords = TryExtractTileCoords(sourceFileName);
            if (coords.HasValue)
                sb.AppendLine($"# Terrain coordinates: {coords.Value.x}, {coords.Value.y}");
            sb.AppendLine();

            // reference to a default material library (user can generate later)
            sb.AppendLine("mtllib pm4_materials.mtl");
            sb.AppendLine();

            string objName = Path.GetFileNameWithoutExtension(sourceFileName ?? "pm4_tile");
            sb.AppendLine($"o {objName}");
            sb.AppendLine($"g {objName}_mesh");
            sb.AppendLine("usemtl default");

            // vertices
            float minX=float.MaxValue,minY=float.MaxValue,minZ=float.MaxValue;
            float maxX=float.MinValue,maxY=float.MinValue,maxZ=float.MinValue;
            foreach (var vVec in verts)
            {
                sb.AppendLine($"v {vVec.X.ToString(CultureInfo.InvariantCulture)} {vVec.Y.ToString(CultureInfo.InvariantCulture)} {vVec.Z.ToString(CultureInfo.InvariantCulture)}");
                if (vVec.X<minX) minX=vVec.X; if(vVec.Y<minY) minY=vVec.Y; if(vVec.Z<minZ) minZ=vVec.Z;
                if (vVec.X>maxX) maxX=vVec.X; if(vVec.Y>maxY) maxY=vVec.Y; if(vVec.Z>maxZ) maxZ=vVec.Z;
            }
            sb.AppendLine();
            sb.AppendLine($"# Bounding Box: Min({minX:F2},{minY:F2},{minZ:F2}) Max({maxX:F2},{maxY:F2},{maxZ:F2})");
            sb.AppendLine($"# Dimensions: W({maxX-minX:F2}) H({maxY-minY:F2}) D({maxZ-minZ:F2})");
            sb.AppendLine();

            // faces (OBJ indices are 1-based) – only if MSVI present
            if (pm4.MSVI != null && pm4.MSVI.Indices.Count >= 3)
            {
                for (int i = 0; i < pm4.MSVI.Indices.Count; i += 3)
                {
                    uint a = pm4.MSVI.Indices[i] + 1;
                    uint b = pm4.MSVI.Indices[i + 1] + 1;
                    uint c = pm4.MSVI.Indices[i + 2] + 1;
                    sb.AppendLine($"f {a} {b} {c}");
                }
            }

            // optionally add position / command points
if (pm4.MPRL != null && pm4.MPRL.Entries.Count > 0)
{
    int startingIndex = verts.Count + 1; // OBJ 1-based
    var posEntries = pm4.MPRL.Entries.Where(e => e.Unknown_0x02 != -1).ToList();
    var cmdEntries = pm4.MPRL.Entries.Where(e => e.Unknown_0x02 == -1).ToList();

    if (posEntries.Count > 0)
    {
        sb.AppendLine();
        sb.AppendLine($"o {objName}_PositionData");
        sb.AppendLine("usemtl positionData");
        foreach (var p in posEntries)
        {
            sb.AppendLine($"v {p.Position.X.ToString(CultureInfo.InvariantCulture)} {p.Position.Y.ToString(CultureInfo.InvariantCulture)} {p.Position.Z.ToString(CultureInfo.InvariantCulture)}");
        }
        sb.Append("p");
        for (int i = 0; i < posEntries.Count; i++) sb.Append($" {startingIndex + i}");
        sb.AppendLine();
        startingIndex += posEntries.Count;
    }
    if (cmdEntries.Count > 0)
    {
        sb.AppendLine();
        sb.AppendLine($"o {objName}_CommandData");
        sb.AppendLine("usemtl commandData");
        foreach (var c in cmdEntries)
        {
            sb.AppendLine($"v {c.Position.Z.ToString(CultureInfo.InvariantCulture)} {c.Position.X.ToString(CultureInfo.InvariantCulture)} {c.Position.Y.ToString(CultureInfo.InvariantCulture)}");
        }
        sb.Append("p");
        for (int i = 0; i < cmdEntries.Count; i++) sb.Append($" {startingIndex + i}");
        sb.AppendLine();
        startingIndex += cmdEntries.Count;
    }
}

await File.WriteAllTextAsync(objPath, sb.ToString());
        }

        public static async Task ExportConsolidatedAsync(IEnumerable<(PM4File file,string name)> tiles, string objPath)
        {
            if (tiles==null) throw new ArgumentNullException(nameof(tiles));
            Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);
            var sb=new StringBuilder();
            sb.AppendLine($"# Consolidated OBJ generated from {tiles.Count()} PM4 tiles");
            sb.AppendLine($"# Generated on {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine();
            int vertexOffset=0;
            foreach(var (pm4,name) in tiles)
            {
                string objName=Path.GetFileNameWithoutExtension(name);
                sb.AppendLine($"o {objName}");
                sb.AppendLine($"g {objName}_mesh");
                sb.AppendLine("usemtl default");
                try
                {
                    var worldVerts = GetWorldVertices(pm4);
                    int addedVerts = worldVerts.Count;
                    foreach (var v in worldVerts)
                    {
                        
                        {
                            sb.AppendLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                        }
                        
                    }

                    if (pm4.MSVI != null && pm4.MSVI.Indices.Count >= 3)
                    {
                        for (int i = 0; i < pm4.MSVI.Indices.Count; i += 3)
                        {
                            uint a = pm4.MSVI.Indices[i] + 1 + (uint)vertexOffset;
                            uint b = pm4.MSVI.Indices[i + 1] + 1 + (uint)vertexOffset;
                            uint c = pm4.MSVI.Indices[i + 2] + 1 + (uint)vertexOffset;
                            sb.AppendLine($"f {a} {b} {c}");
                        }
                    }

                    vertexOffset += addedVerts;
                }
                catch (Exception ex) when (ex is ArgumentNullException || ex is InvalidOperationException)
                {
                    if (ex.Message.Contains("no vertex data", StringComparison.OrdinalIgnoreCase) ||
                         ex.Message.Contains("no MSPV", StringComparison.OrdinalIgnoreCase) ||
                         ex.Message.Contains("MSPV/MSVT", StringComparison.OrdinalIgnoreCase))
                    {
                        sb.AppendLine($"# Skipping {name}: {ex.Message}");
                    }
                    else
                    {
                        throw;
                    }
                    // skipped tile due to missing data
                }
                
                sb.AppendLine();
            }
            await File.WriteAllTextAsync(objPath,sb.ToString());
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
