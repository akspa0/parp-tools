using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
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
        public static async Task ExportAsync(PM4File pm4, string objPath, string? sourceFileName = null)
        {
            if (pm4 == null) throw new ArgumentNullException(nameof(pm4));
            if (pm4.MSPV == null || pm4.MSPV.Vertices.Count == 0)
                throw new InvalidOperationException("PM4 file contains no vertex data.");
            if (pm4.MSVI == null || pm4.MSVI.Indices.Count == 0)
                throw new InvalidOperationException("PM4 file contains no index data.");

            Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);

            var sb = new StringBuilder();
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
            foreach (var v in pm4.MSPV.Vertices)
            {
                sb.AppendLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                if (v.X<minX) minX=v.X; if(v.Y<minY) minY=v.Y; if(v.Z<minZ) minZ=v.Z;
                if (v.X>maxX) maxX=v.X; if(v.Y>maxY) maxY=v.Y; if(v.Z>maxZ) maxZ=v.Z;
            }
            sb.AppendLine();
            sb.AppendLine($"# Bounding Box: Min({minX:F2},{minY:F2},{minZ:F2}) Max({maxX:F2},{maxY:F2},{maxZ:F2})");
            sb.AppendLine($"# Dimensions: W({maxX-minX:F2}) H({maxY-minY:F2}) D({maxZ-minZ:F2})");
            sb.AppendLine();

            // faces (OBJ indices are 1-based)
            for (int i = 0; i < pm4.MSVI.Indices.Count; i += 3)
            {
                uint a = pm4.MSVI.Indices[i] + 1;
                uint b = pm4.MSVI.Indices[i + 1] + 1;
                uint c = pm4.MSVI.Indices[i + 2] + 1;
                sb.AppendLine($"f {a} {b} {c}");
            }

            // optionally add position / command points
if (pm4.MPRL != null && pm4.MPRL.Entries.Count > 0)
{
    int startingIndex = pm4.MSPV.Vertices.Count + 1; // OBJ 1-based
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
            sb.AppendLine($"v {c.Position.X.ToString(CultureInfo.InvariantCulture)} {c.Position.Y.ToString(CultureInfo.InvariantCulture)} {c.Position.Z.ToString(CultureInfo.InvariantCulture)}");
        }
        sb.Append("p");
        for (int i = 0; i < cmdEntries.Count; i++) sb.Append($" {startingIndex + i}");
        sb.AppendLine();
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
                // vertices
                foreach(var v in pm4.MSPV!.Vertices)
                {
                    sb.AppendLine($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                }
                // faces
                for(int i=0;i<pm4.MSVI!.Indices.Count;i+=3)
                {
                    uint a=pm4.MSVI.Indices[i]+1+(uint)vertexOffset;
                    uint b=pm4.MSVI.Indices[i+1]+1+(uint)vertexOffset;
                    uint c=pm4.MSVI.Indices[i+2]+1+(uint)vertexOffset;
                    sb.AppendLine($"f {a} {b} {c}");
                }
                vertexOffset+=pm4.MSPV.Vertices.Count;
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
