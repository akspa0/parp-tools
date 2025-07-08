using System;
using System.Globalization;
using System.IO;
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

            using var writer = new StreamWriter(objPath);
            await writer.WriteLineAsync("# OBJ generated from PM4 " + (sourceFileName ?? "unknown"));
            await writer.WriteLineAsync("o " + Path.GetFileNameWithoutExtension(sourceFileName ?? "pm4_tile"));

            // vertices (Y is up in PM4, keep as-is)
            foreach (var v in pm4.MSPV.Vertices)
            {
                await writer.WriteLineAsync($"v {v.X.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
            }

            await writer.WriteLineAsync();

            // faces (add +1 for OBJ indexing)
            for (int i = 0; i < pm4.MSVI.Indices.Count; i += 3)
            {
                uint a = pm4.MSVI.Indices[i] + 1;
                uint b = pm4.MSVI.Indices[i + 1] + 1;
                uint c = pm4.MSVI.Indices[i + 2] + 1;
                await writer.WriteLineAsync($"f {a} {b} {c}");
            }
        }
    }
}
