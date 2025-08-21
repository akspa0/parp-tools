using System;
using System.Globalization;
using System.IO;
using WoWToolbox.Core.Models;

namespace WoWToolbox.Core.Helpers
{
    public static class PM4MeshExporter
    {
        /// <summary>
        /// Exports a MeshData (PM4 render mesh) to OBJ format using validated logic.
        /// </summary>
        public static void SaveMeshDataToObj(MeshData meshData, string outputPath)
        {
            if (meshData == null)
                throw new ArgumentNullException(nameof(meshData));
            string? directoryPath = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directoryPath) && !Directory.Exists(directoryPath))
                Directory.CreateDirectory(directoryPath);
            using (var writer = new StreamWriter(outputPath, false))
            {
                CultureInfo culture = CultureInfo.InvariantCulture;
                writer.WriteLine("# Mesh saved by WoWToolbox.Core.Helpers.PM4MeshExporter");
                writer.WriteLine("# POINT CLOUD ONLY: Faces are omitted due to unreliable index data");
                writer.WriteLine("# NOTE: X axis is flipped (negated) for alignment with WMO");
                writer.WriteLine($"# Vertices: {meshData.Vertices.Count}");
                writer.WriteLine($"# Generated: {DateTime.Now}");
                writer.WriteLine();
                if (meshData.Vertices.Count > 0)
                {
                    writer.WriteLine("# Vertex Definitions");
                    foreach (var vertex in meshData.Vertices)
                    {
                        writer.WriteLine(string.Format(culture, "v {0} {1} {2}", -vertex.X, vertex.Y, vertex.Z));
                    }
                    writer.WriteLine();
                }
                // Faces are intentionally omitted for point cloud export
            }
        }
    }
} 