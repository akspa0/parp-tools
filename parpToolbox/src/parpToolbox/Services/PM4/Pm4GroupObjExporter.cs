namespace ParpToolbox.Services.PM4;

using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

/// <summary>
/// Exports each <see cref="SurfaceGroup"/> in a <see cref="Pm4Scene"/> to a separate OBJ file.
/// The file names will be &lt;baseName&gt;_&lt;group.Name&gt;.obj inside the specified output directory.
/// </summary>
internal static class Pm4GroupObjExporter
{
    /// <param name="scene">Scene to export – must contain populated <see cref="Pm4Scene.Groups"/>.</param>
    /// <param name="outputDir">Directory to place OBJ/MTL files in (will be created).</param>
    /// <param name="writeFaces">Write face definitions instead of point cloud when true.</param>
    /// <param name="flipX">Invert X coordinate to correct mirroring.</param>
    public static void Export(Pm4Scene scene, string outputDir, bool writeFaces = false, bool flipX = true)
    {
        // Use NewPm4Exporter with equivalent options for MSUR grouping
        var options = new NewPm4Exporter.ExportOptions
        {
            Format = NewPm4Exporter.ExportFormat.Obj,
            MinTriangles = 0, // No filtering
            ApplyXAxisInversion = flipX,
            IncludeM2Objects = true, // Include all objects
            EnableMprlTransformations = false, // No MPRL transforms for legacy compatibility
            EnableCrossTileResolution = false, // No cross-tile resolution for legacy compatibility
        };

        if (!writeFaces)
        {
            ConsoleLogger.WriteLine("Warning: Point-cloud export mode is deprecated and has been removed. Exporting faces instead.");
        }

        var exporter = new NewPm4Exporter(scene, options);
        exporter.Export(outputDir);
    }
}
