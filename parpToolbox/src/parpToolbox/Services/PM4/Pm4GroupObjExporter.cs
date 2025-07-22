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
        // Delegates to unified Pm4Exporter with MSUR grouping.
        var exporter = new Pm4Exporter(
            scene,
            outputDir,
            new Pm4Exporter.ExportOptions
            {
                Grouping = Pm4Exporter.GroupingStrategy.MsurSurfaceGroup,
                SeparateFiles = true,
                FlipX = flipX,
                // The unified exporter always writes faces; legacy behaviour of point-cloud export when writeFaces == false
                // is no longer supported. We emit a warning once to inform the user.
                Verbose = true,
            });

        if (!writeFaces)
        {
            ConsoleLogger.WriteLine("Warning: Point-cloud export mode is deprecated and has been removed. Exporting faces instead.");
        }

        exporter.Export();
    }
}
