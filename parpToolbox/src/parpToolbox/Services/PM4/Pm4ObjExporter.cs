namespace ParpToolbox.Services.PM4;

using System.Globalization;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

/// <summary>
/// Simple OBJ exporter for <see cref="Pm4Scene"/>. Writes single OBJ with default material.
/// </summary>
/// <summary>
/// Legacy wrapper delegating to unified <see cref="Pm4Exporter"/>.
/// Retained for backwards compatibility with existing call sites.
/// </summary>
internal static class Pm4ObjExporter
{
    /// <summary>
    /// Exports the given scene to Wavefront OBJ.
    /// </summary>
    /// <param name="scene">Scene to export.</param>
    /// <param name="filename">Target OBJ (or any path without extension).</param>
    /// <param name="writeFaces">If true, writes face definitions; otherwise points only.</param>
    public static void Export(Pm4Scene scene, string filename, bool writeFaces = false, bool flipX = true)
    {
        string dir = Path.GetDirectoryName(filename) ?? ".";
        Directory.CreateDirectory(dir);

        // Note: unified exporter always writes faces. If caller requested point-cloud we warn once.
        if (!writeFaces)
        {
            ConsoleLogger.WriteLine("Warning: Point-cloud export mode is deprecated and has been removed. Exporting faces instead.");
        }

        var exporter = new Pm4Exporter(
            scene,
            dir,
            new Pm4Exporter.ExportOptions
            {
                Grouping = Pm4Exporter.GroupingStrategy.None,
                SeparateFiles = false,
                FlipX = flipX,
                Verbose = true,
            });

        exporter.Export();

        // The unified exporter writes combined.obj / combined.mtl by default when SeparateFiles=false.
        // Rename to legacy naming convention expected by callers.
        string generatedObj = Path.Combine(dir, "combined.obj");
        string generatedMtl = Path.Combine(dir, "combined.mtl");
        string targetObj = Path.ChangeExtension(filename, ".obj");
        string targetMtl = Path.ChangeExtension(filename, ".mtl");

        try
        {
            if (File.Exists(generatedObj))
            {
                if (File.Exists(targetObj)) File.Delete(targetObj);
                File.Move(generatedObj, targetObj);
            }
            if (File.Exists(generatedMtl))
            {
                if (File.Exists(targetMtl)) File.Delete(targetMtl);
                File.Move(generatedMtl, targetMtl);
            }
        }
        catch (IOException ex)
        {
            ConsoleLogger.WriteLine($"Warning: Failed to rename combined export files: {ex.Message}");
        }
    }
}
