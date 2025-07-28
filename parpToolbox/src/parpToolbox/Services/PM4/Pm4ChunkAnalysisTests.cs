using System;
using System.IO;
using System.Linq;
using System.Text;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

public static class Pm4ChunkAnalysisTests
{
    public static void DumpChunkDataForAnalysis()
    {
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        var pm4Path = Path.Combine(projectRoot, "test_data", "original_development", "development_00_00.pm4");

        if (!File.Exists(pm4Path))
        {
            ConsoleLogger.WriteLine($"[Analysis] Test asset not found: {pm4Path}");
            return;
        }

        var outputDir = Path.Combine(projectRoot, "project_output", $"analysis_{DateTime.Now:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(outputDir);

        ConsoleLogger.WriteLine($"[Analysis] Loading scene: {pm4Path}");
        var adapter = new Pm4Adapter();
        var scene = adapter.Load(pm4Path, new Pm4LoadOptions { VerboseLogging = false });

        // Dump MSUR data
        var msurCsvPath = Path.Combine(outputDir, "msur_analysis.csv");
        var msurSb = new StringBuilder();
        msurSb.AppendLine("MsurIndex,Index_0x01,MsviFirstIndex,IndexCount,SurfaceKey"); // Unknown0x10 removed due to build error
        for (var i = 0; i < scene.Surfaces.Count; i++)
        {
            var s = scene.Surfaces[i];
            msurSb.AppendLine($"{i},{s.IndexCount},{s.MsviFirstIndex},{s.IndexCount},{s.SurfaceKey}");
        }
        File.WriteAllText(msurCsvPath, msurSb.ToString());
        ConsoleLogger.WriteLine($"[Analysis] Wrote MSUR data to {msurCsvPath}");

        // Dump MSLK data
        var mslkCsvPath = Path.Combine(outputDir, "mslk_analysis.csv");
        var mslkSb = new StringBuilder();
        mslkSb.AppendLine("MslkIndex,Unknown_0x00,ReferenceIndex,ParentIndex,MspiFirstIndex,MspiIndexCount,HasGeometry");
        for (var i = 0; i < scene.Links.Count; i++)
        {
            var l = scene.Links[i];
            mslkSb.AppendLine($"{i},{l.Flags_0x00},{l.ReferenceIndex},{l.ParentIndex},{l.MspiFirstIndex},{l.MspiIndexCount},{l.HasGeometry}");
        }
        File.WriteAllText(mslkCsvPath, mslkSb.ToString());
        ConsoleLogger.WriteLine($"[Analysis] Wrote MSLK data to {mslkCsvPath}");
    }
}
