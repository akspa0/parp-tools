using System;
using System.IO;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Services.PM4;
using WoWToolbox.Core.v2.Foundation.PM4;

// Simple CLI front-end for Pm4BatchProcessor. Usage:
//   dotnet Pm4BatchTool.dll <pm4-file-or-directory> [--wmo <wmoDataDir>]
// Outputs OBJ + summary to ProjectOutput/pm4.

class Program
{
    static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: Pm4BatchTool <pm4-file-or-directory> [--wmo <wmoDataDir>] [--diag] [--force]");
            return 1;
        }

        string inputPath = args[0];
        string? wmoDir = null;
        bool dumpDiag = args.Contains("--diag");
        bool force = args.Contains("--force");
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--wmo" when i + 1 < args.Length:
                    wmoDir = args[i + 1];
                    i++;
                    break;
                case "--diag":
                    dumpDiag = true;
                    break;
            }
        }

        if (!File.Exists(inputPath) && !Directory.Exists(inputPath))
        {
            Console.Error.WriteLine($"Input path '{inputPath}' does not exist.");
            return 1;
        }

        // Direct instantiation to avoid DI complexity
        var coordinateService = new WoWToolbox.Core.v2.Services.PM4.CoordinateService();
        IBuildingExtractionService extraction = new WoWToolbox.Core.v2.Services.PM4.PM4BuildingExtractionService(coordinateService);
        WoWToolbox.Core.v2.Services.PM4.IWmoMatcher matcher = new WoWToolbox.Core.v2.Services.PM4.WmoMatcher(wmoDir ?? Path.Combine(Environment.CurrentDirectory, "wmo_data"));
        WoWToolbox.Core.v2.Services.PM4.IPm4BatchProcessor processor = new WoWToolbox.Core.v2.Services.PM4.Pm4BatchProcessor(extraction, matcher);

        if (File.Exists(inputPath))
        {
            if (force)
            {
                string stem = Path.GetFileNameWithoutExtension(inputPath);
                string dir = Path.Combine(processor.RunDirectory, stem);
                if (Directory.Exists(dir))
                {
                    try { Directory.Delete(dir, true); } catch { }
                }
            }

            if (dumpDiag)
            {
                DumpDiagnostics(inputPath);
            }
            var res = processor.Process(inputPath);
            Console.WriteLine(res.Success ? "Success" : $"Failed: {res.ErrorMessage}");
        }
        else
        {
            var pm4Files = Directory.EnumerateFiles(inputPath, "*.pm4", SearchOption.AllDirectories);
            int ok = 0;
            int fail = 0;
            foreach (var file in pm4Files)
            {
                if (force)
                {
                    // remove previous output folder if it exists
                    string stem = Path.GetFileNameWithoutExtension(file);
                    string dir = Path.Combine(processor.RunDirectory, stem);
                    if (Directory.Exists(dir))
                    {
                        try { Directory.Delete(dir, true); } catch { /* ignore */ }
                    }
                }

                if (dumpDiag)
                {
                    DumpDiagnostics(file);
                }

                var r = processor.Process(file);
                if (r.Success)
                {
                    ok++;
                }
                else
                {
                    fail++;
                    Console.WriteLine($"Failed {file}: {r.ErrorMessage}");
                }
            }
            Console.WriteLine($"Processed {ok + fail} files: {ok} succeeded, {fail} failed.");
        }

        return 0;

        // Local function to dump diagnostic CSV for a single PM4
        void DumpDiagnostics(string pm4Path)
        {
            try
            {
                var pm4 = PM4File.FromFile(pm4Path);
                string tsDir = Path.Combine(Environment.CurrentDirectory, "project_output", "diagnostics");
                Directory.CreateDirectory(tsDir);
                string csvName = Path.GetFileNameWithoutExtension(pm4Path) + ".csv";
                MsLkDiagnostics.DumpEntryCsv(pm4, Path.Combine(tsDir, csvName));
                Console.WriteLine($"  ↳ diagnostics → {csvName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ↳ diagnostics failed: {ex.Message}");
            }
        }
    }
}
