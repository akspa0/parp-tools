using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Services.PM4;
using WoWToolbox.Core.v2.Foundation.PM4;

// Simple CLI front-end for Pm4BatchProcessor. Usage:
//   dotnet Pm4BatchTool.dll <pm4-file-or-directory> [--wmo <wmoDataDir>]
// Outputs OBJ + summary to ProjectOutput/pm4.

class Program
{
    static async Task<int> Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: Pm4BatchTool <pm4-file-or-directory> [--wmo <wmoDataDir>] [--diag] [--force] [--debug-chunks]");
            return 1;
        }

        string inputPath = args[0];
        string? wmoDir = null;
        bool dumpDiag = args.Contains("--diag");
        bool force = args.Contains("--force");
        bool debugChunks = args.Contains("--debug-chunks");
        bool chunkQuilts = args.Contains("--chunk-quilts");
        for (int i = 1; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--wmo" when i + 1 < args.Length:
                    wmoDir = args[i + 1];
                    i++;
                    break;
                case "--chunk-quilts":
                    chunkQuilts = true;
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

            // Export debug chunks if requested
            if (debugChunks)
            {
                try
                {
                    // Reload PM4 (processor already exported OBJ inside Process)
                    var pm4ForDebug = PM4File.FromFile(inputPath);
                    string outputDir = Path.Combine(processor.RunDirectory, Path.GetFileNameWithoutExtension(inputPath));
                    string basePath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputPath));
                    await ChunkDebugExporter.ExportChunksAsync(pm4ForDebug, basePath);
                    Console.WriteLine($"  Exported debug chunks to: {basePath}_*.obj");
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error exporting debug chunks: {ex.Message}");
                }
            }
        }
        else
        {
            var pm4Files = Directory.EnumerateFiles(inputPath, "*.pm4", SearchOption.AllDirectories);
            int ok = 0;
            int fail = 0;
            var processed = new List<(PM4File file,string name)>();
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
                    try { processed.Add((PM4File.FromFile(file), Path.GetFileNameWithoutExtension(file))); } catch {}                }
                else
                {
                    fail++;
                    Console.WriteLine($"Failed {file}: {r.ErrorMessage}");
                }
            }
            Console.WriteLine($"Processed {ok + fail} files: {ok} succeeded, {fail} failed.");

            if (chunkQuilts)
            {
                try
                {
                    string outDir = Path.Combine(Environment.CurrentDirectory, "project_output", "analysis");
                    Directory.CreateDirectory(outDir);
                    WoWToolbox.Core.v2.Services.PM4.TileQuiltChunkExporter.ExportChunkQuilts(inputPath, outDir);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  Error exporting chunk quilts: {ex.Message}");
                }
            }

            // MSLK cross-tile analysis
            if (processed.Count>0)
            {
                var graph = WoWToolbox.Core.v2.Services.PM4.MslkLinkGraphBuilder.Build(processed);
                string outDir = Path.Combine(Environment.CurrentDirectory,"project_output","analysis");
                Directory.CreateDirectory(outDir);
                string csvPath = Path.Combine(outDir,"mslk_links.csv");
                using var sw = new StreamWriter(csvPath);
                sw.WriteLine("LinkIdHex,TileCount,EntryCount,TileList");
                foreach(var kv in graph.OrderBy(k=>k.Key))
                {
                    var tilesStr = string.Join(";", kv.Value.Select(e=>$"{e.TileX},{e.TileY}"));
                    sw.WriteLine($"0x{kv.Key:X8},{kv.Value.Select(e=> (e.TileX,e.TileY)).Distinct().Count()},{kv.Value.Count},{tilesStr}");
                }
                Console.WriteLine($"  ↳ MSLK analysis → {csvPath}");

                // MSLK structure audit per tile
                var auditRows = new List<WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.AuditRow>();
                foreach(var (pm4, name) in processed)
                    auditRows.AddRange(WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.Audit(pm4,name));
                var auditCsv = Path.Combine(outDir,"mslk_structure_audit.csv");
                WoWToolbox.Core.v2.Services.PM4.MslkStructureAuditor.WriteCsv(auditRows,auditCsv);
                Console.WriteLine($"  ↳ MSLK structure audit → {auditCsv}");
            }
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
