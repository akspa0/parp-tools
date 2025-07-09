using System;
using System.Collections.Generic;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Infrastructure;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class Pm4BatchProcessor : IPm4BatchProcessor
    {
        private readonly IBuildingExtractionService _buildingExtractionService;
        private readonly IWmoMatcher _wmoMatcher;
        public string RunDirectory { get; }

        public Pm4BatchProcessor(IBuildingExtractionService buildingExtractionService, IWmoMatcher wmoMatcher)
        {
            _buildingExtractionService = buildingExtractionService;
            _wmoMatcher = wmoMatcher;
            RunDirectory = Infrastructure.ProjectOutput.GetPath("pm4");
        }

        public BatchProcessResult Process(string pm4FilePath)
        {
            var result = new BatchProcessResult();
            // declare counts so they are visible to catch/final summary
            int mspvCount = 0;
            int msvtCount = 0;
            int msviCount = 0;
            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);
                mspvCount = pm4File.MSPV?.Vertices.Count ?? 0;
                msvtCount = pm4File.MSVT?.Vertices.Count ?? 0;
                msviCount = pm4File.MSVI?.Indices.Count ?? 0;
                var buildingFragments = _buildingExtractionService.ExtractBuildings(pm4File);
                result.BuildingFragments.AddRange(buildingFragments);
                var wmoMatches = _wmoMatcher.Match(buildingFragments);
                result.WmoMatches.AddRange(wmoMatches);

                // Export geometry to OBJ for parity with legacy batch tool (only if any vertices present)
                string fileStem = Path.GetFileNameWithoutExtension(pm4FilePath);
                string dir = Path.Combine(RunDirectory, fileStem);
                Directory.CreateDirectory(dir);
                string objPath = Path.Combine(dir, fileStem + ".obj");

                if (mspvCount > 0 || msvtCount > 0)
                {
                    LegacyObjExporter.ExportAsync(pm4File, objPath, Path.GetFileName(pm4FilePath)).GetAwaiter().GetResult();
                }
                else
                {
                    result.ErrorMessage = "No vertex chunks present (MSPV/MSVT) – OBJ not generated.";
                }

                result.Success = true;
                // write simple summary
                WriteSummary(pm4FilePath, result, mspvCount, msvtCount, msviCount);
            }
            catch (Exception ex)
            {
                // Missing geometry chunks (e.g., MSPV / MSVT absent) are common and not critical – we still
                // want a summary and diagnostics.  Mark success true but record a warning.
                if (ex.Message.Contains("no MSPV vertices", StringComparison.OrdinalIgnoreCase) ||
                    ex.Message.Contains("Chunk \"MSVT\" not found", StringComparison.OrdinalIgnoreCase))
                {
                    result.Success = true;
                    result.ErrorMessage = $"Warning: {ex.Message}";
                    WriteSummary(pm4FilePath, result, mspvCount, msvtCount, msviCount);
                }
                else
                {
                    result.Success = false;
                    result.ErrorMessage = ex.Message;
                }
            }
            return result;

            void WriteSummary(string pm4Path, BatchProcessResult r, int mspv, int msvt, int msvi)
            {
                try
                {
                    string fileName = Path.GetFileNameWithoutExtension(pm4Path);
                    string dir = Path.Combine(RunDirectory, fileName);
                    Directory.CreateDirectory(dir);
                    string summaryPath = Path.Combine(dir, "summary.txt");
                    File.WriteAllLines(summaryPath, new[]
                    {
                        $"Success: {r.Success}",
                        $"Fragments: {r.BuildingFragments.Count}",
                        $"Matches: {r.WmoMatches.Count}",
                        $"MSPV vertices: {mspv}",
                        $"MSVT vertices: {msvt}",
                        $"MSVI indices: {msvi}"
                    });
                }
                catch { /* non-fatal */ }
            }
        }
    }
}
