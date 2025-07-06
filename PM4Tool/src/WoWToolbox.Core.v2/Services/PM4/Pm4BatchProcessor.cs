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
            try
            {
                var pm4File = PM4File.FromFile(pm4FilePath);
                var buildingFragments = _buildingExtractionService.ExtractBuildings(pm4File);
                result.BuildingFragments.AddRange(buildingFragments);
                var wmoMatches = _wmoMatcher.Match(buildingFragments);
                result.WmoMatches.AddRange(wmoMatches);
                result.Success = true;
                // write simple summary
                WriteSummary(pm4FilePath, result);
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }
            return result;

            void WriteSummary(string pm4Path, BatchProcessResult r)
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
                        $"Matches: {r.WmoMatches.Count}"
                    });
                }
                catch { /* non-fatal */ }
            }
        }
    }
}
