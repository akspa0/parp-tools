using System;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class Pm4BatchProcessor : IPm4BatchProcessor
    {
        private readonly IBuildingExtractionService _buildingExtractionService;
        private readonly IWmoMatcher _wmoMatcher;

        public Pm4BatchProcessor(IBuildingExtractionService buildingExtractionService, IWmoMatcher wmoMatcher)
        {
            _buildingExtractionService = buildingExtractionService;
            _wmoMatcher = wmoMatcher;
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
            }
            catch (Exception ex)
            {
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }
            return result;
        }
    }
}
