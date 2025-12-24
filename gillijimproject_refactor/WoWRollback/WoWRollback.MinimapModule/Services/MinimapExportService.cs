using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WoWRollback.Core.Services;

namespace WoWRollback.MinimapModule.Services
{
    public class MinimapExportService
    {
        private readonly ILogger<MinimapExportService> _logger;
        public MinimapExportService(ILogger<MinimapExportService> logger)
        {
            _logger = logger;
        }

        public async Task ExportGroundTruthAsync(string mapName, string outputDir)
        {
            _logger.LogInformation($"Starting Ground Truth Export for map: {mapName}");
            // TODO: Implementation
            // 1. Load WDT
            // 2. Iterate Tiles
            // 3. Extract Minimap BLP -> PNG
            // 4. Extract ADT Height -> TIFF
        }
    }
}
