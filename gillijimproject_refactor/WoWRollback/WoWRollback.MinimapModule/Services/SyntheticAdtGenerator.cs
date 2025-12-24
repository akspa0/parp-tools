using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace WoWRollback.MinimapModule.Services
{
    public class SyntheticAdtGenerator
    {
        private readonly ILogger<SyntheticAdtGenerator> _logger;

        public SyntheticAdtGenerator(ILogger<SyntheticAdtGenerator> logger)
        {
            _logger = logger;
        }

        public async Task GenerateCalibrationMapAsync(string outputDir)
        {
            _logger.LogInformation("Generating calibration ADT data...");
            // TODO: 
            // 1. Create flat ADT
            // 2. Apply slope gradients
            // 3. Apply texture palettes
            // 4. Save to disk
            await Task.CompletedTask;
        }
    }
}
