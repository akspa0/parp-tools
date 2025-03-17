using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Exports PM4 file data in CSV format for analysis.
    /// </summary>
    public class PM4DataExporter
    {
        private readonly ILogger _logger;
        private readonly PM4CsvGenerator _csvGenerator;

        public PM4DataExporter(ILogger? logger = null)
        {
            _logger = logger ?? NullLogger.Instance;
            _csvGenerator = new PM4CsvGenerator(logger as ILogger<PM4CsvGenerator>);
        }

        /// <summary>
        /// Exports data from a PM4 analysis result as CSV for analysis.
        /// </summary>
        /// <param name="result">The PM4 analysis result to export data from</param>
        /// <param name="outputDir">The directory to save output files to</param>
        /// <returns>An asynchronous task</returns>
        public async Task ExportDataAsync(PM4AnalysisResult result, string outputDir)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            if (string.IsNullOrEmpty(outputDir))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDir));

            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Export data as CSV
            await _csvGenerator.GenerateReportsAsync(result, outputDir);
            
            _logger.LogInformation("Generated CSV reports for {FileName}", result.FileName);
        }
    }
} 