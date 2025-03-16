using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Exports PM4 file data in various formats for analysis and visualization.
    /// </summary>
    public class PM4DataExporter
    {
        private readonly ILogger? _logger;
        private readonly PM4CsvGenerator _csvGenerator;
        private readonly PM4TerrainExporter _terrainExporter;

        public PM4DataExporter(ILogger? logger = null)
        {
            _logger = logger;
            _csvGenerator = new PM4CsvGenerator(logger);
            _terrainExporter = new PM4TerrainExporter(logger);
        }

        /// <summary>
        /// Exports data from a PM4 file in various formats for analysis and visualization.
        /// </summary>
        /// <param name="file">The PM4 file to export data from</param>
        /// <param name="outputDir">The directory to save output files to</param>
        /// <returns>An asynchronous task</returns>
        public async Task ExportDataAsync(PM4File file, string outputDir)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));
            if (string.IsNullOrEmpty(outputDir))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDir));

            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            // Export position data as CSV
            if (file.PositionDataChunk != null && file.PositionDataChunk.Entries.Count > 0)
            {
                string baseFileName = Path.GetFileNameWithoutExtension(file.FileName ?? "unknown");
                await _csvGenerator.GenerateReportsAsync(file.PositionDataChunk, baseFileName, outputDir);
                
                // Export as terrain points (combining Special entries with Position entries)
                await _terrainExporter.ExportTerrainDataAsync(file, outputDir);
            }
            else
            {
                _logger?.LogWarning("No position data found in {FileName}", file.FileName ?? "unknown");
            }
        }
    }
} 