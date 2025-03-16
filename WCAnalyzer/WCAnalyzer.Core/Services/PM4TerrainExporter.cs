using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Exports terrain data from PM4 files to various formats.
    /// </summary>
    public class PM4TerrainExporter
    {
        private readonly ILogger<PM4TerrainExporter>? _logger;
        private readonly string _outputDirectory;

        /// <summary>
        /// Initializes a new instance of the <see cref="PM4TerrainExporter"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="outputDirectory">The output directory.</param>
        public PM4TerrainExporter(ILogger<PM4TerrainExporter>? logger, string outputDirectory)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
            
            // Create output directory if it doesn't exist
            Directory.CreateDirectory(_outputDirectory);
        }

        /// <summary>
        /// Extracts terrain data from PM4 position data.
        /// </summary>
        /// <param name="results">The PM4 analysis results.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task ExtractTerrainDataAsync(IEnumerable<PM4AnalysisResult> results)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));

            var resultsList = results.ToList();
            if (resultsList.Count == 0)
            {
                _logger?.LogWarning("No PM4 analysis results to extract terrain data from.");
                return;
            }

            _logger?.LogInformation("Extracting terrain data from {Count} PM4 files", resultsList.Count);

            // Create a terrain data directory
            var terrainDirectory = Path.Combine(_outputDirectory, "terrain_data");
            Directory.CreateDirectory(terrainDirectory);

            // Extract terrain data from each PM4 file
            foreach (var result in resultsList)
            {
                if (result.PM4File == null)
                {
                    _logger?.LogWarning("PM4 file is null for {FileName}, skipping terrain extraction", result.FileName);
                    continue;
                }

                try
                {
                    await ExtractTerrainDataFromFileAsync(result, terrainDirectory);
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error extracting terrain data from {FileName}", result.FileName);
                }
            }

            _logger?.LogInformation("Terrain data extraction completed");
        }

        /// <summary>
        /// Extracts terrain data from a single PM4 file.
        /// </summary>
        /// <param name="result">The PM4 analysis result.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task ExtractTerrainDataFromFileAsync(PM4AnalysisResult result, string outputDirectory)
        {
            if (result.PM4File == null)
                return;

            var fileName = Path.GetFileNameWithoutExtension(result.FileName);
            _logger?.LogInformation("Extracting terrain data from {FileName}", fileName);

            // Get position data from MPRL chunk
            var mprlChunk = result.PM4File.PositionDataChunk;
            if (mprlChunk == null)
            {
                _logger?.LogWarning("No MPRL chunk found in {FileName}, skipping terrain extraction", fileName);
                return;
            }

            // Extract terrain data from position data
            var terrainData = ExtractTerrainData(mprlChunk, fileName);
            if (terrainData.Count == 0)
            {
                _logger?.LogWarning("No terrain data extracted from {FileName}", fileName);
                return;
            }

            // Write terrain data to CSV file
            var csvPath = Path.Combine(outputDirectory, $"{fileName}_terrain.csv");
            await File.WriteAllTextAsync(csvPath, GenerateTerrainCsv(terrainData));
            _logger?.LogInformation("Wrote terrain data to {CsvPath}", csvPath);

            // Write terrain data to OBJ file
            var objPath = Path.Combine(outputDirectory, $"{fileName}_terrain.obj");
            await File.WriteAllTextAsync(objPath, GenerateTerrainObj(terrainData));
            _logger?.LogInformation("Wrote terrain data to {ObjPath}", objPath);
        }

        /// <summary>
        /// Extracts terrain data from position data.
        /// </summary>
        /// <param name="mprlChunk">The MPRL chunk.</param>
        /// <param name="fileName">The file name.</param>
        /// <returns>A list of terrain data points.</returns>
        private List<TerrainDataPoint> ExtractTerrainData(MPRLChunk mprlChunk, string fileName)
        {
            var terrainData = new List<TerrainDataPoint>();

            // Process each position
            for (int i = 0; i < mprlChunk.Entries.Count; i++)
            {
                var entry = mprlChunk.Entries[i];
                
                // Skip special entries
                if (entry.IsSpecialEntry)
                    continue;

                // Create terrain data point
                var dataPoint = new TerrainDataPoint
                {
                    FileName = fileName,
                    PositionIndex = entry.Index,
                    Position = new Vector3(entry.CoordinateX, entry.CoordinateY, entry.CoordinateZ),
                    Flag = (uint)entry.SpecialValue
                };

                terrainData.Add(dataPoint);
            }

            return terrainData;
        }

        /// <summary>
        /// Generates a CSV file from terrain data.
        /// </summary>
        /// <param name="terrainData">The terrain data.</param>
        /// <returns>A CSV string.</returns>
        private string GenerateTerrainCsv(List<TerrainDataPoint> terrainData)
        {
            var sb = new StringBuilder();
            sb.AppendLine("FileName,PositionIndex,X,Y,Z,Flag");

            foreach (var point in terrainData)
            {
                sb.AppendLine($"{point.FileName},{point.PositionIndex},{point.Position.X},{point.Position.Y},{point.Position.Z},{point.Flag}");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Generates an OBJ file from terrain data.
        /// </summary>
        /// <param name="terrainData">The terrain data.</param>
        /// <returns>An OBJ string.</returns>
        private string GenerateTerrainObj(List<TerrainDataPoint> terrainData)
        {
            var sb = new StringBuilder();
            sb.AppendLine("# Terrain data extracted from PM4 file");
            sb.AppendLine("# Generated by WCAnalyzer");
            sb.AppendLine();

            // Write vertices
            foreach (var point in terrainData)
            {
                // Convert coordinates to OBJ format (Y up)
                sb.AppendLine($"v {point.Position.X} {point.Position.Y} {point.Position.Z}");
            }

            return sb.ToString();
        }

        /// <summary>
        /// Represents a terrain data point.
        /// </summary>
        private class TerrainDataPoint
        {
            /// <summary>
            /// Gets or sets the file name.
            /// </summary>
            public string FileName { get; set; } = string.Empty;

            /// <summary>
            /// Gets or sets the position index.
            /// </summary>
            public int PositionIndex { get; set; }

            /// <summary>
            /// Gets or sets the position.
            /// </summary>
            public Vector3 Position { get; set; }

            /// <summary>
            /// Gets or sets the flag.
            /// </summary>
            public uint Flag { get; set; }
        }
    }
} 