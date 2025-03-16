using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using Microsoft.Extensions.Logging;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using WCAnalyzer.Core.Models.PM4;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for generating 2D map visualizations from PM4 position data.
    /// </summary>
    public class PM4MapImageGenerator
    {
        private readonly ILogger<PM4MapImageGenerator>? _logger;
        private readonly string _outputDirectory;
        
        // Default image dimensions
        private const int DefaultImageWidth = 4096;
        private const int DefaultImageHeight = 4096;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="PM4MapImageGenerator"/> class.
        /// </summary>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="outputDirectory">Output directory for generated images</param>
        public PM4MapImageGenerator(ILogger<PM4MapImageGenerator>? logger = null, string? outputDirectory = null)
        {
            _logger = logger;
            _outputDirectory = outputDirectory ?? System.IO.Path.Combine(Directory.GetCurrentDirectory(), "output");
        }
        
        /// <summary>
        /// Generates a 2D map visualization from position data in PM4 files.
        /// </summary>
        /// <param name="results">The PM4 analysis results containing position data</param>
        /// <param name="outputFileName">Name of the output image file (without path)</param>
        /// <param name="width">Width of the output image in pixels (default: 4096)</param>
        /// <param name="height">Height of the output image in pixels (default: 4096)</param>
        /// <returns>True if the image was generated successfully</returns>
        public bool GenerateMapImage(
            IEnumerable<PM4AnalysisResult> results, 
            string outputFileName = "azeroth_map.png",
            int width = DefaultImageWidth,
            int height = DefaultImageHeight)
        {
            try
            {
                _logger?.LogInformation("Generating 2D map visualization from position data");
                
                // Ensure output directory exists
                Directory.CreateDirectory(_outputDirectory);
                
                // Extract positions where IsControlRecord = false (terrain points)
                var positions = new List<Vector2>();
                
                foreach (var result in results)
                {
                    if (result.PM4File?.PositionDataChunk == null) continue;
                    
                    var filePositions = result.PM4File.PositionDataChunk.Entries
                        .Where(e => !e.IsControlRecord)  // Skip control records
                        // Map the 3D coordinates to 2D: X = CoordinateX, Y = CoordinateY
                        .Select(p => new Vector2(p.CoordinateX, p.CoordinateY))
                        .Where(p => !float.IsNaN(p.X) && !float.IsNaN(p.Y))  // Filter invalid values
                        .ToList();
                    
                    positions.AddRange(filePositions);
                    
                    _logger?.LogInformation("Extracted {Count} valid positions from {FileName}", 
                        filePositions.Count, result.FileName);
                }
                
                if (!positions.Any())
                {
                    _logger?.LogWarning("No valid position data found in any PM4 files");
                    return false;
                }
                
                _logger?.LogInformation("Total positions for visualization: {Count}", positions.Count);
                
                // Find the actual data range
                float minX = positions.Min(p => p.X);
                float maxX = positions.Max(p => p.X);
                float minY = positions.Min(p => p.Y);
                float maxY = positions.Max(p => p.Y);
                
                _logger?.LogInformation("Actual coordinate range: X: {MinX} to {MaxX}, Y: {MinY} to {MaxY}",
                    minX, maxX, minY, maxY);
                
                float rangeX = maxX - minX;
                float rangeY = maxY - minY;
                
                // Add a small margin (5%)
                float marginX = rangeX * 0.05f;
                float marginY = rangeY * 0.05f;
                
                minX -= marginX;
                maxX += marginX;
                minY -= marginY;
                maxY += marginY;
                
                // Create simple scatter plot
                string scatterPlotPath = System.IO.Path.Combine(_outputDirectory, outputFileName);
                CreateScatterPlot(positions, scatterPlotPath, minX, maxX, minY, maxY, width, height);
                
                // Create a CSV file for Excel import
                string csvPath = System.IO.Path.Combine(
                    _outputDirectory, 
                    System.IO.Path.GetFileNameWithoutExtension(outputFileName) + "_points.csv");
                
                CreateCsvFile(positions, csvPath);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to generate map visualization: {Message}", ex.Message);
                return false;
            }
        }
        
        /// <summary>
        /// Creates a simple scatter plot of the position data.
        /// </summary>
        private void CreateScatterPlot(
            List<Vector2> positions, 
            string outputPath, 
            float minX, float maxX, 
            float minY, float maxY, 
            int width, 
            int height)
        {
            // Create new image
            using var image = new Image<Rgba32>(width, height);
            
            // Fill background with dark color
            image.Mutate(ctx => ctx.Fill(Color.Black));
            
            // Calculate scale factors for coordinate mapping
            float scaleX = width / (maxX - minX);
            float scaleY = height / (maxY - minY);
            
            // Draw points
            const int pointRadius = 2;
            var pointColor = Color.FromRgba(64, 128, 255, 200);
            
            foreach (var pos in positions)
            {
                // Convert world coordinates to image coordinates
                int x = (int)((pos.X - minX) * scaleX);
                int y = (int)(height - ((pos.Y - minY) * scaleY));  // Y is inverted in image space
                
                // Ensure point is within image bounds
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    image.Mutate(ctx => ctx.Fill(new SolidBrush(pointColor), new EllipsePolygon(new PointF(x, y), pointRadius)));
                }
            }
            
            // Draw border lines (axes)
            var axisColor = Color.FromRgba(100, 100, 100, 255);
            
            // Draw simple axes
            image.Mutate(ctx => {
                // Draw border - use individual DrawLine calls instead of DrawLines
                ctx.DrawLine(axisColor, 1f, new PointF(0, 0), new PointF(width-1, 0));
                ctx.DrawLine(axisColor, 1f, new PointF(width-1, 0), new PointF(width-1, height-1));
                ctx.DrawLine(axisColor, 1f, new PointF(width-1, height-1), new PointF(0, height-1));
                ctx.DrawLine(axisColor, 1f, new PointF(0, height-1), new PointF(0, 0));
            });
            
            // Draw coordinate info
            try
            {
                var font = new Font(SystemFonts.Get("Arial"), 16);
                
                // Title and count
                image.Mutate(ctx => ctx.DrawText(
                    $"Azeroth World Map - {positions.Count:N0} points",
                    font,
                    Color.White,
                    new PointF(20, 20)));
                
                // Coordinate ranges
                image.Mutate(ctx => ctx.DrawText(
                    $"X: {minX:F1} to {maxX:F1}",
                    font,
                    Color.LightGray,
                    new PointF(20, 50)));
                
                image.Mutate(ctx => ctx.DrawText(
                    $"Y: {minY:F1} to {maxY:F1}",
                    font,
                    Color.LightGray,
                    new PointF(20, 80)));
                
                // Note about the discovery
                var noteFont = new Font(SystemFonts.Get("Arial"), 12, FontStyle.Italic);
                image.Mutate(ctx => ctx.DrawText(
                    "WoW terrain map from March 2002 development",
                    noteFont,
                    Color.FromRgba(200, 200, 255, 255),
                    new PointF(20, height - 30)));
            }
            catch (Exception ex)
            {
                _logger?.LogWarning("Error drawing text: {Message}", ex.Message);
            }
            
            // Save the image
            image.Save(outputPath);
            _logger?.LogInformation("Scatter plot saved to: {Path}", outputPath);
        }
        
        /// <summary>
        /// Creates a CSV file with the position data for import into Excel.
        /// </summary>
        private void CreateCsvFile(List<Vector2> positions, string outputPath)
        {
            try
            {
                using var writer = new StreamWriter(outputPath);
                
                // Write header
                writer.WriteLine("X,Y");
                
                // Write position data with fixed decimal format to avoid scientific notation
                foreach (var pos in positions)
                {
                    // Format with fixed decimal notation (F6 ensures 6 decimal places)
                    writer.WriteLine($"{pos.X.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)},{pos.Y.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)}");
                }
                
                _logger?.LogInformation("CSV file for Excel saved to: {Path}", outputPath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to create CSV file: {Message}", ex.Message);
            }
        }
    }
} 