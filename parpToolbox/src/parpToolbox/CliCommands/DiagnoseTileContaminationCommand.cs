using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Diagnostic command to identify cross-tile contamination in PM4 objects
    /// </summary>
    public class DiagnoseTileContaminationCommand
    {
        public void Execute(string inputPath, string outputPath)
        {
            ConsoleLogger.WriteLine("=== PM4 Tile Contamination Diagnostic ===");
            ConsoleLogger.WriteLine($"Input: {inputPath}");
            ConsoleLogger.WriteLine($"Output: {outputPath}");

            try
            {
                // Validate input file
                if (!File.Exists(inputPath))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file not found: {inputPath}");
                    return;
                }

                // Create output directory
                Directory.CreateDirectory(outputPath);

                // Load PM4 scene
                ConsoleLogger.WriteLine("Loading PM4 scene...");
                var adapter = new Pm4Adapter();
                var scene = adapter.LoadRegion(inputPath);

                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load PM4 scene");
                    return;
                }

                ConsoleLogger.WriteLine($"Scene loaded: {scene.Links?.Count ?? 0} links, {scene.Surfaces?.Count ?? 0} surfaces");

                // Analyze MSLK tile coordinate validity
                ConsoleLogger.WriteLine("\n=== MSLK Tile Coordinate Analysis ===");
                int totalLinks = scene.Links?.Count ?? 0;
                int validTileCoords = 0;
                int invalidTileCoords = 0;
                var tileFrequency = new Dictionary<(int x, int y), int>();

                if (scene.Links != null)
                {
                    foreach (var link in scene.Links)
                    {
                        if (link.TryDecodeTileCoordinates(out int tileX, out int tileY))
                        {
                            validTileCoords++;
                            var coords = (tileX, tileY);
                            tileFrequency[coords] = tileFrequency.GetValueOrDefault(coords, 0) + 1;
                        }
                        else
                        {
                            invalidTileCoords++;
                        }
                    }
                }

                ConsoleLogger.WriteLine($"Valid tile coordinates: {validTileCoords}/{totalLinks} ({(double)validTileCoords/totalLinks*100:F1}%)");
                ConsoleLogger.WriteLine($"Invalid tile coordinates: {invalidTileCoords}/{totalLinks} ({(double)invalidTileCoords/totalLinks*100:F1}%)");

                // Report tile distribution
                ConsoleLogger.WriteLine("\n=== Tile Distribution ===");
                foreach (var kvp in tileFrequency.OrderByDescending(x => x.Value))
                {
                    ConsoleLogger.WriteLine($"Tile ({kvp.Key.x}, {kvp.Key.y}): {kvp.Value} links");
                }

                // Identify primary tile
                (int, int)? primaryTile = null;
                if (tileFrequency.Count > 0)
                {
                    primaryTile = tileFrequency.OrderByDescending(x => x.Value).First().Key;
                }
                
                if (primaryTile.HasValue)
                {
                    ConsoleLogger.WriteLine($"\nPrimary tile: ({primaryTile.Value.Item1}, {primaryTile.Value.Item2})");
                    
                    // Count cross-tile contamination
                    int primaryTileLinks = tileFrequency[primaryTile.Value];
                    int crossTileLinks = validTileCoords - primaryTileLinks;
                    int unknownTileLinks = invalidTileCoords;

                    ConsoleLogger.WriteLine($"Primary tile links: {primaryTileLinks}");
                    ConsoleLogger.WriteLine($"Cross-tile links: {crossTileLinks}");
                    ConsoleLogger.WriteLine($"Unknown tile links: {unknownTileLinks}");
                    ConsoleLogger.WriteLine($"Contamination rate: {(double)(crossTileLinks + unknownTileLinks)/totalLinks*100:F1}%");
                }
                else
                {
                    ConsoleLogger.WriteLine("\nWARNING: No primary tile identified - all coordinates invalid!");
                }

                // Write detailed report
                string reportPath = Path.Combine(outputPath, "tile_contamination_report.txt");
                using (var writer = new StreamWriter(reportPath))
                {
                    writer.WriteLine("PM4 Tile Contamination Analysis Report");
                    writer.WriteLine($"Generated: {DateTime.Now}");
                    writer.WriteLine($"Input file: {inputPath}");
                    writer.WriteLine();
                    
                    writer.WriteLine($"Total MSLK entries: {totalLinks}");
                    writer.WriteLine($"Valid tile coordinates: {validTileCoords} ({(double)validTileCoords/totalLinks*100:F1}%)");
                    writer.WriteLine($"Invalid tile coordinates: {invalidTileCoords} ({(double)invalidTileCoords/totalLinks*100:F1}%)");
                    writer.WriteLine();
                    
                    writer.WriteLine("Tile Distribution:");
                    foreach (var kvp in tileFrequency.OrderByDescending(x => x.Value))
                    {
                        writer.WriteLine($"  Tile ({kvp.Key.x}, {kvp.Key.y}): {kvp.Value} links");
                    }
                    
                    if (primaryTile.HasValue)
                    {
                        writer.WriteLine();
                        writer.WriteLine($"Primary tile: ({primaryTile.Value.Item1}, {primaryTile.Value.Item2})");
                        int primaryTileLinks = tileFrequency[primaryTile.Value];
                        int crossTileLinks = validTileCoords - primaryTileLinks;
                        int unknownTileLinks = invalidTileCoords;
                        writer.WriteLine($"Contamination analysis:");
                        writer.WriteLine($"  Primary tile links: {primaryTileLinks}");
                        writer.WriteLine($"  Cross-tile links: {crossTileLinks}");
                        writer.WriteLine($"  Unknown tile links: {unknownTileLinks}");
                        writer.WriteLine($"  Contamination rate: {(double)(crossTileLinks + unknownTileLinks)/totalLinks*100:F1}%");
                    }
                }

                ConsoleLogger.WriteLine($"\nDetailed report saved to: {reportPath}");
                ConsoleLogger.WriteLine("=== Tile Contamination Diagnostic Complete ===");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }
    }
}
