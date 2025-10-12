using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Handles minimap extraction and organization for the viewer from loose PNG files.
/// Supports minimaps located in the same directory as ADT files.
/// </summary>
public sealed class MinimapHandler
{
    /// <summary>
    /// Scans for minimap PNGs and copies them to the output directory.
    /// Expected structure: World\Textures\Minimap\ or World\Textures\Minimap\{mapName}\
    /// Supports naming patterns: {mapName}_{X}_{Y}.png or map{X}_{Y}.png
    /// </summary>
    public MinimapResult ProcessMinimaps(string mapDirectory, string mapName, string outputDir)
    {
        try
        {
            // Resolve minimap directory
            // From: World\Maps\{mapName}\
            // To:   World\Textures\Minimap\ or World\Textures\Minimap\{mapName}\
            var minimapDir = ResolveMinimapDirectory(mapDirectory, mapName);
            
            if (minimapDir == null || !Directory.Exists(minimapDir))
            {
                return new MinimapResult(
                    Success: true,
                    TilesCopied: 0,
                    MinimapDir: null,
                    ErrorMessage: null); // Not an error - minimaps are optional
            }

            var minimapOutputDir = Path.Combine(outputDir, "minimaps");
            Directory.CreateDirectory(minimapOutputDir);

            var patterns = new[]
            {
                $"{mapName}_*.png",      // development_30_45.png
                "map*.png"               // map30_45.png
            };

            var foundFiles = new List<(string SourcePath, int TileX, int TileY)>();

            foreach (var pattern in patterns)
            {
                var files = Directory.GetFiles(minimapDir, pattern, SearchOption.TopDirectoryOnly);
                
                foreach (var file in files)
                {
                    var fileName = Path.GetFileNameWithoutExtension(file);
                    
                    // Try to parse coordinates
                    if (TryParseMinimapCoordinates(fileName, mapName, out var tileX, out var tileY))
                    {
                        foundFiles.Add((file, tileX, tileY));
                    }
                }
            }

            if (foundFiles.Count == 0)
            {
                return new MinimapResult(
                    Success: true,
                    TilesCopied: 0,
                    MinimapDir: null,
                    ErrorMessage: null); // Not an error - just no minimaps found
            }

            // Copy files to output directory with standardized naming
            int copied = 0;
            foreach (var (sourcePath, tileX, tileY) in foundFiles)
            {
                try
                {
                    var destFileName = $"{mapName}_{tileX}_{tileY}.png";
                    var destPath = Path.Combine(minimapOutputDir, destFileName);
                    
                    File.Copy(sourcePath, destPath, overwrite: true);
                    copied++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MinimapHandler] Warning: Failed to copy {Path.GetFileName(sourcePath)}: {ex.Message}");
                }
            }

            return new MinimapResult(
                Success: true,
                TilesCopied: copied,
                MinimapDir: minimapOutputDir,
                ErrorMessage: null);
        }
        catch (Exception ex)
        {
            return new MinimapResult(
                Success: false,
                TilesCopied: 0,
                ErrorMessage: $"Minimap processing failed: {ex.Message}");
        }
    }

    /// <summary>
    /// Resolves the minimap directory from the ADT map directory.
    /// Converts: World\Maps\{mapName}\ -> World\Textures\Minimap\{mapName}\ or World\Textures\Minimap\
    /// </summary>
    private static string? ResolveMinimapDirectory(string mapDirectory, string mapName)
    {
        try
        {
            // Convert to absolute path for reliable navigation
            var absoluteMapDir = Path.GetFullPath(mapDirectory);
            Console.WriteLine($"[MinimapHandler] Resolving minimaps from: {absoluteMapDir}");

            // Navigate from World\Maps\{mapName}\ to World\
            // Input: .../test_data/development/World/Maps/development/
            // We need to find the "World" directory by walking up the tree
            
            string? worldDir = null;
            var currentDir = absoluteMapDir;
            
            // Walk up directory tree looking for "World"
            for (int i = 0; i < 5; i++)  // Max 5 levels up
            {
                var dirName = Path.GetFileName(currentDir);
                Console.WriteLine($"[MinimapHandler] Checking level {i}: {currentDir} (name: {dirName})");
                
                if (dirName?.Equals("World", StringComparison.OrdinalIgnoreCase) == true)
                {
                    worldDir = currentDir;
                    break;
                }
                
                var parent = Path.GetDirectoryName(currentDir);
                if (string.IsNullOrEmpty(parent) || parent == currentDir)
                    break;
                    
                currentDir = parent;
            }
            
            if (string.IsNullOrEmpty(worldDir))
            {
                Console.WriteLine($"[MinimapHandler] Could not resolve World directory");
                return null;
            }

            Console.WriteLine($"[MinimapHandler] World directory: {worldDir}");

            // Check common locations for minimaps
            var candidates = new[]
            {
                Path.Combine(worldDir, "Textures", "Minimap", mapName),  // World\Textures\Minimap\{mapName}\
                Path.Combine(worldDir, "Textures", "Minimap"),           // World\Textures\Minimap\
                Path.Combine(worldDir, "textures", "minimap", mapName),  // lowercase
                Path.Combine(worldDir, "textures", "minimap"),           // lowercase
                Path.Combine(worldDir, "Minimaps", mapName),             // Alternative structure
                Path.Combine(worldDir, "Minimaps")                       // Alternative structure
            };

            for (int i = 0; i < candidates.Length; i++)
            {
                var candidate = candidates[i];
                bool exists = Directory.Exists(candidate);
                Console.WriteLine($"[MinimapHandler] Candidate {i + 1}: {candidate} (exists: {exists})");
                
                if (exists)
                {
                    // Verify it contains PNG files
                    var pngFiles = Directory.EnumerateFiles(candidate, "*.png", SearchOption.TopDirectoryOnly).ToList();
                    if (pngFiles.Count > 0)
                    {
                        Console.WriteLine($"[MinimapHandler] ✓ Found {pngFiles.Count} PNG files - using this directory");
                        return candidate;
                    }
                    else
                    {
                        Console.WriteLine($"[MinimapHandler] Directory exists but contains no PNG files");
                    }
                }
            }

            Console.WriteLine($"[MinimapHandler] No minimap directory found");
            return null;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[MinimapHandler] Error resolving minimap directory: {ex.Message}");
            return null;
        }
    }

    private static bool TryParseMinimapCoordinates(string fileName, string mapName, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;

        // Try pattern: {mapName}_{X}_{Y}
        if (fileName.StartsWith(mapName + "_", StringComparison.OrdinalIgnoreCase))
        {
            var suffix = fileName.Substring(mapName.Length + 1);
            var parts = suffix.Split('_');
            
            if (parts.Length >= 2 &&
                int.TryParse(parts[0], out tileX) &&
                int.TryParse(parts[1], out tileY))
            {
                return true;
            }
        }

        // Try pattern: map{X}_{Y}
        if (fileName.StartsWith("map", StringComparison.OrdinalIgnoreCase))
        {
            var suffix = fileName.Substring(3);
            var parts = suffix.Split('_');
            
            if (parts.Length >= 2 &&
                int.TryParse(parts[0], out tileX) &&
                int.TryParse(parts[1], out tileY))
            {
                return true;
            }
        }

        return false;
    }
}

/// <summary>
/// Result of minimap processing.
/// </summary>
public record MinimapResult(
    bool Success,
    int TilesCopied,
    string? MinimapDir = null,
    string? ErrorMessage = null);
