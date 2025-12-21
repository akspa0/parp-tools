// PM4 Pipeline Orchestrator - Main entry point for clean PM4 to ADT pipeline
// Coordinates extraction, matching, building, and patching
// Part of the PM4 Clean Reimplementation

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

using WoWRollback.PM4Module.Generation;

namespace WoWRollback.PM4Module.Pipeline;

/// <summary>
/// Main orchestrator for the PM4 to ADT pipeline.
/// Coordinates all pipeline components to extract PM4 objects, match to WMOs,
/// build MODF entries, and patch ADT files.
/// </summary>
public class Pm4PipelineOrchestrator
{
    private readonly Pm4ObjectExtractor _extractor = new();
    private readonly Pm4DebugWmoWriter _debugWriter = new();
    private WmoMatcherService? _wmoMatcher;
    private readonly ModfEntryBuilder _modfBuilder = new();
    private readonly AdtPatcherV2 _adtPatcher = new();
    
    #region Public API
    
    /// <summary>
    /// Execute the full PM4 to ADT pipeline.
    /// </summary>
    public PipelineResult Execute(PipelineConfig config)
    {
        var stopwatch = Stopwatch.StartNew();
        var errors = new List<string>();
        var tileResults = new List<TileResult>();
        
        Console.WriteLine("\n=== PM4 Clean Pipeline V2 ===\n");
        Console.WriteLine($"PM4 Directory: {config.Pm4Directory}");
        Console.WriteLine($"Output Directory: {config.OutputDirectory}");
        Console.WriteLine($"WMO Library: {config.WmoLibraryPath ?? "None"}");
        Console.WriteLine($"Museum ADTs: {config.MuseumAdtDirectory ?? "None"}");
        Console.WriteLine($"Dry Run: {config.DryRun}");
        Console.WriteLine();
        
        // Validate inputs
        if (!Directory.Exists(config.Pm4Directory))
        {
            errors.Add($"PM4 directory not found: {config.Pm4Directory}");
            return CreateResult(0, 0, 0, 0, tileResults, errors, stopwatch.Elapsed);
        }
        
        // Create output directory
        if (!config.DryRun)
        {
            Directory.CreateDirectory(config.OutputDirectory);
            if (config.ExportCsv)
            {
                Directory.CreateDirectory(Path.Combine(config.OutputDirectory, "csv"));
            }
        }
        
        // Initialize matcher as null
        _wmoMatcher = null;
        
        // Step 2: Extract PM4 objects
        Console.WriteLine("\n[Step 2] Extracting PM4 objects...");
        var candidates = _extractor.ExtractAllWmoCandidates(config.Pm4Directory).ToList();
        Console.WriteLine($"[INFO] Extracted {candidates.Count} WMO candidates");
        
        if (candidates.Count == 0)
        {
            errors.Add("No WMO candidates extracted from PM4 files");
            return CreateResult(0, 0, 0, 0, tileResults, errors, stopwatch.Elapsed);
        }
        
        // Filter to single tile if specified
        if (!string.IsNullOrEmpty(config.SingleTile))
        {
            var parts = config.SingleTile.Split('_');
            if (parts.Length == 2 && int.TryParse(parts[0], out int tx) && int.TryParse(parts[1], out int ty))
            {
                candidates = candidates.Where(c => c.TileX == tx && c.TileY == ty).ToList();
                Console.WriteLine($"[INFO] Filtered to tile {config.SingleTile}: {candidates.Count} candidates");
            }
        }

        // Step 3: Debug Export (DISABLED - focus on matching real WMOs)
        // if (config.ExportDebugWmos && !config.DryRun)
        // {
        //     Console.WriteLine("\n[Step 2.5] Exporting Debug WMOs...");
        //     var debugDir = Path.Combine(config.OutputDirectory, "debug_wmos");
        //     
        //     // Group by CK24 to handle instance ID
        //     var byCk24 = candidates.GroupBy(c => c.CK24);
        //     int exportedCount = 0;
        //     
        //     foreach (var group in byCk24)
        //     {
        //         int instanceIdx = 0;
        //         foreach (var candidate in group)
        //         {
        //             // Naming: Debug_{CK24}_{TileId}_{InstanceId}
        //             var name = $"Debug_{candidate.CK24:X6}_{candidate.TileId}_{instanceIdx}";
        //             _debugWriter.WriteWmo(candidate, debugDir, name);
        //             instanceIdx++;
        //             exportedCount++;
        //         }
        //     }
        //     Console.WriteLine($"[INFO] Exported {exportedCount} Debug WMO files to {debugDir}");
        // }
        
        // Step 4: Load WMO library & Match
        if (string.IsNullOrEmpty(config.WmoLibraryPath))
        {
            Console.WriteLine("\n[INFO] No WMO library provided - skipping matching");
            // Return success with extracted counts
            return CreateResult(candidates.Count, 0, 0, 0, tileResults, errors, stopwatch.Elapsed);
        }

        Console.WriteLine("\n[Step 4] Loading WMO library...");
        
        // Check if library path is a directory or file
        if (Directory.Exists(config.WmoLibraryPath))
        {
            // Scan directory and build cache
            _wmoMatcher = new WmoMatcherService(null, config.SizeTolerance);
            
            // Check for existing cache file in the directory first
            string cacheFile = Path.Combine(config.WmoLibraryPath, "wmo_library_cache.json");
            if (File.Exists(cacheFile))
            {
                Console.WriteLine($"[INFO] Found existing cache: {cacheFile}");
                _wmoMatcher.LoadLibrary(cacheFile);
            }
            else
            {
                // Build it
                _wmoMatcher.BuildLibraryFromDirectory(config.WmoLibraryPath);
            }
        }
        else
        {
            // Load from specific file
            _wmoMatcher = new WmoMatcherService(config.WmoLibraryPath, config.SizeTolerance);
        }
        
        if (_wmoMatcher.LibraryCount == 0)
        {
            errors.Add("WMO library is empty - cannot match");
            return CreateResult(candidates.Count, 0, 0, 0, tileResults, errors, stopwatch.Elapsed);
        }

        Console.WriteLine("\n[Step 5] Matching to WMO library...");
        var matches = _wmoMatcher.FindAllMatches(candidates).ToList();
        Console.WriteLine($"[INFO] Matched {matches.Count} / {candidates.Count} candidates");
        
        if (matches.Count == 0)
        {
            errors.Add("No WMO matches found");
            return CreateResult(candidates.Count, 0, 0, 0, tileResults, errors, stopwatch.Elapsed);
        }
        
        // Step 6: Build MODF entries
        Console.WriteLine("\n[Step 6] Building MODF entries...");
        var modfEntries = _modfBuilder.CreateEntries(matches);
        var wmoNames = _modfBuilder.GetWmoNames();
        
        // Export CSV if requested
        if (config.ExportCsv && !config.DryRun)
        {
            var csvDir = Path.Combine(config.OutputDirectory, "csv");
            _modfBuilder.ExportToCsv(modfEntries, Path.Combine(csvDir, "modf_entries.csv"));
            _modfBuilder.ExportWmoNamesToCsv(Path.Combine(csvDir, "mwmo_names.csv"));
        }
        
        // Step 7: Patch ADT files
        Console.WriteLine("\n[Step 7] Patching ADT files...");
        int totalWmosPlaced = 0;
        int tilesProcessed = 0;
        int failedTiles = 0;
        
        if (config.DryRun)
        {
            Console.WriteLine("[DRY RUN] Skipping ADT patching");
            
            // Just count what would be processed
            var entriesByTile = modfEntries.GroupBy(e => GetTileFromEntry(e));
            foreach (var tileGroup in entriesByTile)
            {
                tilesProcessed++;
                totalWmosPlaced += tileGroup.Count();
                tileResults.Add(new TileResult(
                    TileId: $"{tileGroup.Key.x}_{tileGroup.Key.y}",
                    WmosPlaced: tileGroup.Count(),
                    M2sPlaced: 0,
                    Success: true
                ));
            }
        }
        else if (!string.IsNullOrEmpty(config.MuseumAdtDirectory) && Directory.Exists(config.MuseumAdtDirectory))
        {
            // Group entries by tile
            var entriesByTile = modfEntries
                .GroupBy(e => GetTileFromEntry(e))
                .ToDictionary(g => g.Key, g => g.ToList());
            
            Console.WriteLine($"[INFO] Patching {entriesByTile.Count} tiles...");
            
            foreach (var (tile, entries) in entriesByTile)
            {
                var sourceAdtPath = Path.Combine(config.MuseumAdtDirectory, $"development_{tile.x}_{tile.y}.adt");
                var outputAdtPath = Path.Combine(config.OutputDirectory, $"development_{tile.x}_{tile.y}.adt");
                
                if (!File.Exists(sourceAdtPath))
                {
                    tileResults.Add(new TileResult(
                        TileId: $"{tile.x}_{tile.y}",
                        WmosPlaced: 0,
                        M2sPlaced: 0,
                        Success: false,
                        Error: "Source ADT not found"
                    ));
                    failedTiles++;
                    continue;
                }
                
                var result = _adtPatcher.PatchWmoPlacements(
                    sourceAdtPath,
                    outputAdtPath,
                    entries,
                    wmoNames.ToList()
                );
                
                tilesProcessed++;
                if (result.Success)
                {
                    totalWmosPlaced += entries.Count;
                    tileResults.Add(new TileResult(
                        TileId: $"{tile.x}_{tile.y}",
                        WmosPlaced: entries.Count,
                        M2sPlaced: 0,
                        Success: true
                    ));
                }
                else
                {
                    failedTiles++;
                    tileResults.Add(new TileResult(
                        TileId: $"{tile.x}_{tile.y}",
                        WmosPlaced: 0,
                        M2sPlaced: 0,
                        Success: false,
                        Error: result.Error
                    ));
                }
            }
        }
        else
        {
            errors.Add("Museum ADT directory not specified or not found - skipping patching");
        }
        
        // Summary
        stopwatch.Stop();
        Console.WriteLine("\n=== Pipeline Complete ===");
        Console.WriteLine($"Candidates extracted: {candidates.Count}");
        Console.WriteLine($"WMOs matched: {matches.Count}");
        Console.WriteLine($"MODF entries: {modfEntries.Count}");
        Console.WriteLine($"Tiles processed: {tilesProcessed}");
        Console.WriteLine($"WMOs placed: {totalWmosPlaced}");
        Console.WriteLine($"Failed tiles: {failedTiles}");
        Console.WriteLine($"Duration: {stopwatch.Elapsed.TotalSeconds:F1}s");
        
        return CreateResult(tilesProcessed, totalWmosPlaced, 0, failedTiles, tileResults, errors, stopwatch.Elapsed);
    }
    
    /// <summary>
    /// Run pipeline with just extraction and matching (no patching).
    /// Useful for testing and CSV generation.
    /// </summary>
    public (List<Pm4WmoCandidate> Candidates, List<WmoMatch> Matches) AnalyzeOnly(PipelineConfig config)
    {
        Console.WriteLine("\n=== PM4 Analysis Mode ===\n");
        
        // Load WMO library
        _wmoMatcher = new WmoMatcherService(config.WmoLibraryPath, config.SizeTolerance);
        
        // Extract PM4 objects
        var candidates = _extractor.ExtractAllWmoCandidates(config.Pm4Directory).ToList();
        
        // Match to WMO library
        var matches = _wmoMatcher.FindAllMatches(candidates).ToList();
        
        return (candidates, matches);
    }
    
    #endregion
    
    #region Private Helpers
    
    private static PipelineResult CreateResult(
        int tilesProcessed,
        int wmosPlaced,
        int m2sPlaced,
        int failedTiles,
        List<TileResult> tileResults,
        List<string> errors,
        TimeSpan duration)
    {
        return new PipelineResult(
            TilesProcessed: tilesProcessed,
            TotalWmosPlaced: wmosPlaced,
            TotalM2sPlaced: m2sPlaced,
            FailedTiles: failedTiles,
            TileResults: tileResults,
            Errors: errors,
            Duration: duration
        );
    }
    
    private static (int x, int y) GetTileFromEntry(ModfEntry entry)
    {
        // Position is stored as XZY in MODF, use X and Z (which is original Y)
        const float TileSize = 533.33333f;
        int tileX = Math.Clamp((int)(32 - (entry.Position.X / TileSize)), 0, 63);
        int tileY = Math.Clamp((int)(32 - (entry.Position.Z / TileSize)), 0, 63);
        return (tileX, tileY);
    }
    
    #endregion
}
