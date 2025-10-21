using System.CommandLine;
using System.Globalization;
using System.Text.Json;
using CsvHelper;
using ScottPlot;
using WoWDataPlot.Extractors;
using WoWDataPlot.Helpers;
using WoWDataPlot.Models;

namespace WoWDataPlot;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("WoW Data Archaeology - Simple visualization tool for WDT/ADT data");

        // plot-uniqueid command
        var plotCommand = new Command("plot-uniqueid", "Plot M2/WMO UniqueID distribution from Alpha WDT");
        var wdtOption = new Option<FileInfo>("--wdt", "Path to Alpha WDT file") { IsRequired = true };
        var outputOption = new Option<FileInfo>("--output", "Output PNG file path") { IsRequired = true };
        var widthOption = new Option<int>("--width", () => 2048, "Image width in pixels");
        var heightOption = new Option<int>("--height", () => 2048, "Image height in pixels");
        var typeFilterOption = new Option<string?>("--type", "Filter by type (M2 or WMO)");
        
        plotCommand.AddOption(wdtOption);
        plotCommand.AddOption(outputOption);
        plotCommand.AddOption(widthOption);
        plotCommand.AddOption(heightOption);
        plotCommand.AddOption(typeFilterOption);
        
        plotCommand.SetHandler(PlotUniqueId, wdtOption, outputOption, widthOption, heightOption, typeFilterOption);
        
        // export-csv command
        var exportCommand = new Command("export-csv", "Export placement data to CSV for external analysis");
        var wdtOptionExport = new Option<FileInfo>("--wdt", "Path to Alpha WDT file") { IsRequired = true };
        var outputOptionExport = new Option<FileInfo>("--output", "Output CSV file path") { IsRequired = true };
        
        exportCommand.AddOption(wdtOptionExport);
        exportCommand.AddOption(outputOptionExport);
        
        exportCommand.SetHandler(ExportCsv, wdtOptionExport, outputOptionExport);
        
        // visualize command - UNIFIED PIPELINE
        var visualizeCommand = new Command("visualize", "Complete pipeline: analyze layers and generate all visualizations");
        var wdtOptionVisualize = new Option<FileInfo>("--wdt", "Path to Alpha WDT file") { IsRequired = true };
        var outputDirOptionVisualize = new Option<DirectoryInfo>("--output-dir", "Output directory for all results") { IsRequired = true };
        var gapThresholdOption = new Option<int>("--gap-threshold", () => 50, "UniqueID gap size to split layers (gaps = dev pauses)");
        var tileSizeOption = new Option<int>("--tile-size", () => 1024, "Tile image size (square)");
        var mapSizeOption = new Option<int>("--map-size", () => 2048, "Map overview size (square)");
        var tileMarkerSizeOption = new Option<float>("--tile-marker-size", () => 8.0f, "Marker size for tile images");
        var mapMarkerSizeOption = new Option<float>("--map-marker-size", () => 5.0f, "Marker size for map overview");
        var mapMaxLayersOption = new Option<int>("--map-max-layers", () => 0, "Max layers to show on map overview (0 = all)");
        var minimapDirOption = new Option<DirectoryInfo?>("--minimap-dir", () => null, "Directory containing minimap tiles (optional)");
        
        visualizeCommand.AddOption(wdtOptionVisualize);
        visualizeCommand.AddOption(outputDirOptionVisualize);
        visualizeCommand.AddOption(gapThresholdOption);
        visualizeCommand.AddOption(tileSizeOption);
        visualizeCommand.AddOption(mapSizeOption);
        visualizeCommand.AddOption(tileMarkerSizeOption);
        visualizeCommand.AddOption(mapMarkerSizeOption);
        visualizeCommand.AddOption(mapMaxLayersOption);
        // minimapDirOption accepted but not wired (coming soon)
        
        visualizeCommand.SetHandler(VisualizeComplete, wdtOptionVisualize, outputDirOptionVisualize, 
            gapThresholdOption, tileSizeOption, mapSizeOption, tileMarkerSizeOption, mapMarkerSizeOption, mapMaxLayersOption);
        
        rootCommand.AddCommand(plotCommand);
        rootCommand.AddCommand(exportCommand);
        rootCommand.AddCommand(visualizeCommand);
        
        return await rootCommand.InvokeAsync(args);
    }
    
    static void PlotUniqueId(FileInfo wdtFile, FileInfo outputFile, int width, int height, string? typeFilter)
    {
        Console.WriteLine($"=== WoW Data Plot - UniqueID Distribution ===");
        Console.WriteLine($"Input: {wdtFile.FullName}");
        Console.WriteLine($"Output: {outputFile.FullName}");
        Console.WriteLine($"Dimensions: {width}x{height}");
        
        try
        {
            // Extract placement data
            var progress = new Progress<string>(msg => Console.WriteLine($"[INFO] {msg}"));
            var records = AlphaPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            // Filter by type if requested
            if (!string.IsNullOrEmpty(typeFilter))
            {
                records = records.Where(r => r.Type.Equals(typeFilter, StringComparison.OrdinalIgnoreCase)).ToList();
                Console.WriteLine($"[INFO] Filtered to {records.Count} {typeFilter} placements");
            }
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found to plot!");
                return;
            }
            
            // Transform WoW world coordinates to plot coordinates
            // This creates a proper top-down view matching in-game orientation
            var plotCoords = records.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
            double[] xData = plotCoords.Select(c => c.plotX).ToArray();
            double[] yData = plotCoords.Select(c => c.plotY).ToArray();
            
            Console.WriteLine($"[INFO] Plotting {records.Count} points with WoW coordinate system...");
            Console.WriteLine($"[INFO] Coordinate system: Top=North, Right=East, Bottom=South, Left=West");
            
            // Create plot
            var plt = new Plot();
            
            // Add scatter plot (points only, no lines) with color gradient by UniqueID
            var scatter = plt.Add.Scatter(xData, yData);
            scatter.MarkerSize = 2;
            scatter.MarkerShape = MarkerShape.FilledCircle;
            scatter.LineWidth = 0; // NO LINES - points only!
            
            // Color by UniqueID range (gradient from blue to red)
            double minId = records.Min(r => r.UniqueId);
            double maxId = records.Max(r => r.UniqueId);
            
            Console.WriteLine($"[INFO] UniqueID range: {minId:F0} - {maxId:F0}");
            
            // Set axis labels and title
            plt.Title($"{Path.GetFileNameWithoutExtension(wdtFile.Name)} - {typeFilter ?? "All"} Placements by UniqueID");
            plt.XLabel("East ← → West (WoW Y-axis, flipped)");
            plt.YLabel("South ← → North (WoW X-axis, flipped)");
            
            // Equal aspect ratio for accurate spatial representation
            plt.Axes.SquareUnits();
            
            // Save to file
            Directory.CreateDirectory(Path.GetDirectoryName(outputFile.FullName) ?? ".");
            plt.SavePng(outputFile.FullName, width, height);
            
            Console.WriteLine($"[SUCCESS] Plot saved to: {outputFile.FullName}");
            Console.WriteLine($"[INFO] Total placements: {records.Count}");
            Console.WriteLine($"[INFO] M2 count: {records.Count(r => r.Type == "M2")}");
            Console.WriteLine($"[INFO] WMO count: {records.Count(r => r.Type == "WMO")}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[ERROR] {ex.InnerException.Message}");
            }
        }
    }
    
    static void ExportCsv(FileInfo wdtFile, FileInfo outputFile)
    {
        Console.WriteLine($"=== WoW Data Plot - CSV Export ===");
        Console.WriteLine($"Input: {wdtFile.FullName}");
        Console.WriteLine($"Output: {outputFile.FullName}");
        
        try
        {
            // Extract placement data
            var progress = new Progress<string>(msg => Console.WriteLine($"[INFO] {msg}"));
            var records = AlphaPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found to export!");
                return;
            }
            
            // Write to CSV
            Directory.CreateDirectory(Path.GetDirectoryName(outputFile.FullName) ?? ".");
            
            using var writer = new StreamWriter(outputFile.FullName);
            using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
            
            csv.WriteRecords(records);
            
            Console.WriteLine($"[SUCCESS] Exported {records.Count} records to: {outputFile.FullName}");
            Console.WriteLine($"[INFO] M2 count: {records.Count(r => r.Type == "M2")}");
            Console.WriteLine($"[INFO] WMO count: {records.Count(r => r.Type == "WMO")}");
            Console.WriteLine($"\n[INFO] CSV can now be analyzed with Python, R, MATLAB, Excel, etc.");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[ERROR] {ex.InnerException.Message}");
            }
        }
    }
    
    static void AnalyzeLayers(FileInfo wdtFile, FileInfo outputFile, int layerSize)
    {
        Console.WriteLine($"=== WoW Data Plot - Layer Analysis ===");
        Console.WriteLine($"Input: {wdtFile.FullName}");
        Console.WriteLine($"Output: {outputFile.FullName}");
        Console.WriteLine($"Layer size: {layerSize} UniqueIDs per layer");
        
        try
        {
            // Extract placement data
            var progress = new Progress<string>(msg => Console.WriteLine($"[INFO] {msg}"));
            var records = AlphaPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found to analyze!");
                return;
            }
            
            Console.WriteLine($"[INFO] Analyzing {records.Count} placements...");
            
            // Calculate global UniqueID range
            uint minId = records.Min(r => r.UniqueId);
            uint maxId = records.Max(r => r.UniqueId);
            
            Console.WriteLine($"[INFO] Global UniqueID range: {minId} - {maxId}");
            
            string[] colors = new[] { "#0000FF", "#00FF00", "#FF0000", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080" };
            
            // Analyze per-tile layers (tile-specific, not global!)
            var tileLayerInfos = new List<TileLayerInfo>();
            var globalLayersCollected = new HashSet<(uint min, uint max)>();
            
            var tilesWithData = records.Where(r => r.TileX >= 0 && r.TileY >= 0)
                                      .GroupBy(r => (r.TileX, r.TileY))
                                      .OrderBy(g => g.Key.TileY)
                                      .ThenBy(g => g.Key.TileX);
            
            Console.WriteLine($"[INFO] Analyzing {tilesWithData.Count()} tiles independently...");
            
            foreach (var tileGroup in tilesWithData)
            {
                var tileRecords = tileGroup.ToList();
                
                // Find unique ID ranges WITHIN THIS TILE
                var tileUniqueIds = tileRecords.Select(r => r.UniqueId).OrderBy(id => id).ToList();
                
                if (tileUniqueIds.Count == 0) continue;
                
                uint tileMin = tileUniqueIds.First();
                uint tileMax = tileUniqueIds.Last();
                
                // Auto-detect layers within this tile by finding gaps/clusters
                var tileLayers = DetectLayersInTile(tileRecords, layerSize, colors);
                
                // Track all unique ranges for global summary
                foreach (var layer in tileLayers)
                {
                    globalLayersCollected.Add((layer.MinUniqueId, layer.MaxUniqueId));
                }
                
                tileLayerInfos.Add(new TileLayerInfo
                {
                    TileX = tileGroup.Key.TileX,
                    TileY = tileGroup.Key.TileY,
                    TotalPlacements = tileRecords.Count,
                    Layers = tileLayers,
                    ImagePath = $"tile_{tileGroup.Key.TileY:D2}_{tileGroup.Key.TileX:D2}_layers.png"
                });
            }
            
            Console.WriteLine($"[INFO] Analyzed {tileLayerInfos.Count} tiles");
            
            // Create global layer summary from all tile-specific layers
            var globalLayers = globalLayersCollected
                .OrderBy(r => r.min)
                .Select((range, idx) => new LayerInfo
                {
                    Name = $"Layer {range.min}-{range.max}",
                    MinUniqueId = range.min,
                    MaxUniqueId = range.max,
                    PlacementCount = records.Count(r => r.UniqueId >= range.min && r.UniqueId <= range.max),
                    Color = colors[idx % colors.Length]
                })
                .ToList();
            
            Console.WriteLine($"[INFO] Found {globalLayers.Count} unique UniqueID ranges across all tiles");
            
            // Create analysis result
            var analysis = new WdtLayerAnalysis
            {
                WdtName = Path.GetFileNameWithoutExtension(wdtFile.Name),
                TotalPlacements = records.Count,
                MinUniqueId = minId,
                MaxUniqueId = maxId,
                GlobalLayers = globalLayers,
                Tiles = tileLayerInfos,
                AnalyzedAt = DateTime.UtcNow
            };
            
            // Write JSON
            Directory.CreateDirectory(Path.GetDirectoryName(outputFile.FullName) ?? ".");
            
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            
            string json = JsonSerializer.Serialize(analysis, options);
            File.WriteAllText(outputFile.FullName, json);
            
            Console.WriteLine($"[SUCCESS] Layer analysis saved to: {outputFile.FullName}");
            Console.WriteLine($"[INFO] Total layers: {globalLayers.Count}");
            Console.WriteLine($"[INFO] Tiles with data: {tileLayerInfos.Count}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[ERROR] {ex.InnerException.Message}");
            }
        }
    }
    
    /// <summary>
    /// Detect layers within a single tile by finding GAPS in UniqueID sequences.
    /// Gaps represent development pauses - they are meaningful data!
    /// Each continuous cluster of IDs = a layer (development burst).
    /// </summary>
    private static List<LayerInfo> DetectLayersInTile(List<PlacementRecord> tileRecords, int gapThreshold, string[] colors)
    {
        if (tileRecords.Count == 0)
            return new List<LayerInfo>();
        
        // Get all unique UniqueIDs in this tile, sorted
        var uniqueIds = tileRecords.Select(r => r.UniqueId).Distinct().OrderBy(id => id).ToList();
        
        if (uniqueIds.Count == 0)
            return new List<LayerInfo>();
        
        var layers = new List<LayerInfo>();
        uint layerStart = uniqueIds[0];
        uint layerEnd = uniqueIds[0];
        
        for (int i = 1; i < uniqueIds.Count; i++)
        {
            uint currentId = uniqueIds[i];
            uint previousId = uniqueIds[i - 1];
            uint gap = currentId - previousId;
            
            if (gap > gapThreshold)
            {
                // Found a GAP! Finalize current layer
                int count = tileRecords.Count(r => r.UniqueId >= layerStart && r.UniqueId <= layerEnd);
                
                layers.Add(new LayerInfo
                {
                    Name = $"{layerStart}-{layerEnd}",
                    MinUniqueId = layerStart,
                    MaxUniqueId = layerEnd,
                    PlacementCount = count,
                    Color = colors[layers.Count % colors.Length]
                });
                
                // Start new layer after gap
                layerStart = currentId;
                layerEnd = currentId;
            }
            else
            {
                // Continuous sequence - extend current layer
                layerEnd = currentId;
            }
        }
        
        // Finalize last layer
        if (layerStart <= layerEnd)
        {
            int count = tileRecords.Count(r => r.UniqueId >= layerStart && r.UniqueId <= layerEnd);
            
            layers.Add(new LayerInfo
            {
                Name = $"{layerStart}-{layerEnd}",
                MinUniqueId = layerStart,
                MaxUniqueId = layerEnd,
                PlacementCount = count,
                Color = colors[layers.Count % colors.Length]
            });
        }
        
        return layers;
    }
    
    static void GenerateTileLayers(FileInfo wdtFile, FileInfo layersJson, DirectoryInfo outputDir, int imageSize)
    {
        Console.WriteLine($"=== WoW Data Plot - Generate Tile Layers ===");
        Console.WriteLine($"Input WDT: {wdtFile.FullName}");
        Console.WriteLine($"Layers JSON: {layersJson.FullName}");
        Console.WriteLine($"Output dir: {outputDir.FullName}");
        Console.WriteLine($"Image size: {imageSize}x{imageSize}");
        
        try
        {
            // Load layer analysis
            string json = File.ReadAllText(layersJson.FullName);
            var options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            var analysis = JsonSerializer.Deserialize<WdtLayerAnalysis>(json, options);
            
            if (analysis == null)
            {
                Console.WriteLine("[ERROR] Failed to parse layer analysis JSON!");
                return;
            }
            
            Console.WriteLine($"[INFO] Loaded analysis: {analysis.GlobalLayers.Count} layers, {analysis.Tiles.Count} tiles");
            
            // Extract placement data
            var progress = new Progress<string>(msg => Console.WriteLine($"[INFO] {msg}"));
            var records = AlphaPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found!");
                return;
            }
            
            // Create output directory
            Directory.CreateDirectory(outputDir.FullName);
            
            // Generate image for each tile
            foreach (var tileInfo in analysis.Tiles)
            {
                var tileRecords = records.Where(r => r.TileX == tileInfo.TileX && r.TileY == tileInfo.TileY).ToList();
                
                if (tileRecords.Count == 0) continue;
                
                Console.WriteLine($"[INFO] Generating image for tile ({tileInfo.TileY:D2}, {tileInfo.TileX:D2}) - {tileRecords.Count} placements");
                
                var plt = new Plot();
                
                // Plot each layer with different color
                foreach (var layer in tileInfo.Layers)
                {
                    var layerRecords = tileRecords.Where(r => r.UniqueId >= layer.MinUniqueId && 
                                                              r.UniqueId <= layer.MaxUniqueId).ToList();
                    
                    if (layerRecords.Count == 0) continue;
                    
                    // Transform to plot coordinates (proper WoW coordinate system)
                    var plotCoords = layerRecords.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
                    double[] xData = plotCoords.Select(c => c.plotX).ToArray();
                    double[] yData = plotCoords.Select(c => c.plotY).ToArray();
                    
                    var scatter = plt.Add.Scatter(xData, yData);
                    scatter.MarkerSize = 3;
                    scatter.MarkerShape = MarkerShape.FilledCircle;
                    scatter.LineWidth = 0; // No lines
                    scatter.LegendText = $"{layer.Name} ({layerRecords.Count})";
                    
                    // Parse color
                    var color = System.Drawing.ColorTranslator.FromHtml(layer.Color);
                    scatter.Color = ScottPlot.Color.FromColor(color);
                }
                
                // Get tile center coordinates for title
                var (centerX, centerY) = CoordinateTransform.TileToWorldCenter(tileInfo.TileX, tileInfo.TileY);
                
                plt.Title($"Tile [{tileInfo.TileX},{tileInfo.TileY}] ({centerX:F0},{centerY:F0}) - {tileInfo.Layers.Count} Layers");
                plt.XLabel("East ← → West");
                plt.YLabel("South ← → North");
                plt.ShowLegend();
                plt.Axes.SquareUnits();
                
                string outputPath = Path.Combine(outputDir.FullName, tileInfo.ImagePath ?? $"tile_{tileInfo.TileY:D2}_{tileInfo.TileX:D2}.png");
                plt.SavePng(outputPath, imageSize, imageSize);
            }
            
            Console.WriteLine($"[SUCCESS] Generated {analysis.Tiles.Count} tile layer images in: {outputDir.FullName}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[ERROR] {ex.InnerException.Message}");
            }
        }
    }
    
    /// <summary>
    /// Try to find a minimap tile image for the given tile coordinates.
    /// Searches for common naming patterns used in WoW Alpha/Beta clients.
    /// </summary>
    private static string? TryFindMinimapTile(DirectoryInfo minimapDir, string mapName, int tileX, int tileY)
    {
        // Common minimap naming patterns in Alpha WoW:
        // - map<Y>_<X>.png (or .blp converted to .png)
        // - <MapName>_<Y>_<X>.png
        // - tile_<Y>_<X>.png
        
        var patterns = new[]
        {
            $"map{tileY}_{tileX}.png",
            $"map{tileY:D2}_{tileX:D2}.png",
            $"{mapName}_{tileY}_{tileX}.png",
            $"{mapName}_{tileY:D2}_{tileX:D2}.png",
            $"tile_{tileY}_{tileX}.png",
            $"tile_{tileY:D2}_{tileX:D2}.png",
            $"{tileY}_{tileX}.png",
            $"{tileY:D2}_{tileX:D2}.png",
        };
        
        foreach (var pattern in patterns)
        {
            var fullPath = Path.Combine(minimapDir.FullName, pattern);
            if (File.Exists(fullPath))
            {
                return fullPath;
            }
        }
        
        // Also check in subdirectory named after the map
        var mapSubDir = Path.Combine(minimapDir.FullName, mapName);
        if (Directory.Exists(mapSubDir))
        {
            foreach (var pattern in patterns)
            {
                var fullPath = Path.Combine(mapSubDir, pattern);
                if (File.Exists(fullPath))
                {
                    return fullPath;
                }
            }
        }
        
        return null;
    }
    
    static void VisualizeComplete(FileInfo wdtFile, DirectoryInfo outputDir, int gapThreshold, int tileSize, int mapSize, float tileMarkerSize, float mapMarkerSize, int mapMaxLayers)
    {
        Console.WriteLine($"=== WoW Data Plot - Complete Visualization Pipeline ===");
        Console.WriteLine($"Input: {wdtFile.FullName}");
        Console.WriteLine($"Output directory: {outputDir.FullName}");
        Console.WriteLine($"Gap threshold: {gapThreshold} (splits layers when UniqueID jumps > {gapThreshold})");
        Console.WriteLine($"Tile image size: {tileSize}x{tileSize}");
        Console.WriteLine($"Map image size: {mapSize}x{mapSize}");
        Console.WriteLine($"Tile marker size: {tileMarkerSize}");
        Console.WriteLine($"Map marker size: {mapMarkerSize}");
        Console.WriteLine($"Map max layers: {(mapMaxLayers == 0 ? "all" : mapMaxLayers.ToString())}");
        Console.WriteLine();
        
        try
        {
            // Create output directory structure
            Directory.CreateDirectory(outputDir.FullName);
            var tilesDir = Directory.CreateDirectory(Path.Combine(outputDir.FullName, "tiles"));
            
            var mapName = Path.GetFileNameWithoutExtension(wdtFile.Name);
            
            // STEP 1: Extract placement data
            Console.WriteLine("═══ STEP 1: Extracting Placement Data ═══");
            var progress = new Progress<string>(msg => Console.WriteLine($"  {msg}"));
            var records = AlphaPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found!");
                return;
            }
            
            Console.WriteLine($"✓ Extracted {records.Count} placements");
            Console.WriteLine();
            
            // STEP 2: Analyze layers per tile
            Console.WriteLine("═══ STEP 2: Analyzing Layers Per Tile ═══");
            
            string[] colors = new[] { "#0000FF", "#00FF00", "#FF0000", "#FFFF00", "#FF00FF", "#00FFFF", "#FFA500", "#800080" };
            
            var tileLayerInfos = new List<TileLayerInfo>();
            var globalLayersCollected = new HashSet<(uint min, uint max)>();
            
            var tilesWithData = records.Where(r => r.TileX >= 0 && r.TileY >= 0)
                                      .GroupBy(r => (r.TileX, r.TileY))
                                      .OrderBy(g => g.Key.TileY)
                                      .ThenBy(g => g.Key.TileX)
                                      .ToList();
            
            Console.WriteLine($"  Processing {tilesWithData.Count} tiles...");
            
            foreach (var tileGroup in tilesWithData)
            {
                var tileRecords = tileGroup.ToList();
                var tileLayers = DetectLayersInTile(tileRecords, gapThreshold, colors);
                
                foreach (var layer in tileLayers)
                {
                    globalLayersCollected.Add((layer.MinUniqueId, layer.MaxUniqueId));
                }
                
                tileLayerInfos.Add(new TileLayerInfo
                {
                    TileX = tileGroup.Key.TileX,
                    TileY = tileGroup.Key.TileY,
                    TotalPlacements = tileRecords.Count,
                    Layers = tileLayers,
                    ImagePath = $"tile_{tileGroup.Key.TileY:D2}_{tileGroup.Key.TileX:D2}.png"
                });
            }
            
            var globalLayers = globalLayersCollected
                .OrderBy(r => r.min)
                .Select((range, idx) => new LayerInfo
                {
                    Name = $"Layer {range.min}-{range.max}",
                    MinUniqueId = range.min,
                    MaxUniqueId = range.max,
                    PlacementCount = records.Count(r => r.UniqueId >= range.min && r.UniqueId <= range.max),
                    Color = colors[idx % colors.Length]
                })
                .ToList();
            
            Console.WriteLine($"✓ Detected {globalLayers.Count} unique UniqueID ranges across all tiles");
            Console.WriteLine();
            
            // STEP 3: Generate per-tile images
            Console.WriteLine("═══ STEP 3: Generating Per-Tile Images ═══");
            
            int tileCount = 0;
            
            foreach (var tileInfo in tileLayerInfos)
            {
                var tileRecords = records.Where(r => r.TileX == tileInfo.TileX && r.TileY == tileInfo.TileY).ToList();
                
                if (tileRecords.Count == 0) continue;
                
                var plt = new Plot();
                
                // Plot each layer with different color
                foreach (var layer in tileInfo.Layers)
                {
                    var layerRecords = tileRecords.Where(r => r.UniqueId >= layer.MinUniqueId && 
                                                              r.UniqueId <= layer.MaxUniqueId).ToList();
                    
                    if (layerRecords.Count == 0) continue;
                    
                    var plotCoords = layerRecords.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
                    double[] xData = plotCoords.Select(c => c.plotX).ToArray();
                    double[] yData = plotCoords.Select(c => c.plotY).ToArray();
                    
                    var scatter = plt.Add.Scatter(xData, yData);
                    scatter.MarkerSize = tileMarkerSize;  // Use configurable marker size
                    scatter.MarkerShape = MarkerShape.FilledCircle;
                    scatter.LineWidth = 0;
                    scatter.LegendText = $"{layer.Name} ({layerRecords.Count})";
                    
                    var color = System.Drawing.ColorTranslator.FromHtml(layer.Color);
                    scatter.Color = ScottPlot.Color.FromColor(color);
                }
                
                var (centerX, centerY) = CoordinateTransform.TileToWorldCenter(tileInfo.TileX, tileInfo.TileY);
                
                plt.Title($"[{tileInfo.TileX},{tileInfo.TileY}] ({centerX:F0},{centerY:F0}) - {tileInfo.Layers.Count} Layers");
                plt.XLabel("East ← → West");
                plt.YLabel("South ← → North");
                plt.ShowLegend();
                plt.Axes.SquareUnits();
                
                string outputPath = Path.Combine(tilesDir.FullName, tileInfo.ImagePath!);
                plt.SavePng(outputPath, tileSize, tileSize);
                
                tileCount++;
                if (tileCount % 50 == 0)
                {
                    Console.WriteLine($"  Generated {tileCount}/{tileLayerInfos.Count} tiles...");
                }
            }
            
            Console.WriteLine($"✓ Generated {tileCount} tile images");
            Console.WriteLine();
            
            // STEP 4: Generate map-wide overview
            Console.WriteLine("═══ STEP 4: Generating Map Overview ═══");
            
            var mapPlot = new Plot();
            
            // Plot ALL placements on map (not layer-by-layer due to high layer count)
            if (mapMaxLayers == 0 || globalLayers.Count <= mapMaxLayers)
            {
                // Plot all placements as a single dataset for performance
                Console.WriteLine($"  Plotting all {records.Count} placements...");
                
                var allPlotCoords = records.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
                double[] allXData = allPlotCoords.Select(c => c.plotX).ToArray();
                double[] allYData = allPlotCoords.Select(c => c.plotY).ToArray();
                
                var allScatter = mapPlot.Add.Scatter(allXData, allYData);
                allScatter.MarkerSize = mapMarkerSize;
                allScatter.MarkerShape = MarkerShape.FilledCircle;
                allScatter.LineWidth = 0;
                allScatter.LegendText = $"All Placements ({records.Count})";
                allScatter.Color = ScottPlot.Color.FromColor(System.Drawing.Color.Blue);
            }
            else
            {
                // Plot first N layers separately with different colors
                Console.WriteLine($"  Plotting first {mapMaxLayers} layers separately...");
                
                foreach (var globalLayer in globalLayers.Take(mapMaxLayers))
                {
                    var layerRecords = records.Where(r => r.UniqueId >= globalLayer.MinUniqueId && 
                                                          r.UniqueId <= globalLayer.MaxUniqueId).ToList();
                    
                    if (layerRecords.Count == 0) continue;
                    
                    var plotCoords = layerRecords.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
                    double[] xData = plotCoords.Select(c => c.plotX).ToArray();
                    double[] yData = plotCoords.Select(c => c.plotY).ToArray();
                    
                    var scatter = mapPlot.Add.Scatter(xData, yData);
                    scatter.MarkerSize = mapMarkerSize;
                    scatter.MarkerShape = MarkerShape.FilledCircle;
                    scatter.LineWidth = 0;
                    scatter.LegendText = $"{globalLayer.Name} ({layerRecords.Count})";
                    
                    var color = System.Drawing.ColorTranslator.FromHtml(globalLayer.Color);
                    scatter.Color = ScottPlot.Color.FromColor(color);
                }
            }
            
            mapPlot.Title($"{mapName} - Complete Distribution ({records.Count} placements)");
            mapPlot.XLabel("East ← → West");
            mapPlot.YLabel("South ← → North");
            if (mapMaxLayers > 0 && globalLayers.Count > mapMaxLayers)
            {
                mapPlot.ShowLegend();
            }
            mapPlot.Axes.SquareUnits();
            
            string mapPath = Path.Combine(outputDir.FullName, $"{mapName}_overview.png");
            mapPlot.SavePng(mapPath, mapSize, mapSize);
            
            Console.WriteLine($"✓ Generated map overview: {mapName}_overview.png");
            Console.WriteLine();
            
            // STEP 5: Save analysis JSON
            Console.WriteLine("═══ STEP 5: Saving Analysis Metadata ═══");
            
            var analysis = new WdtLayerAnalysis
            {
                WdtName = mapName,
                TotalPlacements = records.Count,
                MinUniqueId = records.Min(r => r.UniqueId),
                MaxUniqueId = records.Max(r => r.UniqueId),
                GlobalLayers = globalLayers,
                Tiles = tileLayerInfos,
                AnalyzedAt = DateTime.UtcNow
            };
            
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };
            
            string jsonPath = Path.Combine(outputDir.FullName, $"{mapName}_analysis.json");
            string json = JsonSerializer.Serialize(analysis, options);
            File.WriteAllText(jsonPath, json);
            
            Console.WriteLine($"✓ Saved analysis: {mapName}_analysis.json");
            Console.WriteLine();
            
            // STEP 6: Summary
            Console.WriteLine("═══════════════════════════════════════════");
            Console.WriteLine("✓ COMPLETE PIPELINE FINISHED");
            Console.WriteLine("═══════════════════════════════════════════");
            Console.WriteLine($"Output directory: {outputDir.FullName}");
            Console.WriteLine($"  ├─ {mapName}_overview.png       (map-wide visualization)");
            Console.WriteLine($"  ├─ {mapName}_analysis.json      (layer metadata)");
            Console.WriteLine($"  └─ tiles/                       ({tileCount} tile images)");
            Console.WriteLine();
            Console.WriteLine($"Total placements: {records.Count}");
            Console.WriteLine($"  M2:  {records.Count(r => r.Type == "M2")}");
            Console.WriteLine($"  WMO: {records.Count(r => r.Type == "WMO")}");
            Console.WriteLine($"Unique layers: {globalLayers.Count}");
            Console.WriteLine($"Tiles processed: {tileCount}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[ERROR] {ex.Message}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"[ERROR] {ex.InnerException.Message}");
            }
            Console.WriteLine(ex.StackTrace);
        }
    }
}
