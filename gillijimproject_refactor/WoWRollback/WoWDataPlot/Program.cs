using System.CommandLine;
using System.Text.Json;
using ScottPlot;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using GillijimProject.WowFiles.Alpha;
using WoWDataPlot.Helpers;
using WoWDataPlot.Extractors;
using WoWRollback.AnalysisModule;
using WoWDataPlot.Models;
using WoWRollback.Core.Services.Viewer;
using WoWFormatLib.FileReaders;
using SixLabors.ImageSharp.Formats.Png;

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
            
            // Analyze per-tile layers (tile-specific, not global!)
            var tileLayerInfos = new List<TileLayerInfo>();
            var globalLayersCollected = new HashSet<(uint min, uint max)>();
            
            var tilesWithData = records.Where(r => r.TileX >= 0 && r.TileY >= 0)
                                      .GroupBy(r => (r.TileX, r.TileY))
                                      .OrderBy(g => g.Key.TileY)
                                      .ThenBy(g => g.Key.TileX);
            
            Console.WriteLine($"  Processing {tilesWithData.Count()} tiles...");
            
            // Calculate GLOBAL UniqueID range for consistent coloring
            uint globalMinId = records.Min(r => r.UniqueId);
            uint globalMaxId = records.Max(r => r.UniqueId);
            
            foreach (var tileGroup in tilesWithData)
            {
                var tileRecords = tileGroup.ToList();
                
                // Find unique ID ranges WITHIN THIS TILE
                var tileUniqueIds = tileRecords.Select(r => r.UniqueId).OrderBy(id => id).ToList();
                
                if (tileUniqueIds.Count == 0) continue;
                
                uint tileMin = tileUniqueIds.First();
                uint tileMax = tileUniqueIds.Last();
                
                // Generate unique colors for this tile's potential layers (estimate max 100 per tile)
                var tileColors = GenerateColorPalette(100)
                    .Select(c => $"#{(byte)(c.Red * 255):X2}{(byte)(c.Green * 255):X2}{(byte)(c.Blue * 255):X2}")
                    .ToArray();
                
                // Auto-detect layers within this tile by finding gaps/clusters
                var tileLayers = DetectLayersInTile(tileRecords, layerSize, tileColors);
                
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
            var globalLayersList = globalLayersCollected.OrderBy(r => r.min).ToList();
            var globalColors = GenerateColorPalette(globalLayersList.Count)
                .Select(c => $"#{(byte)(c.Red * 255):X2}{(byte)(c.Green * 255):X2}{(byte)(c.Blue * 255):X2}")
                .ToArray();
                
            var globalLayers = globalLayersList
                .Select((range, idx) => new WoWDataPlot.Models.LayerInfo
                {
                    Name = $"{range.min}-{range.max}",
                    MinUniqueId = range.min,
                    MaxUniqueId = range.max,
                    PlacementCount = records.Count(r => r.UniqueId >= range.min && r.UniqueId <= range.max),
                    Color = globalColors[idx]
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
    /// Generate an HTML page for a single tile with its layers and legend.
    /// </summary>
    private static void GenerateTileHtmlPage(string htmlPath, string mapName, TileLayerInfo tileInfo, string imageName, string? minimapDir, uint globalMinId, uint globalMaxId)
    {
        using var writer = new StreamWriter(htmlPath);
        
        // Check if minimap exists for this tile
        string? minimapFile = null;
        if (!string.IsNullOrEmpty(minimapDir))
        {
            var minimapPath = Path.Combine(minimapDir, $"{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.png");
            if (File.Exists(minimapPath))
            {
                minimapFile = Path.GetFileName(minimapPath);
            }
        }
        
        writer.WriteLine("<!DOCTYPE html>");
        writer.WriteLine("<html>");
        writer.WriteLine("<head>");
        writer.WriteLine($"    <title>{mapName} - Tile [{tileInfo.TileX}, {tileInfo.TileY}]</title>");
        writer.WriteLine("    <style>");
        writer.WriteLine("        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #d4d4d4; }");
        writer.WriteLine("        h1 { color: #4ec9b0; }");
        writer.WriteLine("        .back-link { color: #569cd6; text-decoration: none; margin-bottom: 20px; display: inline-block; }");
        writer.WriteLine("        .back-link:hover { text-decoration: underline; }");
        writer.WriteLine("        .tile-info { background: #252526; padding: 15px; border-radius: 5px; margin: 20px 0; }");
        writer.WriteLine("        .tile-images { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }");
        writer.WriteLine("        .tile-image { text-align: center; }");
        writer.WriteLine("        .tile-image h3 { color: #569cd6; margin-bottom: 10px; }");
        writer.WriteLine("        .tile-image img { max-width: 100%; border: 2px solid #3c3c3c; border-radius: 5px; }");
        writer.WriteLine("        .single-image { grid-column: 1 / -1; }");
        writer.WriteLine("        .layer-list { background: #252526; padding: 15px; border-radius: 5px; }");
        writer.WriteLine("        .layer-item { display: flex; align-items: center; padding: 8px; margin: 5px 0; background: #2d2d30; border-radius: 3px; }");
        writer.WriteLine("        .color-box { width: 30px; height: 20px; margin-right: 10px; border: 1px solid #3c3c3c; }");
        writer.WriteLine("        .layer-info-text { color: #d4d4d4; }");
        writer.WriteLine("    </style>");
        writer.WriteLine("</head>");
        writer.WriteLine("<body>");
        writer.WriteLine($"    <a href='../{mapName}_legend.html' class='back-link'>← Back to Overview</a>");
        writer.WriteLine($"    <h1>Tile [{tileInfo.TileX}, {tileInfo.TileY}] - {tileInfo.Layers.Count} Layers</h1>");
        
        writer.WriteLine("    <div class='tile-info'>");
        writer.WriteLine($"        <strong>Total Placements:</strong> {tileInfo.Layers.Sum(l => l.PlacementCount)}<br>");
        writer.WriteLine($"        <strong>Layers:</strong> {tileInfo.Layers.Count}<br>");
        writer.WriteLine("        <strong>Marker Shapes:</strong> ● = M2 Models | ■ = WMO Buildings");
        writer.WriteLine("    </div>");
        
        // Interactive layer viewer with minimap base + toggleable overlays
        var overlayDir = $"overlays_{tileInfo.TileX}_{tileInfo.TileY}";
        var overlayPath = Path.Combine(Path.GetDirectoryName(htmlPath)!, overlayDir);
        var hasOverlays = Directory.Exists(overlayPath);
        
        if (minimapFile != null && hasOverlays)
        {
            writer.WriteLine("    <h3 style='color: #4ec9b0;'>Interactive Layer Viewer</h3>");
            writer.WriteLine("    <div style='display: grid; grid-template-columns: 300px 1fr; gap: 20px; margin: 20px 0;'>");
            
            // Layer controls sidebar
            writer.WriteLine("        <div style='background: #252526; padding: 15px; border-radius: 5px; max-height: 800px; overflow-y: auto;'>");
            writer.WriteLine("            <h4 style='color: #569cd6; margin-top: 0;'>Toggle Layers</h4>");
            writer.WriteLine("            <div style='margin-bottom: 10px;'>");
            writer.WriteLine("                <button onclick='toggleAll(true)' style='padding: 5px 10px; margin-right: 5px; background: #4ec9b0; border: none; border-radius: 3px; color: #1e1e1e; cursor: pointer;'>All On</button>");
            writer.WriteLine("                <button onclick='toggleAll(false)' style='padding: 5px 10px; background: #ce9178; border: none; border-radius: 3px; color: #1e1e1e; cursor: pointer;'>All Off</button>");
            writer.WriteLine("            </div>");
            writer.WriteLine("            <div style='margin: 15px 0; padding: 10px; background: #2d2d30; border-radius: 5px;'>");
            writer.WriteLine("                <h5 style='color: #ce9178; margin-top: 0;'>🔧 Debug Transforms</h5>");
            writer.WriteLine("                <label style='display: block; margin: 5px 0; cursor: pointer;'>");
            writer.WriteLine("                    <input type='checkbox' id='flipX' onchange='updateTransform()' style='margin-right: 5px;'>");
            writer.WriteLine("                    Flip X (horizontal mirror)");
            writer.WriteLine("                </label>");
            writer.WriteLine("                <label style='display: block; margin: 5px 0; cursor: pointer;'>");
            writer.WriteLine("                    <input type='checkbox' id='flipY' onchange='updateTransform()' style='margin-right: 5px;'>");
            writer.WriteLine("                    Flip Y (vertical mirror)");
            writer.WriteLine("                </label>");
            writer.WriteLine("                <label style='display: block; margin: 5px 0; cursor: pointer;'>");
            writer.WriteLine("                    <input type='checkbox' id='swapXY' onchange='updateTransform()' style='margin-right: 5px;'>");
            writer.WriteLine("                    Swap X↔Y");
            writer.WriteLine("                </label>");
            writer.WriteLine("                <button onclick='resetTransform()' style='margin-top: 5px; padding: 3px 8px; background: #555; border: none; border-radius: 3px; color: #ddd; cursor: pointer; font-size: 0.9em;'>Reset</button>");
            writer.WriteLine("            </div>");
            
            foreach (var (layer, idx) in tileInfo.Layers.Select((l, i) => (l, i)))
            {
                // Calculate ACTUAL color based on global gradient (average UniqueID of this layer)
                uint layerMidId = (layer.MinUniqueId + layer.MaxUniqueId) / 2;
                float normalizedPos = (layerMidId - globalMinId) / (float)(Math.Max(globalMaxId - globalMinId, 1));
                float hue = (1.0f - normalizedPos) * 240f; // Blue (early) -> Red (late)
                var layerColor = ColorFromHSV(hue, 0.85f, 0.95f);
                string htmlColor = $"#{layerColor.R:X2}{layerColor.G:X2}{layerColor.B:X2}";
                
                writer.WriteLine($"            <div style='margin: 8px 0; padding: 8px; background: #2d2d30; border-radius: 3px;'>");
                writer.WriteLine($"                <label style='display: flex; align-items: center; cursor: pointer;'>");
                writer.WriteLine($"                    <input type='checkbox' id='layer{idx}' onchange='toggleLayer({idx})' checked style='margin-right: 8px;'>");
                writer.WriteLine($"                    <div style='width: 20px; height: 20px; background: {htmlColor}; margin-right: 8px; border: 1px solid #3c3c3c; border-radius: 2px;'></div>");
                writer.WriteLine($"                    <div style='font-size: 0.85em;'>");
                writer.WriteLine($"                        <div style='color: #4ec9b0; font-weight: bold;'>{layer.Name}</div>");
                writer.WriteLine($"                        <div style='color: #858585; font-size: 0.9em;'>{layer.PlacementCount} items</div>");
                writer.WriteLine($"                    </div>");
                writer.WriteLine($"                </label>");
                writer.WriteLine($"            </div>");
            }
            
            writer.WriteLine("        </div>");
            
            // Canvas viewer
            writer.WriteLine("        <div style='position: relative; max-width: 1024px;'>");
            writer.WriteLine("            <canvas id='viewer' width='1024' height='1024' style='border: 2px solid #3c3c3c; border-radius: 5px; width: 100%; height: auto;'></canvas>");
            writer.WriteLine("        </div>");
            
            writer.WriteLine("    </div>");
            
            // JavaScript for layer management
            writer.WriteLine("    <script>");
            writer.WriteLine("        const canvas = document.getElementById('viewer');");
            writer.WriteLine("        const ctx = canvas.getContext('2d');");
            writer.WriteLine("        const baseImg = new Image();");
            writer.WriteLine($"        baseImg.src = '../minimaps/{minimapFile}';");
            writer.WriteLine("        const layerImgs = [];");
            writer.WriteLine("        const layerVisible = [];");
            
            foreach (var (layer, idx) in tileInfo.Layers.Select((l, i) => (l, i)))
            {
                writer.WriteLine($"        layerImgs[{idx}] = new Image();");
                writer.WriteLine($"        layerImgs[{idx}].src = '{overlayDir}/layer_{layer.MinUniqueId}_{layer.MaxUniqueId}.png';");
                writer.WriteLine($"        layerVisible[{idx}] = true;");
            }
            
            writer.WriteLine("        let transformState = { flipX: false, flipY: false, swapXY: false };");
            writer.WriteLine("        function redraw() {");
            writer.WriteLine("            ctx.clearRect(0, 0, canvas.width, canvas.height);");
            writer.WriteLine("            // Draw minimap base - always static, no transform");
            writer.WriteLine("            ctx.drawImage(baseImg, 0, 0, canvas.width, canvas.height);");
            writer.WriteLine("            // Draw layers with transform applied");
            writer.WriteLine("            for (let i = 0; i < layerImgs.length; i++) {");
            writer.WriteLine("                if (layerVisible[i] && layerImgs[i].complete) {");
            writer.WriteLine("                    ctx.save();");
            writer.WriteLine("                    ctx.translate(canvas.width / 2, canvas.height / 2);");
            writer.WriteLine("                    if (transformState.swapXY) {");
            writer.WriteLine("                        ctx.rotate(Math.PI / 2);");
            writer.WriteLine("                        ctx.scale(1, -1);");
            writer.WriteLine("                    }");
            writer.WriteLine("                    ctx.scale(transformState.flipX ? -1 : 1, transformState.flipY ? -1 : 1);");
            writer.WriteLine("                    ctx.translate(-canvas.width / 2, -canvas.height / 2);");
            writer.WriteLine("                    ctx.drawImage(layerImgs[i], 0, 0, canvas.width, canvas.height);");
            writer.WriteLine("                    ctx.restore();");
            writer.WriteLine("                }");
            writer.WriteLine("            }");
            writer.WriteLine("        }");
            writer.WriteLine("        function toggleLayer(idx) {");
            writer.WriteLine("            layerVisible[idx] = document.getElementById('layer' + idx).checked;");
            writer.WriteLine("            redraw();");
            writer.WriteLine("        }");
            writer.WriteLine("        function toggleAll(visible) {");
            writer.WriteLine("            for (let i = 0; i < layerVisible.length; i++) {");
            writer.WriteLine("                layerVisible[i] = visible;");
            writer.WriteLine("                document.getElementById('layer' + i).checked = visible;");
            writer.WriteLine("            }");
            writer.WriteLine("            redraw();");
            writer.WriteLine("        }");
            writer.WriteLine("        function updateTransform() {");
            writer.WriteLine("            transformState.flipX = document.getElementById('flipX').checked;");
            writer.WriteLine("            transformState.flipY = document.getElementById('flipY').checked;");
            writer.WriteLine("            transformState.swapXY = document.getElementById('swapXY').checked;");
            writer.WriteLine("            redraw();");
            writer.WriteLine("        }");
            writer.WriteLine("        function resetTransform() {");
            writer.WriteLine("            document.getElementById('flipX').checked = false;");
            writer.WriteLine("            document.getElementById('flipY').checked = false;");
            writer.WriteLine("            document.getElementById('swapXY').checked = false;");
            writer.WriteLine("            transformState = { flipX: false, flipY: false, swapXY: false };");
            writer.WriteLine("            redraw();");
            writer.WriteLine("        }");
            writer.WriteLine("        baseImg.onload = redraw;");
            writer.WriteLine("        layerImgs.forEach(img => img.onload = redraw);");
            writer.WriteLine("    </script>");
        }
        else
        {
            // Fallback to static image if no minimap/overlays
            writer.WriteLine("    <div class='tile-images'>");
            writer.WriteLine("        <div class='tile-image single-image'>");
            writer.WriteLine("            <h3>Placement Heatmap</h3>");
            writer.WriteLine($"            <img src='{imageName}' alt='Placements [{tileInfo.TileX}, {tileInfo.TileY}]' />");
            writer.WriteLine("        </div>");
            writer.WriteLine("    </div>");
        }
        
        writer.WriteLine("    <h2>Layers in This Tile ({0} local layers detected)</h2>", tileInfo.Layers.Count);
        writer.WriteLine("    <p style='color: #858585; font-size: 0.9em; margin-bottom: 10px;'>These layers were detected in this specific tile based on UniqueID gaps. Use the toggles above to show/hide individual layers.</p>");
        writer.WriteLine("    <div class='layer-list'>");
        
        foreach (var (layer, idx) in tileInfo.Layers.Select((l, i) => (l, i)))
        {
            // Calculate global color for this layer
            uint layerMidId = (layer.MinUniqueId + layer.MaxUniqueId) / 2;
            float normalizedPos = (layerMidId - globalMinId) / (float)Math.Max(globalMaxId - globalMinId, 1);
            float hue = (1.0f - normalizedPos) * 240f;
            var globalColor = ColorFromHSV(hue, 0.85f, 0.95f);
            string htmlColor = $"#{globalColor.R:X2}{globalColor.G:X2}{globalColor.B:X2}";
            
            var m2Count = layer.PlacementCount; // This is already the tile-specific count
            var layerType = layer.MinUniqueId == layer.MaxUniqueId ? "Single object" : "Range";
            
            writer.WriteLine("        <div class='layer-item'>");
            writer.WriteLine($"            <div class='color-box' style='background-color: {htmlColor};'></div>");
            writer.WriteLine($"            <div class='layer-info-text'>");
            writer.WriteLine($"                <strong>Layer {idx + 1}:</strong> UniqueID {layer.MinUniqueId:N0} - {layer.MaxUniqueId:N0} ({m2Count} placements)");
            writer.WriteLine($"            </div>");
            writer.WriteLine("        </div>");
        }
        
        writer.WriteLine("    </div>");
        writer.WriteLine("</body>");
        writer.WriteLine("</html>");
    }
    
    /// <summary>
    /// Generate a diverse color palette for visualizing many layers.
    /// Uses HSV color space to distribute hues evenly.
    /// </summary>
    private static ScottPlot.Color[] GenerateColorPalette(int count)
    {
        var colors = new ScottPlot.Color[count];
        
        for (int i = 0; i < count; i++)
        {
            // Distribute hues evenly across the color wheel
            float hue = (float)i / count * 360f;
            
            // Vary saturation and value to create more distinct colors
            float saturation = 0.7f + ((i % 3) * 0.1f);  // 0.7, 0.8, 0.9
            float value = 0.8f + ((i % 2) * 0.2f);       // 0.8, 1.0
            
            var color = ColorFromHSV(hue, saturation, value);
            colors[i] = ScottPlot.Color.FromColor(color);
        }
        
        return colors;
    }
    
    /// <summary>
    /// Convert HSV to RGB color.
    /// </summary>
    private static System.Drawing.Color ColorFromHSV(float hue, float saturation, float value)
    {
        // Clamp inputs to valid ranges
        hue = Math.Clamp(hue, 0, 360);
        saturation = Math.Clamp(saturation, 0, 1);
        value = Math.Clamp(value, 0, 1);
        
        int hi = (int)Math.Floor(hue / 60) % 6;
        float f = hue / 60 - (float)Math.Floor(hue / 60);

        value = value * 255;
        int v = (int)Math.Clamp(value, 0, 255);
        int p = (int)Math.Clamp(value * (1 - saturation), 0, 255);
        int q = (int)Math.Clamp(value * (1 - f * saturation), 0, 255);
        int t = (int)Math.Clamp(value * (1 - (1 - f) * saturation), 0, 255);

        if (hi == 0)
            return System.Drawing.Color.FromArgb(255, v, t, p);
        else if (hi == 1)
            return System.Drawing.Color.FromArgb(255, q, v, p);
        else if (hi == 2)
            return System.Drawing.Color.FromArgb(255, p, v, t);
        else if (hi == 3)
            return System.Drawing.Color.FromArgb(255, p, q, v);
        else if (hi == 4)
            return System.Drawing.Color.FromArgb(255, t, p, v);
        else
            return System.Drawing.Color.FromArgb(255, v, p, q);
    }
    
    /// <summary>
    /// Detect layers within a single tile by finding GAPS in UniqueID sequences.
    /// Gaps represent development pauses - they are meaningful data!
    /// Each continuous cluster of IDs = a layer (development burst).
    /// </summary>
    private static List<WoWDataPlot.Models.LayerInfo> DetectLayersInTile(List<PlacementRecord> tileRecords, int gapThreshold, string[] colors)
    {
        if (tileRecords.Count == 0)
            return new List<WoWDataPlot.Models.LayerInfo>();
        
        // Get all unique UniqueIDs in this tile, sorted
        var uniqueIds = tileRecords.Select(r => r.UniqueId).Distinct().OrderBy(id => id).ToList();
        
        if (uniqueIds.Count == 0)
            return new List<WoWDataPlot.Models.LayerInfo>();
        
        var layers = new List<WoWDataPlot.Models.LayerInfo>();
        uint layerStart = uniqueIds[0];
        uint layerEnd = uniqueIds[0];
        
        for (int i = 1; i < uniqueIds.Count; i++)
        {
            uint currentId = uniqueIds[i];
            uint previousId = uniqueIds[i - 1];
            uint gap = currentId - previousId;
            
            if (gap > gapThreshold)
            {
                // Gap detected - finalize current layer
                int placementCount = tileRecords.Count(r => r.UniqueId >= layerStart && r.UniqueId <= layerEnd);
                string color = colors[layers.Count % colors.Length];
                
                layers.Add(new WoWDataPlot.Models.LayerInfo
                {
                    Name = $"{layerStart}-{layerEnd}",
                    MinUniqueId = layerStart,
                    MaxUniqueId = layerEnd,
                    PlacementCount = placementCount,
                    Color = color
                });
                
                // Start new layer
                layerStart = currentId;
            }
            
            layerEnd = currentId;
        }
        
        // Add final layer
        {
            int placementCount = tileRecords.Count(r => r.UniqueId >= layerStart && r.UniqueId <= layerEnd);
            string color = colors[layers.Count % colors.Length];
            
            layers.Add(new WoWDataPlot.Models.LayerInfo
            {
                Name = $"{layerStart}-{layerEnd}",
                MinUniqueId = layerStart,
                MaxUniqueId = layerEnd,
                PlacementCount = placementCount,
                Color = color
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
    
    static async Task VisualizeComplete(FileInfo wdtFile, DirectoryInfo outputDir, int gapThreshold, int tileSize, int mapSize, float tileMarkerSize, float mapMarkerSize, int mapMaxLayers)
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
            
            // Process minimaps using existing provider system
            Console.WriteLine("═══ STEP 0: Processing Minimap Tiles ═══");
            
            string? minimapDir = null;
            int tilesCopied = 0;
            
            try
            {
                var minimapOutputDir = Path.Combine(outputDir.FullName, "minimaps");
                Directory.CreateDirectory(minimapOutputDir);
                
                // Use LooseFileMinimapProvider to extract minimaps
                // WDT: test_data/0.5.3/tree/World/Maps/Azeroth/Azeroth.wdt
                // Need: test_data as root (6 levels up from WDT file)
                var wdtPath = wdtFile.FullName;
                var testDataRoot = Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetDirectoryName(wdtPath)))))); // Go up 6 levels to test_data
                var versions = new List<string> { "0.5.3" };
                
                Console.WriteLine($"  Test data root: {testDataRoot}");
                
                var provider = WoWRollback.Core.Services.Viewer.LooseFileMinimapProvider.Build(testDataRoot ?? "", versions);
                var composer = new WoWRollback.Core.Services.Viewer.MinimapComposer();
                var viewerOptions = WoWRollback.Core.Services.Viewer.ViewerOptions.CreateDefault();
                
                // Get all tiles for this map
                var availableTiles = provider.EnumerateTiles("0.5.3", mapName).ToList();
                
                if (availableTiles.Count > 0)
                {
                    Console.WriteLine($"  Found {availableTiles.Count} minimap tiles");
                    
                    foreach (var (tileX, tileY) in availableTiles)
                    {
                        var stream = await provider.OpenTileAsync("0.5.3", mapName, tileX, tileY);
                        if (stream != null)
                        {
                            using (stream)
                            {
                                var outputPath = Path.Combine(minimapOutputDir, $"{mapName}_{tileX}_{tileY}.png");
                                await composer.ComposeAsync(stream, outputPath, viewerOptions, CancellationToken.None);
                                tilesCopied++;
                            }
                        }
                    }
                    
                    if (tilesCopied > 0)
                    {
                        minimapDir = minimapOutputDir;
                        Console.WriteLine($"✓ Converted {tilesCopied} BLP minimap tiles to PNG");
                    }
                }
                else
                {
                    Console.WriteLine($"⚠ No minimap tiles found (optional - continuing without them)");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠ Minimap processing failed: {ex.Message}");
                Console.WriteLine($"  Continuing without minimaps");
            }
            
            var minimapResult = new MinimapResult(
                Success: tilesCopied > 0,
                TilesCopied: tilesCopied,
                MinimapDir: minimapDir,
                ErrorMessage: null);
            Console.WriteLine();
            
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
                
                // Generate unique colors for this tile's layers (max 100 per tile)
                var tileColors = GenerateColorPalette(100)
                    .Select(c => $"#{(byte)(c.Red * 255):X2}{(byte)(c.Green * 255):X2}{(byte)(c.Blue * 255):X2}")
                    .ToArray();
                    
                var tileLayers = DetectLayersInTile(tileRecords, gapThreshold, tileColors);
                
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
            
            var globalLayersList2 = globalLayersCollected.OrderBy(r => r.min).ToList();
            var globalColors2 = GenerateColorPalette(globalLayersList2.Count)
                .Select(c => $"#{(byte)(c.Red * 255):X2}{(byte)(c.Green * 255):X2}{(byte)(c.Blue * 255):X2}")
                .ToArray();
                
            var globalLayers = globalLayersList2
                .Select((range, idx) => new WoWDataPlot.Models.LayerInfo
                {
                    Name = $"{range.min}-{range.max}",
                    MinUniqueId = range.min,
                    MaxUniqueId = range.max,
                    PlacementCount = records.Count(r => r.UniqueId >= range.min && r.UniqueId <= range.max),
                    Color = globalColors2[idx]
                })
                .ToList();
            
            Console.WriteLine($"✓ Detected {globalLayers.Count} unique UniqueID ranges across all tiles");
            Console.WriteLine();
            
            // STEP 3: Generate per-tile images
            Console.WriteLine("═══ STEP 3: Generating Per-Tile Images ═══");
            
            // Use GLOBAL UniqueID range for consistent colors across all tiles
            uint visualGlobalMinId = records.Min(r => r.UniqueId);
            uint visualGlobalMaxId = records.Max(r => r.UniqueId);
            uint visualGlobalRange = Math.Max(visualGlobalMaxId - visualGlobalMinId, 1);
            
            int tileCount = 0;
            
            foreach (var tileInfo in tileLayerInfos)
            {
                var tileRecords = records.Where(r => r.TileX == tileInfo.TileX && r.TileY == tileInfo.TileY).ToList();
                
                if (tileRecords.Count == 0) continue;
                
                var plt = new Plot();
                
                uint tileMinId = tileRecords.Min(r => r.UniqueId);
                uint tileMaxId = tileRecords.Max(r => r.UniqueId);
                
                // HEATMAP: Color each placement based on GLOBAL UniqueID position
                // Group by type for different marker shapes
                var m2Records = tileRecords.Where(r => r.Type == "M2").ToList();
                var wmoRecords = tileRecords.Where(r => r.Type == "WMO").ToList();
                
                // Plot M2 placements (circles)
                foreach (var record in m2Records)
                {
                    float normalizedPos = (record.UniqueId - visualGlobalMinId) / (float)visualGlobalRange;
                    float hue = (1.0f - normalizedPos) * 240f;
                    var pointColor = ColorFromHSV(hue, 0.85f, 0.95f);
                    
                    var plotCoord = CoordinateTransform.WorldToPlot(record.X, record.Y);
                    var scatter = plt.Add.Scatter(new[] { plotCoord.plotX }, new[] { plotCoord.plotY });
                    scatter.MarkerSize = tileMarkerSize;
                    scatter.MarkerShape = MarkerShape.FilledCircle; // M2 = Circle
                    scatter.LineWidth = 0;
                    scatter.Color = ScottPlot.Color.FromColor(pointColor);
                }
                
                // Plot WMO placements (squares)
                foreach (var record in wmoRecords)
                {
                    float normalizedPos = (record.UniqueId - visualGlobalMinId) / (float)visualGlobalRange;
                    float hue = (1.0f - normalizedPos) * 240f;
                    var pointColor = ColorFromHSV(hue, 0.85f, 0.95f);
                    
                    var plotCoord = CoordinateTransform.WorldToPlot(record.X, record.Y);
                    var scatter = plt.Add.Scatter(new[] { plotCoord.plotX }, new[] { plotCoord.plotY });
                    scatter.MarkerSize = tileMarkerSize * 1.2f; // Slightly larger for visibility
                    scatter.MarkerShape = MarkerShape.FilledSquare; // WMO = Square
                    scatter.LineWidth = 0;
                    scatter.Color = ScottPlot.Color.FromColor(pointColor);
                }
                
                var (centerX, centerY) = CoordinateTransform.TileToWorldCenter(tileInfo.TileX, tileInfo.TileY);
                
                plt.Title($"[{tileInfo.TileX},{tileInfo.TileY}] ({centerX:F0},{centerY:F0}) - Global UniqueID {visualGlobalMinId}-{visualGlobalMaxId}");
                plt.XLabel("East ← → West");
                plt.YLabel("South ← → North");
                plt.Axes.SquareUnits();
                
                string outputPath = Path.Combine(tilesDir.FullName, tileInfo.ImagePath!);
                plt.SavePng(outputPath, tileSize, tileSize);
                
                // Generate layer overlays BEFORE HTML (if minimap exists for this tile)
                if (!string.IsNullOrEmpty(minimapResult.MinimapDir))
                {
                    var minimapPath = Path.Combine(minimapResult.MinimapDir, $"{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.png");
                    if (File.Exists(minimapPath))
                    {
                        // Create overlay directory for this tile
                        var overlayDir = Path.Combine(tilesDir.FullName, $"overlays_{tileInfo.TileX}_{tileInfo.TileY}");
                        Directory.CreateDirectory(overlayDir);
                        
                        Console.WriteLine($"[OVERLAY] Generating overlays for tile [{tileInfo.TileX},{tileInfo.TileY}] with {tileInfo.Layers.Count} layers, {tileRecords.Count} records");
                        
                        // Generate one transparent PNG per layer with GLOBAL coloring
                        foreach (var layer in tileInfo.Layers)
                        {
                            var layerRecords = tileRecords.Where(r => r.UniqueId >= layer.MinUniqueId && 
                                                                      r.UniqueId <= layer.MaxUniqueId).ToList();
                            if (layerRecords.Count == 0) continue;
                            
                            var layerPlt = new Plot();
                            layerPlt.Layout.Frameless();
                            
                            // Make background TRANSPARENT so minimap shows through
                            layerPlt.FigureBackground.Color = ScottPlot.Colors.Transparent;
                            layerPlt.DataBackground.Color = ScottPlot.Colors.Transparent;
                            
                            // Color each point based on GLOBAL UniqueID gradient with type-specific shapes
                            var m2LayerRecords = layerRecords.Where(r => r.Type == "M2").ToList();
                            var wmoLayerRecords = layerRecords.Where(r => r.Type == "WMO").ToList();
                            
                            Console.WriteLine($"[OVERLAY] Layer {layer.Name}: {m2LayerRecords.Count} M2 + {wmoLayerRecords.Count} WMO");
                            
                            // Sample coordinate debug for first record
                            if (m2LayerRecords.Count > 0)
                            {
                                var sample = m2LayerRecords[0];
                                var samplePixel = CoordinateTransform.WorldToTilePixel(sample.X, sample.Y, tileSize, tileSize);
                                Console.WriteLine($"[OVERLAY]   Sample M2: world({sample.X:F1}, {sample.Y:F1}) -> pixel({samplePixel.pixelX:F1}, {samplePixel.pixelY:F1})");
                            }
                            
                            // Plot M2 (circles) - use EXACT pixel coordinates for 1:1 alignment
                            foreach (var record in m2LayerRecords)
                            {
                                float normalizedPos = (record.UniqueId - visualGlobalMinId) / (float)visualGlobalRange;
                                float hue = (1.0f - normalizedPos) * 240f;
                                var pointColor = ColorFromHSV(hue, 0.85f, 0.95f);
                                
                                var pixelCoord = CoordinateTransform.WorldToTilePixel(record.X, record.Y, tileSize, tileSize);
                                var scatter = layerPlt.Add.Scatter(new[] { pixelCoord.pixelX }, new[] { pixelCoord.pixelY });
                                scatter.MarkerSize = tileMarkerSize * 1.5f;
                                scatter.MarkerShape = MarkerShape.FilledCircle;
                                scatter.LineWidth = 0;
                                scatter.Color = ScottPlot.Color.FromColor(pointColor);
                            }
                            
                            // Plot WMO (squares) - use EXACT pixel coordinates for 1:1 alignment
                            foreach (var record in wmoLayerRecords)
                            {
                                float normalizedPos = (record.UniqueId - visualGlobalMinId) / (float)visualGlobalRange;
                                float hue = (1.0f - normalizedPos) * 240f;
                                var pointColor = ColorFromHSV(hue, 0.85f, 0.95f);
                                
                                var pixelCoord = CoordinateTransform.WorldToTilePixel(record.X, record.Y, tileSize, tileSize);
                                var scatter = layerPlt.Add.Scatter(new[] { pixelCoord.pixelX }, new[] { pixelCoord.pixelY });
                                scatter.MarkerSize = tileMarkerSize * 1.8f;
                                scatter.MarkerShape = MarkerShape.FilledSquare;
                                scatter.LineWidth = 0;
                                scatter.Color = ScottPlot.Color.FromColor(pointColor);
                            }
                            
                            // Set axes to EXACT pixel space (0 to imageSize) for perfect alignment
                            layerPlt.Axes.SetLimits(0, tileSize, 0, tileSize);
                            layerPlt.Axes.SquareUnits();
                            layerPlt.HideGrid();
                            
                            string overlayPath = Path.Combine(overlayDir, $"layer_{layer.MinUniqueId}_{layer.MaxUniqueId}.png");
                            layerPlt.SavePng(overlayPath, tileSize, tileSize);
                        }
                    }
                }
                
                // Generate HTML page for this tile (overlays now exist!)
                string tileHtmlPath = Path.Combine(tilesDir.FullName, $"{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.html");
                GenerateTileHtmlPage(tileHtmlPath, mapName, tileInfo, Path.GetFileName(outputPath), minimapResult.MinimapDir, visualGlobalMinId, visualGlobalMaxId);
                
                tileCount++;
                if (tileCount % 50 == 0)
                {
                    Console.WriteLine($"  Generated {tileCount}/{tileLayerInfos.Count} tiles...");
                }
            }
            
            Console.WriteLine($"✓ Generated {tileCount} tile images + HTML pages with layer overlays");
            Console.WriteLine();
            
            // Old STEP 3b removed - now done inline above
            if (false && !string.IsNullOrEmpty(minimapResult.MinimapDir))
            {
                Console.WriteLine("═══ STEP 3b: Generating Layer Overlay PNGs ═══");
                int overlaysGenerated = 0;
                
                foreach (var tileInfo in tileLayerInfos)
                {
                    var minimapPath = Path.Combine(minimapResult.MinimapDir, $"{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.png");
                    if (!File.Exists(minimapPath)) continue;
                    
                    var tileRecords = records.Where(r => r.TileX == tileInfo.TileX && r.TileY == tileInfo.TileY).ToList();
                    if (tileRecords.Count == 0) continue;
                    
                    // Create overlay directory for this tile
                    var overlayDir = Path.Combine(tilesDir.FullName, $"overlays_{tileInfo.TileX}_{tileInfo.TileY}");
                    Directory.CreateDirectory(overlayDir);
                    
                    // Generate one transparent PNG per layer with GLOBAL coloring
                    foreach (var layer in tileInfo.Layers)
                    {
                        var layerRecords = tileRecords.Where(r => r.UniqueId >= layer.MinUniqueId && 
                                                                  r.UniqueId <= layer.MaxUniqueId).ToList();
                        if (layerRecords.Count == 0) continue;
                        
                        var plt = new Plot();
                        plt.Layout.Frameless();
                        
                        // Color each point based on GLOBAL UniqueID gradient for consistency
                        foreach (var record in layerRecords)
                        {
                            float normalizedPos = (record.UniqueId - visualGlobalMinId) / (float)visualGlobalRange;
                            float hue = (1.0f - normalizedPos) * 240f;
                            var pointColor = ColorFromHSV(hue, 0.85f, 0.95f);
                            
                            var plotCoord = CoordinateTransform.WorldToPlot(record.X, record.Y);
                            double[] xData = new[] { plotCoord.plotX };
                            double[] yData = new[] { plotCoord.plotY };
                            
                            var scatter = plt.Add.Scatter(xData, yData);
                            scatter.MarkerSize = tileMarkerSize * 1.5f;
                            scatter.MarkerShape = MarkerShape.FilledCircle;
                            scatter.LineWidth = 0;
                            scatter.Color = ScottPlot.Color.FromColor(pointColor);
                        }
                        
                        plt.Axes.SquareUnits();
                        plt.HideGrid();
                        
                        string overlayPath = Path.Combine(overlayDir, $"layer_{layer.MinUniqueId}_{layer.MaxUniqueId}.png");
                        plt.SavePng(overlayPath, tileSize, tileSize);
                        overlaysGenerated++;
                    }
                }
                
                Console.WriteLine($"✓ Generated {overlaysGenerated} layer overlay images");
                Console.WriteLine();
            }
            
            // STEP 4: Generate map-wide overview as HEATMAP
            Console.WriteLine("═══ STEP 4: Generating Map Overview Heatmap ═══");
            Console.WriteLine($"  Found {globalLayers.Count} global layers across all tiles");
            
            var mapPlot = new Plot();
            
            uint globalMinId = records.Min(r => r.UniqueId);
            uint globalMaxId = records.Max(r => r.UniqueId);
            uint globalRange = Math.Max(globalMaxId - globalMinId, 1);
            
            // Divide into 100 color buckets for smooth gradient across entire map
            int mapBuckets = 100;
            
            Console.WriteLine($"  Plotting heatmap with {mapBuckets} color gradients (UniqueID {globalMinId}-{globalMaxId})...");
            
            for (int bucket = 0; bucket < mapBuckets; bucket++)
            {
                uint bucketStart = globalMinId + (uint)((globalRange * bucket) / mapBuckets);
                uint bucketEnd = globalMinId + (uint)((globalRange * (bucket + 1)) / mapBuckets);
                
                var bucketRecords = records.Where(r => r.UniqueId >= bucketStart && r.UniqueId < bucketEnd).ToList();
                if (bucket == mapBuckets - 1) // Last bucket includes max
                    bucketRecords = records.Where(r => r.UniqueId >= bucketStart && r.UniqueId <= bucketEnd).ToList();
                
                if (bucketRecords.Count == 0) continue;
                
                var plotCoords = bucketRecords.Select(r => CoordinateTransform.WorldToPlot(r.X, r.Y)).ToArray();
                double[] xData = plotCoords.Select(c => c.plotX).ToArray();
                double[] yData = plotCoords.Select(c => c.plotY).ToArray();
                
                // Color gradient: Blue (early) -> Green -> Yellow -> Red (late)
                float normalizedPos = bucket / (float)(mapBuckets - 1);
                float hue = (1.0f - normalizedPos) * 240f; // 240° = blue, 0° = red
                var bucketColor = ColorFromHSV(hue, 0.85f, 0.95f);
                
                var scatter = mapPlot.Add.Scatter(xData, yData);
                scatter.MarkerSize = mapMarkerSize;
                scatter.MarkerShape = MarkerShape.FilledCircle;
                scatter.LineWidth = 0;
                scatter.Color = ScottPlot.Color.FromColor(bucketColor);
                
                if ((bucket + 1) % 10 == 0)
                {
                    Console.WriteLine($"    Plotted {bucket + 1}/{mapBuckets} gradient buckets...");
                }
            }
            
            mapPlot.Title($"{mapName} - {globalLayers.Count} Development Layers ({records.Count} placements)");
            mapPlot.XLabel("East ← → West");
            mapPlot.YLabel("South ← → North");
            mapPlot.Axes.SquareUnits();
            
            // Get the actual plot bounds for coordinate mapping
            var plotBounds = mapPlot.Axes.GetLimits();
            
            string mapPath = Path.Combine(outputDir.FullName, $"{mapName}_overview.png");
            mapPlot.SavePng(mapPath, mapSize, mapSize);
            
            Console.WriteLine($"✓ Generated map overview: {mapName}_overview.png");
            Console.WriteLine();
            
            // STEP 4b: Generate HTML legend with overview
            Console.WriteLine("═══ STEP 4b: Generating Interactive Legend ═══");
            
            string htmlPath = Path.Combine(outputDir.FullName, $"{mapName}_legend.html");
            using (var writer = new StreamWriter(htmlPath))
            {
                writer.WriteLine("<!DOCTYPE html>");
                writer.WriteLine("<html>");
                writer.WriteLine("<head>");
                writer.WriteLine($"    <title>{mapName} - Development Layer Analysis</title>");
                writer.WriteLine("    <style>");
                writer.WriteLine("        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #1e1e1e; color: #d4d4d4; height: 100vh; overflow-y: auto; }");
                writer.WriteLine("        h1 { color: #4ec9b0; margin: 0 0 10px 0; font-size: 1.5em; }");
                writer.WriteLine("        h2 { color: #569cd6; margin: 15px 0 10px 0; font-size: 1.2em; }");
                writer.WriteLine("        .stats { background: #252526; padding: 10px; border-radius: 5px; margin: 10px 0; }");
                writer.WriteLine("        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }");
                writer.WriteLine("        .stat-item { background: #2d2d30; padding: 8px; border-radius: 3px; }");
                writer.WriteLine("        .stat-label { color: #858585; font-size: 0.8em; }");
                writer.WriteLine("        .stat-value { color: #4ec9b0; font-size: 1.2em; font-weight: bold; }");
                writer.WriteLine("        .overview { text-align: center; margin: 10px 0; }");
                writer.WriteLine("        .overview img { max-width: 100%; max-height: 50vh; width: auto; height: auto; border: 2px solid #3c3c3c; border-radius: 5px; object-fit: contain; }");
                writer.WriteLine("        .legend-container { margin: 10px 0; max-height: 35vh; overflow-y: auto; }");
                writer.WriteLine("        .legend-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 3px; }");
                writer.WriteLine("        .legend-item { display: flex; align-items: center; padding: 6px; background: #252526; border-radius: 3px; font-size: 0.85em; }");
                writer.WriteLine("        .legend-item:hover { background: #2d2d30; }");
                writer.WriteLine("        a.legend-item { color: inherit; text-decoration: none; transition: all 0.2s; }");
                writer.WriteLine("        a.legend-item:hover { background: #3c3c3c; transform: translateX(3px); }");
                writer.WriteLine("        .color-box { width: 25px; height: 15px; margin-right: 8px; border: 1px solid #3c3c3c; border-radius: 2px; flex-shrink: 0; }");
                writer.WriteLine("        .layer-info { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }");
                writer.WriteLine("        .layer-id { color: #569cd6; font-weight: bold; }");
                writer.WriteLine("        .layer-range { color: #4ec9b0; }");
                writer.WriteLine("        .layer-count { color: #858585; }");
                writer.WriteLine("        .filter { margin: 8px 0; }");
                writer.WriteLine("        .filter input { padding: 6px; width: 250px; background: #252526; border: 1px solid #3c3c3c; color: #d4d4d4; border-radius: 3px; font-size: 0.9em; }");
                writer.WriteLine("    </style>");
                writer.WriteLine("</head>");
                writer.WriteLine("<body>");
                writer.WriteLine($"    <h1>{mapName} - Development Layer Analysis</h1>");
                
                // Statistics
                writer.WriteLine("    <div class='stats'>");
                writer.WriteLine("        <div class='stats-grid'>");
                writer.WriteLine($"            <div class='stat-item'><div class='stat-label'>Total Layers</div><div class='stat-value'>{globalLayers.Count:N0}</div></div>");
                writer.WriteLine($"            <div class='stat-item'><div class='stat-label'>Total Placements</div><div class='stat-value'>{records.Count:N0}</div></div>");
                writer.WriteLine($"            <div class='stat-item'><div class='stat-label'>M2 Models</div><div class='stat-value'>{records.Count(r => r.Type == "M2"):N0}</div></div>");
                writer.WriteLine($"            <div class='stat-item'><div class='stat-label'>WMO Buildings</div><div class='stat-value'>{records.Count(r => r.Type == "WMO"):N0}</div></div>");
                writer.WriteLine("        </div>");
                writer.WriteLine("    </div>");
                
                // Overview image
                writer.WriteLine("    <h2>Continental Overview - Temporal Heatmap</h2>");
                writer.WriteLine("    <div class='overview'>");
                writer.WriteLine($"        <img src='{mapName}_overview.png' alt='Map Overview' />");
                writer.WriteLine("    </div>");
                
                // Heatmap Legend
                writer.WriteLine("    <h3 style='color: #4ec9b0;'>Heatmap Legend</h3>");
                writer.WriteLine("    <div style='background: #252526; padding: 20px; border-radius: 5px; margin: 20px 0;'>");
                writer.WriteLine("        <p style='color: #d4d4d4; margin-bottom: 15px;'>Colors represent <strong>when content was added</strong> based on UniqueID temporal ordering:</p>");
                
                // Gradient bar
                writer.WriteLine("        <div style='background: linear-gradient(to right, #0000FF 0%, #00FFFF 25%, #00FF00 50%, #FFFF00 75%, #FF0000 100%); height: 40px; border-radius: 5px; border: 2px solid #3c3c3c; position: relative;'>");
                
                // Add markers for key UniqueID values
                int numMarkers = 5;
                for (int i = 0; i < numMarkers; i++)
                {
                    float position = i / (float)(numMarkers - 1);
                    uint markerValue = globalMinId + (uint)(globalRange * position);
                    int leftPercent = (int)(position * 100);
                    
                    writer.WriteLine($"            <div style='position: absolute; left: {leftPercent}%; top: 100%; transform: translateX(-50%); margin-top: 5px; color: #d4d4d4; font-size: 0.85em; white-space: nowrap;'>{markerValue:N0}</div>");
                }
                
                writer.WriteLine("        </div>");
                writer.WriteLine("        <div style='display: flex; justify-content: space-between; margin-top: 50px; font-size: 0.9em;'>");
                writer.WriteLine("            <div style='color: #569cd6;'><strong>← Early Development</strong> (Blue)</div>");
                writer.WriteLine("            <div style='color: #ce9178;'><strong>Late Development →</strong> (Red)</div>");
                writer.WriteLine("        </div>");
                writer.WriteLine($"        <p style='color: #858585; font-size: 0.85em; margin-top: 15px;'>UniqueID Range: {globalMinId:N0} - {globalMaxId:N0} ({records.Count:N0} total placements)</p>");
                writer.WriteLine("    </div>");
                
                // Tile Browser - 2D Map Grid with proper continent layout
                if (tileLayerInfos.Count > 0)
                {
                    writer.WriteLine("    <h2>Tile Browser - Continent Map ({0} tiles with data)</h2>", tileLayerInfos.Count);
                    writer.WriteLine("    <div style='background: #252526; padding: 15px; border-radius: 5px; margin: 10px 0;'>");
                    writer.WriteLine("        <p style='color: #d4d4d4; margin-bottom: 15px;'>Visual representation of the map. Click any tile to view placement layers:</p>");
                    
                    // Calculate tile grid bounds
                    int minX = tileLayerInfos.Min(t => t.TileX);
                    int maxX = tileLayerInfos.Max(t => t.TileX);
                    int minY = tileLayerInfos.Min(t => t.TileY);
                    int maxY = tileLayerInfos.Max(t => t.TileY);
                    int gridWidth = maxX - minX + 1;
                    int gridHeight = maxY - minY + 1;
                    
                    // Create a lookup for quick tile access
                    var tileLookup = tileLayerInfos.ToDictionary(t => (t.TileX, t.TileY));
                    
                    writer.WriteLine($"        <div style='display: grid; grid-template-columns: repeat({gridWidth}, 64px); gap: 2px; max-height: 70vh; overflow: auto; padding: 5px; background: #1e1e1e; border-radius: 3px; justify-content: center;'>");
                    
                    // Generate grid in proper 2D layout (Y rows, X columns)
                    for (int y = minY; y <= maxY; y++)
                    {
                        for (int x = minX; x <= maxX; x++)
                        {
                            if (tileLookup.TryGetValue((x, y), out var tileInfo))
                            {
                                // Tile with data - show minimap thumbnail
                                var totalPlacements = tileInfo.Layers.Sum(l => l.PlacementCount);
                                var minimapPath = $"minimaps/{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.png";
                                var tileExists = File.Exists(Path.Combine(outputDir.FullName, minimapPath));
                                
                                if (tileExists)
                                {
                                    writer.WriteLine($"            <a href='tiles/{mapName}_{tileInfo.TileX}_{tileInfo.TileY}.html' title='Tile [{tileInfo.TileX},{tileInfo.TileY}] - {tileInfo.Layers.Count} layers, {totalPlacements} placements' style='display: block; width: 64px; height: 64px; border: 2px solid #569cd6; border-radius: 2px; overflow: hidden; transition: all 0.2s;' onmouseover='this.style.borderColor=\"#4ec9b0\"; this.style.transform=\"scale(1.05)\"; this.style.zIndex=\"10\";' onmouseout='this.style.borderColor=\"#569cd6\"; this.style.transform=\"scale(1)\"; this.style.zIndex=\"1\";'>");
                                    writer.WriteLine($"                <img src='{minimapPath}' style='width: 100%; height: 100%; object-fit: cover;' alt='[{tileInfo.TileX},{tileInfo.TileY}]' />");
                                    writer.WriteLine($"            </a>");
                                }
                                else
                                {
                                    // Data exists but no minimap - show placeholder
                                    writer.WriteLine($"            <div title='Tile [{x},{y}] - No minimap' style='width: 64px; height: 64px; border: 1px solid #3c3c3c; background: #2d2d30; border-radius: 2px;'></div>");
                                }
                            }
                            else
                            {
                                // No placement data - check if minimap exists
                                var minimapPath = $"minimaps/{mapName}_{x}_{y}.png";
                                var minimapExists = File.Exists(Path.Combine(outputDir.FullName, minimapPath));
                                
                                if (minimapExists)
                                {
                                    // Show minimap thumbnail (non-clickable, no placement data)
                                    writer.WriteLine($"            <div title='Tile [{x},{y}] - No placement data' style='width: 64px; height: 64px; border: 1px solid #3c3c3c; border-radius: 2px; overflow: hidden; opacity: 0.5;'>");
                                    writer.WriteLine($"                <img src='{minimapPath}' style='width: 100%; height: 100%; object-fit: cover;' alt='[{x},{y}]' />");
                                    writer.WriteLine($"            </div>");
                                }
                                else
                                {
                                    // No minimap, no data - empty placeholder
                                    writer.WriteLine($"            <div title='Tile [{x},{y}] - No data' style='width: 64px; height: 64px; border: 1px solid #1e1e1e; background: #252526; border-radius: 2px;'></div>");
                                }
                            }
                        }
                    }
                    
                    writer.WriteLine("        </div>");
                    writer.WriteLine($"        <p style='color: #858585; font-size: 0.85em; margin-top: 10px;'>Grid: {gridWidth}×{gridHeight} tiles (X: {minX}-{maxX}, Y: {minY}-{maxY})</p>");
                    writer.WriteLine("    </div>");
                }
                else
                {
                    writer.WriteLine("    <h2>Tile Browser</h2>");
                    writer.WriteLine("    <div style='background: #252526; padding: 15px; border-radius: 5px; margin: 10px 0;'>");
                    writer.WriteLine("        <p style='color: #ce9178;'>⚠️ No tiles with valid placement data after coordinate filtering.</p>");
                    writer.WriteLine("        <p style='color: #858585; font-size: 0.9em;'>This may indicate all placements were 'spanned' duplicates that don't belong to their assigned tiles.</p>");
                    writer.WriteLine("    </div>");
                }
                
                // JavaScript for filtering (kept for future use)
                writer.WriteLine("    <script>");
                writer.WriteLine("        function filterTiles() {");
                writer.WriteLine("            const input = document.getElementById('filterTileInput');");
                writer.WriteLine("            const filter = input.value.toLowerCase();");
                writer.WriteLine("            const items = document.querySelectorAll('#tileGrid .legend-item');");
                writer.WriteLine("            items.forEach(item => {");
                writer.WriteLine("                const text = item.getAttribute('data-search').toLowerCase();");
                writer.WriteLine("                item.style.display = text.includes(filter) ? '' : 'none';");
                writer.WriteLine("            });");
                writer.WriteLine("        }");
                writer.WriteLine("    </script>");
                
                writer.WriteLine("</body>");
                writer.WriteLine("</html>");
            }
            
            Console.WriteLine($"✓ Generated legend: {mapName}_legend.html ({globalLayers.Count} layers)");
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
            Console.WriteLine($"  ├─ {mapName}_legend.html        ⭐ OPEN THIS! Clickable map + full legend");
            Console.WriteLine($"  ├─ {mapName}_overview.png       (map with {globalLayers.Count} colored layers)");
            Console.WriteLine($"  ├─ {mapName}_analysis.json      (layer metadata)");
            Console.WriteLine($"  └─ tiles/                       ({tileCount} clickable tile pages)");
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
