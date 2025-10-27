using System.CommandLine;
using System.Text.Json;
using ScottPlot;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using GillijimProject.WowFiles.Alpha;
using WoWDataPlot.Helpers;
using WoWDataPlot.Extractors;
using WoWDataPlot.Services;
using WoWRollback.AnalysisModule;
using WoWDataPlot.Models;
using WoWRollback.Core.Services.Viewer;
using WoWRollback.Core.Services;
using WoWRollback.DbcModule;
using WoWFormatLib.FileReaders;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
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
        var wdtOptionVisualize = new Option<FileInfo>("--wdt", "Path to WDT/ADT file") { IsRequired = true };
        var outputDirOptionVisualize = new Option<DirectoryInfo>("--output-dir", "Output directory for all results") { IsRequired = true };
        var gapThresholdOption = new Option<int>("--gap-threshold", () => 50, "UniqueID gap size to split layers (gaps = dev pauses)");
        var tileSizeOption = new Option<int>("--tile-size", () => 256, "Tile image size (square) - should match minimap tile size");
        var mapSizeOption = new Option<int>("--map-size", () => 2048, "Map overview size (square)");
        var tileMarkerSizeOption = new Option<float>("--tile-marker-size", () => 8.0f, "Marker size for tile images");
        var mapMarkerSizeOption = new Option<float>("--map-marker-size", () => 5.0f, "Marker size for map overview");
        var mapMaxLayersOption = new Option<int>("--map-max-layers", () => 0, "Max layers to show on map overview (0 = all)");
        var minimapDirOption = new Option<DirectoryInfo?>("--minimap-dir", () => null, "External directory containing minimap PNGs (e.g., World/Textures/Minimap)");
        var globalRangeFileOption = new Option<FileInfo?>("--global-range-file", () => null, "JSON file with global UniqueID ranges (from analyze-client)");
        
        visualizeCommand.AddOption(wdtOptionVisualize);
        visualizeCommand.AddOption(outputDirOptionVisualize);
        visualizeCommand.AddOption(gapThresholdOption);
        visualizeCommand.AddOption(tileSizeOption);
        visualizeCommand.AddOption(mapSizeOption);
        visualizeCommand.AddOption(tileMarkerSizeOption);
        visualizeCommand.AddOption(mapMarkerSizeOption);
        visualizeCommand.AddOption(mapMaxLayersOption);
        visualizeCommand.AddOption(minimapDirOption);
        visualizeCommand.AddOption(globalRangeFileOption);
        
        visualizeCommand.SetHandler(async (context) => 
        {
            var wdt = context.ParseResult.GetValueForOption(wdtOptionVisualize)!;
            var outputDir = context.ParseResult.GetValueForOption(outputDirOptionVisualize)!;
            var gapThreshold = context.ParseResult.GetValueForOption(gapThresholdOption);
            var tileSize = context.ParseResult.GetValueForOption(tileSizeOption);
            var mapSize = context.ParseResult.GetValueForOption(mapSizeOption);
            var tileMarkerSize = context.ParseResult.GetValueForOption(tileMarkerSizeOption);
            var mapMarkerSize = context.ParseResult.GetValueForOption(mapMarkerSizeOption);
            var mapMaxLayers = context.ParseResult.GetValueForOption(mapMaxLayersOption);
            var minimapDir = context.ParseResult.GetValueForOption(minimapDirOption);
            var globalRangeFile = context.ParseResult.GetValueForOption(globalRangeFileOption);
            
            await VisualizeComplete(wdt, outputDir, gapThreshold, tileSize, mapSize, tileMarkerSize, mapMarkerSize, mapMaxLayers, minimapDir, globalRangeFile);
        });
        
        // analyze-client command - NEW: Global UniqueID range tracking
        var analyzeClientCommand = new Command("analyze-client", "Analyze UniqueID ranges across all maps in a WoW client");
        var clientRootOption = new Option<DirectoryInfo>("--client-root", "Path to WoW client root (contains World/Maps)") { IsRequired = true };
        var globalOutputOption = new Option<FileInfo>("--output", "Output JSON file path") { IsRequired = true };
        
        analyzeClientCommand.AddOption(clientRootOption);
        analyzeClientCommand.AddOption(globalOutputOption);
        
        analyzeClientCommand.SetHandler(AnalyzeClient, clientRootOption, globalOutputOption);
        
        // rollback command - CORE FEATURE: Filter content by UniqueID threshold
        var rollbackCommand = new Command("rollback", "Roll back map content to a specific UniqueID threshold");
        var inputWdtOption = new Option<FileInfo>("--input", "Input WDT file to rollback") { IsRequired = true };
        var outputWdtOption = new Option<FileInfo>("--output", "Output WDT file path") { IsRequired = true };
        var maxUniqueIdOption = new Option<uint>("--max-uniqueid", "Maximum UniqueID to keep (placements above this will be buried)") { IsRequired = true };
        var buryDepthOption = new Option<float>("--bury-depth", () => -5000.0f, "Z-coordinate to bury filtered placements (negative = underground)");
        
        rollbackCommand.AddOption(inputWdtOption);
        rollbackCommand.AddOption(outputWdtOption);
        rollbackCommand.AddOption(maxUniqueIdOption);
        rollbackCommand.AddOption(buryDepthOption);
        
        rollbackCommand.SetHandler(RollbackWdt, inputWdtOption, outputWdtOption, maxUniqueIdOption, buryDepthOption);
        
        // layers-ui command (lightweight per-tile layers UI generator)
        var layersUiCommand = new Command("layers-ui", "Generate a simple per-tile layers UI (static HTML) from an Alpha/LK WDT");
        var wdtOpt = new Option<FileInfo>("--wdt", "Path to WDT file") { IsRequired = true };
        var outDirOpt = new Option<DirectoryInfo>("--output-dir", "Output directory for all results") { IsRequired = true };
        var gapOpt = new Option<int>("--gap-threshold", () => 50, "UniqueID gap size to split layers per tile");
        var minimapOpt = new Option<DirectoryInfo?>("--minimap-dir", () => null, "Directory containing minimap PNGs for the map (optional)");
        var dbdDirOpt = new Option<DirectoryInfo?>("--dbd-dir", () => null, "Path to WoWDBDefs directory (optional, enables area name enrichment)");
        var dbcDirOpt = new Option<DirectoryInfo?>("--dbc-dir", () => null, "Path to DBFilesClient directory (optional; deduced if omitted)");
        var buildOpt = new Option<string?>("--build", () => null, "Build version (optional; deduced if omitted)");
        var areaAdtDirOpt = new Option<DirectoryInfo?>("--area-adt-dir", () => null, "Directory containing ADTs to read AreaIDs from (e.g., LK World/Maps/<map>)");
        layersUiCommand.AddOption(wdtOpt);
        layersUiCommand.AddOption(outDirOpt);
        layersUiCommand.AddOption(gapOpt);
        layersUiCommand.AddOption(minimapOpt);
        layersUiCommand.AddOption(dbdDirOpt);
        layersUiCommand.AddOption(dbcDirOpt);
        layersUiCommand.AddOption(buildOpt);
        layersUiCommand.AddOption(areaAdtDirOpt);
        layersUiCommand.SetHandler(async (FileInfo wdt, DirectoryInfo outDir, int gap, DirectoryInfo? minimaps, DirectoryInfo? dbdDir, DirectoryInfo? dbcDir, string? build, DirectoryInfo? areaAdtDir) =>
        {
            await GenerateLayersUi(wdt, outDir, gap, minimaps, dbdDir, dbcDir, build, areaAdtDir);
        }, wdtOpt, outDirOpt, gapOpt, minimapOpt, dbdDirOpt, dbcDirOpt, buildOpt, areaAdtDirOpt);
        
        // layers-ui-serve command (optional HTTP hosting via ViewerModule)
        var serveCommand = new Command("layers-ui-serve", "Serve a Layers UI folder via embedded HTTP server (ViewerModule)");
        var serveDirOpt = new Option<DirectoryInfo>("--dir", "Directory containing index.html (e.g., layers_ui_out/layers_ui)") { IsRequired = true };
        var portOpt = new Option<int>("--port", () => 8080, "HTTP port");
        serveCommand.AddOption(serveDirOpt);
        serveCommand.AddOption(portOpt);
        serveCommand.SetHandler((DirectoryInfo dir, int port) =>
        {
            var abs = Path.GetFullPath(dir.FullName);
            if (!Directory.Exists(abs)) { Console.Error.WriteLine($"[ERROR] Directory not found: {abs}"); return; }
            Console.WriteLine($"[serve] Serving {abs} at http://localhost:{port}/ (Ctrl+C to stop)");
            using var server = new WoWRollback.ViewerModule.ViewerServer();
            server.Start(abs, port);
            var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
            try { Task.Delay(Timeout.Infinite, cts.Token).Wait(); } catch { }
            server.Stop();
        }, serveDirOpt, portOpt);

        rootCommand.AddCommand(plotCommand);
        rootCommand.AddCommand(exportCommand);
        rootCommand.AddCommand(visualizeCommand);
        rootCommand.AddCommand(analyzeClientCommand);
        rootCommand.AddCommand(rollbackCommand);
        rootCommand.AddCommand(layersUiCommand);
        rootCommand.AddCommand(serveCommand);

        return await rootCommand.InvokeAsync(args);
    }

    // === layers-ui implementation ===
    private static async Task GenerateLayersUi(FileInfo wdtFile, DirectoryInfo outputDir, int gapThreshold, DirectoryInfo? minimapDir, DirectoryInfo? dbdDir, DirectoryInfo? dbcDir, string? build, DirectoryInfo? areaAdtDir)
    {
        Console.WriteLine($"=== WoWDataPlot - Layers UI (no server) ===");
        Console.WriteLine($"Input WDT:   {wdtFile.FullName}");
        Console.WriteLine($"Output Dir:  {outputDir.FullName}");
        Directory.CreateDirectory(outputDir.FullName);

        var mapName = Path.GetFileNameWithoutExtension(wdtFile.Name);
        var placementsCsv = Path.Combine(outputDir.FullName, "placements.csv");

        // 1) Extract placements and write CSV in UniqueIdAnalyzer schema
        Console.WriteLine("[1/5] Extracting placements...");
        var progress = new Progress<string>(msg => Console.WriteLine($"[info] {msg}"));
        var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
        using (var sw = new StreamWriter(placementsCsv))
        {
            // Header expected by UniqueIdAnalyzer.AnalyzeFromPlacementsCsv
            sw.WriteLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set");
            foreach (var r in records)
            {
                var line = string.Join(',',
                    Csv(mapName),
                    r.TileX.ToString(CultureInfo.InvariantCulture),
                    r.TileY.ToString(CultureInfo.InvariantCulture),
                    Csv(r.Type),
                    Csv(r.Name ?? string.Empty),
                    r.UniqueId.ToString(CultureInfo.InvariantCulture),
                    r.X.ToString(CultureInfo.InvariantCulture),
                    r.Y.ToString(CultureInfo.InvariantCulture),
                    r.Z.ToString(CultureInfo.InvariantCulture),
                    "0","0","0","1","",""
                );
                sw.WriteLine(line);
            }
        }

        // 2) Analyze per-tile layers using AnalysisModule
        Console.WriteLine("[2/5] Analyzing per-tile layers...");
        var analyzer = new UniqueIdAnalyzer(gapThreshold);
        var result = analyzer.AnalyzeFromPlacementsCsv(placementsCsv, mapName, outputDir.FullName);
        if (!result.Success)
        {
            Console.WriteLine($"[ERROR] Analysis failed: {result.ErrorMessage}");
            return;
        }

        // 2b) Generate areas.csv (best-effort) by scanning ADTs for AreaID majority per tile
        try
        {
            Console.WriteLine("[2b/5] Computing per-tile AreaIDs (areas.csv)...");
            GenerateAreasCsvFromWdt(wdtFile.FullName, outputDir.FullName, areaAdtDir?.FullName);
            TryEnrichAreasCsv(wdtFile.FullName, outputDir.FullName, dbdDir, dbcDir, build);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[warn] areas.csv generation skipped or not enriched: {ex.Message}");
        }

        // 3) Normalize filenames to fixed names for the UI
        var mapTileLayersCsv = Path.Combine(outputDir.FullName, $"{mapName}_tile_layers.csv");
        var fixedTileLayersCsv = Path.Combine(outputDir.FullName, "tile_layers.csv");
        if (File.Exists(mapTileLayersCsv)) File.Copy(mapTileLayersCsv, fixedTileLayersCsv, overwrite: true);
        var mapLayersJson = Path.Combine(outputDir.FullName, $"{mapName}_layers.json");
        var fixedLayersJson = Path.Combine(outputDir.FullName, "layers.json");
        if (File.Exists(mapLayersJson)) File.Copy(mapLayersJson, fixedLayersJson, overwrite: true);

        // 4) Optionally copy minimaps (recursive) and build index
        string miniIndexJson = "{}";
        bool hasMinimap = false;
        if (minimapDir != null && minimapDir.Exists)
        {
            var dst = Path.Combine(outputDir.FullName, "minimap");
            Directory.CreateDirectory(dst);
            var miniIndex = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            int copied = 0;
            foreach (var png in Directory.EnumerateFiles(minimapDir.FullName, "*.png", SearchOption.AllDirectories))
            {
                var rel = Path.GetRelativePath(minimapDir.FullName, png);
                var outPath = Path.Combine(dst, rel);
                Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);
                File.Copy(png, outPath, overwrite: true);
                copied++;

                // mapName_X_Y.png => key "X,Y"
                var fileName = Path.GetFileName(png);
                var m = System.Text.RegularExpressions.Regex.Match(
                    fileName,
                    "^" + System.Text.RegularExpressions.Regex.Escape(mapName) + "_(\\d+)_(\\d+)\\.png$",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase);
                if (m.Success)
                {
                    var key = m.Groups[1].Value + "," + m.Groups[2].Value;
                    var webRel = rel.Replace('\\','/');
                    miniIndex[key] = webRel;
                }
            }
            hasMinimap = miniIndex.Count > 0;
            miniIndexJson = "{" + string.Join(',', miniIndex.Select(kvp => System.Text.Json.JsonSerializer.Serialize(kvp.Key) + ":" + System.Text.Json.JsonSerializer.Serialize(kvp.Value))) + "}";
            Console.WriteLine($"[ok] Copied {copied} minimaps: {dst}");
        }

        // 5) Generate static UI with embedded data
        Console.WriteLine("[3/5] Preparing embedded data...");
        var tileData = ParseTileLayersCsv(fixedTileLayersCsv);
        var globalLayersJson = File.Exists(fixedLayersJson) ? File.ReadAllText(fixedLayersJson) : "{}";

        Console.WriteLine("[4/5] Emitting UI assets...");
        var uiDir = Path.Combine(outputDir.FullName, "layers_ui");
        Directory.CreateDirectory(uiDir);

        // index.html with embedded data variables (built with StringBuilder to avoid escape issues)
        var sbIndex = new System.Text.StringBuilder();
        sbIndex.AppendLine("<!DOCTYPE html>");
        sbIndex.AppendLine("<html>");
        sbIndex.AppendLine("<head>");
        sbIndex.AppendLine("  <meta charset=\"utf-8\" />");
        sbIndex.AppendLine("  <title>Layers UI - " + mapName + "</title>");
        sbIndex.AppendLine("  <link rel=\"stylesheet\" href=\"./styles.css\" />");
        sbIndex.AppendLine("</head>");
        sbIndex.AppendLine("<body>");
        sbIndex.AppendLine("  <div class=\"container\">");
        sbIndex.AppendLine("    <h1>" + mapName + " - Layers UI</h1>");
        sbIndex.AppendLine("    <div class=\"panel\">");
        sbIndex.AppendLine("      <div class=\"row\">");
        sbIndex.AppendLine("        <label>Tile</label>");
        sbIndex.AppendLine("        <select id=\"tileSelect\"></select>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("      <div id=\"minimapSlot\"></div>");
        sbIndex.AppendLine("      <div class=\"row\">");
        sbIndex.AppendLine("        <label>Mode</label>");
        sbIndex.AppendLine("        <select id=\"mode\">");
        sbIndex.AppendLine("          <option value=\"show\">Show Only</option>");
        sbIndex.AppendLine("          <option value=\"hide\">Hide</option>");
        sbIndex.AppendLine("          <option value=\"dim\">Dim</option>");
        sbIndex.AppendLine("        </select>");
        sbIndex.AppendLine("        <label style=\"margin-left:10px\"><input type=\"checkbox\" id=\"currentTileOnly\"/> Current Tile</label>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("    </div>");
        sbIndex.AppendLine("    <div class=\"columns\">");
        sbIndex.AppendLine("      <div class=\"col\">");
        sbIndex.AppendLine("        <h3>Per-Tile Layers</h3>");
        sbIndex.AppendLine("        <div class=\"row\"><label><input type=\"checkbox\" id=\"m2Toggle\" checked/> M2</label> <label style=\"margin-left:10px\"><input type=\"checkbox\" id=\"wmoToggle\" checked/> WMO</label></div>");
        sbIndex.AppendLine("        <div id=\"tileLayersList\"></div>");
        sbIndex.AppendLine("        <h3 style=\"margin-top:10px\">Per-Tile Custom Ranges</h3>");
        sbIndex.AppendLine("        <div id=\"tileCustomList\"></div>");
        sbIndex.AppendLine("        <div class=\"row\">");
        sbIndex.AppendLine("          <input id=\"rangeMinTile\" type=\"number\" placeholder=\"Min UniqueID\" style=\"width:120px\"/>");
        sbIndex.AppendLine("          <input id=\"rangeMaxTile\" type=\"number\" placeholder=\"Max UniqueID\" style=\"width:120px\"/>");
        sbIndex.AppendLine("          <select id=\"rangeTypeTile\"><option value=\"M2\">M2</option><option value=\"WMO\">WMO</option><option value=\"both\">Both</option></select>");
        sbIndex.AppendLine("          <button id=\"addTileRange\">Add Tile Range</button>");
        sbIndex.AppendLine("          <button id=\"clearTileCustom\">Clear Tile Custom</button>");
        sbIndex.AppendLine("        </div>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("      <div class=\"col\">");
        sbIndex.AppendLine("        <h3>Global Ranges</h3>");
        sbIndex.AppendLine("        <div id=\"globalLayersList\"></div>");
        sbIndex.AppendLine("        <div class=\"row\">");
        sbIndex.AppendLine("          <input id=\"rangeMin\" type=\"number\" placeholder=\"Min UniqueID\" style=\"width:120px\"/>");
        sbIndex.AppendLine("          <input id=\"rangeMax\" type=\"number\" placeholder=\"Max UniqueID\" style=\"width:120px\"/>");
        sbIndex.AppendLine("          <select id=\"rangeType\"><option value=\"both\">Both</option><option value=\"M2\">M2</option><option value=\"WMO\">WMO</option></select>");
        sbIndex.AppendLine("          <button id=\"addGlobalRange\">Add Global Range</button>");
        sbIndex.AppendLine("        </div>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("    </div>");
        sbIndex.AppendLine("    <div class=\"columns\">");
        sbIndex.AppendLine("      <div class=\"col\">");
        sbIndex.AppendLine("        <h3>Preset</h3>");
        sbIndex.AppendLine("        <div class=\"row\"><button id=\"savePreset\">Save Preset</button> <input type=\"file\" id=\"loadPreset\" accept=\"application/json\" /></div>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("      <div class=\"col\">");
        sbIndex.AppendLine("        <h3>Command Generator</h3>");
        sbIndex.AppendLine("        <div class=\"row\"><input id=\"inputWdt\" placeholder=\"Alpha WDT path\" style=\"width:100%\"/></div>");
        sbIndex.AppendLine("        <div class=\"row\"><input id=\"outRoot\" placeholder=\"Output root folder\" style=\"width:100%\"/></div>");
        sbIndex.AppendLine("        <div class=\"row\"><input id=\"lkOut\" placeholder=\"LK ADT out (optional)\" style=\"width:100%\"/></div>");
        sbIndex.AppendLine("        <div class=\"row\"><input id=\"lkClient\" placeholder=\"LK client path (optional)\" style=\"width:100%\"/></div>");
        sbIndex.AppendLine("        <div class=\"row\"><button id=\"genAlphaToLk\">Generate alpha-to-lk</button> <button id=\"genLkToAlpha\">Generate lk-to-alpha</button></div>");
        sbIndex.AppendLine("        <textarea id=\"cmdOut\" rows=\"6\" style=\"width:100%;\"></textarea>");
        sbIndex.AppendLine("      </div>");
        sbIndex.AppendLine("    </div>");
        sbIndex.AppendLine("  </div>");
        sbIndex.AppendLine("  <script>");
        sbIndex.AppendLine("    window.__mapName = " + JsonEscape(mapName) + ";");
        sbIndex.AppendLine("    window.__tileLayers = " + tileData + ";");
        sbIndex.AppendLine("    window.__globalLayers = " + globalLayersJson + ";");
        sbIndex.AppendLine("    window.__hasMinimap = " + (hasMinimap?"true":"false") + ";");
        sbIndex.AppendLine("    window.__minimapIndex = " + miniIndexJson + ";");
        sbIndex.AppendLine("  </script>");
        sbIndex.AppendLine("  <script src=\"./app.js\"></script>");
        sbIndex.AppendLine("</body>");
        sbIndex.AppendLine("</html>");
        File.WriteAllText(Path.Combine(uiDir, "index.html"), sbIndex.ToString());

        // app.js (use single quotes and build with StringBuilder to avoid escaping headaches)
        var sbJs = new System.Text.StringBuilder();
        sbJs.AppendLine("(function(){");
        sbJs.AppendLine("  const state = { mode:'show', currentTileOnly:false, m2:true, wmo:true, preset:{ global:{m2:[], wmo:[]}, tiles:{} } };");
        sbJs.AppendLine("  const TL = window.__tileLayers || [];");
        sbJs.AppendLine("  const GL = (window.__globalLayers && (window.__globalLayers.globalLayers || window.__globalLayers.GlobalLayers)) || [];");
        sbJs.AppendLine("  const mapName = window.__mapName || '';");
        sbJs.AppendLine("  let tileState = {};");
        sbJs.AppendLine();
        sbJs.AppendLine("  function tileKey(x,y){ return 'r'+y+'_c'+x; }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function ensureGlobalState(){");
        sbJs.AppendLine("    if (state.preset.global.m2.length===0 && state.preset.global.wmo.length===0){");
        sbJs.AppendLine("      GL.forEach(g=>{ const min = g.idRangeStart ?? g.IdRangeStart ?? g.minId ?? 0; const max = g.idRangeEnd ?? g.IdRangeEnd ?? g.maxId ?? 0; state.preset.global.m2.push({min,max,enabled:true}); state.preset.global.wmo.push({min,max,enabled:true}); });");
        sbJs.AppendLine("    }");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function ensureTileState(x,y){");
        sbJs.AppendLine("    const key = tileKey(x,y);");
        sbJs.AppendLine("    let ts = tileState[key];");
        sbJs.AppendLine("    if (!ts){");
        sbJs.AppendLine("      ts = { base:{m2:[], wmo:[]}, custom:{m2:[], wmo:[]}, _hydrated:false };");
        sbJs.AppendLine("      const rows = TL.filter(r=>r.tile_x===x && r.tile_y===y).sort((a,b)=>a.layer-b.layer);");
        sbJs.AppendLine("      rows.forEach(r=>{ const e = { layer:r.layer, min:r.range_start, max:r.range_end, count:r.count, enabled:true }; if(r.type==='M2') ts.base.m2.push(e); else ts.base.wmo.push(e); });");
        sbJs.AppendLine("      tileState[key] = ts;");
        sbJs.AppendLine("    }");
        sbJs.AppendLine("    if (!ts._hydrated){");
        sbJs.AppendLine("      const presetTile = (state.preset && state.preset.tiles && state.preset.tiles[key]) || null;");
        sbJs.AppendLine("      if (presetTile){");
        sbJs.AppendLine("        const applyArr = (arr, baseArr, customArr) => { (arr||[]).forEach(e=>{ const on = (e.enabled!==false); const m = baseArr.find(b=>b.layer===e.layer && b.min===e.min && b.max===e.max); if(m){ m.enabled = on; } else { customArr.push({ layer: (typeof e.layer==='number'?e.layer:-1), min:e.min, max:e.max, count: e.count||0, enabled: on }); } }); };");
        sbJs.AppendLine("        applyArr(presetTile.m2, ts.base.m2, ts.custom.m2);");
        sbJs.AppendLine("        applyArr(presetTile.wmo, ts.base.wmo, ts.custom.wmo);");
        sbJs.AppendLine("      }");
        sbJs.AppendLine("      ts._hydrated = true;");
        sbJs.AppendLine("    }");
        sbJs.AppendLine("    return ts;");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function init(){");
        sbJs.AppendLine("    const sel = document.getElementById('tileSelect');");
        sbJs.AppendLine("    const tiles = new Set(TL.map(r=>r.tile_x+','+r.tile_y));");
        sbJs.AppendLine("    Array.from(tiles).sort((a,b)=>{ const [ax,ay]=a.split(',').map(Number); const [bx,by]=b.split(',').map(Number); return (ay - by) || (ax - bx); }).forEach(t=>{ const o=document.createElement('option'); o.value=t; o.textContent='['+t+']'; sel.appendChild(o); });");
        sbJs.AppendLine("    if (sel.options.length > 0) sel.selectedIndex = 0;");
        sbJs.AppendLine("    sel.addEventListener('change', ()=>{ renderTileLayers(); renderTileCustom(); });");
        sbJs.AppendLine("    document.getElementById('m2Toggle').addEventListener('change',e=>{state.m2=e.target.checked; renderTileLayers();});");
        sbJs.AppendLine("    document.getElementById('wmoToggle').addEventListener('change',e=>{state.wmo=e.target.checked; renderTileLayers();});");
        sbJs.AppendLine("    document.getElementById('mode').addEventListener('change',e=>{state.mode=e.target.value;});");
        sbJs.AppendLine("    document.getElementById('currentTileOnly').addEventListener('change',e=>{state.currentTileOnly=e.target.checked;});");
        sbJs.AppendLine("    document.getElementById('savePreset').addEventListener('click',savePreset);");
        sbJs.AppendLine("    document.getElementById('loadPreset').addEventListener('change',loadPreset);");
        sbJs.AppendLine("    document.getElementById('genAlphaToLk').addEventListener('click',genAlphaToLk);");
        sbJs.AppendLine("    document.getElementById('genLkToAlpha').addEventListener('click',genLkToAlpha);");
        sbJs.AppendLine("    const addBtn = document.getElementById('addGlobalRange'); if(addBtn) addBtn.addEventListener('click', addGlobalRange);");
        sbJs.AppendLine("    const addTileBtn = document.getElementById('addTileRange'); if(addTileBtn) addTileBtn.addEventListener('click', addTileRange);");
        sbJs.AppendLine("    const clearTileBtn = document.getElementById('clearTileCustom'); if(clearTileBtn) clearTileBtn.addEventListener('click', clearTileCustom);");
        sbJs.AppendLine("    ensureGlobalState();");
        sbJs.AppendLine("    renderGlobal();");
        sbJs.AppendLine("    renderTileLayers();");
        sbJs.AppendLine("    renderTileCustom();");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function renderTileLayers(){");
        sbJs.AppendLine("    const list = document.getElementById('tileLayersList'); list.innerHTML='';");
        sbJs.AppendLine("    const val = document.getElementById('tileSelect').value || '0,0';");
        sbJs.AppendLine("    const parts = val.split(','); const x = parseInt(parts[0],10)||0; const y = parseInt(parts[1],10)||0;");
        sbJs.AppendLine("    if(window.__hasMinimap){");
        sbJs.AppendLine("      const slot=document.getElementById('minimapSlot'); slot.innerHTML=''; const idx=window.__minimapIndex||{}; const k = x+','+y; const rel = idx[k]; if(rel){ const img=new Image(); img.src='../minimap/'+rel; img.style.maxWidth='256px'; img.style.border='1px solid #444'; img.onerror=()=>{ img.style.display='none'; }; slot.appendChild(img);} }");
        sbJs.AppendLine("    const tState = ensureTileState(x,y);");
        sbJs.AppendLine("    const rows = [];");
        sbJs.AppendLine("    if (state.m2) tState.base.m2.forEach(e=>rows.push({type:'M2', layer:e.layer, min:e.min, max:e.max, count:e.count, enabled:e.enabled}));");
        sbJs.AppendLine("    if (state.wmo) tState.base.wmo.forEach(e=>rows.push({type:'WMO', layer:e.layer, min:e.min, max:e.max, count:e.count, enabled:e.enabled}));");
        sbJs.AppendLine("    rows.sort((a,b)=>a.layer-b.layer);");
        sbJs.AppendLine("    rows.forEach(r=>{ const id='tile_'+x+'_'+y+'_'+r.type+'_'+r.layer; const div=document.createElement('div'); div.className='item'; div.innerHTML = '<label><input type=\\'checkbox\\' id=\\''+id+'\\' '+(r.enabled?'checked':'')+'/> ['+r.type+'] L'+r.layer+': '+r.min+'-'+r.max+' <span class=\\'muted\\'>('+r.count+')</span></label>'; list.appendChild(div); document.getElementById(id).addEventListener('change', e=>{ if(r.type==='M2'){ const o = tState.base.m2.find(x=>x.layer===r.layer && x.min===r.min && x.max===r.max); if(o) o.enabled = e.target.checked; } else { const o = tState.base.wmo.find(x=>x.layer===r.layer && x.min===r.min && x.max===r.max); if(o) o.enabled = e.target.checked; } }); });");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function renderTileCustom(){");
        sbJs.AppendLine("    const list = document.getElementById('tileCustomList'); if(!list) return; list.innerHTML='';");
        sbJs.AppendLine("    const val = document.getElementById('tileSelect').value || '0,0'; const parts = val.split(','); const x = parseInt(parts[0],10)||0; const y = parseInt(parts[1],10)||0; const tState = ensureTileState(x,y);");
        sbJs.AppendLine("    const rows = []; tState.custom.m2.forEach(e=>rows.push({type:'M2', ref:e})); tState.custom.wmo.forEach(e=>rows.push({type:'WMO', ref:e}));");
        sbJs.AppendLine("    let idx=0; rows.forEach(r=>{ const e=r.ref; const id='cust_'+idx; const del='del_'+id; const div=document.createElement('div'); div.className='item'; div.innerHTML = '<label><input type=\\'checkbox\\' id=\\''+id+'\\' '+(e.enabled?'checked':'')+'/> [Custom '+r.type+'] '+e.min+'-'+e.max+'</label> <button id=\\''+del+'\\' style=\\'float:right\\'>\u2715</button>'; list.appendChild(div); document.getElementById(id).addEventListener('change', ev=>{ e.enabled = ev.target.checked; }); document.getElementById(del).addEventListener('click', ()=>{ const arr = (r.type==='M2') ? tState.custom.m2 : tState.custom.wmo; const j = arr.indexOf(e); if(j>=0){ arr.splice(j,1); renderTileCustom(); } }); idx++; });");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function renderGlobal(){");
        sbJs.AppendLine("    ensureGlobalState();");
        sbJs.AppendLine("    const list = document.getElementById('globalLayersList'); list.innerHTML='';");
        sbJs.AppendLine("    const byKey = {};");
        sbJs.AppendLine("    state.preset.global.m2.forEach(e=>{ const k=e.min+','+e.max; byKey[k] = byKey[k]||{min:e.min,max:e.max,m2:false,wmo:false}; byKey[k].m2 = e.enabled; });");
        sbJs.AppendLine("    state.preset.global.wmo.forEach(e=>{ const k=e.min+','+e.max; byKey[k] = byKey[k]||{min:e.min,max:e.max,m2:false,wmo:false}; byKey[k].wmo = e.enabled; });");
        sbJs.AppendLine("    Object.values(byKey).sort((a,b)=>a.min-b.min).forEach((g,idx)=>{ const id='global_'+idx; const div=document.createElement('div'); div.className='item'; const both = (g.m2 && g.wmo); div.innerHTML = '<label><input type=\\'checkbox\\' id=\\''+id+'\\' '+(both?'checked':'')+'/> '+g.min+'-'+g.max+'</label>'; list.appendChild(div); document.getElementById(id).addEventListener('change', e=>{ const setEntry=(arr)=>{ const t = arr.find(x=>x.min===g.min && x.max===g.max); if(t) t.enabled = e.target.checked; else arr.push({min:g.min,max:g.max,enabled:e.target.checked}); }; setEntry(state.preset.global.m2); setEntry(state.preset.global.wmo); }); });");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function addGlobalRange(){ const min = parseInt(document.getElementById('rangeMin').value,10); const max = parseInt(document.getElementById('rangeMax').value,10); const type = document.getElementById('rangeType').value; if(isNaN(min)||isNaN(max)||max<min){ alert('Invalid range'); return; } const add=(arr)=>{ const ex = arr.find(e=>e.min===min && e.max===max); if(ex) ex.enabled=true; else arr.push({min,max,enabled:true}); }; if(type==='both'){ add(state.preset.global.m2); add(state.preset.global.wmo);} else if(type==='M2'){ add(state.preset.global.m2);} else { add(state.preset.global.wmo);} renderGlobal(); }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function buildPreset(){");
        sbJs.AppendLine("    const tilesOut = {};");
        sbJs.AppendLine("    Object.keys(tileState).forEach(k=>{ const t = tileState[k]; const m2 = []; const wmo = []; t.base.m2.forEach(e=>{ if(e.enabled) m2.push({layer:e.layer,min:e.min,max:e.max,enabled:true}); }); t.custom.m2.forEach(e=>{ if(e.enabled) m2.push({layer:e.layer,min:e.min,max:e.max,enabled:true}); }); t.base.wmo.forEach(e=>{ if(e.enabled) wmo.push({layer:e.layer,min:e.min,max:e.max,enabled:true}); }); t.custom.wmo.forEach(e=>{ if(e.enabled) wmo.push({layer:e.layer,min:e.min,max:e.max,enabled:true}); }); if(m2.length||wmo.length){ tilesOut[k] = { m2, wmo }; } });");
        sbJs.AppendLine("    const savedTiles = (state.preset && state.preset.tiles) || {}; Object.keys(savedTiles).forEach(k=>{ if(!tilesOut[k]){ const st = savedTiles[k]; const m2 = (st.m2||[]).filter(e=>e.enabled!==false).map(e=>({layer:(typeof e.layer==='number'?e.layer:-1),min:e.min,max:e.max,enabled:true})); const wmo = (st.wmo||[]).filter(e=>e.enabled!==false).map(e=>({layer:(typeof e.layer==='number'?e.layer:-1),min:e.min,max:e.max,enabled:true})); if(m2.length||wmo.length){ tilesOut[k] = { m2, wmo }; } }});");
        sbJs.AppendLine("    const globalOut = { m2: state.preset.global.m2.filter(e=>e.enabled).map(e=>({min:e.min,max:e.max,enabled:true})), wmo: state.preset.global.wmo.filter(e=>e.enabled).map(e=>({min:e.min,max:e.max,enabled:true})) };");
        sbJs.AppendLine("    return { map: mapName, global: globalOut, tiles: tilesOut };");
        sbJs.AppendLine("  }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function savePreset(){ const preset = buildPreset(); const blob = new Blob([JSON.stringify(preset,null,2)], {type:'application/json'}); const a = document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=mapName+'_preset.json'; a.click(); }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function loadPreset(ev){ const f = ev.target.files && ev.target.files[0]; if(!f) return; const reader = new FileReader(); reader.onload = ()=>{ try{ state.preset = JSON.parse(reader.result); tileState = {}; ensureGlobalState(); renderGlobal(); renderTileLayers(); renderTileCustom(); }catch(e){ alert('Invalid preset'); } }; reader.readAsText(f); }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function addTileRange(){ const min = parseInt(document.getElementById('rangeMinTile').value,10); const max = parseInt(document.getElementById('rangeMaxTile').value,10); const type = document.getElementById('rangeTypeTile').value; if(isNaN(min)||isNaN(max)||max<min){ alert('Invalid range'); return; } const val = document.getElementById('tileSelect').value || '0,0'; const parts = val.split(','); const x = parseInt(parts[0],10)||0; const y = parseInt(parts[1],10)||0; const tState = ensureTileState(x,y); const pushUnique=(arr)=>{ const ex = arr.find(e=>e.layer===-1 && e.min===min && e.max===max); if(ex) ex.enabled=true; else arr.push({layer:-1,min,max,count:0,enabled:true}); }; if(type==='both'){ pushUnique(tState.custom.m2); pushUnique(tState.custom.wmo);} else if(type==='M2'){ pushUnique(tState.custom.m2);} else { pushUnique(tState.custom.wmo);} renderTileCustom(); }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function clearTileCustom(){ const val = document.getElementById('tileSelect').value || '0,0'; const parts = val.split(','); const x = parseInt(parts[0],10)||0; const y = parseInt(parts[1],10)||0; const tState = ensureTileState(x,y); tState.custom.m2 = []; tState.custom.wmo = []; renderTileCustom(); }");
        sbJs.AppendLine("  function genAlphaToLk(){ const input = document.getElementById('inputWdt').value.trim(); const out = document.getElementById('outRoot').value.trim(); const lkOut = document.getElementById('lkOut').value.trim(); const lkClient = document.getElementById('lkClient').value.trim(); const presetPath = mapName+'_preset.json'; const cmd = 'dotnet run --project WoWRollback.Cli -- alpha-to-lk  --input \"'+input+'\" --out \"'+out+'\"'+(lkOut?(' --lk-out \"'+lkOut+'\"'):'')+(lkClient?(' --lk-client-path \"'+lkClient+'\"'):'')+' --preset-json \"'+presetPath+'\" --fix-holes --disable-mcsh'; document.getElementById('cmdOut').value = cmd + '\\n# save preset as '+presetPath+' in working dir before running'; }");
        sbJs.AppendLine();
        sbJs.AppendLine("  function genLkToAlpha(){ const lkDir = document.getElementById('lkOut').value.trim(); const out = document.getElementById('outRoot').value.trim(); const presetPath = mapName+'_preset.json'; const cmd = 'dotnet run --project WoWRollback.Cli -- lk-to-alpha --lk-adts-dir \"'+lkDir+'\" --map '+mapName+' --out \"'+out+'\" --preset-json \"'+presetPath+'\" --fix-holes --disable-mcsh'; document.getElementById('cmdOut').value = cmd + '\\n# save preset as '+presetPath+' in working dir before running'; }");
        sbJs.AppendLine();
        sbJs.AppendLine("  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();");
        sbJs.AppendLine("})();");
        File.WriteAllText(Path.Combine(uiDir, "app.js"), sbJs.ToString());

        // styles.css
        var styles = @"body{font-family:Segoe UI,Arial;background:#1e1e1e;color:#dedede;margin:20px} .container{max-width:1100px;margin:0 auto}
h1{color:#4ec9b0} .panel,.col{background:#252526;border-radius:6px;padding:12px;margin:10px 0}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.row{display:flex;gap:8px;align-items:center;margin:6px 0}
.item{padding:6px;border-bottom:1px solid #3c3c3c}
.muted{color:#888}
";
        File.WriteAllText(Path.Combine(uiDir, "styles.css"), styles);

        Console.WriteLine($"[5/5] UI ready: {Path.Combine(uiDir, "index.html")}");
        await Task.CompletedTask;

        static string Csv(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            return (s.Contains(',') || s.Contains('"')) ? '"' + s.Replace("\"", "\"\"") + '"' : s;
        }

        static string JsonEscape(string s) => System.Text.Json.JsonSerializer.Serialize(s);

        static string ParseTileLayersCsv(string path)
        {
            if (!File.Exists(path)) return "[]";
            var lines = File.ReadAllLines(path);
            if (lines.Length <= 1) return "[]";
            var list = new List<string>();
            // header: map,tile_x,tile_y,type,layer,range_start,range_end,count
            for (int i = 1; i < lines.Length; i++)
            {
                var parts = lines[i].Split(',');
                if (parts.Length < 8) continue;
                string obj = $"{{\"map\":{JsonEscape(parts[0])},\"tile_x\":{parts[1]},\"tile_y\":{parts[2]},\"type\":{JsonEscape(parts[3])},\"layer\":{parts[4]},\"range_start\":{parts[5]},\"range_end\":{parts[6]},\"count\":{parts[7]} }}";
                list.Add(obj);
            }
            return "[" + string.Join(',', list) + "]";
        }
    }

    private static void GenerateAreasCsvFromWdt(string wdtPath, string outDir, string? adtDirOpt)
    {
        var wdtDir = Path.GetDirectoryName(wdtPath) ?? string.Empty;
        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var adtDir = !string.IsNullOrWhiteSpace(adtDirOpt) ? adtDirOpt! : wdtDir;
        if (string.IsNullOrWhiteSpace(adtDir) || string.IsNullOrWhiteSpace(mapName)) return;

        var results = new Dictionary<(int x, int y), int>();

        // Prefer split ADT format (_obj0.adt); fall back to regular ADTs
        var objAdtFiles = Directory.GetFiles(adtDir, $"{mapName}_*_obj0.adt", SearchOption.TopDirectoryOnly);
        string[] adtFiles;
        if (objAdtFiles.Length > 0)
        {
            adtFiles = objAdtFiles;
        }
        else
        {
            adtFiles = Directory.GetFiles(adtDir, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
                .Where(f => !f.Contains("_obj", StringComparison.OrdinalIgnoreCase)
                         && !f.Contains("_tex", StringComparison.OrdinalIgnoreCase)
                         && !f.Contains("_lgt", StringComparison.OrdinalIgnoreCase)
                         && !f.Contains("_occ", StringComparison.OrdinalIgnoreCase))
                .ToArray();
        }

        foreach (var adtFile in adtFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(adtFile);
            var parts = fileName.Split('_');
            if (parts.Length < 3) continue;
            if (!int.TryParse(parts[^2], out var tx)) continue;
            if (!int.TryParse(parts[^1], out var ty)) continue;
            var area = LkAdtReader.ReadTileMajorAreaId(adtFile);
            if (area.HasValue)
            {
                results[(tx, ty)] = area.Value;
            }
        }

        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "areas.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("tile_x,tile_y,area_id,area_name,parent_area_id,parent_area_name");
        if (results.Count > 0)
        {
            foreach (var kv in results.OrderBy(k => k.Key.y).ThenBy(k => k.Key.x))
            {
                sw.WriteLine($"{kv.Key.x},{kv.Key.y},{kv.Value},,,");
            }
        }
        Console.WriteLine($"[ok] Wrote areas: {path} ({results.Count} tiles)");
    }

    private static void TryEnrichAreasCsv(string wdtPath, string outDir, DirectoryInfo? dbdDir, DirectoryInfo? dbcDirOpt, string? buildOpt)
    {
        var csvPath = Path.Combine(outDir, "areas.csv");
        if (!File.Exists(csvPath)) return;
        if (dbdDir == null || !dbdDir.Exists) return;
        var dbcDir = dbcDirOpt != null && dbcDirOpt.Exists ? dbcDirOpt.FullName : DeduceDbcDirFromWdt(wdtPath);
        if (string.IsNullOrWhiteSpace(dbcDir) || !Directory.Exists(dbcDir)) return;
        var build = !string.IsNullOrWhiteSpace(buildOpt) ? buildOpt! : GuessBuildFromPath(wdtPath) ?? "3.3.5.12340";

        var dbdPath = dbdDir.FullName;
        var defs = Path.Combine(dbdPath, "definitions");
        if (Directory.Exists(defs)) dbdPath = defs;
        var reader = new AreaDbcReader(dbdPath);
        var data = reader.ReadAreas(build, dbcDir);
        if (!data.Success) return;
        var nameById = data.NameById;
        var parentById = data.ParentById;

        static string Esc(string s)
        {
            if (string.IsNullOrEmpty(s)) return s;
            return (s.Contains(',') || s.Contains('"')) ? '"' + s.Replace("\"", "\"\"") + '"' : s;
        }

        var lines = File.ReadAllLines(csvPath);
        if (lines.Length == 0) return;
        var outLines = new List<string>();
        outLines.Add("tile_x,tile_y,area_id,area_name,parent_area_id,parent_area_name");
        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i]; if (string.IsNullOrWhiteSpace(line)) continue;
            var p = line.Split(','); if (p.Length < 3) continue;
            if (!int.TryParse(p[0], out var tx)) continue;
            if (!int.TryParse(p[1], out var ty)) continue;
            if (!int.TryParse(p[2], out var aid)) continue;
            var aname = nameById.TryGetValue(aid, out var n) ? n : "";
            var pid = parentById.TryGetValue(aid, out var pidVal) ? pidVal : 0;
            var pname = pid > 0 && nameById.TryGetValue(pid, out var pn) ? pn : "";
            outLines.Add($"{tx},{ty},{aid},{Esc(aname)},{pid},{Esc(pname)}");
        }
        File.WriteAllLines(csvPath, outLines);
        Console.WriteLine("[ok] Enriched areas.csv with names and parents");
    }

    private static string? DeduceDbcDirFromWdt(string wdtPath)
    {
        var dir = new DirectoryInfo(Path.GetDirectoryName(wdtPath) ?? ".");
        for (var cur = dir; cur != null; cur = cur.Parent)
        {
            var dbc = Path.Combine(cur.FullName, "DBFilesClient");
            if (Directory.Exists(dbc)) return dbc;
        }
        return null;
    }

    private static string? GuessBuildFromPath(string path)
    {
        var s = path.ToLowerInvariant();
        if (s.Contains("3.3.5") || s.Contains("12340") || s.Contains("335")) return "3.3.5.12340";
        if (s.Contains("0.5.3") || s.Contains("053") || s.Contains("3368")) return "0.5.3.3368";
        if (s.Contains("0.6.0") || s.Contains("060") || s.Contains("3592")) return "0.6.0.3592";
        return null;
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
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
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
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
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
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
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
    
    static void AnalyzeLayersBySize(FileInfo wdtFile, FileInfo outputFile, int layerSize)
    {
        Console.WriteLine($"=== WoW Data Plot - Layer Analysis (By Size) ===");
        Console.WriteLine($"Input: {wdtFile.FullName}");
        Console.WriteLine($"Output: {outputFile.FullName}");
        Console.WriteLine($"Layer size: {layerSize} UniqueIDs per layer");
        
        try
        {
            // Extract placement data
            var progress = new Progress<string>(msg => Console.WriteLine($"[INFO] {msg}"));
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
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
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
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
    
    static async Task VisualizeComplete(FileInfo wdtFile, DirectoryInfo outputDir, int gapThreshold, int tileSize, int mapSize, float tileMarkerSize, float mapMarkerSize, int mapMaxLayers, DirectoryInfo? externalMinimapDir, FileInfo? globalRangeFile)
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
        Console.WriteLine($"External minimap dir: {(externalMinimapDir != null ? externalMinimapDir.FullName : "none (auto-detect)")}");
        Console.WriteLine($"Global range file: {(globalRangeFile != null ? globalRangeFile.FullName : "none (using local ranges)")}");
        Console.WriteLine();
        
        // Load global ranges if provided
        GlobalRangeResult? globalRanges = null;
        if (globalRangeFile != null && globalRangeFile.Exists)
        {
            Console.WriteLine("Loading global UniqueID ranges...");
            globalRanges = GlobalRangeAnalyzer.LoadFromJson(globalRangeFile.FullName);
            Console.WriteLine($"  Global range: {globalRanges.GlobalMinUniqueId:N0} - {globalRanges.GlobalMaxUniqueId:N0}");
            Console.WriteLine($"  Covering {globalRanges.Maps.Count} maps, {globalRanges.TotalPlacementCount:N0} placements");
            Console.WriteLine();
        }
        
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
                
                // If external minimap directory is provided, copy PNGs directly
                if (externalMinimapDir != null && externalMinimapDir.Exists)
                {
                    Console.WriteLine($"  Using external minimap directory: {externalMinimapDir.FullName}");
                    
                    // Try multiple naming patterns:
                    // 1. {mapName}*.png (e.g., development_32_32.png)
                    // 2. map*.png (e.g., map32_32.png)
                    // 3. *.png (all PNGs - will rename to standard format)
                    var patterns = new[] { $"{mapName}*.png", "map*.png", "*.png" };
                    
                    foreach (var pattern in patterns)
                    {
                        var pngFiles = Directory.GetFiles(externalMinimapDir.FullName, pattern, SearchOption.AllDirectories);
                        
                        if (pngFiles.Length > 0)
                        {
                            Console.WriteLine($"  Found {pngFiles.Length} files matching pattern '{pattern}'");
                            
                            foreach (var pngFile in pngFiles)
                            {
                                var fileName = Path.GetFileName(pngFile);
                                
                                // Try to extract tile coords from filename (supports: map32_32.png, development_32_32.png, 32_32.png)
                                var destFileName = fileName;
                                if (!fileName.StartsWith(mapName))
                                {
                                    // Rename to standard format: {mapName}_{Y}_{X}.png
                                    // Extract coordinates if possible
                                    var parts = Path.GetFileNameWithoutExtension(fileName).Split('_');
                                    if (parts.Length >= 2 && int.TryParse(parts[^2], out _) && int.TryParse(parts[^1], out _))
                                    {
                                        destFileName = $"{mapName}_{parts[^2]}_{parts[^1]}.png";
                                    }
                                }
                                
                                var destPath = Path.Combine(minimapOutputDir, destFileName);
                                File.Copy(pngFile, destPath, overwrite: true);
                                tilesCopied++;
                            }
                            break; // Stop after first successful pattern
                        }
                    }
                    
                    if (tilesCopied > 0)
                    {
                        minimapDir = minimapOutputDir;
                        Console.WriteLine($"✓ Copied {tilesCopied} PNG minimap tiles");
                    }
                    else
                    {
                        Console.WriteLine($"⚠ No PNG files found in external minimap directory");
                    }
                }
                else
                {
                    // Auto-detect and extract from BLP
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
            var records = UnifiedPlacementExtractor.Extract(wdtFile.FullName, progress);
            
            if (records.Count == 0)
            {
                Console.WriteLine("[ERROR] No placements found!");
                return;
            }
            
            Console.WriteLine($"✓ Extracted {records.Count} placements");
            Console.WriteLine();
            
            // STEP 1b: Generate white placeholder tiles for tiles with placements but no minimap
            if (minimapResult.MinimapDir != null)
            {
                Console.WriteLine("═══ STEP 1b: Generating Placeholder Tiles ═══");
                var tilesWithPlacements = records.Where(r => r.TileX >= 0 && r.TileY >= 0)
                    .Select(r => (r.TileX, r.TileY))
                    .Distinct()
                    .ToList();
                
                int placeholdersGenerated = 0;
                foreach (var (tileX, tileY) in tilesWithPlacements)
                {
                    var minimapPath = Path.Combine(minimapResult.MinimapDir, $"{mapName}_{tileX}_{tileY}.png");
                    if (!File.Exists(minimapPath))
                    {
                        // Create white 512x512 placeholder
                        using var image = new Image<Rgb24>(512, 512);
                        image.Mutate(ctx => ctx.BackgroundColor(SixLabors.ImageSharp.Color.White));
                        await image.SaveAsPngAsync(minimapPath);
                        placeholdersGenerated++;
                    }
                }
                
                if (placeholdersGenerated > 0)
                {
                    Console.WriteLine($"✓ Generated {placeholdersGenerated} white placeholder tiles");
                }
                else
                {
                    Console.WriteLine($"  All tiles have minimaps (no placeholders needed)");
                }
                Console.WriteLine();
            }
            
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
            uint visualGlobalMinId;
            uint visualGlobalMaxId;
            uint visualGlobalRange;
            
            if (globalRanges != null)
            {
                // Use cross-map global ranges for TRUE temporal positioning
                visualGlobalMinId = globalRanges.GlobalMinUniqueId;
                visualGlobalMaxId = globalRanges.GlobalMaxUniqueId;
                visualGlobalRange = Math.Max(visualGlobalMaxId - visualGlobalMinId, 1);
                Console.WriteLine($"  Using GLOBAL ranges: {visualGlobalMinId:N0} - {visualGlobalMaxId:N0} (cross-map normalization)");
            }
            else
            {
                // Use local map ranges only
                visualGlobalMinId = records.Min(r => r.UniqueId);
                visualGlobalMaxId = records.Max(r => r.UniqueId);
                visualGlobalRange = Math.Max(visualGlobalMaxId - visualGlobalMinId, 1);
                Console.WriteLine($"  Using LOCAL ranges: {visualGlobalMinId:N0} - {visualGlobalMaxId:N0} (map-only normalization)");
            }
            
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
            
            // Reuse the same global ranges from STEP 3 for consistency
            uint globalMinId = visualGlobalMinId;
            uint globalMaxId = visualGlobalMaxId;
            uint globalRange = visualGlobalRange;
            
            // Divide into 100 color buckets for smooth gradient across entire map
            int mapBuckets = 100;
            
            if (globalRanges != null)
            {
                Console.WriteLine($"  Plotting GLOBAL heatmap with {mapBuckets} color gradients (UniqueID {globalMinId}-{globalMaxId})...");
                Console.WriteLine($"  This map's actual range: {records.Min(r => r.UniqueId):N0} - {records.Max(r => r.UniqueId):N0}");
            }
            else
            {
                Console.WriteLine($"  Plotting LOCAL heatmap with {mapBuckets} color gradients (UniqueID {globalMinId}-{globalMaxId})...");
            }
            
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
                
                // Rollback Controls
                writer.WriteLine("    <h2>🎮 WoWRollback Controls</h2>");
                writer.WriteLine("    <div class='stats' style='padding: 20px;'>");
                writer.WriteLine("        <p style='color: #d4d4d4; margin-bottom: 15px;'>Roll back to a specific point in development. Move the slider to keep only content added up to that UniqueID:</p>");
                writer.WriteLine("        ");
                writer.WriteLine("        <div style='margin: 20px 0;'>");
                writer.WriteLine($"            <label style='color: #4ec9b0; font-weight: bold; display: block; margin-bottom: 10px;'>Development Rollback Point</label>");
                writer.WriteLine("            ");
                writer.WriteLine("            <!-- Gradient bar background -->");
                writer.WriteLine("            <div style='position: relative; background: linear-gradient(to right, #0000FF 0%, #00FFFF 25%, #00FF00 50%, #FFFF00 75%, #FF0000 100%); height: 40px; border-radius: 5px; border: 2px solid #3c3c3c; margin-bottom: 10px;'>");
                writer.WriteLine($"                <input type='range' id='maxSlider' min='{globalMinId}' max='{globalMaxId}' value='{globalMaxId}' style='position: absolute; width: 100%; height: 100%; opacity: 0.01; cursor: pointer; margin: 0;' oninput='updateRollbackDisplay()'>");
                writer.WriteLine("                <div id='sliderMarker' style='position: absolute; right: 0; top: -2px; bottom: -2px; width: 4px; background: white; box-shadow: 0 0 5px rgba(0,0,0,0.5); pointer-events: none;'></div>");
                writer.WriteLine("            </div>");
                writer.WriteLine("            ");
                writer.WriteLine($"            <div id='rollbackDisplay' style='color: #569cd6; font-size: 1.2em; font-weight: bold;'>Keep content up to UniqueID: {globalMaxId:N0} <span style='color: #858585; font-size: 0.8em;'>(All content)</span></div>");
                writer.WriteLine("        </div>");
                writer.WriteLine("        ");
                writer.WriteLine("        <div style='margin: 20px 0; display: flex; gap: 10px;'>");
                writer.WriteLine("            <button onclick='savePreferences()' style='padding: 10px 20px; background: #0e639c; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em;'>💾 Save Settings</button>");
                writer.WriteLine("            <button onclick='loadPreferences()' style='padding: 10px 20px; background: #0e639c; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em;'>📂 Load Settings</button>");
                writer.WriteLine($"            <button onclick='recompileMap()' style='padding: 10px 20px; background: #2d7d2d; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 1em; font-weight: bold;'>⚙️ Recompile Map</button>");
                writer.WriteLine("        </div>");
                writer.WriteLine("        ");
                writer.WriteLine("        <p style='color: #858585; font-size: 0.9em; margin-top: 15px;'>💡 <strong>Tip:</strong> Click any tile below for per-tile layer control.</p>");
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
                    
                    // Calculate tile grid bounds from BOTH placement tiles AND minimap files
                    var minimapDirPath = Path.Combine(outputDir.FullName, "minimaps");
                    var allCoords = new List<(int x, int y)>();
                    
                    // Add coordinates from tiles with placement data
                    foreach (var tile in tileLayerInfos)
                    {
                        allCoords.Add((tile.TileX, tile.TileY));
                    }
                    
                    // Add coordinates from minimap files
                    if (Directory.Exists(minimapDirPath))
                    {
                        var minimapFilesForBounds = Directory.GetFiles(minimapDirPath, $"{mapName}_*.png");
                        foreach (var file in minimapFilesForBounds)
                        {
                            var name = Path.GetFileNameWithoutExtension(file);
                            var parts = name.Split('_');
                            if (parts.Length >= 3 && int.TryParse(parts[^2], out int x) && int.TryParse(parts[^1], out int y))
                            {
                                allCoords.Add((x, y));
                            }
                        }
                    }
                    
                    // Calculate bounds from ALL coordinates (use 0,0 as origin for absolute positioning)
                    int minX = Math.Min(0, allCoords.Min(c => c.x));
                    int maxX = allCoords.Max(c => c.x);
                    int minY = Math.Min(0, allCoords.Min(c => c.y));
                    int maxY = allCoords.Max(c => c.y);
                    
                    // Build lookup maps for tiles with data and minimap files
                    var tileLookup = tileLayerInfos.ToDictionary(t => (t.TileX, t.TileY));
                    var minimapLookup = new Dictionary<(int x, int y), string>();
                    
                    var minimapFiles = Directory.GetFiles(minimapDirPath, $"{mapName}_*.png");
                    foreach (var file in minimapFiles)
                    {
                        var name = Path.GetFileNameWithoutExtension(file);
                        var parts = name.Split('_');
                        if (parts.Length >= 3 && int.TryParse(parts[^2], out int x) && int.TryParse(parts[^1], out int y))
                        {
                            minimapLookup[(x, y)] = Path.GetFileName(file);
                        }
                    }
                    
                    // Create debug log file
                    var debugLogPath = Path.Combine(outputDir.FullName, "tile_browser_debug.log");
                    using (var debugLog = new StreamWriter(debugLogPath))
                    {
                        debugLog.WriteLine("=== TILE BROWSER DEBUG LOG ===");
                        debugLog.WriteLine($"Timestamp: {DateTime.Now}");
                        debugLog.WriteLine();
                        debugLog.WriteLine($"Minimap directory: {minimapDirPath}");
                        debugLog.WriteLine($"Tiles with placement data: {tileLayerInfos.Count}");
                        debugLog.WriteLine($"allCoords has {allCoords.Count} entries");
                        debugLog.WriteLine($"Sample allCoords: {string.Join(", ", allCoords.Take(10))}");
                        debugLog.WriteLine($"Grid bounds: X=[{minX},{maxX}], Y=[{minY},{maxY}]");
                        debugLog.WriteLine();
                        debugLog.WriteLine($"Found {minimapFiles.Length} minimap files");
                        debugLog.WriteLine($"Loaded {minimapLookup.Count} minimap entries into lookup");
                        debugLog.WriteLine($"Sample lookup entries: {string.Join(", ", minimapLookup.Keys.Take(10))}");
                        debugLog.WriteLine();
                        
                        // Log first 20 grid cells to see what's rendered
                        debugLog.WriteLine("First 20 grid cells:");
                        int cellCount = 0;
                        for (int y = minY; y <= maxY && cellCount < 20; y++)
                        {
                            for (int x = minX; x <= maxX && cellCount < 20; x++)
                            {
                                bool hasTileData = tileLookup.ContainsKey((x, y));
                                bool hasMinimapFile = minimapLookup.ContainsKey((x, y));
                                debugLog.WriteLine($"  [{x},{y}]: TileData={hasTileData}, Minimap={hasMinimapFile}" + 
                                    (hasMinimapFile ? $" (file: {minimapLookup[(x, y)]})" : ""));
                                cellCount++;
                            }
                        }
                    } // Close using block for debugLog
                    
                    int gridWidth = maxX - minX + 1;
                    int gridHeight = maxY - minY + 1;
                    
                    writer.WriteLine($"        <div style='display: grid; grid-template-columns: repeat({gridWidth}, 64px); gap: 2px; max-height: 70vh; overflow: auto; padding: 5px; background: #1e1e1e; border-radius: 3px; justify-content: center;'>");
                    
                    // Render COMPLETE grid starting from origin (proper absolute positioning)
                    for (int y = minY; y <= maxY; y++)
                    {
                        for (int x = minX; x <= maxX; x++)
                        {
                            bool hasTileData = tileLookup.TryGetValue((x, y), out var tileInfo);
                            bool hasMinimapFile = minimapLookup.TryGetValue((x, y), out var minimapFile);
                            
                            if (hasTileData && hasMinimapFile)
                            {
                                // Tile with placement data AND minimap - clickable
                                var totalPlacements = tileInfo.Layers.Sum(l => l.PlacementCount);
                                writer.WriteLine($"            <a href='tiles/{mapName}_{x}_{y}.html' title='Tile [{x},{y}] - {tileInfo.Layers.Count} layers, {totalPlacements} placements' style='display: block; width: 64px; height: 64px; border: 2px solid #569cd6; border-radius: 2px; overflow: hidden; transition: all 0.2s;' onmouseover='this.style.borderColor=\"#4ec9b0\"; this.style.transform=\"scale(1.05)\"; this.style.zIndex=\"10\";' onmouseout='this.style.borderColor=\"#569cd6\"; this.style.transform=\"scale(1)\"; this.style.zIndex=\"1\";'>");
                                writer.WriteLine($"                <img src='minimaps/{minimapFile}' style='width: 100%; height: 100%; object-fit: cover;' alt='[{x},{y}]' />");
                                writer.WriteLine($"            </a>");
                            }
                            else if (hasMinimapFile)
                            {
                                // Minimap exists but no placement data - non-clickable, dimmed
                                writer.WriteLine($"            <div title='Tile [{x},{y}] - No placement data' style='width: 64px; height: 64px; border: 1px solid #3c3c3c; border-radius: 2px; overflow: hidden; opacity: 0.5;'>");
                                writer.WriteLine($"                <img src='minimaps/{minimapFile}' style='width: 100%; height: 100%; object-fit: cover;' alt='[{x},{y}]' />");
                                writer.WriteLine($"            </div>");
                            }
                            else if (hasTileData)
                            {
                                // Data exists but no minimap - show placeholder
                                writer.WriteLine($"            <div title='Tile [{x},{y}] - No minimap' style='width: 64px; height: 64px; border: 1px solid #3c3c3c; background: #2d2d30; border-radius: 2px;'></div>");
                            }
                            else
                            {
                                // Empty cell - no data, no minimap - dark background to show grid
                                writer.WriteLine($"            <div style='width: 64px; height: 64px; background: #1e1e1e;'></div>");
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
                
                // JavaScript for WoWRollback controls
                writer.WriteLine("    <script>");
                writer.WriteLine("        let globalSettings = {");
                writer.WriteLine($"            maxUniqueId: {globalMaxId},");
                writer.WriteLine("            perTileSettings: {}");
                writer.WriteLine("        };");
                writer.WriteLine("        ");
                writer.WriteLine($"        const GLOBAL_MIN = {globalMinId};");
                writer.WriteLine($"        const GLOBAL_MAX = {globalMaxId};");
                writer.WriteLine("        ");
                writer.WriteLine("        function updateRollbackDisplay() {");
                writer.WriteLine("            const maxSlider = document.getElementById('maxSlider');");
                writer.WriteLine("            const display = document.getElementById('rollbackDisplay');");
                writer.WriteLine("            const marker = document.getElementById('sliderMarker');");
                writer.WriteLine("            ");
                writer.WriteLine("            const maxValue = parseInt(maxSlider.value);");
                writer.WriteLine("            globalSettings.maxUniqueId = maxValue;");
                writer.WriteLine("            ");
                writer.WriteLine("            // Calculate position percentage");
                writer.WriteLine("            const percentage = ((maxValue - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)) * 100;");
                writer.WriteLine("            marker.style.left = percentage + '%';");
                writer.WriteLine("            ");
                writer.WriteLine("            // Update text with helpful context");
                writer.WriteLine("            let suffix = '';");
                writer.WriteLine("            if (maxValue >= GLOBAL_MAX) {");
                writer.WriteLine("                suffix = '<span style=\"color: #858585; font-size: 0.8em;\">(All content)</span>';");
                writer.WriteLine("            } else if (percentage < 25) {");
                writer.WriteLine("                suffix = '<span style=\"color: #569cd6; font-size: 0.8em;\">(Early development)</span>';");
                writer.WriteLine("            } else if (percentage < 50) {");
                writer.WriteLine("                suffix = '<span style=\"color: #00FFFF; font-size: 0.8em;\">(Mid development)</span>';");
                writer.WriteLine("            } else if (percentage < 75) {");
                writer.WriteLine("                suffix = '<span style=\"color: #FFFF00; font-size: 0.8em;\">(Late development)</span>';");
                writer.WriteLine("            } else {");
                writer.WriteLine("                suffix = '<span style=\"color: #FF9900; font-size: 0.8em;\">(Nearly complete)</span>';");
                writer.WriteLine("            }");
                writer.WriteLine("            ");
                writer.WriteLine("            display.innerHTML = `Keep content up to UniqueID: ${maxValue.toLocaleString()} ${suffix}`;");
                writer.WriteLine("        }");
                writer.WriteLine("        ");
                writer.WriteLine("        function savePreferences() {");
                writer.WriteLine("            const dataStr = JSON.stringify(globalSettings, null, 2);");
                writer.WriteLine("            const dataBlob = new Blob([dataStr], { type: 'application/json' });");
                writer.WriteLine($"            const url = URL.createObjectURL(dataBlob);");
                writer.WriteLine("            const link = document.createElement('a');");
                writer.WriteLine($"            link.href = url;");
                writer.WriteLine($"            link.download = '{mapName}_rollback_settings.json';");
                writer.WriteLine("            link.click();");
                writer.WriteLine("            URL.revokeObjectURL(url);");
                writer.WriteLine("            alert('✅ Preferences saved to JSON file!');");
                writer.WriteLine("        }");
                writer.WriteLine("        ");
                writer.WriteLine("        function loadPreferences() {");
                writer.WriteLine("            const input = document.createElement('input');");
                writer.WriteLine("            input.type = 'file';");
                writer.WriteLine("            input.accept = '.json';");
                writer.WriteLine("            input.onchange = (e) => {");
                writer.WriteLine("                const file = e.target.files[0];");
                writer.WriteLine("                const reader = new FileReader();");
                writer.WriteLine("                reader.onload = (ev) => {");
                writer.WriteLine("                    try {");
                writer.WriteLine("                        globalSettings = JSON.parse(ev.target.result);");
                writer.WriteLine("                        document.getElementById('maxSlider').value = globalSettings.maxUniqueId;");
                writer.WriteLine("                        updateRollbackDisplay();");
                writer.WriteLine("                        alert('✅ Settings loaded!');");
                writer.WriteLine("                    } catch (err) {");
                writer.WriteLine("                        alert('❌ Error loading preferences: ' + err.message);");
                writer.WriteLine("                    }");
                writer.WriteLine("                };");
                writer.WriteLine("                reader.readAsText(file);");
                writer.WriteLine("            };");
                writer.WriteLine("            input.click();");
                writer.WriteLine("        }");
                writer.WriteLine("        ");
                writer.WriteLine("        function recompileMap() {");
                writer.WriteLine("            const settingsJson = JSON.stringify(globalSettings, null, 2);");
                writer.WriteLine("            alert('⚙️ Recompile feature coming soon!\\n\\nYour settings:\\n' + settingsJson + '\\n\\nThis will generate filtered ADT/WDT files based on your UniqueID range selection.');");
                writer.WriteLine("            // TODO: Call WoWRollback CLI tool with settings");
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
    
    static void AnalyzeClient(DirectoryInfo clientRoot, FileInfo outputFile)
    {
        Console.WriteLine("=== WoW Data Archaeology - Global UniqueID Analysis ===");
        Console.WriteLine($"Client Root: {clientRoot.FullName}");
        Console.WriteLine($"Output: {outputFile.FullName}");
        Console.WriteLine();
        
        try
        {
            var progress = new Progress<string>(msg => Console.WriteLine(msg));
            
            Console.WriteLine("═══ Scanning All Maps ═══");
            var result = GlobalRangeAnalyzer.AnalyzeClient(clientRoot.FullName, progress);
            
            Console.WriteLine();
            Console.WriteLine("═══ Global Statistics ═══");
            Console.WriteLine($"Total maps analyzed: {result.Maps.Count}");
            Console.WriteLine($"Total placements: {result.TotalPlacementCount:N0}");
            Console.WriteLine($"Global UniqueID range: {result.GlobalMinUniqueId:N0} - {result.GlobalMaxUniqueId:N0}");
            Console.WriteLine($"Range span: {result.GlobalMaxUniqueId - result.GlobalMinUniqueId:N0}");
            Console.WriteLine();
            
            Console.WriteLine("═══ Per-Map Breakdown ═══");
            foreach (var map in result.Maps.OrderBy(m => m.Value.MinUniqueId))
            {
                var info = map.Value;
                Console.WriteLine($"{map.Key,-20} | {info.MinUniqueId,8:N0} - {info.MaxUniqueId,8:N0} | {info.PlacementCount,7:N0} placements ({info.M2Count,6:N0} M2, {info.WmoCount,5:N0} WMO)");
            }
            Console.WriteLine();
            
            // Save to JSON
            Console.WriteLine("═══ Saving Results ═══");
            GlobalRangeAnalyzer.SaveToJson(result, outputFile.FullName);
            Console.WriteLine($"✓ Saved: {outputFile.FullName}");
            Console.WriteLine();
            
            Console.WriteLine("✓ Analysis Complete!");
            Console.WriteLine();
            Console.WriteLine("Use this file with the visualize command:");
            Console.WriteLine($"  --global-range-file \"{outputFile.FullName}\"");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error analyzing client: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    static void RollbackWdt(FileInfo inputFile, FileInfo outputFile, uint maxUniqueId, float buryDepth)
    {
        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine("          🎮 WoWRollback - CORE FEATURE");
        Console.WriteLine("══════════════════════════════════════════");
        Console.WriteLine($"Input WDT:      {inputFile.FullName}");
        Console.WriteLine($"Output WDT:     {outputFile.FullName}");
        Console.WriteLine($"Max UniqueID:   {maxUniqueId:N0}");
        Console.WriteLine($"Bury Depth:     {buryDepth:F1}");
        Console.WriteLine();
        
        try
        {
            // Use existing WdtAlpha class to properly parse the structure
            Console.WriteLine("═══ STEP 1: Loading Alpha WDT ===");
            var wdt = new WdtAlpha(inputFile.FullName);
            var existingAdts = wdt.GetExistingAdtsNumbers();
            Console.WriteLine($"✓ Loaded WDT with {existingAdts.Count} ADT tiles");
            Console.WriteLine();
            
            // Read raw file bytes for direct modification
            byte[] wdtBytes = File.ReadAllBytes(inputFile.FullName);
            
            Console.WriteLine("═══ STEP 2: Processing All ADT Tiles ===");
            int totalPlacements = 0;
            int buriedPlacements = 0;
            int keptPlacements = 0;
            int tilesProcessed = 0;
            
            var adtOffsets = wdt.GetAdtOffsetsInMain();
            
            // Process each ADT tile
            foreach (var adtNum in existingAdts)
            {
                int adtOffset = adtOffsets[adtNum];
                if (adtOffset == 0) continue;
                
                int tileX = adtNum % 64;
                int tileY = adtNum / 64;
                
                // Parse the ADT at this offset
                var adt = new AdtAlpha(inputFile.FullName, adtOffset, adtNum);
                var mddf = adt.GetMddf();
                var modf = adt.GetModf();
                
                // Process MDDF (M2) placements - 36 bytes per entry
                // +0x00: nameId (4 bytes)
                // +0x04: uniqueId (4 bytes)
                // +0x08: position X (4 bytes)
                // +0x0C: position Z (4 bytes) <- MODIFY THIS
                // +0x10: position Y (4 bytes)
                const int mddfEntrySize = 36;
                for (int offset = 0; offset + mddfEntrySize <= mddf.Data.Length; offset += mddfEntrySize)
                {
                    uint uniqueId = BitConverter.ToUInt32(mddf.Data, offset + 4);
                    totalPlacements++;
                    
                    if (uniqueId > maxUniqueId)
                    {
                        // Bury it! Modify Z coordinate (offset +0x0C)
                        byte[] newZ = BitConverter.GetBytes(buryDepth);
                        Array.Copy(newZ, 0, mddf.Data, offset + 12, 4);
                        buriedPlacements++;
                    }
                    else
                    {
                        keptPlacements++;
                    }
                }
                
                // Process MODF (WMO) placements - 64 bytes per entry
                // +0x00: nameId (4 bytes)
                // +0x04: uniqueId (4 bytes)
                // +0x08: position X (4 bytes)
                // +0x0C: position Z (4 bytes) <- MODIFY THIS
                // +0x10: position Y (4 bytes)
                const int modfEntrySize = 64;
                for (int offset = 0; offset + modfEntrySize <= modf.Data.Length; offset += modfEntrySize)
                {
                    uint uniqueId = BitConverter.ToUInt32(modf.Data, offset + 4);
                    totalPlacements++;
                    
                    if (uniqueId > maxUniqueId)
                    {
                        // Bury it! Modify Z coordinate (offset +0x0C)
                        byte[] newZ = BitConverter.GetBytes(buryDepth);
                        Array.Copy(newZ, 0, modf.Data, offset + 12, 4);
                        buriedPlacements++;
                    }
                    else
                    {
                        keptPlacements++;
                    }
                }
                
                // Copy modified chunk data back to wdtBytes
                if (mddf.Data.Length > 0)
                {
                    int mddfFileOffset = adt.GetMddfDataOffset();
                    Array.Copy(mddf.Data, 0, wdtBytes, mddfFileOffset, mddf.Data.Length);
                }
                if (modf.Data.Length > 0)
                {
                    int modfFileOffset = adt.GetModfDataOffset();
                    Array.Copy(modf.Data, 0, wdtBytes, modfFileOffset, modf.Data.Length);
                }
                
                tilesProcessed++;
                if (tilesProcessed % 50 == 0)
                {
                    Console.WriteLine($"  Processed {tilesProcessed}/{existingAdts.Count} tiles...");
                }
            }
            
            Console.WriteLine($"✓ Processed {tilesProcessed} ADT tiles");
            Console.WriteLine();
            Console.WriteLine($"Total Placements:  {totalPlacements:N0}");
            Console.WriteLine($"  ✅ Kept:         {keptPlacements:N0} (UniqueID <= {maxUniqueId:N0})");
            Console.WriteLine($"  🪦 Buried:       {buriedPlacements:N0} (UniqueID > {maxUniqueId:N0})");
            Console.WriteLine();
            
            // Write modified WDT
            Console.WriteLine("═══ STEP 3: Writing Output WDT ═══");
            Directory.CreateDirectory(Path.GetDirectoryName(outputFile.FullName)!);
            File.WriteAllBytes(outputFile.FullName, wdtBytes);
            
            Console.WriteLine($"✓ Saved: {outputFile.FullName}");
            
            // Generate MD5 checksum for minimap compatibility
            Console.WriteLine();
            Console.WriteLine("═══ STEP 4: Generating MD5 Checksum ═══");
            using (var md5 = System.Security.Cryptography.MD5.Create())
            {
                var hash = md5.ComputeHash(wdtBytes);
                var hashString = BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
                
                var md5FileName = Path.GetFileNameWithoutExtension(outputFile.FullName) + ".md5";
                var md5FilePath = Path.Combine(Path.GetDirectoryName(outputFile.FullName)!, md5FileName);
                File.WriteAllText(md5FilePath, hashString);
                
                Console.WriteLine($"✓ MD5 Hash: {hashString}");
                Console.WriteLine($"✓ Saved: {md5FilePath}");
            }
            Console.WriteLine();
            Console.WriteLine("══════════════════════════════════════════");
            Console.WriteLine("✅ ROLLBACK COMPLETE!");
            Console.WriteLine("══════════════════════════════════════════");
            Console.WriteLine($"Your filtered map is ready to use.");
            Console.WriteLine($"Placements with UniqueID > {maxUniqueId:N0} have been buried at Z={buryDepth:F1}");
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error during rollback: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
        }
    }
    
    static List<int> FindAllChunks(byte[] data, string chunkName)
    {
        var positions = new List<int>();
        byte[] pattern = System.Text.Encoding.ASCII.GetBytes(chunkName);
        
        for (int i = 0; i < data.Length - pattern.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < pattern.Length; j++)
            {
                if (data[i + j] != pattern[j])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                positions.Add(i);
                i += 8; // Skip past chunk header to avoid false positives
            }
        }
        return positions;
    }
}
