using System;
using System.IO;
using System.Text;
using System.Text.Json;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Generates a simple standalone viewer for single-map analysis results.
/// </summary>
public sealed class SimpleViewerGenerator
{
    public ViewerGenerationResult Generate(string analysisOutputDir, string mapName)
    {
        try
        {
            var viewerDir = Path.Combine(analysisOutputDir, "viewer");
            Directory.CreateDirectory(viewerDir);

            // Copy minimaps if they exist
            var minimapSourceDir = Path.Combine(analysisOutputDir, "minimaps");
            if (Directory.Exists(minimapSourceDir))
            {
                var minimapDestDir = Path.Combine(viewerDir, "minimaps");
                Directory.CreateDirectory(minimapDestDir);
                
                foreach (var file in Directory.GetFiles(minimapSourceDir, "*.png"))
                {
                    var destFile = Path.Combine(minimapDestDir, Path.GetFileName(file));
                    File.Copy(file, destFile, overwrite: true);
                }
            }

            // Copy analysis JSONs
            var dataDir = Path.Combine(viewerDir, "data");
            Directory.CreateDirectory(dataDir);

            CopyIfExists(Path.Combine(analysisOutputDir, $"{mapName}_layers.json"), 
                        Path.Combine(dataDir, "layers.json"));
            CopyIfExists(Path.Combine(analysisOutputDir, $"{mapName}_spatial_clusters.json"), 
                        Path.Combine(dataDir, "clusters.json"));
            CopyIfExists(Path.Combine(analysisOutputDir, $"{mapName}_patterns.json"), 
                        Path.Combine(dataDir, "patterns.json"));
            CopyIfExists(Path.Combine(analysisOutputDir, $"{mapName}_placements.csv"), 
                        Path.Combine(dataDir, "placements.csv"));

            // Generate viewer config
            var config = new
            {
                mapName = mapName,
                hasMinimaps = Directory.Exists(minimapSourceDir),
                hasLayers = File.Exists(Path.Combine(dataDir, "layers.json")),
                hasClusters = File.Exists(Path.Combine(dataDir, "clusters.json")),
                hasPatterns = File.Exists(Path.Combine(dataDir, "patterns.json"))
            };

            File.WriteAllText(
                Path.Combine(viewerDir, "config.json"),
                JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true }));

            // Generate index.html
            GenerateIndexHtml(viewerDir, mapName);

            // Generate viewer.js
            GenerateViewerScript(viewerDir);

            // Generate styles.css
            GenerateStyles(viewerDir);

            return new ViewerGenerationResult
            {
                Success = true,
                ViewerPath = viewerDir
            };
        }
        catch (Exception ex)
        {
            return new ViewerGenerationResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    private static void CopyIfExists(string source, string dest)
    {
        if (File.Exists(source))
        {
            File.Copy(source, dest, overwrite: true);
        }
    }

    private static void GenerateIndexHtml(string viewerDir, string mapName)
    {
        var html = $@"<!DOCTYPE html>
<html lang=""en"">
<head>
    <meta charset=""UTF-8"">
    <meta name=""viewport"" content=""width=device-width, initial-scale=1.0"">
    <title>{mapName} - WoW Archive Viewer</title>
    <link rel=""stylesheet"" href=""styles.css"">
</head>
<body>
    <div class=""header"">
        <h1>🏺 {mapName} Archaeological Analysis</h1>
        <p>Temporal layers, spatial clusters, and placement patterns</p>
    </div>

    <div class=""container"">
        <div class=""sidebar"">
            <h2>Layers</h2>
            <div id=""layers-list""></div>
            
            <h2>Clusters</h2>
            <div id=""clusters-summary""></div>
            
            <h2>Patterns</h2>
            <div id=""patterns-summary""></div>
        </div>

        <div class=""main-view"">
            <div id=""map-viewer"">
                <canvas id=""map-canvas""></canvas>
            </div>
            <div id=""details-panel""></div>
        </div>
    </div>

    <script src=""viewer.js""></script>
</body>
</html>";

        File.WriteAllText(Path.Combine(viewerDir, "index.html"), html);
    }

    private static void GenerateViewerScript(string viewerDir)
    {
        var js = @"// Simple viewer for archaeological analysis
let config = null;
let layers = null;
let clusters = null;
let patterns = null;

async function init() {
    try {
        config = await fetch('config.json').then(r => r.json());
        
        if (config.hasLayers) {
            layers = await fetch('data/layers.json').then(r => r.json());
            renderLayers();
        }
        
        if (config.hasClusters) {
            clusters = await fetch('data/clusters.json').then(r => r.json());
            renderClustersSummary();
        }
        
        if (config.hasPatterns) {
            patterns = await fetch('data/patterns.json').then(r => r.json());
            renderPatternsSummary();
        }
    } catch (err) {
        console.error('Failed to load data:', err);
    }
}

function renderLayers() {
    const container = document.getElementById('layers-list');
    if (!layers || !layers.globalLayers) return;
    
    const html = layers.globalLayers.map((layer, idx) => `
        <div class=""layer-item"">
            <strong>Layer ${idx + 1}</strong><br>
            Range: ${layer.idRangeStart} - ${layer.idRangeEnd}<br>
            Objects: ${layer.objectCount}
        </div>
    `).join('');
    
    container.innerHTML = html;
}

function renderClustersSummary() {
    const container = document.getElementById('clusters-summary');
    if (!clusters || !clusters.tiles) return;
    
    const totalClusters = clusters.tiles.reduce((sum, t) => sum + t.clusters.length, 0);
    const stampClusters = clusters.tiles
        .flatMap(t => t.clusters)
        .filter(c => c.isPlacementStamp)
        .length;
    
    container.innerHTML = `
        <p>Total Clusters: ${totalClusters}</p>
        <p>Placement Stamps: ${stampClusters}</p>
        <p>Tiles: ${clusters.tiles.length}</p>
    `;
}

function renderPatternsSummary() {
    const container = document.getElementById('patterns-summary');
    if (!patterns) return;
    
    container.innerHTML = `
        <p>Unique Patterns: ${patterns.totalPatterns || 0}</p>
        <p>(Recurring prefabs/brushes detected)</p>
    `;
}

init();
";

        File.WriteAllText(Path.Combine(viewerDir, "viewer.js"), js);
    }

    private static void GenerateStyles(string viewerDir)
    {
        var css = @"* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #1a1a1a;
    color: #e0e0e0;
}

.header {
    background: #2a2a2a;
    padding: 20px;
    border-bottom: 2px solid #ff6b35;
    text-align: center;
}

.header h1 {
    color: #ff6b35;
    margin-bottom: 10px;
}

.header p {
    color: #999;
}

.container {
    display: flex;
    height: calc(100vh - 100px);
}

.sidebar {
    width: 300px;
    background: #252525;
    padding: 20px;
    overflow-y: auto;
    border-right: 1px solid #333;
}

.sidebar h2 {
    color: #ff6b35;
    font-size: 18px;
    margin: 20px 0 10px 0;
}

.layer-item {
    background: #2a2a2a;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 4px;
    border-left: 3px solid #ff6b35;
}

.main-view {
    flex: 1;
    display: flex;
    flex-direction: column;
}

#map-viewer {
    flex: 1;
    background: #1a1a1a;
    display: flex;
    align-items: center;
    justify-content: center;
}

#map-canvas {
    border: 1px solid #333;
}

#details-panel {
    height: 200px;
    background: #252525;
    border-top: 1px solid #333;
    padding: 20px;
    overflow-y: auto;
}
";

        File.WriteAllText(Path.Combine(viewerDir, "styles.css"), css);
    }
}

public record ViewerGenerationResult
{
    public required bool Success { get; init; }
    public string? ViewerPath { get; init; }
    public string? ErrorMessage { get; init; }
}
