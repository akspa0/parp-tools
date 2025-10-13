using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Builds terrain_complete overlay JSON files from MCNK terrain CSV data.
/// </summary>
public sealed class TerrainOverlayBuilder
{
    public void BuildTerrainOverlays(
        string terrainCsvPath,
        string mapName,
        string version,
        string viewerRoot)
    {
        if (!File.Exists(terrainCsvPath))
        {
            Console.WriteLine($"[TerrainOverlayBuilder] CSV not found: {terrainCsvPath}");
            return;
        }

        Console.WriteLine($"[TerrainOverlayBuilder] Building terrain overlays from {terrainCsvPath}");

        // Parse CSV
        var records = ParseTerrainCsv(terrainCsvPath);
        Console.WriteLine($"[TerrainOverlayBuilder] Parsed {records.Count} MCNK chunks");

        // Group by tile
        var byTile = records.GroupBy(r => (r.TileRow, r.TileCol)).ToList();
        Console.WriteLine($"[TerrainOverlayBuilder] Grouped into {byTile.Count} tiles");

        // Create output directory
        var overlaysDir = Path.Combine(viewerRoot, "overlays", version, mapName, "terrain_complete");
        Directory.CreateDirectory(overlaysDir);

        int filesGenerated = 0;
        foreach (var tileGroup in byTile)
        {
            var (row, col) = tileGroup.Key;
            var chunks = tileGroup.ToList();

            // Build overlay JSON
            var overlayJson = BuildTileOverlayJson(mapName, version, row, col, chunks);

            // Write to file: tile_{col}_{row}.json
            var outputPath = Path.Combine(overlaysDir, $"tile_{col}_{row}.json");
            File.WriteAllText(outputPath, overlayJson);
            filesGenerated++;
        }

        Console.WriteLine($"[TerrainOverlayBuilder] Generated {filesGenerated} terrain overlay JSON files in {overlaysDir}");
    }

    private string BuildTileOverlayJson(
        string mapName,
        string version,
        int tileRow,
        int tileCol,
        List<TerrainRecord> chunks)
    {
        // Build terrain properties data
        var terrainProperties = chunks.Select(c => new
        {
            chunk_x = c.ChunkX,
            chunk_y = c.ChunkY,
            flags = c.Flags
        }).ToList();

        // Build liquids data
        var liquids = chunks
            .Where(c => c.HasLiquids)
            .Select(c => new
            {
                chunk_x = c.ChunkX,
                chunk_y = c.ChunkY,
                has_liquids = true
            })
            .ToList();

        // Build holes data  
        var holes = chunks
            .Where(c => c.HasHoles)
            .Select(c => new
            {
                chunk_x = c.ChunkX,
                chunk_y = c.ChunkY,
                has_holes = true
            })
            .ToList();

        // Build area IDs data
        var areaIds = chunks.Select(c => new
        {
            chunk_x = c.ChunkX,
            chunk_y = c.ChunkY,
            area_id = c.AreaId
        }).ToList();

        // Build complete overlay structure
        var overlay = new
        {
            map = mapName,
            tile_row = tileRow,
            tile_col = tileCol,
            layers = new[]
            {
                new
                {
                    version,
                    terrain_properties = terrainProperties.Count > 0 ? terrainProperties : null,
                    liquids = liquids.Count > 0 ? liquids : null,
                    holes = holes.Count > 0 ? holes : null,
                    area_ids = areaIds.Count > 0 ? areaIds : null
                }
            }
        };

        return JsonSerializer.Serialize(overlay, new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        });
    }

    private List<TerrainRecord> ParseTerrainCsv(string csvPath)
    {
        var records = new List<TerrainRecord>();
        var lines = File.ReadAllLines(csvPath);

        // Skip header
        for (int i = 1; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var fields = line.Split(',');
            if (fields.Length < 11) // Need all fields including MapName
                continue;

            try
            {
                // CSV format: MapName,TileX,TileY,ChunkX,ChunkY,AreaId,Flags,TextureLayers,HasLiquids,HasHoles,IsImpassible
                // Note: TileX=col, TileY=row in ADT coordinate system
                records.Add(new TerrainRecord(
                    TileRow: int.Parse(fields[2]),  // TileY = row
                    TileCol: int.Parse(fields[1]),  // TileX = col
                    ChunkX: int.Parse(fields[3]),
                    ChunkY: int.Parse(fields[4]),
                    AreaId: int.Parse(fields[5]),
                    Flags: int.Parse(fields[6]),
                    TextureLayers: int.Parse(fields[7]),
                    HasLiquids: fields[8].Equals("true", StringComparison.OrdinalIgnoreCase),
                    HasHoles: fields[9].Equals("true", StringComparison.OrdinalIgnoreCase)
                ));
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TerrainOverlayBuilder] Line {i + 1} parse error: {ex.Message}");
            }
        }

        return records;
    }
}

internal record TerrainRecord(
    int TileRow,
    int TileCol,
    int ChunkX,
    int ChunkY,
    int AreaId,
    int Flags,
    int TextureLayers,
    bool HasLiquids,
    bool HasHoles);
