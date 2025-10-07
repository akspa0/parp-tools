using System.Text;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Generates MCNK terrain metadata CSVs from converted LK ADTs.
/// </summary>
public sealed class TerrainCsvGenerator
{
    /// <summary>
    /// Generates terrain CSVs for all tiles in a map.
    /// </summary>
    /// <param name="adtMapDir">Directory containing ADT files</param>
    /// <param name="mapName">Map name</param>
    /// <param name="outputDir">Output directory for CSVs</param>
    /// <returns>Result with paths to generated CSVs</returns>
    public TerrainCsvResult Generate(string adtMapDir, string mapName, string outputDir)
    {
        try
        {
            var adtFiles = Directory.GetFiles(adtMapDir, "*.adt", SearchOption.TopDirectoryOnly);
            if (adtFiles.Length == 0)
            {
                return new TerrainCsvResult(
                    string.Empty,
                    string.Empty,
                    0,
                    Success: false,
                    ErrorMessage: $"No ADT files found in {adtMapDir}");
            }

            var terrainRecords = new List<McnkTerrainRecord>();

            // Extract terrain data from each ADT
            foreach (var adtPath in adtFiles)
            {
                var records = ExtractTerrainData(adtPath, mapName);
                terrainRecords.AddRange(records);
            }

            // Export CSVs
            var terrainCsvPath = Path.Combine(outputDir, $"{mapName}_mcnk_terrain.csv");
            var propertiesCsvPath = Path.Combine(outputDir, $"{mapName}_mcnk_properties.csv");

            ExportTerrainCsv(terrainRecords, terrainCsvPath);
            ExportPropertiesCsv(terrainRecords, propertiesCsvPath);

            return new TerrainCsvResult(
                TerrainCsvPath: terrainCsvPath,
                PropertiesCsvPath: propertiesCsvPath,
                ChunkCount: terrainRecords.Count,
                Success: true);
        }
        catch (Exception ex)
        {
            return new TerrainCsvResult(
                string.Empty,
                string.Empty,
                0,
                Success: false,
                ErrorMessage: $"Terrain CSV generation failed: {ex.Message}");
        }
    }

    private List<McnkTerrainRecord> ExtractTerrainData(string adtPath, string mapName)
    {
        var records = new List<McnkTerrainRecord>();

        try
        {
            // TODO: Read LK ADT and extract MCNK chunks
            // For now, return empty list as placeholder
            // Need to implement LK ADT reading to extract MCNK data

            // Parse tile coordinates from filename
            var fileName = Path.GetFileNameWithoutExtension(adtPath);
            var parts = fileName.Split('_');
            if (parts.Length < 3 || !int.TryParse(parts[^2], out var tileX) || !int.TryParse(parts[^1], out var tileY))
            {
                return records;
            }

            // Placeholder: Will implement actual MCNK extraction
            // Each ADT has 16x16 = 256 MCNK chunks
            for (int chunkY = 0; chunkY < 16; chunkY++)
            {
                for (int chunkX = 0; chunkX < 16; chunkX++)
                {
                    records.Add(new McnkTerrainRecord
                    {
                        MapName = mapName,
                        TileX = tileX,
                        TileY = tileY,
                        ChunkX = chunkX,
                        ChunkY = chunkY,
                        AreaId = 0, // Placeholder
                        Flags = 0,
                        TextureLayers = 0,
                        HasLiquids = false,
                        HasHoles = false,
                        IsImpassible = false
                    });
                }
            }
        }
        catch
        {
            // Skip problematic files
        }

        return records;
    }

    private void ExportTerrainCsv(List<McnkTerrainRecord> records, string csvPath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("MapName,TileX,TileY,ChunkX,ChunkY,AreaId,Flags,TextureLayers,HasLiquids,HasHoles,IsImpassible");

        foreach (var record in records.OrderBy(r => r.TileY).ThenBy(r => r.TileX).ThenBy(r => r.ChunkY).ThenBy(r => r.ChunkX))
        {
            csv.AppendLine($"{record.MapName},{record.TileX},{record.TileY}," +
                $"{record.ChunkX},{record.ChunkY},{record.AreaId},{record.Flags}," +
                $"{record.TextureLayers},{record.HasLiquids},{record.HasHoles},{record.IsImpassible}");
        }

        File.WriteAllText(csvPath, csv.ToString());
    }

    private void ExportPropertiesCsv(List<McnkTerrainRecord> records, string csvPath)
    {
        var csv = new StringBuilder();
        csv.AppendLine("MapName,TotalChunks,WithLiquids,WithHoles,Impassible,MaxTextureLayers");

        var grouped = records.GroupBy(r => r.MapName);

        foreach (var group in grouped)
        {
            var mapName = group.Key;
            var totalChunks = group.Count();
            var withLiquids = group.Count(r => r.HasLiquids);
            var withHoles = group.Count(r => r.HasHoles);
            var impassible = group.Count(r => r.IsImpassible);
            var maxLayers = group.Max(r => r.TextureLayers);

            csv.AppendLine($"{mapName},{totalChunks},{withLiquids},{withHoles},{impassible},{maxLayers}");
        }

        File.WriteAllText(csvPath, csv.ToString());
    }
}
