using System.Text.Json.Serialization;

namespace WoWRollback.MinimapModule.Models;

/// <summary>
/// Complete training sample for VLM - maps to JSON output.
/// </summary>
public record VlmTrainingSample(
    [property: JsonPropertyName("adt_tile")] string AdtTile,
    [property: JsonPropertyName("textures")] List<ChunkTextureInfo> Textures,
    [property: JsonPropertyName("objects")] List<ObjectPlacement> Objects,
    [property: JsonPropertyName("terrain")] TerrainSummary Terrain,
    [property: JsonPropertyName("mesh_path")] string? MeshPath = null);

/// <summary>
/// Per-chunk texture layer info (256 chunks per ADT).
/// </summary>
public record ChunkTextureInfo(
    [property: JsonPropertyName("chunk")] int[] Chunk,      // [x, y] 0-15
    [property: JsonPropertyName("layers")] string[] Layers); // texture asset paths

/// <summary>
/// Object placement from MODF/MDDF.
/// </summary>
public record ObjectPlacement(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("x")] float X,
    [property: JsonPropertyName("y")] float Y,
    [property: JsonPropertyName("z")] float Z,
    [property: JsonPropertyName("category")] string Category);  // "wmo" or "m2"

/// <summary>
/// Terrain height summary.
/// </summary>
public record TerrainSummary(
    [property: JsonPropertyName("height_min")] float HeightMin,
    [property: JsonPropertyName("height_max")] float HeightMax,
    [property: JsonPropertyName("water")] bool HasWater);
