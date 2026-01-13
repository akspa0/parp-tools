using System.Text.Json.Serialization;

namespace WoWRollback.MinimapModule.Models;

/// <summary>
/// Complete training sample for VLM - maps to JSON output.
/// </summary>
public record VlmTrainingSample(
    [property: JsonPropertyName("image")] string ImagePath,
    [property: JsonPropertyName("terrain_data")] VlmTerrainData TerrainData
);

public record VlmTerrainData(
    [property: JsonPropertyName("adt_tile")] string AdtTile,
    [property: JsonPropertyName("obj_content")] string ObjContent,
    [property: JsonPropertyName("mtl_content")] string MtlContent,
    [property: JsonPropertyName("alpha_maps")] string? AlphaMaps,     // Base64 MCAL
    [property: JsonPropertyName("shadow_map")] string? ShadowMap,     // Base64 MCSH
    [property: JsonPropertyName("layer_masks")] List<string> LayerMasks, // Paths to alpha mask images
    [property: JsonPropertyName("textures")] List<string> Textures,   // Unique texture files
    [property: JsonPropertyName("layers")] List<VlmTextureLayer> Layers, // Texture layer definitions
    [property: JsonPropertyName("objects")] List<ObjectPlacement> Objects,
    [property: JsonPropertyName("height_min")] float HeightMin,
    [property: JsonPropertyName("height_max")] float HeightMax
);

/// <summary>
/// Object placement from MODF/MDDF.
/// </summary>
public record ObjectPlacement(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("unique_id")] uint UniqueId,
    [property: JsonPropertyName("x")] float X,
    [property: JsonPropertyName("y")] float Y,
    [property: JsonPropertyName("z")] float Z,
    [property: JsonPropertyName("rot_x")] float RotX,
    [property: JsonPropertyName("rot_y")] float RotY,
    [property: JsonPropertyName("rot_z")] float RotZ,
    [property: JsonPropertyName("scale")] float Scale,
    [property: JsonPropertyName("category")] string Category);  // "wmo" or "m2"

