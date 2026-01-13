using System.Text.Json.Serialization;

namespace WoWMapConverter.Core.VLM;

/// <summary>
/// Complete training sample for VLM - maps to JSON output.
/// </summary>
public record VlmTrainingSample(
    [property: JsonPropertyName("image")] string ImagePath,
    [property: JsonPropertyName("depth")] string? DepthPath,
    [property: JsonPropertyName("terrain_data")] VlmTerrainData TerrainData
);

/// <summary>
/// Complete terrain data for a single ADT tile.
/// </summary>
public record VlmTerrainData(
    [property: JsonPropertyName("adt_tile")] string AdtTile,
    
    // Compact Height Data (145 heights per chunk × 256 chunks)
    [property: JsonPropertyName("heights")] VlmChunkHeights[]? Heights,
    [property: JsonPropertyName("chunk_positions")] float[]? ChunkPositions,  // 256 × 3 (x,y,z)
    [property: JsonPropertyName("holes")] int[]? Holes,                       // 256 hole bitmasks
    
    // Shadow Maps - paths to per-chunk PNGs
    [property: JsonPropertyName("shadow_maps")] string[]? ShadowMaps,
    
    // Alpha Masks - paths to per-layer PNGs
    [property: JsonPropertyName("alpha_masks")] string[]? AlphaMasks,
    
    // Textures
    [property: JsonPropertyName("textures")] List<string> Textures,
    
    // Per-Chunk Layer Data (MCLY)
    [property: JsonPropertyName("chunk_layers")] VlmChunkLayers[]? ChunkLayers,
    
    // Liquids (MH2O/MCLQ)
    [property: JsonPropertyName("liquids")] VlmLiquidData[]? Liquids,
    
    // Objects (MDDF/MODF)
    [property: JsonPropertyName("objects")] List<VlmObjectPlacement> Objects,
    
    // Stats
    [property: JsonPropertyName("height_min")] float HeightMin,
    [property: JsonPropertyName("height_max")] float HeightMax
);

/// <summary>
/// Compact height data for a single MCNK chunk (145 values).
/// </summary>
public record VlmChunkHeights(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("h")] float[] Heights
);

/// <summary>
/// Per-chunk texture layer data (MCLY).
/// </summary>
public record VlmChunkLayers(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("layers")] VlmTextureLayer[] Layers
);

/// <summary>
/// Single texture layer definition (from MCLY).
/// </summary>
public record VlmTextureLayer(
    [property: JsonPropertyName("tex_id")] uint TextureId,
    [property: JsonPropertyName("flags")] uint Flags,
    [property: JsonPropertyName("alpha_off")] uint AlphaOffset,
    [property: JsonPropertyName("effect_id")] uint EffectId
);

/// <summary>
/// Liquid data per chunk (MH2O or MCLQ).
/// </summary>
public record VlmLiquidData(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("type")] int LiquidType,
    [property: JsonPropertyName("min_height")] float MinHeight,
    [property: JsonPropertyName("max_height")] float MaxHeight,
    [property: JsonPropertyName("mask_path")] string? MaskPath,
    [property: JsonPropertyName("heights")] float[]? Heights  // 9×9 = 81 values if present
);

/// <summary>
/// Object placement from MDDF/MODF.
/// </summary>
public record VlmObjectPlacement(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("name_id")] uint NameId,
    [property: JsonPropertyName("unique_id")] uint UniqueId,
    [property: JsonPropertyName("x")] float X,
    [property: JsonPropertyName("y")] float Y,
    [property: JsonPropertyName("z")] float Z,
    [property: JsonPropertyName("rot_x")] float RotX,
    [property: JsonPropertyName("rot_y")] float RotY,
    [property: JsonPropertyName("rot_z")] float RotZ,
    [property: JsonPropertyName("scale")] float Scale,
    [property: JsonPropertyName("category")] string Category  // "wmo" or "m2"
);

/// <summary>
/// Result of VLM dataset export.
/// </summary>
public record VlmExportResult(
    int TilesExported,
    int TilesSkipped,
    int UniqueTextures,
    string OutputDirectory
);
