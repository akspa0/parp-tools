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
    [property: JsonPropertyName("heightmap")] string? HeightmapPath,          // Path to 16-bit PNG heightmap
    [property: JsonPropertyName("heightmap_local")] string? HeightmapLocalPath,
    [property: JsonPropertyName("heightmap_global")] string? HeightmapGlobalPath,
    [property: JsonPropertyName("normalmap")] string? NormalmapPath,
    [property: JsonPropertyName("mccv_map")] string? MccvMapPath,
    
    // Shadow Maps - paths to per-chunk PNGs
    [property: JsonPropertyName("shadow_maps")] string[]? ShadowMaps,
    
    // Shadow Maps - raw bit data (64 bytes per chunk = 512 bits = 64x8 shadow map)
    [property: JsonPropertyName("shadow_bits")] VlmChunkShadowBits[]? ShadowBits,
    
    // Alpha Masks - paths to per-layer PNGs
    [property: JsonPropertyName("alpha_masks")] string[]? AlphaMasks,
    
    // Liquid Stitched Maps
    [property: JsonPropertyName("liquid_mask")] string? LiquidMaskPath,
    [property: JsonPropertyName("liquid_height")] string? LiquidHeightPath,
    [property: JsonPropertyName("liquid_min")] float LiquidMinHeight,
    [property: JsonPropertyName("liquid_max")] float LiquidMaxHeight,
    
    // Textures
    [property: JsonPropertyName("textures")] List<string> Textures,
    
    // Per-Chunk Layer Data (MCLY)
    [property: JsonPropertyName("chunk_layers")] VlmChunkLayers[]? ChunkLayers,
    
    // Liquids (MH2O/MCLQ)
    [property: JsonPropertyName("liquids")] VlmLiquidData[]? Liquids,
    
    // Objects (MDDF/MODF)
    [property: JsonPropertyName("objects")] List<VlmObjectPlacement> Objects,
    
    // WDL Low-Res Heightmap
    [property: JsonPropertyName("wdl_heights")] VlmWdlData? WdlHeights,
    
    // Stats
    [property: JsonPropertyName("height_min")] float HeightMin,
    [property: JsonPropertyName("height_max")] float HeightMax,
    [property: JsonPropertyName("height_global_min")] float HeightGlobalMin,
    [property: JsonPropertyName("height_global_max")] float HeightGlobalMax,
    [property: JsonPropertyName("is_interleaved")] bool IsInterleaved
);

/// <summary>
/// WDL low-resolution heightmap data (17x17 + 16x16 shorts).
/// </summary>
public record VlmWdlData(
    [property: JsonPropertyName("outer_17")] short[] Height17, // Flattened 17x17 = 289
    [property: JsonPropertyName("inner_16")] short[] Height16  // Flattened 16x16 = 256
);

/// <summary>
/// Compact height data for a single MCNK chunk (145 values).
/// </summary>
public record VlmChunkHeights(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("h")] float[] Heights
);

/// <summary>
/// Raw shadow map bit data for a single MCNK chunk (64 bytes = 512 bits).
/// Encoded as Base64 for JSON serialization.
/// </summary>
public record VlmChunkShadowBits(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("bits")] string BitsBase64  // 64 bytes -> Base64
);

/// <summary>
/// Per-chunk texture layer data (MCLY) with all data for terrain reconstruction.
/// </summary>
public record VlmChunkLayers(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("layers")] VlmTextureLayer[] Layers,
    // Per-chunk paths for reconstruction
    [property: JsonPropertyName("shadow_path")] string? ShadowPath = null,
    [property: JsonPropertyName("normals")] sbyte[]? Normals = null,  // MCNR 448 bytes (145 * 3 + 13 padding)
    [property: JsonPropertyName("mccv_colors")] byte[]? MccvColors = null,  // MCCV vertex colors (145 * 4 RGBA = 580 bytes)
    [property: JsonPropertyName("area_id")] uint AreaId = 0,
    [property: JsonPropertyName("flags")] uint Flags = 0
);

/// <summary>
/// Single texture layer definition (from MCLY) with full texture path for reconstruction.
/// </summary>
public record VlmTextureLayer(
    [property: JsonPropertyName("tex_id")] uint TextureId,
    [property: JsonPropertyName("texture_path")] string? TexturePath,  // Actual path from MTEX
    [property: JsonPropertyName("flags")] uint Flags,
    [property: JsonPropertyName("alpha_off")] uint AlphaOffset,
    [property: JsonPropertyName("effect_id")] uint EffectId,
    [property: JsonPropertyName("ground_effects")] string[]? GroundEffects = null,
    // Raw alpha mask data (64 bytes = 64x64 / 8 for 1-bit, or 4096 bytes for 8-bit)
    [property: JsonPropertyName("alpha_bits")] string? AlphaBitsBase64 = null,
    // Path to exported alpha PNG for this layer
    [property: JsonPropertyName("alpha_path")] string? AlphaPath = null,
    
    // Raw alpha bytes (Not serialized to JSON, used for .bin export)
    [property: JsonIgnore] byte[]? AlphaData = null
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
/// Object placement from MDDF/MODF with bounding box data.
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
    [property: JsonPropertyName("category")] string Category,  // "wmo" or "m2"
    [property: JsonPropertyName("bounds_min")] float[]? BoundsMin = null,  // [x, y, z] local min
    [property: JsonPropertyName("bounds_max")] float[]? BoundsMax = null   // [x, y, z] local max
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
