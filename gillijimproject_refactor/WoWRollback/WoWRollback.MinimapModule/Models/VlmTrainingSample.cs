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
    
    // Compact Height Data (replaces OBJ string)
    [property: JsonPropertyName("heights")] VlmChunkHeights[]? Heights,        // 256 chunks, each 145 heights
    [property: JsonPropertyName("chunk_positions")] float[]? ChunkPositions,   // Flattened: 256 * 3 (x,y,z)
    [property: JsonPropertyName("holes")] int[]? Holes,                        // 256 hole bitmasks
    
    // Legacy OBJ/MTL (optional, for backward compat or visualization)
    [property: JsonPropertyName("obj_content")] string? ObjContent,
    [property: JsonPropertyName("mtl_content")] string? MtlContent,
    
    // Raw Alpha/Shadow Data
    [property: JsonPropertyName("alpha_maps")] string? AlphaMaps,              // Base64 MCAL
    [property: JsonPropertyName("shadow_map")] string? ShadowMap,              // Base64 MCSH
    [property: JsonPropertyName("layer_masks")] List<string> LayerMasks,       // Paths to alpha mask PNGs
    
    // Textures
    [property: JsonPropertyName("textures")] List<string> Textures,            // Unique texture paths from MTEX
    [property: JsonPropertyName("textures_extracted")] List<string>? TexturesExtracted, // Paths to extracted PNGs
    
    // Per-Chunk Layer Data (for precise reconstruction)
    [property: JsonPropertyName("chunk_layers")] VlmChunkLayers[]? ChunkLayers, // 256 entries
    
    // Objects
    [property: JsonPropertyName("objects")] List<ObjectPlacement> Objects,
    
    // WDL Global Context
    [property: JsonPropertyName("wdl_heights")] short[]? WdlHeights,            // 17x17 low-res heightmap
    
    // Stats
    [property: JsonPropertyName("height_min")] float HeightMin,
    [property: JsonPropertyName("height_max")] float HeightMax
);

/// <summary>
/// Compact height data for a single MCNK chunk (145 values = 9x9 outer + 8x8 inner).
/// </summary>
public record VlmChunkHeights(
    [property: JsonPropertyName("idx")] int ChunkIndex,          // 0-255
    [property: JsonPropertyName("h")] float[] Heights            // 145 floats (MCVT)
);

/// <summary>
/// Per-chunk texture layer data for reconstruction (MCLY + MCAL).
/// </summary>
public record VlmChunkLayers(
    [property: JsonPropertyName("idx")] int ChunkIndex,
    [property: JsonPropertyName("layers")] VlmTextureLayer[] Layers,
    [property: JsonPropertyName("mcal")] string? McalBase64       // Raw MCAL bytes for this chunk
);

/// <summary>
/// Single texture layer definition (from MCLY).
/// </summary>
public record VlmTextureLayer(
    [property: JsonPropertyName("tex_id")] uint TextureId,
    [property: JsonPropertyName("flags")] uint Flags,
    [property: JsonPropertyName("alpha_off")] uint AlphaOffset,
    [property: JsonPropertyName("effect_id")] uint EffectId);

/// <summary>
/// Object placement from MODF/MDDF.
/// </summary>
public record ObjectPlacement(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("name_id")] uint NameId,        // Raw index for fallback correlation
    [property: JsonPropertyName("unique_id")] uint UniqueId,
    [property: JsonPropertyName("x")] float X,
    [property: JsonPropertyName("y")] float Y,
    [property: JsonPropertyName("z")] float Z,
    [property: JsonPropertyName("rot_x")] float RotX,
    [property: JsonPropertyName("rot_y")] float RotY,
    [property: JsonPropertyName("rot_z")] float RotZ,
    [property: JsonPropertyName("scale")] float Scale,
    [property: JsonPropertyName("category")] string Category);  // "wmo" or "m2"

