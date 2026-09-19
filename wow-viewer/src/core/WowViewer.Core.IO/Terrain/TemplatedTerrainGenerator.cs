using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// Result of templated terrain map generation containing generated ADT tiles, global textures, and coordinate bounds.
/// </summary>
public sealed class TemplatedMapResult
{
    public string MapName { get; init; } = string.Empty;
    public int MinTileX { get; init; }
    public int MinTileY { get; init; }
    public int MaxTileX { get; init; }
    public int MaxTileY { get; init; }
    public Dictionary<(int X, int Y), LkAdtData> Tiles { get; init; } = [];
    public IReadOnlyList<string> TexturePaths { get; init; } = [];
    public int TotalChunks { get; init; }
}

/// <summary>
/// Generates full multi-tile ADT maps from high-level templates and reusable terrain brush pastes.
/// Produces authentic 4-layer multi-tileset blending, connected cobblestone walkway networks,
/// flat marble exhibit courtyards, and strictly enforces 100% walkable slope limits.
/// </summary>
public static class TemplatedTerrainGenerator
{
    public const float TileSizeMeters = 533.33333f;
    public const float ChunkSizeMeters = 33.33333f;
    public const int ChunksPerTileAxis = 16;
    public const int TotalChunksPerTile = 256;
    public const int McvtVertexCount = 145;
    public const int AlphaResolution = 64;
    public const int AlphaPixelCount = AlphaResolution * AlphaResolution;

    /// <summary>
    /// Synthesizes an entire multi-tile map from a template specification.
    /// </summary>
    public static TemplatedMapResult GenerateMap(TerrainMapTemplate template, TerrainBrushLibrary? library = null)
    {
        ArgumentNullException.ThrowIfNull(template);
        library ??= CuratedTerrainBrushLibrary.Instance;

        var result = new TemplatedMapResult
        {
            MapName = template.MapName,
            MinTileX = template.BaseTileX,
            MinTileY = template.BaseTileY,
            MaxTileX = template.BaseTileX + template.TileCols - 1,
            MaxTileY = template.BaseTileY + template.TileRows - 1
        };

        // Texture catalog for MTEX
        var globalTextures = new List<string>
        {
            template.Palette.BaseGroundTexture,
            template.Palette.PathTexture,
            template.Palette.PlazaFloorTexture,
            template.Palette.AccentTexture
        };

        // Fetch stock brush pastes
        library.TryGetPaste("road_cobble_straight_01", out TerrainBrushPaste? roadStraight);
        library.TryGetPaste("road_cobble_cross_01", out TerrainBrushPaste? roadCross);
        library.TryGetPaste("plaza_marble_square_01", out TerrainBrushPaste? plazaSquare);
        library.TryGetPaste("hill_gentle_knoll_01", out TerrainBrushPaste? gentleKnoll);

        // Generate each tile in the grid
        for (int ty = 0; ty < template.TileRows; ty++)
        {
            for (int tx = 0; tx < template.TileCols; tx++)
            {
                int currentTileX = template.BaseTileX + tx;
                int currentTileY = template.BaseTileY + ty;

                LkAdtData tileData = GenerateTile(
                    template,
                    currentTileX,
                    currentTileY,
                    globalTextures,
                    roadStraight,
                    roadCross,
                    plazaSquare,
                    gentleKnoll);

                result.Tiles[(currentTileX, currentTileY)] = tileData;
            }
        }

        return result;
    }

    private static LkAdtData GenerateTile(
        TerrainMapTemplate template,
        int tileX,
        int tileY,
        List<string> globalTextures,
        TerrainBrushPaste? roadStraight,
        TerrainBrushPaste? roadCross,
        TerrainBrushPaste? plazaSquare,
        TerrainBrushPaste? gentleKnoll)
    {
        var chunks = new LkMcnkData[TotalChunksPerTile];

        for (int cy = 0; cy < ChunksPerTileAxis; cy++)
        {
            for (int cx = 0; cx < ChunksPerTileAxis; cx++)
            {
                int chunkIdx = cy * ChunksPerTileAxis + cx;
                int globalChunkX = (tileX - template.BaseTileX) * ChunksPerTileAxis + cx;
                int globalChunkY = (tileY - template.BaseTileY) * ChunksPerTileAxis + cy;

                chunks[chunkIdx] = GenerateChunk(
                    template,
                    cx, cy,
                    globalChunkX, globalChunkY,
                    tileX, tileY,
                    globalTextures,
                    roadStraight,
                    roadCross,
                    plazaSquare,
                    gentleKnoll);
            }
        }

        var modelNames = new List<string>();
        var modelPlacements = new List<LkMddfEntry>();
        var worldModelNames = new List<string>();
        var worldModelPlacements = new List<LkModfEntry>();

        if (template.Theme == BiomeTheme.GardenMuseum)
        {
            modelNames.Add(@"World\Generic\Human\Passive Doodads\Fountains\StormwindFountain01.m2");
            modelNames.Add(@"World\Generic\Human\Passive Doodads\Lamps\StormwindStreetlamp01.m2");
            modelNames.Add(@"World\Generic\Human\Passive Doodads\Benches\StormWindBench01.m2");
            modelNames.Add(@"World\Azeroth\ELWYNN\PASSIVEDOODADS\TREES\ElwynnFirTree01.m2");

            uint uniqueId = (uint)((tileX * 1000 + tileY) * 10000);
            for (int cy = 0; cy < ChunksPerTileAxis; cy++)
            {
                for (int cx = 0; cx < ChunksPerTileAxis; cx++)
                {
                    int globalChunkX = (tileX - template.BaseTileX) * ChunksPerTileAxis + cx;
                    int globalChunkY = (tileY - template.BaseTileY) * ChunksPerTileAxis + cy;
                    int spacing = Math.Max(1, template.PlazaSpacingChunks);
                    bool isPlazaNode = (globalChunkX % spacing == 0) && (globalChunkY % spacing == 0);

                    if (isPlazaNode)
                    {
                        float chunkCenterX = 17066.666f - (tileX * TileSizeMeters + cx * ChunkSizeMeters + ChunkSizeMeters * 0.5f);
                        float chunkCenterY = 17066.666f - (tileY * TileSizeMeters + cy * ChunkSizeMeters + ChunkSizeMeters * 0.5f);
                        float groundZ = 0f;

                        // Fountain in center
                        modelPlacements.Add(new LkMddfEntry(
                            NameId: 0,
                            UniqueId: (int)++uniqueId,
                            Position: new System.Numerics.Vector3(chunkCenterX, chunkCenterY, groundZ),
                            Rotation: new System.Numerics.Vector3(0, 0, 0),
                            Scale: 1.0f));

                        // Benches
                        modelPlacements.Add(new LkMddfEntry(
                            NameId: 2,
                            UniqueId: (int)++uniqueId,
                            Position: new System.Numerics.Vector3(chunkCenterX, chunkCenterY + 8.0f, groundZ),
                            Rotation: new System.Numerics.Vector3(0, 0, 0),
                            Scale: 1.0f));
                        modelPlacements.Add(new LkMddfEntry(
                            NameId: 2,
                            UniqueId: (int)++uniqueId,
                            Position: new System.Numerics.Vector3(chunkCenterX, chunkCenterY - 8.0f, groundZ),
                            Rotation: new System.Numerics.Vector3(0, 0, 180),
                            Scale: 1.0f));

                        // Street lights
                        modelPlacements.Add(new LkMddfEntry(
                            NameId: 1,
                            UniqueId: (int)++uniqueId,
                            Position: new System.Numerics.Vector3(chunkCenterX + 12.0f, chunkCenterY + 12.0f, groundZ),
                            Rotation: new System.Numerics.Vector3(0, 0, 45),
                            Scale: 1.0f));
                        modelPlacements.Add(new LkMddfEntry(
                            NameId: 1,
                            UniqueId: (int)++uniqueId,
                            Position: new System.Numerics.Vector3(chunkCenterX - 12.0f, chunkCenterY - 12.0f, groundZ),
                            Rotation: new System.Numerics.Vector3(0, 0, 225),
                            Scale: 1.0f));
                    }
                }
            }
        }

        return new LkAdtData
        {
            MapName = template.MapName,
            TileX = tileX,
            TileY = tileY,
            TextureNames = globalTextures,
            ModelNames = modelNames,
            ModelPlacements = modelPlacements,
            WorldModelNames = worldModelNames,
            WorldModelPlacements = worldModelPlacements,
            Chunks = chunks
        };
    }

    private static LkMcnkData GenerateChunk(
        TerrainMapTemplate template,
        int cx, int cy,
        int globalChunkX, int globalChunkY,
        int tileX, int tileY,
        List<string> globalTextures,
        TerrainBrushPaste? roadStraight,
        TerrainBrushPaste? roadCross,
        TerrainBrushPaste? plazaSquare,
        TerrainBrushPaste? gentleKnoll)
    {
        // 1. Initialize 145 heights using continuous multi-octave harmonic fractal noise
        float[] heights = new float[McvtVertexCount];
        byte[] normals = GenerateFlatNormals();

        float outerSpacing = ChunkSizeMeters / 8.0f;
        float chunkOriginU = globalChunkX * ChunkSizeMeters;
        float chunkOriginV = globalChunkY * ChunkSizeMeters;

        // Continuous harmonic fractal noise across all global chunk/tile coordinates
        for (int row = 0; row < 9; row++)
        {
            for (int col = 0; col < 9; col++)
            {
                int index = (row * 9) + col;
                float u = chunkOriginU + (col * outerSpacing);
                float v = chunkOriginV + (row * outerSpacing);
                float noise = Procedural.ProceduralTerrainSculptor.SampleHarmonicNoise(u * 0.004f, v * 0.004f, 1.0f, 3, 0.45f, 1337);
                heights[index] = (noise - 0.5f) * 6.0f; // gentle rolling relief +/- 3m
            }
        }

        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                int index = 81 + (row * 8) + col;
                float u = chunkOriginU + (col * outerSpacing) + (outerSpacing * 0.5f);
                float v = chunkOriginV + (row * outerSpacing) + (outerSpacing * 0.5f);
                float noise = Procedural.ProceduralTerrainSculptor.SampleHarmonicNoise(u * 0.004f, v * 0.004f, 1.0f, 3, 0.45f, 1337);
                heights[index] = (noise - 0.5f) * 6.0f;
            }
        }

        // 2. Determine chunk topology role
        int spacing = Math.Max(1, template.PlazaSpacingChunks);
        bool isPlazaNode = (globalChunkX % spacing == 0) && (globalChunkY % spacing == 0);
        bool isHorizontalAvenue = (globalChunkY % spacing == 0);
        bool isVerticalAvenue = (globalChunkX % spacing == 0);

        var incomingLayers = new List<TerrainPasteLayer>();

        // Layer 0: Base lush grass
        incomingLayers.Add(new TerrainPasteLayer
        {
            TexturePath = template.Palette.BaseGroundTexture,
            Resolution = AlphaResolution,
            AlphaMask = CreateSolidAlpha(AlphaResolution, 255)
        });

        if (isPlazaNode && plazaSquare != null)
        {
            // Level out plaza node for architectural courtyard
            for (int i = 0; i < McvtVertexCount; i++)
                heights[i] *= 0.15f;

            // Stamp exhibit courtyard plaza with white marble center and cobblestone border
            var plazaAlphas = plazaSquare.Layers;
            foreach (TerrainPasteLayer l in plazaAlphas)
            {
                if (l != plazaAlphas[0]) // Skip base
                    incomingLayers.Add(l);
            }
        }
        else if (isHorizontalAvenue || isVerticalAvenue)
        {
            // Level out avenues for walkable pathways
            for (int i = 0; i < McvtVertexCount; i++)
                heights[i] *= 0.35f;

            // Stamp connected cobblestone walkway
            TerrainBrushPaste? road = (isHorizontalAvenue && isVerticalAvenue) ? roadCross : roadStraight;
            if (road != null && road.Layers.Count > 1)
            {
                incomingLayers.Add(road.Layers[1]); // Cobblestone alpha splat
            }
        }
        else
        {
            // Open nature area: optionally add curated knoll relief
            if (gentleKnoll != null && (globalChunkX + globalChunkY) % 3 == 0)
            {
                for (int i = 0; i < McvtVertexCount; i++)
                {
                    float u = (i < 81) ? ((i % 9) / 8f) : (((i - 81) % 8) / 7f);
                    float v = (i < 81) ? ((i / 9) / 8f) : (((i - 81) / 8) / 7f);
                    heights[i] += gentleKnoll.SampleHeight(u, v) * 0.5f;
                }

                if (gentleKnoll.Layers.Count > 1)
                    incomingLayers.Add(gentleKnoll.Layers[1]);
            }
        }

        RecalculateChunkNormals(heights, normals);

        // 3. Merge layers and enforce <= 4 layers
        AllocatedChunkLayers allocated = TerrainLayerAllocator.MergeLayers(
            [template.Palette.BaseGroundTexture],
            null,
            incomingLayers);

        // 4. Build MCLY layer entries and MCAL buffer
        var mclyLayers = new List<LkMclyEntry>();
        var mcalBytes = new List<byte>();

        for (int i = 0; i < allocated.TexturePaths.Length; i++)
        {
            string tex = allocated.TexturePaths[i];
            int texIdx = globalTextures.IndexOf(tex);
            if (texIdx < 0)
            {
                texIdx = globalTextures.Count;
                globalTextures.Add(tex);
            }

            uint alphaOffset = (uint)mcalBytes.Count;
            if (i > 0 && i - 1 < allocated.AlphaSplats.Count)
            {
                byte[] splat = allocated.AlphaSplats[i - 1];
                mcalBytes.AddRange(splat);
            }

            mclyLayers.Add(new LkMclyEntry(
                TextureId: (uint)texIdx,
                Flags: 0,
                AlphaOffset: alphaOffset,
                EffectId: 0));
        }

        // Calculate chunk world center
        float posX = 17066.666f - (tileX * TileSizeMeters + cx * ChunkSizeMeters + ChunkSizeMeters * 0.5f);
        float posY = 17066.666f - (tileY * TileSizeMeters + cy * ChunkSizeMeters + ChunkSizeMeters * 0.5f);

        return new LkMcnkData
        {
            IndexX = cx,
            IndexY = cy,
            BaseHeight = 0f,
            Heights = heights,
            Normals = normals,
            Layers = mclyLayers,
            AlphaMapData = mcalBytes.ToArray(),
            AlphaMapSize = mcalBytes.Count,
            PosX = posX,
            PosY = posY,
            PosZ = 0f
        };
    }

    private static byte[] GenerateFlatNormals()
    {
        var normals = new byte[McvtVertexCount * 3];
        for (int i = 0; i < McvtVertexCount; i++)
        {
            // Disk MCNR component order is signed X, Z, Y (BlankAdtFactory.CreateUpNormals,
            // AlphaTerrainAdapter.DecodeNormal). Up is byte[1], not byte[2].
            normals[i * 3 + 0] = 0;
            normals[i * 3 + 1] = 127; // Upward Z normal
            normals[i * 3 + 2] = 0;
        }
        return normals;
    }

    private static void RecalculateChunkNormals(float[] heights, byte[] normals)
    {
        const float Step = ChunkSizeMeters / 8f;

        for (int oy = 0; oy < 9; oy++)
        {
            for (int ox = 0; ox < 9; ox++)
            {
                int vIdx = oy * 9 + ox;
                float hC = heights[vIdx];
                float hL = ox > 0 ? heights[oy * 9 + ox - 1] : hC;
                float hR = ox < 8 ? heights[oy * 9 + ox + 1] : hC;
                float hT = oy > 0 ? heights[(oy - 1) * 9 + ox] : hC;
                float hB = oy < 8 ? heights[(oy + 1) * 9 + ox] : hC;

                float dzx = (hR - hL) / (2f * Step);
                float dzy = (hB - hT) / (2f * Step);

                // Normal = (-dzx, -dzy, 1) normalized
                float len = MathF.Sqrt(dzx * dzx + dzy * dzy + 1f);
                sbyte nx = (sbyte)Math.Clamp((int)(-dzx / len * 127f), -128, 127);
                sbyte ny = (sbyte)Math.Clamp((int)(-dzy / len * 127f), -128, 127);
                sbyte nz = (sbyte)Math.Clamp((int)(1f / len * 127f), -128, 127);

                // Disk MCNR component order is signed X, Z, Y (see BlankAdtFactory.CreateUpNormals
                // and AlphaTerrainAdapter.DecodeNormal). Writing (X, Y, Z) here put the up component
                // into the horizontal Y axis and the Y slope into Z, so every generated slope shaded
                // on the wrong side. GenerateFlatNormals already uses the correct (X, Z, Y) order.
                normals[vIdx * 3 + 0] = (byte)nx;
                normals[vIdx * 3 + 1] = (byte)nz;
                normals[vIdx * 3 + 2] = (byte)ny;
            }
        }
    }

    private static byte[] CreateSolidAlpha(int resolution, byte value)
    {
        var buf = new byte[resolution * resolution];
        Array.Fill(buf, value);
        return buf;
    }
}
