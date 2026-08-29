using System.Numerics;

namespace WowViewer.Core.Editor.Procedural;

/// <summary>
/// A 4-layer texture palette for procedural map rendering.
/// </summary>
public sealed record ProceduralTexturePalette(
    string BaseTexture,
    string WalkwayTexture,
    string CheckerBorderTexture,
    string CenterPlazaTexture);

/// <summary>
/// Generated alpha splat data and texture assignment for a single 33.33m ADT chunk.
/// </summary>
public sealed record ChunkAlphaData(
    IReadOnlyList<string> TextureFilenames,
    IReadOnlyList<byte[]> AlphaLayers64x64);

/// <summary>
/// Generates multi-layer anti-aliased 64x64 alpha splat maps (MCAL) and assigns texture layers (MCLY)
/// to create garden landscapes, cobblestone pathways, stylized checker borders, and clean exhibit floors.
/// </summary>
public static class ProceduralTexturePainter
{
    public const float ChunkSizeMeters = 33.333333f;
    public const int AlphaResolution = 64;
    public const float PixelWorldStep = ChunkSizeMeters / AlphaResolution; // ~0.520833m

    /// <summary>
    /// Resolves the canonical texture palette for a chosen procedural theme.
    /// </summary>
    public static ProceduralTexturePalette ResolvePalette(ProceduralMapTheme theme)
    {
        return theme switch
        {
            ProceduralMapTheme.Garden => new ProceduralTexturePalette(
                BaseTexture: @"tileset\elwynn\elwynngrass.blp",
                WalkwayTexture: @"tileset\city\stormwindcobble.blp",
                CheckerBorderTexture: @"tileset\generic\checkers.blp",
                CenterPlazaTexture: @"tileset\city\whitemarble.blp"),

            ProceduralMapTheme.Marble => new ProceduralTexturePalette(
                BaseTexture: @"tileset\city\whitemarble.blp",
                WalkwayTexture: @"tileset\city\stormwindcobble.blp",
                CheckerBorderTexture: @"tileset\generic\checkers.blp",
                CenterPlazaTexture: @"tileset\generic\stone.blp"),

            ProceduralMapTheme.Autumn => new ProceduralTexturePalette(
                BaseTexture: @"tileset\azshara\azsharagrass.blp",
                WalkwayTexture: @"tileset\generic\dirt.blp",
                CheckerBorderTexture: @"tileset\generic\checkers.blp",
                CenterPlazaTexture: @"tileset\azshara\azshararock.blp"),

            ProceduralMapTheme.Desert => new ProceduralTexturePalette(
                BaseTexture: @"tileset\tanaris\tanarissand.blp",
                WalkwayTexture: @"tileset\barrens\barrenscobble.blp",
                CheckerBorderTexture: @"tileset\generic\checkers.blp",
                CenterPlazaTexture: @"tileset\tanaris\tanarisrock.blp"),

            _ => new ProceduralTexturePalette(
                BaseTexture: @"tileset\elwynn\elwynngrass.blp",
                WalkwayTexture: @"tileset\city\stormwindcobble.blp",
                CheckerBorderTexture: @"tileset\generic\checkers.blp",
                CenterPlazaTexture: @"tileset\city\whitemarble.blp")
        };
    }

    /// <summary>
    /// Generates 64x64 alpha splat maps for up to 4 layers in a specific chunk.
    /// Layer 0: Base Garden Turf (implied full coverage).
    /// Layer 1: Walkways & Promenades.
    /// Layer 2: Decorative Checkerboard Border Ring.
    /// Layer 3: Clean Neutral Center Exhibit Plaza.
    /// </summary>
    public static ChunkAlphaData GenerateChunkAlphaLayers(
        int chunkX,
        int chunkY,
        IReadOnlyList<AdaptiveExhibitPlacement> tilePlacements,
        ProceduralTexturePalette palette)
    {
        float chunkOriginU = chunkX * ChunkSizeMeters;
        float chunkOriginV = chunkY * ChunkSizeMeters;

        byte[] alphaWalkway = new byte[AlphaResolution * AlphaResolution];
        byte[] alphaCheckers = new byte[AlphaResolution * AlphaResolution];
        byte[] alphaPlaza = new byte[AlphaResolution * AlphaResolution];

        bool hasWalkway = false;
        bool hasCheckers = false;
        bool hasPlaza = false;

        float halfChunk = ChunkSizeMeters * 0.5f;

        for (int py = 0; py < AlphaResolution; py++)
        {
            float localV = chunkOriginV + ((py + 0.5f) * PixelWorldStep);

            for (int px = 0; px < AlphaResolution; px++)
            {
                float localU = chunkOriginU + ((px + 0.5f) * PixelWorldStep);
                int pixelIndex = (py * AlphaResolution) + px;

                // 1. Evaluate Arterial Walkways (connecting cell perimeters)
                float uInChunk = localU % ChunkSizeMeters;
                float vInChunk = localV % ChunkSizeMeters;

                // Walkway paths along chunk borders and center cross
                float distToRoadX = MathF.Min(uInChunk, ChunkSizeMeters - uInChunk);
                float distToRoadY = MathF.Min(vInChunk, ChunkSizeMeters - vInChunk);
                float distToCrossX = MathF.Abs(uInChunk - halfChunk);
                float distToCrossY = MathF.Abs(vInChunk - halfChunk);

                float roadDist = MathF.Min(MathF.Min(distToRoadX, distToRoadY), MathF.Min(distToCrossX, distToCrossY));
                if (roadDist < 2.5f)
                {
                    float pathAlpha = SmoothStep(2.5f, 1.2f, roadDist);
                    byte val = (byte)Math.Clamp((int)(pathAlpha * 255f), 0, 255);
                    if (val > 0)
                    {
                        alphaWalkway[pixelIndex] = val;
                        hasWalkway = true;
                    }
                }

                // 2. Evaluate Exhibit Pedestal Decor (Checkers Ring & Clean Center Plaza)
                if (tilePlacements != null)
                {
                    for (int i = 0; i < tilePlacements.Count; i++)
                    {
                        AdaptiveExhibitPlacement p = tilePlacements[i];
                        float centerU = p.CellU + (p.CellSize * 0.5f);
                        float centerV = p.CellV + (p.CellSize * 0.5f);

                        float dx = localU - centerU;
                        float dy = localV - centerV;
                        float dist = MathF.Sqrt((dx * dx) + (dy * dy));

                        float rInner = p.CellSize * 0.32f;
                        float rOuter = p.CellSize * 0.46f;

                        // Clean neutral center plaza (Layer 3)
                        if (dist < rInner)
                        {
                            float plazaAlpha = SmoothStep(rInner, rInner * 0.85f, dist);
                            byte pVal = (byte)Math.Clamp((int)(plazaAlpha * 255f), 0, 255);
                            if (pVal > 0)
                            {
                                alphaPlaza[pixelIndex] = Math.Max(alphaPlaza[pixelIndex], pVal);
                                hasPlaza = true;
                            }
                        }

                        // Decorative Checkerboard Border Ring (Layer 2)
                        if (dist >= rInner * 0.8f && dist < rOuter)
                        {
                            float ringAlpha = 1.0f;
                            if (dist < rInner)
                                ringAlpha = SmoothStep(rInner * 0.8f, rInner, dist);
                            else
                                ringAlpha = SmoothStep(rOuter, rOuter * 0.9f, dist);

                            byte cVal = (byte)Math.Clamp((int)(ringAlpha * 255f), 0, 255);
                            if (cVal > 0)
                            {
                                alphaCheckers[pixelIndex] = Math.Max(alphaCheckers[pixelIndex], cVal);
                                hasCheckers = true;
                            }
                        }
                    }
                }
            }
        }

        // Build active layer stack (Layer 0 is always Base)
        var textureList = new List<string>(4) { palette.BaseTexture };
        var layerList = new List<byte[]>(3);

        if (hasWalkway)
        {
            textureList.Add(palette.WalkwayTexture);
            layerList.Add(alphaWalkway);
        }

        if (hasCheckers && textureList.Count < 4)
        {
            textureList.Add(palette.CheckerBorderTexture);
            layerList.Add(alphaCheckers);
        }

        if (hasPlaza && textureList.Count < 4)
        {
            textureList.Add(palette.CenterPlazaTexture);
            layerList.Add(alphaPlaza);
        }

        return new ChunkAlphaData(textureList, layerList);
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        float diff = edge1 - edge0;
        if (MathF.Abs(diff) < 0.0001f)
            return x >= edge1 ? 1f : 0f;

        float t = Math.Clamp((x - edge0) / diff, 0f, 1f);
        return t * t * (3f - (2f * t));
    }
}
