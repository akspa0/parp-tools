using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Deterministically synthesizes one terrain-only minimap tile from already-decoded client data.
/// This is a derived artifact: it composes BLP pixels with MCLY/MCAL weights and applies the
/// caller-declared lighting profile. It never reads or substitutes an authored minimap image.
/// </summary>
public static class TerrainMinimapCompositor
{
    public const int DefaultResolution = 256;

    public static Image<Rgba32> Compose(
        TerrainTileTensorPack pack,
        IReadOnlyDictionary<int, byte[,,]> texturesById,
        TerrainMinimapCompositionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(texturesById);

        options ??= TerrainMinimapCompositionOptions.Default;
        options.Validate();

        // No MTEX names is an explicit empty-tile state, not a stale material reference. Preserve
        // the empty terrain signal as unlit white rather than inventing colour from an unrelated
        // catalog BLP or allowing lighting to turn the placeholder grey.
        if (!HasDeclaredTileset(pack))
            return new Image<Rgba32>(options.Resolution, options.Resolution, new Rgba32(255, 255, 255, 255));

        float[,,]? alpha = pack.McalAlphaPack256;
        int[,,] textureIds = pack.MclyTextureIds ?? CreateFallbackTextureGrid();
        bool[,,]? layerMask = pack.MclyLayerMask;

        if (textureIds.GetLength(0) <= 0 || textureIds.GetLength(1) <= 0 || textureIds.GetLength(2) < 4)
        {
            // A few early-client tiles are geometrically readable but omit a usable MCLY table.
            // The caller supplies a declared catalog RGB proxy at ID zero in that case, so keep
            // the tile renderable instead of discarding its terrain, lighting, and liquid output.
            textureIds = CreateFallbackTextureGrid();
            layerMask = null;
        }
        if (alpha is not null && alpha.GetLength(2) < 4)
        {
            throw new InvalidDataException(
                "Synthetic minimap composition requires four MCAL layer channels when MCAL is present.");
        }

        // MCAL is optional for a base-only tile. Do not invent overlay weights when it is absent:
        // render the declared layer-zero material, retain the normal/lighting path, and leave the
        // tile exportable instead of failing an otherwise readable whole-map export.
        int alphaHeight = alpha?.GetLength(0) ?? DefaultResolution;
        int alphaWidth = alpha?.GetLength(1) ?? DefaultResolution;
        if (alphaWidth <= 0 || alphaHeight <= 0)
            throw new InvalidDataException("Synthetic minimap alpha dimensions must be positive.");

        var image = new Image<Rgba32>(options.Resolution, options.Resolution);
        var textureSampler = new TerrainTextureSampler(texturesById);

        for (int y = 0; y < options.Resolution; y++)
        {
            int sourceY = ScaleCoordinate(y, options.Resolution, alphaHeight);
            int chunkY = Math.Min(textureIds.GetLength(0) - 1, sourceY * textureIds.GetLength(0) / alphaHeight);

            for (int x = 0; x < options.Resolution; x++)
            {
                int sourceX = ScaleCoordinate(x, options.Resolution, alphaWidth);
                int chunkX = Math.Min(textureIds.GetLength(1) - 1, sourceX * textureIds.GetLength(1) / alphaWidth);

                Vector3 blended = BlendLayers(
                    alpha,
                    textureIds,
                    layerMask,
                    textureSampler,
                    sourceX,
                    sourceY,
                    chunkX,
                    chunkY);

                float lambert = ResolveInterpolatedLambert(
                    pack,
                    sourceX,
                    sourceY,
                    alphaWidth,
                    alphaHeight,
                    options.Lighting.LightDirection);
                // MCSH is a terrain-side static-shadow signal, not normal minimap content. Keep it
                // separate by default; the rare historically baked-shadow minimap is an explicit
                // diagnostic/export case and must never silently contaminate a general RGB corpus.
                float shadowMask = options.Lighting.ApplyMcshToMinimap
                    ? ResolveShadowMask(pack.McshShadowMask256, sourceX, sourceY, alphaWidth, alphaHeight)
                    : 0f;
                Vector3 lighting = TerrainLightingMath.Evaluate(
                    lambert,
                    options.Lighting.DirectionalColor,
                    options.Lighting.AmbientColor,
                    shadowMask,
                    options.Lighting.McshShadowStrength);

                Vector3 shaded = Vector3.Max(Vector3.Zero, blended * lighting);
                image[x, y] = ToRgba(shaded);
            }
        }

        return image;
    }

    private static bool HasDeclaredTileset(TerrainTileTensorPack pack)
    {
        foreach (string textureName in pack.MclyTextureNames)
        {
            if (!string.IsNullOrWhiteSpace(textureName))
                return true;
        }

        return false;
    }

    private static Vector3 BlendLayers(
        float[,,]? alpha,
        int[,,] textureIds,
        bool[,,]? layerMask,
        TerrainTextureSampler textureSampler,
        int sourceX,
        int sourceY,
        int chunkX,
        int chunkY)
    {
        Vector3 color = Vector3.Zero;
        bool hasColor = false;

        // Match the terrain fragment shader: layer zero is the opaque base and
        // each subsequent MCAL layer is composed over the current result in file
        // order. MCAL layers are not mutually-exclusive weights, so normalizing
        // their sum produces visibly incorrect mixes where overlays overlap.
        for (int layer = 0; layer < 4; layer++)
        {
            if (layerMask is not null
                && (chunkY >= layerMask.GetLength(0)
                    || chunkX >= layerMask.GetLength(1)
                    || layer >= layerMask.GetLength(2)
                    || !layerMask[chunkY, chunkX, layer]))
            {
                continue;
            }

            int textureId = textureIds[chunkY, chunkX, layer];
            if (textureId < 0 || !textureSampler.TrySample(textureId, out Vector3 layerColor))
                continue;

            if (layer == 0 || !hasColor)
            {
                color = layerColor;
                hasColor = true;
                continue;
            }

            if (alpha is null)
                continue;

            float overlayAlpha = Math.Clamp(alpha[sourceY, sourceX, layer], 0f, 1f);
            color = Vector3.Lerp(color, layerColor, overlayAlpha);
        }

        if (hasColor)
            return color;

        return textureSampler.TrySampleFallback(out color) ? color : Vector3.Zero;
    }

    private static int[,,] CreateFallbackTextureGrid()
    {
        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;
        return textureIds;
    }

    private sealed class TerrainTextureSampler
    {
        private readonly IReadOnlyDictionary<int, byte[,,]> _texturesById;
        private readonly Dictionary<int, Vector3> _materialColors = [];

        public TerrainTextureSampler(IReadOnlyDictionary<int, byte[,,]> texturesById)
        {
            _texturesById = texturesById;
        }

        public bool TrySample(int textureId, out Vector3 color)
        {
            color = Vector3.Zero;
            if (!_texturesById.TryGetValue(textureId, out byte[,,]? texture)
                || texture.GetLength(0) <= 0
                || texture.GetLength(1) <= 0
                || texture.GetLength(2) < 3)
            {
                return false;
            }

            // Sampling a low mip at one projected UV retains diffuse-repeat phase and creates
            // moire/interpolation patterns. A minimap is instead a material view: use each BLP's
            // phase-independent full-texture average, then retain spatial detail from chunk
            // selection and MCAL masks.
            if (!_materialColors.TryGetValue(textureId, out color))
            {
                color = CalculateAverageColor(texture);
                _materialColors[textureId] = color;
            }

            return true;
        }

        public bool TrySampleFallback(out Vector3 color)
        {
            foreach (int textureId in _texturesById.Keys.OrderBy(static id => id))
            {
                if (TrySample(textureId, out color))
                    return true;
            }

            color = Vector3.Zero;
            return false;
        }

        private static Vector3 CalculateAverageColor(byte[,,] texture)
        {
            int height = texture.GetLength(0);
            int width = texture.GetLength(1);
            long red = 0;
            long green = 0;
            long blue = 0;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    red += texture[y, x, 0];
                    green += texture[y, x, 1];
                    blue += texture[y, x, 2];
                }
            }

            float inversePixelCount = 1f / (height * width * 255f);
            return new Vector3(red * inversePixelCount, green * inversePixelCount, blue * inversePixelCount);
        }
    }

    private static float ResolveInterpolatedLambert(
        TerrainTileTensorPack pack,
        int sourceX,
        int sourceY,
        int sourceWidth,
        int sourceHeight,
        Vector3 lightDirection)
    {
        float[,,]? normals = pack.McnrNormalXyz;
        if (normals is null || normals.GetLength(2) < 3)
            return Lambert(Vector3.UnitZ, lightDirection);

        bool[,]? mask = pack.McnrMask257;
        int normalHeight = normals.GetLength(0);
        int normalWidth = normals.GetLength(1);

        // MCNR is a staggered 9x9/8x8 vertex lattice per chunk. The dense 257x257
        // compatibility raster leaves the alternating positions empty. Sampling that raster
        // directly turns every missing position into UnitZ and produces a checkerboard lighting
        // pattern. The native terrain path evaluates N.L at the five real lattice vertices in a
        // cell and interpolates the scalar across its four triangles.
        if (mask is null
            || mask.GetLength(0) != normalHeight
            || mask.GetLength(1) != normalWidth
            || normalWidth < 3
            || normalHeight < 3
            || ((normalWidth - 1) & 1) != 0
            || ((normalHeight - 1) & 1) != 0)
        {
            int normalY = ScaleCoordinate(sourceY, sourceHeight, normalHeight);
            int normalX = ScaleCoordinate(sourceX, sourceWidth, normalWidth);
            return Lambert(ReadNormal(normals, mask, normalX, normalY), lightDirection);
        }

        float normalXCoordinate = Math.Clamp(
            (sourceX + 0.5f) * (normalWidth - 1f) / sourceWidth,
            0f,
            normalWidth - 1f);
        float normalYCoordinate = Math.Clamp(
            (sourceY + 0.5f) * (normalHeight - 1f) / sourceHeight,
            0f,
            normalHeight - 1f);

        int cellX = Math.Min((int)(normalXCoordinate * 0.5f), (normalWidth - 3) / 2);
        int cellY = Math.Min((int)(normalYCoordinate * 0.5f), (normalHeight - 3) / 2);
        int baseX = cellX * 2;
        int baseY = cellY * 2;
        float u = Math.Clamp((normalXCoordinate - baseX) * 0.5f, 0f, 1f);
        float v = Math.Clamp((normalYCoordinate - baseY) * 0.5f, 0f, 1f);

        float topLeft = Lambert(ReadNormal(normals, mask, baseX, baseY), lightDirection);
        float topRight = Lambert(ReadNormal(normals, mask, baseX + 2, baseY), lightDirection);
        float center = Lambert(ReadNormal(normals, mask, baseX + 1, baseY + 1), lightDirection);
        float bottomLeft = Lambert(ReadNormal(normals, mask, baseX, baseY + 2), lightDirection);
        float bottomRight = Lambert(ReadNormal(normals, mask, baseX + 2, baseY + 2), lightDirection);

        if (v <= u && v <= 1f - u)
            return ((1f - u - v) * topLeft) + ((u - v) * topRight) + ((2f * v) * center);

        if (u >= v && u >= 1f - v)
            return ((u - v) * topRight) + ((u + v - 1f) * bottomRight) + ((2f * (1f - u)) * center);

        if (v >= u && v >= 1f - u)
            return ((v - u) * bottomLeft) + ((u + v - 1f) * bottomRight) + ((2f * (1f - v)) * center);

        return ((1f - u - v) * topLeft) + ((v - u) * bottomLeft) + ((2f * u) * center);
    }

    private static Vector3 ReadNormal(float[,,] normals, bool[,]? mask, int x, int y)
    {
        if ((uint)y >= normals.GetLength(0)
            || (uint)x >= normals.GetLength(1)
            || (mask is not null
                && ((uint)y >= mask.GetLength(0)
                    || (uint)x >= mask.GetLength(1)
                    || !mask[y, x])))
        {
            return Vector3.UnitZ;
        }

        Vector3 normal = new(normals[y, x, 0], normals[y, x, 1], normals[y, x, 2]);
        return normal.LengthSquared() > 1e-10f && float.IsFinite(normal.X) && float.IsFinite(normal.Y) && float.IsFinite(normal.Z)
            ? Vector3.Normalize(normal)
            : Vector3.UnitZ;
    }

    private static float Lambert(Vector3 normal, Vector3 lightDirection)
    {
        Vector3 light = lightDirection.LengthSquared() > 1e-10f
            ? Vector3.Normalize(lightDirection)
            : Vector3.UnitZ;
        return MathF.Max(0f, Vector3.Dot(normal, light));
    }

    private static float ResolveShadowMask(float[,]? shadows, int sourceX, int sourceY, int sourceWidth, int sourceHeight)
    {
        if (shadows is null || shadows.GetLength(0) == 0 || shadows.GetLength(1) == 0)
            return 0f;

        int shadowY = ScaleCoordinate(sourceY, sourceHeight, shadows.GetLength(0));
        int shadowX = ScaleCoordinate(sourceX, sourceWidth, shadows.GetLength(1));
        return float.IsFinite(shadows[shadowY, shadowX])
            ? Math.Clamp(shadows[shadowY, shadowX], 0f, 1f)
            : 0f;
    }

    private static int ScaleCoordinate(int coordinate, int sourceSize, int targetSize)
    {
        if (targetSize <= 1 || sourceSize <= 1)
            return 0;

        return Math.Clamp((int)MathF.Round(coordinate * (targetSize - 1f) / (sourceSize - 1f)), 0, targetSize - 1);
    }

    private static Rgba32 ToRgba(Vector3 color)
    {
        return new Rgba32(
            (byte)Math.Clamp(MathF.Round(color.X * 255f), 0f, 255f),
            (byte)Math.Clamp(MathF.Round(color.Y * 255f), 0f, 255f),
            (byte)Math.Clamp(MathF.Round(color.Z * 255f), 0f, 255f),
            255);
    }
}

/// <summary>Explicit lighting inputs for a derived minimap export.</summary>
public sealed record TerrainMinimapLighting(
    Vector3 LightDirection,
    Vector3 DirectionalColor,
    Vector3 AmbientColor,
    float McshShadowStrength,
    bool ApplyMcshToMinimap = false)
{
    /// <summary>Visible neutral composition for callers that intentionally do not grade lighting.</summary>
    public static TerrainMinimapLighting Neutral { get; } = new(
        Vector3.UnitZ,
        Vector3.Zero,
        Vector3.One,
        0f);

    /// <summary>
    /// The synthesized-minimap lighting contract: achromatic sunlight from the raster's top edge.
    /// LIT tracks are world/viewer data, not a source of minimap colour or direction. The modest
    /// achromatic ambient term keeps terrain-readable slopes without tinting the materials.
    /// </summary>
    public static TerrainMinimapLighting CreateWhiteTopEdge(float gameTime)
    {
        return new TerrainMinimapLighting(
            TerrainSolarDirection.Evaluate(gameTime),
            Vector3.One,
            new Vector3(0.25f),
            0f);
    }
}

/// <summary>Controls deterministic terrain minimap composition without carrying client-source policy.</summary>
public sealed record TerrainMinimapCompositionOptions(
    int Resolution,
    TerrainMinimapLighting Lighting)
{
    public static TerrainMinimapCompositionOptions Default { get; } = new(
        TerrainMinimapCompositor.DefaultResolution,
        TerrainMinimapLighting.Neutral);

    internal void Validate()
    {
        if (Resolution <= 0 || Resolution > 4096)
            throw new ArgumentOutOfRangeException(nameof(Resolution), "Resolution must be within 1..4096 pixels.");
        ArgumentNullException.ThrowIfNull(Lighting);
    }
}
