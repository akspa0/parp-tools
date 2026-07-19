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
        var textureSampler = new TerrainTextureSampler(
            texturesById,
            options.Resolution,
            16f * TerrainMinimapCompositionOptions.TextureRepeatsPerChunk);

        for (int y = 0; y < options.Resolution; y++)
        {
            int sourceY = ScaleCoordinate(y, options.Resolution, alphaHeight);
            int chunkY = Math.Min(textureIds.GetLength(0) - 1, sourceY * textureIds.GetLength(0) / alphaHeight);

            for (int x = 0; x < options.Resolution; x++)
            {
                int sourceX = ScaleCoordinate(x, options.Resolution, alphaWidth);
                int chunkX = Math.Min(textureIds.GetLength(1) - 1, sourceX * textureIds.GetLength(1) / alphaWidth);

                // Detail mode (Spec 113): sample real texels at the production terrain UV.
                // TerrainMeshBuilder supplies 0..1 per chunk and both production terrain shaders
                // multiply that by 8, so the diffuse material repeats eight times per chunk.
                float detailU = 0f, detailV = 0f;
                if (options.DetailTexels)
                {
                    float chunkPosX = (x + 0.5f) / options.Resolution * 16f * TerrainMinimapCompositionOptions.TextureRepeatsPerChunk;
                    float chunkPosY = (y + 0.5f) / options.Resolution * 16f * TerrainMinimapCompositionOptions.TextureRepeatsPerChunk;
                    detailU = chunkPosX - MathF.Floor(chunkPosX);
                    detailV = chunkPosY - MathF.Floor(chunkPosY);
                }

                Vector3 blended = BlendLayers(
                    alpha,
                    textureIds,
                    layerMask,
                    textureSampler,
                    sourceX,
                    sourceY,
                    chunkX,
                    chunkY,
                    options.DetailTexels,
                    detailU,
                    detailV);

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
        int chunkY,
        bool detailTexels = false,
        float detailU = 0f,
        float detailV = 0f)
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
            bool sampled = detailTexels
                ? textureSampler.TrySampleTexel(textureId, detailU, detailV, out Vector3 layerColor)
                : textureSampler.TrySample(textureId, out layerColor);
            if (textureId < 0 || !sampled)
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
        private readonly Dictionary<int, IReadOnlyList<byte[,,]>> _detailMipChains = [];
        private readonly int _outputResolution;
        private readonly float _textureRepeatsAcrossTile;

        public TerrainTextureSampler(
            IReadOnlyDictionary<int, byte[,,]> texturesById,
            int outputResolution,
            float textureRepeatsAcrossTile)
        {
            _texturesById = texturesById;
            _outputResolution = outputResolution;
            _textureRepeatsAcrossTile = textureRepeatsAcrossTile;
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

        /// <summary>
        /// Spec 113 detail mode: the real BLP texel at (u, v) in [0,1)², bilinear with wraparound
        /// (terrain textures tile). A deterministic box-filtered mip is selected from the decoded
        /// pixels for the output footprint; sampling the full-resolution base texture after an 8x
        /// minification would alias and recreate the exact moire this feature must avoid. Falls
        /// back to the same decodability rules as
        /// <see cref="TrySample"/> — a missing/undecodable texture is a miss, never a fabricated
        /// texel.
        /// </summary>
        public bool TrySampleTexel(int textureId, float u, float v, out Vector3 color)
        {
            color = Vector3.Zero;
            if (!_texturesById.TryGetValue(textureId, out byte[,,]? texture)
                || texture.GetLength(0) <= 0
                || texture.GetLength(1) <= 0
                || texture.GetLength(2) < 3)
            {
                return false;
            }

            IReadOnlyList<byte[,,]> mipChain = GetOrBuildMipChain(textureId, texture);
            float texelFootprint = MathF.Max(texture.GetLength(0), texture.GetLength(1))
                * _textureRepeatsAcrossTile
                / _outputResolution;
            int mipLevel = texelFootprint <= 1f
                ? 0
                : Math.Clamp((int)MathF.Floor(MathF.Log2(texelFootprint)), 0, mipChain.Count - 1);
            return TrySampleBilinear(mipChain[mipLevel], u, v, out color);
        }

        private IReadOnlyList<byte[,,]> GetOrBuildMipChain(int textureId, byte[,,] texture)
        {
            if (_detailMipChains.TryGetValue(textureId, out IReadOnlyList<byte[,,]>? cached))
                return cached;

            var levels = new List<byte[,,]> { texture };
            byte[,,] current = texture;
            while (current.GetLength(0) > 1 || current.GetLength(1) > 1)
            {
                current = BuildNextMip(current);
                levels.Add(current);
            }

            _detailMipChains[textureId] = levels;
            return levels;
        }

        private static byte[,,] BuildNextMip(byte[,,] source)
        {
            int sourceHeight = source.GetLength(0);
            int sourceWidth = source.GetLength(1);
            int targetHeight = Math.Max(1, (sourceHeight + 1) / 2);
            int targetWidth = Math.Max(1, (sourceWidth + 1) / 2);
            var target = new byte[targetHeight, targetWidth, 3];
            for (int y = 0; y < targetHeight; y++)
            {
                int y0 = Math.Min(sourceHeight - 1, y * 2);
                int y1 = (y0 + 1) % sourceHeight;
                for (int x = 0; x < targetWidth; x++)
                {
                    int x0 = Math.Min(sourceWidth - 1, x * 2);
                    int x1 = (x0 + 1) % sourceWidth;
                    for (int channel = 0; channel < 3; channel++)
                    {
                        int sum = source[y0, x0, channel]
                            + source[y0, x1, channel]
                            + source[y1, x0, channel]
                            + source[y1, x1, channel];
                        target[y, x, channel] = (byte)((sum + 2) / 4);
                    }
                }
            }

            return target;
        }

        private static bool TrySampleBilinear(byte[,,] texture, float u, float v, out Vector3 color)
        {
            color = Vector3.Zero;
            int height = texture.GetLength(0);
            int width = texture.GetLength(1);
            float px = u * width - 0.5f;
            float py = v * height - 0.5f;
            int x0 = (int)MathF.Floor(px);
            int y0 = (int)MathF.Floor(py);
            float fx = px - x0;
            float fy = py - y0;
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            // wraparound (tiling texture)
            x0 = ((x0 % width) + width) % width;
            x1 = ((x1 % width) + width) % width;
            y0 = ((y0 % height) + height) % height;
            y1 = ((y1 % height) + height) % height;

            Vector3 c00 = new(texture[y0, x0, 0], texture[y0, x0, 1], texture[y0, x0, 2]);
            Vector3 c10 = new(texture[y0, x1, 0], texture[y0, x1, 1], texture[y0, x1, 2]);
            Vector3 c01 = new(texture[y1, x0, 0], texture[y1, x0, 1], texture[y1, x0, 2]);
            Vector3 c11 = new(texture[y1, x1, 0], texture[y1, x1, 1], texture[y1, x1, 2]);
            Vector3 top = Vector3.Lerp(c00, c10, fx);
            Vector3 bottom = Vector3.Lerp(c01, c11, fx);
            color = Vector3.Lerp(top, bottom, fy) / 255f;
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

        Vector3 adtNormal = new(normals[y, x, 0], normals[y, x, 1], normals[y, x, 2]);
        // McnrNormalXyz is stored in ADT grid axes, while TerrainMinimapLighting.LightDirection is
        // expressed in renderer/world axes. Terrain grid X advances along -world Y and grid Y
        // advances along -world X. Dotting these spaces directly reverses the horizontal hillshade
        // for the fixed diagonal sun; convert through the shared numeric terrain contract first.
        return adtNormal.LengthSquared() > 1e-10f
            && float.IsFinite(adtNormal.X)
            && float.IsFinite(adtNormal.Y)
            && float.IsFinite(adtNormal.Z)
            ? TerrainNormalGeometry.TransformAdtNormalToRenderer(adtNormal)
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

    internal static int ScaleCoordinate(int coordinate, int sourceSize, int targetSize)
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
    /// Achromatic diagnostic light using the shared solar direction. The modest ambient term keeps
    /// terrain-readable slopes without tinting its materials.
    /// </summary>
    public static TerrainMinimapLighting CreateWhiteTopEdge(float gameTime)
    {
        return new TerrainMinimapLighting(
            TerrainSolarDirection.Evaluate(gameTime),
            Vector3.One,
            new Vector3(0.25f),
            0f);
    }

    /// <summary>
    /// Production synthetic-minimap light: one fixed noon, achromatic global light. Authored
    /// minimaps are not runtime world renders, so map LIT data and local/exact-build Light DBC
    /// profiles must not color-grade this composition.
    /// </summary>
    public static TerrainMinimapLighting CreateNoonWhiteGlobal() => CreateWhiteTopEdge(0.5f);
}

/// <summary>Controls deterministic terrain minimap composition without carrying client-source policy.</summary>
/// <remarks>
/// <see cref="DetailTexels"/> (Spec 113 US1): when true, layer colors come from real BLP texels at
/// the terrain UV (bilinear) instead of each texture's phase-independent average color. The
/// material-average default exists because texel sampling while downsampling hard to 256px produced
/// moire; at high output resolutions (1024+) the downsample is gentle and real texel detail becomes
/// viable — the premise of the minimap super-resolution HR target. The terrain UV convention
/// (eight texture repeats per chunk) matches the production renderer: TerrainMeshBuilder emits
/// 0..1 chunk UVs and the terrain shaders sample <c>vTexCoord * 8.0</c>.
/// </remarks>
public sealed record TerrainMinimapCompositionOptions(
    int Resolution,
    TerrainMinimapLighting Lighting,
    bool DetailTexels = false)
{
    public static TerrainMinimapCompositionOptions Default { get; } = new(
        TerrainMinimapCompositor.DefaultResolution,
        TerrainMinimapLighting.Neutral);

    /// <summary>Diffuse texture repeats per chunk edge, matching both production terrain shaders.</summary>
    public const float TextureRepeatsPerChunk = 8f;

    internal void Validate()
    {
        if (Resolution <= 0 || Resolution > 4096)
            throw new ArgumentOutOfRangeException(nameof(Resolution), "Resolution must be within 1..4096 pixels.");
        ArgumentNullException.ThrowIfNull(Lighting);
    }
}
