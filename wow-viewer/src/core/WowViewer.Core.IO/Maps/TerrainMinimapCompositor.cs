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

        // A tile with no MTEX names is an untextured terrain tile, NOT an empty tile: it still
        // carries valid MCVT heights and MCNR normals, so it must render with its real terrain
        // shape (Lambert hillshading + cast shadows) over a neutral white base. Do not invent
        // colour from an unrelated catalog BLP, and do not short-circuit to a flat unlit white
        // image that discards the tile's shape.
        bool untextured = !HasDeclaredTileset(pack);

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

        // Analytic cast shadows are built once per tile over the 257x257 heightfield and then
        // sampled per output pixel, so the cost is independent of the export resolution.
        float[,]? castShadow = options.Lighting.ApplyCastShadows
            ? TerrainCastShadowMap.Compute(
                pack.Height257,
                options.Lighting.LightDirection,
                options.Lighting.CastShadowSoftness)
            : null;

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
                    detailV,
                    untextured);

                float lambert = ResolveInterpolatedLambert(
                    pack,
                    sourceX,
                    sourceY,
                    alphaWidth,
                    alphaHeight,
                    options.Lighting.LightDirection);
                // MCSH is the client-authored terrain-side static shadow map, and is a SEPARATE
                // signal from the analytic cast shadows below: MCSH is baked client data that
                // authored minimaps demonstrably do not contain, while the cast-shadow pass is
                // derived from MCVT against the current sun. Keep both opt-in and independent.
                float mcshMask = options.Lighting.ApplyMcshToMinimap
                    ? ResolveShadowMask(pack.McshShadowMask256, sourceX, sourceY, alphaWidth, alphaHeight)
                    : 0f;
                // Sampled from OUTPUT pixel space, not alpha space. MCAL can legitimately be far
                // coarser than the export (and is only 2x2 in some fixtures), whereas the cast
                // shadow map always spans the tile at heightfield resolution; routing it through
                // the alpha grid would quantise a 257-sample shadow down to the alpha's cell count.
                float castMask = castShadow is null
                    ? 0f
                    : ResolveCastShadow(castShadow, x, y, options.Resolution, options.Resolution);

                // Combine the two occluders multiplicatively on visibility rather than adding or
                // maxing their masks, so overlapping shadows cannot drive the surface past black
                // and each keeps its own independently calibrated strength.
                float visibility =
                    (1f - (Math.Clamp(mcshMask, 0f, 1f) * Math.Clamp(options.Lighting.McshShadowStrength, 0f, 1f)))
                    * (1f - (Math.Clamp(castMask, 0f, 1f) * Math.Clamp(options.Lighting.CastShadowStrength, 0f, 1f)));

                Vector3 lighting = TerrainLightingMath.Evaluate(
                    lambert,
                    options.Lighting.DirectionalColor,
                    options.Lighting.AmbientColor,
                    1f - visibility,
                    1f,
                    options.Lighting.ToneMapped,
                    options.Lighting.ToneMapExposure);

                image[x, y] = ToRgba(ApplyAlbedo(blended, lighting, options.Lighting));
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
        float detailV = 0f,
        bool untextured = false)
    {
        // Untextured tile (no MTEX): neutral white base so the terrain shape renders via
        // lighting instead of being discarded.
        if (untextured)
            return Vector3.One;

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

    /// <summary>
    /// Multiplies albedo by the evaluated light. In <see cref="TerrainMinimapLighting.LinearSpaceShading"/>
    /// mode the sRGB-authored BLP albedo is decoded to linear first and the result re-encoded, which
    /// is the only place the shading curve is physically correct; the legacy path multiplies
    /// directly in sRGB space and is retained so <see cref="MinimapShadingMatch"/>'s hour sweep and
    /// every other existing caller keep their exact previous response.
    /// </summary>
    private static Vector3 ApplyAlbedo(Vector3 albedoSrgb, Vector3 lighting, TerrainMinimapLighting profile)
    {
        if (!profile.LinearSpaceShading)
            return Vector3.Max(Vector3.Zero, albedoSrgb * lighting);

        float gain = float.IsFinite(profile.LinearLightGain) && profile.LinearLightGain > 0f
            ? profile.LinearLightGain
            : 1f;
        Vector3 linear = TerrainLightingMath.SrgbToLinear(albedoSrgb) * lighting * gain;
        return TerrainLightingMath.LinearToSrgb(Vector3.Max(Vector3.Zero, linear));
    }

    /// <summary>
    /// Samples the analytic cast-shadow map bilinearly. The map is built on the 257x257 heightfield
    /// while the pixel loop walks the output raster; nearest sampling across that rescale
    /// reintroduces exactly the stair-stepping the softness ramp exists to remove.
    /// </summary>
    private static float ResolveCastShadow(float[,] castShadow, int pixelX, int pixelY, int width, int height)
    {
        int size = castShadow.GetLength(0);
        if (size <= 1 || width <= 1 || height <= 1)
            return 0f;

        float row = pixelY * (size - 1f) / (height - 1f);
        float column = pixelX * (castShadow.GetLength(1) - 1f) / (width - 1f);
        float value = TerrainCastShadowMap.SampleBilinear(castShadow, row, column);
        return float.IsFinite(value) ? Math.Clamp(value, 0f, 1f) : 0f;
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
/// <param name="ApplyCastShadows">
/// Enables the analytic <see cref="TerrainCastShadowMap"/> pass: a ray march from every heightfield
/// sample toward <paramref name="LightDirection"/>, producing shadows that terrain throws across
/// other terrain. Independent of <paramref name="ApplyMcshToMinimap"/> -- Lambert alone only shades
/// slopes by their facing and can never darken flat ground sitting behind a ridge.
/// </param>
/// <param name="CastShadowStrength">
/// How dark a fully occluded cast shadow gets, 0..1. Deliberately separate from
/// <paramref name="McshShadowStrength"/>; see
/// <see cref="TerrainLightingMath.DefaultCastShadowStrength"/>.
/// </param>
/// <param name="LinearSpaceShading">
/// Decode sRGB albedo to linear, light it there, and re-encode on output. Off by default: every
/// pre-existing caller (notably <see cref="MinimapShadingMatch"/>'s hour sweep) is calibrated
/// against the legacy sRGB-space multiply and must not silently shift.
/// </param>
/// <param name="LinearLightGain">
/// Linear-space brightness gain, used instead of the tone map when
/// <paramref name="LinearSpaceShading"/> is set. Derive it with
/// <see cref="TerrainLightingMath.ResolveLinearLightGain"/> rather than hardcoding a value.
/// </param>
/// <param name="CastShadowSoftness">
/// Penumbra width of the cast-shadow ramp in world units. Lower values give crisper, narrower
/// shadow edges in crevices; see <see cref="TerrainCastShadowMap.DefaultSoftnessWorldUnits"/>.
/// </param>
public sealed record TerrainMinimapLighting(
    Vector3 LightDirection,
    Vector3 DirectionalColor,
    Vector3 AmbientColor,
    float McshShadowStrength,
    bool ApplyMcshToMinimap = false,
    bool ToneMapped = false,
    float ToneMapExposure = TerrainLightingMath.ToneMapExposure,
    bool ApplyCastShadows = false,
    float CastShadowStrength = TerrainLightingMath.DefaultCastShadowStrength,
    bool LinearSpaceShading = false,
    float LinearLightGain = 1f,
    float CastShadowSoftness = TerrainCastShadowMap.DefaultSoftnessWorldUnits)
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
    /// <remarks>
    /// Deliberately linear/untone-mapped: <see cref="MinimapShadingMatch"/> sweeps this factory
    /// across every hour of the day to recover an authored minimap's unknown capture time from its
    /// shading pattern, and needs the raw linear response across that whole sweep, not just at
    /// noon. Tone mapping is scoped to <see cref="CreateNoonWhiteGlobal"/> only.
    /// </remarks>
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
    /// <remarks>
    /// Tone mapping calibrated 2026-07-20 against the T010b 2.4.3/Expansion01 comparison set: the
    /// raw linear response measured a 2.41-3.18x (mean 2.79x) brightness deficit against authored
    /// minimaps, with R/G and B/G channel ratios matching authored almost exactly (e.g. tile 27,27:
    /// auth R/G=1.56 vs synth R/G=1.58) -- confirming the gap was pure underexposure, not a hue/tint
    /// bug. A flat linear multiplier fixed the deficit but clipped ~4% of pixels to hard white on
    /// steep, well-lit slopes; <see cref="TerrainLightingMath.ToneMapExposure"/>'s Reinhard curve
    /// (see <see cref="ToneMapped"/>) closes the same deficit without hard-clipping highlights.
    ///
    /// SUPERSEDED by <see cref="CreateShadedTerrain"/> for synthetic-minimap export: that
    /// calibration only ever checked mean brightness and clipping, never shading contrast, and the
    /// curve it settled on flattens the entire Lambert range into 12.8% of albedo.
    /// </remarks>
    public static TerrainMinimapLighting CreateNoonWhiteGlobal() =>
        CreateWhiteTopEdge(0.5f) with { ToneMapped = true };

    /// <summary>
    /// Production synthetic-minimap light with terrain shape actually visible: linear-space shading
    /// with an sRGB output encode, plus analytic sun-driven cast shadows from the tile's own
    /// heightfield. Replaces <see cref="CreateNoonWhiteGlobal"/> for export.
    /// </summary>
    /// <remarks>
    /// Two independent fixes, both aimed at the same symptom (synthesized tiles reading as flat,
    /// shadowless albedo next to authored minimaps):
    /// <list type="number">
    /// <item>The exposure-20 Reinhard curve compressed the hillshade to 12.8% of albedo. Replaced by
    /// a linear gain in linear light space -- same mid-tone, roughly 5x the shading range. See
    /// <see cref="TerrainLightingMath.SyntheticMinimapLinearLightGain"/>.</item>
    /// <item>Nothing in this codebase ever cast a shadow. Lambert N·L shades slopes by facing only;
    /// <see cref="TerrainCastShadowMap"/> adds the missing ridge-onto-ground occlusion.</item>
    /// </list>
    /// Neither the gain nor <see cref="DefaultCastShadowStrength"/> has been re-measured against a
    /// real authored comparison set yet. Re-running that check is a user-run step, and it must
    /// assert shading contrast alongside mean brightness.
    /// </remarks>
    public static TerrainMinimapLighting CreateShadedTerrain(float gameTime) =>
        CreateShadedTerrain(
            gameTime,
            TerrainLightingMath.DefaultSyntheticMinimapAmbient,
            TerrainLightingMath.DefaultCastShadowStrength);

    /// <summary>
    /// <see cref="CreateShadedTerrain(float)"/> with the two contrast controls exposed. The light
    /// gain is always DERIVED from the ambient term (see
    /// <see cref="TerrainLightingMath.ResolveLinearLightGain"/>), so raising or lowering ambient
    /// changes how deep shadows read without shifting the image's overall brightness -- these knobs
    /// are safe to sweep against an authored comparison one render at a time.
    /// </summary>
    public static TerrainMinimapLighting CreateShadedTerrain(
        float gameTime,
        float ambient,
        float castShadowStrength) =>
        CreateShadedTerrain(gameTime, new Vector3(ambient), castShadowStrength);

    /// <summary>
    /// <see cref="CreateShadedTerrain(float, float, float)"/> with a per-channel ambient, so sky
    /// light can be tinted. Shadowed ground is lit by ambient alone, so a non-neutral ambient is
    /// what gives shadows a different hue from lit ground rather than just a darker version of it.
    /// The gain is derived from the ambient's luminance.
    /// </summary>
    public static TerrainMinimapLighting CreateShadedTerrain(
        float gameTime,
        Vector3 ambientColor,
        float castShadowStrength,
        float castShadowSoftness = TerrainCastShadowMap.DefaultSoftnessWorldUnits)
    {
        float ambientLuminance = (ambientColor.X + ambientColor.Y + ambientColor.Z) / 3f;
        return CreateWhiteTopEdge(gameTime) with
        {
            AmbientColor = ambientColor,
            ToneMapped = false,
            LinearSpaceShading = true,
            LinearLightGain = TerrainLightingMath.ResolveLinearLightGain(ambientLuminance),
            ApplyCastShadows = true,
            CastShadowStrength = castShadowStrength,
            CastShadowSoftness = castShadowSoftness,
        };
    }
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
