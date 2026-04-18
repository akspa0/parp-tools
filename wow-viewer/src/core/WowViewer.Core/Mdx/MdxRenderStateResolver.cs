using System.Numerics;

namespace WowViewer.Core.Mdx;

public readonly record struct MdxResolvedMaterialState(
    string? TexturePath,
    uint ReplaceableId,
    int TransformId,
    int CoordId,
    bool IsTransparent,
    bool IsAdditive,
    bool DepthWrite,
    bool AlphaCutout,
    float Alpha,
    uint BlendMode);

public readonly record struct MdxResolvedTextureTransform(
    bool UsesTransform,
    Vector2 Translation,
    Vector2 Scale,
    Vector2 RotationRow0,
    Vector2 RotationRow1)
{
    public static MdxResolvedTextureTransform Identity { get; } =
        new(false, Vector2.Zero, Vector2.One, new Vector2(1.0f, 0.0f), new Vector2(0.0f, 1.0f));
}

public readonly record struct MdxResolvedGeosetRenderState(
    bool ReceivesLighting,
    bool DepthTest,
    bool DepthWrite,
    Vector3 BaseColor,
    float Alpha);

public static class MdxRenderStateResolver
{
    private const uint MdxBlendModeTransparentKey = 1;
    private const uint MdxBlendModeAdditive = 3;
    private const uint MdxBlendModeAddAlpha = 4;
    private const uint MdxGeosetFlagUnshaded = 0x1;
    private const uint MdxGeosetFlagNoDepthTest = 0x40;
    private const uint MdxGeosetFlagNoDepthSet = 0x80;

    public static MdxResolvedMaterialState ResolveMaterial(MdxSummary summary, int materialId)
    {
        ArgumentNullException.ThrowIfNull(summary);

        if (materialId < 0 || materialId >= summary.Materials.Count)
            return new MdxResolvedMaterialState(null, 0, -1, 0, false, false, true, false, 1.0f, 0);

        MdxMaterialSummary material = summary.Materials[materialId];
        if (material.LayerCount == 0)
            return new MdxResolvedMaterialState(null, 0, -1, 0, false, false, true, false, 1.0f, 0);

        MdxMaterialLayerSummary layer = material.Layers[0];
        string? texturePath = null;
        uint replaceableId = 0;
        if (layer.TextureId >= 0 && layer.TextureId < summary.Textures.Count)
        {
            MdxTextureSummary texture = summary.Textures[layer.TextureId];
            texturePath = texture.Path;
            replaceableId = texture.ReplaceableId;
        }

        float alpha = Math.Clamp(layer.StaticAlpha <= 0.0f ? 1.0f : layer.StaticAlpha, 0.0f, 1.0f);
        bool isTransparent = layer.BlendMode != 0 || alpha < 0.999f;
        bool alphaCutout = layer.BlendMode == MdxBlendModeTransparentKey;
        bool isAdditive = layer.BlendMode is MdxBlendModeAdditive or MdxBlendModeAddAlpha;
        bool depthWrite = !isTransparent || alphaCutout;
        return new MdxResolvedMaterialState(texturePath, replaceableId, layer.TransformId, layer.CoordId, isTransparent, isAdditive, depthWrite, alphaCutout, alpha, layer.BlendMode);
    }

    public static MdxResolvedGeosetRenderState ResolveGeosetRenderState(
        MdxSummary summary,
        MdxGeosetAnimationFile? geosetAnimationFile,
        int sequenceIndex,
        int timeMs,
        MdxGeosetGeometry geoset,
        MdxResolvedMaterialState material)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentNullException.ThrowIfNull(geoset);

        bool receivesLighting = (geoset.Flags & MdxGeosetFlagUnshaded) == 0;
        bool depthTest = (geoset.Flags & MdxGeosetFlagNoDepthTest) == 0;
        bool depthWrite = material.DepthWrite && (geoset.Flags & MdxGeosetFlagNoDepthSet) == 0;
        Vector3 baseColor = Vector3.One;
        float alpha = material.Alpha;

        if (TryGetGeosetAnimation(geosetAnimationFile, geoset.Index, out MdxGeosetAnimation? geosetAnimation) && geosetAnimation is not null)
        {
            alpha *= Math.Clamp(
                MdxAnimationSampler.SampleScalarTrack(geosetAnimation.AlphaTrack, summary, sequenceIndex, timeMs, geosetAnimation.StaticAlpha),
                0.0f,
                1.0f);
            if (geosetAnimation.UsesStaticColor || geosetAnimation.ColorTrack is not null)
                baseColor *= MdxAnimationSampler.SampleColorTrack(geosetAnimation.ColorTrack, summary, sequenceIndex, timeMs, geosetAnimation.StaticColor);
        }
        else if (TryGetGeosetAnimation(summary, geoset.Index, out MdxGeosetAnimationSummary? geosetAnimationSummary) && geosetAnimationSummary is not null)
        {
            alpha *= Math.Clamp(geosetAnimationSummary.StaticAlpha, 0.0f, 1.0f);
            if (geosetAnimationSummary.UsesStaticColor)
                baseColor *= geosetAnimationSummary.StaticColor;
        }

        return new MdxResolvedGeosetRenderState(receivesLighting, depthTest, depthWrite, baseColor, Math.Clamp(alpha, 0.0f, 1.0f));
    }

    public static MdxResolvedTextureTransform ResolveTextureTransform(
        MdxSummary summary,
        MdxTextureAnimationFile? textureAnimationFile,
        int sequenceIndex,
        int timeMs,
        MdxResolvedMaterialState material)
    {
        ArgumentNullException.ThrowIfNull(summary);

        if (textureAnimationFile is null || material.TransformId < 0 || material.TransformId >= textureAnimationFile.TextureAnimationCount)
            return MdxResolvedTextureTransform.Identity;

        MdxTextureAnimation animation = textureAnimationFile.TextureAnimations[material.TransformId];
        Vector3 translation = MdxAnimationSampler.SampleVector3Track(animation.TranslationTrack, summary, sequenceIndex, timeMs, Vector3.Zero);
        Vector3 scale = MdxAnimationSampler.SampleVector3Track(animation.ScalingTrack, summary, sequenceIndex, timeMs, Vector3.One);
        Quaternion rotation = MdxAnimationSampler.SampleQuaternionTrack(animation.RotationTrack, summary, sequenceIndex, timeMs, Quaternion.Identity);
        Matrix4x4 rotationMatrix = Matrix4x4.CreateFromQuaternion(rotation);

        bool usesTransform = animation.HasTranslationTrack || animation.HasRotationTrack || animation.HasScalingTrack;
        return new MdxResolvedTextureTransform(
            usesTransform,
            new Vector2(translation.X, translation.Y),
            new Vector2(scale.X, scale.Y),
            new Vector2(rotationMatrix.M11, rotationMatrix.M12),
            new Vector2(rotationMatrix.M21, rotationMatrix.M22));
    }

    private static bool TryGetGeosetAnimation(MdxSummary summary, int geosetIndex, out MdxGeosetAnimationSummary? geosetAnimation)
    {
        foreach (MdxGeosetAnimationSummary candidate in summary.GeosetAnimations)
        {
            if (candidate.GeosetId != (uint)geosetIndex)
                continue;

            geosetAnimation = candidate;
            return true;
        }

        geosetAnimation = null;
        return false;
    }

    private static bool TryGetGeosetAnimation(MdxGeosetAnimationFile? geosetAnimationFile, int geosetIndex, out MdxGeosetAnimation? geosetAnimation)
    {
        if (geosetAnimationFile is not null)
        {
            foreach (MdxGeosetAnimation candidate in geosetAnimationFile.GeosetAnimations)
            {
                if (candidate.GeosetId != (uint)geosetIndex)
                    continue;

                geosetAnimation = candidate;
                return true;
            }
        }

        geosetAnimation = null;
        return false;
    }
}
