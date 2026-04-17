namespace WowViewer.Core.Mdx;

public readonly record struct MdxRenderCharacteristics(
    bool HasOpaqueRenderContent,
    bool HasTransparentRenderContent);

public static class MdxRenderCharacteristicsAnalyzer
{
    public static MdxRenderCharacteristics Analyze(MdxSummary summary)
    {
        ArgumentNullException.ThrowIfNull(summary);

        bool hasOpaque = false;
        bool hasTransparent = summary.ParticleEmitter2Count > 0 || summary.RibbonCount > 0;

        foreach (MdxMaterialSummary material in summary.Materials)
        {
            if (material.LayerCount == 0)
            {
                hasOpaque = true;
                continue;
            }

            foreach (MdxMaterialLayerSummary layer in material.Layers)
            {
                if (layer.BlendMode == 0 && layer.StaticAlpha >= 0.999f)
                    hasOpaque = true;
                else
                    hasTransparent = true;
            }
        }

        if (!hasOpaque && summary.GeosetCount > 0 && summary.MaterialCount == 0)
            hasOpaque = true;

        return new MdxRenderCharacteristics(hasOpaque, hasTransparent);
    }
}