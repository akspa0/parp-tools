using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapLightingProvenanceTests
{
    [Fact]
    public void Infer_RecordsTintMcshCorrelationAndConservativeTimeBucket()
    {
        var baseline = new byte[8, 8, 3];
        var authored = new byte[8, 8, 3];
        var mcsh = new float[8, 8];
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                baseline[y, x, 0] = 100;
                baseline[y, x, 1] = 100;
                baseline[y, x, 2] = 100;
                bool shadowed = x < 2 && y < 2;
                mcsh[y, x] = shadowed ? 1f : 0f;
                float multiplier = shadowed ? 0.7f : 1f;
                authored[y, x, 0] = (byte)(80f * multiplier);
                authored[y, x, 1] = (byte)(100f * multiplier);
                authored[y, x, 2] = (byte)(60f * multiplier);
            }
        }

        MinimapLightingProvenance result = MinimapLightingProvenance.Infer(
            authored,
            baseline,
            mcsh,
            [
                new MinimapLightingTimeCandidate(6f, 0.6f, 0.7f, 1f, "fixture-dawn"),
                new MinimapLightingTimeCandidate(12f, 0.8f, 1f, 0.6f, "fixture-noon"),
            ]);

        Assert.Equal("baked_tint_and_mcsh_likely", result.InferenceStatus);
        Assert.NotNull(result.TintRed);
        Assert.NotNull(result.TintGreen);
        Assert.NotNull(result.TintBlue);
        Assert.Equal(0.8f, result.TintRed!.Value, 2);
        Assert.Equal(1f, result.TintGreen!.Value, 2);
        Assert.Equal(0.6f, result.TintBlue!.Value, 2);
        Assert.True(result.McshDarkeningCorrelation > 0.9f);
        Assert.Equal(12f, result.EstimatedTimeOfDayHours);
        Assert.Equal("inferred_global_lighting_chroma_match_not_capture_proof", result.TimeOfDayEvidence);
    }

    [Fact]
    public void Infer_DoesNotInventTimeWhenNoTintIsDetected()
    {
        var image = new byte[8, 8, 3];
        for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
                for (int channel = 0; channel < 3; channel++)
                    image[y, x, channel] = 100;

        MinimapLightingProvenance result = MinimapLightingProvenance.Infer(
            image,
            image,
            mcshShadowMask256: null,
            [new MinimapLightingTimeCandidate(12f, 1f, 1f, 1f, "fixture-noon")]);

        Assert.Equal("unlit_or_unclassified", result.InferenceStatus);
        Assert.Null(result.EstimatedTimeOfDayHours);
        Assert.Equal("no_baked_tint_detected", result.TimeOfDayEvidence);
    }
}
