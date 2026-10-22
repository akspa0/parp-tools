using System.Numerics;
using WoWViewer;

namespace WowViewer.Core.Tests;

/// <summary>
/// Spec 286 U1: one text-scale owner. These are policy assertions over the same code the viewer
/// draws with; loaded-scene retention, restart and on-screen appearance remain operator proof.
/// </summary>
public sealed class ViewPreferencePolicyTests
{
    [Theory]
    [InlineData(1.0f, 1.0f)]
    [InlineData(0.85f, 0.85f)]
    [InlineData(2.20f, 2.20f)]
    [InlineData(0.75f, 0.85f)]   // legacy loader accepted 0.75-2.50
    [InlineData(2.50f, 2.20f)]
    [InlineData(10f, 2.20f)]
    [InlineData(0.6f, 0.85f)]
    public void NormalizeUserScale_ClampsToTheSupportedRange(float value, float expected)
    {
        Assert.Equal(expected, ViewPreferencePolicy.NormalizeUserScale(value));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(0.5f)]   // the legacy loader treated <= 0.5 as damaged and used the default
    public void NormalizeUserScale_DamagedValuesFallBackToTheDefault(float value)
    {
        Assert.Equal(ViewPreferencePolicy.DefaultUserScale, ViewPreferencePolicy.NormalizeUserScale(value));
    }

    [Fact]
    public void NormalizeUserScale_IsIdempotent()
    {
        foreach (float raw in new[] { float.NaN, -3f, 0.3f, 0.75f, 0.85f, 1.35f, 2.2f, 2.5f, 99f })
        {
            float once = ViewPreferencePolicy.NormalizeUserScale(raw);
            Assert.Equal(once, ViewPreferencePolicy.NormalizeUserScale(once));
        }
    }

    [Theory]
    [InlineData(1.0f, false)]
    [InlineData(1.5f, false)]
    [InlineData(0.85f, false)]
    [InlineData(2.20f, false)]
    [InlineData(2.50f, true)]
    [InlineData(0.75f, true)]
    [InlineData(float.NaN, true)]
    public void NormalizePersisted_ReportsCorrectionOnlyForOutliers(float persisted, bool expectedCorrected)
    {
        float result = ViewPreferencePolicy.NormalizePersisted(persisted, out bool corrected);

        Assert.Equal(expectedCorrected, corrected);
        Assert.InRange(result, ViewPreferencePolicy.MinUserScale, ViewPreferencePolicy.MaxUserScale);

        // Normalising the corrected value again is a no-op: the correction happens once.
        ViewPreferencePolicy.NormalizePersisted(result, out bool correctedAgain);
        Assert.False(correctedAgain);
    }

    [Fact]
    public void Presets_AreAllInsideTheSupportedRangeAndAscending()
    {
        Assert.Contains(ViewPreferencePolicy.DefaultUserScale, ViewPreferencePolicy.Presets);
        float previous = 0f;
        foreach (float preset in ViewPreferencePolicy.Presets)
        {
            Assert.Equal(preset, ViewPreferencePolicy.NormalizeUserScale(preset));
            Assert.True(preset > previous);
            previous = preset;
        }
    }

    // ViewResetKind is internal to the viewer, so the truth table is one fact, not a theory.
    [Fact]
    public void Resolve_NamesExactlyWhatEachResetChanges()
    {
        Assert.Equal(new ViewResetEffect(ResetLayout: true, ResetUserScale: false),
            ViewPreferencePolicy.Resolve(ViewResetKind.LayoutOnly));
        Assert.Equal(new ViewResetEffect(ResetLayout: false, ResetUserScale: true),
            ViewPreferencePolicy.Resolve(ViewResetKind.TextScaleOnly));
        Assert.Equal(new ViewResetEffect(ResetLayout: true, ResetUserScale: true),
            ViewPreferencePolicy.Resolve(ViewResetKind.LayoutAndTextScale));
    }

    [Fact]
    public void Resolve_RejectsAnUndefinedKindInsteadOfResettingNothingSilently()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ViewPreferencePolicy.Resolve((ViewResetKind)99));
    }

    [Theory]
    [InlineData(1920, 1080, 1.0f, 1.0f)]
    [InlineData(1920, 1080, 1.5f, 1.5f)]
    [InlineData(2560, 1440, 1.0f, 1.3333334f)]
    [InlineData(3840, 2160, 1.0f, 2.0f)]
    [InlineData(3840, 2160, 1.5f, 3.0f)]
    [InlineData(3840, 2160, 2.20f, 4.0f)]   // 4.4 composes above the shell ceiling
    [InlineData(1920, 1080, 0.85f, 0.85f)]
    public void EffectiveFontScale_ComposesUserAdjustmentWithDisplayScale(int width, int height, float user, float expected)
    {
        float display = UiLayout.ScaleFor(new Vector2(width, height));

        Assert.Equal(expected, ViewPreferencePolicy.EffectiveFontScale(user, display), 3);
    }

    [Theory]
    [InlineData(1920, 1080)]
    [InlineData(2560, 1440)]
    [InlineData(3840, 2160)]
    public void ScaleReset_RestoresTheUserDefaultButKeepsDisplayAwareSizing(int width, int height)
    {
        float display = UiLayout.ScaleFor(new Vector2(width, height));
        float afterReset = ViewPreferencePolicy.EffectiveFontScale(ViewPreferencePolicy.DefaultUserScale, display);

        // Reset returns the user factor to 1.0; it must not collapse a 4K display to 1080p text.
        Assert.Equal(display, afterReset, 3);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(-2f)]
    [InlineData(0f)]
    public void EffectiveFontScale_SurvivesInvalidInputs(float bad)
    {
        float withBadUser = ViewPreferencePolicy.EffectiveFontScale(bad, 1f);
        float withBadDisplay = ViewPreferencePolicy.EffectiveFontScale(1f, bad);

        Assert.Equal(ViewPreferencePolicy.DefaultUserScale, withBadUser);
        Assert.Equal(1f, withBadDisplay);
    }

    [Theory]
    [InlineData(1.0f, "100%")]
    [InlineData(1.2f, "120%")]
    [InlineData(1.35f, "135%")]
    [InlineData(0.85f, "85%")]
    public void FormatPercent_MatchesThePresetLabels(float scale, string expected)
    {
        Assert.Equal(expected, ViewPreferencePolicy.FormatPercent(scale));
    }

    [Fact]
    public void IsSameScale_MatchesWithinSliderPrecision()
    {
        Assert.True(ViewPreferencePolicy.IsSameScale(1.2f, 1.2004f));
        Assert.False(ViewPreferencePolicy.IsSameScale(1.2f, 1.25f));
    }
}
