namespace WoWViewer;

/// <summary>What a named view reset changes. Every reset keeps the loaded client and scene.</summary>
internal enum ViewResetKind
{
    /// <summary>Panel placement, open tool windows and dock sizes only.</summary>
    LayoutOnly,
    /// <summary>The user's text-size adjustment only; display-aware sizing stays active.</summary>
    TextScaleOnly,
    /// <summary>Both of the above.</summary>
    LayoutAndTextScale,
}

internal readonly record struct ViewResetEffect(bool ResetLayout, bool ResetUserScale);

/// <summary>
/// Pure rules for the user's view preferences (Spec 286 U1). The viewer composes the interface
/// text size from two independent factors: the user's adjustment (this policy) and the display
/// scale (<see cref="UiLayout.ScaleFor"/>). Resetting the user factor therefore restores 1.0
/// without collapsing a 4K display to 1080p text.
/// </summary>
internal static class ViewPreferencePolicy
{
    internal const float MinUserScale = 0.85f;
    internal const float MaxUserScale = 2.20f;
    internal const float DefaultUserScale = 1.0f;

    // The shell never draws text smaller or larger than this after display scaling.
    internal const float MinEffectiveFontScale = 0.75f;
    internal const float MaxEffectiveFontScale = 4.0f;

    /// <summary>Legacy builds treated anything at or below this as damaged and used the default.</summary>
    private const float DamagedThreshold = 0.5f;

    /// <summary>One-click sizes offered next to the slider.</summary>
    internal static IReadOnlyList<float> Presets { get; } = [1.00f, 1.20f, 1.35f, 1.50f, 1.75f];

    /// <summary>
    /// A non-finite or damaged value recovers to the default rather than to the minimum, so a
    /// hand-edited file does not shrink the interface; anything else clamps to the supported range.
    /// </summary>
    internal static float NormalizeUserScale(float value) =>
        !float.IsFinite(value) || value <= DamagedThreshold
            ? DefaultUserScale
            : Math.Clamp(value, MinUserScale, MaxUserScale);

    /// <summary>Normalises a stored value and reports whether it had to be corrected.</summary>
    internal static float NormalizePersisted(float persisted, out bool corrected)
    {
        float normalized = NormalizeUserScale(persisted);
        corrected = !(normalized == persisted);
        return normalized;
    }

    /// <summary>The one font scale for the whole shell: user adjustment x display scale.</summary>
    internal static float EffectiveFontScale(float userScale, float displayScale)
    {
        float display = float.IsFinite(displayScale) && displayScale > 0f ? displayScale : 1f;
        return Math.Clamp(NormalizeUserScale(userScale) * display, MinEffectiveFontScale, MaxEffectiveFontScale);
    }

    internal static ViewResetEffect Resolve(ViewResetKind kind) => kind switch
    {
        ViewResetKind.LayoutOnly => new(ResetLayout: true, ResetUserScale: false),
        ViewResetKind.TextScaleOnly => new(ResetLayout: false, ResetUserScale: true),
        ViewResetKind.LayoutAndTextScale => new(ResetLayout: true, ResetUserScale: true),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown view reset."),
    };

    /// <summary>Slider-precision equality, so the current preset can be shown as selected.</summary>
    internal static bool IsSameScale(float a, float b) => MathF.Abs(a - b) < 0.005f;

    internal static string FormatPercent(float scale) => $"{MathF.Round(scale * 100f):0}%";
}
