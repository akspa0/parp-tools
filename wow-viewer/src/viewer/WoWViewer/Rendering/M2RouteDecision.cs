namespace WoWViewer.Rendering;

/// <summary>
/// Declared world M2 route type from spec data-model.md M2RouteDecision.
/// PrimaryRoute and AppliedRoute use this enum.
/// </summary>
public enum M2RouteType
{
    /// <summary>Primary: direct M2 adapter + skin.</summary>
    AdapterSkin,
    /// <summary>Primary for early builds: embedded root-profile geometry (no external .skin).</summary>
    AdapterEmbeddedProfile,
    /// <summary>Native static M2 renderer fed by an embedded legacy root profile.</summary>
    NativeEmbeddedProfile,
    /// <summary>Fallback: byte-level M2-to-MDX conversion.</summary>
    ConversionFallback,
    /// <summary>Legacy: standard MDX loader (non-M2 model).</summary>
    MdxDirect,
}

/// <summary>
/// Captures exactly how a world M2 instance reached its draw path.
/// Defined in data-model.md as M2RouteDecision.
/// </summary>
public sealed record M2RouteDecision(
    string ModelPath,
    string BuildProfileId,
    M2RouteType PrimaryRoute,
    M2RouteType AppliedRoute,
    string? SelectedSkinPath,
    string? FallbackReason,
    DateTime TimestampUtc)
{
    public static M2RouteDecision Create(string modelPath, string buildProfileId, M2RouteType primary, M2RouteType applied,
        string? selectedSkinPath = null, string? fallbackReason = null)
        => new(modelPath, buildProfileId, primary, applied, selectedSkinPath, fallbackReason, DateTime.UtcNow);
}
