namespace WoWRollback.LkToAlphaModule;

public sealed record LkToAlphaOptions
{
    public bool SkipLiquids { get; init; } = true;
    public bool SkipWmos { get; init; }
    public string? TextureMappingPath { get; init; }
    public bool Validate { get; init; }
    public bool Verbose { get; init; }
    public bool VerboseLogging { get; init; }
    public int? ForceAreaId { get; init; }
    public float? DebugFlatMcvt { get; init; }
    public string? BaseTexture { get; init; }
    /// <summary>
    /// If true, MAIN cell offsets point to MHDR.data (+8 from letters); if false, point to MHDR letters.
    /// Use inspector to determine correct behavior for target client.
    /// </summary>
    public bool MainPointToMhdrData { get; init; } = false;
    // Asset gating: target listfile (e.g., 3.3.5) and optional modern listfile for diffing/reporting
    public string? TargetListfilePath { get; init; }
    public bool StrictTargetAssets { get; init; } = true;
    public string? ModernListfilePath { get; init; }
}
