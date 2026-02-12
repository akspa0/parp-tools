namespace WoWRollback.LkToAlphaModule;

public sealed record LkToAlphaOptions
{
    public bool SkipLiquids { get; init; } = true;
    public bool SkipWmos { get; init; }
    public bool SkipM2 { get; init; }
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
    public string? ExportMccvDir { get; init; }
    public bool PreferTexLayers { get; init; }
    public bool RawCopyLkLayers { get; init; }

    public bool ExtractAssets { get; init; }
    public string? AssetsOut { get; init; }
    public string AssetScope { get; init; } = "textures";
    public bool IncludeMissingModels { get; init; } = true;
    public bool WriteAssetManifest { get; init; } = true;
    public string? RunTag { get; init; }

    public bool ConvertModelsToLegacy { get; init; }
    public bool ConvertWmosToLegacy { get; init; }
    public string? AssetsSourceRoot { get; init; }
}
