namespace WoWViewer;

/// <summary>
/// Spec 286 U0: immutable projection of the counters the status bar already collects.
/// <para>
/// The compact status line and the FPS-hover disclosure both read this snapshot. It is built once
/// per frame from existing state, so the hover path never triggers asset, DBC or disk I/O and the
/// viewer does not grow a second telemetry pipeline. Pure data, source-linked into the headless
/// test project.
/// </para>
/// </summary>
internal readonly record struct FooterDiagnosticsSnapshot(
    bool HasRuntimeScene,
    double Fps,
    string Area,
    double CpuMs,
    int LoadedTiles,
    int ChunksRendered,
    int ChunksCulled,
    int VisibleWmo,
    int TotalWmo,
    int VisibleMdx,
    int TotalMdx,
    int MdxLoaded,
    int MdxFailed,
    int ActiveAudio,
    int ResidentAudio,
    int PendingAssetLoads,
    string Coordinates,
    string Tile,
    string Facing,
    string StatusMessage)
{
    /// <summary>Snapshot used before a scene exists.</summary>
    public static FooterDiagnosticsSnapshot Empty { get; } = new(
        HasRuntimeScene: false,
        Fps: 0d,
        Area: "Area: Unknown",
        CpuMs: 0d,
        LoadedTiles: 0,
        ChunksRendered: 0,
        ChunksCulled: 0,
        VisibleWmo: 0,
        TotalWmo: 0,
        VisibleMdx: 0,
        TotalMdx: 0,
        MdxLoaded: 0,
        MdxFailed: 0,
        ActiveAudio: 0,
        ResidentAudio: 0,
        PendingAssetLoads: 0,
        Coordinates: string.Empty,
        Tile: "Tile: --",
        Facing: string.Empty,
        StatusMessage: string.Empty);

    /// <summary>FPS readout. Always present, even before a scene loads.</summary>
    public string FpsText => Fps > 0d ? $"FPS {Fps:0}" : "FPS --";

    /// <summary>
    /// Short suffix that fits beside the FPS readout. Detailed counters live in
    /// <see cref="DetailLines"/> and are shown on hover.
    /// </summary>
    public string CompactSuffix => HasRuntimeScene
        ? $" | {Area} | Pending {PendingAssetLoads}"
        : string.IsNullOrWhiteSpace(StatusMessage)
            ? " | Ready"
            : $" | {StatusMessage.Replace(Environment.NewLine, " ").Trim()}";

    /// <summary>Left-side camera readout, empty when no scene is loaded.</summary>
    public string LeftText
    {
        get
        {
            if (!HasRuntimeScene) return string.Empty;
            var parts = new List<string>(3);
            if (!string.IsNullOrEmpty(Coordinates)) parts.Add(Coordinates);
            if (!string.IsNullOrEmpty(Tile)) parts.Add(Tile);
            if (!string.IsNullOrEmpty(Facing)) parts.Add(Facing);
            return string.Join("  ", parts);
        }
    }

    /// <summary>Full runtime detail for the FPS-hover disclosure.</summary>
    public IReadOnlyList<string> DetailLines
    {
        get
        {
            var lines = new List<string>(12)
            {
                FpsText,
                Area,
            };

            if (!string.IsNullOrEmpty(Coordinates)) lines.Add(Coordinates);
            if (!string.IsNullOrEmpty(Tile)) lines.Add(Tile);
            if (!string.IsNullOrEmpty(Facing)) lines.Add(Facing);

            if (HasRuntimeScene)
            {
                lines.Add($"CPU: {CpuMs:0.0} ms");
                lines.Add($"Tiles loaded: {LoadedTiles}");
                lines.Add($"Chunks: {ChunksRendered} rendered / {ChunksCulled} culled");
                lines.Add($"WMO: {VisibleWmo} visible / {TotalWmo} total");
                lines.Add($"MDX: {VisibleMdx} visible / {TotalMdx} total ({MdxLoaded} ok / {MdxFailed} failed)");
                lines.Add($"Audio emitters: {ActiveAudio} active / {ResidentAudio} resident");
                lines.Add($"Pending asset loads: {PendingAssetLoads}");
            }
            else
            {
                lines.Add($"Tiles loaded: {LoadedTiles}");
            }

            return lines;
        }
    }
}
