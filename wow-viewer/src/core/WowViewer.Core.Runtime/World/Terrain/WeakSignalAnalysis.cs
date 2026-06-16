namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WeakSignalAnalysis
{
    public int TileX { get; set; }
    public int TileY { get; set; }
    public float HeightRange { get; init; }
    public float MinHeight { get; init; }
    public float MaxHeight { get; init; }
    public bool IsWeakSignalCandidate { get; init; }
    public float AmplificationFactor { get; set; }
    public float AnchorHeight { get; set; }
    public string FactorSource { get; set; } = string.Empty;
    public string Severity { get; init; } = "none";
    public bool WasPatched { get; set; }
    public string? Error { get; set; }
}
