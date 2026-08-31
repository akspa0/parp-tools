namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Options configuring chunk transposition, transformation, and placement translation.
/// </summary>
public sealed class ChunkTranspositionOptions
{
    public bool IncludeHeights { get; set; } = true;
    public bool RelativeHeights { get; set; } = true;
    public float HeightOffset { get; set; } = 0f;
    public bool IncludeTextures { get; set; } = true;
    public bool IncludeHoles { get; set; } = true;
    public bool IncludeLiquid { get; set; } = true;
    public bool IncludeVertexShading { get; set; } = true;
    public bool IncludeM2Placements { get; set; } = true;
    public bool IncludeWmoPlacements { get; set; } = true;
    public int RotationDegrees { get; set; } = 0; // 0, 90, 180, 270
    public bool MirrorX { get; set; } = false;
    public bool MirrorY { get; set; } = false;
    public bool OverwriteDestination { get; set; } = true;
}
