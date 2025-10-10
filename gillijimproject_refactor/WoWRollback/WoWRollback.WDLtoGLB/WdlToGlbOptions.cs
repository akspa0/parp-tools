namespace WoWRollback.WDLtoGLB;

/// <summary>
/// Options for controlling WDL → GLB conversion.
/// </summary>
public sealed class WdlToGlbOptions
{
    /// <summary>
    /// Logical map name used for resolving per-tile minimap images (e.g., Map_X_Y.png).
    /// </summary>
    public string? MapName { get; init; }

    public string? TextureOverridePath { get; init; }
    /// <summary>
    /// Optional folder containing per-tile minimap images mapping 1:1 to tiles.
    /// Supported file names (case-insensitive):
    ///   {MapName}_{x}_{y}.png|jpg|jpeg|webp   or   tile_{x}_{y}.png|jpg|jpeg|webp
    /// </summary>
    public string? MinimapFolder { get; init; }

    /// <summary>
    /// Root of minimap data that contains md5translate.trs (or .txt) and the textures/minimap tree.
    /// When provided (or TRS path provided), TRS resolution is preferred over filename heuristics.
    /// Example: test_data/0.5.3/tree/textures/Minimap
    /// </summary>
    public string? MinimapRoot { get; init; }

    /// <summary>
    /// Optional explicit path to md5translate.trs (or .txt). If not provided, auto-detected under MinimapRoot.
    /// </summary>
    public string? TrsPath { get; init; }

    public bool InvertX { get; init; } = false;
    public float Scale { get; init; } = 1.0f;
    public bool SRGB { get; init; } = true;
    public int? AnisotropyHint { get; init; }

    /// <summary>
    /// Apply a +90° rotation around Y to the whole export (merged and per-tile) after placement.
    /// </summary>
    public bool RotateY90 { get; init; } = true;

    /// <summary>
    /// Invert Z axis globally (mirror across X/Y plane). Winding will be corrected.
    /// </summary>
    public bool InvertZ { get; init; } = true;
}
