using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// Options configuring chunk transposition, transformation, and placement translation.
/// Backed by <see cref="PhaseDataChannel"/> to reconcile the single unified channel model (Constitution II).
/// </summary>
public sealed class ChunkTranspositionOptions
{
    /// <summary>
    /// Granular channels selected for transposition.
    /// </summary>
    public PhaseDataChannel Channels { get; set; } = PhaseDataChannel.All;

    public bool IncludeHeights
    {
        get => Channels.HasFlag(PhaseDataChannel.Heightmap);
        set => SetChannel(PhaseDataChannel.Heightmap | PhaseDataChannel.Normals, value);
    }

    public bool RelativeHeights { get; set; } = true;
    public float HeightOffset { get; set; } = 0f;

    public bool IncludeTextures
    {
        get => Channels.HasFlag(PhaseDataChannel.TextureLayers);
        set => SetChannel(PhaseDataChannel.TextureLayers, value);
    }

    public bool IncludeHoles
    {
        get => Channels.HasFlag(PhaseDataChannel.Holes);
        set => SetChannel(PhaseDataChannel.Holes, value);
    }

    public bool IncludeLiquid
    {
        get => Channels.HasFlag(PhaseDataChannel.Liquid);
        set => SetChannel(PhaseDataChannel.Liquid, value);
    }

    public bool IncludeVertexShading
    {
        get => Channels.HasFlag(PhaseDataChannel.VertexColors) || Channels.HasFlag(PhaseDataChannel.Shadows);
        set => SetChannel(PhaseDataChannel.VertexColors | PhaseDataChannel.Shadows, value);
    }

    public bool IncludeM2Placements
    {
        get => Channels.HasFlag(PhaseDataChannel.Doodads);
        set => SetChannel(PhaseDataChannel.Doodads, value);
    }

    public bool IncludeWmoPlacements
    {
        get => Channels.HasFlag(PhaseDataChannel.WorldObjects);
        set => SetChannel(PhaseDataChannel.WorldObjects, value);
    }

    public int RotationDegrees { get; set; } = 0; // 0, 90, 180, 270
    public bool MirrorX { get; set; } = false;
    public bool MirrorY { get; set; } = false;
    public bool OverwriteDestination { get; set; } = true;

    private void SetChannel(PhaseDataChannel flag, bool enabled)
    {
        if (enabled)
            Channels |= flag;
        else
            Channels &= ~flag;
    }
}
