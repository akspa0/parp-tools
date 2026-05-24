namespace WowViewer.Core.Renderer.Scene;

public readonly record struct RenderVariant
{
    public static readonly RenderVariant Primary = new() { HideTerrain = false, HideLiquids = false, HideObjects = false };
    public static readonly RenderVariant NoLiquids = new() { HideTerrain = false, HideLiquids = true, HideObjects = false };
    public static readonly RenderVariant NoObjects = new() { HideTerrain = false, HideLiquids = false, HideObjects = true };
    public static readonly RenderVariant ObjectsOnly = new() { HideTerrain = true, HideLiquids = true, HideObjects = false };

    public bool HideTerrain { get; init; }
    public bool HideLiquids { get; init; }
    public bool HideObjects { get; init; }
}
