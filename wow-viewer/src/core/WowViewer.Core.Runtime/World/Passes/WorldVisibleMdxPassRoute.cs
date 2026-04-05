namespace WowViewer.Core.Runtime.World.Passes;

public readonly record struct WorldVisibleMdxPassRoute(
    int VisibleMdxIndex,
    bool RequiresUnbatchedRender);