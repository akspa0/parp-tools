namespace WoWViewer.Rendering;

public readonly record struct WmoRenderStats(
    int DrawCalls,
    int BatchDrawCalls,
    int OpaqueBatchInstanceCount,
    int GroupFallbackDrawCalls,
    int LiquidDrawCalls,
    int DoodadSubmissions,
    int VisibleGroupSubmissions,
    int VisibleLiquidMeshes,
    int PortalTestedCount,
    int PortalFallbackCount,
    int PortalAdmittedGroupCount);
