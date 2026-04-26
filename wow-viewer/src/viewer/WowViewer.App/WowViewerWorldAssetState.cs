namespace WowViewer.App;

internal sealed class WowViewerWorldAssetState
{
    public static readonly WowViewerWorldAssetState Empty = new(
        referencedWmoAssetCount: 0,
        referencedMdxAssetCount: 0,
        wmoInstanceCount: 0,
        mdxInstanceCount: 0,
        readyWmoCount: 0,
        readyMdxCount: 0,
        visibleWmoCount: 0,
        visibleMdxCount: 0,
        culledWmoCount: 0,
        culledMdxCount: 0,
        skyboxBackdropCount: 0,
        pendingAssetKeys: Array.Empty<string>());

    public WowViewerWorldAssetState(
        int referencedWmoAssetCount,
        int referencedMdxAssetCount,
        int wmoInstanceCount,
        int mdxInstanceCount,
        int readyWmoCount,
        int readyMdxCount,
        int visibleWmoCount,
        int visibleMdxCount,
        int culledWmoCount,
        int culledMdxCount,
        int skyboxBackdropCount,
        IReadOnlyList<string> pendingAssetKeys)
    {
        ReferencedWmoAssetCount = referencedWmoAssetCount;
        ReferencedMdxAssetCount = referencedMdxAssetCount;
        WmoInstanceCount = wmoInstanceCount;
        MdxInstanceCount = mdxInstanceCount;
        ReadyWmoCount = readyWmoCount;
        ReadyMdxCount = readyMdxCount;
        VisibleWmoCount = visibleWmoCount;
        VisibleMdxCount = visibleMdxCount;
        CulledWmoCount = culledWmoCount;
        CulledMdxCount = culledMdxCount;
        SkyboxBackdropCount = skyboxBackdropCount;
        PendingAssetKeys = pendingAssetKeys;
    }

    public int ReferencedWmoAssetCount { get; }

    public int ReferencedMdxAssetCount { get; }

    public int WmoInstanceCount { get; }

    public int MdxInstanceCount { get; }

    public int ReadyWmoCount { get; }

    public int ReadyMdxCount { get; }

    public int VisibleWmoCount { get; }

    public int VisibleMdxCount { get; }

    public int CulledWmoCount { get; }

    public int CulledMdxCount { get; }

    public int SkyboxBackdropCount { get; }

    public IReadOnlyList<string> PendingAssetKeys { get; }

    public int PendingAssetLoadCount => PendingAssetKeys.Count;

    public int VisibleObjectCount => VisibleWmoCount + VisibleMdxCount;

    public static WowViewerWorldAssetState FromRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);
        return new WowViewerWorldAssetState(
            runtimeFrame.WmoInstances.Select(static instance => instance.ModelKey).Where(static key => !string.IsNullOrWhiteSpace(key)).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
            runtimeFrame.MdxInstances.Select(static instance => instance.ModelKey).Where(static key => !string.IsNullOrWhiteSpace(key)).Distinct(StringComparer.OrdinalIgnoreCase).Count(),
            runtimeFrame.WmoInstances.Count,
            runtimeFrame.MdxInstances.Count,
            runtimeFrame.ReadyWmoCount,
            runtimeFrame.ReadyMdxCount,
            runtimeFrame.Visibility.VisibleWmos.Count,
            runtimeFrame.Visibility.VisibleMdx.Count,
            runtimeFrame.CulledWmoCount,
            runtimeFrame.CulledMdxCount,
            runtimeFrame.SkyboxBackdropInstances.Count,
            runtimeFrame.PendingAssetKeys.ToArray());
    }
}