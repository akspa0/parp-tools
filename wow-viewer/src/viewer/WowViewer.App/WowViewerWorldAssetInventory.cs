using WowViewer.Core.Runtime.World;

namespace WowViewer.App;

internal sealed class WowViewerWorldAssetInventory
{
    private readonly HashSet<string> _referencedWmoAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _referencedMdxAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private string[] _pendingAssetKeys = Array.Empty<string>();
    private int _wmoInstanceCount;
    private int _mdxInstanceCount;
    private int _readyWmoCount;
    private int _readyMdxCount;
    private int _visibleWmoCount;
    private int _visibleMdxCount;
    private int _culledWmoCount;
    private int _culledMdxCount;
    private int _skyboxBackdropCount;

    public void Reset()
    {
        _referencedWmoAssetKeys.Clear();
        _referencedMdxAssetKeys.Clear();
        _pendingAssetKeys = Array.Empty<string>();
        _wmoInstanceCount = 0;
        _mdxInstanceCount = 0;
        _readyWmoCount = 0;
        _readyMdxCount = 0;
        _visibleWmoCount = 0;
        _visibleMdxCount = 0;
        _culledWmoCount = 0;
        _culledMdxCount = 0;
        _skyboxBackdropCount = 0;
    }

    public void ObserveRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);

        Reset();

        _wmoInstanceCount = runtimeFrame.WmoInstances.Count;
        _mdxInstanceCount = runtimeFrame.MdxInstances.Count;
        _readyWmoCount = runtimeFrame.ReadyWmoCount;
        _readyMdxCount = runtimeFrame.ReadyMdxCount;
        _visibleWmoCount = runtimeFrame.Visibility.VisibleWmos.Count;
        _visibleMdxCount = runtimeFrame.Visibility.VisibleMdx.Count;
        _culledWmoCount = runtimeFrame.CulledWmoCount;
        _culledMdxCount = runtimeFrame.CulledMdxCount;
        _skyboxBackdropCount = runtimeFrame.SkyboxBackdropInstances.Count;

        foreach (WorldObjectInstance instance in runtimeFrame.WmoInstances)
        {
            if (!string.IsNullOrWhiteSpace(instance.ModelKey))
                _referencedWmoAssetKeys.Add(instance.ModelKey);
        }

        foreach (WorldObjectInstance instance in runtimeFrame.MdxInstances)
        {
            if (!string.IsNullOrWhiteSpace(instance.ModelKey))
                _referencedMdxAssetKeys.Add(instance.ModelKey);
        }

        _pendingAssetKeys = runtimeFrame.PendingAssetKeys
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    public WowViewerWorldAssetState CreateState()
    {
        return new WowViewerWorldAssetState(
            referencedWmoAssetCount: _referencedWmoAssetKeys.Count,
            referencedMdxAssetCount: _referencedMdxAssetKeys.Count,
            wmoInstanceCount: _wmoInstanceCount,
            mdxInstanceCount: _mdxInstanceCount,
            readyWmoCount: _readyWmoCount,
            readyMdxCount: _readyMdxCount,
            visibleWmoCount: _visibleWmoCount,
            visibleMdxCount: _visibleMdxCount,
            culledWmoCount: _culledWmoCount,
            culledMdxCount: _culledMdxCount,
            skyboxBackdropCount: _skyboxBackdropCount,
            pendingAssetKeys: _pendingAssetKeys);
    }
}