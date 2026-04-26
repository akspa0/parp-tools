using WowViewer.Core.Runtime.World;

namespace WowViewer.App;

internal sealed class WowViewerWorldAssetInventory
{
    private readonly HashSet<string> _referencedWmoAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _referencedMdxAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _readyWmoAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _readyMdxAssetKeys = new(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _priorityMdxLoads = new();
    private readonly Queue<string> _pendingMdxLoads = new();
    private readonly HashSet<string> _queuedMdxLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _priorityQueuedMdxLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly Queue<string> _priorityWmoLoads = new();
    private readonly Queue<string> _pendingWmoLoads = new();
    private readonly HashSet<string> _queuedWmoLoads = new(StringComparer.OrdinalIgnoreCase);
    private readonly HashSet<string> _priorityQueuedWmoLoads = new(StringComparer.OrdinalIgnoreCase);
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
        _readyWmoAssetKeys.Clear();
        _readyMdxAssetKeys.Clear();
        _priorityMdxLoads.Clear();
        _pendingMdxLoads.Clear();
        _queuedMdxLoads.Clear();
        _priorityQueuedMdxLoads.Clear();
        _priorityWmoLoads.Clear();
        _pendingWmoLoads.Clear();
        _queuedWmoLoads.Clear();
        _priorityQueuedWmoLoads.Clear();
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

        HashSet<string> pendingAssetKeys = runtimeFrame.PendingAssetKeys
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        HashSet<string> visibleWmoKeys = runtimeFrame.Visibility.VisibleWmos
            .Select(static entry => entry.Instance.ModelKey)
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        HashSet<string> visibleMdxKeys = runtimeFrame.Visibility.VisibleMdx
            .Select(static entry => entry.Instance.ModelKey)
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

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
            {
                _referencedWmoAssetKeys.Add(instance.ModelKey);
                if (pendingAssetKeys.Contains(instance.ModelKey))
                {
                    if (visibleWmoKeys.Contains(instance.ModelKey))
                        PrioritizeWmoLoad(instance.ModelKey);
                    else
                        QueueWmoLoad(instance.ModelKey);
                }
                else
                {
                    MarkWmoReady(instance.ModelKey);
                }
            }
        }

        foreach (WorldObjectInstance instance in runtimeFrame.MdxInstances)
        {
            if (!string.IsNullOrWhiteSpace(instance.ModelKey))
            {
                _referencedMdxAssetKeys.Add(instance.ModelKey);
                if (pendingAssetKeys.Contains(instance.ModelKey))
                {
                    if (visibleMdxKeys.Contains(instance.ModelKey))
                        PrioritizeMdxLoad(instance.ModelKey);
                    else
                        QueueMdxLoad(instance.ModelKey);
                }
                else
                {
                    MarkMdxReady(instance.ModelKey);
                }
            }
        }
    }

    public void QueueMdxLoad(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || _readyMdxAssetKeys.Contains(modelKey))
            return;

        if (_queuedMdxLoads.Add(modelKey))
            _pendingMdxLoads.Enqueue(modelKey);
    }

    public void PrioritizeMdxLoad(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || _readyMdxAssetKeys.Contains(modelKey))
            return;

        if (_queuedMdxLoads.Add(modelKey))
            _pendingMdxLoads.Enqueue(modelKey);

        if (_priorityQueuedMdxLoads.Add(modelKey))
            _priorityMdxLoads.Enqueue(modelKey);
    }

    public void QueueWmoLoad(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || _readyWmoAssetKeys.Contains(modelKey))
            return;

        if (_queuedWmoLoads.Add(modelKey))
            _pendingWmoLoads.Enqueue(modelKey);
    }

    public void PrioritizeWmoLoad(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey) || _readyWmoAssetKeys.Contains(modelKey))
            return;

        if (_queuedWmoLoads.Add(modelKey))
            _pendingWmoLoads.Enqueue(modelKey);

        if (_priorityQueuedWmoLoads.Add(modelKey))
            _priorityWmoLoads.Enqueue(modelKey);
    }

    private void MarkMdxReady(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey))
            return;

        _readyMdxAssetKeys.Add(modelKey);
        _queuedMdxLoads.Remove(modelKey);
        _priorityQueuedMdxLoads.Remove(modelKey);
    }

    private void MarkWmoReady(string modelKey)
    {
        if (string.IsNullOrWhiteSpace(modelKey))
            return;

        _readyWmoAssetKeys.Add(modelKey);
        _queuedWmoLoads.Remove(modelKey);
        _priorityQueuedWmoLoads.Remove(modelKey);
    }

    private string[] BuildPendingAssetKeys()
    {
        List<string> orderedKeys = new();
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);

        AppendPendingKeys(_priorityWmoLoads, seen, orderedKeys);
        AppendPendingKeys(_priorityMdxLoads, seen, orderedKeys);
        AppendPendingKeys(_pendingWmoLoads, seen, orderedKeys);
        AppendPendingKeys(_pendingMdxLoads, seen, orderedKeys);

        return orderedKeys.ToArray();
    }

    private static void AppendPendingKeys(IEnumerable<string> source, ISet<string> seen, IList<string> target)
    {
        foreach (string key in source)
        {
            if (string.IsNullOrWhiteSpace(key) || !seen.Add(key))
                continue;

            target.Add(key);
        }
    }

    public WowViewerWorldAssetState CreateState()
    {
        return new WowViewerWorldAssetState(
            referencedWmoAssetCount: _referencedWmoAssetKeys.Count,
            referencedMdxAssetCount: _referencedMdxAssetKeys.Count,
            readyWmoAssetCount: _readyWmoAssetKeys.Count,
            readyMdxAssetCount: _readyMdxAssetKeys.Count,
            pendingWmoAssetCount: _queuedWmoLoads.Count,
            pendingMdxAssetCount: _queuedMdxLoads.Count,
            priorityPendingAssetCount: _priorityQueuedWmoLoads.Count + _priorityQueuedMdxLoads.Count,
            wmoInstanceCount: _wmoInstanceCount,
            mdxInstanceCount: _mdxInstanceCount,
            readyWmoCount: _readyWmoCount,
            readyMdxCount: _readyMdxCount,
            visibleWmoCount: _visibleWmoCount,
            visibleMdxCount: _visibleMdxCount,
            culledWmoCount: _culledWmoCount,
            culledMdxCount: _culledMdxCount,
            skyboxBackdropCount: _skyboxBackdropCount,
            pendingAssetKeys: BuildPendingAssetKeys());
    }
}