using WowViewer.Core.Runtime.World;
using WowViewer.Core.Runtime.World.Visibility;

namespace WowViewer.App;

internal enum WorldSelectionKind
{
    Wmo = 0,
    Mdx = 1,
}

internal readonly record struct WorldObjectSelection(
    WorldSelectionKind Kind,
    int TileX,
    int TileY,
    int PlacementEntryIndex,
    int UniqueId,
    string ModelKey);

internal readonly record struct WorldNavigatorEntry(
    WorldSelectionKind Kind,
    WorldObjectInstance Instance,
    bool IsVisible,
    bool AssetReady,
    float? CenterDistance,
    bool IsTaxiActor,
    bool HasOpaqueRoute,
    bool HasTransparentRoute,
    bool RequiresUnbatchedRender,
    bool WasAnimated);

internal sealed class WowViewerWorldNavigatorState
{
    private readonly Dictionary<WorldObjectSelection, WorldNavigatorEntry> _entriesBySelection;

    public static readonly WowViewerWorldNavigatorState Empty = new(-1, -1, Array.Empty<WorldNavigatorEntry>(), null);

    private WowViewerWorldNavigatorState(
        int selectedTileX,
        int selectedTileY,
        IReadOnlyList<WorldNavigatorEntry> entries,
        WorldObjectSelection? defaultSelection)
    {
        SelectedTileX = selectedTileX;
        SelectedTileY = selectedTileY;
        Entries = entries;
        DefaultSelection = defaultSelection;
        _entriesBySelection = entries
            .Select(entry => new KeyValuePair<WorldObjectSelection, WorldNavigatorEntry>(CreateSelection(entry, selectedTileX, selectedTileY), entry))
            .ToDictionary(static pair => pair.Key, static pair => pair.Value);
    }

    public int SelectedTileX { get; }

    public int SelectedTileY { get; }

    public IReadOnlyList<WorldNavigatorEntry> Entries { get; }

    public WorldObjectSelection? DefaultSelection { get; }

    public bool HasEntries => Entries.Count > 0;

    public WorldObjectSelection CreateSelection(WorldNavigatorEntry entry)
    {
        return CreateSelection(entry, SelectedTileX, SelectedTileY);
    }

    public bool TryResolveEntry(WorldObjectSelection selection, out WorldNavigatorEntry entry)
    {
        if (selection.TileX != SelectedTileX || selection.TileY != SelectedTileY)
        {
            entry = default;
            return false;
        }

        return _entriesBySelection.TryGetValue(selection, out entry);
    }

    public static WowViewerWorldNavigatorState FromRuntimeFrame(WowViewerWorldRuntimeFrameResult runtimeFrame)
    {
        ArgumentNullException.ThrowIfNull(runtimeFrame);

        List<WorldNavigatorEntry> entries = BuildEntries(runtimeFrame);
        WorldObjectSelection? defaultSelection = SelectDefault(entries, runtimeFrame.SelectedTileX, runtimeFrame.SelectedTileY);
        return new WowViewerWorldNavigatorState(runtimeFrame.SelectedTileX, runtimeFrame.SelectedTileY, entries, defaultSelection);
    }

    private static List<WorldNavigatorEntry> BuildEntries(WowViewerWorldRuntimeFrameResult result)
    {
        Dictionary<int, WorldVisibleWmoEntry> visibleWmoByIndex = result.Visibility.VisibleWmos.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        Dictionary<int, WorldVisibleMdxEntry> visibleMdxByIndex = result.Visibility.VisibleMdx.ToDictionary(static entry => entry.Instance.PlacementEntryIndex);
        HashSet<int> opaqueRoutes = result.PassFrame.OpaqueVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> transparentRoutes = result.PassFrame.TransparentVisibleMdxRoutes.Select(static route => route.VisibleMdxIndex).ToHashSet();
        HashSet<int> unbatchedRoutes = result.PassFrame.UnbatchedVisibleMdxIndices;
        HashSet<string> animatedModels = result.PassFrame.UpdatedMdxModelKeys;

        List<WorldNavigatorEntry> entries = new(result.WmoInstances.Count + result.MdxInstances.Count);
        foreach (WorldObjectInstance instance in result.WmoInstances)
        {
            if (visibleWmoByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleWmoEntry visibleWmo))
                entries.Add(CreateEntry(result, WorldSelectionKind.Wmo, visibleWmo.Instance, visibleWmo.CenterDistanceSq, isVisible: true, isTaxiActor: false));
            else
                entries.Add(CreateEntry(result, WorldSelectionKind.Wmo, instance, centerDistanceSq: null, isVisible: false, isTaxiActor: false));
        }

        for (int index = 0; index < result.MdxInstances.Count; index++)
        {
            WorldObjectInstance instance = result.MdxInstances[index];
            if (visibleMdxByIndex.TryGetValue(instance.PlacementEntryIndex, out WorldVisibleMdxEntry visibleMdx))
            {
                entries.Add(CreateEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    visibleMdx.Instance,
                    visibleMdx.CenterDistanceSq,
                    isVisible: true,
                    visibleMdx.IsTaxiActor,
                    opaqueRoutes.Contains(index),
                    transparentRoutes.Contains(index),
                    unbatchedRoutes.Contains(index),
                    animatedModels.Contains(visibleMdx.Instance.ModelKey)));
            }
            else
            {
                entries.Add(CreateEntry(
                    result,
                    WorldSelectionKind.Mdx,
                    instance,
                    centerDistanceSq: null,
                    isVisible: false,
                    isTaxiActor: false,
                    hasOpaqueRoute: false,
                    hasTransparentRoute: false,
                    requiresUnbatchedRender: false,
                    wasAnimated: animatedModels.Contains(instance.ModelKey)));
            }
        }

        return entries;
    }

    private static WorldNavigatorEntry CreateEntry(
        WowViewerWorldRuntimeFrameResult result,
        WorldSelectionKind kind,
        WorldObjectInstance instance,
        float? centerDistanceSq,
        bool isVisible,
        bool isTaxiActor,
        bool hasOpaqueRoute = false,
        bool hasTransparentRoute = false,
        bool requiresUnbatchedRender = false,
        bool wasAnimated = false)
    {
        bool assetReady = kind == WorldSelectionKind.Wmo
            ? result.WmoInstances.Any(candidate => candidate.PlacementEntryIndex == instance.PlacementEntryIndex && candidate.BoundsResolved)
            : result.MdxInstances.Any(candidate => candidate.PlacementEntryIndex == instance.PlacementEntryIndex && candidate.BoundsResolved);

        return new WorldNavigatorEntry(
            kind,
            instance,
            isVisible,
            assetReady,
            centerDistanceSq,
            isTaxiActor,
            hasOpaqueRoute,
            hasTransparentRoute,
            requiresUnbatchedRender,
            wasAnimated);
    }

    private static WorldObjectSelection? SelectDefault(IReadOnlyList<WorldNavigatorEntry> entries, int tileX, int tileY)
    {
        WorldNavigatorEntry? firstVisibleWmo = entries
            .Where(static entry => entry.Kind == WorldSelectionKind.Wmo && entry.IsVisible)
            .OrderBy(static entry => entry.CenterDistance ?? float.MaxValue)
            .Cast<WorldNavigatorEntry?>()
            .FirstOrDefault();
        if (firstVisibleWmo.HasValue)
            return CreateSelection(firstVisibleWmo.Value, tileX, tileY);

        WorldNavigatorEntry? firstVisibleMdx = entries
            .Where(static entry => entry.Kind == WorldSelectionKind.Mdx && entry.IsVisible)
            .OrderBy(static entry => entry.CenterDistance ?? float.MaxValue)
            .Cast<WorldNavigatorEntry?>()
            .FirstOrDefault();
        if (firstVisibleMdx.HasValue)
            return CreateSelection(firstVisibleMdx.Value, tileX, tileY);

        WorldNavigatorEntry? firstWmo = entries.FirstOrDefault(static entry => entry.Kind == WorldSelectionKind.Wmo);
        if (firstWmo.HasValue)
            return CreateSelection(firstWmo.Value, tileX, tileY);

        WorldNavigatorEntry? firstMdx = entries.FirstOrDefault(static entry => entry.Kind == WorldSelectionKind.Mdx);
        if (firstMdx.HasValue)
            return CreateSelection(firstMdx.Value, tileX, tileY);

        return null;
    }

    private static WorldObjectSelection CreateSelection(WorldNavigatorEntry entry, int tileX, int tileY)
    {
        return new WorldObjectSelection(entry.Kind, tileX, tileY, entry.Instance.PlacementEntryIndex, entry.Instance.UniqueId, entry.Instance.ModelKey);
    }
}