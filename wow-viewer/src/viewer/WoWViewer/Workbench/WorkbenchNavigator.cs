namespace WoWViewer.Workbench;

/// <summary>
/// Compatibility page identifiers for the former Model route.
/// </summary>
public enum ModelBottomTab
{
    Info = 0,
    Animations = 1,
    Actions = 2,
}

/// <summary>
/// Compatibility page identifiers for the former World route.
/// </summary>
public enum WorldBottomTab
{
    Placements = 0,
    Tiles = 1,
    SelectionTools = 2,
    Lod = 3,
}

/// <summary>
/// Compact pages exposed by the canonical Inspect destination.
/// </summary>
public enum InspectBottomTab
{
    Context = 0,
    SceneInvestigation = 1,
    Mcnk = 2,
    WorldContext = 3,
    Archeology = 4,
    Animations = 5,
    Actions = 6,
}

/// <summary>
/// Compatibility page identifiers for the former Tools route.
/// Each value maps to a former 069 page's content.
/// </summary>
public enum ToolsBottomTab
{
    Quick = 0,
    Archeology = 1,
    Pm4 = 2,
    Terrain = 3,
    Utilities = 4,
    Converters = 5,
}

/// <summary>
/// Page identifiers under the Utilities destination.
/// </summary>
public enum UtilitiesBottomTab
{
    Minimap = 0,
    Log = 1,
    Perf = 2,
    RenderQuality = 3,
    Taxi = 4,
    Capture = 5,
    AssetCatalog = 6,
    RuntimeStats = 7,
    Lighting = 8,
    Audio = 9,
}

/// <summary>
/// Page identifiers under Experimental &gt; Terrain Lab.
/// </summary>
public enum TerrainBottomTab
{
    Clipboard = 0,
    Analysis = 1,
    Mcnk = 2,
    WeakSignal = 3,
    Export = 4,
    Tools = 5,
}

/// <summary>
/// Page identifiers under Experimental &gt; PM4.
/// </summary>
public enum Pm4BottomTab
{
    Overlay = 0,
    Selection = 1,
    Correlation = 2,
    Info = 3,
    Match = 4,
    Alignment = 5,
}

/// <summary>
/// Page identifiers under Experimental &gt; Archeology.
/// </summary>
public enum ArcheologyBottomTab
{
    Range = 0,
    Layers = 1,
    Playback = 2,
    Capture = 3,
}

/// <summary>
/// Helpers for mapping the task-oriented workbench destinations to their
/// single optional page selector.
/// </summary>
public static class WorkbenchNavigator
{
    public static string[] GetBottomTabLabels(WorkbenchTab tab) => tab switch
    {
        WorkbenchTab.Quick => [],
        WorkbenchTab.Inspect => GetInspectBottomTabLabels(),
        WorkbenchTab.Scene => ["Placements", "LOD"],
        WorkbenchTab.Utilities => GetUtilitiesBottomTabLabels(),
        WorkbenchTab.Experimental => ["Terrain Lab", "PM4", "Converters", "Population"],
        _ => [],
    };

    /// <summary>Labels for <see cref="InspectBottomTab"/>; order must match the enum.</summary>
    public static string[] GetInspectBottomTabLabels() =>
        ["Context", "Scene Investigation", "MCNK / ADT", "World Context", "Archeology", "Animations", "Actions"];

    public static string[] GetTerrainBottomTabLabels() => ["Clipboard", "Analysis", "MCNK", "Weak Signal", "Export", "Tools"];

    /// <summary>Labels for <see cref="Pm4BottomTab"/>; order must match the enum.</summary>
    public static string[] GetPm4BottomTabLabels() => ["Overlay", "Selection", "Correlation", "Info", "Match", "Alignment"];

    /// <summary>Labels for <see cref="ArcheologyBottomTab"/>; order must match the enum.</summary>
    public static string[] GetArcheologyBottomTabLabels() => ["Range", "Layers", "Playback", "Capture"];

    /// <summary>Labels for <see cref="UtilitiesBottomTab"/>; order must match the enum.</summary>
    public static string[] GetUtilitiesBottomTabLabels() =>
        ["Minimap", "Log", "Perf", "Render Quality", "Taxi", "Capture", "Asset Catalog", "Runtime Stats", "Lighting", "Audio"];
}
