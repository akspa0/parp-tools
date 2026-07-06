namespace WoWViewer.Workbench;

/// <summary>
/// Sub-tab identifiers for the Model top tab.
/// </summary>
public enum ModelBottomTab
{
    Info = 0,
    Animations = 1,
    Actions = 2,
    Lod = 3,
}

/// <summary>
/// Sub-tab identifiers for the World top tab.
/// </summary>
public enum WorldBottomTab
{
    Source = 0,
    Placements = 1,
    Tiles = 2,
    SelectionTools = 3,
    Lod = 4,
}

/// <summary>
/// Sub-tab identifiers for the Tools top tab.
/// Each value maps to a former 069 top tab's content.
/// </summary>
public enum ToolsBottomTab
{
    Quick = 0,
    Archeology = 1,
    Pm4 = 2,
    Terrain = 3,
    Utilities = 4,
}

/// <summary>
/// Sub-tab identifiers under Tools &gt; Utilities.
/// </summary>
public enum UtilitiesBottomTab
{
    Minimap = 0,
    Log = 1,
    Perf = 2,
    RenderQuality = 3,
    Taxi = 4,
    CaptureAutomation = 5,
    AssetCatalog = 6,
    RuntimeStats = 7,
}

/// <summary>
/// Sub-tab identifiers under Tools &gt; Terrain.
/// </summary>
public enum TerrainBottomTab
{
    Clipboard = 0,
    Analysis = 1,
    Mcnk = 2,
    WeakSignal = 3,
    Export = 4,
}

/// <summary>
/// Sub-tab identifiers under Tools &gt; PM4.
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
/// Sub-tab identifiers under Tools &gt; Archeology.
/// </summary>
public enum ArcheologyBottomTab
{
    Range = 0,
    Layers = 1,
    Playback = 2,
    Capture = 3,
}

/// <summary>
/// Helpers for mapping between 071 workbench tabs and their sub-tab labels.
/// </summary>
public static class WorkbenchNavigator
{
    public static string[] GetBottomTabLabels(WorkbenchTab tab) => tab switch
    {
        WorkbenchTab.Model => ["Info", "Animations", "Actions", "LOD"],
        WorkbenchTab.World => ["Source", "Placements", "Tiles", "Selection Tools", "LOD"],
        WorkbenchTab.Tools => ["Quick", "Archeology", "PM4", "Terrain", "Utilities"],
        _ => [],
    };
}
