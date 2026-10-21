namespace WoWViewer.Workbench;
public static class WorkbenchNavigator
{
    public static string[] GetPageLabels(WorkbenchTab destination) => destination switch
    {
        WorkbenchTab.Editor => ["Placement & Objects", "Terrain Tools", "Data I/O", "Converters"],
        WorkbenchTab.Archaeology => ["Weak Signal & Stratigraphy", "UniqueId Timeline", "Layers & Provenance", "Playback & Capture", "Cartography"],
        _ => [],
    };
}
