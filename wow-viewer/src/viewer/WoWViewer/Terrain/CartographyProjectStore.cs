using WowViewer.Core.Maps;

namespace WoWViewer.Terrain;

/// <summary>
/// Spec 232 Phase 2 (FR-2/FR-8): layer-project storage — one human-editable JSON file per base
/// map, under the viewer's project output area. Reloaded automatically on map load (FR-2) and
/// via explicit Save/Load buttons in the Layers panel (FR-8).
/// </summary>
internal static class CartographyProjectStore
{
    /// <summary>Same convention as the viewer's OutputDir: exe-relative output/projects.</summary>
    public static string ProjectsDir { get; set; } =
        Path.Combine(AppContext.BaseDirectory, "output", "projects");

    public static string ProjectPath(string mapName)
        => Path.Combine(ProjectsDir, "cartography", mapName + ".layers.json");

    public static bool Exists(string mapName) => File.Exists(ProjectPath(mapName));
}
