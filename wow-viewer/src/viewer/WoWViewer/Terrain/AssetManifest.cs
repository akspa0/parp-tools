namespace WoWViewer.Terrain;

/// <summary>
/// Describes the set of unique assets referenced by a map.
/// Built before loading so we know the full scope.
/// </summary>
public class AssetManifest
{
    public HashSet<string> ReferencedMdx { get; } = new(StringComparer.OrdinalIgnoreCase);
    public HashSet<string> ReferencedWmo { get; } = new(StringComparer.OrdinalIgnoreCase);
}
