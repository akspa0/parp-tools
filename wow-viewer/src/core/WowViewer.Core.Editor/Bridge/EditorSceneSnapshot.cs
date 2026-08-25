using System.Numerics;

namespace WowViewer.Core.Editor.Bridge;

public enum EditorSelectionKind
{
    Model,
    WorldModel,
}

/// <summary>The build-neutral read model a plugin sees for one selected placement.</summary>
public sealed record EditorSelectionEntry(
    EditorSelectionKind Kind,
    string MapName,
    int TileX,
    int TileY,
    int EntryIndex,
    int UniqueId,
    string ModelPath,
    Vector3 Position);

/// <summary>The renderer-agnostic camera the bridge exposes to plugins.</summary>
public readonly record struct EditorCamera(Vector3 Position, Vector3 Forward, Vector3 Up);

/// <summary>A single loaded tile coordinate.</summary>
public readonly record struct EditorLoadedTile(string MapName, int TileX, int TileY);

/// <summary>
/// An immutable, renderer-free snapshot of live scene state. Plugins read through this; they never
/// see <c>WorldScene</c>, the renderer, or <c>ViewerApp</c> (Spec 167 FR-001).
/// </summary>
public sealed record EditorSceneSnapshot(
    string? MapName,
    EditorCamera Camera,
    IReadOnlyList<EditorLoadedTile> LoadedTiles,
    IReadOnlyList<EditorSelectionEntry> Selection)
{
    public static EditorSceneSnapshot Empty { get; } = new(null, default, [], []);
}