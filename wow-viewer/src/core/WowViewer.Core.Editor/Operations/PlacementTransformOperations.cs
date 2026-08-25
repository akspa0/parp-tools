using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <summary>A rotation-only placement edit expressed as data.</summary>
public sealed class PlacementRotateOperation : EditorOperation
{
    public PlacementRotateOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        AdtPlacementKind kind,
        int entryIndex,
        int uniqueId,
        Vector3 oldRotation,
        Vector3 newRotation)
        : base(operationId, originPluginId, $"Rotate placement {kind} #{entryIndex} (id {uniqueId})", undoable: true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryIndex);

        SourcePath = sourcePath;
        Kind = kind;
        EntryIndex = entryIndex;
        UniqueId = uniqueId;
        OldRotation = oldRotation;
        NewRotation = newRotation;
    }

    public string SourcePath { get; }
    public AdtPlacementKind Kind { get; }
    public int EntryIndex { get; }
    public int UniqueId { get; }
    public Vector3 OldRotation { get; }
    public Vector3 NewRotation { get; }

    public override IReadOnlyList<string> AffectedPaths => [SourcePath];

    public override EditorOperation CreateReverse()
        => new PlacementRotateOperation(
            $"reverse:{OperationId}", OriginPluginId, SourcePath, Kind, EntryIndex, UniqueId, NewRotation, OldRotation);
}

/// <summary>A scale-only placement edit expressed as data.</summary>
public sealed class PlacementScaleOperation : EditorOperation
{
    public PlacementScaleOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        AdtPlacementKind kind,
        int entryIndex,
        int uniqueId,
        float oldScale,
        float newScale)
        : base(operationId, originPluginId, $"Scale placement {kind} #{entryIndex} (id {uniqueId})", undoable: true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryIndex);

        SourcePath = sourcePath;
        Kind = kind;
        EntryIndex = entryIndex;
        UniqueId = uniqueId;
        OldScale = oldScale;
        NewScale = newScale;
    }

    public string SourcePath { get; }
    public AdtPlacementKind Kind { get; }
    public int EntryIndex { get; }
    public int UniqueId { get; }
    public float OldScale { get; }
    public float NewScale { get; }

    public override IReadOnlyList<string> AffectedPaths => [SourcePath];

    public override EditorOperation CreateReverse()
        => new PlacementScaleOperation(
            $"reverse:{OperationId}", OriginPluginId, SourcePath, Kind, EntryIndex, UniqueId, NewScale, OldScale);
}

/// <summary>A delete operation expressed as data. The reverse re-adds the placement at its prior
/// transform; the source row identity is preserved so the reverse can be applied deterministically.</summary>
public sealed class PlacementDeleteOperation : EditorOperation
{
    public PlacementDeleteOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        AdtPlacementKind kind,
        int entryIndex,
        int uniqueId)
        : base(operationId, originPluginId, $"Delete placement {kind} #{entryIndex} (id {uniqueId})", undoable: true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryIndex);

        SourcePath = sourcePath;
        Kind = kind;
        EntryIndex = entryIndex;
        UniqueId = uniqueId;
    }

    public string SourcePath { get; }
    public AdtPlacementKind Kind { get; }
    public int EntryIndex { get; }
    public int UniqueId { get; }

    public override IReadOnlyList<string> AffectedPaths => [SourcePath];

    public override EditorOperation CreateReverse()
        => new PlacementDeleteOperation(
            $"reverse:{OperationId}", OriginPluginId, SourcePath, Kind, EntryIndex, UniqueId);
}