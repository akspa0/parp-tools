using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// A translation-only placement move expressed as data. This is the narrowest operation that proves
/// the whole bridge: the source placement identity, the before/after position, and the reverse are
/// all fully recorded, so the host can undo it without a plugin (Spec 167 FR-006/FR-007).
/// </summary>
public sealed class PlacementMoveOperation : EditorOperation
{
    public PlacementMoveOperation(
        string operationId,
        string originPluginId,
        string sourcePath,
        AdtPlacementKind kind,
        int entryIndex,
        int uniqueId,
        Vector3 oldPosition,
        Vector3 newPosition)
        : base(operationId, originPluginId, $"Move placement {kind} #{entryIndex} (id {uniqueId})", undoable: true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryIndex);

        SourcePath = sourcePath;
        Kind = kind;
        EntryIndex = entryIndex;
        UniqueId = uniqueId;
        OldPosition = oldPosition;
        NewPosition = newPosition;
    }

    public string SourcePath { get; }

    public AdtPlacementKind Kind { get; }

    public int EntryIndex { get; }

    public int UniqueId { get; }

    public Vector3 OldPosition { get; }

    public Vector3 NewPosition { get; }

    public override IReadOnlyList<string> AffectedPaths => [SourcePath];

    public override EditorOperation CreateReverse()
        => new PlacementMoveOperation(
            $"reverse:{OperationId}",
            OriginPluginId,
            SourcePath,
            Kind,
            EntryIndex,
            UniqueId,
            NewPosition,
            OldPosition);
}