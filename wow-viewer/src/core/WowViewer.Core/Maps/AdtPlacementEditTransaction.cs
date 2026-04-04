using System.Numerics;

namespace WowViewer.Core.Maps;

public enum AdtPlacementKind
{
    Model,
    WorldModel,
}

public sealed class AdtPlacementReference
{
    public AdtPlacementReference(AdtPlacementKind kind, int entryIndex, int uniqueId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(entryIndex);

        Kind = kind;
        EntryIndex = entryIndex;
        UniqueId = uniqueId;
    }

    public AdtPlacementKind Kind { get; }

    public int EntryIndex { get; }

    public int UniqueId { get; }
}

public sealed class AdtPlacementMove
{
    public AdtPlacementMove(AdtPlacementReference placement, Vector3 newPosition, string? reason = null)
    {
        ArgumentNullException.ThrowIfNull(placement);

        Placement = placement;
        NewPosition = newPosition;
        Reason = reason;
    }

    public AdtPlacementReference Placement { get; }

    public Vector3 NewPosition { get; }

    public string? Reason { get; }
}

public sealed class AdtPlacementEditTransaction
{
    public AdtPlacementEditTransaction(string sourcePath, IReadOnlyList<AdtPlacementMove> moves)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(moves);

        AdtPlacementMove[] copiedMoves = new AdtPlacementMove[moves.Count];
        for (int index = 0; index < moves.Count; index++)
        {
            ArgumentNullException.ThrowIfNull(moves[index]);
            copiedMoves[index] = moves[index];
        }

        SourcePath = sourcePath;
        Moves = copiedMoves;
    }

    public string SourcePath { get; }

    public IReadOnlyList<AdtPlacementMove> Moves { get; }
}