namespace WowViewer.Core.Maps;

public sealed class WdtMainFlagsSummary
{
    public WdtMainFlagsSummary(
        int cellsWithAnyFlags,
        int cellsWithHasAdt,
        int cellsWithAllWater,
        int cellsWithLoaded,
        int cellsWithUnknownFlags,
        int cellsWithAsyncId,
        IReadOnlyList<WdtMainFlagValueSummary> distinctNonZeroValues)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithAnyFlags);
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithHasAdt);
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithAllWater);
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithLoaded);
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithUnknownFlags);
        ArgumentOutOfRangeException.ThrowIfNegative(cellsWithAsyncId);
        ArgumentNullException.ThrowIfNull(distinctNonZeroValues);

        CellsWithAnyFlags = cellsWithAnyFlags;
        CellsWithHasAdt = cellsWithHasAdt;
        CellsWithAllWater = cellsWithAllWater;
        CellsWithLoaded = cellsWithLoaded;
        CellsWithUnknownFlags = cellsWithUnknownFlags;
        CellsWithAsyncId = cellsWithAsyncId;
        DistinctNonZeroValues = distinctNonZeroValues;
    }

    public int CellsWithAnyFlags { get; }

    public int CellsWithHasAdt { get; }

    public int CellsWithAllWater { get; }

    public int CellsWithLoaded { get; }

    public int CellsWithUnknownFlags { get; }

    public int CellsWithAsyncId { get; }

    public IReadOnlyList<WdtMainFlagValueSummary> DistinctNonZeroValues { get; }
}