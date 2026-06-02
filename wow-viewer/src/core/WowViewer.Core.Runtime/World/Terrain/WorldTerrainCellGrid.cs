namespace WowViewer.Core.Runtime.World.Terrain;

public sealed class WorldTerrainCellGrid
{
    public const int CellsPerAxis = 8;
    public const int CellCount = CellsPerAxis * CellsPerAxis;

    private readonly WorldTerrainCell[] _cells;

    private WorldTerrainCellGrid(WorldTerrainCell[] cells)
    {
        _cells = cells;
    }

    public IReadOnlyList<WorldTerrainCell> Cells => _cells;

    public static WorldTerrainCellGrid CreateDefault(ushort holeMask)
    {
        WorldTerrainHoleMask holeMaskState = new(holeMask);
        WorldTerrainCell[] cells = new WorldTerrainCell[CellCount];

        for (int cellY = 0; cellY < CellsPerAxis; cellY++)
        {
            for (int cellX = 0; cellX < CellsPerAxis; cellX++)
            {
                int index = GetCellIndex(cellX, cellY);
                cells[index] = new WorldTerrainCell(
                    cellX,
                    cellY,
                    holeMaskState.IsCellHoled(cellX, cellY),
                    OuterIndex(cellY, cellX),
                    OuterIndex(cellY, cellX + 1),
                    OuterIndex(cellY + 1, cellX),
                    OuterIndex(cellY + 1, cellX + 1),
                    InnerIndex(cellY, cellX));
            }
        }

        return new WorldTerrainCellGrid(cells);
    }

    public ref readonly WorldTerrainCell GetCell(int cellX, int cellY)
    {
        ValidateCellCoordinate(cellX, nameof(cellX));
        ValidateCellCoordinate(cellY, nameof(cellY));
        return ref _cells[GetCellIndex(cellX, cellY)];
    }

    public static int GetCellIndex(int cellX, int cellY)
    {
        ValidateCellCoordinate(cellX, nameof(cellX));
        ValidateCellCoordinate(cellY, nameof(cellY));
        return (cellY * CellsPerAxis) + cellX;
    }

    public static void GetVertexPosition(int index, out int row, out int col, out bool isInner)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        if (index >= 145)
            throw new ArgumentOutOfRangeException(nameof(index));

        int remaining = index;
        row = 0;
        col = 0;
        isInner = false;

        for (int currentRow = 0; currentRow < 17; currentRow++)
        {
            int rowSize = (currentRow % 2 == 0) ? 9 : 8;
            if (remaining < rowSize)
            {
                row = currentRow;
                col = remaining;
                isInner = (currentRow % 2) != 0;
                return;
            }

            remaining -= rowSize;
        }

        throw new ArgumentOutOfRangeException(nameof(index));
    }

    public static int OuterIndex(int outerRow, int outerCol)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(outerRow, nameof(outerRow));
        ArgumentOutOfRangeException.ThrowIfNegative(outerCol, nameof(outerCol));
        if (outerRow > 8)
            throw new ArgumentOutOfRangeException(nameof(outerRow));
        if (outerCol > 8)
            throw new ArgumentOutOfRangeException(nameof(outerCol));

        return (outerRow * 17) + outerCol;
    }

    public static int InnerIndex(int innerRow, int innerCol)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(innerRow, nameof(innerRow));
        ArgumentOutOfRangeException.ThrowIfNegative(innerCol, nameof(innerCol));
        if (innerRow >= CellsPerAxis)
            throw new ArgumentOutOfRangeException(nameof(innerRow));
        if (innerCol >= CellsPerAxis)
            throw new ArgumentOutOfRangeException(nameof(innerCol));

        return (innerRow * 17) + 9 + innerCol;
    }

    private static void ValidateCellCoordinate(int value, string paramName)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, paramName);
        if (value >= CellsPerAxis)
            throw new ArgumentOutOfRangeException(paramName);
    }
}

public readonly record struct WorldTerrainCell(
    int CellX,
    int CellY,
    bool IsHoled,
    int TopLeftVertexIndex,
    int TopRightVertexIndex,
    int BottomLeftVertexIndex,
    int BottomRightVertexIndex,
    int CenterVertexIndex);
