namespace WowViewer.Core.Runtime.World.Terrain;

public readonly struct WorldTerrainHoleMask
{
    public const int CellsPerAxis = 8;
    public const int HoleGroupsPerAxis = 4;

    public WorldTerrainHoleMask(ushort rawValue)
    {
        RawValue = rawValue;
        RawValue64 = Maps.TerrainHoleMath.UpsampleLowResToHighRes(rawValue);
    }

    public WorldTerrainHoleMask(ulong rawValue64, ushort rawValue = 0)
    {
        RawValue64 = rawValue64;
        RawValue = rawValue != 0 ? rawValue : Maps.TerrainHoleMath.DownsampleHighResToLowRes(rawValue64);
    }

    public ushort RawValue { get; }
    public ulong RawValue64 { get; }

    public bool HasHoles => RawValue64 != 0UL || RawValue != 0;

    public bool IsCellHoled(int cellX, int cellY)
    {
        ValidateCellCoordinate(cellX, nameof(cellX));
        ValidateCellCoordinate(cellY, nameof(cellY));

        if (RawValue64 != 0UL)
            return Maps.TerrainHoleMath.IsCellHoled64(RawValue64, cellX, cellY);

        return IsHoleGroupSet(cellX / 2, cellY / 2);
    }

    public bool IsHoleGroupSet(int holeGroupX, int holeGroupY)
    {
        ValidateHoleGroupCoordinate(holeGroupX, nameof(holeGroupX));
        ValidateHoleGroupCoordinate(holeGroupY, nameof(holeGroupY));

        int holeBit = 1 << ((holeGroupY * HoleGroupsPerAxis) + holeGroupX);
        return (RawValue & holeBit) != 0;
    }

    private static void ValidateCellCoordinate(int value, string paramName)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, paramName);
        if (value >= CellsPerAxis)
            throw new ArgumentOutOfRangeException(paramName);
    }

    private static void ValidateHoleGroupCoordinate(int value, string paramName)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, paramName);
        if (value >= HoleGroupsPerAxis)
            throw new ArgumentOutOfRangeException(paramName);
    }
}
