namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Separates resident object admission from directional terrain submission.
/// A retained tile may contribute objects; per-object bounds and frustum tests
/// still decide whether an admitted placement is rendered.
/// </summary>
public static class WorldObjectTileAdmission
{
    public static bool IsResident(
        IReadOnlyList<(int tileX, int tileY)> detailedTiles,
        IReadOnlyList<(int tileX, int tileY)> retainedTiles,
        (int tileX, int tileY) tile)
    {
        return Contains(detailedTiles, tile) || Contains(retainedTiles, tile);
    }

    private static bool Contains(
        IReadOnlyList<(int tileX, int tileY)> tiles,
        (int tileX, int tileY) tile)
    {
        for (int index = 0; index < tiles.Count; index++)
        {
            if (tiles[index] == tile)
                return true;
        }

        return false;
    }
}
