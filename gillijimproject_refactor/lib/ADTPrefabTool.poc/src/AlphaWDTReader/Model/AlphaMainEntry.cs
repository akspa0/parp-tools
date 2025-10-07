namespace AlphaWDTReader.Model;

public readonly record struct AlphaMainEntry(uint Offset, uint Size, int TileX, int TileY)
{
    public int TileId => TileY * 64 + TileX;
}
