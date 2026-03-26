using WowViewer.Core.Chunks;

namespace WowViewer.Core.Maps;

public readonly record struct MapChunkLocation(FourCC Id, uint Size, long HeaderOffset, long DataOffset)
{
    public long EndOffset => checked(DataOffset + Size);
}