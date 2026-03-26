namespace WowViewer.Core.Chunks;

public readonly record struct ChunkHeader(FourCC Id, uint Size)
{
    public const int SizeInBytes = 8;
}