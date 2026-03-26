using WowViewer.Core.Chunks;

namespace WowViewer.Core.IO.Chunked;

public readonly record struct ChunkSpan(ChunkHeader Header, long HeaderOffset, long DataOffset)
{
    public long EndOffset => checked(DataOffset + Header.Size);
}