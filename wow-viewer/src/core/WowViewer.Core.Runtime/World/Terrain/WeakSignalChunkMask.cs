namespace WowViewer.Core.Runtime.World.Terrain;

public struct WeakSignalChunkMask
{
    public const int ChunksPerDim = 16;
    public const int ChunksPerTile = ChunksPerDim * ChunksPerDim;

    public bool[] Mask; // length 256, true = weak signal chunk
    public float[] ChunkMinHeights; // length 256
    public float[] ChunkMaxHeights; // length 256
    public int WeakChunkCount;

    public bool IsWeak(int cx, int cy) => (uint)cx < ChunksPerDim && (uint)cy < ChunksPerDim && Mask[cy * ChunksPerDim + cx];
}
