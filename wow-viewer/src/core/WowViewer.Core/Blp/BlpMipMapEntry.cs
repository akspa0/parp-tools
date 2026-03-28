namespace WowViewer.Core.Blp;

public sealed class BlpMipMapEntry
{
    public BlpMipMapEntry(int level, int width, int height, uint offset, uint sizeBytes, bool isInBounds)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(level);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        Level = level;
        Width = width;
        Height = height;
        Offset = offset;
        SizeBytes = sizeBytes;
        IsInBounds = isInBounds;
    }

    public int Level { get; }

    public int Width { get; }

    public int Height { get; }

    public uint Offset { get; }

    public uint SizeBytes { get; }

    public bool IsInBounds { get; }
}