namespace WowViewer.Core.Mdx;

public sealed class MdxTextureSummary
{
    public MdxTextureSummary(int index, uint replaceableId, string? path, uint flags)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);

        Index = index;
        ReplaceableId = replaceableId;
        Path = string.IsNullOrWhiteSpace(path) ? null : path;
        Flags = flags;
    }

    public int Index { get; }

    public uint ReplaceableId { get; }

    public string? Path { get; }

    public uint Flags { get; }

    public bool HasPath => !string.IsNullOrWhiteSpace(Path);

    public bool IsReplaceable => ReplaceableId != 0;
}