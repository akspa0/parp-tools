namespace WowViewer.Core.Maps;

public sealed class AdtTextureFile
{
    public AdtTextureFile(
        string sourcePath,
        MapFileKind kind,
        AdtMcalDecodeProfile decodeProfile,
        IReadOnlyList<string> textureNames,
        IReadOnlyList<AdtTextureChunk> chunks)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentNullException.ThrowIfNull(textureNames);
        ArgumentNullException.ThrowIfNull(chunks);

        SourcePath = sourcePath;
        Kind = kind;
        DecodeProfile = decodeProfile;
        TextureNames = textureNames;
        Chunks = chunks;
    }

    public string SourcePath { get; }

    public MapFileKind Kind { get; }

    public AdtMcalDecodeProfile DecodeProfile { get; }

    public IReadOnlyList<string> TextureNames { get; }

    public IReadOnlyList<AdtTextureChunk> Chunks { get; }
}