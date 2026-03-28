namespace WowViewer.Core.Maps;

public sealed class AdtTileFamily
{
    public AdtTileFamily(
        string sourcePath,
        string basePath,
        string rootPath,
        string tex0Path,
        string obj0Path,
        string lodPath,
        bool hasRoot,
        bool hasTex0,
        bool hasObj0,
        bool hasLod)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(basePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(rootPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(tex0Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(obj0Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(lodPath);

        SourcePath = sourcePath;
        BasePath = basePath;
        RootPath = rootPath;
        Tex0Path = tex0Path;
        Obj0Path = obj0Path;
        LodPath = lodPath;
        HasRoot = hasRoot;
        HasTex0 = hasTex0;
        HasObj0 = hasObj0;
        HasLod = hasLod;
    }

    public string SourcePath { get; }

    public string BasePath { get; }

    public string RootPath { get; }

    public string Tex0Path { get; }

    public string Obj0Path { get; }

    public string LodPath { get; }

    public bool HasRoot { get; }

    public bool HasTex0 { get; }

    public bool HasObj0 { get; }

    public bool HasLod { get; }

    public string? TextureSourcePath => HasTex0 ? Tex0Path : HasRoot ? RootPath : null;

    public MapFileKind? TextureSourceKind => HasTex0 ? MapFileKind.AdtTex : HasRoot ? MapFileKind.Adt : null;

    public string? PlacementSourcePath => HasObj0 ? Obj0Path : HasRoot ? RootPath : null;

    public MapFileKind? PlacementSourceKind => HasObj0 ? MapFileKind.AdtObj : HasRoot ? MapFileKind.Adt : null;
}