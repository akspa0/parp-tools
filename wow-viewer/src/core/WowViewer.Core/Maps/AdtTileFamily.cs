namespace WowViewer.Core.Maps;

public sealed class AdtTileFamily
{
    public AdtTileFamily(
        string sourcePath,
        string basePath,
        string rootPath,
        string tex0Path,
        string obj0Path,
        string tex1Path,
        string obj1Path,
        string lodPath,
        bool hasRoot,
        bool hasTex0,
        bool hasObj0,
        bool hasTex1,
        bool hasObj1,
        bool hasLod)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(basePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(rootPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(tex0Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(obj0Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(tex1Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(obj1Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(lodPath);

        SourcePath = sourcePath;
        BasePath = basePath;
        RootPath = rootPath;
        Tex0Path = tex0Path;
        Obj0Path = obj0Path;
        Tex1Path = tex1Path;
        Obj1Path = obj1Path;
        LodPath = lodPath;
        HasRoot = hasRoot;
        HasTex0 = hasTex0;
        HasObj0 = hasObj0;
        HasTex1 = hasTex1;
        HasObj1 = hasObj1;
        HasLod = hasLod;
    }

    public string SourcePath { get; }

    public string BasePath { get; }

    public string RootPath { get; }

    public string Tex0Path { get; }

    public string Obj0Path { get; }

    public string Tex1Path { get; }

    public string Obj1Path { get; }

    public string LodPath { get; }

    public bool HasRoot { get; }

    public bool HasTex0 { get; }

    public bool HasObj0 { get; }

    public bool HasTex1 { get; }

    public bool HasObj1 { get; }

    public bool HasLod { get; }

    public bool HasCompanion(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasTex1 || HasObj1
            : HasTex0 || HasObj0;

    public bool HasCompleteCompanionPair(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasTex1 && HasObj1
            : HasTex0 && HasObj0;

    /// <summary>
    /// Selects one native resident companion band. A complete pair wins over a
    /// partial pair, and the requested band wins when both are available.
    /// </summary>
    public AdtLodBand? SelectCompanionBand(AdtLodBand preferred = AdtLodBand.Band0)
    {
        AdtLodBand alternate = preferred == AdtLodBand.Band0
            ? AdtLodBand.Band1
            : AdtLodBand.Band0;

        if (HasCompleteCompanionPair(preferred))
            return preferred;

        if (HasCompleteCompanionPair(alternate))
            return alternate;

        if (HasCompanion(preferred))
            return preferred;

        return HasCompanion(alternate) ? alternate : null;
    }

    /// <summary>
    /// Returns the best available texture source for the requested native band.
    /// Band 0 remains the compatibility default exposed by <see cref="TextureSourcePath"/>.
    /// </summary>
    public string? GetTextureSourcePath(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasTex1 ? Tex1Path : HasTex0 ? Tex0Path : HasRoot ? RootPath : null
            : HasTex0 ? Tex0Path : HasTex1 ? Tex1Path : HasRoot ? RootPath : null;

    public MapFileKind? GetTextureSourceKind(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasTex1 ? MapFileKind.AdtTex1 : HasTex0 ? MapFileKind.AdtTex : HasRoot ? MapFileKind.Adt : null
            : HasTex0 ? MapFileKind.AdtTex : HasTex1 ? MapFileKind.AdtTex1 : HasRoot ? MapFileKind.Adt : null;

    /// <summary>
    /// Returns the best available placement source for the requested native band.
    /// Band 0 remains the compatibility default exposed by <see cref="PlacementSourcePath"/>.
    /// </summary>
    public string? GetPlacementSourcePath(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasObj1 ? Obj1Path : HasObj0 ? Obj0Path : HasRoot ? RootPath : null
            : HasObj0 ? Obj0Path : HasObj1 ? Obj1Path : HasRoot ? RootPath : null;

    public MapFileKind? GetPlacementSourceKind(AdtLodBand band)
        => band == AdtLodBand.Band1
            ? HasObj1 ? MapFileKind.AdtObj1 : HasObj0 ? MapFileKind.AdtObj : HasRoot ? MapFileKind.Adt : null
            : HasObj0 ? MapFileKind.AdtObj : HasObj1 ? MapFileKind.AdtObj1 : HasRoot ? MapFileKind.Adt : null;

    public string? TextureSourcePath => GetTextureSourcePath(AdtLodBand.Band0);

    public MapFileKind? TextureSourceKind => GetTextureSourceKind(AdtLodBand.Band0);

    public string? PlacementSourcePath => GetPlacementSourcePath(AdtLodBand.Band0);

    public MapFileKind? PlacementSourceKind => GetPlacementSourceKind(AdtLodBand.Band0);
}
