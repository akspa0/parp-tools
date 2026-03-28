using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtTileFamilyResolver
{
    public static AdtTileFamily Resolve(string sourcePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        string fullPath = Path.GetFullPath(sourcePath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? throw new InvalidDataException($"Could not resolve directory for '{sourcePath}'.");
        string fileName = Path.GetFileName(fullPath);
        if (!fileName.EndsWith(".adt", StringComparison.OrdinalIgnoreCase))
            throw new InvalidDataException($"ADT family resolution requires an .adt file, but found '{sourcePath}'.");

        string stem = Path.GetFileNameWithoutExtension(fullPath);
        string baseStem;
        if (stem.EndsWith("_tex0", StringComparison.OrdinalIgnoreCase))
            baseStem = stem[..^5];
        else if (stem.EndsWith("_obj0", StringComparison.OrdinalIgnoreCase))
            baseStem = stem[..^5];
        else if (stem.EndsWith("_lod", StringComparison.OrdinalIgnoreCase))
            baseStem = stem[..^4];
        else
            baseStem = stem;

        string basePath = Path.Combine(directory, baseStem);
        string rootPath = basePath + ".adt";
        string tex0Path = basePath + "_tex0.adt";
        string obj0Path = basePath + "_obj0.adt";
        string lodPath = basePath + "_lod.adt";

        return new AdtTileFamily(
            fullPath,
            basePath,
            rootPath,
            tex0Path,
            obj0Path,
            lodPath,
            hasRoot: File.Exists(rootPath),
            hasTex0: File.Exists(tex0Path),
            hasObj0: File.Exists(obj0Path),
            hasLod: File.Exists(lodPath));
    }
}