using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Converters;

/// <summary>
/// Loads a complete WMO (root file + every group file) via <see cref="IArchiveCatalog"/>.
/// <see cref="WmoV14ToV17Converter.ParseWmoV14"/> only reads the root file, which carries no
/// group geometry -- per-face materials, vertices, and UVs live in the separate
/// <c>Name_000.wmo</c>/<c>Name_001.wmo</c>/... group files (MOPY/MOVT/MOVI/MOTV). Parsing the
/// root alone leaves every group empty, which silently collapses every face onto material 0 in
/// any consumer that renders/exports by material id. This is the one canonical, catalog-agnostic
/// place that loading is done correctly; both the viewer (via its own IDataSource-based
/// ScreenshotRenderer.LoadWmoData) and headless IArchiveCatalog-based tools should route through
/// the same root+group-loading shape rather than re-deriving it per caller.
/// </summary>
public static class WmoFullLoader
{
    /// <summary>Read and parse a root WMO plus every group file it declares. Returns null if the
    /// root file cannot be read or parsed.</summary>
    public static WmoV14ToV17Converter.WmoV14Data? Load(IArchiveCatalog catalog, string wmoPath)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        if (string.IsNullOrWhiteSpace(wmoPath))
            return null;

        string normalized = wmoPath.Replace('/', '\\');
        byte[]? rootBytes = catalog.ReadFile(normalized);
        if (rootBytes is null || rootBytes.Length == 0)
            return null;

        var converter = new WmoV14ToV17Converter();
        string tempPath = Path.Combine(Path.GetTempPath(), $"wmo_load_{Guid.NewGuid():N}.wmo");
        WmoV14ToV17Converter.WmoV14Data wmo;
        try
        {
            File.WriteAllBytes(tempPath, rootBytes);
            wmo = converter.ParseWmoV14(tempPath);
        }
        catch
        {
            return null;
        }
        finally
        {
            try { File.Delete(tempPath); } catch { /* best-effort cleanup */ }
        }

        if (wmo.Groups.Count == 0 && wmo.GroupCount > 0)
        {
            string modelDir = Path.GetDirectoryName(normalized) ?? "";
            string baseName = Path.GetFileNameWithoutExtension(normalized);
            for (int gi = 0; gi < wmo.GroupCount; gi++)
            {
                string groupName = $"{baseName}_{gi:D3}.wmo";
                string groupPath = string.IsNullOrEmpty(modelDir) ? groupName : $"{modelDir}\\{groupName}";
                byte[]? groupBytes = catalog.ReadFile(groupPath);
                if (groupBytes is { Length: > 0 })
                    converter.ParseGroupFile(groupBytes, wmo, gi);
            }
        }

        return wmo;
    }
}
