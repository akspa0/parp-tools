namespace WowViewer.Core.IO.Files;

/// <summary>
/// Spec 239: bridges FileDataID references into the path-based asset plumbing. Loaders that meet a
/// FileDataID (M2 TXID/SFID, WMO MOMT/GFID/MODI) call <see cref="Resolve"/> and pass the result
/// wherever a virtual path is expected. The active data source installs <see cref="Resolver"/>
/// (listfile lookup); ids without a listfile name become <c>fdid:&lt;id&gt;</c>, which FileDataID-aware
/// data sources read directly.
/// </summary>
public static class FileDataIdPaths
{
    public const string Prefix = "fdid:";

    /// <summary>Id → listfile path, installed by the active data source. Null when no FileDataID-aware source is open.</summary>
    public static Func<uint, string?>? Resolver { get; set; }

    public static string ToVirtualPath(uint fileDataId) => Prefix + fileDataId;

    public static string Resolve(uint fileDataId) =>
        Resolver?.Invoke(fileDataId) is { Length: > 0 } path ? path : ToVirtualPath(fileDataId);

    public static bool TryParse(string? path, out uint fileDataId)
    {
        fileDataId = 0;
        return path is not null
            && path.StartsWith(Prefix, StringComparison.OrdinalIgnoreCase)
            && uint.TryParse(path.AsSpan(Prefix.Length), out fileDataId);
    }
}
