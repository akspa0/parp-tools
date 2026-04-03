using System.Text.Json;

namespace WowViewer.Core.IO.Files;

public sealed record ArchiveListfileCacheManifest(
    int FormatVersion,
    string CacheKey,
    string[] ArchiveRoots,
    DateTimeOffset GeneratedAtUtc,
    string[] TrustedInternalEntries,
    string[] SupplementalEntries)
{
    public IReadOnlyList<string> AllEntries { get; } = BuildAllEntries(TrustedInternalEntries, SupplementalEntries);

    private static IReadOnlyList<string> BuildAllEntries(
        IReadOnlyList<string> trustedInternalEntries,
        IReadOnlyList<string> supplementalEntries)
    {
        HashSet<string> combined = new(StringComparer.OrdinalIgnoreCase);

        foreach (string entry in trustedInternalEntries)
            combined.Add(entry);

        foreach (string entry in supplementalEntries)
            combined.Add(entry);

        return combined.OrderBy(static entry => entry, StringComparer.OrdinalIgnoreCase).ToArray();
    }
}

public static class ArchiveListfileCache
{
    public const int CurrentFormatVersion = 1;

    public static ArchiveListfileCacheManifest? TryRead(string cacheDirectoryPath, string cacheKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDirectoryPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheKey);

        string cachePath = GetCachePath(cacheDirectoryPath, cacheKey);
        if (!File.Exists(cachePath))
            return null;

        try
        {
            string json = File.ReadAllText(cachePath);
            ArchiveListfileCacheManifest? manifest = JsonSerializer.Deserialize<ArchiveListfileCacheManifest>(json);
            if (manifest is null || manifest.FormatVersion != CurrentFormatVersion)
                return null;

            return manifest with
            {
                TrustedInternalEntries = NormalizeEntries(manifest.TrustedInternalEntries),
                SupplementalEntries = NormalizeEntries(manifest.SupplementalEntries),
            };
        }
        catch
        {
            return null;
        }
    }

    public static string Write(
        string cacheDirectoryPath,
        string cacheKey,
        IEnumerable<string> archiveRoots,
        IEnumerable<string> trustedInternalEntries,
        IEnumerable<string> supplementalEntries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDirectoryPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheKey);
        ArgumentNullException.ThrowIfNull(archiveRoots);
        ArgumentNullException.ThrowIfNull(trustedInternalEntries);
        ArgumentNullException.ThrowIfNull(supplementalEntries);

        Directory.CreateDirectory(cacheDirectoryPath);

        ArchiveListfileCacheManifest manifest = new(
            CurrentFormatVersion,
            cacheKey,
            NormalizeEntries(archiveRoots),
            DateTimeOffset.UtcNow,
            NormalizeEntries(trustedInternalEntries),
            NormalizeEntries(supplementalEntries));

        string cachePath = GetCachePath(cacheDirectoryPath, cacheKey);
        JsonSerializerOptions options = new() { WriteIndented = true };
        File.WriteAllText(cachePath, JsonSerializer.Serialize(manifest, options));
        return cachePath;
    }

    public static string GetCachePath(string cacheDirectoryPath, string cacheKey)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheDirectoryPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(cacheKey);

        return Path.Combine(cacheDirectoryPath, $"{SanitizeFileName(cacheKey)}.json");
    }

    private static string[] NormalizeEntries(IEnumerable<string> entries)
    {
        HashSet<string> normalized = new(StringComparer.OrdinalIgnoreCase);
        foreach (string? entry in entries)
        {
            string trimmed = entry?.Trim() ?? string.Empty;
            if (string.IsNullOrWhiteSpace(trimmed))
                continue;

            normalized.Add(trimmed.Replace('/', '\\'));
        }

        return normalized.OrderBy(static entry => entry, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static string SanitizeFileName(string value)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        char[] sanitized = value
            .Select(character => invalid.Contains(character) ? '_' : character)
            .ToArray();
        return new string(sanitized);
    }
}