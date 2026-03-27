using System.Text;

namespace WowViewer.Core.IO.Files;

public sealed class Md5TranslateIndex
{
    public Dictionary<string, string> HashToPlain { get; } = new(StringComparer.OrdinalIgnoreCase);

    public Dictionary<string, string> PlainToHash { get; } = new(StringComparer.OrdinalIgnoreCase);

    public string Normalize(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        return value.Replace('\\', '/').Trim().TrimStart('/').ToLowerInvariant();
    }

    public void Add(string left, string right)
    {
        string normalizedLeft = Normalize(left);
        string normalizedRight = Normalize(right);

        if (!HashToPlain.ContainsKey(normalizedLeft))
            HashToPlain[normalizedLeft] = normalizedRight;

        if (!PlainToHash.ContainsKey(normalizedRight))
            PlainToHash[normalizedRight] = normalizedLeft;

        if (!HashToPlain.ContainsKey(normalizedRight))
            HashToPlain[normalizedRight] = normalizedLeft;

        if (!PlainToHash.ContainsKey(normalizedLeft))
            PlainToHash[normalizedLeft] = normalizedRight;
    }
}

public static class Md5TranslateResolver
{
    private static readonly string[] Candidates =
    [
        "textures/Minimap/md5translate.txt",
        "textures/Minimap/md5translate.trs",
        "textures/minimap/md5translate.txt",
        "textures/minimap/md5translate.trs",
        "world/textures/Minimap/md5translate.txt",
        "world/textures/Minimap/md5translate.trs",
        "world/textures/minimap/md5translate.txt",
        "world/textures/minimap/md5translate.trs",
        "Data/textures/Minimap/md5translate.txt",
        "Data/textures/Minimap/md5translate.trs",
    ];

    public static bool TryLoad(
        IEnumerable<string> searchPaths,
        Func<string, bool> archiveFileExists,
        Func<string, byte[]?> archiveReadFile,
        out Md5TranslateIndex? index,
        IEnumerable<string>? extraCandidates = null)
    {
        ArgumentNullException.ThrowIfNull(searchPaths);
        ArgumentNullException.ThrowIfNull(archiveFileExists);
        ArgumentNullException.ThrowIfNull(archiveReadFile);

        index = null;

        Md5TranslateIndex translateIndex = new();
        bool foundAny = false;

        if (extraCandidates is not null)
        {
            foreach (string candidate in extraCandidates)
            {
                if (TryLoadArchiveCandidate(candidate, archiveFileExists, archiveReadFile, translateIndex, inferDirectoryFromCandidate: true))
                    foundAny = true;
            }
        }

        foreach (string candidate in Candidates)
        {
            if (TryLoadArchiveCandidate(candidate, archiveFileExists, archiveReadFile, translateIndex, inferDirectoryFromCandidate: false))
                foundAny = true;
        }

        foreach (string basePath in searchPaths)
        {
            foreach (string candidate in Candidates)
            {
                string fullPath = Path.Combine(basePath, candidate);
                if (!File.Exists(fullPath))
                    continue;

                using FileStream stream = File.OpenRead(fullPath);
                ParseStream(stream, translateIndex);
                foundAny = true;
            }
        }

        if (!foundAny)
            return false;

        index = translateIndex;
        return true;
    }

    private static bool TryLoadArchiveCandidate(
        string candidate,
        Func<string, bool> archiveFileExists,
        Func<string, byte[]?> archiveReadFile,
        Md5TranslateIndex index,
        bool inferDirectoryFromCandidate)
    {
        string archiveKey = candidate.Replace("/", "\\", StringComparison.Ordinal);
        if (!archiveFileExists(archiveKey))
            return false;

        byte[]? data = archiveReadFile(archiveKey);
        if (data is null)
            return false;

        string? initialDirectory = inferDirectoryFromCandidate ? TryInferInitialDirectory(archiveKey, index) : null;
        using MemoryStream stream = new(data);
        ParseStream(stream, index, initialDirectory);
        return true;
    }

    private static string? TryInferInitialDirectory(string archiveKey, Md5TranslateIndex index)
    {
        if (!archiveKey.Contains("World\\Maps\\", StringComparison.OrdinalIgnoreCase))
            return null;

        string[] parts = archiveKey.Split('\\', StringSplitOptions.RemoveEmptyEntries);
        for (int i = 0; i < parts.Length - 1; i++)
        {
            if (string.Equals(parts[i], "Maps", StringComparison.OrdinalIgnoreCase))
                return index.Normalize(parts[i + 1]).Trim('/');
        }

        return null;
    }

    private static void ParseStream(Stream stream, Md5TranslateIndex index, string? initialDirectory = null)
    {
        using StreamReader reader = new(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true, leaveOpen: true);
        string? currentDirectory = initialDirectory;

        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string trimmed = line.Trim();
            if (trimmed.StartsWith("#", StringComparison.Ordinal))
                continue;

            if (trimmed.StartsWith("dir:", StringComparison.OrdinalIgnoreCase))
            {
                string directoryName = trimmed[4..].Trim();
                if (!string.IsNullOrWhiteSpace(directoryName))
                    currentDirectory = index.Normalize(directoryName).Trim('/');

                continue;
            }

            string[] parts = trimmed.Split((char[])null!, 2, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2)
            {
                AddWithVariants(index, parts[0], parts[1], currentDirectory);
                continue;
            }

            if (parts.Length > 2)
                AddWithVariants(index, parts[0], parts[^1], currentDirectory);
        }
    }

    private static void AddWithVariants(Md5TranslateIndex index, string plainRaw, string hashedRaw, string? currentDirectory)
    {
        string plain = index.Normalize(plainRaw);
        string hashed = index.Normalize(hashedRaw);

        string plainDirectory = plain.Contains('/') ? string.Empty : (currentDirectory ?? string.Empty);
        string plainPath = !string.IsNullOrEmpty(plainDirectory)
            ? $"textures/minimap/{plainDirectory}/{plain}"
            : (plain.StartsWith("textures/minimap/") || plain.StartsWith("world/textures/minimap/"))
                ? plain
                : $"textures/minimap/{plain}";

        string hashFile = hashed.Contains('/') ? Path.GetFileName(hashed) : hashed;
        string hashedTexturePath = $"textures/minimap/{hashFile}";
        string hashedWorldPath = $"world/textures/minimap/{hashFile}";
        string hashedPluralPath = $"textures/minimaps/{hashFile}";

        index.Add(plainPath, hashedTexturePath);
        index.Add(plainPath, hashedWorldPath);
        index.Add(plainPath, hashedPluralPath);

        string shortPlain = plain;
        if (!shortPlain.StartsWith("textures/minimap/") && !shortPlain.StartsWith("world/textures/minimap/"))
        {
            if (!string.IsNullOrEmpty(plainDirectory) && !shortPlain.StartsWith(plainDirectory + "/", StringComparison.Ordinal))
                shortPlain = $"{plainDirectory}/{shortPlain}";

            index.Add(shortPlain, hashedTexturePath);
            index.Add(shortPlain, hashedWorldPath);
            index.Add(shortPlain, hashedPluralPath);
        }
    }
}