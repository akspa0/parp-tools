namespace WowViewer.Core.Audio;

/// <summary>
/// Build-scoped SoundEntries metadata. File paths come from the loaded DBC;
/// this type does not infer a sound directory from an ID or filename.
/// </summary>
public sealed record AlphaSoundEntry(
    int Id,
    IReadOnlyList<string> Files,
    string DirectoryBase,
    float Volume,
    float MinDistance,
    float MaxDistance,
    float DistanceCutoff)
{
    public bool HasClientFilePaths => Files.Count > 0;

    public IEnumerable<string> EnumerateVirtualPaths()
    {
        foreach (string file in Files)
        {
            string normalizedFile = Normalize(file);
            if (normalizedFile.Length == 0)
                continue;

            string combined = string.IsNullOrWhiteSpace(DirectoryBase)
                ? normalizedFile
                : Normalize($"{DirectoryBase}/{normalizedFile}");

            yield return combined;
    }
}

    private static string Normalize(string value)
        => value.Trim().Replace('/', '\\').TrimStart('\\');
}

public sealed class AlphaSoundEntriesCatalog
{
    public AlphaSoundEntriesCatalog(IReadOnlyDictionary<int, AlphaSoundEntry> entries)
    {
        Entries = entries ?? throw new ArgumentNullException(nameof(entries));
    }

    public IReadOnlyDictionary<int, AlphaSoundEntry> Entries { get; }

    public bool TryResolve(uint soundNameId, out AlphaSoundEntry entry)
        => Entries.TryGetValue(checked((int)soundNameId), out entry!);
}
