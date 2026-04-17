using System.Collections.ObjectModel;

namespace WowViewer.Core.Lit;

public sealed class LitSummary
{
    public LitSummary(
        string sourcePath,
        uint versionNumber,
        int lightCount,
        int listEntryCount,
        IReadOnlyList<LitListEntrySummary> entries,
        bool usesSinglePartialEntry,
        bool hasDefaultFirstEntry,
        int namedEntryCount,
        int remainingPayloadBytes)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(listEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(namedEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(remainingPayloadBytes);
        ArgumentNullException.ThrowIfNull(entries);

        SourcePath = sourcePath;
        VersionNumber = versionNumber;
        LightCount = lightCount;
        ListEntryCount = listEntryCount;
        Entries = new ReadOnlyCollection<LitListEntrySummary>(entries.ToArray());
        UsesSinglePartialEntry = usesSinglePartialEntry;
        HasDefaultFirstEntry = hasDefaultFirstEntry;
        NamedEntryCount = namedEntryCount;
        RemainingPayloadBytes = remainingPayloadBytes;
    }

    public string SourcePath { get; }

    public uint VersionNumber { get; }

    public int LightCount { get; }

    public int ListEntryCount { get; }

    public IReadOnlyList<LitListEntrySummary> Entries { get; }

    public bool UsesSinglePartialEntry { get; }

    public bool HasDefaultFirstEntry { get; }

    public int NamedEntryCount { get; }

    public int RemainingPayloadBytes { get; }
}