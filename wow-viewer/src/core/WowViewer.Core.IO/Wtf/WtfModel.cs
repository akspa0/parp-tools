namespace WowViewer.Core.IO.Wtf;

/// <summary>What a WTF line's statement shape was recognized as.</summary>
public enum WtfLineKind
{
    /// <summary><c>SET name "value"</c> — a client configuration variable.</summary>
    Set,

    /// <summary><c>bind KEY ACTION</c> — a keybinding, confirmed real via WTF\DefaultBindings.wtf.</summary>
    Bind,

    /// <summary>Did not match any recognized statement shape. The finding this tool exists to surface.</summary>
    Unrecognized,
}

/// <summary>
/// One non-blank line from a WTF file, classified. <see cref="RawText"/> is always the exact original
/// text — an unrecognized line's real syntax is the entire point of sweeping, so it is never paraphrased
/// or dropped.
/// </summary>
public sealed record WtfLine(string RawText, WtfLineKind Kind, string? Name = null, string? Value = null);

/// <summary>Where a WTF file's bytes actually came from.</summary>
public enum WtfFileSource
{
    /// <summary>Loose on disk.</summary>
    Loose,

    /// <summary>Packed inside one of the build's own data archives.</summary>
    Archive,
}

/// <summary>One WTF file's swept results.</summary>
public sealed record WtfFileSurvey(
    string Path,
    WtfFileSource Source,
    IReadOnlyList<WtfLine> Lines,
    string? FailureDetail = null)
{
    public bool Readable => FailureDetail is null;

    public int RecognizedCount => Lines.Count(static l => l.Kind != WtfLineKind.Unrecognized);

    public int UnrecognizedCount => Lines.Count(static l => l.Kind == WtfLineKind.Unrecognized);
}

/// <summary>One build's aggregated WTF sweep results.</summary>
public sealed record WtfBuildSurvey(string BuildLabel, IReadOnlyList<WtfFileSurvey> Files)
{
    public int FilesFound => Files.Count;

    public int FilesUnreadable => Files.Count(static f => !f.Readable);

    public int TotalLines => Files.Sum(static f => f.Readable ? f.Lines.Count : 0);

    public int TotalRecognized => Files.Sum(static f => f.Readable ? f.RecognizedCount : 0);

    public int TotalUnrecognized => Files.Sum(static f => f.Readable ? f.UnrecognizedCount : 0);

    /// <summary>Distinct unrecognized line texts across every readable file, so a shape repeated
    /// thousands of times is visible once rather than burying the reader in repetition.</summary>
    public IReadOnlyList<string> DistinctUnrecognizedLines
        => Files
            .Where(static f => f.Readable)
            .SelectMany(static f => f.Lines)
            .Where(static l => l.Kind == WtfLineKind.Unrecognized)
            .Select(static l => l.RawText)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToList();
}

/// <summary>The outcome of testing one guessed file name directly against a build's archives.</summary>
public sealed record WtfCandidateProbeResult(string CandidateName, bool Resolved, WtfFileSurvey? Survey);
