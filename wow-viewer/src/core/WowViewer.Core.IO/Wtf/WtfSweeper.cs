using System.Text;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Wtf;

/// <summary>
/// Finds and classifies every WTF file a build actually contains — loose on disk or packed inside the
/// build's own data archives.
/// </summary>
/// <remarks>
/// <para>
/// WTF files are not scanned into <see cref="MpqArchiveCatalog"/>'s loose-container tracking the way
/// WMO/WDT/WDL per-asset containers are — nothing calls a WTF-specific scan. The corpus here is instead
/// the union of each archive's own internal <c>(listfile)</c> (<see cref="IArchiveCatalog.ExtractInternalListfiles"/>,
/// which is how <c>WTF\DefaultBindings.wtf</c> was actually found packed inside 0.5.3.3368) and the
/// catalogue's broader known-file set (<see cref="IArchiveCatalog.GetAllKnownFiles"/>). Neither alone is
/// guaranteed complete — the same lesson Spec 155 already learned for the WMO corpus applies here.
/// </para>
/// <para>
/// A file whose name was never catalogued in either source is invisible to this sweep by construction —
/// that gap is what <see cref="ProbeCandidate"/> exists for: it tests a specific guessed name directly
/// against the archive's own hash table, independent of any listfile.
/// </para>
/// </remarks>
public sealed class WtfSweeper
{
    private readonly ArchiveCatalogSession _session;

    public WtfSweeper(ArchiveCatalogSession session)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
    }

    public IReadOnlyList<string> EnumerateCorpus()
    {
        HashSet<string> paths = new(StringComparer.OrdinalIgnoreCase);
        foreach (string path in _session.ArchiveCatalog.ExtractInternalListfiles())
        {
            if (path.EndsWith(".wtf", StringComparison.OrdinalIgnoreCase))
                paths.Add(path);
        }

        foreach (string path in _session.ArchiveCatalog.GetAllKnownFiles())
        {
            if (path.EndsWith(".wtf", StringComparison.OrdinalIgnoreCase))
                paths.Add(path);
        }

        // Neither source above sees an arbitrary loose file — nothing scans for .wtf the way
        // ScanWmoMpqArchives scans for .wmo.mpq. Config.wtf, realmlist.wtf, and every per-account
        // cache file are genuinely loose-only and would otherwise be invisible to this sweep entirely.
        foreach (string archiveRoot in _session.ArchiveRoots)
        {
            if (string.IsNullOrWhiteSpace(archiveRoot) || !Directory.Exists(archiveRoot))
                continue;

            string fullRoot = Path.GetFullPath(archiveRoot);
            foreach (string diskPath in Directory.EnumerateFiles(fullRoot, "*.wtf", SearchOption.AllDirectories))
            {
                string relative = Path.GetRelativePath(fullRoot, diskPath);
                paths.Add(relative);
            }
        }

        return paths.Order(StringComparer.OrdinalIgnoreCase).ToList();
    }

    public WtfBuildSurvey Sweep(string buildLabel)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(buildLabel);

        List<WtfFileSurvey> results = EnumerateCorpus().Select(SweepFile).ToList();
        return new WtfBuildSurvey(buildLabel, results);
    }

    public WtfFileSurvey SweepFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        (byte[]? bytes, WtfFileSource source) = ReadWithSource(path);
        if (bytes is null)
            return new WtfFileSurvey(path, WtfFileSource.Loose, [], "could not be read from the build");

        try
        {
            string text = Encoding.UTF8.GetString(bytes);
            IReadOnlyList<WtfLine> lines = WtfLineClassifier.ClassifyFile(text);
            return new WtfFileSurvey(path, source, lines);
        }
        catch (DecoderFallbackException ex)
        {
            return new WtfFileSurvey(path, source, [], $"not decodable as UTF-8: {ex.Message}");
        }
    }

    /// <summary>
    /// Tests one guessed file name directly against the build's archives, bypassing every listfile.
    /// Resolution goes through the same <see cref="ArchiveCatalogSession"/> hash-table lookup every
    /// other read in this build uses — a name that resolves here was never "found" by any sweep, it was
    /// correctly guessed.
    /// </summary>
    public WtfCandidateProbeResult ProbeCandidate(string candidateName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(candidateName);

        (byte[]? bytes, WtfFileSource source) = ReadWithSource(candidateName);
        if (bytes is null)
            return new WtfCandidateProbeResult(candidateName, Resolved: false, Survey: null);

        string text = Encoding.UTF8.GetString(bytes);
        IReadOnlyList<WtfLine> lines = WtfLineClassifier.ClassifyFile(text);
        return new WtfCandidateProbeResult(candidateName, Resolved: true, new WtfFileSurvey(candidateName, source, lines));
    }

    private (byte[]? Bytes, WtfFileSource Source) ReadWithSource(string path)
    {
        byte[]? loose = _session.TryReadFileFromDisk(path);
        if (loose is { Length: > 0 })
            return (loose, WtfFileSource.Loose);

        byte[]? archived = _session.ReadFile(path);
        if (archived is { Length: > 0 })
            return (archived, WtfFileSource.Archive);

        return (null, WtfFileSource.Loose);
    }
}
