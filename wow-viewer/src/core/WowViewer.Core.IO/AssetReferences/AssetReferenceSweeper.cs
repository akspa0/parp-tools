using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.AssetReferences;

/// <summary>
/// Collects every asset reference a build's world objects and models make, and resolves each against
/// what the build actually contains.
/// </summary>
/// <remarks>
/// <para>
/// The corpus comes from the archive catalogue's known-file set, which merges internal listfile
/// entries with per-asset wrapper archives scanned from disk. It is deliberately <b>not</b> taken from
/// internal listfiles alone: per-asset wrapper archives carry none, so that surface reports a single
/// world object for the Alpha corpus against an actual 532.
/// </para>
/// <para>
/// A single unreadable asset never aborts a sweep, and it is recorded distinctly from an asset that
/// read and referenced nothing. "Could not check" must never present as "nothing missing".
/// </para>
/// </remarks>
public sealed class AssetReferenceSweeper
{
    private static readonly string[] WorldObjectExtensions = [".wmo"];
    private static readonly string[] ModelExtensions = [".mdx", ".m2"];

    private readonly ArchiveCatalogSession _session;
    private readonly Dictionary<string, (AssetResolution Resolution, string? ResolvedPath)> _resolutionCache =
        new(StringComparer.OrdinalIgnoreCase);

    public AssetReferenceSweeper(ArchiveCatalogSession session)
    {
        _session = session ?? throw new ArgumentNullException(nameof(session));
    }

    /// <summary>Every file the build is known to contain, including per-asset wrapper archives.</summary>
    public IReadOnlyList<string> EnumerateCorpus() => _session.ArchiveCatalog.GetAllKnownFiles();

    /// <summary>
    /// Only the entries declared by archive internal listfiles. This is the "catalogued" set, and it
    /// is not the corpus — the difference between the two is itself a finding.
    /// </summary>
    public IReadOnlyList<string> EnumerateCatalogued() => _session.ArchiveCatalog.ExtractInternalListfiles();

    public static bool IsWorldObject(string path) => HasExtension(path, WorldObjectExtensions);

    public static bool IsModel(string path) => HasExtension(path, ModelExtensions);

    /// <summary>
    /// Sweeps the whole build. <paramref name="progress"/> is invoked with the number of assets
    /// examined so far, so a long corpus run is observable rather than silent.
    /// </summary>
    public SweepReport Sweep(BuildIdentity build, Action<int, int>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(build);

        IReadOnlyList<string> corpus = EnumerateCorpus();
        List<string> worldObjects = corpus.Where(IsWorldObject).Order(StringComparer.OrdinalIgnoreCase).ToList();
        List<string> models = corpus.Where(IsModel).Order(StringComparer.OrdinalIgnoreCase).ToList();

        List<ReferencingAssetResult> results = new(worldObjects.Count + models.Count);
        int total = worldObjects.Count + models.Count;
        int examined = 0;

        foreach (string path in worldObjects)
        {
            results.Add(SweepWorldObject(path));
            progress?.Invoke(++examined, total);
        }

        foreach (string path in models)
        {
            results.Add(SweepModel(path));
            progress?.Invoke(++examined, total);
        }

        return new SweepReport
        {
            Build = build,
            Results = results,
            BlockedRoutes = [],
            ReferenceKindsSwept =
            [
                AssetReferenceKind.PlacedDoodad,
                AssetReferenceKind.WorldObjectTexture,
                AssetReferenceKind.ModelTexture,
            ],
            WorldObjectsExamined = worldObjects.Count,
            ModelsExamined = models.Count,
        };
    }

    /// <summary>Extracts and resolves one world object's references.</summary>
    public ReferencingAssetResult SweepWorldObject(string path)
    {
        byte[]? bytes = ReadAsset(path);
        if (bytes is null)
            return Unreadable(path, "world object could not be read from the build");

        try
        {
            using MemoryStream stream = new(bytes, writable: false);
            IReadOnlyList<(AssetReferenceKind Kind, string Path)> extracted =
                WmoReferenceExtractor.Extract(stream, path);

            List<AssetReference> references = new(extracted.Count);
            foreach ((AssetReferenceKind kind, string target) in extracted)
            {
                (AssetResolution resolution, string? resolvedPath) = Resolve(target);
                references.Add(new AssetReference(path, kind, target, resolution, resolvedPath));
            }

            return new ReferencingAssetResult(path, ReferencingAssetState.Read, references);
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or EndOfStreamException or IndexOutOfRangeException or ArgumentException)
        {
            return Unreadable(path, ex.Message);
        }
    }

    /// <summary>Extracts and resolves one model's texture references.</summary>
    public ReferencingAssetResult SweepModel(string path)
    {
        byte[]? bytes = ReadAsset(path);
        if (bytes is null)
            return Unreadable(path, "model could not be read from the build");

        try
        {
            IReadOnlyList<string> textures = ModelReferenceExtractor.Extract(bytes, path);
            List<AssetReference> references = new(textures.Count);
            foreach (string target in textures)
            {
                (AssetResolution resolution, string? resolvedPath) = Resolve(target);
                references.Add(new AssetReference(path, AssetReferenceKind.ModelTexture, target, resolution, resolvedPath));
            }

            return new ReferencingAssetResult(path, ReferencingAssetState.Read, references);
        }
        catch (Exception ex) when (ex is InvalidDataException or NotSupportedException or EndOfStreamException or IndexOutOfRangeException or ArgumentException or OverflowException)
        {
            // A model whose format route does not read yet lands here. Its textures are unknown, which
            // is deliberately not the same as having none.
            return Unreadable(path, ex.Message);
        }
    }

    /// <summary>
    /// Source-format extensions that authored references use but builds do not ship, mapped to the
    /// compiled extension the client actually loads. Without this, authored model references read as
    /// missing en masse and bury the genuine misses among them.
    /// </summary>
    private static readonly (string From, string To)[] ExtensionSubstitutions =
    [
        (".mdl", ".mdx"),
        (".mdx", ".m2"),
        (".m2", ".mdx"),
    ];

    /// <summary>
    /// Determines whether a referenced asset can actually be obtained, and under what path. Cached,
    /// because a texture referenced by hundreds of models should be probed once.
    /// </summary>
    public (AssetResolution Resolution, string? ResolvedPath) Resolve(string targetPath)
    {
        if (string.IsNullOrWhiteSpace(targetPath))
            return (AssetResolution.Absent, null);

        if (_resolutionCache.TryGetValue(targetPath, out (AssetResolution, string?) cached))
            return cached;

        (AssetResolution, string?) resolved = ResolveUncached(targetPath);
        _resolutionCache[targetPath] = resolved;
        return resolved;
    }

    private (AssetResolution Resolution, string? ResolvedPath) ResolveUncached(string targetPath)
    {
        AssetResolution direct = Probe(targetPath);
        if (direct != AssetResolution.Absent)
            return (direct, null);

        foreach ((string from, string to) in ExtensionSubstitutions)
        {
            if (!targetPath.EndsWith(from, StringComparison.OrdinalIgnoreCase))
                continue;

            string candidate = string.Concat(targetPath.AsSpan(0, targetPath.Length - from.Length), to);
            AssetResolution substituted = Probe(candidate);
            if (substituted == AssetResolution.Present)
                return (AssetResolution.Present, candidate);
        }

        return (AssetResolution.Absent, null);
    }

    private AssetResolution Probe(string targetPath)
    {
        try
        {
            byte[]? bytes = _session.ReadFile(targetPath) ?? _session.TryReadFileFromDisk(targetPath);
            return bytes is { Length: > 0 } ? AssetResolution.Present : AssetResolution.Absent;
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            // Something answers to that path but cannot be read. Reporting this as Absent would
            // invent a missing asset, so it keeps its own state.
            return AssetResolution.Unreadable;
        }
    }

    private byte[]? ReadAsset(string path)
    {
        try
        {
            return _session.ReadFile(path) ?? _session.TryReadFileFromDisk(path);
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static ReferencingAssetResult Unreadable(string path, string detail)
        => new(path, ReferencingAssetState.Unreadable, [], detail);

    private static bool HasExtension(string path, string[] extensions)
    {
        foreach (string extension in extensions)
        {
            if (path.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }
}
