namespace WowViewer.Core.Maps;

/// <summary>
/// Narrow recovery policy for terrain-material consumers that need an RGB proxy when a referenced
/// diffuse BLP is absent or undecodable. It deliberately does not claim that the companion carries
/// equivalent engine material semantics.
/// </summary>
public static class TerrainTextureFallbackPolicy
{
    public const string SpecularCompanionRgbProxy = "specular_companion_rgb_proxy";
    public const string RelatedDiffuseRgbProxy = "related_diffuse_rgb_proxy";
    public const string CatalogRgbLastResortProxy = "catalog_rgb_last_resort_proxy";

    private const int MaximumRelatedCandidates = 16;
    private const int MaximumCatalogLastResortCandidates = 64;

    /// <summary>
    /// Returns the same-stem <c>_s.blp</c> companion for a non-specular BLP path. The caller must
    /// still prove the candidate exists and decodes before using it.
    /// </summary>
    public static string? GetSpecularCompanionPath(string requestedPath)
    {
        if (string.IsNullOrWhiteSpace(requestedPath)
            || !requestedPath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        int extensionIndex = requestedPath.LastIndexOf('.');
        string stem = requestedPath[..extensionIndex];
        if (stem.EndsWith("_s", StringComparison.OrdinalIgnoreCase))
            return null;

        return stem + "_s" + requestedPath[extensionIndex..];
    }

    /// <summary>
    /// Returns an ordered, deliberately narrow set of RGB-proxy candidates. The original caller
    /// must first attempt <paramref name="requestedPath"/> itself, and must prove that every
    /// returned candidate decodes before accepting it. Related diffuse candidates are searched
    /// across the known catalog because early clients moved assets without repairing every MTEX
    /// reference. Exact/strong basename matches rank first, then shared directory-theme tokens;
    /// material companions such as <c>_s</c>, <c>_n</c>, and <c>_h</c> are excluded from that tier.
    /// </summary>
    public static IReadOnlyList<TerrainTextureFallbackCandidate> GetRgbProxyCandidates(
        string requestedPath,
        IEnumerable<string> knownTexturePaths)
    {
        ArgumentNullException.ThrowIfNull(knownTexturePaths);

        if (string.IsNullOrWhiteSpace(requestedPath)
            || !requestedPath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        var candidates = new List<TerrainTextureFallbackCandidate>();

        string? companionPath = GetSpecularCompanionPath(requestedPath);
        if (companionPath is not null)
        {
            candidates.Add(new TerrainTextureFallbackCandidate(
                companionPath,
                SpecularCompanionRgbProxy));
        }

        candidates.AddRange(GetRelatedDiffuseRgbProxyCandidates(requestedPath, knownTexturePaths));
        return candidates;
    }

    /// <summary>
    /// Returns only catalog-wide related diffuse candidates, after a caller has separately tried
    /// the original path and its same-stem <c>_s.blp</c> companion. This split keeps the common
    /// companion recovery path from scanning the whole listfile.
    /// </summary>
    public static IReadOnlyList<TerrainTextureFallbackCandidate> GetRelatedDiffuseRgbProxyCandidates(
        string requestedPath,
        IEnumerable<string> knownTexturePaths)
    {
        ArgumentNullException.ThrowIfNull(knownTexturePaths);

        if (string.IsNullOrWhiteSpace(requestedPath)
            || !requestedPath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        string normalizedRequestedPath = NormalizeVirtualPath(requestedPath);
        string requestedDirectory = GetDirectory(normalizedRequestedPath);
        if (string.IsNullOrWhiteSpace(requestedDirectory))
            return [];

        string requestedStem = GetFileStem(normalizedRequestedPath);
        HashSet<string> requestedTokens = GetMeaningfulTokens(requestedStem, requestedDirectory);
        HashSet<string> requestedDirectoryTokens = GetDirectoryThemeTokens(requestedDirectory);
        if (requestedTokens.Count == 0)
            return [];

        var related = new List<(string Path, int Score)>();
        var addedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string rawCandidatePath in knownTexturePaths)
        {
            if (string.IsNullOrWhiteSpace(rawCandidatePath))
                continue;

            string candidatePath = NormalizeVirtualPath(rawCandidatePath);
            if (candidatePath.Equals(normalizedRequestedPath, StringComparison.OrdinalIgnoreCase)
                || !candidatePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            string candidateStem = GetFileStem(candidatePath);
            if (!IsOrdinaryDiffuseStem(candidateStem))
                continue;

            string candidateDirectory = GetDirectory(candidatePath);
            int score = ScoreRelatedDiffuse(
                requestedStem,
                requestedTokens,
                requestedDirectory,
                requestedDirectoryTokens,
                candidateStem,
                candidateDirectory);
            if (score <= 0 || !addedPaths.Add(candidatePath))
                continue;

            related.Add((rawCandidatePath, score));
        }

        return related
            .OrderByDescending(static candidate => candidate.Score)
            .ThenBy(static candidate => NormalizeVirtualPath(candidate.Path), StringComparer.OrdinalIgnoreCase)
            .Take(MaximumRelatedCandidates)
            .Select(static candidate => new TerrainTextureFallbackCandidate(
                candidate.Path,
                RelatedDiffuseRgbProxy))
            .ToArray();
    }

    /// <summary>
    /// Returns a deterministic last-resort list of ordinary, cataloged BLPs. This tier is for a
    /// readable terrain tile whose MTEX paths cannot be resolved after the same-stem and related
    /// recovery tiers. It deliberately prefers the original directory, then the same top-level
    /// terrain family (for example <c>Tileset</c>), before a generic ordinary diffuse BLP. The
    /// consumer must still prove a candidate decodes and record the substitution.
    /// </summary>
    public static IReadOnlyList<TerrainTextureFallbackCandidate> GetCatalogRgbLastResortCandidates(
        string requestedPath,
        IEnumerable<string> knownTexturePaths)
    {
        ArgumentNullException.ThrowIfNull(knownTexturePaths);

        string normalizedRequestedPath = NormalizeVirtualPath(requestedPath);
        string requestedDirectory = GetDirectory(normalizedRequestedPath);
        string requestedRoot = GetTopLevelDirectory(requestedDirectory);
        HashSet<string> requestedDirectoryTokens = GetDirectoryThemeTokens(requestedDirectory);

        var candidates = new List<(string Path, int Score)>();
        var addedPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (string rawCandidatePath in knownTexturePaths)
        {
            if (string.IsNullOrWhiteSpace(rawCandidatePath))
                continue;

            string candidatePath = NormalizeVirtualPath(rawCandidatePath);
            if (candidatePath.Equals(normalizedRequestedPath, StringComparison.OrdinalIgnoreCase)
                || !candidatePath.EndsWith(".blp", StringComparison.OrdinalIgnoreCase)
                || !IsOrdinaryDiffuseStem(GetFileStem(candidatePath))
                || !addedPaths.Add(candidatePath))
            {
                continue;
            }

            string candidateDirectory = GetDirectory(candidatePath);
            int score = ScoreCatalogLastResort(
                requestedDirectory,
                requestedRoot,
                requestedDirectoryTokens,
                candidateDirectory);
            candidates.Add((rawCandidatePath, score));
        }

        return candidates
            .OrderByDescending(static candidate => candidate.Score)
            .ThenBy(static candidate => NormalizeVirtualPath(candidate.Path), StringComparer.OrdinalIgnoreCase)
            .Take(MaximumCatalogLastResortCandidates)
            .Select(static candidate => new TerrainTextureFallbackCandidate(
                candidate.Path,
                CatalogRgbLastResortProxy))
            .ToArray();
    }

    private static int ScoreRelatedDiffuse(
        string requestedStem,
        IReadOnlySet<string> requestedTokens,
        string requestedDirectory,
        IReadOnlySet<string> requestedDirectoryTokens,
        string candidateStem,
        string candidateDirectory)
    {
        HashSet<string> candidateTokens = GetMeaningfulTokens(candidateStem, candidateDirectory);
        int sharedTokenCharacters = requestedTokens
            .Intersect(candidateTokens, StringComparer.OrdinalIgnoreCase)
            .Sum(static token => token.Length);

        int commonPrefixLength = 0;
        int comparedLength = Math.Min(requestedStem.Length, candidateStem.Length);
        while (commonPrefixLength < comparedLength
               && char.ToUpperInvariant(requestedStem[commonPrefixLength])
                   == char.ToUpperInvariant(candidateStem[commonPrefixLength]))
        {
            commonPrefixLength++;
        }

        int longestNameLength = Math.Max(requestedStem.Length, candidateStem.Length);
        int prefixPercent = longestNameLength == 0
            ? 0
            : (100 * commonPrefixLength) / longestNameLength;
        bool exactStemMatch = requestedStem.Equals(candidateStem, StringComparison.OrdinalIgnoreCase);
        bool strongNameMatch = exactStemMatch || prefixPercent >= 70 || sharedTokenCharacters >= 3;
        if (!strongNameMatch)
            return 0;

        int sharedDirectoryTokenCharacters = requestedDirectoryTokens
            .Intersect(GetDirectoryThemeTokens(candidateDirectory), StringComparer.OrdinalIgnoreCase)
            .Sum(static token => token.Length);
        int score = (sharedTokenCharacters * 100) + prefixPercent + (sharedDirectoryTokenCharacters * 10);
        if (requestedDirectory.Equals(candidateDirectory, StringComparison.OrdinalIgnoreCase))
            score += 75;
        if (prefixPercent >= 70)
            score += 500;
        if (exactStemMatch)
            score += 10_000;

        return score;
    }

    private static int ScoreCatalogLastResort(
        string requestedDirectory,
        string requestedRoot,
        IReadOnlySet<string> requestedDirectoryTokens,
        string candidateDirectory)
    {
        if (!string.IsNullOrWhiteSpace(requestedDirectory)
            && requestedDirectory.Equals(candidateDirectory, StringComparison.OrdinalIgnoreCase))
        {
            return 100_000;
        }

        string candidateRoot = GetTopLevelDirectory(candidateDirectory);
        int score = 0;
        if (!string.IsNullOrWhiteSpace(requestedRoot)
            && requestedRoot.Equals(candidateRoot, StringComparison.OrdinalIgnoreCase))
        {
            score += 10_000;
        }

        int sharedDirectoryTokenCharacters = requestedDirectoryTokens
            .Intersect(GetDirectoryThemeTokens(candidateDirectory), StringComparer.OrdinalIgnoreCase)
            .Sum(static token => token.Length);
        score += sharedDirectoryTokenCharacters * 100;

        // A generic terrain-family texture is a safer emergency material than an arbitrary UI or
        // character BLP when the original directory provided no usable candidates.
        if (candidateRoot.Equals("tileset", StringComparison.OrdinalIgnoreCase))
            score += 1_000;
        else if (candidateRoot.Equals("terrain", StringComparison.OrdinalIgnoreCase))
            score += 750;
        else if (candidateRoot.Equals("world", StringComparison.OrdinalIgnoreCase))
            score += 500;

        return score;
    }

    private static HashSet<string> GetMeaningfulTokens(string stem, string directory)
    {
        string folderName = GetFolderName(directory);
        HashSet<string> tokens = Tokenize(stem);
        tokens.Remove(folderName);
        return tokens;
    }

    private static HashSet<string> GetDirectoryThemeTokens(string directory)
    {
        var ignored = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "art", "environment", "terrain", "texture", "textures", "tileset", "world",
        };
        HashSet<string> tokens = Tokenize(directory);
        tokens.ExceptWith(ignored);
        return tokens;
    }

    private static HashSet<string> Tokenize(string text)
    {
        var tokens = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var current = new System.Text.StringBuilder();

        void AddCurrentToken()
        {
            if (current.Length >= 3)
                tokens.Add(current.ToString());

            current.Clear();
        }

        for (int index = 0; index < text.Length; index++)
        {
            char currentCharacter = text[index];
            if (!char.IsLetterOrDigit(currentCharacter))
            {
                AddCurrentToken();
                continue;
            }

            char? previous = index > 0 ? text[index - 1] : null;
            char? next = index + 1 < text.Length ? text[index + 1] : null;
            bool startsCamelCaseToken = char.IsUpper(currentCharacter)
                && previous.HasValue
                && (char.IsLower(previous.Value)
                    || (char.IsUpper(previous.Value) && next.HasValue && char.IsLower(next.Value)));
            bool startsNumericToken = char.IsDigit(currentCharacter)
                && previous.HasValue
                && !char.IsDigit(previous.Value);
            bool resumesWordToken = char.IsLetter(currentCharacter)
                && previous.HasValue
                && char.IsDigit(previous.Value);
            if (startsCamelCaseToken || startsNumericToken || resumesWordToken)
                AddCurrentToken();

            current.Append(char.ToLowerInvariant(currentCharacter));
        }

        AddCurrentToken();
        return tokens;
    }

    private static bool IsOrdinaryDiffuseStem(string stem)
    {
        string[] excludedMaterialSuffixes =
        [
            "_s", "_n", "_h", "_m", "_g", "_r", "_a",
            "_spec", "_normal", "_height", "_gloss", "_emissive", "_opacity", "_mask",
        ];

        return !excludedMaterialSuffixes.Any(suffix => stem.EndsWith(suffix, StringComparison.OrdinalIgnoreCase));
    }

    private static string NormalizeVirtualPath(string path) => path.Trim().Replace('/', '\\');

    private static string GetDirectory(string path)
    {
        int separatorIndex = path.LastIndexOf('\\');
        return separatorIndex > 0 ? path[..separatorIndex] : string.Empty;
    }

    private static string GetFolderName(string directory)
    {
        int separatorIndex = directory.LastIndexOf('\\');
        return separatorIndex >= 0 ? directory[(separatorIndex + 1)..] : directory;
    }

    private static string GetTopLevelDirectory(string directory)
    {
        int separatorIndex = directory.IndexOf('\\');
        return separatorIndex >= 0 ? directory[..separatorIndex] : directory;
    }

    private static string GetFileStem(string path)
    {
        int separatorIndex = path.LastIndexOf('\\');
        int extensionIndex = path.LastIndexOf('.');
        return extensionIndex > separatorIndex ? path[(separatorIndex + 1)..extensionIndex] : path[(separatorIndex + 1)..];
    }
}

/// <summary>
/// A candidate selected by the deterministic terrain RGB-proxy policy. It remains a candidate
/// until the consumer has successfully decoded the referenced BLP.
/// </summary>
public sealed record TerrainTextureFallbackCandidate(
    string ResolvedPath,
    string ResolutionKind);

/// <summary>
/// A declared terrain-texture substitution. The original MTEX entry remains authoritative; this
/// record states that a decodable RGB proxy was used only for a derived artifact.
/// </summary>
public sealed record TerrainTextureFallbackResolution(
    int TextureId,
    string RequestedPath,
    string ResolvedPath,
    string ResolutionKind);
