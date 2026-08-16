namespace WowViewer.Core.IO.AssetReferences;

/// <summary>What kind of claim an asset is making.</summary>
public enum AssetReferenceKind
{
    /// <summary>A world object placing a doodad model.</summary>
    PlacedDoodad,

    /// <summary>A texture named by a world object's material table.</summary>
    WorldObjectTexture,

    /// <summary>A texture named by a model.</summary>
    ModelTexture,
}

/// <summary>
/// Whether a referenced asset could be obtained. Three states, deliberately: collapsing
/// <see cref="Unreadable"/> into <see cref="Absent"/> would manufacture missing assets, which is the
/// one direction this analysis must never be wrong in.
/// </summary>
public enum AssetResolution
{
    /// <summary>The asset was read from the build.</summary>
    Present,

    /// <summary>The asset could not be obtained from the build by any route.</summary>
    Absent,

    /// <summary>Something is there under that path but it could not be read.</summary>
    Unreadable,
}

/// <summary>Why a referencing asset produced no references.</summary>
public enum ReferencingAssetState
{
    /// <summary>Read successfully; any reference count including zero is a real observation.</summary>
    Read,

    /// <summary>This individual asset failed to read. Its references are unknown, not absent.</summary>
    Unreadable,

    /// <summary>This asset's format route is known not to read in this build. Not attempted.</summary>
    RouteBlocked,
}

/// <summary>The build an observation came from. Travels with every record.</summary>
public sealed record BuildIdentity(string Label, string RootLabel)
{
    public override string ToString() => Label;
}

/// <summary>
/// One claim by one asset that another asset exists.
/// </summary>
/// <param name="ResolvedPath">
/// The path the asset was actually obtained from, when that differs from <paramref name="TargetPath"/>.
/// Authored references routinely name a source-format extension the build does not ship; the client
/// substitutes the compiled one. Recording the substitution keeps it a visible fact rather than
/// hiding it behind a successful resolve — 363 of 364 apparently-missing model references in the
/// Alpha corpus are this, and treating them as missing would bury the 3 that genuinely are.
/// </param>
public sealed record AssetReference(
    string SourcePath,
    AssetReferenceKind Kind,
    string TargetPath,
    AssetResolution Resolution,
    string? ResolvedPath = null)
{
    /// <summary>True when the asset was found under a different path than the one referenced.</summary>
    public bool ResolvedBySubstitution => ResolvedPath is not null;
}

/// <summary>
/// The result of examining one referencing asset. A zero-length <see cref="References"/> means
/// something different for each <see cref="State"/>, which is why the state is carried rather than
/// inferred from the count.
/// </summary>
public sealed record ReferencingAssetResult(
    string Path,
    ReferencingAssetState State,
    IReadOnlyList<AssetReference> References,
    string? FailureDetail = null);

/// <summary>A format route that was not swept, and how much was skipped because of it.</summary>
public sealed record BlockedRoute(string Route, int AssetCount, string Reason);

/// <summary>
/// One build's sweep. Its job is to make incompleteness impossible to miss: examined counts are
/// always present, and a report carrying blocked routes is not a complete picture.
/// </summary>
public sealed record SweepReport
{
    public required BuildIdentity Build { get; init; }

    public required IReadOnlyList<ReferencingAssetResult> Results { get; init; }

    public required IReadOnlyList<BlockedRoute> BlockedRoutes { get; init; }

    public required IReadOnlyList<AssetReferenceKind> ReferenceKindsSwept { get; init; }

    public int WorldObjectsExamined { get; init; }

    public int ModelsExamined { get; init; }

    /// <summary>
    /// WMO group files (the <c>Name_NNN.wmo</c> convention) found alongside root files in the corpus
    /// and excluded from <see cref="WorldObjectsExamined"/> — they carry geometry, not root-level
    /// references, so sweeping them would always fail and would misreport as unreadable assets. Kept
    /// as its own count so the corpus accounting stays fully explained rather than presenting a smaller
    /// examined number with no stated reason.
    /// </summary>
    public int WorldObjectGroupFilesExcluded { get; init; }

    public int AssetsUnreadable => Results.Count(static r => r.State == ReferencingAssetState.Unreadable);

    public IEnumerable<AssetReference> AllReferences => Results.SelectMany(static r => r.References);

    public int ReferenceCount => Results.Sum(static r => r.References.Count);

    public int UnresolvedReferenceCount
        => Results.Sum(static r => r.References.Count(static reference => reference.Resolution == AssetResolution.Absent));

    /// <summary>
    /// References that only resolved because the build ships the asset under a different extension.
    /// A large number here is a property of how the data was authored, not a defect.
    /// </summary>
    public int SubstitutedReferenceCount
        => Results.Sum(static r => r.References.Count(static reference => reference.ResolvedBySubstitution));

    /// <summary>
    /// Distinct asset paths that were referenced but could not be obtained. This is the headline
    /// number: the size of the missing-asset population.
    /// </summary>
    public IReadOnlyList<string> DistinctMissingTargets
        => AllReferences
            .Where(static reference => reference.Resolution == AssetResolution.Absent)
            .Select(static reference => reference.TargetPath)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Order(StringComparer.OrdinalIgnoreCase)
            .ToList();

    /// <summary>
    /// False when any route was blocked or any asset was unreadable. A consumer presenting findings
    /// from an incomplete sweep must say so — "could not check" must never read as "nothing missing".
    /// </summary>
    public bool Complete => BlockedRoutes.Count == 0 && AssetsUnreadable == 0;
}
