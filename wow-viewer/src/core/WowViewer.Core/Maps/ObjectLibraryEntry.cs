using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.Maps;

/// <summary>
/// Asset type for an entry in the per-object capture library.
/// Aligned with the spec 077 §1.1 contract: <c>m2 | mdx | wmo</c>.
/// </summary>
public enum ObjectAssetType
{
    Unknown,
    M2,
    Mdx,
    Wmo,
}

/// <summary>
/// Capture status for an object library entry. An entry may be observed in
/// harvested placement data long before a capture run ever touches the asset.
/// The <see cref="NotAttempted"/> value is the explicit no-signal state
/// required by spec 077 FR-026.
/// </summary>
public enum ObjectCaptureStatus
{
    NotAttempted,
    Captured,
    Partial,
    Failed,
}

/// <summary>
/// Whether the asset is expected to contribute visibly to the baked minimap.
/// Spec 077 FR-026 requires explicit low-visibility / clutter states rather
/// than silently dropping them from the library.
/// </summary>
public enum ObjectVisibilityClass
{
    Unknown,
    RoofVisible,
    LikelyVisible,
    LikelyHidden,
    ClutterFiltered,
}

/// <summary>
/// Operator review state for an object library entry. Library consumers MUST
/// NOT treat <see cref="Unreviewed"/> entries as authoritative.
/// </summary>
public enum ObjectReviewState
{
    Unreviewed,
    Accepted,
    Rejected,
    NeedsFollowup,
}

/// <summary>
/// Canonical per-asset record for the spec 077 object library. One entry
/// exists per normalized asset path. The <see cref="LibraryId"/> is a
/// deterministic SHA1-derived identifier keyed on
/// <see cref="NormalizedAssetPath"/>; callers MUST NOT regenerate it on
/// every load.
/// </summary>
public sealed record ObjectLibraryEntry
{
    public string LibraryId { get; init; } = string.Empty;
    public string OriginalAssetPath { get; init; } = string.Empty;
    public string NormalizedAssetPath { get; init; } = string.Empty;
    public ObjectAssetType AssetType { get; init; } = ObjectAssetType.Unknown;
    public ObjectCaptureStatus CaptureStatus { get; init; } = ObjectCaptureStatus.NotAttempted;
    public ObjectVisibilityClass VisibilityClass { get; init; } = ObjectVisibilityClass.Unknown;
    public ObjectReviewState ReviewState { get; init; } = ObjectReviewState.Unreviewed;
    public IReadOnlyList<string> SourceBuilds { get; init; } = Array.Empty<string>();
    public IReadOnlyList<string> SourceMaps { get; init; } = Array.Empty<string>();
    public int PlacementObservationCount { get; init; }
    public string? PreferredVariantId { get; init; }

    /// <summary>
    /// Deterministic library id for a given normalized asset path. Stable
    /// across runs and platforms; see spec 077 §1.1.
    /// </summary>
    public static string ComputeLibraryId(string normalizedAssetPath)
    {
        if (string.IsNullOrWhiteSpace(normalizedAssetPath))
            return string.Empty;
        Span<byte> hash = stackalloc byte[20];
        SHA1.HashData(Encoding.UTF8.GetBytes(normalizedAssetPath), hash);
        string hex = Convert.ToHexString(hash).ToLowerInvariant();
        return $"objlib_{hex[..14]}";
    }
}
