using System.Numerics;
using System.Security.Cryptography;
using System.Text;

namespace WowViewer.Core.Maps;

/// <summary>
/// How an <see cref="ObjectCaptureVariant"/> image and mask were produced.
/// The first spec 077 implementation slice uses
/// <see cref="OrthographicTopdown"/> for M2/WMO captures rendered through
/// the headless validation-capture lane; <see cref="GeometryProjection"/>
/// is reserved for future synthetic captures derived from placement
/// geometry without an OpenGL render pass.
/// </summary>
public enum ObjectCaptureMode
{
    Unknown,
    OrthographicTopdown,
    GeometryProjection,
    Hybrid,
}

/// <summary>
/// Inclusive tile-space bounding box used by the object library. Coordinates
/// are row/column pixel positions aligned to the same coordinate system as
/// the minimap tile (top-left origin, axis aligned, no rotation applied).
/// </summary>
public readonly record struct ObjectLibraryBoundingBox(int X0, int Y0, int X1, int Y1)
{
    public int Width => Math.Max(0, X1 - X0);
    public int Height => Math.Max(0, Y1 - Y0);

    public bool IsEmpty => Width <= 0 || Height <= 0;
}

/// <summary>
/// One concrete captured view of an object library asset. Multiple variants
/// per asset are allowed when rotation, scale, or visibility class
/// materially changes the top-down appearance; see spec 077 FR-007.
/// </summary>
public sealed record ObjectCaptureVariant
{
    public string VariantId { get; init; } = string.Empty;
    public string LibraryId { get; init; } = string.Empty;
    public string CaptureBuild { get; init; } = string.Empty;
    public ObjectCaptureMode CaptureMode { get; init; } = ObjectCaptureMode.Unknown;
    public ObjectAssetType AssetType { get; init; } = ObjectAssetType.Unknown;
    public string ImageKey { get; init; } = string.Empty;
    public string MaskKey { get; init; } = string.Empty;
    public ObjectLibraryBoundingBox BboxLocal { get; init; }
    public Vector3 Rotation { get; init; }
    public float Scale { get; init; } = 1f;
    public string CaptureNotes { get; init; } = string.Empty;
    public float CaptureConfidence { get; init; }

    /// <summary>
    /// Deterministic variant id derived from
    /// <paramref name="libraryId"/> + capture pose. Stable across runs and
    /// platforms.
    /// </summary>
    public static string ComputeVariantId(
        string libraryId,
        string captureBuild,
        ObjectCaptureMode captureMode,
        Vector3 rotation,
        float scale)
    {
        if (string.IsNullOrWhiteSpace(libraryId))
            return string.Empty;
        string payload = string.Join(
            '|',
            libraryId,
            captureBuild ?? string.Empty,
            FormatCaptureMode(captureMode),
            rotation.X.ToString("G9", System.Globalization.CultureInfo.InvariantCulture),
            rotation.Y.ToString("G9", System.Globalization.CultureInfo.InvariantCulture),
            rotation.Z.ToString("G9", System.Globalization.CultureInfo.InvariantCulture),
            scale.ToString("G9", System.Globalization.CultureInfo.InvariantCulture));
        Span<byte> hash = stackalloc byte[20];
        SHA1.HashData(Encoding.UTF8.GetBytes(payload), hash);
        string hex = Convert.ToHexString(hash).ToLowerInvariant();
        return $"objvar_{hex[..16]}";
    }

    private static string FormatCaptureMode(ObjectCaptureMode captureMode) => captureMode switch
    {
        ObjectCaptureMode.OrthographicTopdown => "orthographic_topdown",
        ObjectCaptureMode.GeometryProjection => "geometry_projection",
        ObjectCaptureMode.Hybrid => "hybrid",
        _ => "unknown",
    };
}
