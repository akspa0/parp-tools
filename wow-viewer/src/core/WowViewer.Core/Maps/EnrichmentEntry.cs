namespace WowViewer.Core.Maps;

/// <summary>
/// One decoded asset from the V22 enrichment stream.
/// </summary>
/// <param name="CanonicalPath">Stable canonical asset path (e.g. "World/M2/Peasant.m2").</param>
/// <param name="Kind">Asset kind: M2, WMO, or BLP/texture.</param>
/// <param name="LoadError">0 = successfully decoded, 1 = decode failed.</param>
/// <param name="Arrays">Named arrays for this entry. For M2: vertices, normals, triangles, etc. For WMO: vertices, triangles, materials, etc. For BLP: texture_rgb, texture_shape.</param>
public sealed record EnrichmentEntry(
    string CanonicalPath,
    AssetKind Kind,
    int LoadError,
    IReadOnlyList<EnrichmentArray> Arrays);

/// <summary>
/// Asset kind in the enrichment stream.
/// </summary>
public enum AssetKind : byte
{
    Unknown = 0,
    M2 = 1,
    Wmo = 2,
    Blp = 3,
}