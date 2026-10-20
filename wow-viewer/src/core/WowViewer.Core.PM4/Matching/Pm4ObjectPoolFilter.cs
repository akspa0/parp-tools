using WowViewer.Core.PM4.Models;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Explicit opt-in search-space restriction. Does not rank candidates or alter geometry scores.
/// Design-kit/class labels are metadata; eligibility uses declared kinds and path selectors only.
/// </summary>
public static class Pm4ObjectPoolFilter
{
    public static Pm4ObjectPoolSelection Select(
        Pm4ObjectPoolManifest manifest, SurfaceCorrelationDatabase database,
        string mapName, int tileX, int tileY)
    {
        Validate(manifest);
        ArgumentNullException.ThrowIfNull(database);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ValidateTile(tileX, tileY);
        var pools = string.Equals(manifest.MapName, mapName, StringComparison.OrdinalIgnoreCase)
            ? manifest.Assignments.Where(a => tileX >= a.MinTileX && tileX <= a.MaxTileX
                && tileY >= a.MinTileY && tileY <= a.MaxTileY)
                .Select(a => manifest.Pools.Single(p => string.Equals(p.Id, a.PoolId, StringComparison.OrdinalIgnoreCase)))
                .DistinctBy(p => p.Id, StringComparer.OrdinalIgnoreCase).ToArray()
            : [];
        if (pools.Length == 0)
            return new Pm4ObjectPoolSelection(mapName, tileX, tileY, false, [], database.Records.Count, database.Records.Count, database);

        var eligible = database.Records.Where(r => pools.Any(p => Allows(p, r))).ToArray();
        var filtered = database with
        {
            Records = eligible,
            WmoCount = eligible.Count(r => r.AssetKind.Equals("wmo", StringComparison.OrdinalIgnoreCase)),
            M2Count = eligible.Count(r => r.AssetKind.Equals("m2", StringComparison.OrdinalIgnoreCase)),
        };
        return new Pm4ObjectPoolSelection(mapName, tileX, tileY, true,
            pools.Select(p => p.Id).Order(StringComparer.OrdinalIgnoreCase).ToArray(), database.Records.Count, eligible.Length, filtered);
    }

    public static void Validate(Pm4ObjectPoolManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        if (manifest.Version != 1) throw new ArgumentException("Unsupported object-pool manifest version.");
        ArgumentException.ThrowIfNullOrWhiteSpace(manifest.MapName);
        ArgumentNullException.ThrowIfNull(manifest.Pools);
        ArgumentNullException.ThrowIfNull(manifest.Assignments);
        var ids = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in manifest.Pools)
        {
            ArgumentNullException.ThrowIfNull(p);
            ArgumentException.ThrowIfNullOrWhiteSpace(p.Id);
            if (!ids.Add(p.Id)) throw new ArgumentException($"Duplicate pool '{p.Id}'.");
            ArgumentNullException.ThrowIfNull(p.AssetKinds);
            ArgumentNullException.ThrowIfNull(p.ObjectClasses);
            ArgumentNullException.ThrowIfNull(p.AssetPaths);
            ArgumentNullException.ThrowIfNull(p.PathPrefixes);
            if (p.AssetKinds.Count == 0 || p.AssetKinds.Any(k => !string.Equals(k, "m2", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(k, "wmo", StringComparison.OrdinalIgnoreCase)))
                throw new ArgumentException($"Pool '{p.Id}' must explicitly name m2 and/or wmo asset kinds.");
            if (p.AssetPaths.Count + p.PathPrefixes.Count == 0)
                throw new ArgumentException($"Pool '{p.Id}' needs at least one explicit asset path or path prefix.");
            foreach (string path in p.AssetPaths.Concat(p.PathPrefixes))
            {
                ArgumentException.ThrowIfNullOrWhiteSpace(path);
                string normalized = Normalize(path);
                if (normalized.Contains('*') || normalized.Contains('?') || normalized.Contains(':')
                    || normalized.StartsWith('/') || normalized.Split('/').Any(s => s is "." or ".."))
                    throw new ArgumentException("Pool selectors must be portable game paths, not globs or filesystem paths.");
            }
        }
        foreach (var a in manifest.Assignments)
        {
            ArgumentNullException.ThrowIfNull(a);
            if (!ids.Contains(a.PoolId)) throw new ArgumentException($"Assignment references unknown pool '{a.PoolId}'.");
            ValidateTile(a.MinTileX, a.MinTileY);
            ValidateTile(a.MaxTileX, a.MaxTileY);
            if (a.MinTileX > a.MaxTileX || a.MinTileY > a.MaxTileY)
                throw new ArgumentException("Tile ranges must be ordered min-to-max.");
        }
    }

    private static bool Allows(Pm4ObjectPool pool, SurfaceCorrelationFingerprint candidate)
    {
        if (!pool.AssetKinds.Contains(candidate.AssetKind, StringComparer.OrdinalIgnoreCase)) return false;
        string path = Normalize(candidate.AssetPath);
        return pool.AssetPaths.Any(p => string.Equals(Normalize(p), path, StringComparison.OrdinalIgnoreCase))
            || pool.PathPrefixes.Any(p => path.StartsWith(Normalize(p).TrimEnd('/') + "/", StringComparison.OrdinalIgnoreCase));
    }

    private static string Normalize(string path) => path.Replace('\\', '/');
    private static void ValidateTile(int x, int y)
    {
        if (x is < 0 or > 63 || y is < 0 or > 63)
            throw new ArgumentOutOfRangeException(nameof(x), "Map tile coordinates must be 0..63.");
    }
}
