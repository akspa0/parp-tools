namespace WoWViewer.Terrain;

public readonly record struct WorldAssetReadStats(
    long ReadRequests,
    long FileCacheHits,
    int FileCacheCount,
    long FileCacheBytes,
    long ResolvedPathCacheHits,
    long PathProbeAttempts,
    long PathProbeResolutions,
    long PathProbeMisses,
    int ResolvedPathCacheCount);
