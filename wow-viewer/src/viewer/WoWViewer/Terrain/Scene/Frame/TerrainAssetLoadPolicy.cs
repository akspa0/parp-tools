namespace WoWViewer.Terrain;

// Moved from WorldScene (Spec 255 W0): formerly a private nested record; body unchanged.
internal readonly record struct TerrainAssetLoadPolicy(
    bool PrewarmTileAssets,
    int MaxNewMdxLoadsPerFrame,
    int MaxNewWmoLoadsPerFrame,
    int MaxDeferredLoadsPerFrame,
    double MaxDeferredLoadBudgetMs,
    int MaxPriorityLoadBacklog);
