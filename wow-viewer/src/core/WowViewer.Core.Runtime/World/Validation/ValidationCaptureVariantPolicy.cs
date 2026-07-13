namespace WowViewer.Core.Runtime.World.Validation;

public readonly record struct ValidationCaptureVariantPolicy(
    bool ShowTerrain,
    bool ShowTerrainLiquids,
    bool ShowObjects,
    bool ShowWmos,
    bool ShowDoodads,
    bool ShowSky,
    bool ShowWdl,
    bool ShowWorldLiquids,
    bool TerrainShadeOnly = false);
