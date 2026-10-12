namespace WowViewer.Core.IO.Terrain;

public enum BiomeTheme
{
    GardenMuseum,
    ElwynnForest,
    CobblestoneCity,
    DunMorogh,
    Barrens,
    Ashenvale
}

public enum RoadNetworkTopology
{
    GridPromenade,
    SpokeWheel,
    OrganicTrails,
    CourtyardAvenue
}

/// <summary>
/// Defines the 4-layer texture palette for a specific biome theme.
/// </summary>
public sealed class BiomePalette
{
    public string BaseGroundTexture { get; init; } = @"TILESET\ELWYNN\ElwynnGrassBase.blp";
    public string PathTexture { get; init; } = @"TILESET\StormwindCity\SW_Cobble_A.blp";
    public string PlazaFloorTexture { get; init; } = @"TILESET\ELWYNN\ElwynnCobbleStoneBase.blp";
    public string AccentTexture { get; init; } = @"TILESET\ELWYNN\ELWYNNDIRTBASE.BLP";

    public static BiomePalette ForTheme(BiomeTheme theme) => theme switch
    {
        BiomeTheme.GardenMuseum => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\ELWYNN\ElwynnGrassBase.blp",
            PathTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            PlazaFloorTexture = @"TILESET\ELWYNN\ElwynnCobbleStoneBase.blp",
            AccentTexture = @"TILESET\ELWYNN\ELWYNNDIRTBASE.BLP"
        },
        BiomeTheme.ElwynnForest => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\ELWYNN\ElwynnGrassBase.blp",
            PathTexture = @"TILESET\ELWYNN\ELWYNNDIRTBASE.BLP",
            PlazaFloorTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            AccentTexture = @"TILESET\ELWYNN\ElwynnCobbleStoneBase.blp"
        },
        BiomeTheme.CobblestoneCity => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            PathTexture = @"TILESET\StormwindCity\SWC_DirtA.blp",
            PlazaFloorTexture = @"TILESET\StormwindCity\SW_Cobble_LeavesA.blp",
            AccentTexture = @"TILESET\StormwindCity\SWC_GrassMidA.blp"
        },
        BiomeTheme.DunMorogh => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\EXPANSION02\DRAGONBLIGHT\DragonBlightFreshSmoothSnowA.blp",
            PathTexture = @"TILESET\ALTERACMTNS\AlteracDirtBase.blp",
            PlazaFloorTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            AccentTexture = @"TILESET\ALTERACMTNS\AlteracGrassBase.blp"
        },
        BiomeTheme.Barrens => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\Barrens\BarrensBaseDirt.blp",
            PathTexture = @"TILESET\Barrens\BarrensBaseDirt02.blp",
            PlazaFloorTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            AccentTexture = @"TILESET\Barrens\BarrensBaseGrass.blp"
        },
        BiomeTheme.Ashenvale => new BiomePalette
        {
            BaseGroundTexture = @"TILESET\ASHENVALE\AshenvaleGrass.blp",
            PathTexture = @"TILESET\ASHENVALE\AshenvaleDirt.blp",
            PlazaFloorTexture = @"TILESET\StormwindCity\SW_Cobble_A.blp",
            AccentTexture = @"TILESET\ASHENVALE\AshenvaleFerns.blp"
        },
        _ => new BiomePalette()
    };
}

/// <summary>
/// High-level generation specification for synthesizing multi-tile maps from terrain templates and brush motifs.
/// </summary>
public sealed class TerrainMapTemplate
{
    public string MapName { get; init; } = "TemplatedGarden";
    public BiomeTheme Theme { get; init; } = BiomeTheme.GardenMuseum;
    public RoadNetworkTopology Topology { get; init; } = RoadNetworkTopology.GridPromenade;
    public int TileRows { get; init; } = 2;
    public int TileCols { get; init; } = 2;
    public int BaseTileX { get; init; } = 30;
    public int BaseTileY { get; init; } = 30;
    public int PlazaSpacingChunks { get; init; } = 2; // Every 2 chunks (66.6m)
    public bool EnablePerimeterRidges { get; init; } = false;
    public float MaxSlopeLimitDegrees { get; init; } = 25.0f;
    public BiomePalette Palette { get; init; } = BiomePalette.ForTheme(BiomeTheme.GardenMuseum);
}
