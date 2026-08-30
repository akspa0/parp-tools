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
    public string BaseGroundTexture { get; init; } = @"tileset\elwynn\elwynngrass.blp";
    public string PathTexture { get; init; } = @"tileset\city\stormwindcobble.blp";
    public string PlazaFloorTexture { get; init; } = @"tileset\city\whitemarble.blp";
    public string AccentTexture { get; init; } = @"tileset\generic\rock.blp";

    public static BiomePalette ForTheme(BiomeTheme theme) => theme switch
    {
        BiomeTheme.GardenMuseum => new BiomePalette
        {
            BaseGroundTexture = @"tileset\elwynn\elwynngrass.blp",
            PathTexture = @"tileset\city\stormwindcobble.blp",
            PlazaFloorTexture = @"tileset\city\whitemarble.blp",
            AccentTexture = @"tileset\generic\dirt.blp"
        },
        BiomeTheme.ElwynnForest => new BiomePalette
        {
            BaseGroundTexture = @"tileset\elwynn\elwynngrass.blp",
            PathTexture = @"tileset\generic\dirt.blp",
            PlazaFloorTexture = @"tileset\city\stormwindcobble.blp",
            AccentTexture = @"tileset\generic\rock.blp"
        },
        BiomeTheme.CobblestoneCity => new BiomePalette
        {
            BaseGroundTexture = @"tileset\city\stormwindcobble.blp",
            PathTexture = @"tileset\city\whitemarble.blp",
            PlazaFloorTexture = @"tileset\city\whitemarble.blp",
            AccentTexture = @"tileset\elwynn\elwynngrass.blp"
        },
        BiomeTheme.DunMorogh => new BiomePalette
        {
            BaseGroundTexture = @"tileset\dunmorogh\dunmoroghsnow.blp",
            PathTexture = @"tileset\generic\dirt.blp",
            PlazaFloorTexture = @"tileset\city\whitemarble.blp",
            AccentTexture = @"tileset\generic\rock.blp"
        },
        BiomeTheme.Barrens => new BiomePalette
        {
            BaseGroundTexture = @"tileset\barrens\barrensdirt.blp",
            PathTexture = @"tileset\generic\dirt.blp",
            PlazaFloorTexture = @"tileset\city\whitemarble.blp",
            AccentTexture = @"tileset\generic\rock.blp"
        },
        BiomeTheme.Ashenvale => new BiomePalette
        {
            BaseGroundTexture = @"tileset\ashenvale\ashenvalegrass.blp",
            PathTexture = @"tileset\generic\dirt.blp",
            PlazaFloorTexture = @"tileset\city\whitemarble.blp",
            AccentTexture = @"tileset\generic\rock.blp"
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
