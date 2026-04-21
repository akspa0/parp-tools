namespace WowViewer.Core.Datasets;

public enum TerrainTrainingSampleSourceKind
{
    Unknown = 0,
    ClientRoot = 1,
    StagedClientRoot = 2,
    MountedArchive = 3,
    LooseOverlayComposite = 4,
    HarvestedDatasetCompatibility = 5,
}

public sealed class TerrainTrainingSignalAvailability
{
    public bool HasRootAdt { get; init; }

    public bool HasObjAdt { get; init; }

    public bool HasTexAdt { get; init; }

    public bool HasWdl { get; init; }

    public bool HasMinimap { get; init; }

    public bool HasTerrainOnlyMinimap { get; init; }

    public bool HasNoLiquidMinimap { get; init; }

    public bool HasNoObjectMinimap { get; init; }

    public bool HasNoMccvMinimap { get; init; }

    public bool HasNormalMap { get; init; }

    public bool HasLiquidMask { get; init; }

    public bool HasLiquidHeight { get; init; }

    public bool HasObjectMask { get; init; }

    public bool HasBrushMask { get; init; }

    public bool HasPm4Mask { get; init; }

    public bool HasHoleMask { get; init; }

    public bool HasAreaIdMap { get; init; }

    public bool HasChunkFlagsMap { get; init; }

    public bool HasAlphaLayers { get; init; }

    public bool HasTextureMetadata { get; init; }
}

public sealed class TerrainTrainingSampleMetrics
{
    public float HeightMin { get; init; }

    public float HeightMax { get; init; }

    public float HeightRange { get; init; }

    public float LiquidCoverage { get; init; }

    public float ObjectCoverage { get; init; }

    public float BrushCoverage { get; init; }

    public float Pm4Coverage { get; init; }

    public float HoleCoverage { get; init; }

    public float MinimapVariance { get; init; }

    public float MinimapGradient { get; init; }

    public float MeanWdlDelta { get; init; }

    public float MaxAbsWdlDelta { get; init; }

    public int TextureLayerCount { get; init; }
}

public sealed class TerrainTrainingSampleDescriptor
{
    public TerrainTrainingSampleDescriptor(
        string sampleId,
        TerrainTrainingSampleSourceKind sourceKind,
        string buildLabel,
        string mapName,
        int tileX,
        int tileY,
        string sourceRoot,
        string rootAdtPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sampleId);
        ArgumentException.ThrowIfNullOrWhiteSpace(buildLabel);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(rootAdtPath);

        SampleId = sampleId;
        SourceKind = sourceKind;
        BuildLabel = buildLabel;
        MapName = mapName;
        TileX = tileX;
        TileY = tileY;
        SourceRoot = sourceRoot;
        RootAdtPath = rootAdtPath;
    }

    public string SampleId { get; }

    public TerrainTrainingSampleSourceKind SourceKind { get; }

    public string BuildLabel { get; }

    public string MapName { get; }

    public int TileX { get; }

    public int TileY { get; }

    public string SourceRoot { get; }

    public string RootAdtPath { get; }

    public string? ObjAdtPath { get; init; }

    public string? TexAdtPath { get; init; }

    public string? LodAdtPath { get; init; }

    public string? WdlPath { get; init; }

    public string? MapDirectory { get; init; }

    public string? LooseOverlayRoot { get; init; }

    public string? CompatibilityTileJsonPath { get; init; }

    public TerrainTrainingSignalAvailability Signals { get; init; } = new();

    public TerrainTrainingSampleMetrics Metrics { get; init; } = new();

    public string TileName => $"{MapName}_{TileX}_{TileY}";
}

public sealed class TerrainTrainingSampleManifest
{
    public TerrainTrainingSampleManifest(
        string schemaVersion,
        DateTimeOffset createdAtUtc,
        string sourceManifestKind,
        IReadOnlyList<TerrainTrainingSampleDescriptor> entries)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(schemaVersion);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourceManifestKind);
        ArgumentNullException.ThrowIfNull(entries);

        SchemaVersion = schemaVersion;
        CreatedAtUtc = createdAtUtc;
        SourceManifestKind = sourceManifestKind;
        Entries = entries;
    }

    public string SchemaVersion { get; }

    public DateTimeOffset CreatedAtUtc { get; }

    public string SourceManifestKind { get; }

    public IReadOnlyList<TerrainTrainingSampleDescriptor> Entries { get; }
}