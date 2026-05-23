namespace WowViewer.Core.Runtime.World.Validation;

public sealed class ValidationCaptureBatchPlan
{
    public ValidationCaptureBatchPlan(
        string datasetRoot,
        string mapName,
        string primaryOutputDirectory,
        string noLiquidsOutputDirectory,
        string noObjectsOutputDirectory,
        string objectsOnlyOutputDirectory,
        int requestedResolution,
        string? buildLabel,
        IReadOnlyList<ValidationCaptureTileRequest> tileRequests)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(datasetRoot);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapName);
        ArgumentException.ThrowIfNullOrWhiteSpace(primaryOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(noLiquidsOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(noObjectsOutputDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(objectsOnlyOutputDirectory);
        ArgumentOutOfRangeException.ThrowIfLessThan(requestedResolution, 1);
        ArgumentNullException.ThrowIfNull(tileRequests);
        if (tileRequests.Any(static request => request is null))
            throw new ArgumentException("Tile requests cannot contain null entries.", nameof(tileRequests));

        DatasetRoot = datasetRoot;
        MapName = mapName;
        PrimaryOutputDirectory = primaryOutputDirectory;
        NoLiquidsOutputDirectory = noLiquidsOutputDirectory;
        NoObjectsOutputDirectory = noObjectsOutputDirectory;
        ObjectsOnlyOutputDirectory = objectsOnlyOutputDirectory;
        RequestedResolution = requestedResolution;
        BuildLabel = buildLabel;
        TileRequests = tileRequests;
    }

    public string DatasetRoot { get; }

    public string MapName { get; }

    public string PrimaryOutputDirectory { get; }

    public string NoLiquidsOutputDirectory { get; }

    public string NoObjectsOutputDirectory { get; }

    public string ObjectsOnlyOutputDirectory { get; }

    public int RequestedResolution { get; }

    public string? BuildLabel { get; }

    public IReadOnlyList<ValidationCaptureTileRequest> TileRequests { get; }

    public int RequestCount => TileRequests.Count;
}