using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <inheritdoc cref="IPlacementWriteService"/>
public sealed class PlacementWriteService : IPlacementWriteService
{
    public byte[] ApplyMove(byte[] sourceBytes, string sourcePath, PlacementMoveOperation operation)
    {
        ArgumentNullException.ThrowIfNull(sourceBytes);
        ArgumentNullException.ThrowIfNull(operation);

        var reference = new AdtPlacementReference(operation.Kind, operation.EntryIndex, operation.UniqueId);
        var move = new AdtPlacementMove(reference, operation.NewPosition);
        var transaction = new AdtPlacementEditTransaction(sourcePath, [move]);

        return AdtPlacementWriter.ApplyTransaction(sourceBytes, sourcePath, transaction);
    }

    public byte[] ApplyMove(string sourcePath, PlacementMoveOperation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);

        byte[] sourceBytes = File.ReadAllBytes(sourcePath);
        return ApplyMove(sourceBytes, sourcePath, operation);
    }

    public void WriteMove(string sourcePath, string outputPath, PlacementMoveOperation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);

        byte[] updatedBytes = ApplyMove(sourcePath, operation);

        string? outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(outputDirectory))
            Directory.CreateDirectory(outputDirectory);

        File.WriteAllBytes(outputPath, updatedBytes);
    }
}