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

        var move = new AdtPlacementMoveEdit(operation.Kind, operation.EntryIndex, operation.UniqueId, operation.NewPosition);
        return AdtPlacementEditor.Apply(sourceBytes, sourcePath, [move]).Bytes;
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
