using WowViewer.Core.Maps;

namespace WowViewer.Core.Editor.Operations;

/// <summary>
/// The single production path for placement writes. This is the sole caller of
/// <c>AdtPlacementWriter</c>, retiring <c>ViewerApp</c>'s parallel translation-only staging
/// implementation (Spec 167 SC-003).
/// </summary>
public interface IPlacementWriteService
{
    /// <summary>Applies a placement move to in-memory ADT bytes and returns the updated bytes.</summary>
    byte[] ApplyMove(byte[] sourceBytes, string sourcePath, PlacementMoveOperation operation);

    /// <summary>Applies a placement move to a source ADT file and returns the updated bytes.</summary>
    byte[] ApplyMove(string sourcePath, PlacementMoveOperation operation);

    /// <summary>Applies a placement move and writes the updated ADT to a distinct output path.</summary>
    void WriteMove(string sourcePath, string outputPath, PlacementMoveOperation operation);
}