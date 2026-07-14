namespace WowViewer.Core.Maps;

/// <summary>
/// One named array inside an enrichment stream entry.
/// </summary>
/// <param name="Name">Array field name (e.g. "vertices", "triangles", "normals", "texture_rgb").</param>
/// <param name="Shape">Shape of the array in row-major order.</param>
/// <param name="DataType">CLR element type (typeof(float), typeof(int), typeof(uint), typeof(byte), typeof(bool), typeof(ushort)).</param>
/// <param name="Data">Flattened row-major bytes.</param>
public sealed record EnrichmentArray(
    string Name,
    int[] Shape,
    Type DataType,
    byte[] Data)
{
    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Length;
}