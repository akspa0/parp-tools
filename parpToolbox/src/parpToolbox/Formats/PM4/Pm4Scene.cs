namespace ParpToolbox.Formats.PM4
{
    /// <summary>
    /// Immutable high-level representation of a PM4 file after parsing and adaption.
    /// Will be populated by <see cref="ParpToolbox.Services.PM4.Pm4Adapter"/>.
    /// </summary>
    public sealed record Pm4Scene
{
    public IReadOnlyList<System.Numerics.Vector3> Vertices { get; init; } = Array.Empty<System.Numerics.Vector3>();
    public IReadOnlyList<(int A,int B,int C)> Triangles { get; init; } = Array.Empty<(int,int,int)>();
}

}
