namespace PM4NextExporter.Model
{
    using System.Collections.Generic;
    using System.Numerics;
    using ParpToolbox.Formats.PM4;
    using ParpToolbox.Formats.P4.Chunks.Common;

    public sealed class Scene
    {
        public string SourcePath { get; init; } = string.Empty;

        // Geometry
        public List<Vector3> Vertices { get; init; } = new();
        public List<int> Indices { get; init; } = new();
        public List<MsurChunk.Entry> Surfaces { get; init; } = new();

        // Convenience counts
        public int VertexCount => Vertices?.Count ?? 0;
        public int SurfaceCount => Surfaces?.Count ?? 0;

        public static Scene Empty(string sourcePath) => new Scene
        {
            SourcePath = sourcePath,
            Vertices = new List<Vector3>(),
            Indices = new List<int>(),
            Surfaces = new List<MsurChunk.Entry>()
        };

        public static Scene FromPm4Scene(Pm4Scene pm4, string sourcePath) => new Scene
        {
            SourcePath = sourcePath,
            Vertices = pm4.Vertices ?? new List<Vector3>(),
            Indices = pm4.Indices ?? new List<int>(),
            Surfaces = pm4.Surfaces ?? new List<MsurChunk.Entry>()
        };
    }
}
