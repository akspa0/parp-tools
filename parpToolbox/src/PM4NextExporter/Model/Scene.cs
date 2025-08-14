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
        public List<Vector3> MscnVertices { get; init; } = new();
        // Optional: tile IDs (linearized) for each MSCN vertex, when region loads are used
        public List<int> MscnTileIds { get; init; } = new();

        // Links (MSLK) for per-instance assembly
        public List<MslkEntry> Links { get; init; } = new();

        // Raw MSPI chunks for diagnostics / advanced grouping
        public List<MspiChunk> Spis { get; init; } = new();

        // Tile offset maps (keyed by linear tile id: Y*64 + X)
        public Dictionary<int, int> TileVertexOffsetByTileId { get; init; } = new();
        public Dictionary<int, int> TileIndexOffsetByTileId { get; init; } = new();
        public Dictionary<int, int> TileVertexCountByTileId { get; init; } = new();
        public Dictionary<int, int> TileIndexCountByTileId { get; init; } = new();
        // Original tile coordinates as parsed from filenames
        public Dictionary<int, TileCoord> TileCoordByTileId { get; init; } = new();

        // Convenience counts
        public int VertexCount => Vertices?.Count ?? 0;
        public int SurfaceCount => Surfaces?.Count ?? 0;

        public static Scene Empty(string sourcePath) => new Scene
        {
            SourcePath = sourcePath,
            Vertices = new List<Vector3>(),
            Indices = new List<int>(),
            Surfaces = new List<MsurChunk.Entry>(),
            Spis = new List<MspiChunk>(),
            MscnVertices = new List<Vector3>(),
            MscnTileIds = new List<int>(),
            Links = new List<MslkEntry>(),
            TileVertexOffsetByTileId = new Dictionary<int, int>(),
            TileIndexOffsetByTileId = new Dictionary<int, int>(),
            TileVertexCountByTileId = new Dictionary<int, int>(),
            TileIndexCountByTileId = new Dictionary<int, int>(),
            TileCoordByTileId = new Dictionary<int, TileCoord>()
        };

        public static Scene FromPm4Scene(Pm4Scene pm4, string sourcePath) => new Scene
        {
            SourcePath = sourcePath,
            Vertices = pm4.Vertices ?? new List<Vector3>(),
            Indices = pm4.Indices ?? new List<int>(),
            Surfaces = pm4.Surfaces ?? new List<MsurChunk.Entry>(),
            Spis = pm4.Spis ?? new List<MspiChunk>(),
            MscnVertices = pm4.MscnVertices ?? new List<Vector3>(),
            MscnTileIds = pm4.MscnVertexTileIds ?? new List<int>(),
            Links = pm4.Links ?? new List<MslkEntry>(),
            TileVertexOffsetByTileId = pm4.TileVertexOffsetByTileId ?? new Dictionary<int, int>(),
            TileIndexOffsetByTileId = pm4.TileIndexOffsetByTileId ?? new Dictionary<int, int>(),
            TileVertexCountByTileId = pm4.TileVertexCountByTileId ?? new Dictionary<int, int>(),
            TileIndexCountByTileId = pm4.TileIndexCountByTileId ?? new Dictionary<int, int>(),
            TileCoordByTileId = pm4.TileCoordByTileId ?? new Dictionary<int, TileCoord>()
        };
    }
}
