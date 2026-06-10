namespace AlphaWDTReader.Model;

public sealed class AlphaScanResult
{
    public required string FilePath { get; init; }
    public int ChunkCount { get; set; }
    public bool HasMver { get; set; }
    public bool HasMphd { get; set; }
    public bool HasMain { get; set; }
    public int? MainDeclaredTiles { get; set; }
    public int DoodadNameCount { get; set; }
    public int WmoNameCount { get; set; }
    public List<string> FirstDoodadNames { get; } = new();
    public List<string> FirstWmoNames { get; } = new();
}
