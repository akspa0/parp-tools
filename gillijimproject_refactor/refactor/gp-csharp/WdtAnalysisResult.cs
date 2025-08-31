using System.Text.Json.Serialization;

namespace GillijimProject;

public class WdtAnalysisResult
{
    public string InputFile { get; set; } = "";
    public DateTime AnalysisTime { get; set; }
    public string Version { get; set; } = "1.0";
    public WdtSummary Summary { get; set; } = new();
    public List<TileInfo> Tiles { get; set; } = new();
    public List<ChunkTypeCount> ChunkTypes { get; set; } = new();
}

public class WdtSummary
{
    public int TotalMainEntries { get; set; }
    public int ValidTiles { get; set; }
    public int EmptyTiles { get; set; }
    public long FileSizeBytes { get; set; }
    public List<int> ValidTileIndices { get; set; } = new();
}

public class TileInfo
{
    public int TileIndex { get; set; }
    public long MhdrOffset { get; set; }
    public int McinEntries { get; set; }
    public long TileStart { get; set; }
    public List<McnkInfo> Mcnks { get; set; } = new();
}

public class McnkInfo
{
    public int McnkIndex { get; set; }
    public long AbsoluteOffset { get; set; }
    public long PayloadStart { get; set; }
    public uint ChunksSize { get; set; }
    public bool McvtOk { get; set; }
    public bool McnrOk { get; set; }
    public bool BoundsOk { get; set; }
    public uint McvtRel { get; set; }
    public uint McnrRel { get; set; }
    public uint MclqRel { get; set; }
    public HeightData? Heights { get; set; }
}

public class HeightData
{
    public float MinHeight { get; set; }
    public float MaxHeight { get; set; }
    public float[] Values { get; set; } = Array.Empty<float>();
}

public class ChunkTypeCount
{
    public string ChunkType { get; set; } = "";
    public int Count { get; set; }
}
