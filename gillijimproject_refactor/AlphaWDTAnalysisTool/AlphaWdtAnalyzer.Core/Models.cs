using System;
using System.Collections.Generic;

namespace AlphaWdtAnalyzer.Core;

public enum AssetType
{
    Unknown,
    Wmo,
    MdxOrM2,
    Blp
}

public sealed record MapTile(int X, int Y, string AdtPath);

public sealed record PlacementRecord(
    AssetType Type,
    string AssetPath,
    string MapName,
    int TileX,
    int TileY,
    int? UniqueId, // optional until we decode Alpha layout reliably
    float WorldX,
    float WorldY,
    float WorldZ
);

public sealed class AnalysisIndex
{
    public string MapName { get; set; } = string.Empty;
    public List<MapTile> Tiles { get; set; } = new();

    public List<string> WmoAssets { get; set; } = new();
    public List<string> M2Assets { get; set; } = new();
    public List<string> BlpAssets { get; set; } = new();

    public List<PlacementRecord> Placements { get; set; } = new();

    public List<string> MissingWmo { get; set; } = new();
    public List<string> MissingM2 { get; set; } = new();
    public List<string> MissingBlp { get; set; } = new();
}
