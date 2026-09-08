using System.Text.Json;
using System.Text.Json.Serialization;

namespace WowViewer.Core.Maps;

/// <summary>
/// Spec 232 Phase 2 (FR-2/FR-8): the persisted layer-stack project for one base map. Human-editable
/// JSON, stored per base map under the viewer's project output area; reloaded automatically on
/// launch / map load. Locked layers render a locked badge and reject accidental edits.
/// </summary>
public sealed class PhaseLayerProjectFile
{
    private static readonly JsonSerializerOptions SerializerOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingDefault,
    };

    public int Version { get; set; } = 1;

    /// <summary>The base map this stack overlays (sanity check on load).</summary>
    public string BaseMap { get; set; } = string.Empty;

    public List<PhaseLayerProjectEntry> Layers { get; set; } = new();

    /// <summary>Spec 232 FR-11: the channels the base map contributes to its own tiles.</summary>
    public PhaseDataChannel BaseChannelKeep { get; set; } = PhaseDataChannel.All;

    public static PhaseLayerProjectFile FromLayers(
        string baseMap, IList<PhaseLayerSettings> layers, PhaseDataChannel baseChannelKeep)
    {
        var file = new PhaseLayerProjectFile { BaseMap = baseMap, BaseChannelKeep = baseChannelKeep };
        foreach (PhaseLayerSettings layer in layers)
            file.Layers.Add(PhaseLayerProjectEntry.FromLayer(layer));
        return file;
    }

    public List<PhaseLayerSettings> ToLayers()
    {
        var layers = new List<PhaseLayerSettings>(Layers.Count);
        foreach (PhaseLayerProjectEntry entry in Layers)
            layers.Add(entry.ToLayer());
        return layers;
    }

    public static PhaseLayerProjectFile Load(string path)
        => JsonSerializer.Deserialize<PhaseLayerProjectFile>(File.ReadAllText(path), SerializerOptions)
            ?? new PhaseLayerProjectFile();

    public void Save(string path)
    {
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(path, JsonSerializer.Serialize(this, SerializerOptions));
    }
}

/// <summary>One persisted layer with all transform, channel, and lock state.</summary>
public sealed class PhaseLayerProjectEntry
{
    public string MapName { get; set; } = string.Empty;
    public bool Enabled { get; set; } = true;
    public PhaseDataChannel Channels { get; set; } = PhaseDataChannel.All;
    public bool OnlyTakeWhatThePhaseCarries { get; set; } = true;
    public int TileOffsetX { get; set; }
    public int TileOffsetY { get; set; }
    public int CellOffsetX { get; set; }
    public int CellOffsetY { get; set; }
    public float RotationDegrees { get; set; }
    public float RotationOriginTileX { get; set; }
    public float RotationOriginTileY { get; set; }
    public bool MirrorHorizontal { get; set; }
    public bool MirrorVertical { get; set; }
    public int FootprintColorIndex { get; set; }

    /// <summary>Spec 232 FR-2: locked layers reject accidental edits until explicitly unlocked.</summary>
    public bool Locked { get; set; }

    public List<PhaseTilePlacementDto> TilePlacements { get; set; } = new();

    public static PhaseLayerProjectEntry FromLayer(PhaseLayerSettings layer) => new()
    {
        MapName = layer.MapName,
        Enabled = layer.Enabled,
        Channels = layer.Channels,
        OnlyTakeWhatThePhaseCarries = layer.OnlyTakeWhatThePhaseCarries,
        TileOffsetX = layer.TileOffsetX,
        TileOffsetY = layer.TileOffsetY,
        CellOffsetX = layer.CellOffsetX,
        CellOffsetY = layer.CellOffsetY,
        RotationDegrees = layer.RotationDegrees,
        RotationOriginTileX = layer.RotationOriginTileX,
        RotationOriginTileY = layer.RotationOriginTileY,
        MirrorHorizontal = layer.MirrorHorizontal,
        MirrorVertical = layer.MirrorVertical,
        FootprintColorIndex = layer.FootprintColorIndex,
        Locked = layer.Locked,
        TilePlacements = layer.TilePlacements.Select(static placement => new PhaseTilePlacementDto
        {
            DonorTileX = placement.DonorTileX,
            DonorTileY = placement.DonorTileY,
            TargetTileX = placement.TargetTileX,
            TargetTileY = placement.TargetTileY,
        }).ToList(),
    };

    public PhaseLayerSettings ToLayer()
    {
        var layer = new PhaseLayerSettings
        {
            MapName = MapName,
            Enabled = Enabled,
            Channels = Channels,
            OnlyTakeWhatThePhaseCarries = OnlyTakeWhatThePhaseCarries,
            TileOffsetX = TileOffsetX,
            TileOffsetY = TileOffsetY,
            CellOffsetX = CellOffsetX,
            CellOffsetY = CellOffsetY,
            RotationDegrees = RotationDegrees,
            RotationOriginTileX = RotationOriginTileX,
            RotationOriginTileY = RotationOriginTileY,
            MirrorHorizontal = MirrorHorizontal,
            MirrorVertical = MirrorVertical,
            FootprintColorIndex = FootprintColorIndex,
            Locked = Locked,
        };
        foreach (PhaseTilePlacementDto placement in TilePlacements)
            layer.TilePlacements.Add(new PhaseTilePlacement(
                placement.DonorTileX, placement.DonorTileY, placement.TargetTileX, placement.TargetTileY));
        return layer;
    }
}

public sealed class PhaseTilePlacementDto
{
    public int DonorTileX { get; set; }
    public int DonorTileY { get; set; }
    public int TargetTileX { get; set; }
    public int TargetTileY { get; set; }
}
