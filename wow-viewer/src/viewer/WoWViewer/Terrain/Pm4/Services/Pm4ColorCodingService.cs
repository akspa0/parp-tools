using System;
using System.Collections.Generic;
using System.Numerics;
using WowViewer.Core.PM4.Services;

namespace WoWViewer.Terrain;

/// <summary>
/// Defines color palette modes for the PM4 collision overlay (Spec 288).
/// </summary>
public enum Pm4ColorPalette
{
    InstanceCluster = 0,
    SemanticCategory = 1,
    ElevationRamp = 2,
    MatchStatus = 3,
    PlacementZ = 4,
    Tile = 5,
    MshdRegion = 6,
    SurfaceCount = 7,
}

/// <summary>
/// Telemetry entry for the interactive sidebar color legend.
/// </summary>
public readonly record struct Pm4LegendEntry(Vector3 Color, string Label, int Count, string Description);

/// <summary>
/// Centralized color coding, palette evaluation, and legend service for PM4 geometry in the viewer (Spec 288).
/// Conforms to AGENTS.md §10 god-class decomposition by delegating pure algorithms to <see cref="Pm4ColorCodingEngine"/>.
/// </summary>
internal sealed class Pm4ColorCodingService
{
    public static readonly Pm4ColorCodingService Instance = new();

    private readonly Pm4ColorCodingEngine _engine = Pm4ColorCodingEngine.Instance;

    public Vector3 GetObjectColor(
        Pm4OverlayObject obj,
        (int tileX, int tileY) tileKey,
        Pm4ColorPalette palette,
        float minZ,
        float maxZ,
        bool isMatched = false,
        bool isDuplicated = false)
    {
        return palette switch
        {
            Pm4ColorPalette.InstanceCluster  => GetInstanceColor(obj, tileKey),
            Pm4ColorPalette.SemanticCategory => GetSemanticCategoryColor(obj.Ck24Type, obj.DominantGroupKey),
            Pm4ColorPalette.ElevationRamp    => GetElevationColor(obj.PlacementAnchor.Z + obj.Center.Z, minZ, maxZ),
            Pm4ColorPalette.MatchStatus      => GetMatchStatusColor(isMatched, isDuplicated, obj.Ck24Type),
            Pm4ColorPalette.PlacementZ       => Pm4ColorCodingEngine.ColorFromSeed(obj.Ck24),
            Pm4ColorPalette.Tile             => Pm4ColorCodingEngine.ColorFromSeed((uint)HashCode.Combine(tileKey.tileX, tileKey.tileY)),
            Pm4ColorPalette.MshdRegion       => Pm4ColorCodingEngine.ColorFromSeed(obj.MshdRegionId + 1u),
            Pm4ColorPalette.SurfaceCount     => Pm4ColorCodingEngine.ColorFromSeed((uint)Math.Max(1, obj.SurfaceCount) * 17u),
            _                                => GetInstanceColor(obj, tileKey)
        };
    }

    public Vector3 GetInstanceColor(Pm4OverlayObject obj, (int tileX, int tileY) tileKey)
    {
        return _engine.ComputeInstanceColor(
            obj.Ck24,
            obj.ObjectPartId,
            tileKey.tileX,
            tileKey.tileY,
            obj.PlacementAnchor,
            obj.SurfaceCount);
    }

    public Vector3 GetSemanticCategoryColor(byte ck24Type, byte dominantGroupKey)
    {
        return _engine.ComputeSemanticCategoryColor(ck24Type, dominantGroupKey);
    }

    public Vector3 GetElevationColor(float worldZ, float minZ, float maxZ)
    {
        return _engine.ComputeElevationColor(worldZ, minZ, maxZ);
    }

    public Vector3 GetMatchStatusColor(bool isMatched, bool isDuplicated, byte ck24Type)
    {
        return _engine.ComputeMatchStatusColor(isMatched, isDuplicated, ck24Type);
    }

    public IReadOnlyList<Pm4LegendEntry> GetLegendEntries(
        Pm4ColorPalette palette,
        IEnumerable<Pm4OverlayObject>? objects = null,
        float minZ = 0f,
        float maxZ = 100f)
    {
        var list = new List<Pm4LegendEntry>();
        switch (palette)
        {
            case Pm4ColorPalette.InstanceCluster:
                list.Add(new Pm4LegendEntry(new Vector3(0.92f, 0.45f, 0.25f), "Unique Instance / Cluster", 0, "Spatial hash ensuring adjacent doodads have contrasting hues"));
                list.Add(new Pm4LegendEntry(new Vector3(0.25f, 0.75f, 0.85f), "Multi-part Geometry", 0, "Distinct part IDs offset by golden ratio"));
                break;

            case Pm4ColorPalette.SemanticCategory:
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorDoodad, "Doodad Mesh (0x03)", 0, "Level-designer doodad collision hulls (M2/MDX)"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorLowest, "Lowest Floor (0x10)", 0, "Ground level interior floor surfaces"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorCommon, "Standard Floor (0x11)", 0, "Navigable horizontal floor planes"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorUpper, "Upper Floor (0x13)", 0, "Multi-story elevated floor planes"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorAngled, "Angled / Roof (0x14)", 0, "Ramps, stairs, and sloped surfaces"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorWall, "Walls (MSPV/MSPI)", 0, "Vertical boundary and obstruction walls"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorConnector, "Connectors (MSCN)", 0, "Cross-tile scene graph anchor points"));
                break;

            case Pm4ColorPalette.ElevationRamp:
                list.Add(new Pm4LegendEntry(new Vector3(0.08f, 0.12f, 0.40f), $"Min Elevation ({minZ:F0} yd)", 0, "Deep valley / sea-level base"));
                list.Add(new Pm4LegendEntry(new Vector3(0.12f, 0.65f, 0.85f), $"Mid-Low ({minZ + (maxZ - minZ) * 0.25f:F0} yd)", 0, "Lower terrace / valley floor"));
                list.Add(new Pm4LegendEntry(new Vector3(0.25f, 0.85f, 0.38f), $"Mid-High ({minZ + (maxZ - minZ) * 0.50f:F0} yd)", 0, "Middle plateau / foundation"));
                list.Add(new Pm4LegendEntry(new Vector3(0.95f, 0.62f, 0.15f), $"High Tier ({minZ + (maxZ - minZ) * 0.75f:F0} yd)", 0, "Upper ridge / battlements"));
                list.Add(new Pm4LegendEntry(new Vector3(0.98f, 0.95f, 0.82f), $"Peak ({maxZ:F0} yd)", 0, "Highest mountain crest"));
                break;

            case Pm4ColorPalette.MatchStatus:
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorMatched, "Matched ADT Placement", 0, "Correlated with existing M2/WMO in scene"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorUnmatched, "Unmatched Candidate", 0, "Authentic Blizzard hull lacking ADT model"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorDuplicated, "Live Duplicated Model", 0, "Newly magnetized and duplicated model"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorIgnored, "Terrain / Structural", 0, "Terrain anchors and static WMO geometry"));
                break;

            case Pm4ColorPalette.PlacementZ:
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFromSeed(0), "Doodads (Z = 0.0)", 0, "Collision hulls lacking authored placement Z"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFromSeed(0x420000), "WMO Group A", 0, "WMO placement anchor group"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFromSeed(0x430000), "WMO Group B", 0, "WMO placement anchor group"));
                break;

            case Pm4ColorPalette.Tile:
                list.Add(new Pm4LegendEntry(new Vector3(0.4f, 0.7f, 0.9f), "Per-Tile Color", 0, "Unique color hash for each 533.33yd tile"));
                break;

            case Pm4ColorPalette.MshdRegion:
                list.Add(new Pm4LegendEntry(new Vector3(0.8f, 0.4f, 0.6f), "MSHD Partition Region", 0, "Spatial partition domain"));
                break;

            case Pm4ColorPalette.SurfaceCount:
                list.Add(new Pm4LegendEntry(new Vector3(0.3f, 0.8f, 0.5f), "Surface Complexity", 0, "Polygon density bucket"));
                break;
        }

        return list;
    }
}
