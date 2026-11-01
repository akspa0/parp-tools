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
            {
                int singlePart = 0, multiPart = 0;
                if (objects != null)
                {
                    foreach (Pm4OverlayObject obj in objects)
                    {
                        if (obj.ObjectPartId > 0) multiPart++;
                        else singlePart++;
                    }
                }
                list.Add(new Pm4LegendEntry(new Vector3(0.92f, 0.45f, 0.25f), "Unique Instance / Cluster", singlePart, "Spatial hash ensuring adjacent doodads have contrasting hues"));
                list.Add(new Pm4LegendEntry(new Vector3(0.25f, 0.75f, 0.85f), "Multi-part Geometry", multiPart, "Distinct part IDs offset by golden ratio"));
                break;
            }

            case Pm4ColorPalette.SemanticCategory:
            {
                int count03 = 0, count10 = 0, count11 = 0, count13 = 0, count14 = 0;
                if (objects != null)
                {
                    foreach (Pm4OverlayObject obj in objects)
                    {
                        byte cls = obj.DominantGroupKey;
                        if (cls == 0x03) count03++;
                        else if (cls == 0x10) count10++;
                        else if (cls == 0x11 || cls == 0x12) count11++;
                        else if (cls == 0x13) count13++;
                        else if (cls >= 0x14) count14++;
                    }
                }
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorDoodad, "Doodad Mesh (0x03)", count03, "Level-designer doodad collision hulls (M2/MDX)"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorLowest, "Lowest Floor (0x10)", count10, "Ground level interior floor surfaces"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorCommon, "Standard Floor (0x11)", count11, "Navigable horizontal floor planes"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorUpper, "Upper Floor (0x13)", count13, "Multi-story elevated floor planes"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorFloorAngled, "Angled / Roof (0x14)", count14, "Ramps, stairs, and sloped surfaces"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorWall, "Walls (MSPV/MSPI)", 0, "Vertical boundary and obstruction walls"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorConnector, "Connectors (MSCN)", 0, "Cross-tile scene graph anchor points"));
                break;
            }

            case Pm4ColorPalette.ElevationRamp:
            {
                int b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
                if (objects != null)
                {
                    float range = MathF.Max(0.01f, maxZ - minZ);
                    foreach (Pm4OverlayObject obj in objects)
                    {
                        float norm = (obj.PlacementAnchor.Z - minZ) / range;
                        if (norm < 0.2f) b1++;
                        else if (norm < 0.4f) b2++;
                        else if (norm < 0.6f) b3++;
                        else if (norm < 0.8f) b4++;
                        else b5++;
                    }
                }
                list.Add(new Pm4LegendEntry(new Vector3(0.08f, 0.12f, 0.40f), $"Min Elevation ({minZ:F0} yd)", b1, "Deep valley / sea-level base"));
                list.Add(new Pm4LegendEntry(new Vector3(0.12f, 0.65f, 0.85f), $"Mid-Low ({minZ + (maxZ - minZ) * 0.25f:F0} yd)", b2, "Lower terrace / valley floor"));
                list.Add(new Pm4LegendEntry(new Vector3(0.25f, 0.85f, 0.38f), $"Mid-High ({minZ + (maxZ - minZ) * 0.50f:F0} yd)", b3, "Middle plateau / foundation"));
                list.Add(new Pm4LegendEntry(new Vector3(0.95f, 0.62f, 0.15f), $"High Tier ({minZ + (maxZ - minZ) * 0.75f:F0} yd)", b4, "Upper ridge / battlements"));
                list.Add(new Pm4LegendEntry(new Vector3(0.98f, 0.95f, 0.82f), $"Peak ({maxZ:F0} yd)", b5, "Highest mountain crest"));
                break;
            }

            case Pm4ColorPalette.MatchStatus:
            {
                int matched = 0, unmatched = 0, duplicated = 0, structural = 0;
                if (objects != null)
                {
                    foreach (Pm4OverlayObject obj in objects)
                    {
                        if (obj.Ck24Type == 0x03) unmatched++;
                        else structural++;
                    }
                }
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorMatched, "Matched ADT Placement", matched, "Correlated with existing M2/WMO in scene"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorUnmatched, "Unmatched Candidate", unmatched, "Authentic Blizzard hull lacking ADT model"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorDuplicated, "Live Duplicated Model", duplicated, "Newly magnetized and duplicated model"));
                list.Add(new Pm4LegendEntry(Pm4ColorCodingEngine.ColorIgnored, "Terrain / Structural", structural, "Terrain anchors and static WMO geometry"));
                break;
            }

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
