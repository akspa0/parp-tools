using System;
using System.Collections.Generic;
using System.Numerics;

namespace WowViewer.Core.PM4.Services;

/// <summary>
/// Color palette modes for PM4 collision overlay (Spec 288).
/// </summary>
public enum Pm4CoreColorPalette
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
/// Telemetry entry for color legends.
/// </summary>
public readonly record struct Pm4CoreLegendEntry(Vector3 Color, string Label, int Count, string Description);

/// <summary>
/// Pure algorithmic color coding, palette evaluation, and legend service for PM4 navmeshes (Spec 288).
/// </summary>
public sealed class Pm4ColorCodingEngine
{
    public static readonly Pm4ColorCodingEngine Instance = new();

    public static readonly Vector3 ColorDoodad = new(0.96f, 0.48f, 0.22f);        // Coral/tangerine (0x03)
    public static readonly Vector3 ColorFloorLowest = new(0.20f, 0.82f, 0.40f);    // Forest green (0x10)
    public static readonly Vector3 ColorFloorCommon = new(0.15f, 0.75f, 0.90f);    // Cyan (0x11)
    public static readonly Vector3 ColorFloorUpper = new(0.30f, 0.52f, 0.95f);     // Royal blue (0x13)
    public static readonly Vector3 ColorFloorAngled = new(0.72f, 0.35f, 0.90f);    // Purple/violet (0x14)
    public static readonly Vector3 ColorWall = new(0.95f, 0.78f, 0.18f);           // Gold/amber (MSPV/MSPI)
    public static readonly Vector3 ColorConnector = new(0.10f, 0.95f, 0.85f);      // Turquoise (MSCN)
    public static readonly Vector3 ColorTerrainLinker = new(0.85f, 0.85f, 0.30f);   // Lime (MPRL)
    public static readonly Vector3 ColorGeneric = new(0.65f, 0.68f, 0.72f);        // Slate

    public static readonly Vector3 ColorMatched = new(0.22f, 0.88f, 0.38f);        // Emerald green
    public static readonly Vector3 ColorUnmatched = new(0.95f, 0.38f, 0.28f);      // Warm vermilion
    public static readonly Vector3 ColorDuplicated = new(1.00f, 0.82f, 0.10f);     // Gold
    public static readonly Vector3 ColorIgnored = new(0.50f, 0.55f, 0.60f);        // Steel gray

    /// <summary>
    /// Computes a distinct hue for each instance/doodad so adjacent models have contrasting colors.
    /// Guarantees that even with Ck24 == 0 (doodads), adjacent objects receive distinct hues.
    /// </summary>
    public Vector3 ComputeInstanceColor(
        uint ck24,
        int objectPartId,
        int tileX,
        int tileY,
        Vector3 worldAnchor,
        int surfaceCount)
    {
        uint hash;
        if (ck24 != 0)
        {
            hash = HashCodeCombine(ck24, (uint)objectPartId, (uint)tileX, (uint)tileY);
        }
        else
        {
            int quantizedX = (int)MathF.Round(worldAnchor.X * 0.2f);
            int quantizedY = (int)MathF.Round(worldAnchor.Y * 0.2f);
            int quantizedZ = (int)MathF.Round(worldAnchor.Z * 0.2f);
            hash = HashCodeCombine(
                (uint)HashCode.Combine(tileX, tileY, objectPartId),
                (uint)quantizedX,
                (uint)quantizedY,
                (uint)(quantizedZ ^ (surfaceCount * 31)));
        }

        uint golden = hash * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return HsvToRgb(hue, 0.78f, 0.92f);
    }

    /// <summary>
    /// Returns the semantic color based on Ck24Type and dominant group flags.
    /// </summary>
    public Vector3 ComputeSemanticCategoryColor(byte ck24Type, byte dominantGroupKey)
    {
        return ck24Type switch
        {
            0x03 => ColorDoodad,
            0x10 => ColorFloorLowest,
            0x11 => ColorFloorCommon,
            0x13 => ColorFloorUpper,
            0x14 => ColorFloorAngled,
            _    => dominantGroupKey switch
            {
                0x01 => ColorTerrainLinker,
                0x02 => ColorConnector,
                _    => ColorGeneric
            }
        };
    }

    /// <summary>
    /// Maps absolute world elevation Z across a smooth perceptual gradient (Navy -> Cyan -> Green -> Orange -> Pale Gold).
    /// </summary>
    public Vector3 ComputeElevationColor(float worldZ, float minZ, float maxZ)
    {
        float span = maxZ - minZ;
        float t = span > 0.01f ? Math.Clamp((worldZ - minZ) / span, 0f, 1f) : 0.5f;

        if (t < 0.25f)
        {
            float seg = t / 0.25f;
            return Vector3.Lerp(new Vector3(0.08f, 0.12f, 0.40f), new Vector3(0.12f, 0.65f, 0.85f), seg);
        }
        if (t < 0.50f)
        {
            float seg = (t - 0.25f) / 0.25f;
            return Vector3.Lerp(new Vector3(0.12f, 0.65f, 0.85f), new Vector3(0.25f, 0.85f, 0.38f), seg);
        }
        if (t < 0.75f)
        {
            float seg = (t - 0.50f) / 0.25f;
            return Vector3.Lerp(new Vector3(0.25f, 0.85f, 0.38f), new Vector3(0.95f, 0.62f, 0.15f), seg);
        }
        else
        {
            float seg = (t - 0.75f) / 0.25f;
            return Vector3.Lerp(new Vector3(0.95f, 0.62f, 0.15f), new Vector3(0.98f, 0.95f, 0.82f), seg);
        }
    }

    /// <summary>
    /// Computes color for match/curation status.
    /// </summary>
    public Vector3 ComputeMatchStatusColor(bool isMatched, bool isDuplicated, byte ck24Type)
    {
        if (isDuplicated)
            return ColorDuplicated;
        if (isMatched)
            return ColorMatched;
        if (ck24Type == 0x03)
            return ColorUnmatched;

        return ColorIgnored;
    }

    private static uint HashCodeCombine(uint a, uint b, uint c, uint d)
    {
        uint hash = 2166136261u;
        hash = (hash ^ a) * 16777619u;
        hash = (hash ^ b) * 16777619u;
        hash = (hash ^ c) * 16777619u;
        hash = (hash ^ d) * 16777619u;
        return hash;
    }

    public static Vector3 ColorFromSeed(uint seed)
    {
        uint golden = seed * 2654435761u;
        float hue = (golden & 0x00FFFFFF) / 16777215f;
        return HsvToRgb(hue, 0.75f, 0.95f);
    }

    public static Vector3 HsvToRgb(float h, float s, float v)
    {
        h = h - MathF.Floor(h);
        float c = v * s;
        float x = c * (1f - MathF.Abs((h * 6f) % 2f - 1f));
        float m = v - c;

        float r;
        float g;
        float b;
        int sector = (int)(h * 6f);
        switch (sector)
        {
            case 0:
                r = c; g = x; b = 0f;
                break;
            case 1:
                r = x; g = c; b = 0f;
                break;
            case 2:
                r = 0f; g = c; b = x;
                break;
            case 3:
                r = 0f; g = x; b = c;
                break;
            case 4:
                r = x; g = 0f; b = c;
                break;
            default:
                r = c; g = 0f; b = x;
                break;
        }

        return new Vector3(r + m, g + m, b + m);
    }
}
