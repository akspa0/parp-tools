using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Rasterizes PM4 path and building-footprint data onto 257×257 masks for a single ADT tile.
/// Scans the entire map directory for all PM4 files (MdxViewer pattern) and rasterizes
/// any geometry that falls within the target tile.
/// </summary>
public static class AdtPm4MaskBuilder
{
    private const int TileHeightmapSize = 257;
    private const float TileSize = 533.33333f;

    /// <summary>
    /// Reads all PM4 files in the map directory and produces path, building, and MPRL portal masks for the given ADT tile.
    /// </summary>
    /// <param name="adtPath">Full path to the root ADT file.</param>
    /// <param name="pathMask">Output 257×257 path mask (1.0 = path, 0.0 = no path).</param>
    /// <param name="buildingFootprintMask">Output 257×257 building footprint mask (0.0–1.0).</param>
    /// <param name="mprlMask">Output 257×257 MPRL portal mask (1.0 = portal region, 0.0 = none).</param>
    /// <returns>True if any PM4 data was found and rasterized.</returns>
    public static bool TryBuild(string adtPath, out float[,]? pathMask, out float[,]? buildingFootprintMask, out float[,]? mprlMask)
    {
        pathMask = null;
        buildingFootprintMask = null;
        mprlMask = null;

        if (!TryParseAdtTileCoords(adtPath, out int tileX, out int tileY))
            return false;

        // Find all PM4 files in the map directory (MdxViewer pattern: scan entire map)
        string? mapDir = Path.GetDirectoryName(adtPath);
        if (string.IsNullOrEmpty(mapDir))
            return false;

        string[] pm4Files;
        try
        {
            pm4Files = Directory.GetFiles(mapDir, "*.pm4", SearchOption.TopDirectoryOnly);
        }
        catch
        {
            return false;
        }

        if (pm4Files.Length == 0)
            return false;

        float[,] path = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] building = new float[TileHeightmapSize, TileHeightmapSize];
        float[,] mprl = new float[TileHeightmapSize, TileHeightmapSize];
        bool anyPath = false;
        bool anyBuilding = false;
        bool anyMprl = false;

        foreach (string pm4Path in pm4Files)
        {
            Pm4ResearchDocument doc;
            try
            {
                doc = Pm4ResearchReader.ReadFile(pm4Path);
            }
            catch
            {
                continue;
            }

            var known = doc.KnownChunks;
            if (known.Mslk.Count == 0 && known.Msur.Count == 0 && known.Msvt.Count == 0 && known.Mspv.Count == 0)
                continue;

            // ── Rasterize MSLK path links ────────────────────────────────────
            if (known.Mslk.Count > 0 && known.Mspv.Count > 0 && known.Mspi.Count > 0)
            {
                foreach (var mslk in known.Mslk)
                {
                    int firstIdx = (int)mslk.MspiFirstIndex;
                    int count = mslk.MspiIndexCount;
                    if (firstIdx < 0 || count <= 0 || firstIdx + count > known.Mspi.Count)
                        continue;

                    var positions = new List<Vector3>(count);
                    for (int i = 0; i < count; i++)
                    {
                        uint mspiIdx = known.Mspi[firstIdx + i];
                        if (mspiIdx < known.Mspv.Count)
                            positions.Add(known.Mspv[(int)mspiIdx]);
                    }

                    if (positions.Count < 2)
                        continue;

                    var tilePositions = new List<Vector3>(positions.Count);
                    foreach (var localPos in positions)
                    {
                        Vector3 adtPos = Pm4CoordinateService.Pm4LocalToAdtPlacement(localPos, tileX, tileY);
                        if (IsWithinTile(adtPos, tileX, tileY))
                            tilePositions.Add(adtPos);
                    }

                    if (tilePositions.Count < 2)
                        continue;

                    anyPath = true;
                    for (int i = 1; i < tilePositions.Count; i++)
                    {
                        RasterizeLine(path, tilePositions[i - 1], tilePositions[i], tileX, tileY, value: 1.0f);
                    }
                }
            }

            // ── Rasterize MSUR building surfaces ─────────────────────────────
            if (known.Msur.Count > 0 && known.Msvt.Count > 0 && known.Msvi.Count > 0)
            {
                foreach (var msur in known.Msur)
                {
                    int firstIdx = (int)msur.MsviFirstIndex;
                    int count = msur.IndexCount;
                    if (firstIdx < 0 || count < 3 || firstIdx + count > known.Msvi.Count)
                        continue;

                    var positions = new List<Vector3>(count);
                    for (int i = 0; i < count; i++)
                    {
                        uint msviIdx = known.Msvi[firstIdx + i];
                        if (msviIdx < known.Msvt.Count)
                            positions.Add(known.Msvt[(int)msviIdx]);
                    }

                    if (positions.Count < 3)
                        continue;

                    var tilePositions = new List<Vector3>(positions.Count);
                    foreach (var localPos in positions)
                    {
                        Vector3 adtPos = Pm4CoordinateService.Pm4LocalToAdtPlacement(localPos, tileX, tileY);
                        if (IsWithinTile(adtPos, tileX, tileY))
                            tilePositions.Add(adtPos);
                    }

                    if (tilePositions.Count < 3)
                        continue;

                    anyBuilding = true;
                    for (int i = 0; i + 2 < tilePositions.Count; i += 3)
                    {
                        RasterizeTriangle(building, tilePositions[i], tilePositions[i + 1], tilePositions[i + 2], tileX, tileY);
                    }
                }
            }

            // ── Rasterize MPRL portal positions ────────────────────────────
            if (known.Mprl.Count > 0)
            {
                foreach (var mprlEntry in known.Mprl)
                {
                    // Unk16 == 0 indicates a normal portal entry (non-terminator).
                    if (mprlEntry.Unk16 != 0)
                        continue;

                    Vector3 worldPos = Pm4CoordinateService.MprlToAdtPlacement(mprlEntry.Position);
                    if (!IsWithinTile(worldPos, tileX, tileY))
                        continue;

                    int px = WorldToPixelX(worldPos.X, tileX);
                    int py = WorldToPixelY(worldPos.Z, tileY);
                    // Paint a small disk — MPRL portals are point-like indicators,
                    // similar in spirit to ADT hole regions.
                    PaintCircle(mprl, px, py, radius: 3f, value: 1.0f);
                    anyMprl = true;
                }
            }
        }

        if (anyPath)
            pathMask = path;
        if (anyBuilding)
            buildingFootprintMask = building;
        if (anyMprl)
            mprlMask = mprl;

        return anyPath || anyBuilding || anyMprl;
    }

    private static bool IsWithinTile(Vector3 adtPosition, int tileX, int tileY)
    {
        float minX = tileX * TileSize;
        float maxX = (tileX + 1) * TileSize;
        float minZ = tileY * TileSize;
        float maxZ = (tileY + 1) * TileSize;

        return adtPosition.X >= minX && adtPosition.X <= maxX
            && adtPosition.Z >= minZ && adtPosition.Z <= maxZ;
    }

    private static void RasterizeLine(float[,] buffer, Vector3 from, Vector3 to, int tileX, int tileY, float value)
    {
        int x0 = WorldToPixelX(from.X, tileX);
        int y0 = WorldToPixelY(from.Z, tileY);
        int x1 = WorldToPixelX(to.X, tileX);
        int y1 = WorldToPixelY(to.Z, tileY);

        int dx = Math.Abs(x1 - x0);
        int dy = Math.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        int lineWidth = 2;
        while (true)
        {
            PaintCircle(buffer, x0, y0, lineWidth, value);
            if (x0 == x1 && y0 == y1)
                break;
            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    private static void RasterizeTriangle(float[,] buffer, Vector3 a, Vector3 b, Vector3 c, int tileX, int tileY)
    {
        int x0 = WorldToPixelX(a.X, tileX);
        int y0 = WorldToPixelY(a.Z, tileY);
        int x1 = WorldToPixelX(b.X, tileX);
        int y1 = WorldToPixelY(b.Z, tileY);
        int x2 = WorldToPixelX(c.X, tileX);
        int y2 = WorldToPixelY(c.Z, tileY);

        int minX = Math.Clamp(Math.Min(x0, Math.Min(x1, x2)), 0, TileHeightmapSize - 1);
        int maxX = Math.Clamp(Math.Max(x0, Math.Max(x1, x2)), 0, TileHeightmapSize - 1);
        int minY = Math.Clamp(Math.Min(y0, Math.Min(y1, y2)), 0, TileHeightmapSize - 1);
        int maxY = Math.Clamp(Math.Max(y0, Math.Max(y1, y2)), 0, TileHeightmapSize - 1);

        float denom = (float)(y1 - y2) * (x0 - x2) + (float)(x2 - x1) * (y0 - y2);
        if (MathF.Abs(denom) < 0.0001f)
            return;

        for (int y = minY; y <= maxY; y++)
        {
            for (int x = minX; x <= maxX; x++)
            {
                float w0 = (float)(y1 - y2) * (x - x2) + (float)(x2 - x1) * (y - y2);
                float w1 = (float)(y2 - y0) * (x - x2) + (float)(x0 - x2) * (y - y2);
                float w2 = (float)(y0 - y1) * (x - x2) + (float)(x1 - x2) * (y - y2);

                w0 /= denom;
                w1 /= denom;
                w2 /= denom;

                if (w0 >= 0 && w1 >= 0 && w2 >= 0)
                {
                    buffer[y, x] = 1.0f;
                }
            }
        }
    }

    private static void PaintCircle(float[,] buffer, int cx, int cy, float radius, float value)
    {
        int r = (int)MathF.Ceiling(radius);
        for (int dy = -r; dy <= r; dy++)
        {
            for (int dx = -r; dx <= r; dx++)
            {
                if ((dx * dx) + (dy * dy) > r * r)
                    continue;

                int x = cx + dx;
                int y = cy + dy;
                if ((uint)x >= TileHeightmapSize || (uint)y >= TileHeightmapSize)
                    continue;

                buffer[y, x] = value;
            }
        }
    }

    private static int WorldToPixelX(float worldX, int tileX)
    {
        float localX = worldX - (tileX * TileSize);
        return Math.Clamp((int)(localX / TileSize * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
    }

    private static int WorldToPixelY(float worldZ, int tileY)
    {
        float localZ = worldZ - (tileY * TileSize);
        return Math.Clamp((int)(localZ / TileSize * (TileHeightmapSize - 1)), 0, TileHeightmapSize - 1);
    }

    private static bool TryParseAdtTileCoords(string adtPath, out int tileX, out int tileY)
    {
        tileX = 0;
        tileY = 0;
        string fileName = Path.GetFileNameWithoutExtension(adtPath);
        int lastUnderscore = fileName.LastIndexOf('_');
        if (lastUnderscore < 0) return false;
        int secondLast = fileName.LastIndexOf('_', lastUnderscore - 1);
        if (secondLast < 0) return false;

        return int.TryParse(fileName.AsSpan(secondLast + 1, lastUnderscore - secondLast - 1), out tileX)
            && int.TryParse(fileName.AsSpan(lastUnderscore + 1), out tileY);
    }
}
