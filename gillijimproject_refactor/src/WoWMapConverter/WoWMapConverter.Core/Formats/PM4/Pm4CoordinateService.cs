using System.Numerics;
using System.Text.RegularExpressions;

namespace WoWMapConverter.Core.Formats.PM4;

/// <summary>
/// Authoritative PM4-to-ADT placement coordinate helpers for active core code.
/// </summary>
public static partial class Pm4CoordinateService
{
    public const float TileSize = 533.33333f;
    public const float HalfMapExtent = 32f * TileSize;
    public const string DefaultDevelopmentMapDirectory = "test_data/development/World/Maps/development";

    public static Vector3 Pm4LocalToAdtPlacement(Vector3 pm4LocalPosition, int tileX, int tileY)
    {
        float placementX = tileX * TileSize + pm4LocalPosition.X;
        float placementY = pm4LocalPosition.Y;
        float placementZ = tileY * TileSize + pm4LocalPosition.Z;
        return new Vector3(placementX, placementY, placementZ);
    }

    public static Vector3 MprlToAdtPlacement(Vector3 mprlPosition)
    {
        return mprlPosition;
    }

    public static bool IsWithinPlacementTileBounds(Vector3 placementPosition, int tileX, int tileY, float tolerance = 0f)
    {
        float minX = tileX * TileSize - tolerance;
        float maxX = (tileX + 1) * TileSize + tolerance;
        float minZ = tileY * TileSize - tolerance;
        float maxZ = (tileY + 1) * TileSize + tolerance;

        return placementPosition.X >= minX && placementPosition.X <= maxX
            && placementPosition.Z >= minZ && placementPosition.Z <= maxZ;
    }

    public static string GetObj0PathForPm4(string pm4Path)
    {
        string directory = Path.GetDirectoryName(pm4Path) ?? string.Empty;
        string fileName = Path.GetFileNameWithoutExtension(pm4Path);
        return Path.Combine(directory, fileName + "_obj0.adt");
    }

    public static string ResolveMapDirectory(string requestedPath)
    {
        if (Path.IsPathFullyQualified(requestedPath) && Directory.Exists(requestedPath))
            return requestedPath;

        string[] searchRoots =
        {
            Directory.GetCurrentDirectory(),
            AppContext.BaseDirectory
        };

        foreach (string root in searchRoots)
        {
            string? current = Path.GetFullPath(root);
            while (!string.IsNullOrEmpty(current))
            {
                string directCandidate = Path.Combine(current, requestedPath);
                if (Directory.Exists(directCandidate))
                    return directCandidate;

                string workspaceCandidate = Path.Combine(current, "gillijimproject_refactor", requestedPath);
                if (Directory.Exists(workspaceCandidate))
                    return workspaceCandidate;

                DirectoryInfo? parent = Directory.GetParent(current);
                current = parent?.FullName;
            }
        }

        return Path.GetFullPath(requestedPath);
    }

    public static bool TryParseTileCoordinates(string pathOrFileName, out int tileX, out int tileY)
    {
        string fileName = Path.GetFileName(pathOrFileName);
        Match match = Pm4TilePattern().Match(fileName);
        if (match.Success
            && int.TryParse(match.Groups[1].Value, out tileX)
            && int.TryParse(match.Groups[2].Value, out tileY))
        {
            return true;
        }

        tileX = 0;
        tileY = 0;
        return false;
    }

    [GeneratedRegex(@"_(\d+)_(\d+)\.pm4$", RegexOptions.IgnoreCase)]
    private static partial Regex Pm4TilePattern();
}