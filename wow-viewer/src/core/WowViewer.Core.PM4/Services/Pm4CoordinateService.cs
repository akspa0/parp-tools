using System.Numerics;
using System.Text.RegularExpressions;

namespace WowViewer.Core.PM4.Services;

public static partial class Pm4CoordinateService
{
    public const float TileSize = 533.33333f;
    public const float HalfMapExtent = 32f * TileSize;
    public const string DefaultDevelopmentMapDirectory = "test_data/development/World/Maps/development";

    /// <summary>
    /// Converts an MSVT vertex to a world placement.
    /// </summary>
    /// <remarks>
    /// MSVT is stored in ABSOLUTE WORLD coordinates on both horizontal axes, with Z as height.
    /// The tile indices are not needed to place it and are accepted only so callers can validate
    /// against <see cref="IsWithinPlacementTileBounds"/>.
    ///
    /// Measured over all 309 non-empty files of the development corpus with `pm4 bounds-audit`:
    /// X lies fully inside the world band of the SECOND number in the filename for 309/309 files,
    /// Y inside the band of the FIRST number for 309/309, and the maximum span of either axis is
    /// exactly one tile (533.33). Z spans -658..596 with a mean span of 173 — height, not a
    /// horizontal axis.
    ///
    /// The previous implementation added <c>tileX * TileSize</c> to X and <c>tileY * TileSize</c>
    /// to Z. That double-offset an already-absolute X using the wrong index, and offset the height
    /// axis as if it were horizontal. For tile 00_00 both indices are zero so both errors cancel,
    /// which is why 0_0 alone appeared correct while every other tile piled up around it.
    ///
    /// MSVT's own X and Y are SWAPPED relative to world: world X comes from MSVT.Y and world Y
    /// from MSVT.X. Confirmed against ground truth rather than inferred — the human tents in
    /// development_01_00.pm4 belong on the tile to the right of 0_0 in the same row, i.e. world
    /// (x=1, y=0). Raw MSVT has X in band 0 (41.0..52.9) and Y in band 1 (778.9..790.6), so only
    /// the swapped reading puts them where they belong. The filename is therefore ordinary
    /// COL_ROW; bounds measurements alone cannot separate "filename is row_col" from "MSVT axes
    /// are swapped", and this is the observation that does.
    ///
    /// <see cref="Pm4PlacementMath.ConvertPm4VertexToWorld"/> already applies this swap in its
    /// XYPlaneZUp case and is correct.
    /// </remarks>
    public static Vector3 Pm4LocalToAdtPlacement(Vector3 msvtPosition, int tileX, int tileY)
        => new(msvtPosition.Y, msvtPosition.X, msvtPosition.Z);

    public static Vector3 MprlToAdtPlacement(Vector3 mprlPosition)
    {
        return mprlPosition;
    }

    /// <summary>
    /// Tests a world placement against a tile's horizontal extent.
    /// </summary>
    /// <remarks>
    /// The horizontal plane is X/Y and Z is height, so Z is not tested. <paramref name="tileX"/>
    /// and <paramref name="tileY"/> are axis-ordered — pass filename numbers through
    /// <see cref="FileOrderToTileAxes"/> first.
    /// </remarks>
    public static bool IsWithinPlacementTileBounds(Vector3 placementPosition, int tileX, int tileY, float tolerance = 0f)
    {
        float minX = tileX * TileSize - tolerance;
        float maxX = (tileX + 1) * TileSize + tolerance;
        float minY = tileY * TileSize - tolerance;
        float maxY = (tileY + 1) * TileSize + tolerance;

        return placementPosition.X >= minX && placementPosition.X <= maxX
            && placementPosition.Y >= minY && placementPosition.Y <= maxY;
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
        [
            Directory.GetCurrentDirectory(),
            AppContext.BaseDirectory
        ];

        foreach (string root in searchRoots)
        {
            string? current = Path.GetFullPath(root);
            while (!string.IsNullOrEmpty(current))
            {
                string candidate = Path.Combine(current, requestedPath);
                if (Directory.Exists(candidate))
                    return candidate;

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