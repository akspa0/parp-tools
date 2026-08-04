using System.Numerics;
using System.Text.RegularExpressions;

namespace WowViewer.Core.PM4.Services;

public static partial class Pm4CoordinateService
{
    public const float TileSize = 533.33333f;
    public const float HalfMapExtent = 32f * TileSize;

    /// <summary>
    /// The map origin used by ADT placement decoding — see <c>AdtPlacementReader.MapOrigin</c>.
    /// MDDF/MODF positions are stored as a distance from this origin and are converted to map
    /// space by subtracting them from it. MSVT is stored in that same pre-subtraction space.
    /// </summary>
    public const float MapOrigin = 17066.666f;

    public const string DefaultDevelopmentMapDirectory = "test_data/development/World/Maps/development";

    /// <summary>
    /// Converts an MSVT vertex into ADT placement space — the same space
    /// <c>AdtPlacementReader</c> emits for MDDF and MODF positions.
    /// </summary>
    /// <remarks>
    /// MEASURED against ADT ground truth, corpus-wide, on 2026-08-03: MSVT stores a
    /// distance-from-origin coordinate, exactly like a raw MDDF position, with its two horizontal
    /// fields in the opposite order to MDDF's. Concretely <c>MSVT.X == MDDF.rawY</c> and
    /// <c>MSVT.Y == MDDF.rawX</c>, so the conversion is a per-axis subtraction from
    /// <see cref="MapOrigin"/> with NO axis swap:
    /// <code>placement = (MapOrigin - MSVT.X, MapOrigin - MSVT.Y, MSVT.Z)</code>
    ///
    /// Evidence: over the 179 development tiles that have both a PM4 and a correctly named
    /// <c>_obj0.adt</c> holding placements, <b>55,978 of 60,560 (92.4%)</b> MDDF/MODF positions
    /// fall inside the paired PM4's MSVT footprint under this mapping. The unswapped alternative
    /// scores <b>412 of 60,560 (0.7%)</b> and is eliminated. Worked example —
    /// <c>development_01_00.pm4</c> has MSVT X = 41.0..52.9 and Y = 778.9..790.6; the three MDDF
    /// entries in <c>development_1_0_obj0.adt</c> have rawY = 42.3..51.1 and rawX = 780.3..789.0,
    /// nested inside it on both axes.
    ///
    /// This supersedes the earlier reading that MSVT was already in absolute map coordinates with
    /// its axes swapped. That reading came from a bounds fit, which proves only which BAND a value
    /// lies in — it cannot see a reflection about the map centre, because reflecting a band keeps
    /// it a band. The raw band measurement it rested on is still correct and still reproduces
    /// (309/309 files have X inside the band of the filename's SECOND number and Y inside the band
    /// of its FIRST); what was wrong was reading a distance-from-origin band as a map tile index.
    /// The map tile index is <c>31 - band</c>.
    ///
    /// <see cref="Pm4PlacementMath.ConvertPm4VertexToWorld"/> is a DIFFERENT and correct
    /// conversion: it emits an intermediate space that the viewer finishes with
    /// <c>renderer = (MapOrigin - world.Y, MapOrigin - world.X, world.Z)</c>. Composing the two
    /// yields exactly the transform above, so that method's axis swap must NOT be removed — it is
    /// cancelled by the swap in the renderer step.
    /// </remarks>
    public static Vector3 Pm4LocalToAdtPlacement(Vector3 msvtPosition)
        => new(MapOrigin - msvtPosition.X, MapOrigin - msvtPosition.Y, msvtPosition.Z);

    /// <inheritdoc cref="Pm4LocalToAdtPlacement(Vector3)"/>
    /// <remarks>
    /// The tile indices are not needed to place a vertex and are accepted only so callers can
    /// validate the result against <see cref="IsWithinPlacementTileBounds"/>.
    /// </remarks>
    public static Vector3 Pm4LocalToAdtPlacement(Vector3 msvtPosition, int tileX, int tileY)
        => Pm4LocalToAdtPlacement(msvtPosition);

    public static Vector3 MprlToAdtPlacement(Vector3 mprlPosition)
    {
        return mprlPosition;
    }

    /// <summary>
    /// Tests an ADT placement-space position against the extent of the tile a PM4 filename names.
    /// </summary>
    /// <remarks>
    /// Takes the filename numbers in the order they appear — <c>development_&lt;first&gt;_&lt;second&gt;</c>
    /// — because that is the only form a caller can read off a path without already knowing the
    /// answer. The second number bounds the X axis and the first bounds Y, which is the measured
    /// pairing recorded on <see cref="Pm4LocalToAdtPlacement(Vector3)"/>.
    ///
    /// Both bands run DOWNWARD from <see cref="MapOrigin"/>, because placement space is a distance
    /// subtracted from the origin. The horizontal plane is X/Y and Z is height, so Z is not tested.
    /// </remarks>
    public static bool IsWithinPlacementTileBounds(Vector3 placementPosition, int tileFirst, int tileSecond, float tolerance = 0f)
    {
        (float minX, float maxX) = GetPlacementTileBand(tileSecond, tolerance);
        (float minY, float maxY) = GetPlacementTileBand(tileFirst, tolerance);

        return placementPosition.X >= minX && placementPosition.X <= maxX
            && placementPosition.Y >= minY && placementPosition.Y <= maxY;
    }

    /// <summary>
    /// The placement-space interval covered by one tile index on a single horizontal axis.
    /// </summary>
    public static (float Min, float Max) GetPlacementTileBand(int tileIndex, float tolerance = 0f)
        => (MapOrigin - (tileIndex + 1) * TileSize - tolerance, MapOrigin - tileIndex * TileSize + tolerance);

    /// <summary>
    /// The tile index whose band contains a placement-space coordinate on one horizontal axis.
    /// </summary>
    public static int PlacementCoordinateToTileIndex(float placementCoordinate)
        => (int)MathF.Floor((MapOrigin - placementCoordinate) / TileSize);

    /// <summary>
    /// Resolves the <c>_obj0.adt</c> that carries the placements for a PM4 tile, or null when the
    /// corpus has no companion for it.
    /// </summary>
    /// <remarks>
    /// PM4 filenames zero-pad their tile numbers to two digits (<c>development_01_00.pm4</c>)
    /// while ADT filenames do not (<c>development_1_0_obj0.adt</c>). Appending the suffix to the
    /// PM4 stem — which is what this used to do — therefore produced a path that exists for NONE
    /// of the 616 corpus files, even though 411 of them do have a companion under the unpadded
    /// name. Callers must handle null rather than falling back to an arbitrary ADT: pairing a PM4
    /// with an unrelated tile's placements is worse than having none.
    /// </remarks>
    public static string? TryGetObj0PathForPm4(string pm4Path)
    {
        string directory = Path.GetDirectoryName(pm4Path) ?? string.Empty;
        string fileName = Path.GetFileNameWithoutExtension(pm4Path);

        // Preferred: strip the PM4 zero padding, which is how the ADTs are named.
        Match match = Pm4TilePattern().Match(fileName + ".pm4");
        if (match.Success
            && int.TryParse(match.Groups[1].Value, out int first)
            && int.TryParse(match.Groups[2].Value, out int second))
        {
            string stem = fileName[..match.Index];
            string unpadded = Path.Combine(directory, $"{stem}_{first}_{second}_obj0.adt");
            if (File.Exists(unpadded))
                return unpadded;
        }

        // Fall back to the literal stem for corpora that pad both file kinds alike.
        string literal = Path.Combine(directory, fileName + "_obj0.adt");
        return File.Exists(literal) ? literal : null;
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