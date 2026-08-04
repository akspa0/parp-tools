using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Tests whether CK24 == 0 is the bucket every M2 doodad collision falls into, and looks for the
/// field that separates individual doodads inside it.
/// </summary>
/// <remarks>
/// <para><b>The claim under test.</b> CK24 keys objects, except that CK24 0 spans 291 tiles and
/// behaves like a sentinel rather than an identity. The hypothesis is that it is not a sentinel at
/// all but a <i>class</i>: the bucket M2 collision lands in, as against WMO collision which gets a
/// real key. The tents are the motivating case — <c>development_01_00.pm4</c> carries
/// <c>CK24 0x000000</c> and resolves to <c>HU_TENT02.M2</c>.</para>
///
/// <para><b>Why this is decidable now.</b> ADT placements split cleanly by asset class: MDDF is
/// doodads (M2) and MODF is world models (WMO). So each PM4 object can be scored against both and
/// the two CK24 populations compared. If CK24 0 objects sit on MDDF positions while CK24 non-zero
/// objects sit inside MODF boxes, the bucket is a class marker. If both populations look alike, it
/// is not.</para>
///
/// <para><b>Asymmetric tests, on purpose.</b> MDDF carries a position but no extent, so a doodad is
/// scored by distance from the object's centroid. MODF carries a world bounding box, so a world
/// model is scored by containment. Forcing one metric on both would handicap whichever chunk it
/// suited less and manufacture the difference this is trying to detect.</para>
///
/// <para><b>Candidate separators.</b> For the objects that do land in the doodad bucket, the counts
/// that would have to line up for a field to be the per-doodad identity are reported next to the
/// number of MDDF placements on the same tile: distinct <c>MSLK.GroupObjectId</c>, distinct
/// <c>MSUR.GroupKey</c>, and the count of MSLK entries with <c>MspiFirstIndex &lt; 0</c> — the 53%
/// of links prior art reads as doodad placements carrying anchors rather than path windows.</para>
/// </remarks>
public static class Pm4DoodadSplitAnalyzer
{
    /// <summary>How close an object centroid must be to an MDDF position to count as sitting on it.</summary>
    public const float DoodadMatchRadius = 24f;

    public static Pm4DoodadSplitReport AnalyzeDirectory(
        string inputDirectory,
        IReadOnlyDictionary<string, Pm4TilePlacements> placementsByFile)
    {
        ArgumentNullException.ThrowIfNull(placementsByFile);
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4DoodadSplitObjectRecord> objects = [];
        List<Pm4DoodadSplitTileRecord> tiles = [];

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName, StringComparer.Ordinal))
        {
            string fileName = Path.GetFileName(path);
            if (!placementsByFile.TryGetValue(fileName, out Pm4TilePlacements placements))
                continue;

            Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
            Pm4KnownChunkSet chunks = document.KnownChunks;
            if (chunks.Msvt.Count == 0 || chunks.Msvi.Count == 0 || chunks.Msur.Count == 0)
                continue;

            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileFirst, out int tileSecond))
                continue;

            tiles.Add(AnalyzeTile(chunks, fileName, tileFirst, tileSecond, placements, objects));
        }

        return BuildReport(resolvedDirectory, tiles, objects);
    }

    private static Pm4DoodadSplitTileRecord AnalyzeTile(
        Pm4KnownChunkSet chunks,
        string fileName,
        int tileFirst,
        int tileSecond,
        Pm4TilePlacements placements,
        List<Pm4DoodadSplitObjectRecord> objects)
    {
        // MSLK.RefIndex points at an MSUR entry, so invert it to reach a surface's links.
        Dictionary<int, List<Pm4MslkEntry>> linksBySurface = [];
        int anchorOnlyLinks = 0;
        foreach (Pm4MslkEntry link in chunks.Mslk)
        {
            if (link.MspiFirstIndex < 0)
                anchorOnlyLinks++;

            if (link.RefIndex >= chunks.Msur.Count)
                continue;

            if (!linksBySurface.TryGetValue(link.RefIndex, out List<Pm4MslkEntry>? bucket))
            {
                bucket = [];
                linksBySurface[link.RefIndex] = bucket;
            }

            bucket.Add(link);
        }

        int zeroObjects = 0;
        int zeroOnDoodad = 0;
        int nonZeroObjects = 0;
        int nonZeroInWorldModel = 0;

        var surfacesByCk24 = chunks.Msur
            .Select(static (surface, index) => (surface, index))
            .GroupBy(static pair => pair.surface.Ck24);

        foreach (var group in surfacesByCk24)
        {
            List<Pm4MsurEntry> surfaces = [.. group.Select(static pair => pair.surface)];
            List<Vector3> vertices = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
            if (vertices.Count == 0)
                continue;

            Vector3 centroid = Vector3.Zero;
            foreach (Vector3 vertex in vertices)
                centroid += vertex;
            centroid /= vertices.Count;

            Vector3 placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(centroid);
            Vector2 planar = new(placement.X, placement.Y);

            float doodadDistance = NearestDoodadDistance(placements.DoodadPositions, planar, out string doodadPath);
            bool insideWorldModel = IsInsideAnyWorldModel(placements.WorldModelBoxes, planar, out string worldModelPath);

            HashSet<uint> groupObjectIds = [];
            HashSet<byte> groupKeys = [];
            int anchorLinks = 0;
            foreach ((Pm4MsurEntry surface, int index) in group)
            {
                groupKeys.Add(surface.GroupKey);
                if (!linksBySurface.TryGetValue(index, out List<Pm4MslkEntry>? links))
                    continue;

                foreach (Pm4MslkEntry link in links)
                {
                    groupObjectIds.Add(link.GroupObjectId);
                    if (link.MspiFirstIndex < 0)
                        anchorLinks++;
                }
            }

            bool isZeroBucket = group.Key == 0u;
            bool onDoodad = doodadDistance <= DoodadMatchRadius;
            if (isZeroBucket)
            {
                zeroObjects++;
                if (onDoodad)
                    zeroOnDoodad++;
            }
            else
            {
                nonZeroObjects++;
                if (insideWorldModel)
                    nonZeroInWorldModel++;
            }

            objects.Add(new Pm4DoodadSplitObjectRecord(
                fileName,
                tileFirst,
                tileSecond,
                chunks.Mshd?.Field04 ?? 0u,
                group.Key,
                isZeroBucket,
                surfaces.Count,
                vertices.Count,
                doodadDistance,
                onDoodad,
                doodadPath,
                insideWorldModel,
                worldModelPath,
                groupObjectIds.Count,
                groupKeys.Count,
                anchorLinks));
        }

        return new Pm4DoodadSplitTileRecord(
            fileName,
            tileFirst,
            tileSecond,
            chunks.Mshd?.Field04 ?? 0u,
            placements.DoodadPositions.Count,
            placements.WorldModelBoxes.Count,
            zeroObjects,
            zeroOnDoodad,
            nonZeroObjects,
            nonZeroInWorldModel,
            chunks.Mslk.Count,
            anchorOnlyLinks,
            chunks.Mprl.Count,
            chunks.Mslk.Select(static link => link.GroupObjectId).Distinct().Count());
    }

    private static float NearestDoodadDistance(
        IReadOnlyList<Pm4NamedPoint> doodads,
        Vector2 point,
        out string modelPath)
    {
        modelPath = string.Empty;
        float best = float.MaxValue;

        foreach (Pm4NamedPoint doodad in doodads)
        {
            float distance = Vector2.Distance(new Vector2(doodad.X, doodad.Y), point);
            if (distance < best)
            {
                best = distance;
                modelPath = doodad.ModelPath;
            }
        }

        return best;
    }

    private static bool IsInsideAnyWorldModel(
        IReadOnlyList<Pm4PlacementBox> boxes,
        Vector2 point,
        out string modelPath)
    {
        modelPath = string.Empty;
        float bestArea = float.MaxValue;
        bool found = false;

        foreach (Pm4PlacementBox box in boxes)
        {
            if (point.X < box.MinX || point.X > box.MaxX || point.Y < box.MinY || point.Y > box.MaxY)
                continue;

            float area = (box.MaxX - box.MinX) * (box.MaxY - box.MinY);
            if (area < bestArea)
            {
                bestArea = area;
                modelPath = box.ModelPath;
                found = true;
            }
        }

        return found;
    }

    private static Pm4DoodadSplitReport BuildReport(
        string resolvedDirectory,
        List<Pm4DoodadSplitTileRecord> tiles,
        List<Pm4DoodadSplitObjectRecord> objects)
    {
        List<Pm4DoodadSplitObjectRecord> zero = [.. objects.Where(static o => o.IsZeroBucket)];
        List<Pm4DoodadSplitObjectRecord> nonZero = [.. objects.Where(static o => !o.IsZeroBucket)];

        double zeroOnDoodad = Fraction(zero.Count(static o => o.SitsOnDoodad), zero.Count);
        double zeroInWorldModel = Fraction(zero.Count(static o => o.InsideWorldModel), zero.Count);
        double nonZeroOnDoodad = Fraction(nonZero.Count(static o => o.SitsOnDoodad), nonZero.Count);
        double nonZeroInWorldModel = Fraction(nonZero.Count(static o => o.InsideWorldModel), nonZero.Count);

        // The two tests are not equally powerful and must not be summarised as one verdict.
        //
        // Containment discriminates: an object either falls inside a WMO's world box or it does
        // not, and being outside every box is a real negative.
        //
        // Proximity does not, on its own. Doodads are dense, so ANY object is near one by chance.
        // The known-WMO population's own proximity rate is therefore the control: whatever
        // fraction of CK24 != 0 objects sit within the radius of some doodad is what "near a
        // doodad" is worth when it means nothing.
        double proximityControl = nonZeroOnDoodad;
        double proximityLift = zeroOnDoodad - proximityControl;
        bool containmentSeparates = nonZeroInWorldModel - zeroInWorldModel >= 0.25d;
        bool proximityHasPower = proximityControl <= 0.35d;

        string verdict = zero.Count == 0 || nonZero.Count == 0
            ? "undecidable — one of the two CK24 populations is empty in this corpus"
            : !containmentSeparates
                ? "not supported — CK24 does not separate by asset class on either test"
                : proximityHasPower && proximityLift >= 0.15d
                    ? "CK24 0 IS the doodad bucket — non-zero CK24 tracks MODF and CK24 0 tracks MDDF"
                    : $"HALF CONFIRMED — a non-zero CK24 is a WMO instance ({nonZeroInWorldModel:P1} sit "
                        + $"inside a MODF box, against {zeroInWorldModel:P1} of CK24 0), so CK24 0 is the "
                        + "NOT-A-WMO bucket. Whether its contents are specifically M2 doodads is NOT "
                        + $"established here: the proximity test has no power, because {proximityControl:P1} "
                        + "of known-WMO objects also sit within the radius of some doodad.";

        // The strongest test available, and a falsifiable one: if a non-zero CK24 IS a WMO
        // instance, then the count of non-zero CK24 objects on a tile must track the tile's WMO
        // placement count, and a tile with no WMOs at all must have none. Counting is far more
        // powerful than either spatial test, because it cannot be satisfied by coincidence the way
        // "near something" can.
        int wmoFreeTiles = tiles.Count(static tile => tile.WorldModelPlacements == 0);
        int wmoFreeTilesWithKeyedObjects =
            tiles.Count(static tile => tile.WorldModelPlacements == 0 && tile.NonZeroObjects > 0);
        int exactCountMatch =
            tiles.Count(static tile => tile.NonZeroObjects == tile.WorldModelPlacements);
        int withinOne =
            tiles.Count(static tile => Math.Abs(tile.NonZeroObjects - tile.WorldModelPlacements) <= 1);
        int tilesWithExactlyOneZeroBucket =
            tiles.Count(static tile => tile.ZeroBucketObjects == 1);
        int tilesWithAnyZeroBucket =
            tiles.Count(static tile => tile.ZeroBucketObjects > 0);

        Pm4Ck24WmoCorrespondence correspondence = new(
            tiles.Count,
            wmoFreeTiles,
            wmoFreeTilesWithKeyedObjects,
            exactCountMatch,
            withinOne,
            tilesWithAnyZeroBucket,
            tilesWithExactlyOneZeroBucket,
            tiles.Sum(static tile => (long)tile.NonZeroObjects),
            tiles.Sum(static tile => (long)tile.WorldModelPlacements));

        // Does any candidate field produce as many groups as the tile has doodads?
        List<Pm4DoodadSeparatorFit> separators =
        [
            BuildSeparatorFit("MSLK.GroupObjectId (distinct, per tile)", tiles,
                static tile => tile.DistinctGroupObjectIds),
            BuildSeparatorFit("MSLK entries with MspiFirstIndex < 0", tiles,
                static tile => tile.AnchorOnlyLinks),
            BuildSeparatorFit("MPRL entry count", tiles,
                static tile => tile.MprlCount)
        ];

        List<string> notes =
        [
            "Doodads are scored by DISTANCE because MDDF carries a position and no extent; world "
                + "models are scored by CONTAINMENT because MODF carries a world bounding box. Using "
                + "one metric for both would handicap whichever chunk it suited less.",
            $"An object counts as sitting on a doodad when its centroid is within {DoodadMatchRadius:F0} "
                + "units of an MDDF position. Doodad collision is small, so this is a generous radius "
                + "rather than a tight one; tightening it can only lower the CK24 0 figure.",
            "Separator fits compare a candidate field's per-tile cardinality against the tile's MDDF "
                + "count. A ratio near 1.0 means the field could carry per-doodad identity; far from "
                + "1.0 means it counts something else. This is a screen, not a proof — a matching "
                + "count still has to be shown to match the right doodads.",
            "Tiles with no companion _obj0.adt are absent entirely, so every figure here is over the "
                + "subset that has ground truth.",
            $"CONTROL: {nonZeroInWorldModel:P1} of CK24 != 0 objects sit inside a MODF box, and those "
                + $"same objects are within {DoodadMatchRadius:F0} units of a doodad {nonZeroOnDoodad:P1} "
                + "of the time. That second number is the chance rate for the proximity test — read the "
                + "CK24 0 proximity figure against it, never against zero.",
            "CK24 0 is a per-tile LUMP, not one object: it is roughly one bucket per tile but some "
                + "buckets carry over a thousand surfaces. Splitting it is the open problem; the next "
                + "step is to derive components geometrically and then look for the field that is "
                + "constant within a component, rather than guessing fields and hoping the count fits."
        ];

        return new Pm4DoodadSplitReport(
            resolvedDirectory,
            tiles.Count,
            objects.Count,
            zero.Count,
            nonZero.Count,
            zeroOnDoodad,
            zeroInWorldModel,
            nonZeroOnDoodad,
            nonZeroInWorldModel,
            correspondence,
            verdict,
            separators,
            [.. tiles.OrderByDescending(static tile => tile.DoodadPlacements).Take(24)],
            [.. zero.Where(static o => o.SitsOnDoodad).Take(24)],
            notes);
    }

    private static Pm4DoodadSeparatorFit BuildSeparatorFit(
        string name,
        IReadOnlyList<Pm4DoodadSplitTileRecord> tiles,
        Func<Pm4DoodadSplitTileRecord, int> selector)
    {
        List<Pm4DoodadSplitTileRecord> usable =
            [.. tiles.Where(static tile => tile.DoodadPlacements > 0)];
        if (usable.Count == 0)
            return new Pm4DoodadSeparatorFit(name, 0, 0d, 0d, 0);

        double meanRatio = usable.Average(tile => (double)selector(tile) / tile.DoodadPlacements);
        double medianRatio = Median([.. usable.Select(tile => (double)selector(tile) / tile.DoodadPlacements)]);
        int exact = usable.Count(tile => selector(tile) == tile.DoodadPlacements);

        return new Pm4DoodadSeparatorFit(name, usable.Count, meanRatio, medianRatio, exact);
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
            return 0d;

        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2d;
    }

    private static double Fraction(int numerator, int denominator)
        => denominator == 0 ? 0d : (double)numerator / denominator;
}
