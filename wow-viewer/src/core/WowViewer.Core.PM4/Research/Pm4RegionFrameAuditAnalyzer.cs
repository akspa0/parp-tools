using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Groups PM4 geometry by <c>MSHD.Field04</c> (region) and asks whether each region's frame agrees
/// with the canonical one, so that "coordinate frames are region-scoped" can be confirmed or killed.
/// </summary>
/// <remarks>
/// <para><b>What this measures, and why it is not the obvious thing.</b> The obvious `--by-region`
/// audit — group raw MSVT bounds by region and check each against the tile band its filename
/// implies — cannot answer the question, because that test is already saturated: all 309 non-empty
/// files pass it. A detector that every input passes reports "uniform" no matter what is true. So
/// the band check is still emitted (it is the continuity baseline) but the discriminating
/// measurements are the two below, both of which can vary per file.</para>
///
/// <para><b>1. Resolved frame.</b> Placement does not use the canonical transform directly. It runs
/// a per-object fitter — <see cref="Pm4PlacementMath.ResolveCoordinateMode"/> and
/// <see cref="Pm4PlacementMath.ResolvePlacementSolution"/> — that scores candidate coordinate modes
/// and planar transforms against MPRL and can also apply a yaw correction. Every object therefore
/// carries a resolved frame that may or may not equal the canonical one, and the fitter's inputs are
/// per file. Since <c>MSHD.Field04</c> is itself a per-file header value, "objects in one region
/// fail identically" and "objects in one file fail identically" are the SAME observation. This audit
/// reports the resolved frame per object so the two can be told apart: a genuinely region-scoped
/// frame must be constant across the several files that share a region, not merely within one file.
/// </para>
///
/// <para><b>2. Reference agreement.</b> When the caller supplies placement-space reference points
/// (the paired <c>_obj0.adt</c>'s MDDF/MODF positions), each file is scored on how many of them land
/// inside its own MSVT footprint. That is external ground truth rather than a self-consistency
/// check, and it is what established the canonical transform in the first place.</para>
///
/// <para>Reference points are passed in rather than read here because <c>WowViewer.Core.IO</c>
/// already depends on this assembly; reading ADTs from inside it would be a reference cycle.</para>
/// </remarks>
public static class Pm4RegionFrameAuditAnalyzer
{
    /// <summary>The frame the canonical transform corresponds to, in <see cref="DescribeFrame"/> form.</summary>
    public const string CanonicalFrame = "WorldSpace/....";

    public static Pm4RegionFrameAuditReport AnalyzeDirectory(
        string inputDirectory,
        IReadOnlyDictionary<string, IReadOnlyList<Vector2>>? referencePointsByFile = null)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4RegionFrameObjectRecord> objects = [];
        List<Pm4RegionFrameFileRecord> files = [];

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName, StringComparer.Ordinal))
        {
            Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
            if (document.KnownChunks.Msvt.Count == 0)
                continue;

            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileFirst, out int tileSecond))
                continue;

            string fileName = Path.GetFileName(path);
            IReadOnlyList<Vector2>? referencePoints = null;
            referencePointsByFile?.TryGetValue(fileName, out referencePoints);

            files.Add(AnalyzeFile(document, fileName, tileFirst, tileSecond, referencePoints, objects));
        }

        return BuildReport(resolvedDirectory, files, objects, referencePointsByFile is not null);
    }

    private static Pm4RegionFrameFileRecord AnalyzeFile(
        Pm4ResearchDocument document,
        string fileName,
        int tileFirst,
        int tileSecond,
        IReadOnlyList<Vector2>? referencePoints,
        List<Pm4RegionFrameObjectRecord> objects)
    {
        Pm4KnownChunkSet chunks = document.KnownChunks;
        IReadOnlyList<Vector3> vertices = chunks.Msvt;
        uint regionId = chunks.Mshd?.Field04 ?? 0u;

        // Canonical placement bounds — the transform verified against ADT ground truth.
        float minX = float.MaxValue, maxX = float.MinValue;
        float minY = float.MaxValue, maxY = float.MinValue;
        float rawMinX = float.MaxValue, rawMaxX = float.MinValue;
        float rawMinY = float.MaxValue, rawMaxY = float.MinValue;

        foreach (Vector3 vertex in vertices)
        {
            Vector3 placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(vertex);
            minX = Math.Min(minX, placement.X);
            maxX = Math.Max(maxX, placement.X);
            minY = Math.Min(minY, placement.Y);
            maxY = Math.Max(maxY, placement.Y);

            rawMinX = Math.Min(rawMinX, vertex.X);
            rawMaxX = Math.Max(rawMaxX, vertex.X);
            rawMinY = Math.Min(rawMinY, vertex.Y);
            rawMaxY = Math.Max(rawMaxY, vertex.Y);
        }

        // The continuity baseline: does the raw band still match the filename? Tolerance absorbs
        // the float boundary case where a value sits exactly on a band edge.
        const float bandTolerance = 1f;
        bool bandFits =
            rawMinX >= tileSecond * Pm4CoordinateService.TileSize - bandTolerance
            && rawMaxX <= (tileSecond + 1) * Pm4CoordinateService.TileSize + bandTolerance
            && rawMinY >= tileFirst * Pm4CoordinateService.TileSize - bandTolerance
            && rawMaxY <= (tileFirst + 1) * Pm4CoordinateService.TileSize + bandTolerance;

        int referenceInside = 0;
        if (referencePoints is { Count: > 0 })
        {
            const float referenceTolerance = 5f;
            foreach (Vector2 point in referencePoints)
            {
                if (point.X >= minX - referenceTolerance && point.X <= maxX + referenceTolerance
                    && point.Y >= minY - referenceTolerance && point.Y <= maxY + referenceTolerance)
                {
                    referenceInside++;
                }
            }
        }

        int objectsBefore = objects.Count;
        int objectsOffCanonical = AnalyzeObjects(document, fileName, tileFirst, tileSecond, regionId, objects);

        return new Pm4RegionFrameFileRecord(
            fileName,
            tileFirst,
            tileSecond,
            regionId,
            vertices.Count,
            minX, maxX, minY, maxY,
            bandFits,
            objects.Count - objectsBefore,
            objectsOffCanonical,
            referencePoints?.Count ?? 0,
            referenceInside,
            referencePoints is { Count: > 0 } ? (double)referenceInside / referencePoints.Count : 0d);
    }

    /// <summary>
    /// Runs the real placement fitter over each CK24 object, exactly as the viewer does, and records
    /// the frame it resolves plus how far that moves the object off the canonical position.
    /// </summary>
    private static int AnalyzeObjects(
        Pm4ResearchDocument document,
        string fileName,
        int tileFirst,
        int tileSecond,
        uint regionId,
        List<Pm4RegionFrameObjectRecord> objects)
    {
        Pm4KnownChunkSet chunks = document.KnownChunks;
        IReadOnlyList<Vector3> vertices = chunks.Msvt;
        IReadOnlyList<uint> indices = chunks.Msvi;
        IReadOnlyList<Pm4MprlEntry> positionRefs = chunks.Mprl;

        if (chunks.Msur.Count == 0 || indices.Count == 0)
            return 0;

        Pm4AxisConvention axisConvention =
            Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(vertices, indices, chunks.Msur);
        Pm4CoordinateMode fallbackMode = Pm4PlacementMath.IsLikelyTileLocal(vertices)
            ? Pm4CoordinateMode.TileLocal
            : Pm4CoordinateMode.WorldSpace;

        int offCanonical = 0;

        foreach (IGrouping<uint, Pm4MsurEntry> group in chunks.Msur.GroupBy(static surface => surface.Ck24))
        {
            List<Pm4MsurEntry> surfaces = [.. group];

            Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
                vertices, indices, surfaces, positionRefs, anchorPositionRefs: null,
                tileFirst, tileSecond, axisConvention, fallbackMode);

            Pm4PlacementSolution solution = Pm4PlacementMath.ResolvePlacementSolution(
                vertices, indices, surfaces, positionRefs, anchorPositionRefs: null,
                tileFirst, tileSecond, resolution.CoordinateMode, axisConvention);

            List<Vector3> objectVertices = Pm4PlacementMath.CollectSurfaceVertices(vertices, indices, surfaces);
            if (objectVertices.Count == 0)
                continue;

            Vector3 centroid = Vector3.Zero;
            foreach (Vector3 vertex in objectVertices)
                centroid += vertex;
            centroid /= objectVertices.Count;

            Vector3 canonical = Pm4CoordinateService.Pm4LocalToAdtPlacement(centroid);
            Vector3 resolved = ToPlacementSpace(Pm4PlacementMath.ConvertPm4VertexToWorld(centroid, solution));

            int canonicalTileX = Pm4CoordinateService.PlacementCoordinateToTileIndex(canonical.X);
            int canonicalTileY = Pm4CoordinateService.PlacementCoordinateToTileIndex(canonical.Y);
            int resolvedTileX = Pm4CoordinateService.PlacementCoordinateToTileIndex(resolved.X);
            int resolvedTileY = Pm4CoordinateService.PlacementCoordinateToTileIndex(resolved.Y);

            string frame = DescribeFrame(resolution.CoordinateMode, resolution.PlanarTransform);
            bool onCanonicalFrame = frame == CanonicalFrame
                && MathF.Abs(solution.WorldYawCorrectionRadians) < 1e-6f;
            if (!onCanonicalFrame)
                offCanonical++;

            objects.Add(new Pm4RegionFrameObjectRecord(
                fileName,
                tileFirst,
                tileSecond,
                regionId,
                group.Key,
                surfaces.Count,
                frame,
                onCanonicalFrame,
                canonicalTileX,
                canonicalTileY,
                resolvedTileX,
                resolvedTileY,
                resolvedTileX - canonicalTileX,
                resolvedTileY - canonicalTileY,
                solution.WorldYawCorrectionRadians * (180f / MathF.PI)));
        }

        return offCanonical;
    }

    /// <summary>
    /// Finishes <see cref="Pm4PlacementMath.ConvertPm4VertexToWorld"/>'s intermediate space into
    /// placement space, applying the same step the viewer's world-to-renderer conversion does.
    /// </summary>
    private static Vector3 ToPlacementSpace(Vector3 intermediate)
        => new(
            Pm4CoordinateService.MapOrigin - intermediate.Y,
            Pm4CoordinateService.MapOrigin - intermediate.X,
            intermediate.Z);

    /// <summary>Renders a resolved frame as a short stable token, e.g. <c>TileLocal/.UV</c>.</summary>
    public static string DescribeFrame(Pm4CoordinateMode mode, Pm4PlanarTransform transform)
    {
        char swap = transform.SwapPlanarAxes ? 'S' : '.';
        char invertU = transform.InvertU ? 'U' : '.';
        char invertV = transform.InvertV ? 'V' : '.';
        return $"{mode}/{swap}{invertU}{invertV}.";
    }

    private static Pm4RegionFrameAuditReport BuildReport(
        string resolvedDirectory,
        List<Pm4RegionFrameFileRecord> files,
        List<Pm4RegionFrameObjectRecord> objects,
        bool hasReference)
    {
        List<Pm4RegionFrameSummary> regions = [];

        foreach (IGrouping<uint, Pm4RegionFrameFileRecord> regionFiles in files
            .GroupBy(static file => file.RegionId)
            .OrderBy(static group => group.Key))
        {
            uint regionId = regionFiles.Key;
            List<Pm4RegionFrameObjectRecord> regionObjects =
                [.. objects.Where(record => record.RegionId == regionId)];

            IReadOnlyList<Pm4FrameFamilyCount> frames = CountFrames(regionObjects);
            IReadOnlyList<Pm4TileOffsetFamilyCount> offsets = CountOffsets(regionObjects);

            int referenceTotal = regionFiles.Sum(static file => file.ReferencePlacements);
            int referenceInside = regionFiles.Sum(static file => file.ReferencePlacementsInside);

            regions.Add(new Pm4RegionFrameSummary(
                regionId,
                IsSharedBucket: regionId == 0u,
                IsEmptyStubRegion: regionId == 1u,
                regionFiles.Count(),
                regionFiles.Sum(static file => file.VertexCount),
                [.. regionFiles.Select(static file => file.FileName).OrderBy(static name => name, StringComparer.Ordinal)],
                regionObjects.Count,
                frames,
                frames.Count <= 1,
                offsets,
                offsets.Count <= 1,
                regionFiles.Count(static file => !file.RawBandMatchesFileName),
                referenceTotal,
                referenceInside,
                referenceTotal == 0 ? 0d : (double)referenceInside / referenceTotal));
        }

        long referencePlacements = files.Sum(static file => (long)file.ReferencePlacements);
        long referenceInsideTotal = files.Sum(static file => (long)file.ReferencePlacementsInside);

        int multiFileRegions = regions.Count(static region => region.FileCount > 1);
        int multiFileRegionsMixed = regions.Count(static region => region.FileCount > 1 && !region.IsFrameHomogeneous);

        List<string> notes =
        [
            "Region is MSHD.Field04, a PER-FILE header value. 'Objects in one region behave alike' and "
                + "'objects in one file behave alike' are therefore the same observation for any "
                + "single-file region — only regions spanning several files can distinguish them.",
            $"Regions spanning more than one file: {multiFileRegions}; of those, mixed-frame: {multiFileRegionsMixed}. "
                + "A region-scoped frame requires this to be 0 AND the frames to differ between regions.",
            "The raw-band check against the filename is the continuity baseline and is expected to pass "
                + "everywhere; it cannot discriminate, because a reflection about the map centre maps a "
                + "band onto a band. It is reported so a regression in it is still visible.",
            "Canonical frame is " + CanonicalFrame + " with zero yaw correction — the composition that "
                + "reproduces ADT MDDF/MODF placements. Any other resolved frame is the per-object "
                + "fitter overriding it, not a property of how the file stores coordinates.",
            hasReference
                ? "Reference agreement is measured against real ADT placements supplied by the caller."
                : "No ADT reference points supplied — pass --placements to score against real placements."
        ];

        return new Pm4RegionFrameAuditReport(
            resolvedDirectory,
            files.Count,
            objects.Count,
            regions.Count,
            multiFileRegions,
            multiFileRegionsMixed,
            objects.Count(static record => record.MatchesCanonicalFrame),
            objects.Count(static record => !record.MatchesCanonicalFrame),
            files.Count(static file => !file.RawBandMatchesFileName),
            CountFrames(objects),
            CountOffsets(objects),
            referencePlacements,
            referenceInsideTotal,
            referencePlacements == 0 ? 0d : (double)referenceInsideTotal / referencePlacements,
            regions,
            files,
            objects,
            notes);
    }

    private static IReadOnlyList<Pm4FrameFamilyCount> CountFrames(IReadOnlyList<Pm4RegionFrameObjectRecord> records)
        => [.. records
            .GroupBy(static record => record.ResolvedFrame, StringComparer.Ordinal)
            .Select(static group => new Pm4FrameFamilyCount(group.Key, group.Count()))
            .OrderByDescending(static family => family.ObjectCount)
            .ThenBy(static family => family.Frame, StringComparer.Ordinal)];

    private static IReadOnlyList<Pm4TileOffsetFamilyCount> CountOffsets(IReadOnlyList<Pm4RegionFrameObjectRecord> records)
        => [.. records
            .GroupBy(static record => (record.TileOffsetX, record.TileOffsetY))
            .Select(static group => new Pm4TileOffsetFamilyCount(group.Key.TileOffsetX, group.Key.TileOffsetY, group.Count()))
            .OrderByDescending(static family => family.ObjectCount)
            .ThenBy(static family => family.OffsetX)
            .ThenBy(static family => family.OffsetY)];
}
