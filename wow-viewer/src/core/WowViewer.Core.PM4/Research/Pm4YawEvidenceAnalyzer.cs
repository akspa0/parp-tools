using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Settles whether the placement pipeline's per-object yaw correction helps or hurts, by scoring
/// each object's geometry against the world bounding box of the WMO placement it sits inside.
/// </summary>
/// <remarks>
/// <para><b>Why a new measurement was needed.</b> The transform that converts MSVT to ADT placement
/// space was established by asking whether MDDF/MODF positions fall inside a PM4's footprint. That
/// test cannot judge the yaw correction, because it compares a point against a bounding box and the
/// correction rotates geometry about its own centroid — which moves neither. MODF is the way out: it
/// carries a world-space bounding box, not just a position, so rotating a non-square object inside
/// it pushes vertices out.</para>
///
/// <para><b>Matching is rotation-invariant on purpose.</b> An object is matched to the MODF box that
/// contains its centroid. A rotation about the centroid leaves the centroid fixed, so the match is
/// decided identically whichever hypothesis is true. Matching on "best containment" instead would
/// have selected for the unrotated reading and then concluded in its favour.</para>
///
/// <para><b>Every object carries its own power check.</b> A bounding box cannot see a rotation when
/// the object is small relative to the box, or when its footprint is near-square. So each object is
/// also scored under a deliberate 45° control rotation, which is known-wrong. If the control does
/// not measurably reduce containment, that box cannot discriminate for that object and it is
/// excluded from the headline and reported separately. Without this, "the yaw makes no difference"
/// would be indistinguishable from "this test cannot tell".</para>
/// </remarks>
public static class Pm4YawEvidenceAnalyzer
{
    /// <summary>Containment drop the 45° control must produce before an object is treated as informative.</summary>
    public const double PowerMarginFraction = 0.02d;

    /// <summary>Difference in containment below which yaw-on and yaw-off are called a tie.</summary>
    public const double DecisionMarginFraction = 0.01d;

    private const float ControlRotationDegrees = 45f;

    public static Pm4YawEvidenceReport AnalyzeDirectory(
        string inputDirectory,
        IReadOnlyDictionary<string, IReadOnlyList<Pm4PlacementBox>> boxesByFile)
    {
        ArgumentNullException.ThrowIfNull(boxesByFile);
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4YawEvidenceObjectRecord> records = [];
        int filesScored = 0;
        int objectsSeen = 0;
        int objectsUnmatched = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName, StringComparer.Ordinal))
        {
            string fileName = Path.GetFileName(path);
            if (!boxesByFile.TryGetValue(fileName, out IReadOnlyList<Pm4PlacementBox>? boxes) || boxes.Count == 0)
                continue;

            Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
            Pm4KnownChunkSet chunks = document.KnownChunks;
            if (chunks.Msvt.Count == 0 || chunks.Msvi.Count == 0 || chunks.Msur.Count == 0)
                continue;

            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileFirst, out int tileSecond))
                continue;

            filesScored++;
            ScoreFile(document, fileName, tileFirst, tileSecond, boxes, records, ref objectsSeen, ref objectsUnmatched);
        }

        return BuildReport(resolvedDirectory, filesScored, objectsSeen, objectsUnmatched, records);
    }

    private static void ScoreFile(
        Pm4ResearchDocument document,
        string fileName,
        int tileFirst,
        int tileSecond,
        IReadOnlyList<Pm4PlacementBox> boxes,
        List<Pm4YawEvidenceObjectRecord> records,
        ref int objectsSeen,
        ref int objectsUnmatched)
    {
        Pm4KnownChunkSet chunks = document.KnownChunks;
        IReadOnlyList<Vector3> vertices = chunks.Msvt;
        IReadOnlyList<uint> indices = chunks.Msvi;

        Pm4AxisConvention axisConvention =
            Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(vertices, indices, chunks.Msur);
        Pm4CoordinateMode fallbackMode = Pm4PlacementMath.IsLikelyTileLocal(vertices)
            ? Pm4CoordinateMode.TileLocal
            : Pm4CoordinateMode.WorldSpace;

        foreach (IGrouping<uint, Pm4MsurEntry> group in chunks.Msur.GroupBy(static surface => surface.Ck24))
        {
            List<Pm4MsurEntry> surfaces = [.. group];
            List<Vector3> objectVertices = Pm4PlacementMath.CollectSurfaceVertices(vertices, indices, surfaces);
            if (objectVertices.Count < 3)
                continue;

            objectsSeen++;

            // Canonical placement-space points, and the centroid the match is decided on.
            List<Vector2> canonical = new(objectVertices.Count);
            Vector2 centroid = Vector2.Zero;
            foreach (Vector3 vertex in objectVertices)
            {
                Vector3 placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(vertex);
                Vector2 point = new(placement.X, placement.Y);
                canonical.Add(point);
                centroid += point;
            }

            centroid /= canonical.Count;

            Pm4PlacementBox? match = FindBoxContaining(boxes, centroid);
            if (match is null)
            {
                objectsUnmatched++;
                continue;
            }

            Pm4PlacementBox box = match.Value;

            Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
                vertices, indices, surfaces, chunks.Mprl, anchorPositionRefs: null,
                tileFirst, tileSecond, axisConvention, fallbackMode);

            Pm4PlacementSolution solution = Pm4PlacementMath.ResolvePlacementSolution(
                vertices, indices, surfaces, chunks.Mprl, anchorPositionRefs: null,
                tileFirst, tileSecond, resolution.CoordinateMode, axisConvention);

            float yawRadians = solution.WorldYawCorrectionRadians;

            Pm4YawDecision decision = Decide(canonical, centroid, yawRadians, box);

            List<Vector2> resolved = new(objectVertices.Count);
            foreach (Vector3 vertex in objectVertices)
            {
                Vector3 world = Pm4PlacementMath.ConvertPm4VertexToWorld(vertex, solution);
                resolved.Add(new Vector2(
                    Pm4CoordinateService.MapOrigin - world.Y,
                    Pm4CoordinateService.MapOrigin - world.X));
            }

            double insideResolved = InsideFraction(resolved, box);

            records.Add(new Pm4YawEvidenceObjectRecord(
                fileName,
                tileFirst,
                tileSecond,
                chunks.Mshd?.Field04 ?? 0u,
                group.Key,
                objectVertices.Count,
                box.ModelPath,
                box.UniqueId,
                yawRadians * (180f / MathF.PI),
                decision.InsideCanonical,
                decision.InsideYawOnly,
                insideResolved,
                decision.InsideControl45,
                decision.HasDiscriminatingPower,
                decision.Verdict));
        }
    }

    /// <summary>
    /// Scores one object's footprint against its box with the yaw off, on, and at a known-wrong
    /// control angle, and turns that into a verdict.
    /// </summary>
    /// <remarks>
    /// Public because it is the whole decision. Reading it in isolation is how you check that the
    /// test can separate a rotation it should catch from one it cannot see, without a corpus.
    /// </remarks>
    public static Pm4YawDecision Decide(
        IReadOnlyList<Vector2> canonicalPoints,
        Vector2 centroid,
        float yawRadians,
        Pm4PlacementBox box)
    {
        ArgumentNullException.ThrowIfNull(canonicalPoints);

        double insideCanonical = InsideFraction(canonicalPoints, box);
        double insideYawOnly = InsideFraction(Rotate(canonicalPoints, centroid, yawRadians), box);
        double insideControl = InsideFraction(
            Rotate(canonicalPoints, centroid, ControlRotationDegrees * MathF.PI / 180f), box);

        bool hasPower = insideCanonical - insideControl >= PowerMarginFraction;
        string verdict = MathF.Abs(yawRadians) < 1e-6f
            ? "no-yaw-applied"
            : !hasPower
                ? "undecidable"
                : insideYawOnly > insideCanonical + DecisionMarginFraction
                    ? "yaw-helps"
                    : insideYawOnly < insideCanonical - DecisionMarginFraction
                        ? "yaw-hurts"
                        : "tie";

        return new Pm4YawDecision(insideCanonical, insideYawOnly, insideControl, hasPower, verdict);
    }

    /// <summary>
    /// The box whose horizontal extent contains the point. When several do, the smallest wins — a
    /// nested WMO is the more specific claim about what the geometry belongs to.
    /// </summary>
    private static Pm4PlacementBox? FindBoxContaining(IReadOnlyList<Pm4PlacementBox> boxes, Vector2 point)
    {
        Pm4PlacementBox? best = null;
        float bestArea = float.MaxValue;

        foreach (Pm4PlacementBox box in boxes)
        {
            if (point.X < box.MinX || point.X > box.MaxX || point.Y < box.MinY || point.Y > box.MaxY)
                continue;

            float area = (box.MaxX - box.MinX) * (box.MaxY - box.MinY);
            if (area < bestArea)
            {
                bestArea = area;
                best = box;
            }
        }

        return best;
    }

    private static List<Vector2> Rotate(IReadOnlyList<Vector2> points, Vector2 pivot, float radians)
    {
        List<Vector2> rotated = new(points.Count);
        float sin = MathF.Sin(radians);
        float cos = MathF.Cos(radians);

        foreach (Vector2 point in points)
        {
            float dx = point.X - pivot.X;
            float dy = point.Y - pivot.Y;
            rotated.Add(new Vector2(pivot.X + dx * cos - dy * sin, pivot.Y + dx * sin + dy * cos));
        }

        return rotated;
    }

    private static double InsideFraction(IReadOnlyList<Vector2> points, Pm4PlacementBox box)
    {
        if (points.Count == 0)
            return 0d;

        int inside = 0;
        foreach (Vector2 point in points)
        {
            if (point.X >= box.MinX && point.X <= box.MaxX && point.Y >= box.MinY && point.Y <= box.MaxY)
                inside++;
        }

        return (double)inside / points.Count;
    }

    private static Pm4YawEvidenceReport BuildReport(
        string resolvedDirectory,
        int filesScored,
        int objectsSeen,
        int objectsUnmatched,
        List<Pm4YawEvidenceObjectRecord> records)
    {
        List<Pm4YawEvidenceObjectRecord> yawed =
            [.. records.Where(static record => MathF.Abs(record.YawCorrectionDegrees) > 1e-4f)];
        List<Pm4YawEvidenceObjectRecord> decidable =
            [.. yawed.Where(static record => record.HasDiscriminatingPower)];

        int helps = decidable.Count(static record => record.Verdict == "yaw-helps");
        int hurts = decidable.Count(static record => record.Verdict == "yaw-hurts");
        int ties = decidable.Count(static record => record.Verdict == "tie");

        double meanCanonical = decidable.Count == 0 ? 0d : decidable.Average(static record => record.InsideCanonical);
        double meanYawOnly = decidable.Count == 0 ? 0d : decidable.Average(static record => record.InsideYawOnly);
        double meanResolved = decidable.Count == 0 ? 0d : decidable.Average(static record => record.InsideResolved);
        double meanControl = decidable.Count == 0 ? 0d : decidable.Average(static record => record.InsideControl45);

        string verdict = decidable.Count == 0
            ? "undecidable — no matched object had a bounding box able to see a rotation"
            : hurts > helps * 2 && hurts >= 10
                ? "yaw correction HURTS — it moves geometry out of the WMO box it belongs in"
                : helps > hurts * 2 && helps >= 10
                    ? "yaw correction HELPS — keep it"
                    : "inconclusive — no clear majority either way";

        List<string> notes =
        [
            "Objects are matched to a MODF box by CENTROID CONTAINMENT, which a rotation about the "
                + "centroid cannot change. Matching on best fit instead would have selected for the "
                + "unrotated reading and then concluded in its favour.",
            $"Each object is also scored under a deliberate {ControlRotationDegrees:F0} degree control "
                + "rotation. An object whose containment the control does not reduce by at least "
                + $"{PowerMarginFraction:P0} cannot discriminate and is excluded from the headline.",
            "Only MODF carries a world bounding box; MDDF doodad placements are points and cannot "
                + "score a rotation, so PM4 objects that are doodad collision are simply unmatched here.",
            "InsideResolved applies the fitter's whole solution (coordinate mode, planar transform and "
                + "yaw). It is reported alongside InsideYawOnly so a mode flip is not mistaken for a "
                + "yaw effect."
        ];

        return new Pm4YawEvidenceReport(
            resolvedDirectory,
            filesScored,
            objectsSeen,
            records.Count,
            objectsUnmatched,
            yawed.Count,
            decidable.Count,
            yawed.Count - decidable.Count,
            helps,
            hurts,
            ties,
            meanCanonical,
            meanYawOnly,
            meanResolved,
            meanControl,
            verdict,
            [.. records.OrderByDescending(static record => record.InsideCanonical - record.InsideYawOnly).Take(24)],
            notes);
    }
}
