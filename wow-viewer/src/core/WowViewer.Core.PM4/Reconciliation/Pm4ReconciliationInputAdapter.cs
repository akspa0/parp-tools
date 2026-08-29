using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Matching;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Reconciliation;

/// <summary>
/// The reconciliation input adapter (Spec 176 Phase 1 step 2). It consumes the existing PM4
/// segment/match results and <see cref="AdtPlacementCatalog"/> records and produces the immutable
/// guide observations, placement snapshots, and candidate lists the proposal engine consumes. It
/// reads no files itself and introduces no second coordinate convention: the world-to-placement
/// composition here is exactly the one documented on
/// <see cref="Pm4CoordinateService.Pm4LocalToAdtPlacement(Vector3)"/>.
/// </summary>
public static class Pm4ReconciliationInputAdapter
{
    /// <summary>
    /// Tolerance applied to the guide geometry's Z span when deciding whether an MSUR
    /// <c>_0x1C</c> value is a plausible placement-height signal rather than unrelated bits.
    /// </summary>
    public const float HeightSignalWindowTolerance = 50f;

    /// <summary>The half-extent of the fallback box used for M2 self-corpus references.</summary>
    public const float M2FallbackBoundsHalfExtent = 2f;

    /// <summary>
    /// Converts a point from the intermediate PM4 "world" space produced by
    /// <c>Pm4PlacementMath.ConvertPm4VertexToWorld</c> (and therefore by
    /// <see cref="Pm4CorrelationObjectState"/>) into ADT placement space. This is the canonical
    /// viewer composition <c>(MapOrigin - world.Y, MapOrigin - world.X, world.Z)</c>, which the
    /// coordinate service documents as composing exactly into
    /// <c>(MapOrigin - MSVT.X, MapOrigin - MSVT.Y, MSVT.Z)</c>.
    /// </summary>
    public static Vector3 WorldToPlacementSpace(Vector3 world)
        => new(
            Pm4CoordinateService.MapOrigin - world.Y,
            Pm4CoordinateService.MapOrigin - world.X,
            world.Z);

    /// <summary>
    /// Builds one guide observation per scored PM4 segment. Positions, bounds, and footprints are
    /// converted into placement space; the height signal is the median MSUR <c>_0x1C</c> value that
    /// is finite and plausible against the segment's own Z span (Spec 185 field evidence), or null
    /// when no surface carries a usable value.
    /// </summary>
    public static IReadOnlyList<Pm4GuideObservation> BuildGuideObservations(
        IReadOnlyList<Pm4SegmentMatchResult> matchResults,
        string sourcePath,
        string buildFingerprint,
        string mapName,
        int tileX,
        int tileY)
    {
        ArgumentNullException.ThrowIfNull(matchResults);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var observations = new List<Pm4GuideObservation>(matchResults.Count);
        foreach (Pm4SegmentMatchResult match in matchResults)
        {
            Pm4BuiltObjectSegment segment = match.Segment;
            Pm4CorrelationObjectState state = segment.CorrelationState;

            // The reflection about the map centre reverses axis ordering, so recompute min/max per
            // axis after converting both corners.
            Vector3 convertedMin = WorldToPlacementSpace(state.BoundsMin);
            Vector3 convertedMax = WorldToPlacementSpace(state.BoundsMax);
            Vector3 boundsMin = Vector3.Min(convertedMin, convertedMax);
            Vector3 boundsMax = Vector3.Max(convertedMin, convertedMax);
            Vector3 position = WorldToPlacementSpace(state.Center);

            var footprint = new List<Vector3>(state.FootprintHull.Count);
            foreach (Vector2 hullPoint in state.FootprintHull)
            {
                // Footprint hull points are planar world (X, Y); Z carries no information here.
                footprint.Add(new Vector3(
                    Pm4CoordinateService.MapOrigin - hullPoint.Y,
                    Pm4CoordinateService.MapOrigin - hullPoint.X,
                    0f));
            }

            double? heightSignal = TryResolveHeightSignal(segment, state.BoundsMin.Z, state.BoundsMax.Z);
            ExpectedAssetKind expectedKind = MapExpectedKind(match.ExpectedAssetKind);

            var evidence = new List<ReconciliationEvidence>
            {
                new("pm4-segment-surfaces", segment.Segment.SurfaceCount, $"segment '{segment.Segment.SegmentId}' MSUR surface count"),
                new("pm4-segment-index-count", segment.Segment.TotalIndexCount, "segment MSVI index count"),
                new("pm4-ck24", segment.Segment.Ck24, "segment CK24 identifier (0 = zero-CK24 seed)"),
                new("pm4-expected-kind", (int)expectedKind, $"scorer expected asset kind '{match.ExpectedAssetKind ?? "unknown"}'"),
            };

            if (heightSignal is not null)
                evidence.Add(new ReconciliationEvidence("pm4-height-signal", heightSignal.Value, "median MSUR _0x1C placement-height (Spec 185) inside the segment Z span"));

            foreach (Pm4SegmentConfidenceFlags flag in Enum.GetValues<Pm4SegmentConfidenceFlags>())
            {
                if (flag != Pm4SegmentConfidenceFlags.None && segment.Segment.ConfidenceFlags.HasFlag(flag))
                    evidence.Add(new ReconciliationEvidence($"pm4-confidence-{flag}", 1.0, "segment confidence flag from the existing segment builder"));
            }

            var identity = new Pm4GuideIdentity(
                segment.Segment.SegmentId,
                sourcePath,
                buildFingerprint,
                mapName,
                tileX,
                tileY,
                segment.Segment.Ck24,
                segment.Segment.Ck24ObjectId,
                GeometryFingerprint: null);

            observations.Add(new Pm4GuideObservation(
                identity,
                position,
                boundsMin,
                boundsMax,
                footprint,
                heightSignal,
                expectedKind,
                segment.Signal.SignalVersion,
                evidence));
        }

        return observations;
    }

    /// <summary>
    /// Captures every MDDF/MODF row of a loaded placement catalog as an immutable snapshot. Entry
    /// indices are the catalog list positions, which is what <c>AdtPlacementEditor</c> resolves.
    /// </summary>
    public static IReadOnlyList<PlacementSnapshot> BuildPlacementSnapshots(
        AdtPlacementCatalog catalog,
        string mapName,
        int tileX,
        int tileY,
        string buildFingerprint)
    {
        ArgumentNullException.ThrowIfNull(catalog);

        var snapshots = new List<PlacementSnapshot>(catalog.ModelPlacements.Count + catalog.WorldModelPlacements.Count);

        for (int index = 0; index < catalog.ModelPlacements.Count; index++)
        {
            AdtModelPlacement placement = catalog.ModelPlacements[index];
            snapshots.Add(new PlacementSnapshot(
                new PlacementIdentity(
                    catalog.SourcePath,
                    mapName,
                    tileX,
                    tileY,
                    ExpectedAssetKind.Model,
                    index,
                    placement.UniqueId,
                    placement.ModelPath,
                    buildFingerprint),
                placement.Position,
                placement.Rotation,
                placement.Scale));
        }

        for (int index = 0; index < catalog.WorldModelPlacements.Count; index++)
        {
            AdtWorldModelPlacement placement = catalog.WorldModelPlacements[index];
            snapshots.Add(new PlacementSnapshot(
                new PlacementIdentity(
                    catalog.SourcePath,
                    mapName,
                    tileX,
                    tileY,
                    ExpectedAssetKind.WorldModel,
                    index,
                    placement.UniqueId,
                    placement.ModelPath,
                    buildFingerprint),
                placement.Position,
                placement.Rotation,
                Scale: 1f));
        }

        return snapshots;
    }

    /// <summary>
    /// Maps the existing scorer's ranked candidates into reconciliation candidates keyed by guide
    /// (segment) id. Score breakdowns, rationale, and status are preserved one-to-one so the
    /// engine's ambiguity semantics operate on the scorer's own numbers.
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyList<ReconciliationCandidate>> BuildCandidatesByGuideId(
        IReadOnlyList<Pm4SegmentMatchResult> matchResults)
    {
        ArgumentNullException.ThrowIfNull(matchResults);

        var byGuideId = new Dictionary<string, IReadOnlyList<ReconciliationCandidate>>(StringComparer.Ordinal);
        foreach (Pm4SegmentMatchResult match in matchResults)
        {
            var candidates = new List<ReconciliationCandidate>(match.Candidates.Count);
            foreach (Pm4AssetMatchCandidate candidate in match.Candidates)
            {
                candidates.Add(new ReconciliationCandidate(
                    candidate.AssetId,
                    candidate.AssetPath,
                    MapExpectedKind(candidate.AssetKind),
                    candidate.Rank,
                    candidate.OverallScore,
                    candidate.ScoreBreakdown,
                    candidate.Rationale,
                    MapCandidateStatus(candidate.Status)));
            }

            byGuideId[match.Segment.Segment.SegmentId] = candidates;
        }

        return byGuideId;
    }

    /// <summary>
    /// Builds asset reference signals from the Museum catalog's own placements — a deliberately
    /// labelled "self corpus" that needs no client archive access. WMO references use the MODF
    /// bounds (already placement space); M2 references use a small fallback box around the
    /// placement position, exactly as the inspect tool's fallback does, and the validation tags
    /// report which of the two was used.
    /// </summary>
    public static IReadOnlyList<Pm4AssetReferenceSignalRecord> BuildSelfCorpusReferences(
        AdtPlacementCatalog catalog,
        string tileCoordinate,
        string buildLabel)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentException.ThrowIfNullOrWhiteSpace(tileCoordinate);

        var assets = new List<Pm4AssetReferenceSignalRecord>(catalog.WorldModelPlacements.Count + catalog.ModelPlacements.Count);

        foreach (AdtWorldModelPlacement placement in catalog.WorldModelPlacements)
        {
            Vector3 boundsMin = placement.BoundsMin;
            Vector3 boundsMax = placement.BoundsMax;
            Vector2[] footprintHull = BuildAabbFootprintHull(boundsMin, boundsMax);

            assets.Add(new Pm4AssetReferenceSignalRecord(
                $"wmo:{placement.UniqueId}",
                placement.ModelPath,
                "wmo",
                buildLabel,
                [tileCoordinate],
                new Pm4Bounds3(boundsMin, boundsMax),
                (boundsMin + boundsMax) * 0.5f,
                footprintHull,
                Pm4CorrelationMath.ComputeFootprintArea(footprintHull),
                placement.Position,
                placement.Rotation,
                1f,
                new Dictionary<string, int>(StringComparer.Ordinal)
                {
                    ["assetKind:wmo"] = 1,
                    ["geometry:resolved"] = 0,
                },
                new Dictionary<string, double>(StringComparer.Ordinal)
                {
                    ["boundsSpanX"] = boundsMax.X - boundsMin.X,
                    ["boundsSpanY"] = boundsMax.Y - boundsMin.Y,
                    ["boundsSpanZ"] = boundsMax.Z - boundsMin.Z,
                },
                Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
                SignalStoreRow: null,
                ["museum-self-corpus", "fallback-placement-bounds"]));
        }

        foreach (AdtModelPlacement placement in catalog.ModelPlacements)
        {
            Vector3 boundsMin = placement.Position - new Vector3(M2FallbackBoundsHalfExtent);
            Vector3 boundsMax = placement.Position + new Vector3(M2FallbackBoundsHalfExtent);
            Vector2[] footprintHull = BuildAabbFootprintHull(boundsMin, boundsMax);

            assets.Add(new Pm4AssetReferenceSignalRecord(
                $"m2:{placement.UniqueId}",
                placement.ModelPath,
                "m2",
                buildLabel,
                [tileCoordinate],
                new Pm4Bounds3(boundsMin, boundsMax),
                placement.Position,
                footprintHull,
                Pm4CorrelationMath.ComputeFootprintArea(footprintHull),
                placement.Position,
                placement.Rotation,
                placement.Scale,
                new Dictionary<string, int>(StringComparer.Ordinal)
                {
                    ["assetKind:m2"] = 1,
                    ["geometry:resolved"] = 0,
                },
                new Dictionary<string, double>(StringComparer.Ordinal)
                {
                    ["boundsSpanX"] = boundsMax.X - boundsMin.X,
                    ["boundsSpanY"] = boundsMax.Y - boundsMin.Y,
                    ["boundsSpanZ"] = boundsMax.Z - boundsMin.Z,
                },
                Pm4AssetMatchScorer.CurrentReferenceSignalVersion,
                SignalStoreRow: null,
                ["museum-self-corpus", "fallback-placement-bounds"]));
        }

        return assets;
    }

    /// <summary>
    /// Builds reference signal records from a complete <see cref="RosettaReferenceLibrary"/>
    /// for consumption by the reconciliation proposal engine (Spec 190 Phase 3).
    /// </summary>
    public static IReadOnlyList<Pm4AssetReferenceSignalRecord> BuildRosettaCorpusReferences(
        RosettaReferenceLibrary library)
    {
        ArgumentNullException.ThrowIfNull(library);
        return library.Assets
            .Select(static asset => asset.ToAssetReferenceSignalRecord())
            .ToList();
    }

    private static double? TryResolveHeightSignal(Pm4BuiltObjectSegment segment, float worldMinZ, float worldMaxZ)
    {
        // MSUR _0x1C is a float bit pattern (Spec 185: equal to the producing placement's
        // Position.Z for 88.84% of WMO objects). Values outside the segment's own Z span are
        // treated as unrelated bits rather than guessed heights. Z is unaffected by the
        // world-to-placement reflection, so the world-space span applies directly.
        var plausible = new List<float>();
        foreach (Pm4ObjectSegmentSurface surface in segment.Surfaces)
        {
            float height = BitConverter.UInt32BitsToSingle(surface.PackedParams);
            if (!float.IsFinite(height))
                continue;

            if (height < worldMinZ - HeightSignalWindowTolerance || height > worldMaxZ + HeightSignalWindowTolerance)
                continue;

            plausible.Add(height);
        }

        if (plausible.Count == 0)
            return null;

        plausible.Sort();
        return plausible[(plausible.Count - 1) / 2];
    }

    private static ExpectedAssetKind MapExpectedKind(string? assetKind)
        => assetKind is null ? ExpectedAssetKind.Unknown
        : string.Equals(assetKind, "wmo", StringComparison.OrdinalIgnoreCase) ? ExpectedAssetKind.WorldModel
        : string.Equals(assetKind, "m2", StringComparison.OrdinalIgnoreCase) ? ExpectedAssetKind.Model
        : ExpectedAssetKind.Unknown;

    private static CandidateStatus MapCandidateStatus(Pm4AssetMatchStatus status)
        => status switch
        {
            Pm4AssetMatchStatus.Matched => CandidateStatus.Matched,
            Pm4AssetMatchStatus.Ambiguous => CandidateStatus.Ambiguous,
            Pm4AssetMatchStatus.Ineligible => CandidateStatus.Ineligible,
            _ => CandidateStatus.Unresolved,
        };

    private static Vector2[] BuildAabbFootprintHull(Vector3 boundsMin, Vector3 boundsMax)
    {
        return
        [
            new Vector2(boundsMin.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMin.Y),
            new Vector2(boundsMax.X, boundsMax.Y),
            new Vector2(boundsMin.X, boundsMax.Y),
        ];
    }
}
