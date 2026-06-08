using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4ObjectSegmentBuilder
{
    public static IReadOnlyList<Pm4BuiltObjectSegment> Build(string pm4Path)
    {
        if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
            throw new InvalidOperationException($"Could not parse tile coordinates from '{pm4Path}'.");

        return Build(Pm4ResearchReader.ReadFile(pm4Path), tileX, tileY);
    }

    public static IReadOnlyList<Pm4BuiltObjectSegment> Build(Pm4ResearchDocument document, int tileX, int tileY)
    {
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        List<IndexedSurface> indexedSurfaces = document.KnownChunks.Msur
            .Select(static (surface, i) => new IndexedSurface(i, surface))
            .Where(static indexed => indexed.Surface.IndexCount >= 3)
            .ToList();
        if (indexedSurfaces.Count == 0)
            return [];

        uint? field04 = document.KnownChunks.Mshd?.Field04;
        string tileCoordinate = $"{tileX}_{tileY}";

        // Detect reused low16 CK24 object IDs across different CK24 families
        HashSet<ushort> reusedLow16ObjectIds = document.KnownChunks.Msur
            .Where(static surface => surface.Ck24 != 0u && surface.Ck24ObjectId != 0)
            .GroupBy(static surface => surface.Ck24ObjectId)
            .Where(static group => group.Select(static surface => surface.Ck24).Distinct().Count() > 1)
            .Select(static group => group.Key)
            .ToHashSet();

        // Build TypeFlags index: RefIndex -> TypeFlags
        Dictionary<int, byte> typeFlagsByRefIndex = [];
        foreach (Pm4MslkEntry link in document.KnownChunks.Mslk)
        {
            if (!typeFlagsByRefIndex.ContainsKey(link.RefIndex))
                typeFlagsByRefIndex[link.RefIndex] = link.TypeFlags;
        }

        // Group by CK24 (non-zero) or (GroupKey, AttributeMask) (zero-CK24)
        List<Ck24Group> ck24Groups = [];
        foreach (IGrouping<uint, IndexedSurface> ck24Group in indexedSurfaces
            .Where(static s => s.Surface.Ck24 != 0u)
            .GroupBy(static s => s.Surface.Ck24)
            .OrderBy(static g => g.Key))
        {
            ck24Groups.Add(BuildCk24Group(ck24Group, typeFlagsByRefIndex, false));
        }
        foreach (IGrouping<(byte GroupKey, byte AttributeMask), IndexedSurface> zeroGroup in indexedSurfaces
            .Where(static s => s.Surface.Ck24 == 0u)
            .GroupBy(static s => (s.Surface.GroupKey, s.Surface.AttributeMask))
            .OrderBy(static g => g.Key.GroupKey).ThenBy(static g => g.Key.AttributeMask))
        {
            ck24Groups.Add(BuildCk24Group(zeroGroup, typeFlagsByRefIndex, true));
        }

        bool fallbackTileLocal = Pm4PlacementMath.IsLikelyTileLocal(meshVertices);
        List<Pm4BuiltObjectSegment> builtSegments = [];
        foreach (Ck24Group group in ck24Groups)
        {
            IndexedSurface exemplar = group.Surfaces[0];
            List<Pm4MsurEntry> groupSurfaces = group.Surfaces.Select(static s => s.Surface).ToList();
            List<IndexedSurface> groupIndexed = group.Surfaces;
            uint ck24 = exemplar.Surface.Ck24 != 0u ? exemplar.Surface.Ck24 : 0u;
            byte ck24Type = exemplar.Surface.Ck24Type;
            ushort ck24ObjectId = exemplar.Surface.Ck24ObjectId;

            Pm4AxisConvention axisConvention = Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(meshVertices, meshIndices, groupSurfaces);
            List<Pm4MprlEntry> positionRefs = CollectLinkedPositionRefs(document, groupIndexed);
            Pm4CoordinateMode fallbackMode = fallbackTileLocal ? Pm4CoordinateMode.TileLocal : Pm4CoordinateMode.WorldSpace;
            Pm4CoordinateModeResolution coordMode = Pm4PlacementMath.ResolveCoordinateMode(meshVertices, meshIndices, groupSurfaces, positionRefs, positionRefs, tileX, tileY, axisConvention, fallbackMode);
            Pm4PlacementSolution placement = Pm4PlacementMath.ResolvePlacementSolution(meshVertices, meshIndices, groupSurfaces, positionRefs, positionRefs, tileX, tileY, coordMode.CoordinateMode, axisConvention);

            // Collect vertices and convert to world space
            List<Vector3> vertices = Pm4PlacementMath.CollectSurfaceVertices(meshVertices, meshIndices, groupSurfaces);
            List<Vector3> worldPoints = vertices.Select(v => Pm4PlacementMath.ConvertPm4VertexToWorld(v, placement)).ToList();

            // Build typed surface groups
            Dictionary<byte, TypedSurfaceGroup> typedGroups = [];
            foreach (IndexedSurface surface in group.Surfaces)
            {
                byte typeFlags = typeFlagsByRefIndex.GetValueOrDefault(surface.SurfaceIndex, (byte)0);
                if (!typedGroups.TryGetValue(typeFlags, out TypedSurfaceGroup? typed))
                {
                    typed = new TypedSurfaceGroup(typeFlags, []);
                    typedGroups[typeFlags] = typed;
                }
                typed.SurfaceIndices.Add(surface.SurfaceIndex);
            }

            IReadOnlyList<uint> linkGroupIds = CollectDistinctLinkGroupIds(document, groupIndexed);
            uint dominantLinkGroupObjectId = SelectDominantGroupObjectId(document, groupIndexed);
            IReadOnlyList<Pm4MprlEntry> componentRefs = CollectLinkedPositionRefs(document, groupIndexed);
            IReadOnlyList<Pm4ObjectSegmentSurface> segmentSurfaces = groupIndexed
                .Select(static s => new Pm4ObjectSegmentSurface(
                    s.SurfaceIndex, s.Surface.GroupKey, s.Surface.AttributeMask,
                    s.Surface.IndexCount, s.Surface.Height, s.Surface.MsviFirstIndex,
                    s.Surface.MscnRefIndex, s.Surface._0x1C, s.Surface.Ck24,
                    s.Surface.Ck24Type, s.Surface.Ck24ObjectId, s.Surface.Normal))
                .ToList();
            Pm4LinkedPositionRefSummary anchorSummary = Pm4PlacementMath.SummarizeLinkedPositionRefs(componentRefs);

            // Compute per-TypeFlags-class bounds
            Dictionary<byte, Pm4Bounds3> typedBounds = ComputeTypedBounds(groupIndexed, meshVertices, meshIndices, placement, typeFlagsByRefIndex);

            int surfaceCount = groupSurfaces.Count;
            int totalIndexCount = groupSurfaces.Sum(static s => s.IndexCount);

            Pm4SegmentConfidenceFlags confidenceFlags = Pm4SegmentConfidenceFlags.None;
            if (group.IsZeroCk24)
                confidenceFlags |= Pm4SegmentConfidenceFlags.ZeroCk24Seed;
            if (linkGroupIds.Count > 1)
                confidenceFlags |= Pm4SegmentConfidenceFlags.MultipleLinkGroupIds;
            if (linkGroupIds.Count == 0)
                confidenceFlags |= Pm4SegmentConfidenceFlags.HasUnlinkedSurfaces;
            if (componentRefs.Count == 0)
                confidenceFlags |= Pm4SegmentConfidenceFlags.MissingPositionRefs;
            if (!field04.HasValue)
                confidenceFlags |= Pm4SegmentConfidenceFlags.SpansMultipleField04Values;
            if (ck24 != 0u && reusedLow16ObjectIds.Contains(ck24ObjectId))
                confidenceFlags |= Pm4SegmentConfidenceFlags.ReusedLow16ObjectId;

            IReadOnlyList<int> surfaceIndices = groupIndexed.Select(static s => s.SurfaceIndex).ToList();
            string segmentId = BuildSegmentId(tileCoordinate, field04, ck24, ck24Type, ck24ObjectId, surfaceIndices, linkGroupIds);

            Pm4ObjectSegment segment = new(
                segmentId, ck24, ck24Type, ck24ObjectId,
                [tileCoordinate],
                field04.HasValue ? [field04.Value] : [],
                surfaceCount, totalIndexCount,
                linkGroupIds, dominantLinkGroupObjectId,
                confidenceFlags);

            // Build correlation input for signal extraction
            Pm4CorrelationObjectInput correlationInput = new(
                tileX, tileY,
                new Pm4ObjectGroupKey(tileX, tileY, ck24 != 0u ? ck24 : 0x80000000u | (uint)builtSegments.Count),
                new Pm4CorrelationObjectDescriptor(
                    ck24, ck24Type, builtSegments.Count,
                    dominantLinkGroupObjectId, surfaceCount, componentRefs.Count,
                    groupSurfaces[0].GroupKey, groupSurfaces[0].AttributeMask,
                    groupSurfaces[0].MscnRefIndex,
                    (float)groupSurfaces.Average(static s => s.Height)),
                worldPoints, placement.WorldPivot);

            IReadOnlyList<Pm4CorrelationObjectState> states = Pm4CorrelationMath.BuildObjectStates([correlationInput]);
            Pm4CorrelationObjectState state = states[0];
            IReadOnlyList<Vector2> anchorPlanarPoints = componentRefs
                .Select(static r => new Vector2(r.Position.X, r.Position.Z))
                .ToList();

            Pm4SegmentSignalRecord signal = Pm4SegmentSignalExtractor.Extract(segment, state, anchorSummary, segmentSurfaces, typedBounds);

            builtSegments.Add(new Pm4BuiltObjectSegment(
                segment, signal, state, anchorSummary, anchorPlanarPoints,
                segmentSurfaces, placement.CoordinateMode, axisConvention,
                placement.PlanarTransform,
                placement.WorldYawCorrectionRadians * 180f / MathF.PI));
        }

        return builtSegments;
    }

    private static Ck24Group BuildCk24Group(IGrouping<uint, IndexedSurface> group, Dictionary<int, byte> typeFlagsByRefIndex, bool isZeroCk24)
    {
        IndexedSurface exemplar = group.First();
        return new Ck24Group(
            exemplar.Surface.Ck24, exemplar.Surface.Ck24Type, exemplar.Surface.Ck24ObjectId,
            isZeroCk24, group.OrderBy(static s => s.SurfaceIndex).ToList());
    }

    private static Ck24Group BuildCk24Group(IGrouping<(byte GroupKey, byte AttributeMask), IndexedSurface> group, Dictionary<int, byte> typeFlagsByRefIndex, bool isZeroCk24)
    {
        return new Ck24Group(0u, 0, 0, true, group.OrderBy(static s => s.SurfaceIndex).ToList());
    }

    private static Dictionary<byte, Pm4Bounds3> ComputeTypedBounds(
        IReadOnlyList<IndexedSurface> surfaces,
        IReadOnlyList<Vector3> meshVertices,
        IReadOnlyList<uint> meshIndices,
        Pm4PlacementSolution placement,
        Dictionary<int, byte> typeFlagsByRefIndex)
    {
        Dictionary<byte, List<Vector3>> verticesByType = [];

        foreach (IndexedSurface surface in surfaces)
        {
            // Get TypeFlags from MSLK entry referencing this surface
            byte typeFlags = typeFlagsByRefIndex.GetValueOrDefault(surface.SurfaceIndex, (byte)0);
            int firstIndex = checked((int)surface.Surface.MsviFirstIndex);
            int endExclusive = Math.Min(firstIndex + surface.Surface.IndexCount, meshIndices.Count);

            if (!verticesByType.TryGetValue(typeFlags, out List<Vector3>? list))
            {
                list = [];
                verticesByType[typeFlags] = list;
            }

            HashSet<int> seen = [];
            for (int i = firstIndex; i < endExclusive; i++)
            {
                int vi = checked((int)meshIndices[i]);
                if ((uint)vi < (uint)meshVertices.Count && seen.Add(vi))
                    list.Add(Pm4PlacementMath.ConvertPm4VertexToWorld(meshVertices[vi], placement));
            }
        }

        Dictionary<byte, Pm4Bounds3> bounds = [];
        foreach (KeyValuePair<byte, List<Vector3>> kv in verticesByType)
        {
            if (kv.Value.Count == 0) continue;
            Vector3 min = new(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new(float.MinValue, float.MinValue, float.MinValue);
            foreach (Vector3 v in kv.Value)
            {
                min = Vector3.Min(min, v);
                max = Vector3.Max(max, v);
            }
            bounds[kv.Key] = new Pm4Bounds3(min, max);
        }

        return bounds;
    }

    private static string BuildSegmentId(
        string tileCoordinate, uint? field04, uint ck24,
        byte ck24Type, ushort ck24ObjectId,
        IReadOnlyList<int> surfaceIndices,
        IReadOnlyList<uint> linkGroupIds)
    {
        string canonical =
            $"tile={tileCoordinate}|field04={(field04.HasValue ? field04.Value.ToString() : "none")}|ck24={ck24:X6}|type={ck24Type:X2}|low16={ck24ObjectId}|surfaces={string.Join(",", surfaceIndices)}|groups={string.Join(",", linkGroupIds)}";
        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(canonical));
        return $"pm4seg-{Convert.ToHexString(hash.AsSpan(0, 8)).ToLowerInvariant()}";
    }

    internal static IReadOnlyList<uint> CollectDistinctLinkGroupIds(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (document.KnownChunks.Mslk.Count == 0 || surfaces.Count == 0)
            return [];

        HashSet<int> surfaceIndices = surfaces.Select(static s => s.SurfaceIndex).ToHashSet();
        return document.KnownChunks.Mslk
            .Where(link => link.GroupObjectId != 0 && surfaceIndices.Contains(link.RefIndex))
            .Select(static link => link.GroupObjectId)
            .Distinct().OrderBy(static id => id).ToList();
    }

    internal static List<Pm4MprlEntry> CollectLinkedPositionRefs(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (document.KnownChunks.Mprl.Count == 0 || document.KnownChunks.Mslk.Count == 0 || surfaces.Count == 0)
            return [];

        HashSet<int> surfaceIndices = surfaces.Select(static s => s.SurfaceIndex).ToHashSet();
        HashSet<int> seen = [];
        List<Pm4MprlEntry> refs = [];
        foreach (Pm4MslkEntry link in document.KnownChunks.Mslk)
        {
            if (surfaceIndices.Contains(link.RefIndex) && (uint)link.RefIndex < (uint)document.KnownChunks.Mprl.Count && seen.Add(link.RefIndex))
                refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
        }
        return refs;
    }

    private static uint SelectDominantGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        HashSet<int> surfaceIndices = surfaces.Select(static s => s.SurfaceIndex).ToHashSet();
        Dictionary<uint, int> counts = [];
        uint best = 0;
        int bestCount = 0;
        foreach (Pm4MslkEntry link in document.KnownChunks.Mslk)
        {
            if (link.GroupObjectId == 0 || !surfaceIndices.Contains(link.RefIndex))
                continue;
            int next = counts.TryGetValue(link.GroupObjectId, out int existing) ? existing + 1 : 1;
            counts[link.GroupObjectId] = next;
            if (next > bestCount) { bestCount = next; best = link.GroupObjectId; }
        }
        return best;
    }

    internal sealed record IndexedSurface(int SurfaceIndex, Pm4MsurEntry Surface);

    private sealed record Ck24Group(
        uint Ck24, byte Ck24Type, ushort Ck24ObjectId,
        bool IsZeroCk24, List<IndexedSurface> Surfaces);

    internal sealed record TypedSurfaceGroup(byte TypeFlags, List<int> SurfaceIndices);
}
