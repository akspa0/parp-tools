using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4Ck24ForensicsAnalyzer
{
    private const float BoundsPadding = 32f;

    public static Pm4Ck24ForensicsReport Analyze(Pm4ResearchDocument document, uint ck24)
    {
        IReadOnlyList<Pm4MsurEntry> allSurfaces = document.KnownChunks.Msur;
        List<int> selectedSurfaceIndices = [];
        List<Pm4MsurEntry> selectedSurfaces = [];
        for (int surfaceIndex = 0; surfaceIndex < allSurfaces.Count; surfaceIndex++)
        {
            Pm4MsurEntry surface = allSurfaces[surfaceIndex];
            if (surface.Ck24 != ck24)
                continue;

            selectedSurfaceIndices.Add(surfaceIndex);
            selectedSurfaces.Add(surface);
        }

        if (selectedSurfaces.Count == 0)
            throw new InvalidOperationException($"CK24 0x{ck24:X6} was not found in '{document.SourcePath ?? "<memory>"}'.");

        List<(uint GroupObjectId, List<int> SurfaceIndices)> linkGroupSurfaceSets = BuildLinkGroupSurfaceSets(
            document.KnownChunks.Mslk,
            selectedSurfaceIndices,
            document.KnownChunks.Msur.Count);

        byte ck24Type = selectedSurfaces[0].Ck24Type;
        ushort ck24ObjectId = selectedSurfaces[0].Ck24ObjectId;
        (int? tileX, int? tileY) = TryParseTileCoordinates(document.SourcePath);
        int resolvedTileX = tileX ?? 0;
        int resolvedTileY = tileY ?? 0;
        Pm4ForensicsPlacementComparison overallPlacement = BuildPlacementComparison(
            document,
            selectedSurfaceIndices,
            resolvedTileX,
            resolvedTileY,
            tileX,
            tileY);

        List<Pm4ForensicsLinkGroupReport> linkGroups = [];
        foreach ((uint groupObjectId, List<int> surfaceSet) in linkGroupSurfaceSets)
        {
            linkGroups.Add(BuildLinkGroupReport(
                document,
                groupObjectId,
                surfaceSet,
                resolvedTileX,
                resolvedTileY,
                tileX,
                tileY));
        }

        List<string> notes =
        [
            "Research-only CK24 forensic export. MPRL rows are exposed as terrain/object seam evidence, not confirmed final runtime semantics.",
            "Linked MPRL refs are collected with the same current RefIndex-to-surface gate used by the viewer-side PM4 graph export so the shared export stays comparable to existing field captures."
        ];
        if (tileX is null || tileY is null)
            notes.Add("Tile coordinates could not be parsed from the PM4 source path. Placement comparison used tile (0,0) as a fallback anchor.");
        int zeroGroupSurfaceCount = linkGroupSurfaceSets.Where(static group => group.GroupObjectId == 0).Sum(static group => group.SurfaceIndices.Count);
        if (zeroGroupSurfaceCount > 0)
            notes.Add($"{zeroGroupSurfaceCount} surfaces under this CK24 have no dominant non-zero MSLK.GroupObjectId component and are exported under linkGroupObjectId 0.");

        return new Pm4Ck24ForensicsReport(
            document.SourcePath,
            document.Version,
            ck24,
            ck24Type,
            ck24ObjectId,
            selectedSurfaces.Count,
            selectedSurfaces.Sum(static surface => surface.IndexCount),
            linkGroups.Count,
            selectedSurfaces.Select(static surface => surface.MdosIndex).Distinct().Count(),
            selectedSurfaces.Select(static surface => surface.AttributeMask).Distinct().Count(),
            selectedSurfaces.Select(static surface => surface.GroupKey).Distinct().Count(),
            Pm4TerminologyCatalog.ForCk24Forensics(),
            selectedSurfaces.Select(static surface => surface.AttributeMask).Distinct().Order().ToList(),
            selectedSurfaces.Select(static surface => surface.GroupKey).Distinct().Order().ToList(),
            selectedSurfaces.Select(static surface => surface.MdosIndex).Distinct().Order().ToList(),
            overallPlacement,
            linkGroups,
            notes,
            document.Diagnostics);
    }

    private static Pm4ForensicsLinkGroupReport BuildLinkGroupReport(
        Pm4ResearchDocument document,
        uint groupObjectId,
        IReadOnlyList<int> surfaceIndices,
        int resolvedTileX,
        int resolvedTileY,
        int? tileX,
        int? tileY)
    {
        IReadOnlyList<Pm4MsurEntry> allSurfaces = document.KnownChunks.Msur;
        HashSet<int> surfaceIndexSet = surfaceIndices.ToHashSet();
        List<Pm4MsurEntry> surfaces = surfaceIndices.Select(index => allSurfaces[index]).ToList();
        List<(Pm4MslkEntry Entry, int LinkIndex)> linkRows = [];
        for (int linkIndex = 0; linkIndex < document.KnownChunks.Mslk.Count; linkIndex++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[linkIndex];
            bool referencesSelectedSurface = surfaceIndexSet.Contains(link.RefIndex);
            if (groupObjectId == 0)
            {
                if (!referencesSelectedSurface)
                    continue;
            }
            else if (link.GroupObjectId != groupObjectId)
            {
                continue;
            }

            linkRows.Add((link, linkIndex));
        }

        List<int> linkedPositionRefIndices = linkRows
            .Select(static item => (int)item.Entry.RefIndex)
            .Where(index => (uint)index < (uint)document.KnownChunks.Mprl.Count)
            .Where(surfaceIndexSet.Contains)
            .Distinct()
            .Order()
            .ToList();
        List<Pm4MprlEntry> linkedPositionRefs = linkedPositionRefIndices.Select(index => document.KnownChunks.Mprl[index]).ToList();
        Pm4LinkedPositionRefSummary summary = SanitizeSummary(Pm4PlacementMath.SummarizeLinkedPositionRefs(linkedPositionRefs));
        Pm4ForensicsPlacementComparison placementComparison = BuildPlacementComparison(
            document,
            surfaceIndices,
            resolvedTileX,
            resolvedTileY,
            tileX,
            tileY,
            linkedPositionRefs);

        List<Pm4ForensicsSurfaceRow> surfaceRows = surfaceIndices
            .Select(index =>
            {
                Pm4MsurEntry surface = allSurfaces[index];
                return new Pm4ForensicsSurfaceRow(
                    index,
                    surface.GroupKey,
                    surface.AttributeMask,
                    surface.MdosIndex,
                    surface.Ck24Type,
                    surface.Ck24ObjectId,
                    surface.IndexCount,
                    surface.MsviFirstIndex,
                    surface.Height,
                    surface.Normal);
            })
            .ToList();

        List<Pm4ForensicsMslkRow> mslkRows = linkRows
            .Select(item => new Pm4ForensicsMslkRow(
                item.LinkIndex,
                item.Entry.TypeFlags,
                item.Entry.Subtype,
                item.Entry.GroupObjectId,
                item.Entry.MspiFirstIndex,
                item.Entry.MspiIndexCount,
                item.Entry.LinkId,
                item.Entry.RefIndex,
                item.Entry.SystemFlag,
                surfaceIndexSet.Contains(item.Entry.RefIndex),
                item.Entry.RefIndex < document.KnownChunks.Mprl.Count))
            .ToList();

        List<Pm4ForensicsMprlRow> mprlRows = linkedPositionRefIndices
            .Select(index =>
            {
                Pm4MprlEntry entry = document.KnownChunks.Mprl[index];
                return new Pm4ForensicsMprlRow(
                    index,
                    entry.Unk00,
                    entry.Unk02,
                    entry.Unk04,
                    entry.Unk06,
                    entry.Position,
                    entry.Unk14,
                    entry.Unk16,
                    entry.Unk04 * (360f / 65536f),
                    entry.Unk16 != 0);
            })
            .ToList();

        return new Pm4ForensicsLinkGroupReport(
            groupObjectId,
            surfaceRows.Count,
            surfaceRows.Sum(static row => row.IndexCount),
            surfaceRows.Select(static row => row.AttributeMask).Distinct().Order().ToList(),
            surfaceRows.Select(static row => row.GroupKey).Distinct().Order().ToList(),
            surfaceRows.Select(static row => row.MdosIndex).Distinct().Order().ToList(),
            surfaceIndices.ToList(),
            linkedPositionRefIndices,
            ComputeSurfaceBounds(document, surfaceIndices, placementComparison),
            summary,
            BuildFootprintSummary(document.KnownChunks.Mprl, linkedPositionRefs, surfaceIndices, placementComparison, document),
            surfaceRows,
            mslkRows,
            mprlRows,
            placementComparison);
    }

    private static Pm4ForensicsPlacementComparison BuildPlacementComparison(
        Pm4ResearchDocument document,
        IReadOnlyList<int> surfaceIndices,
        int resolvedTileX,
        int resolvedTileY,
        int? tileX,
        int? tileY,
        IReadOnlyList<Pm4MprlEntry>? linkedPositionRefs = null)
    {
        List<Pm4MsurEntry> surfaces = surfaceIndices.Select(index => document.KnownChunks.Msur[index]).ToList();
        Pm4AxisConvention axisConvention = Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(document.KnownChunks.Msvt, document.KnownChunks.Msvi, surfaces);
        Pm4CoordinateMode fallbackCoordinateMode = Pm4PlacementMath.IsLikelyTileLocal(document.KnownChunks.Msvt)
            ? Pm4CoordinateMode.TileLocal
            : Pm4CoordinateMode.WorldSpace;
        IReadOnlyList<Pm4MprlEntry> refs = linkedPositionRefs ?? CollectLinkedPositionRefs(document, surfaceIndices);

        Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            surfaces,
            refs,
            anchorPositionRefs: refs,
            resolvedTileX,
            resolvedTileY,
            axisConvention,
            fallbackCoordinateMode);

        Pm4PlacementSolution placement = Pm4PlacementMath.ResolvePlacementSolution(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            surfaces,
            refs,
            anchorPositionRefs: refs,
            resolvedTileX,
            resolvedTileY,
            resolution.CoordinateMode,
            axisConvention);

        Pm4LinkedPositionRefSummary summary = SanitizeSummary(Pm4PlacementMath.SummarizeLinkedPositionRefs(refs));
        float? meanHeading = summary.HasNormalHeadings ? summary.HeadingMeanDegrees : null;
        float frameYawDegrees = placement.WorldYawCorrectionRadians * (180f / MathF.PI);

        return new Pm4ForensicsPlacementComparison(
            tileX,
            tileY,
            axisConvention,
            resolution.CoordinateMode,
            resolution.PlanarTransform,
            resolution.UsedFallback,
            SanitizeFinite(resolution.TileLocalScore),
            SanitizeFinite(resolution.WorldSpaceScore),
            placement.WorldPivot,
            frameYawDegrees,
            meanHeading,
            meanHeading.HasValue ? NormalizeDegrees(meanHeading.Value - frameYawDegrees) : null);
    }

    private static List<Pm4MprlEntry> CollectLinkedPositionRefs(Pm4ResearchDocument document, IReadOnlyList<int> surfaceIndices)
    {
        if (surfaceIndices.Count == 0 || document.KnownChunks.Mprl.Count == 0 || document.KnownChunks.Mslk.Count == 0)
            return [];

        HashSet<int> surfaceIndexSet = surfaceIndices.ToHashSet();
        HashSet<int> seen = [];
        List<Pm4MprlEntry> refs = [];

        for (int linkIndex = 0; linkIndex < document.KnownChunks.Mslk.Count; linkIndex++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[linkIndex];
            if ((uint)link.RefIndex >= (uint)document.KnownChunks.Mprl.Count)
                continue;

            if (!surfaceIndexSet.Contains(link.RefIndex))
                continue;

            if (!seen.Add(link.RefIndex))
                continue;

            refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
        }

        return refs;
    }

    private static List<(uint GroupObjectId, List<int> SurfaceIndices)> BuildLinkGroupSurfaceSets(
        IReadOnlyList<Pm4MslkEntry> links,
        IReadOnlyList<int> selectedSurfaceIndices,
        int surfaceCount)
    {
        if (selectedSurfaceIndices.Count == 0)
            return [];

        if (selectedSurfaceIndices.Count == 1 || links.Count == 0)
            return [(0, selectedSurfaceIndices.Order().ToList())];

        Dictionary<int, int> surfaceIndexToLocal = [];
        for (int localIndex = 0; localIndex < selectedSurfaceIndices.Count; localIndex++)
            surfaceIndexToLocal[selectedSurfaceIndices[localIndex]] = localIndex;

        Dictionary<uint, HashSet<int>> groupToMembers = [];
        for (int linkIndex = 0; linkIndex < links.Count; linkIndex++)
        {
            Pm4MslkEntry link = links[linkIndex];
            if (link.GroupObjectId == 0)
                continue;

            if (link.RefIndex >= surfaceCount || !surfaceIndexToLocal.TryGetValue(link.RefIndex, out int localRefIndex))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = [];
                groupToMembers.Add(link.GroupObjectId, members);
            }

            members.Add(localRefIndex);
        }

        if (groupToMembers.Count == 0)
            return [(0, selectedSurfaceIndices.Order().ToList())];

        int[] parent = new int[selectedSurfaceIndices.Count];
        for (int index = 0; index < parent.Length; index++)
            parent[index] = index;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        HashSet<int> linkedLocalIndices = [];
        foreach (HashSet<int> members in groupToMembers.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linkedLocalIndices.Add(first);
            foreach (int member in members)
            {
                linkedLocalIndices.Add(member);
                Union(parent, first, member);
            }
        }

        if (linkedLocalIndices.Count < 2)
            return [(0, selectedSurfaceIndices.Order().ToList())];

        Dictionary<int, List<int>> linkedComponents = [];
        for (int localIndex = 0; localIndex < selectedSurfaceIndices.Count; localIndex++)
        {
            if (!linkedLocalIndices.Contains(localIndex))
                continue;

            int root = Find(parent, localIndex);
            if (!linkedComponents.TryGetValue(root, out List<int>? component))
            {
                component = [];
                linkedComponents.Add(root, component);
            }

            component.Add(selectedSurfaceIndices[localIndex]);
        }

        List<(uint GroupObjectId, List<int> SurfaceIndices)> groups = linkedComponents.Values
            .OrderBy(static component => component.Min())
            .Select(component => (SelectDominantGroupObjectId(links, component, surfaceCount), component.Order().ToList()))
            .ToList();

        List<int> unlinked = [];
        for (int localIndex = 0; localIndex < selectedSurfaceIndices.Count; localIndex++)
        {
            if (!linkedLocalIndices.Contains(localIndex))
                unlinked.Add(selectedSurfaceIndices[localIndex]);
        }

        if (unlinked.Count > 0)
            groups.Add((0, unlinked.Order().ToList()));

        return groups;
    }

    private static uint SelectDominantGroupObjectId(IReadOnlyList<Pm4MslkEntry> links, IReadOnlyList<int> surfaceIndices, int surfaceCount)
    {
        if (surfaceIndices.Count == 0 || links.Count == 0)
            return 0;

        HashSet<int> surfaceIndexSet = surfaceIndices.ToHashSet();
        Dictionary<uint, int> counts = [];
        uint bestGroupObjectId = 0;
        int bestCount = 0;
        for (int linkIndex = 0; linkIndex < links.Count; linkIndex++)
        {
            Pm4MslkEntry link = links[linkIndex];
            if (link.GroupObjectId == 0 || link.RefIndex >= surfaceCount || !surfaceIndexSet.Contains(link.RefIndex))
                continue;

            int nextCount = 1;
            if (counts.TryGetValue(link.GroupObjectId, out int existingCount))
                nextCount = existingCount + 1;

            counts[link.GroupObjectId] = nextCount;
            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestGroupObjectId = link.GroupObjectId;
            }
        }

        return bestGroupObjectId;
    }

    private static Pm4MprlFootprintSummary BuildFootprintSummary(
        IReadOnlyList<Pm4MprlEntry> tileRefs,
        IReadOnlyList<Pm4MprlEntry> linkedRefs,
        IReadOnlyList<int> surfaceIndices,
        Pm4ForensicsPlacementComparison placementComparison,
        Pm4ResearchDocument document)
    {
        Pm4Bounds3? bounds = ComputeSurfaceBounds(document, surfaceIndices, placementComparison);
        if (bounds is null)
        {
            return new Pm4MprlFootprintSummary(
                tileRefs.Count,
                linkedRefs.Count,
                linkedRefs.Count(static entry => entry.Unk16 == 0),
                linkedRefs.Count(static entry => entry.Unk16 != 0),
                0,
                0,
                0,
                0,
                linkedRefs.Where(static entry => entry.Unk16 == 0).Select(static entry => (short?)entry.Unk14).Min(),
                linkedRefs.Where(static entry => entry.Unk16 == 0).Select(static entry => (short?)entry.Unk14).Max());
        }

        return new Pm4MprlFootprintSummary(
            tileRefs.Count,
            linkedRefs.Count,
            linkedRefs.Count(static entry => entry.Unk16 == 0),
            linkedRefs.Count(static entry => entry.Unk16 != 0),
            CountRefsInBounds(tileRefs, bounds, padding: 0f),
            CountRefsInBounds(tileRefs, bounds, padding: BoundsPadding),
            CountRefsInBounds(linkedRefs, bounds, padding: 0f),
            CountRefsInBounds(linkedRefs, bounds, padding: BoundsPadding),
            linkedRefs.Where(static entry => entry.Unk16 == 0).Select(static entry => (short?)entry.Unk14).Min(),
            linkedRefs.Where(static entry => entry.Unk16 == 0).Select(static entry => (short?)entry.Unk14).Max());
    }

    private static int CountRefsInBounds(IReadOnlyList<Pm4MprlEntry> refs, Pm4Bounds3 bounds, float padding)
    {
        int count = 0;
        float minX = bounds.Min.X - padding;
        float minY = bounds.Min.Y - padding;
        float minZ = bounds.Min.Z - padding;
        float maxX = bounds.Max.X + padding;
        float maxY = bounds.Max.Y + padding;
        float maxZ = bounds.Max.Z + padding;

        for (int index = 0; index < refs.Count; index++)
        {
            Vector3 position = refs[index].Position;
            if (position.X < minX || position.X > maxX || position.Y < minY || position.Y > maxY || position.Z < minZ || position.Z > maxZ)
                continue;

            count++;
        }

        return count;
    }

    private static Pm4Bounds3? ComputeSurfaceBounds(
        Pm4ResearchDocument document,
        IReadOnlyList<int> surfaceIndices,
        Pm4ForensicsPlacementComparison placementComparison)
    {
        List<Vector3> worldVertices = [];
        foreach (int surfaceIndex in surfaceIndices)
        {
            Pm4MsurEntry surface = document.KnownChunks.Msur[surfaceIndex];
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, document.KnownChunks.Msvi.Count);
            for (int index = firstIndex; index < endExclusive; index++)
            {
                uint vertexIndex = document.KnownChunks.Msvi[index];
                if (vertexIndex >= document.KnownChunks.Msvt.Count)
                    continue;

                worldVertices.Add(Pm4PlacementMath.ConvertPm4VertexToWorld(
                    document.KnownChunks.Msvt[(int)vertexIndex],
                    placementComparison.TileX ?? 0,
                    placementComparison.TileY ?? 0,
                    placementComparison.CoordinateMode,
                    placementComparison.AxisConvention,
                    placementComparison.PlanarTransform,
                    placementComparison.WorldPivot,
                    placementComparison.FrameYawDegrees * (MathF.PI / 180f)));
            }
        }

        if (worldVertices.Count == 0)
            return null;

        Vector3 min = worldVertices[0];
        Vector3 max = worldVertices[0];
        for (int index = 1; index < worldVertices.Count; index++)
        {
            min = Vector3.Min(min, worldVertices[index]);
            max = Vector3.Max(max, worldVertices[index]);
        }

        return new Pm4Bounds3(min, max);
    }

    private static Pm4LinkedPositionRefSummary SanitizeSummary(Pm4LinkedPositionRefSummary summary)
    {
        return new Pm4LinkedPositionRefSummary(
            summary.TotalCount,
            summary.NormalCount,
            summary.TerminatorCount,
            summary.FloorMin,
            summary.FloorMax,
            SanitizeFinite(summary.HeadingMinDegrees) ?? 0f,
            SanitizeFinite(summary.HeadingMaxDegrees) ?? 0f,
            SanitizeFinite(summary.HeadingMeanDegrees) ?? 0f);
    }

    private static float? SanitizeFinite(float value)
    {
        return float.IsFinite(value) ? value : null;
    }

    private static float NormalizeDegrees(float value)
    {
        while (value <= -180f)
            value += 360f;
        while (value > 180f)
            value -= 360f;
        return value;
    }

    private static (int? TileX, int? TileY) TryParseTileCoordinates(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return (null, null);

        string fileName = Path.GetFileNameWithoutExtension(sourcePath);
        if (string.IsNullOrWhiteSpace(fileName))
            return (null, null);

        string[] parts = fileName.Split('_', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (parts.Length < 2)
            return (null, null);

        if (!int.TryParse(parts[^2], out int tileX) || !int.TryParse(parts[^1], out int tileY))
            return (null, null);

        return (tileX, tileY);
    }
}