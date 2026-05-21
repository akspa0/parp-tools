using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Groups PM4 objects by MSHD.Field04 (region), then by CK24 (object), then by MSLK.GroupObjectId (sub-object).
/// This is the primary entry point for region-aware PM4 object decomposition.
/// </summary>
public static class Pm4RegionObjectGrouper
{
    /// <summary>
    /// Read all PM4 files in a directory and produce a region-first grouping report.
    /// </summary>
    public static Pm4RegionGroupingReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<(string Path, Pm4ResearchDocument Doc)> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(path => (path, Pm4ResearchReader.ReadFile(path)))
            .ToList();

        int totalFiles = files.Count;
        int nonEmptyFiles = files.Count(f => f.Doc.KnownChunks.Msur is { Count: > 0 });

        // Phase 1: Group tiles by MSHD.Field04 → region.
        Dictionary<uint, RegionAccumulator> regionAccumulators = new();

        foreach ((string path, Pm4ResearchDocument doc) in files)
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int tileX, out int tileY))
                continue;

            string tileCoord = $"{tileX}_{tileY}";
            uint regionId = doc.KnownChunks.Mshd?.Field04 ?? 0;

            if (!regionAccumulators.TryGetValue(regionId, out RegionAccumulator? regionAccum))
            {
                regionAccum = new RegionAccumulator(regionId);
                regionAccumulators[regionId] = regionAccum;
            }

            regionAccum.TileCoordinates.Add(tileCoord);

            IReadOnlyList<Pm4MsurEntry> msur = doc.KnownChunks.Msur;
            IReadOnlyList<Pm4MslkEntry> mslk = doc.KnownChunks.Mslk;
            IReadOnlyList<Pm4MprlEntry> mprl = doc.KnownChunks.Mprl;
            IReadOnlyList<Vector3> msvt = doc.KnownChunks.Msvt;
            IReadOnlyList<uint> msvi = doc.KnownChunks.Msvi;

            // Store tile data with a global surface offset for cross-tile reference mapping.
            int globalSurfOffset = regionAccum.AllMsurEntries.Count;
            regionAccum.AllMsurEntries.AddRange(msur);
            regionAccum.AllMslkEntries.AddRange(mslk);
            regionAccum.AllMprlEntries.AddRange(mprl);
            regionAccum.AllMsvtVertices.AddRange(msvt);
            regionAccum.AllMsviIndices.AddRange(msvi);

            TileData tileData = new(tileCoord, tileX, tileY, globalSurfOffset, msur.Count, mslk, mprl, msvt, msvi);
            regionAccum.Tiles.Add(tileData);

            // Phase 2: Within each tile, group surfaces by CK24 → object.
            for (int i = 0; i < msur.Count; i++)
            {
                Pm4MsurEntry surface = msur[i];
                uint ck24 = surface.Ck24;

                if (!regionAccum.Ck24Objects.TryGetValue(ck24, out Ck24ObjectAccumulator? objAccum))
                {
                    objAccum = new Ck24ObjectAccumulator(ck24, surface.Ck24Type, surface.Ck24ObjectId);
                    regionAccum.Ck24Objects[ck24] = objAccum;
                }

                objAccum.TileCoordinates.Add(tileCoord);
                objAccum.TotalSurfaceCount++;
                objAccum.TotalIndexCount += surface.IndexCount;

                // Store the surface with its global index (globalSurfOffset + localIndex).
                int globalIndex = globalSurfOffset + i;
                objAccum.SurfaceEntries.Add(new SurfaceEntry(globalIndex, tileCoord, tileX, tileY, surface));
            }
        }

        // Phase 3: Within each CK24 object, partition by MSLK.GroupObjectId → sub-objects.
        List<Pm4Region> regions = new();
        int totalObjects = 0;
        int totalSubObjects = 0;

        foreach (RegionAccumulator regionAccum in regionAccumulators.Values.OrderBy(r => r.RegionId))
        {
            List<Pm4RegionObject> regionObjects = new();
            int regionSurfaceCount = 0;

            foreach (Ck24ObjectAccumulator objAccum in regionAccum.Ck24Objects.Values.OrderBy(o => o.Ck24))
            {
                List<Pm4SubObject> subObjects = PartitionByGroupObjectId(regionAccum, objAccum);

                int objectSurfaceCount = objAccum.SurfaceEntries.Count;
                regionSurfaceCount += objectSurfaceCount;

                Pm4RegionObject regionObject = new(
                    objAccum.Ck24,
                    objAccum.Ck24Type,
                    objAccum.Ck24ObjectId,
                    subObjects,
                    objAccum.TileCoordinates.Distinct().OrderBy(t => t).ToList(),
                    objectSurfaceCount,
                    objAccum.TotalIndexCount);

                regionObjects.Add(regionObject);
                totalObjects++;
                totalSubObjects += subObjects.Count;
            }

            List<string> regionTiles = regionAccum.TileCoordinates.Distinct().OrderBy(t => t).ToList();

            Pm4Region region = new(
                regionAccum.RegionId,
                regionObjects,
                regionTiles,
                regionSurfaceCount,
                regionObjects.Count);

            regions.Add(region);
        }

        List<string> notes = new()
        {
            $"Region-first grouping: {regions.Count} regions from {totalFiles} PM4 files ({nonEmptyFiles} non-empty).",
            $"Total objects: {totalObjects}, total sub-objects: {totalSubObjects}.",
            $"Non-empty regions: {regions.Count(r => !r.IsEmptyStubRegion)}.",
            $"Regions with Field04=1 (empty stubs): {regions.Count(r => r.IsEmptyStubRegion)}.",
            "Each region groups all CK24 objects belonging to a single MSHD.Field04 scene division.",
            "Within each object, MSLK.GroupObjectId partitions surfaces into linked sub-objects."
        };

        return new Pm4RegionGroupingReport(
            resolvedDirectory,
            totalFiles,
            nonEmptyFiles,
            regions.Count,
            totalObjects,
            totalSubObjects,
            regions,
            notes);
    }

    /// <summary>
    /// Partition a CK24 object's surfaces into sub-objects by MSLK.GroupObjectId.
    /// Uses the global surface index to match MSLK.RefIndex entries.
    /// </summary>
    private static List<Pm4SubObject> PartitionByGroupObjectId(
        RegionAccumulator regionAccum,
        Ck24ObjectAccumulator objAccum)
    {
        List<SurfaceEntry> surfaces = objAccum.SurfaceEntries;
        if (surfaces.Count == 0)
            return new List<Pm4SubObject>();

        // Build a set of global surface indices belonging to this CK24 object.
        HashSet<int> objectGlobalIndices = new();
        foreach (SurfaceEntry se in surfaces)
            objectGlobalIndices.Add(se.GlobalIndex);

        // Build a mapping from global MSUR index → local surface entry index.
        Dictionary<int, int> globalToLocal = new();
        for (int i = 0; i < surfaces.Count; i++)
            globalToLocal[surfaces[i].GlobalIndex] = i;

        // Scan MSLK entries to find GroupObjectId → surface mappings.
        // MSLK.RefIndex references MSUR — in the concatenated region, this is a global index.
        Dictionary<uint, List<int>> groupToMembers = new();
        Dictionary<int, uint> surfaceToGroup = new();
        List<uint> groupIds = new();

        foreach (Pm4MslkEntry link in regionAccum.AllMslkEntries)
        {
            if (link.GroupObjectId == 0)
                continue;

            if (link.RefIndex >= regionAccum.AllMsurEntries.Count)
                continue;

            // Check if the referenced surface belongs to our CK24 object.
            if (!objectGlobalIndices.Contains(link.RefIndex))
                continue;

            // Map global index to local surface index.
            if (!globalToLocal.TryGetValue(link.RefIndex, out int localIdx))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out List<int>? members))
            {
                members = new List<int>();
                groupToMembers[link.GroupObjectId] = members;
                groupIds.Add(link.GroupObjectId);
            }

            if (!surfaceToGroup.ContainsKey(localIdx))
            {
                surfaceToGroup[localIdx] = link.GroupObjectId;
                members.Add(localIdx);
            }
        }

        // Surfaces not assigned to any group.
        List<int> unlinkedIndices = new();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!surfaceToGroup.ContainsKey(i))
                unlinkedIndices.Add(i);
        }

        // Build sub-objects from groups.
        List<Pm4SubObject> subObjects = new();

        foreach (uint groupId in groupIds.OrderBy(id => id))
        {
            List<int> memberIndices = groupToMembers[groupId];
            if (memberIndices.Count == 0)
                continue;

            List<Pm4MprlEntry> positionRefs = CollectPositionRefsForGroup(
                regionAccum, surfaces, memberIndices, groupIds);

            Pm4Bounds3? bounds = ComputeBounds(regionAccum, surfaces, memberIndices);
            float avgHeight = memberIndices.Count > 0
                ? memberIndices.Average(idx => surfaces[idx].Surface.Height)
                : 0f;

            Pm4SubObject subObject = new(
                groupId,
                memberIndices.Select(idx => surfaces[idx].GlobalIndex).ToList(),
                positionRefs,
                groupIds,
                groupId,
                bounds,
                avgHeight);

            subObjects.Add(subObject);
        }

        // Unlinked surfaces get their own sub-object.
        if (unlinkedIndices.Count > 0)
        {
            List<Pm4MprlEntry> positionRefs = CollectPositionRefsForGroup(
                regionAccum, surfaces, unlinkedIndices, groupIds);

            Pm4Bounds3? bounds = ComputeBounds(regionAccum, surfaces, unlinkedIndices);
            float avgHeight = unlinkedIndices.Count > 0
                ? unlinkedIndices.Average(idx => surfaces[idx].Surface.Height)
                : 0f;

            Pm4SubObject unlinkedSubObject = new(
                0,
                unlinkedIndices.Select(idx => surfaces[idx].GlobalIndex).ToList(),
                positionRefs,
                groupIds,
                0,
                bounds,
                avgHeight);

            subObjects.Add(unlinkedSubObject);
        }

        return subObjects;
    }

    /// <summary>
    /// Collect MPRL position references for a group of surfaces via MSLK GroupObjectId linking.
    /// </summary>
    private static List<Pm4MprlEntry> CollectPositionRefsForGroup(
        RegionAccumulator regionAccum,
        List<SurfaceEntry> surfaces,
        List<int> memberIndices,
        List<uint> allGroupIds)
    {
        // Collect the GroupObjectIds that members of this sub-object participate in.
        HashSet<uint> memberGroupIds = new();
        foreach (int idx in memberIndices)
        {
            TileData? tile = regionAccum.Tiles.FirstOrDefault(t => t.TileCoord == surfaces[idx].TileCoord);
            if (tile is null)
                continue;

            foreach (Pm4MslkEntry link in tile.Mslk)
            {
                if (link.GroupObjectId == 0)
                    continue;
                if (link.RefIndex == surfaces[idx].GlobalIndex)
                    memberGroupIds.Add(link.GroupObjectId);
            }
        }

        // Collect all MSLK entries with matching GroupObjectIds, then use their RefIndex as MPRL indices.
        HashSet<int> seenRefIndices = new();
        List<Pm4MprlEntry> refs = new();

        if (memberGroupIds.Count > 0)
        {
            foreach (Pm4MslkEntry link in regionAccum.AllMslkEntries)
            {
                if (link.GroupObjectId == 0 || !memberGroupIds.Contains(link.GroupObjectId))
                    continue;

                if (link.RefIndex >= regionAccum.AllMprlEntries.Count)
                    continue;

                if (!seenRefIndices.Add(link.RefIndex))
                    continue;

                refs.Add(regionAccum.AllMprlEntries[link.RefIndex]);
            }

            if (refs.Count > 0)
                return refs;
        }

        // Fallback: direct surface-to-MSLK matching.
        HashSet<int> memberGlobalIndices = new();
        foreach (int idx in memberIndices)
            memberGlobalIndices.Add(surfaces[idx].GlobalIndex);

        foreach (Pm4MslkEntry link in regionAccum.AllMslkEntries)
        {
            if (link.RefIndex >= regionAccum.AllMprlEntries.Count)
                continue;

            if (!memberGlobalIndices.Contains(link.RefIndex))
                continue;

            if (!seenRefIndices.Add(link.RefIndex))
                continue;

            refs.Add(regionAccum.AllMprlEntries[link.RefIndex]);
        }

        return refs;
    }

    /// <summary>
    /// Compute axis-aligned bounding box for a set of surfaces.
    /// </summary>
    private static Pm4Bounds3? ComputeBounds(
        RegionAccumulator regionAccum,
        List<SurfaceEntry> surfaces,
        List<int> indices)
    {
        if (indices.Count == 0)
            return null;

        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);

        foreach (int idx in indices)
        {
            SurfaceEntry se = surfaces[idx];
            Pm4MsurEntry surface = se.Surface;

            // Find the tile's vertex/index data.
            TileData? tile = regionAccum.Tiles.FirstOrDefault(t => t.TileCoord == se.TileCoord);
            if (tile is null)
                continue;

            int startIndex = (int)surface.MsviFirstIndex;
            int indexCount = surface.IndexCount;

            for (int j = 0; j < indexCount; j++)
            {
                int vi = startIndex + j;
                if (vi < 0 || vi >= tile.Msvi.Count)
                    continue;

                uint vertexIndex = tile.Msvi[vi];
                if (vertexIndex >= (uint)tile.Msvt.Count)
                    continue;

                Vector3 vertex = tile.Msvt[(int)vertexIndex];
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
        }

        if (min.X > max.X)
            return null;

        return new Pm4Bounds3(min, max);
    }

    private sealed class RegionAccumulator
    {
        public uint RegionId { get; }
        public HashSet<string> TileCoordinates { get; } = new(StringComparer.Ordinal);
        public Dictionary<uint, Ck24ObjectAccumulator> Ck24Objects { get; } = new();
        public List<TileData> Tiles { get; } = new();
        public List<Pm4MsurEntry> AllMsurEntries { get; } = new();
        public List<Pm4MslkEntry> AllMslkEntries { get; } = new();
        public List<Pm4MprlEntry> AllMprlEntries { get; } = new();
        public List<Vector3> AllMsvtVertices { get; } = new();
        public List<uint> AllMsviIndices { get; } = new();

        public RegionAccumulator(uint regionId)
        {
            RegionId = regionId;
        }
    }

    private sealed class Ck24ObjectAccumulator
    {
        public uint Ck24 { get; }
        public byte Ck24Type { get; }
        public ushort Ck24ObjectId { get; }
        public HashSet<string> TileCoordinates { get; } = new(StringComparer.Ordinal);
        public int TotalSurfaceCount { get; set; }
        public int TotalIndexCount { get; set; }
        public List<SurfaceEntry> SurfaceEntries { get; } = new();

        public Ck24ObjectAccumulator(uint ck24, byte ck24Type, ushort ck24ObjectId)
        {
            Ck24 = ck24;
            Ck24Type = ck24Type;
            Ck24ObjectId = ck24ObjectId;
        }
    }

    private sealed class TileData
    {
        public string TileCoord { get; }
        public int TileX { get; }
        public int TileY { get; }
        public int GlobalSurfOffset { get; }
        public int LocalSurfCount { get; }
        public IReadOnlyList<Pm4MslkEntry> Mslk { get; }
        public IReadOnlyList<Pm4MprlEntry> Mprl { get; }
        public IReadOnlyList<Vector3> Msvt { get; }
        public IReadOnlyList<uint> Msvi { get; }

        public TileData(string tileCoord, int tileX, int tileY, int globalSurfOffset, int localSurfCount,
            IReadOnlyList<Pm4MslkEntry> mslk, IReadOnlyList<Pm4MprlEntry> mprl,
            IReadOnlyList<Vector3> msvt, IReadOnlyList<uint> msvi)
        {
            TileCoord = tileCoord;
            TileX = tileX;
            TileY = tileY;
            GlobalSurfOffset = globalSurfOffset;
            LocalSurfCount = localSurfCount;
            Mslk = mslk;
            Mprl = mprl;
            Msvt = msvt;
            Msvi = msvi;
        }
    }

    private sealed class SurfaceEntry
    {
        public int GlobalIndex { get; }
        public string TileCoord { get; }
        public int TileX { get; }
        public int TileY { get; }
        public Pm4MsurEntry Surface { get; }

        public SurfaceEntry(int globalIndex, string tileCoord, int tileX, int tileY, Pm4MsurEntry surface)
        {
            GlobalIndex = globalIndex;
            TileCoord = tileCoord;
            TileX = tileX;
            TileY = tileY;
            Surface = surface;
        }
    }
}
