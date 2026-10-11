using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Resolution method used to identify a PM4 object.
/// </summary>
public enum Pm4PlacementResolutionMethod
{
    AdjacentSpillover,
    LedgerExactZ,
    NeighborThematic,
    ShapeScore,
    UnknownFallback,
}

/// <summary>
/// Ground truth placement record from the persistent Rosetta ledger.
/// </summary>
public sealed record Pm4LedgerPlacementRecord(
    string AssetPath,
    string AssetKind,
    float PosX,
    float PosY,
    float PosZ,
    float RotX,
    float RotY,
    float RotZ,
    float MinX,
    float MinY,
    float MinZ,
    float MaxX,
    float MaxY,
    float MaxZ,
    int SurfaceCount);

/// <summary>
/// Record of an individual resolved object placement with its provenance evidence.
/// </summary>
public sealed record Pm4ResolvedObjectRecord(
    string ModelPath,
    string AssetKind,
    uint Ck24,
    float PlacementZ,
    int SurfaceCount,
    int VertexCount,
    Vector3 Position,
    Vector3 Rotation,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    Vector3 ObservedShape,
    Pm4PlacementResolutionMethod Method,
    double Confidence,
    string SourceNote);

/// <summary>
/// Result of resolving all objects on an unauthored PM4 tile.
/// </summary>
public sealed record Pm4TileResolutionResult(
    string TileName,
    int TileX,
    int TileY,
    AdtPlacementCatalog Catalog,
    IReadOnlyList<Pm4ResolvedObjectRecord> ResolvedWmos,
    IReadOnlyList<Pm4ResolvedObjectRecord> ResolvedDoodads);

/// <summary>
/// Resolves authentic WMO and M2 object placements for unauthored PM4 tiles by synthesizing
/// cross-tile spillover from adjacent authored tiles, bit-exact Z keys from the persistent ledger,
/// neighbor thematic asset libraries, and multi-signal shape matching (Spec 270 Phase 2).
/// </summary>
public static class Pm4UnauthoredPlacementResolver
{
    public const float WeldEpsilon = 0.25f;
    private const string UnknownWmo = @"World\wmo\missingwmo.wmo";
    private const int SyntheticUniqueIdBase = 0x50000000;

    /// <summary>
    /// Resolves authentic placements for a given unauthored PM4 tile.
    /// </summary>
    public static Pm4TileResolutionResult ResolveTile(
        string pm4Path,
        string mapDirectory,
        Func<string, AdtPlacementCatalog>? neighborCatalogReader = null,
        IReadOnlyDictionary<string, List<Vector3>>? shapeLibrary = null,
        int uniqueIdOffset = 0,
        IReadOnlyDictionary<uint, Pm4LedgerPlacementRecord>? ledgerLookup = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pm4Path);
        ArgumentException.ThrowIfNullOrWhiteSpace(mapDirectory);

        string fileName = Path.GetFileName(pm4Path);
        string tileStem = Path.GetFileNameWithoutExtension(pm4Path);

        if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileFirst, out int tileSecond))
            throw new ArgumentException($"Cannot parse tile coordinates from '{pm4Path}'.");

        // Canonical tile indices: tileFirst and tileSecond in filename map directly to tile grid coordinates
        int tileX = tileFirst;
        int tileY = tileSecond;

        Pm4ResearchDocument doc = Pm4ResearchReader.ReadFile(pm4Path);
        Pm4KnownChunkSet chunks = doc.KnownChunks;

        // 1. Load neighbor authored ADT placement catalogs (8-neighborhood)
        Dictionary<string, AdtPlacementCatalog> neighborCatalogs = LoadNeighborCatalogs(mapDirectory, tileFirst, tileSecond, neighborCatalogReader);

        // 2. Extract WMO objects (Keyed CK24 / PackedParams != 0)
        List<Pm4RawObjectGroup> wmoGroups = ExtractWmoGroups(chunks);

        int nextUniqueId = SyntheticUniqueIdBase + uniqueIdOffset;
        var resolvedWmos = new List<Pm4ResolvedObjectRecord>();
        var modfPlacements = new List<AdtWorldModelPlacement>();
        var wmoNames = new List<string>();
        var claimedSpillovers = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (Pm4RawObjectGroup group in wmoGroups)
        {
            Pm4ResolvedObjectRecord resolved = ResolveWmo(group, neighborCatalogs, shapeLibrary, ledgerLookup);
            resolvedWmos.Add(resolved);

            // Honor spanned PM4 objects: only take the placement that can exist in the tile itself in ADT space.
            // Spanned objects (e.g. Stormwind Harbor) spanning across multiple tiles are owned and emitted
            // exclusively by the tile holding the placement origin, preventing multi-instance copies.
            if (!Pm4CoordinateService.IsWithinPlacementTileBounds(resolved.Position, tileFirst, tileSecond))
            {
                continue;
            }

            if (resolved.Method == Pm4PlacementResolutionMethod.AdjacentSpillover)
            {
                // Unique instance key based on model and parent position to avoid multi-instance stacking
                string instanceKey = $"{resolved.ModelPath}_{resolved.Position.X:F1}_{resolved.Position.Y:F1}";
                if (!claimedSpillovers.Add(instanceKey))
                {
                    // Spillover already represented in placement catalog, skip duplicate MODF entry
                    continue;
                }
            }

            string normPath = resolved.ModelPath;
            int nameIndex = wmoNames.FindIndex(n => n.Equals(normPath, StringComparison.OrdinalIgnoreCase));
            if (nameIndex < 0)
            {
                nameIndex = wmoNames.Count;
                wmoNames.Add(normPath);
            }

            modfPlacements.Add(new AdtWorldModelPlacement(
                nameIndex,
                normPath,
                nextUniqueId++,
                resolved.Position,
                resolved.Rotation,
                resolved.BoundsMin,
                resolved.BoundsMax,
                Flags: 0,
                DoodadSet: 0,
                NameSet: 0,
                Scale: 1024));
        }

        // 3. Extract M2 doodad components from CK24 == 0 remainder
        List<Pm4RawObjectGroup> doodadGroups = ExtractDoodadComponents(chunks);
        var resolvedDoodads = new List<Pm4ResolvedObjectRecord>();
        var mddfPlacements = new List<AdtModelPlacement>();
        var m2Names = new List<string>();

        foreach (Pm4RawObjectGroup group in doodadGroups)
        {
            Pm4ResolvedObjectRecord? resolved = ResolveDoodad(group, neighborCatalogs, shapeLibrary);
            if (resolved is null)
                continue;

            resolvedDoodads.Add(resolved);

            // Honor spanned PM4 doodads: only take placements that exist in this tile itself in ADT space.
            if (!Pm4CoordinateService.IsWithinPlacementTileBounds(resolved.Position, tileFirst, tileSecond))
            {
                continue;
            }

            string normPath = resolved.ModelPath;
            int nameIndex = m2Names.FindIndex(n => n.Equals(normPath, StringComparison.OrdinalIgnoreCase));
            if (nameIndex < 0)
            {
                nameIndex = m2Names.Count;
                m2Names.Add(normPath);
            }

            mddfPlacements.Add(new AdtModelPlacement(
                nameIndex,
                normPath,
                nextUniqueId++,
                resolved.Position,
                resolved.Rotation,
                Scale: 1.0f));
        }

        var catalog = new AdtPlacementCatalog(
            pm4Path,
            MapFileKind.AdtObj,
            m2Names,
            wmoNames,
            mddfPlacements,
            modfPlacements);

        return new Pm4TileResolutionResult(
            tileStem,
            tileX,
            tileY,
            catalog,
            resolvedWmos,
            resolvedDoodads);
    }

    private static Pm4ResolvedObjectRecord ResolveWmo(
        Pm4RawObjectGroup group,
        Dictionary<string, AdtPlacementCatalog> neighborCatalogs,
        IReadOnlyDictionary<string, List<Vector3>>? shapeLibrary,
        IReadOnlyDictionary<uint, Pm4LedgerPlacementRecord>? ledgerLookup = null)
    {
        float zFloat = group.PlacementZ;
        Vector3 shape = ShapeOf(group.BoundsMin, group.BoundsMax);
        float footprintYaw = ComputeFootprintYawDegrees(group.Vertices);
        Vector3 defaultRotation = new(0f, 0f, footprintYaw);

        // Strategy 0: Check persistent Rosetta ledger for bit-exact Z key
        if (ledgerLookup != null && group.Key != 0 && ledgerLookup.TryGetValue(group.Key, out Pm4LedgerPlacementRecord? ledgerRec))
        {
            if (!string.IsNullOrWhiteSpace(ledgerRec.AssetPath) && !ledgerRec.AssetPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
            {
                Vector3 ledgerPos = new(ledgerRec.PosX, ledgerRec.PosY, ledgerRec.PosZ);
                Vector3 ledgerRot = new(ledgerRec.RotX, ledgerRec.RotY, ledgerRec.RotZ);
                return new Pm4ResolvedObjectRecord(
                    ledgerRec.AssetPath,
                    ledgerRec.AssetKind ?? "WMO",
                    group.Key,
                    zFloat,
                    group.SurfaceCount,
                    group.VertexCount,
                    ledgerPos,
                    ledgerRot,
                    group.BoundsMin,
                    group.BoundsMax,
                    shape,
                    Pm4PlacementResolutionMethod.LedgerExactZ,
                    1.0,
                    $"Bit-exact Z {zFloat:F3} matched persistent Rosetta ledger entry {ledgerRec.AssetPath}");
            }
        }

        // Strategy 1: Check neighbor catalogs for identical placement Z or overlapping bounds
        foreach (var (neighborTile, cat) in neighborCatalogs)
        {
            foreach (AdtWorldModelPlacement placed in cat.WorldModelPlacements)
            {
                if (placed.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                    continue;

                // Bit-exact or near-exact Z match (within 0.05 yds)
                if (MathF.Abs(placed.Position.Z - zFloat) <= 0.05f)
                {
                    Vector3 rot = placed.Rotation != Vector3.Zero ? placed.Rotation : defaultRotation;
                    return new Pm4ResolvedObjectRecord(
                        placed.ModelPath,
                        "WMO",
                        group.Key,
                        zFloat,
                        group.SurfaceCount,
                        group.VertexCount,
                        placed.Position,
                        rot,
                        group.BoundsMin,
                        group.BoundsMax,
                        shape,
                        Pm4PlacementResolutionMethod.AdjacentSpillover,
                        1.0,
                        $"Exact Z {zFloat:F3} matched neighbor {neighborTile} placement {placed.ModelPath}");
                }

                // Check 3D bounding box containment with neighbor WMO
                if (BoxesOverlap2D(group.BoundsMin, group.BoundsMax, placed.BoundsMin, placed.BoundsMax, padding: 5.0f)
                    && MathF.Abs(placed.Position.Z - zFloat) <= 2.5f)
                {
                    Vector3 rot = placed.Rotation != Vector3.Zero ? placed.Rotation : defaultRotation;
                    return new Pm4ResolvedObjectRecord(
                        placed.ModelPath,
                        "WMO",
                        group.Key,
                        zFloat,
                        group.SurfaceCount,
                        group.VertexCount,
                        placed.Position,
                        rot,
                        group.BoundsMin,
                        group.BoundsMax,
                        shape,
                        Pm4PlacementResolutionMethod.AdjacentSpillover,
                        0.95,
                        $"Spatial bounds overlap with neighbor {neighborTile} placement {placed.ModelPath}");
                }
            }
        }

        // Strategy 2: Neighbor Thematic Asset Pool
        var neighborModels = neighborCatalogs.Values
            .SelectMany(c => c.WorldModelPlacements)
            .Where(p => !p.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
            .GroupBy(p => p.ModelPath, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (neighborModels.Count > 0)
        {
            var bestThematic = neighborModels
                .Select(g =>
                {
                    double dist = double.MaxValue;
                    Vector3 sampleSpan = g.First().BoundsMax - g.First().BoundsMin;
                    if (shapeLibrary != null && shapeLibrary.TryGetValue(g.Key, out var refShapes) && refShapes.Count > 0)
                    {
                        dist = refShapes.Min(s => ShapeDistance(shape, s));
                    }
                    else
                    {
                        Vector3 refShape = ShapeOf(Vector3.Zero, sampleSpan);
                        dist = ShapeDistance(shape, refShape);
                    }
                    return (Model: g.Key, Rot: g.First().Rotation, Span: sampleSpan, Dist: dist);
                })
                .OrderBy(t => t.Dist)
                .FirstOrDefault();

            if (bestThematic.Model is not null && bestThematic.Dist <= 0.18)
            {
                float alignedYaw = ComputeModelAlignedYaw(group.Vertices, bestThematic.Span);
                Vector3 rot = MathF.Abs(alignedYaw) > 0.01f ? new Vector3(0f, 0f, alignedYaw) : (bestThematic.Rot != Vector3.Zero ? bestThematic.Rot : defaultRotation);
                Vector3 attractedPos = AttractPositionToPm4Mesh(group.Centroid, zFloat, rot.Z, Vector3.Zero, bestThematic.Span);

                return new Pm4ResolvedObjectRecord(
                    bestThematic.Model,
                    "WMO",
                    group.Key,
                    zFloat,
                    group.SurfaceCount,
                    group.VertexCount,
                    attractedPos,
                    rot,
                    group.BoundsMin,
                    group.BoundsMax,
                    shape,
                    Pm4PlacementResolutionMethod.NeighborThematic,
                    Math.Max(0.5, 1.0 - bestThematic.Dist),
                    $"Thematic match from adjacent tiles with shape distance {bestThematic.Dist:F3}");
            }
        }

        // Strategy 3: Global Shape Library matching
        if (shapeLibrary is { Count: > 0 })
        {
            string? bestAsset = null;
            Vector3 bestRefShape = Vector3.Zero;
            double bestScore = double.MaxValue;

            foreach (var (asset, shapes) in shapeLibrary)
            {
                if (asset.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                    continue;

                foreach (Vector3 refShape in shapes)
                {
                    double d = ShapeDistance(shape, refShape);
                    if (d < bestScore)
                    {
                        bestScore = d;
                        bestAsset = asset;
                        bestRefShape = refShape;
                    }
                }
            }

            if (bestAsset is not null && bestScore <= 0.18)
            {
                float alignedYaw = ComputeModelAlignedYaw(group.Vertices, bestRefShape);
                Vector3 rot = new(0f, 0f, alignedYaw);
                Vector3 attractedPos = AttractPositionToPm4Mesh(group.Centroid, zFloat, rot.Z, Vector3.Zero, bestRefShape);

                return new Pm4ResolvedObjectRecord(
                    bestAsset,
                    "WMO",
                    group.Key,
                    zFloat,
                    group.SurfaceCount,
                    group.VertexCount,
                    attractedPos,
                    rot,
                    group.BoundsMin,
                    group.BoundsMax,
                    shape,
                    Pm4PlacementResolutionMethod.ShapeScore,
                    Math.Max(0.4, 1.0 - bestScore),
                    $"Global shape library match with score {bestScore:F3}");
            }
        }

        // Fallback: missingwmo.wmo
        return new Pm4ResolvedObjectRecord(
            UnknownWmo,
            "WMO",
            group.Key,
            zFloat,
            group.SurfaceCount,
            group.VertexCount,
            group.Centroid,
            defaultRotation,
            group.BoundsMin,
            group.BoundsMax,
            shape,
            Pm4PlacementResolutionMethod.UnknownFallback,
            0.1,
            "No neighbor, thematic, or shape candidate cleared acceptance threshold");
    }

    private static Pm4ResolvedObjectRecord? ResolveDoodad(
        Pm4RawObjectGroup group,
        Dictionary<string, AdtPlacementCatalog> neighborCatalogs,
        IReadOnlyDictionary<string, List<Vector3>>? shapeLibrary = null)
    {
        // Only components with sufficient surfaces/vertices are physical doodads (filter trivial terrain remnants)
        if (group.SurfaceCount < 1 || group.VertexCount < 3)
            return null;

        Vector2 planar = new(group.Centroid.X, group.Centroid.Y);
        float doodadYaw = ComputeFootprintYawDegrees(group.Vertices);
        Vector3 defaultRotation = new(0f, 0f, doodadYaw);

        // Strategy 1: Check neighbor M2 placements for spatial coincidence (r <= 35.0 yds)
        AdtModelPlacement? bestNeighbor = null;
        float bestDist = float.MaxValue;
        string bestNeighborTile = "";

        foreach (var (neighborTile, cat) in neighborCatalogs)
        {
            foreach (AdtModelPlacement m in cat.ModelPlacements)
            {
                if (m.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
                    continue;

                float d = Vector2.Distance(new Vector2(m.Position.X, m.Position.Y), planar);
                if (d <= 35.0f && d < bestDist)
                {
                    bestDist = d;
                    bestNeighbor = m;
                    bestNeighborTile = neighborTile;
                }
            }
        }

        if (bestNeighbor != null)
        {
            Vector3 rot = MathF.Abs(doodadYaw) > 0.1f ? defaultRotation : bestNeighbor.Rotation;
            return new Pm4ResolvedObjectRecord(
                bestNeighbor.ModelPath,
                "M2",
                group.Key,
                group.Centroid.Z,
                group.SurfaceCount,
                group.VertexCount,
                group.Centroid,
                rot,
                group.BoundsMin,
                group.BoundsMax,
                ShapeOf(group.BoundsMin, group.BoundsMax),
                Pm4PlacementResolutionMethod.AdjacentSpillover,
                1.0,
                $"Matched neighbor {bestNeighborTile} M2 doodad at distance {bestDist:F2} yds");
        }

        // Strategy 2: Neighbor Thematic M2 Pool matching by shape
        var neighborM2s = neighborCatalogs.Values
            .SelectMany(c => c.ModelPlacements)
            .Where(m => !m.ModelPath.Contains("OILPLATFORM", StringComparison.OrdinalIgnoreCase))
            .GroupBy(m => m.ModelPath, StringComparer.OrdinalIgnoreCase)
            .ToList();

        if (neighborM2s.Count > 0 && shapeLibrary != null)
        {
            Vector3 doodadShape = ShapeOf(group.BoundsMin, group.BoundsMax);
            var bestThematic = neighborM2s
                .Select(g =>
                {
                    double dist = double.MaxValue;
                    if (shapeLibrary.TryGetValue(g.Key, out var refShapes) && refShapes.Count > 0)
                    {
                        dist = refShapes.Min(s => ShapeDistance(doodadShape, s));
                    }
                    return (Model: g.Key, First: g.First(), Dist: dist);
                })
                .Where(t => t.Dist <= 0.22)
                .OrderBy(t => t.Dist)
                .FirstOrDefault();

            if (bestThematic.Model is not null)
            {
                return new Pm4ResolvedObjectRecord(
                    bestThematic.Model,
                    "M2",
                    group.Key,
                    group.Centroid.Z,
                    group.SurfaceCount,
                    group.VertexCount,
                    group.Centroid,
                    defaultRotation,
                    group.BoundsMin,
                    group.BoundsMax,
                    doodadShape,
                    Pm4PlacementResolutionMethod.NeighborThematic,
                    Math.Max(0.5, 1.0 - bestThematic.Dist),
                    $"Thematic M2 match from adjacent tiles with shape distance {bestThematic.Dist:F3}");
            }
        }

        return null;
    }

    private static List<Pm4RawObjectGroup> ExtractWmoGroups(Pm4KnownChunkSet chunks)
    {
        var groups = new List<Pm4RawObjectGroup>();
        if (chunks.Msur.Count == 0 || chunks.Msvt.Count == 0 || chunks.Msvi.Count == 0)
            return groups;

        var byKey = new Dictionary<uint, List<int>>();
        for (int i = 0; i < chunks.Msur.Count; i++)
        {
            uint key = chunks.Msur[i].PackedParams;
            if (key == 0u)
                continue;

            if (!byKey.TryGetValue(key, out List<int>? list))
            {
                list = [];
                byKey[key] = list;
            }
            list.Add(i);
        }

        foreach (var (key, surfaceIndices) in byKey)
        {
            List<Pm4MsurEntry> surfaces = [.. surfaceIndices.Select(idx => chunks.Msur[idx])];
            List<Vector3> verts = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
            if (verts.Count == 0)
                continue;

            List<Vector3> placedVerts = new(verts.Count);
            Vector3 min = new(float.MaxValue), max = new(float.MinValue), sum = Vector3.Zero;
            foreach (Vector3 v in verts)
            {
                Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(v);
                placedVerts.Add(p);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
                sum += p;
            }

            Vector3 centroid = sum / verts.Count;
            float placementZ = BitConverter.UInt32BitsToSingle(key);

            groups.Add(new Pm4RawObjectGroup(
                key,
                placementZ,
                surfaceIndices.Count,
                verts.Count,
                min,
                max,
                centroid,
                placedVerts));
        }

        return groups;
    }

    private static List<Pm4RawObjectGroup> ExtractDoodadComponents(Pm4KnownChunkSet chunks)
    {
        var groups = new List<Pm4RawObjectGroup>();
        if (chunks.Msur.Count == 0 || chunks.Msvt.Count == 0 || chunks.Msvi.Count == 0)
            return groups;

        List<int> zeroIndices = [];
        for (int i = 0; i < chunks.Msur.Count; i++)
        {
            if (chunks.Msur[i].PackedParams == 0u)
                zeroIndices.Add(i);
        }

        if (zeroIndices.Count == 0)
            return groups;

        // Disjoint-set spatial clustering of zero-bucket surfaces at epsilon 0.25
        Dictionary<(int, int, int), List<int>> cellMap = [];
        foreach (int sIdx in zeroIndices)
        {
            Pm4MsurEntry s = chunks.Msur[sIdx];
            int first = (int)s.MsviFirstIndex;
            int end = Math.Min(first + s.IndexCount, chunks.Msvi.Count);
            for (int k = first; k < end; k++)
            {
                int vi = (int)chunks.Msvi[k];
                if ((uint)vi >= (uint)chunks.Msvt.Count)
                    continue;

                Vector3 v = chunks.Msvt[vi];
                (int, int, int) cell = (
                    (int)MathF.Round(v.X / WeldEpsilon),
                    (int)MathF.Round(v.Y / WeldEpsilon),
                    (int)MathF.Round(v.Z / WeldEpsilon));

                if (!cellMap.TryGetValue(cell, out List<int>? bucket))
                {
                    bucket = [];
                    cellMap[cell] = bucket;
                }
                bucket.Add(sIdx);
            }
        }

        Dictionary<int, int> parent = [];
        foreach (int idx in zeroIndices)
            parent[idx] = idx;

        int Find(int i)
        {
            if (parent[i] == i) return i;
            return parent[i] = Find(parent[i]);
        }

        void Union(int a, int b)
        {
            int ra = Find(a), rb = Find(b);
            if (ra != rb) parent[ra] = rb;
        }

        foreach (List<int> cellSurfs in cellMap.Values)
        {
            if (cellSurfs.Count <= 1) continue;
            int first = cellSurfs[0];
            for (int k = 1; k < cellSurfs.Count; k++)
                Union(first, cellSurfs[k]);
        }

        Dictionary<int, List<int>> components = [];
        foreach (int idx in zeroIndices)
        {
            int root = Find(idx);
            if (!components.TryGetValue(root, out List<int>? list))
            {
                list = [];
                components[root] = list;
            }
            list.Add(idx);
        }

        foreach (List<int> compSurfaces in components.Values)
        {
            // Skip tiny single-triangle remnants (often terrain borders)
            if (compSurfaces.Count < 2)
                continue;

            List<Pm4MsurEntry> surfaces = [.. compSurfaces.Select(idx => chunks.Msur[idx])];
            List<Vector3> verts = Pm4PlacementMath.CollectSurfaceVertices(chunks.Msvt, chunks.Msvi, surfaces);
            if (verts.Count < 4)
                continue;

            List<Vector3> placedVerts = new(verts.Count);
            Vector3 min = new(float.MaxValue), max = new(float.MinValue), sum = Vector3.Zero;
            foreach (Vector3 v in verts)
            {
                Vector3 p = Pm4CoordinateService.Pm4LocalToAdtPlacement(v);
                placedVerts.Add(p);
                min = Vector3.Min(min, p);
                max = Vector3.Max(max, p);
                sum += p;
            }

            Vector3 centroid = sum / verts.Count;
            Vector3 extent = max - min;

            // Doodads are bounded in size (e.g. max extent <= 40 yards)
            if (extent.X > 40f || extent.Y > 40f || extent.Z > 40f)
                continue;

            groups.Add(new Pm4RawObjectGroup(
                0u,
                centroid.Z,
                compSurfaces.Count,
                verts.Count,
                min,
                max,
                centroid,
                placedVerts));
        }

        return groups;
    }

    private static Dictionary<string, AdtPlacementCatalog> LoadNeighborCatalogs(
        string mapDirectory,
        int centerFirst,
        int centerSecond,
        Func<string, AdtPlacementCatalog>? neighborCatalogReader)
    {
        var catalogs = new Dictionary<string, AdtPlacementCatalog>(StringComparer.OrdinalIgnoreCase);
        if (neighborCatalogReader is null)
            return catalogs;

        for (int df = -1; df <= 1; df++)
        {
            for (int ds = -1; ds <= 1; ds++)
            {
                if (df == 0 && ds == 0)
                    continue;

                int nf = centerFirst + df;
                int ns = centerSecond + ds;
                if (nf < 0 || nf >= 64 || ns < 0 || ns >= 64)
                    continue;

                // Probe for candidate files in mapDirectory: monolithic .adt, split _obj0.adt, unpadded and padded
                string[] candidates =
                [
                    Path.Combine(mapDirectory, $"development_{nf}_{ns}.adt"),
                    Path.Combine(mapDirectory, $"development_{nf}_{ns}_obj0.adt"),
                    Path.Combine(mapDirectory, $"development_{nf:02d}_{ns:02d}.adt"),
                    Path.Combine(mapDirectory, $"development_{nf:02d}_{ns:02d}_obj0.adt")
                ];

                string? companion = candidates.FirstOrDefault(File.Exists);

                if (companion is not null)
                {
                    try
                    {
                        AdtPlacementCatalog cat = neighborCatalogReader(companion);
                        // Filter out dummy Noggit OILPLATFORM_LOW placeholders
                        var filteredWmos = cat.WorldModelPlacements
                            .Where(w => !w.ModelPath.Contains("OILPLATFORM_LOW", StringComparison.OrdinalIgnoreCase))
                            .ToList();

                        if (filteredWmos.Count > 0 || cat.ModelPlacements.Count > 0)
                        {
                            catalogs[Path.GetFileName(companion)] = new AdtPlacementCatalog(
                                cat.SourcePath,
                                cat.Kind,
                                cat.ModelNames,
                                cat.WorldModelNames,
                                cat.ModelPlacements,
                                filteredWmos);
                        }
                    }
                    catch
                    {
                        // Ignore unreadable companion
                    }
                }
            }
        }

        return catalogs;
    }

    /// <summary>
    /// Computes the principal 2D yaw orientation (in degrees) of a set of surface vertices
    /// using covariance analysis on the horizontal placement plane (X, Y).
    /// </summary>
    public static float ComputeFootprintYawDegrees(IReadOnlyList<Vector3> vertices)
    {
        if (vertices == null || vertices.Count < 3)
            return 0f;

        float sumX = 0f, sumY = 0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            sumX += vertices[i].X;
            sumY += vertices[i].Y;
        }
        float meanX = sumX / vertices.Count;
        float meanY = sumY / vertices.Count;

        float cxx = 0f, cyy = 0f, cxy = 0f;
        for (int i = 0; i < vertices.Count; i++)
        {
            float dx = vertices[i].X - meanX;
            float dy = vertices[i].Y - meanY;
            cxx += dx * dx;
            cyy += dy * dy;
            cxy += dx * dy;
        }

        if (MathF.Abs(cxx - cyy) < 1e-4f && MathF.Abs(cxy) < 1e-4f)
            return 0f;

        float angleRad = 0.5f * MathF.Atan2(2f * cxy, cxx - cyy);
        float angleDeg = angleRad * (180f / MathF.PI);

        while (angleDeg > 180f) angleDeg -= 360f;
        while (angleDeg < -180f) angleDeg += 360f;

        return angleDeg;
    }

    /// <summary>
    /// Computes the principal horizontal yaw (in degrees around Z) from the PM4 vertices,
    /// aligned to the canonical forward orientation of the model.
    /// </summary>
    public static float ComputeModelAlignedYaw(IReadOnlyList<Vector3> vertices, Vector3 modelSpan)
    {
        float footprintYaw = ComputeFootprintYawDegrees(vertices);

        // In WoW model coordinates, if the model's primary axis is along local Y instead of local X:
        if (MathF.Abs(modelSpan.Y) > MathF.Abs(modelSpan.X) * 1.15f)
        {
            footprintYaw += 90f;
            while (footprintYaw > 180f) footprintYaw -= 360f;
            while (footprintYaw < -180f) footprintYaw += 360f;
        }

        return footprintYaw;
    }

    /// <summary>
    /// Magnetically attracts an object's placement origin so that the model's geometric center
    /// coincides with the PM4 collision mesh centroid, correcting for local model coordinate offsets.
    /// </summary>
    public static Vector3 AttractPositionToPm4Mesh(
        Vector3 pm4Centroid,
        float targetZ,
        float yawDegrees,
        Vector3 modelBoundsMin,
        Vector3 modelBoundsMax)
    {
        Vector3 modelCenter = (modelBoundsMin + modelBoundsMax) * 0.5f;
        float rotZRad = yawDegrees * (MathF.PI / 180f);
        Matrix4x4 rotMatrix = Matrix4x4.CreateRotationZ(MathF.PI) * Matrix4x4.CreateRotationZ(rotZRad);
        Vector3 rotatedCenter = Vector3.Transform(modelCenter, rotMatrix);

        float z = targetZ != 0f ? targetZ : (pm4Centroid.Z - rotatedCenter.Z);
        return new Vector3(
            pm4Centroid.X - rotatedCenter.X,
            pm4Centroid.Y - rotatedCenter.Y,
            z);
    }

    private static Vector3 ShapeOf(Vector3 min, Vector3 max)
    {
        Vector3 span = max - min;
        float sx = MathF.Abs(span.X);
        float sy = MathF.Abs(span.Y);
        float sz = MathF.Abs(span.Z);
        return new Vector3(MathF.Min(sx, sy), MathF.Max(sx, sy), sz);
    }

    private static double ShapeDistance(Vector3 a, Vector3 b)
    {
        double dx = Math.Abs(a.X - b.X) / Math.Max(0.5, Math.Max(a.X, b.X));
        double dy = Math.Abs(a.Y - b.Y) / Math.Max(0.5, Math.Max(a.Y, b.Y));
        double dz = Math.Abs(a.Z - b.Z) / Math.Max(0.5, Math.Max(a.Z, b.Z));
        return (dx + dy + dz * 2.0) / 4.0;
    }

    private static bool BoxesOverlap2D(Vector3 minA, Vector3 maxA, Vector3 minB, Vector3 maxB, float padding)
    {
        return !(maxA.X + padding < minB.X || minA.X - padding > maxB.X ||
                 maxA.Y + padding < minB.Y || minA.Y - padding > maxB.Y);
    }

    private readonly record struct Pm4RawObjectGroup(
        uint Key,
        float PlacementZ,
        int SurfaceCount,
        int VertexCount,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        Vector3 Centroid,
        List<Vector3> Vertices);
}
