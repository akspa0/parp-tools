using System.Numerics;
using System.Text.Json;
using WoWViewer.Logging;
using WoWViewer.Terrain;
using WowViewer.Core.PM4.Footprint;
using WowViewer.Core.PM4.Matching;
using ObjectInstance = WowViewer.Core.Runtime.World.WorldObjectInstance;

namespace WoWViewer;

/// <summary>
/// Information about a similar or identical PM4 hull identified across loaded tiles.
/// </summary>
internal sealed record SimilarPm4HullInfo(
    int TileX,
    int TileY,
    Pm4OverlayObject Object,
    float SimilarityScore,
    bool IsExactGeometry,
    bool IsOccupied,
    float BestYawOffsetDegrees = 0f);

/// <summary>
/// A curated spatial relationship between an ADT scene model and a PM4 collision hull.
/// Human curation confirms the relationship, which is then used to automate matching across all tiles.
/// </summary>
internal sealed class CuratedSpatialRelationship
{
    public string ModelName { get; set; } = "";
    public string ModelPath { get; set; } = "";
    public string AssetKind { get; set; } = "M2";
    public uint Ck24 { get; set; }
    public byte DominantGroupKey { get; set; }
    public float Length { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
    public float SolvedScale { get; set; } = 1.0f;
    public float YawOffset { get; set; }
    public Vector3 PositionOffset { get; set; }
    public float FitConfidence { get; set; }
    public DateTime DetectedUtc { get; set; } = DateTime.UtcNow;
    public bool Verified { get; set; }
    public float[]? VerticalSliceRadii { get; set; }
    public Vector3 InertiaTensor { get; set; }
    public Vector3 NormalDistribution { get; set; }

    public CuratedSpatialRelationship() { }

    public CuratedSpatialRelationship(
        string modelName,
        string modelPath,
        string assetKind,
        uint ck24,
        byte dominantGroupKey,
        float length,
        float width,
        float height,
        float solvedScale,
        float yawOffset,
        Vector3 positionOffset,
        float fitConfidence,
        DateTime detectedUtc,
        bool verified,
        float[]? verticalSliceRadii = null,
        Vector3 inertiaTensor = default,
        Vector3 normalDistribution = default)
    {
        ModelName = modelName;
        ModelPath = modelPath;
        AssetKind = assetKind;
        Ck24 = ck24;
        DominantGroupKey = dominantGroupKey;
        Length = length;
        Width = width;
        Height = height;
        SolvedScale = solvedScale;
        YawOffset = yawOffset;
        PositionOffset = positionOffset;
        FitConfidence = fitConfidence;
        DetectedUtc = detectedUtc;
        Verified = verified;
        VerticalSliceRadii = verticalSliceRadii;
        InertiaTensor = inertiaTensor;
        NormalDistribution = normalDistribution;
    }
}

/// <summary>
/// Service managing cross-tile PM4 shape identification, spatial relationship detection,
/// human curation logging, and automated cross-tile propagation.
/// </summary>
internal sealed class Pm4CurationAndSpatialMatchingService
{
    private readonly IViewerAppHost _host;
    private readonly Pm4DoodadDuplicationService _parent;
    private readonly List<CuratedSpatialRelationship> _curatedRelationships = new();

    // Cache to avoid scanning thousands of hulls every render frame
    private (int tileX, int tileY, uint ck24, int objectPartId)? _cachedSimilarKey;
    private int _cachedTileCount;
    private readonly List<SimilarPm4HullInfo> _cachedSimilarHulls = new();

    public IReadOnlyList<CuratedSpatialRelationship> CuratedRelationships => _curatedRelationships;

    public Pm4CurationAndSpatialMatchingService(IViewerAppHost host, Pm4DoodadDuplicationService parent)
    {
        _host = host;
        _parent = parent;
        LoadCuratedRelationshipsFromDisk();
    }

    /// <summary>
    /// Identifies similar or same PM4 object shapes across all loaded tiles for a target hull.
    /// Uses 3D geometry shape signatures (vertical cross sections, 3D inertia moments,
    /// surface normal distribution, and 3D Chamfer point set distance).
    /// </summary>
    public IReadOnlyList<SimilarPm4HullInfo> FindSimilarHullsAcrossTiles(
        Pm4OverlayObject target,
        int sourceTileX,
        int sourceTileY,
        WorldScene? worldScene)
    {
        if (worldScene == null || worldScene.Pm4Overlay._pm4TileObjects.Count == 0)
            return Array.Empty<SimilarPm4HullInfo>();

        var key = (sourceTileX, sourceTileY, target.Ck24, target.ObjectPartId);
        int tileCount = worldScene.Pm4Overlay._pm4TileObjects.Count;
        if (_cachedSimilarKey == key && _cachedTileCount == tileCount)
            return _cachedSimilarHulls;

        _cachedSimilarKey = key;
        _cachedTileCount = tileCount;
        _cachedSimilarHulls.Clear();

        var targetShape = target.Shape3D;
        float maxExtentTarget = MathF.Max(targetShape.Length, targetShape.Height);
        float occRadius = MathF.Max(2.5f, targetShape.Width * 0.75f);

        foreach (var (tileCoord, objects) in worldScene.Pm4Overlay._pm4TileObjects)
        {
            if (objects == null || objects.Count == 0) continue;

            for (int i = 0; i < objects.Count; i++)
            {
                var other = objects[i];
                if (ReferenceEquals(other, target)) continue;
                if (other.Ck24 == target.Ck24 && other.ObjectPartId == target.ObjectPartId
                    && tileCoord.tileX == sourceTileX && tileCoord.tileY == sourceTileY)
                    continue;

                bool exactCk = target.Ck24 != 0 && target.Ck24 == other.Ck24;

                var otherShape = other.Shape3D;
                float maxExtentOther = MathF.Max(otherShape.Length, otherShape.Height);
                float extentRatio = MathF.Max(maxExtentTarget, maxExtentOther) / MathF.Max(0.1f, MathF.Min(maxExtentTarget, maxExtentOther));

                // Quick pre-filter: if not exact Ck24 and max extent differs by more than 2.5x, skip Chamfer
                if (!exactCk && extentRatio > 2.5f)
                    continue;

                float shapeScore = Pm4Geometry3DShapeSignature.Compare3D(targetShape, otherShape, out float bestYawOffset);
                if (exactCk)
                {
                    shapeScore = MathF.Max(shapeScore, 0.95f);
                }

                bool isExactGeometry = exactCk || shapeScore >= 0.88f;

                if (shapeScore >= 0.65f || exactCk)
                {
                    bool isOccupied = IsHullOccupied(worldScene, other, tileCoord.tileX, tileCoord.tileY, occRadius);
                    _cachedSimilarHulls.Add(new SimilarPm4HullInfo(
                        tileCoord.tileX,
                        tileCoord.tileY,
                        other,
                        shapeScore,
                        isExactGeometry,
                        isOccupied,
                        bestYawOffset));
                }
            }
        }

        _cachedSimilarHulls.Sort((a, b) =>
        {
            if (a.IsExactGeometry != b.IsExactGeometry) return b.IsExactGeometry ? 1 : -1;
            return b.SimilarityScore.CompareTo(a.SimilarityScore);
        });

        return _cachedSimilarHulls;
    }

    /// <summary>
    /// Detects if an ADT scene object sits properly within spatial relationships with a PM4 hull.
    /// If an explicit instance is provided (e.g. human selection), it uses that instance.
    /// Otherwise, it scans resident placements on the tile to find an overlapping or enclosing object.
    /// Automatically logs detected relationships into the curation ledger.
    /// </summary>
    public bool TryDetectSpatialRelationship(
        Pm4OverlayObject pm4Obj,
        int pm4TileX,
        int pm4TileY,
        WorldScene? worldScene,
        out CuratedSpatialRelationship? relationship,
        ObjectInstance? explicitInst = null)
    {
        relationship = null;
        if (worldScene == null) return false;

        Vector3 pm4Centroid = Vector3.Transform(pm4Obj.Center, pm4Obj.BaseTransform);
        WorldScene.TransformBounds(pm4Obj.BoundsMin, pm4Obj.BoundsMax, pm4Obj.BaseTransform, out Vector3 pm4Min, out Vector3 pm4Max);
        Vector3 pm4Size = Vector3.Abs(pm4Obj.BoundsMax - pm4Obj.BoundsMin);
        float pm4Len = MathF.Max(pm4Size.X, pm4Size.Y);
        float pm4Wid = MathF.Min(pm4Size.X, pm4Size.Y);
        float pm4H = MathF.Max(0.01f, pm4Size.Z);

        ObjectInstance? candidate = explicitInst;
        float bestConfidence = explicitInst.HasValue ? 1.0f : 0f;

        // If no explicit instance was selected, scan resident placements on this tile
        if (!candidate.HasValue)
        {
            float searchRadius = MathF.Max(6.0f, MathF.Max(pm4Len, pm4Wid) * 0.90f);
            float searchRadiusSq = searchRadius * searchRadius;

            void CheckPlacements(List<ObjectInstance>? list)
            {
                if (list == null) return;
                for (int i = 0; i < list.Count; i++)
                {
                    var inst = list[i];
                    float distSq = Vector3.DistanceSquared(inst.PlacementPosition, pm4Centroid);
                    if (distSq > searchRadiusSq) continue;

                    float dist = MathF.Sqrt(distSq);
                    float centroidScore = 1.0f - Math.Clamp(dist / searchRadius, 0f, 1f);

                    // Local dimension similarity
                    Vector3 instLocalSize = Vector3.Abs(inst.LocalBoundsMax - inst.LocalBoundsMin);
                    float instLen = MathF.Max(instLocalSize.X, instLocalSize.Y) * inst.PlacementScale;
                    float instWid = MathF.Min(instLocalSize.X, instLocalSize.Y) * inst.PlacementScale;
                    float instAspect = instLen / MathF.Max(0.01f, instWid);
                    float pm4Aspect = pm4Len / MathF.Max(0.01f, pm4Wid);
                    float aspectDiff = MathF.Abs(instAspect - pm4Aspect) / MathF.Max(instAspect, pm4Aspect);
                    float aspectScore = 1.0f - Math.Clamp(aspectDiff, 0f, 1f);

                    float conf = centroidScore * 0.60f + aspectScore * 0.40f;
                    if (conf > bestConfidence)
                    {
                        bestConfidence = conf;
                        candidate = inst;
                    }
                }
            }

            IPm4OverlayHost pm4Host = worldScene;
            pm4Host.TileMdxInstances.TryGetValue((pm4TileX, pm4TileY), out var mdxList);
            pm4Host.TileWmoInstances.TryGetValue((pm4TileX, pm4TileY), out var wmoList);
            CheckPlacements(mdxList);
            CheckPlacements(wmoList);
        }

        if (candidate.HasValue && bestConfidence >= 0.55f)
        {
            var inst = candidate.Value;
            Vector3 posOffset = inst.PlacementPosition - pm4Centroid;

            // Extrude footprint prism to compute yaw reference
            var surfaceVerts = Pm4DoodadDuplicationService.ExtractCompositeSurfaceVertices(worldScene, pm4Obj, pm4TileX, pm4TileY, out float minWorldZ);
            worldScene.Pm4Overlay._pm4TilePositionRefs.TryGetValue((pm4TileX, pm4TileY), out var mprlList);
            float? placementZ = pm4Obj.Ck24 != 0 ? BitConverter.UInt32BitsToSingle(pm4Obj.Ck24 << 8) : null;
            Pm4FootprintPrism prism = Pm4FootprintExtruder.ExtrudeFootprint(surfaceVerts, mprlList, placementZ, minWorldZ);

            float yawOffset = inst.PlacementRotation.Z - prism.YawDegrees;
            while (yawOffset > 180f) yawOffset -= 360f;
            while (yawOffset < -180f) yawOffset += 360f;

            // Check if already in curated relationships
            var existing = _curatedRelationships.FirstOrDefault(r =>
                string.Equals(r.ModelPath, inst.ModelPath, StringComparison.OrdinalIgnoreCase) &&
                (r.Ck24 == pm4Obj.Ck24 || MathF.Abs(r.Length - pm4Len) < 1.0f));

            if (existing != null)
            {
                relationship = existing;
                return true;
            }

            bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(inst.ModelPath, inst.AssetKind);
            string assetKind = isWmo ? "WMO" : Pm4DoodadDuplicationService.ResolveAssetKind(inst.ModelPath, inst.AssetKind);
            float solvedScale = isWmo ? 1.0f : inst.PlacementScale;

            var shape3D = pm4Obj.Shape3D;
            var rel = new CuratedSpatialRelationship
            {
                ModelName = inst.ModelName,
                ModelPath = inst.ModelPath,
                AssetKind = assetKind,
                Ck24 = pm4Obj.Ck24,
                DominantGroupKey = pm4Obj.DominantGroupKey,
                Length = pm4Len,
                Width = pm4Wid,
                Height = pm4H,
                SolvedScale = solvedScale,
                YawOffset = yawOffset,
                PositionOffset = posOffset,
                FitConfidence = bestConfidence,
                DetectedUtc = DateTime.UtcNow,
                Verified = explicitInst.HasValue,
                VerticalSliceRadii = shape3D.VerticalSliceRadii,
                InertiaTensor = shape3D.InertiaTensor,
                NormalDistribution = shape3D.NormalDistribution
            };

            _curatedRelationships.RemoveAll(r =>
                string.Equals(r.ModelPath, rel.ModelPath, StringComparison.OrdinalIgnoreCase) && r.Ck24 == rel.Ck24);
            _curatedRelationships.Add(rel);
            SaveCuratedRelationshipsToDisk();

            relationship = rel;
            return true;
        }

        return false;
    }

    /// <summary>
    /// Confirms human curation of a spatial relationship and propagates it across all matching
    /// unoccupied PM4 hulls on ALL loaded tiles.
    /// </summary>
    public int VerifyAndPropagateAcrossAllTiles(
        CuratedSpatialRelationship relationship,
        WorldScene worldScene,
        out int distinctTilesPopulated)
    {
        relationship.Verified = true;
        SaveCuratedRelationshipsToDisk();

        bool isWmo = Pm4DoodadDuplicationService.IsWmoAsset(relationship.ModelPath, relationship.AssetKind);
        float scale = isWmo ? 1.0f : relationship.SolvedScale;
        string assetKind = isWmo ? "WMO" : Pm4DoodadDuplicationService.ResolveAssetKind(relationship.ModelPath, relationship.AssetKind);

        // 1. Ensure a TaughtPairing exists for this relationship in the automation library
        var localMin = -new Vector3(relationship.Length, relationship.Width, relationship.Height) * 0.5f;
        var localMax = new Vector3(relationship.Length, relationship.Width, relationship.Height) * 0.5f;

        var taught = new TaughtPairing(
            relationship.Ck24,
            relationship.DominantGroupKey,
            relationship.ModelName,
            relationship.ModelPath,
            assetKind,
            scale,
            0f,
            relationship.Length,
            relationship.Width,
            relationship.Height,
            DateTime.UtcNow,
            localMin,
            localMax,
            relationship.YawOffset,
            relationship.VerticalSliceRadii,
            relationship.InertiaTensor,
            relationship.NormalDistribution);

        _parent.TeachPairingExplicit(taught);

        // 2. Propagate across all loaded tiles
        int totalAdded = _parent.ApplyTaughtPairingAcrossAllTiles(taught);
        distinctTilesPopulated = worldScene.Pm4Overlay._pm4TileObjects.Count;

        string msg = $"✓ Verified & automated: Populated {totalAdded} matching PM4 hulls with {relationship.ModelName} across loaded tiles.";
        _host.StatusMessage = msg;
        _parent.LastOperationMessage = msg;

        return totalAdded;
    }

    private static bool IsHullOccupied(WorldScene worldScene, Pm4OverlayObject obj, int tileX, int tileY, float radius)
    {
        IPm4OverlayHost pm4Host = worldScene;
        WorldScene.TransformBounds(obj.BoundsMin, obj.BoundsMax, obj.BaseTransform, out Vector3 hullMin, out Vector3 hullMax);
        var target = new Pm4DoodadTarget(
            tileX,
            tileY,
            obj.Ck24,
            obj.ObjectPartId,
            Vector3.Transform(obj.Center, obj.BaseTransform),
            hullMin,
            hullMax,
            obj.BaseRotationRadians,
            obj.DominantGroupKey);

        var placements = new List<PlacementBounds>();
        Collect(tileX, tileY);
        if (tileX != tileY)
            Collect(tileY, tileX);

        return Pm4DoodadDuplicationEngine.IsTargetOccupied(target, placements, radius);

        void Collect(int tx, int ty)
        {
            if (pm4Host.TileMdxInstances.TryGetValue((tx, ty), out var mdxList) && mdxList != null)
            {
                for (int i = 0; i < mdxList.Count; i++)
                    placements.Add(new PlacementBounds(mdxList[i].BoundsMin, mdxList[i].BoundsMax, mdxList[i].PlacementPosition, mdxList[i].BoundsResolved));
            }

            if (pm4Host.TileWmoInstances.TryGetValue((tx, ty), out var wmoList) && wmoList != null)
            {
                for (int i = 0; i < wmoList.Count; i++)
                    placements.Add(new PlacementBounds(wmoList[i].BoundsMin, wmoList[i].BoundsMax, wmoList[i].PlacementPosition, wmoList[i].BoundsResolved));
            }
        }
    }

    private void SaveCuratedRelationshipsToDisk()
    {
        try
        {
            string outDir = Path.Combine(Directory.GetCurrentDirectory(), "output", "curation");
            Directory.CreateDirectory(outDir);
            string json = JsonSerializer.Serialize(_curatedRelationships, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(Path.Combine(outDir, "pm4_curated_relationships.json"), json);
        }
        catch { }
    }

    private void LoadCuratedRelationshipsFromDisk()
    {
        try
        {
            string filePath = Path.Combine(Directory.GetCurrentDirectory(), "output", "curation", "pm4_curated_relationships.json");
            if (File.Exists(filePath))
            {
                string json = File.ReadAllText(filePath);
                var loaded = JsonSerializer.Deserialize<List<CuratedSpatialRelationship>>(json);
                if (loaded != null && loaded.Count > 0)
                {
                    _curatedRelationships.Clear();
                    bool dirty = false;
                    for (int i = 0; i < loaded.Count; i++)
                    {
                        var r = loaded[i];
                        if (Pm4DoodadDuplicationService.IsWmoAsset(r.ModelPath, r.AssetKind))
                        {
                            if (!string.Equals(r.AssetKind, "WMO", StringComparison.OrdinalIgnoreCase) || r.SolvedScale != 1.0f)
                            {
                                r.AssetKind = "WMO";
                                r.SolvedScale = 1.0f;
                                dirty = true;
                            }
                        }
                        _curatedRelationships.Add(r);
                    }
                    if (dirty)
                    {
                        SaveCuratedRelationshipsToDisk();
                    }
                }
            }
        }
        catch { }
    }
}
