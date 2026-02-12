using System.Numerics;
using WoWRollback.Core.Services.PM4;

namespace WoWRollback.PM4Module.Decoding;

/// <summary>
/// Discovers MISSING objects by comparing MSCN collision data (grouped by CK24) 
/// against existing ADT placements. Uses CK24 linkage via MdosIndex for proper object grouping.
/// </summary>
public static class MscnObjectDiscovery
{
    /// <summary>
    /// Represents an MSCN cluster that might be a missing object.
    /// Now includes CK24 identifier for proper object grouping.
    /// </summary>
    public record MscnCluster(
        uint CK24,
        int TileX,
        int TileY,
        List<Vector3> MscnVertices,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        Vector3 Centroid,
        Vector3 Dimensions
    );
    
    /// <summary>
    /// Represents an existing ADT placement (WMO or M2).
    /// </summary>
    public record ExistingPlacement(
        string Path,
        Vector3 Position,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        bool IsWmo
    );

    /// <summary>
    /// Extract MSCN clusters GROUPED BY CK24 from a PM4 file.
    /// Uses MdosIndex to link surfaces to MSCN vertices for each CK24 object.
    /// Filters out clusters that overlap with existing ADT placements.
    /// </summary>
    public static List<MscnCluster> ExtractMissingObjects(
        Pm4FileStructure pm4, 
        int tileX, 
        int tileY,
        List<ExistingPlacement> existingPlacements,
        float overlapThreshold = 0.5f)
    {
        var missing = new List<MscnCluster>();
        
        if (pm4.SceneNodes == null || pm4.SceneNodes.Count == 0)
            return missing;
        if (pm4.Surfaces == null || pm4.Surfaces.Count == 0)
            return missing;
        
        // Group surfaces by CK24
        var ck24Groups = pm4.Surfaces
            .GroupBy(s => s.CK24)
            .ToDictionary(g => g.Key, g => g.ToList());
        
        foreach (var (ck24, surfaces) in ck24Groups)
        {
            // Skip terrain/nav mesh (CK24 = 0)
            if (ck24 == 0) continue;
            
            // Collect MSCN vertices linked via MdosIndex for this CK24
            var mscnVertices = new List<Vector3>();
            foreach (var surf in surfaces)
            {
                if (surf.MdosIndex < pm4.SceneNodes.Count)
                {
                    mscnVertices.Add(pm4.SceneNodes[(int)surf.MdosIndex]);
                }
            }
            
            if (mscnVertices.Count < 3) continue;
            
            // Calculate bounds
            var (min, max) = CalculateBounds(mscnVertices);
            var dims = max - min;
            
            // DEBUG: Show raw MSCN values vs expected ADT tile bounds
            // Only print for first few clusters per tile to avoid spam
            bool shouldDebug = (ck24 != 0) && (mscnVertices.Count > 10);
            if (shouldDebug)
            {
                // Expected ADT tile bounds (world coords)
                const float TileSize = 533.33333f;
                const float MapCenter = 17066.66656f;
                float tileWorldMinX = MapCenter - (tileX + 1) * TileSize;
                float tileWorldMaxX = MapCenter - tileX * TileSize;
                float tileWorldMinY = MapCenter - (tileY + 1) * TileSize;
                float tileWorldMaxY = MapCenter - tileY * TileSize;
                
                Console.WriteLine($"[DEBUG MSCN] Tile {tileX}_{tileY} CK24=0x{ck24:X6} ({mscnVertices.Count} verts)");
                Console.WriteLine($"[DEBUG MSCN]   Expected ADT X range: {tileWorldMinX:F0} to {tileWorldMaxX:F0}");
                Console.WriteLine($"[DEBUG MSCN]   Expected ADT Y range: {tileWorldMinY:F0} to {tileWorldMaxY:F0}");
                Console.WriteLine($"[DEBUG MSCN]   RAW MSCN X: {min.X:F1} to {max.X:F1} (range: {dims.X:F1})");
                Console.WriteLine($"[DEBUG MSCN]   RAW MSCN Y: {min.Y:F1} to {max.Y:F1} (range: {dims.Y:F1})");
                Console.WriteLine($"[DEBUG MSCN]   RAW MSCN Z: {min.Z:F1} to {max.Z:F1} (range: {dims.Z:F1})");
            }
            
            // Skip tiny noise or huge terrain
            if (dims.X < 0.5f && dims.Y < 0.5f && dims.Z < 0.5f) continue;
            if (dims.X > 300f || dims.Y > 300f) continue; // Object-scale, not terrain
            
            var centroid = (min + max) / 2f;
            
            // MSCN data is ALREADY in world coordinates (not tile-local)
            // Just need axis swap: MSCN_X=WoW_Y, MSCN_Y=WoW_X
            var worldCentroid = PipelineCoordinateService.MscnToAdtPosition(centroid);
            var worldMin = PipelineCoordinateService.MscnToAdtPosition(min);
            var worldMax = PipelineCoordinateService.MscnToAdtPosition(max);
            
            if (shouldDebug)
            {
                Console.WriteLine($"[DEBUG MSCN]   RAW centroid: ({centroid.X:F1}, {centroid.Y:F1}, {centroid.Z:F1})");
                Console.WriteLine($"[DEBUG MSCN]   TRANSFORMED: ({worldCentroid.X:F1}, {worldCentroid.Y:F1}, {worldCentroid.Z:F1})");
            }
            
            // Fix min/max after transform (coords may flip due to axis conversion)
            var actualWorldMin = Vector3.Min(worldMin, worldMax);
            var actualWorldMax = Vector3.Max(worldMin, worldMax);
            
            var cluster = new MscnCluster(
                CK24: ck24,
                TileX: tileX,
                TileY: tileY,
                MscnVertices: mscnVertices,
                BoundsMin: actualWorldMin,
                BoundsMax: actualWorldMax,
                Centroid: worldCentroid,
                Dimensions: dims  // Keep local dims for shape matching
            );
            
            // Check if this cluster overlaps with any existing placement (NOW in same coord space)
            bool isExisting = false;
            foreach (var existing in existingPlacements)
            {
                if (BoundsOverlap(actualWorldMin, actualWorldMax, existing.BoundsMin, existing.BoundsMax, overlapThreshold))
                {
                    isExisting = true;
                    break;
                }
            }
            
            if (!isExisting)
            {
                missing.Add(cluster);
            }
        }
        
        return missing;
    }
    
    /// <summary>
    /// Match an MSCN cluster against WMO/M2 geometry library.
    /// Returns the best match with confidence score.
    /// </summary>
    public static (string? AssetPath, float Confidence, bool IsWmo) MatchClusterToAsset(
        MscnCluster cluster,
        List<Pm4ModfReconstructor.WmoReference> wmoLibrary,
        List<Pm4ModfReconstructor.M2Reference> m2Library,
        float minConfidence = 0.70f)
    {
        string? bestPath = null;
        float bestScore = 0f;
        bool isWmo = false;
        
        var clusterDims = new[] { cluster.Dimensions.X, cluster.Dimensions.Y, cluster.Dimensions.Z }
            .OrderByDescending(x => x).ToArray();
        
        // Match against WMOs - use PrincipalExtents if Dimensions is zero
        foreach (var wmo in wmoLibrary)
        {
            float[] assetDims;
            if (wmo.Stats.Dimensions.X > 0.1f)
                assetDims = new[] { wmo.Stats.Dimensions.X, wmo.Stats.Dimensions.Y, wmo.Stats.Dimensions.Z };
            else
                assetDims = wmo.Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();
            
            float score = CompareDimensionsArray(clusterDims, assetDims);
            if (score > bestScore && score >= minConfidence)
            {
                bestScore = score;
                bestPath = wmo.WmoPath;
                isWmo = true;
            }
        }
        
        // Match against M2s - use PrincipalExtents since Dimensions doesn't serialize
        foreach (var m2 in m2Library)
        {
            float[] assetDims;
            if (m2.Stats.Dimensions.X > 0.1f)
                assetDims = new[] { m2.Stats.Dimensions.X, m2.Stats.Dimensions.Y, m2.Stats.Dimensions.Z };
            else
                assetDims = m2.Stats.PrincipalExtents.OrderByDescending(x => x).ToArray();
            
            float score = CompareDimensionsArray(clusterDims, assetDims);
            if (score > bestScore && score >= minConfidence)
            {
                bestScore = score;
                bestPath = m2.M2Path;
                isWmo = false;
            }
        }
        
        return (bestPath, bestScore, isWmo);
    }
    
    private static float CompareDimensionsArray(float[] clusterDims, float[] assetDims)
    {
        if (assetDims.Length < 3 || clusterDims.Length < 3) return 0f;
        if (assetDims[0] < 0.1f || clusterDims[0] < 0.1f) return 0f;
        
        float clusterScale = clusterDims[0];
        float assetScale = assetDims[0];
        
        // CRITICAL: Reject if absolute sizes are too different
        // Allow maximum 2x scale difference for WMOs (they can't be scaled)
        float sizeRatio = clusterScale / assetScale;
        if (sizeRatio < 0.5f || sizeRatio > 2.0f)
            return 0f;  // Size mismatch - small building can't match dungeon!
        
        // Also reject objects that are too large (dungeons/raids are typically 200+ units)
        // MSCN building clusters should be under 150 units in their largest dimension
        if (assetScale > 200f && clusterScale < 100f)
            return 0f;  // Asset is way too big for this cluster
        
        // Now compare shape ratios
        float[] clusterNorm = { 1f, clusterDims[1] / clusterScale, clusterDims[2] / clusterScale };
        float[] assetNorm = { 1f, assetDims[1] / assetScale, assetDims[2] / assetScale };
        
        // Compare normalized ratios
        float diff1 = Math.Abs(clusterNorm[0] - assetNorm[0]); // Always 0
        float diff2 = Math.Abs(clusterNorm[1] - assetNorm[1]);
        float diff3 = Math.Abs(clusterNorm[2] - assetNorm[2]);
        
        float avgDiff = (diff1 + diff2 + diff3) / 3f;
        float shapeScore = Math.Max(0f, 1f - avgDiff * 2f);
        
        // Bonus for close absolute size match
        float sizeDiff = Math.Abs(1f - sizeRatio);
        float sizeScore = Math.Max(0f, 1f - sizeDiff);
        
        // Combined score: 70% shape, 30% size match
        return shapeScore * 0.7f + sizeScore * 0.3f;
    }
    
    private static float CompareDimensions(float[] clusterDims, Vector3 assetDimVec)
    {
        var assetDims = new[] { assetDimVec.X, assetDimVec.Y, assetDimVec.Z }
            .OrderByDescending(x => x).ToArray();
        
        if (assetDims[0] < 0.1f || clusterDims[0] < 0.1f) return 0f;
        
        // Normalize to largest dimension (compare SHAPE, not absolute size)
        float clusterScale = clusterDims[0];
        float assetScale = assetDims[0];
        
        float[] clusterNorm = { 1f, clusterDims[1] / clusterScale, clusterDims[2] / clusterScale };
        float[] assetNorm = { 1f, assetDims[1] / assetScale, assetDims[2] / assetScale };
        
        // Compare normalized ratios
        float diff1 = Math.Abs(clusterNorm[0] - assetNorm[0]); // Always 0
        float diff2 = Math.Abs(clusterNorm[1] - assetNorm[1]);
        float diff3 = Math.Abs(clusterNorm[2] - assetNorm[2]);
        
        float avgDiff = (diff1 + diff2 + diff3) / 3f;
        return Math.Max(0f, 1f - avgDiff * 2f);
    }
    
    private static bool BoundsOverlap(Vector3 aMin, Vector3 aMax, Vector3 bMin, Vector3 bMax, float threshold)
    {
        // Check if bounding boxes overlap by at least threshold %
        float overlapX = Math.Max(0, Math.Min(aMax.X, bMax.X) - Math.Max(aMin.X, bMin.X));
        float overlapY = Math.Max(0, Math.Min(aMax.Y, bMax.Y) - Math.Max(aMin.Y, bMin.Y));
        float overlapZ = Math.Max(0, Math.Min(aMax.Z, bMax.Z) - Math.Max(aMin.Z, bMin.Z));
        
        float aVol = (aMax.X - aMin.X) * (aMax.Y - aMin.Y) * Math.Max(1f, aMax.Z - aMin.Z);
        float bVol = (bMax.X - bMin.X) * (bMax.Y - bMin.Y) * Math.Max(1f, bMax.Z - bMin.Z);
        float overlapVol = overlapX * overlapY * Math.Max(1f, overlapZ);
        
        float minVol = Math.Min(aVol, bVol);
        if (minVol < 0.001f) return false;
        
        return (overlapVol / minVol) >= threshold;
    }
    
    private static (Vector3 min, Vector3 max) CalculateBounds(List<Vector3> vertices)
    {
        var min = new Vector3(float.MaxValue);
        var max = new Vector3(float.MinValue);
        
        foreach (var v in vertices)
        {
            min = Vector3.Min(min, v);
            max = Vector3.Max(max, v);
        }
        
        return (min, max);
    }
}
