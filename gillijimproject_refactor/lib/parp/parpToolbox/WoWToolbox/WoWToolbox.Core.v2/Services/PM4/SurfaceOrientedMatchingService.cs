using System.Numerics;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.Core.v2.Services.PM4;

namespace WoWToolbox.Core.v2.Services.PM4;

/// <summary>
/// Service for matching PM4 surfaces to WMO assets using surface orientation analysis.
/// Enables precise top-surface to top-surface and bottom-surface to bottom-surface matching.
/// </summary>
public class SurfaceOrientedMatchingService
{
    /// <summary>
    /// WMO asset with surface analysis data
    /// </summary>
    public class WmoAssetWithSurfaces
    {
        public string FileName { get; set; } = "";
        public string Category { get; set; } = "";
        public int VertexCount { get; set; }
        public BoundingBox3D BoundingBox { get; set; }
        public WMOSurfaceProfile SurfaceProfile { get; set; } = new();
    }

    /// <summary>
    /// Surface analysis profile for a WMO asset
    /// </summary>
    public class WMOSurfaceProfile
    {
        public BoundingBox3D? TopSurfaceBounds { get; set; }
        public BoundingBox3D? BottomSurfaceBounds { get; set; }
        public float TopSurfaceArea { get; set; }
        public float BottomSurfaceArea { get; set; }
        public float TopComplexity { get; set; }
        public float BottomComplexity { get; set; }
        public Vector3 PrimaryTopNormal { get; set; }
        public Vector3 PrimaryBottomNormal { get; set; }
    }

    /// <summary>
    /// Result of surface-to-surface matching
    /// </summary>
    public class SurfaceMatchResult
    {
        public WmoAssetWithSurfaces WmoAsset { get; set; } = new();
        public float TopSurfaceCorrelation { get; set; }
        public float BottomSurfaceCorrelation { get; set; }
        public float OrientationCompatibility { get; set; }
        public float OverallConfidence { get; set; }
        public List<string> MatchFeatures { get; set; } = new();
        public string MatchReason { get; set; } = "";
        public MSURSurfaceExtractionService.SurfaceOrientation MatchedOrientation { get; set; }
    }

    private readonly List<WmoAssetWithSurfaces> _wmoDatabase = new();

    /// <summary>
    /// Load WMO database with surface analysis
    /// </summary>
    public void LoadWMODatabase(List<WmoAssetWithSurfaces> wmoAssets)
    {
        _wmoDatabase.Clear();
        _wmoDatabase.AddRange(wmoAssets);
    }

    /// <summary>
    /// Find WMO matches for a PM4 surface using orientation-aware matching
    /// </summary>
    public List<SurfaceMatchResult> FindWMOMatchesBySurface(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        int maxResults = 10)
    {
        var matches = new List<SurfaceMatchResult>();

        foreach (var wmo in _wmoDatabase)
        {
            var match = AnalyzeSurfaceMatch(pm4Surface, wmo);
            if (match.OverallConfidence > 0.1f) // Minimum threshold
            {
                matches.Add(match);
            }
        }

        return matches
            .OrderByDescending(m => m.OverallConfidence)
            .Take(maxResults)
            .ToList();
    }

    /// <summary>
    /// Analyze surface-to-surface match using orientation and geometric similarity
    /// </summary>
    public SurfaceMatchResult AnalyzeSurfaceMatch(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WmoAssetWithSurfaces wmo)
    {
        var result = new SurfaceMatchResult
        {
            WmoAsset = wmo,
            MatchedOrientation = pm4Surface.Orientation
        };

        try
        {
            // 1. Orientation-Specific Matching (40% weight)
            switch (pm4Surface.Orientation)
            {
                case MSURSurfaceExtractionService.SurfaceOrientation.TopFacing:
                    result.TopSurfaceCorrelation = AnalyzeTopSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures);
                    result.BottomSurfaceCorrelation = 0f;
                    break;
                    
                case MSURSurfaceExtractionService.SurfaceOrientation.BottomFacing:
                    result.TopSurfaceCorrelation = 0f;
                    result.BottomSurfaceCorrelation = AnalyzeBottomSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures);
                    break;
                    
                case MSURSurfaceExtractionService.SurfaceOrientation.Vertical:
                    // For vertical surfaces, consider both but weight differently
                    result.TopSurfaceCorrelation = AnalyzeTopSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures) * 0.3f;
                    result.BottomSurfaceCorrelation = AnalyzeBottomSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures) * 0.3f;
                    break;
                    
                default:
                    // Mixed orientation - analyze both
                    result.TopSurfaceCorrelation = AnalyzeTopSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures) * 0.5f;
                    result.BottomSurfaceCorrelation = AnalyzeBottomSurfaceMatch(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures) * 0.5f;
                    break;
            }

            // 2. Normal Vector Compatibility (30% weight)
            result.OrientationCompatibility = AnalyzeNormalCompatibility(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures);

            // 3. Surface Area Similarity (20% weight)
            var areaSimilarity = AnalyzeSurfaceAreaSimilarity(pm4Surface, wmo.SurfaceProfile, result.MatchFeatures);

            // 4. Boundary Box Compatibility (10% weight)
            var boundsSimilarity = AnalyzeBoundsSimilarity(pm4Surface, wmo, result.MatchFeatures);

            // Calculate overall confidence with orientation-aware weighting
            var surfaceMatch = Math.Max(result.TopSurfaceCorrelation, result.BottomSurfaceCorrelation);
            result.OverallConfidence = 
                (surfaceMatch * 0.4f) +
                (result.OrientationCompatibility * 0.3f) +
                (areaSimilarity * 0.2f) +
                (boundsSimilarity * 0.1f);

            result.MatchReason = GenerateMatchReason(result);

            return result;
        }
        catch (Exception ex)
        {
            result.MatchFeatures.Add($"ERROR: {ex.Message}");
            result.OverallConfidence = 0.0f;
            result.MatchReason = "Surface analysis failed";
            return result;
        }
    }

    /// <summary>
    /// Analyze top surface matching (roofs, upper geometry)
    /// </summary>
    private float AnalyzeTopSurfaceMatch(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WMOSurfaceProfile wmoProfile,
        List<string> features)
    {
        if (!wmoProfile.TopSurfaceBounds.HasValue || wmoProfile.TopSurfaceArea <= 0)
        {
            features.Add("WMO has no identifiable top surface");
            return 0.1f;
        }

        if (!pm4Surface.SurfaceBounds.HasValue)
        {
            features.Add("PM4 surface has no valid bounds");
            return 0.1f;
        }

        // Compare top surface dimensions
        var pm4Size = pm4Surface.SurfaceBounds.Value.Size;
        var wmoTopSize = wmoProfile.TopSurfaceBounds.Value.Size;

        var dimensionalSimilarity = CalculateDimensionalSimilarity(pm4Size, wmoTopSize);
        
        // Compare surface areas
        var areaSimilarity = Math.Min(pm4Surface.SurfaceArea, wmoProfile.TopSurfaceArea) / 
                           Math.Max(pm4Surface.SurfaceArea, wmoProfile.TopSurfaceArea);

        var topMatch = (dimensionalSimilarity * 0.7f) + (areaSimilarity * 0.3f);

        features.Add($"Top surface match: Dimensional({dimensionalSimilarity:P1}), Area({areaSimilarity:P1})");
        return Math.Max(0f, topMatch);
    }

    /// <summary>
    /// Analyze bottom surface matching (foundations, walkable areas)
    /// </summary>
    private float AnalyzeBottomSurfaceMatch(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WMOSurfaceProfile wmoProfile,
        List<string> features)
    {
        if (!wmoProfile.BottomSurfaceBounds.HasValue || wmoProfile.BottomSurfaceArea <= 0)
        {
            features.Add("WMO has no identifiable bottom surface");
            return 0.1f;
        }

        if (!pm4Surface.SurfaceBounds.HasValue)
        {
            features.Add("PM4 surface has no valid bounds");
            return 0.1f;
        }

        // Compare bottom surface dimensions
        var pm4Size = pm4Surface.SurfaceBounds.Value.Size;
        var wmoBottomSize = wmoProfile.BottomSurfaceBounds.Value.Size;

        var dimensionalSimilarity = CalculateDimensionalSimilarity(pm4Size, wmoBottomSize);
        
        // Compare surface areas
        var areaSimilarity = Math.Min(pm4Surface.SurfaceArea, wmoProfile.BottomSurfaceArea) / 
                           Math.Max(pm4Surface.SurfaceArea, wmoProfile.BottomSurfaceArea);

        var bottomMatch = (dimensionalSimilarity * 0.7f) + (areaSimilarity * 0.3f);

        features.Add($"Bottom surface match: Dimensional({dimensionalSimilarity:P1}), Area({areaSimilarity:P1})");
        return Math.Max(0f, bottomMatch);
    }

    /// <summary>
    /// Analyze normal vector compatibility between surfaces
    /// </summary>
    private float AnalyzeNormalCompatibility(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WMOSurfaceProfile wmoProfile,
        List<string> features)
    {
        var pm4Normal = Vector3.Normalize(pm4Surface.SurfaceNormal);
        
        Vector3 wmoNormal;
        string comparisonType;
        
        if (pm4Surface.Orientation == MSURSurfaceExtractionService.SurfaceOrientation.TopFacing)
        {
            wmoNormal = wmoProfile.PrimaryTopNormal;
            comparisonType = "top";
        }
        else if (pm4Surface.Orientation == MSURSurfaceExtractionService.SurfaceOrientation.BottomFacing)
        {
            wmoNormal = wmoProfile.PrimaryBottomNormal;
            comparisonType = "bottom";
        }
        else
        {
            // For vertical/mixed, choose the better match
            var topDot = Vector3.Dot(pm4Normal, wmoProfile.PrimaryTopNormal);
            var bottomDot = Vector3.Dot(pm4Normal, wmoProfile.PrimaryBottomNormal);
            
            if (Math.Abs(topDot) > Math.Abs(bottomDot))
            {
                wmoNormal = wmoProfile.PrimaryTopNormal;
                comparisonType = "top-mixed";
            }
            else
            {
                wmoNormal = wmoProfile.PrimaryBottomNormal;
                comparisonType = "bottom-mixed";
            }
        }

        // Calculate normal alignment (dot product)
        var normalAlignment = Math.Abs(Vector3.Dot(pm4Normal, wmoNormal));
        
        features.Add($"Normal compatibility ({comparisonType}): {normalAlignment:P1}");
        return normalAlignment;
    }

    /// <summary>
    /// Analyze surface area similarity
    /// </summary>
    private float AnalyzeSurfaceAreaSimilarity(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WMOSurfaceProfile wmoProfile,
        List<string> features)
    {
        var relevantWmoArea = pm4Surface.Orientation == MSURSurfaceExtractionService.SurfaceOrientation.TopFacing 
            ? wmoProfile.TopSurfaceArea 
            : wmoProfile.BottomSurfaceArea;

        if (relevantWmoArea <= 0 || pm4Surface.SurfaceArea <= 0)
        {
            features.Add("Invalid surface areas for comparison");
            return 0.3f;
        }

        var areaSimilarity = Math.Min(pm4Surface.SurfaceArea, relevantWmoArea) / 
                           Math.Max(pm4Surface.SurfaceArea, relevantWmoArea);

        features.Add($"Surface area similarity: PM4({pm4Surface.SurfaceArea:F1}) vs WMO({relevantWmoArea:F1}) = {areaSimilarity:P1}");
        return areaSimilarity;
    }

    /// <summary>
    /// Analyze bounding box similarity
    /// </summary>
    private float AnalyzeBoundsSimilarity(
        MSURSurfaceExtractionService.SurfaceGeometry pm4Surface,
        WmoAssetWithSurfaces wmo,
        List<string> features)
    {
        if (!pm4Surface.SurfaceBounds.HasValue)
        {
            features.Add("PM4 surface missing bounds");
            return 0.3f;
        }

        var pm4Size = pm4Surface.SurfaceBounds.Value.Size;
        var wmoSize = wmo.BoundingBox.Size;

        var similarity = CalculateDimensionalSimilarity(pm4Size, wmoSize);
        
        features.Add($"Bounds similarity: PM4({pm4Size.X:F1}×{pm4Size.Y:F1}×{pm4Size.Z:F1}) vs WMO({wmoSize.X:F1}×{wmoSize.Y:F1}×{wmoSize.Z:F1}) = {similarity:P1}");
        return similarity;
    }

    /// <summary>
    /// Calculate dimensional similarity between two size vectors
    /// </summary>
    private float CalculateDimensionalSimilarity(Vector3 size1, Vector3 size2)
    {
        var xSim = 1.0f - Math.Min(Math.Abs(size1.X - size2.X) / Math.Max(size1.X, size2.X), 0.9f);
        var ySim = 1.0f - Math.Min(Math.Abs(size1.Y - size2.Y) / Math.Max(size1.Y, size2.Y), 0.9f);
        var zSim = 1.0f - Math.Min(Math.Abs(size1.Z - size2.Z) / Math.Max(size1.Z, size2.Z), 0.9f);

        return (xSim + ySim + zSim) / 3f;
    }

    /// <summary>
    /// Generate human-readable match reason
    /// </summary>
    private string GenerateMatchReason(SurfaceMatchResult result)
    {
        var reasons = new List<string>();

        var surfaceMatch = Math.Max(result.TopSurfaceCorrelation, result.BottomSurfaceCorrelation);
        var surfaceType = result.TopSurfaceCorrelation > result.BottomSurfaceCorrelation ? "top" : "bottom";

        if (surfaceMatch > 0.8f) reasons.Add($"Excellent {surfaceType} surface match");
        else if (surfaceMatch > 0.6f) reasons.Add($"Good {surfaceType} surface match");
        else if (surfaceMatch > 0.4f) reasons.Add($"Moderate {surfaceType} surface match");
        else reasons.Add($"Poor {surfaceType} surface match");

        if (result.OrientationCompatibility > 0.8f) reasons.Add("Strong normal alignment");
        else if (result.OrientationCompatibility > 0.6f) reasons.Add("Good normal alignment");

        return string.Join(", ", reasons);
    }

    /// <summary>
    /// Create WMO surface profile from basic geometry data
    /// This is a simplified version - in production you'd analyze actual WMO geometry
    /// </summary>
    public static WMOSurfaceProfile CreateBasicSurfaceProfile(BoundingBox3D bounds, int vertexCount)
    {
        // Simplified surface analysis - in production this would analyze actual WMO vertices
        var topBounds = new BoundingBox3D(
            new Vector3(bounds.Min.X, bounds.Max.Y - bounds.Size.Y * 0.1f, bounds.Min.Z),
            bounds.Max
        );

        var bottomBounds = new BoundingBox3D(
            bounds.Min,
            new Vector3(bounds.Max.X, bounds.Min.Y + bounds.Size.Y * 0.1f, bounds.Max.Z)
        );

        return new WMOSurfaceProfile
        {
            TopSurfaceBounds = topBounds,
            BottomSurfaceBounds = bottomBounds,
            TopSurfaceArea = topBounds.Size.X * topBounds.Size.Z,
            BottomSurfaceArea = bottomBounds.Size.X * bottomBounds.Size.Z,
            TopComplexity = vertexCount / 100f,
            BottomComplexity = vertexCount / 200f, // Foundations typically simpler
            PrimaryTopNormal = Vector3.UnitY,
            PrimaryBottomNormal = -Vector3.UnitY
        };
    }
} 