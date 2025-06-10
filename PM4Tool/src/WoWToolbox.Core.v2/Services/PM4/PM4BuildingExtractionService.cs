using System.Numerics;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.Core.v2.Services.PM4;

namespace WoWToolbox.Core.v2.Services.PM4;

/// <summary>
/// Enhanced PM4 building extraction service that integrates surface-oriented processing
/// with dual-format support for newer (MDOS) and legacy PM4 files.
/// </summary>
public class PM4BuildingExtractionService
{
    private readonly MSURSurfaceExtractionService _surfaceExtractor;
    private readonly SurfaceOrientedMatchingService _surfaceMatchingService;

    /// <summary>
    /// PM4 format detection results
    /// </summary>
    public enum PM4FormatVersion
    {
        Unknown,
        NewerWithMDOS,      // Has MDOS chunk (e.g., development_00_00)
        LegacyPreMDOS       // Pre-MDOS chunk format (most development files)
    }

    /// <summary>
    /// Extraction configuration for different PM4 formats
    /// </summary>
    public class ExtractionConfiguration
    {
        public PM4FormatVersion DetectedFormat { get; set; }
        public bool UseMSURSurfaceExtraction { get; set; } = true;
        public bool UseSurfaceOrientedMatching { get; set; } = true;
        public bool FallbackToLegacyExtraction { get; set; } = true;
        public float SpatialClusteringTolerance { get; set; } = 10.0f;
        public int MaxSurfaceGroupsPerBuilding { get; set; } = 50;
    }

    /// <summary>
    /// Results of PM4 building extraction
    /// </summary>
    public class ExtractionResult
    {
        public List<CompleteWMOModel> Buildings { get; set; } = new();
        public List<MSURSurfaceExtractionService.SurfaceGroup> SurfaceGroups { get; set; } = new();
        public PM4FormatVersion DetectedFormat { get; set; }
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = "";
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Individual surface-based navigation object (replaces blob-based objects)
    /// </summary>
    public class SurfaceBasedNavigationObject
    {
        public string PM4FileName { get; set; } = "";
        public int SurfaceGroupIndex { get; set; }
        public string ObjectId { get; set; } = "";
        public MSURSurfaceExtractionService.SurfaceGroup SurfaceGroup { get; set; } = new();
        public List<MSURSurfaceExtractionService.SurfaceGeometry> Surfaces { get; set; } = new();
        
        // Top/Bottom surface separation
        public List<MSURSurfaceExtractionService.SurfaceGeometry> TopSurfaces { get; set; } = new();
        public List<MSURSurfaceExtractionService.SurfaceGeometry> BottomSurfaces { get; set; } = new();
        public List<MSURSurfaceExtractionService.SurfaceGeometry> VerticalSurfaces { get; set; } = new();
        
        // Aggregate metrics
        public int TotalVertexCount { get; set; }
        public int TotalTriangleCount { get; set; }
        public BoundingBox3D? ObjectBounds { get; set; }
        public float TotalSurfaceArea { get; set; }
        public string EstimatedObjectType { get; set; } = "Unknown";
    }

    public PM4BuildingExtractionService()
    {
        _surfaceExtractor = new MSURSurfaceExtractionService();
        _surfaceMatchingService = new SurfaceOrientedMatchingService();
    }

    /// <summary>
    /// Extract buildings from PM4 file using surface-oriented approach
    /// </summary>
    public ExtractionResult ExtractAndExportBuildings(string pm4FilePath, string outputDirectory)
    {
        var result = new ExtractionResult();
        
        try
        {
            // 1. Load and analyze PM4 file
            var pm4File = PM4File.FromFile(pm4FilePath);
            var config = CreateExtractionConfiguration(pm4File);
            result.DetectedFormat = config.DetectedFormat;

            // 2. Extract surface groups using appropriate method
            if (config.UseMSURSurfaceExtraction && pm4File.MSUR?.Entries?.Any() == true)
            {
                result.SurfaceGroups = _surfaceExtractor.GroupSurfacesIntoBuildings(pm4File);
                result.Metadata["ExtractionMethod"] = "MSUR_Surface_Based";
            }
            else if (config.FallbackToLegacyExtraction)
            {
                result.SurfaceGroups = ExtractSurfaceGroupsLegacyMethod(pm4File);
                result.Metadata["ExtractionMethod"] = "Legacy_Fallback";
            }
            else
            {
                result.ErrorMessage = "No suitable extraction method available";
                return result;
            }

            // 3. Create buildings from surface groups
            var sourceFileName = Path.GetFileNameWithoutExtension(pm4FilePath);
            for (int i = 0; i < result.SurfaceGroups.Count; i++)
            {
                var building = _surfaceExtractor.CreateBuildingFromSurfaceGroup(
                    pm4File, result.SurfaceGroups[i], sourceFileName, i);
                
                if (building.HasGeometry)
                {
                    result.Buildings.Add(building);
                }
            }

            // 4. Set success metadata
            result.Success = true;
            result.Metadata["PM4_SourceFile"] = pm4FilePath;
            result.Metadata["BuildingCount"] = result.Buildings.Count;
            result.Metadata["SurfaceGroupCount"] = result.SurfaceGroups.Count;
            result.Metadata["DetectedFormat"] = config.DetectedFormat.ToString();

            return result;
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = ex.Message;
            result.Metadata["Exception"] = ex.ToString();
            return result;
        }
    }

    /// <summary>
    /// Extract surface-based navigation objects for enhanced matching
    /// </summary>
    public List<SurfaceBasedNavigationObject> ExtractSurfaceBasedNavigationObjects(string pm4FilePath)
    {
        var objects = new List<SurfaceBasedNavigationObject>();
        
        try
        {
            var pm4File = PM4File.FromFile(pm4FilePath);
            var surfaceGroups = _surfaceExtractor.GroupSurfacesIntoBuildings(pm4File);
            var sourceFileName = Path.GetFileNameWithoutExtension(pm4FilePath);

            for (int i = 0; i < surfaceGroups.Count; i++)
            {
                var surfaceGroup = surfaceGroups[i];
                var navObject = new SurfaceBasedNavigationObject
                {
                    PM4FileName = sourceFileName,
                    SurfaceGroupIndex = i,
                    ObjectId = $"SURFACE_OBJ_{i:D3}",
                    SurfaceGroup = surfaceGroup,
                    ObjectBounds = surfaceGroup.GroupBounds
                };

                // Extract individual surface geometries
                foreach (var surfaceIndex in surfaceGroup.SurfaceIndices)
                {
                    var surfaceGeometry = _surfaceExtractor.ExtractSurfaceGeometry(pm4File, surfaceIndex);
                    navObject.Surfaces.Add(surfaceGeometry);

                    // Separate by orientation
                    switch (surfaceGeometry.Orientation)
                    {
                        case MSURSurfaceExtractionService.SurfaceOrientation.TopFacing:
                            navObject.TopSurfaces.Add(surfaceGeometry);
                            break;
                        case MSURSurfaceExtractionService.SurfaceOrientation.BottomFacing:
                            navObject.BottomSurfaces.Add(surfaceGeometry);
                            break;
                        case MSURSurfaceExtractionService.SurfaceOrientation.Vertical:
                            navObject.VerticalSurfaces.Add(surfaceGeometry);
                            break;
                        // Mixed surfaces handled in aggregate calculations
                    }

                    // Aggregate metrics
                    navObject.TotalVertexCount += surfaceGeometry.Vertices.Count;
                    navObject.TotalTriangleCount += surfaceGeometry.TriangleIndices.Count / 3;
                    navObject.TotalSurfaceArea += surfaceGeometry.SurfaceArea;
                }

                // Estimate object type based on surface characteristics
                navObject.EstimatedObjectType = EstimateObjectTypeFromSurfaces(navObject);
                
                objects.Add(navObject);
            }

            return objects;
        }
        catch (Exception)
        {
            return objects; // Return what we have, even if incomplete
        }
    }

    /// <summary>
    /// Detect PM4 format version based on chunk availability
    /// </summary>
    public PM4FormatVersion DetectPM4FormatVersion(string pm4FilePath)
    {
        try
        {
            var pm4File = PM4File.FromFile(pm4FilePath);
            var chunkAvailability = pm4File.GetChunkAvailability();

            // Check for MDOS chunk presence (newer format indicator)
            if (chunkAvailability.HasMDOS && pm4File.MDOS?.Entries?.Any() == true)
            {
                return PM4FormatVersion.NewerWithMDOS;
            }

            // Check filename patterns (development_00_00 is known newer format)
            var fileName = Path.GetFileNameWithoutExtension(pm4FilePath);
            if (fileName.Contains("00_00"))
            {
                return PM4FormatVersion.NewerWithMDOS;
            }

            // Default to legacy format
            return PM4FormatVersion.LegacyPreMDOS;
        }
        catch
        {
            return PM4FormatVersion.Unknown;
        }
    }

    /// <summary>
    /// Create extraction configuration based on PM4 format analysis
    /// </summary>
    private ExtractionConfiguration CreateExtractionConfiguration(PM4File pm4File)
    {
        var config = new ExtractionConfiguration();
        var chunkAvailability = pm4File.GetChunkAvailability();

        // Detect format version
        if (chunkAvailability.HasMDOS && pm4File.MDOS?.Entries?.Any() == true)
        {
            config.DetectedFormat = PM4FormatVersion.NewerWithMDOS;
            config.UseMSURSurfaceExtraction = true;
            config.UseSurfaceOrientedMatching = true;
        }
        else
        {
            config.DetectedFormat = PM4FormatVersion.LegacyPreMDOS;
            config.UseMSURSurfaceExtraction = chunkAvailability.HasMSUR && pm4File.MSUR?.Entries?.Any() == true;
            config.UseSurfaceOrientedMatching = config.UseMSURSurfaceExtraction;
        }

        return config;
    }

    /// <summary>
    /// Legacy extraction method for PM4s without MSUR data
    /// </summary>
    private List<MSURSurfaceExtractionService.SurfaceGroup> ExtractSurfaceGroupsLegacyMethod(PM4File pm4File)
    {
        var groups = new List<MSURSurfaceExtractionService.SurfaceGroup>();

        // Fallback to using other chunk data (MSLK, MSPV, etc.)
        // This would use the original PM4 extraction logic from PM4FileTests.cs
        // For now, create a single group representing the entire file
        
        var chunkAvailability = pm4File.GetChunkAvailability();
        if (chunkAvailability.HasMSLK && pm4File.MSLK?.Entries?.Any() == true)
        {
            // Use MSLK-based extraction (from existing logic)
            var group = new MSURSurfaceExtractionService.SurfaceGroup
            {
                SurfaceIndices = new List<int>(), // Would be populated from MSLK analysis
                GroupType = "Legacy_MSLK_Based",
                EstimatedBuildingIndex = 0
            };

            groups.Add(group);
        }

        return groups;
    }

    /// <summary>
    /// Estimate object type based on surface characteristics
    /// </summary>
    private string EstimateObjectTypeFromSurfaces(SurfaceBasedNavigationObject navObject)
    {
        var topCount = navObject.TopSurfaces.Count;
        var bottomCount = navObject.BottomSurfaces.Count;
        var verticalCount = navObject.VerticalSurfaces.Count;
        var totalVertices = navObject.TotalVertexCount;

        // Complex building detection
        if (topCount > 5 && bottomCount > 5 && verticalCount > 10)
            return "Complex Building Structure";

        // Building with clear roof/foundation
        if (topCount > 0 && bottomCount > 0 && verticalCount > 0)
            return "Building";

        // Mostly top surfaces (roof elements)
        if (topCount > bottomCount * 2)
            return "Roof Structure";

        // Mostly bottom surfaces (foundations, platforms)
        if (bottomCount > topCount * 2)
            return "Foundation Platform";

        // Mostly vertical (walls, barriers)
        if (verticalCount > (topCount + bottomCount) * 2)
            return "Wall Structure";

        // Size-based classification
        if (totalVertices > 1000)
            return "Large Structure";
        else if (totalVertices > 500)
            return "Medium Structure";
        else if (totalVertices > 100)
            return "Small Structure";
        else
            return "Simple Element";
    }

    /// <summary>
    /// Load WMO database for surface-oriented matching
    /// </summary>
    public void LoadWMODatabaseForSurfaceMatching(List<SurfaceOrientedMatchingService.WmoAssetWithSurfaces> wmoAssets)
    {
        _surfaceMatchingService.LoadWMODatabase(wmoAssets);
    }

    /// <summary>
    /// Find WMO matches for a surface-based navigation object
    /// </summary>
    public List<SurfaceOrientedMatchingService.SurfaceMatchResult> FindSurfaceOrientedMatches(
        SurfaceBasedNavigationObject navObject,
        int maxMatches = 10)
    {
        var allMatches = new List<SurfaceOrientedMatchingService.SurfaceMatchResult>();

        // Match each surface individually, prioritizing by orientation
        var surfacesToMatch = new List<MSURSurfaceExtractionService.SurfaceGeometry>();
        
        // Prioritize top and bottom surfaces for matching
        surfacesToMatch.AddRange(navObject.TopSurfaces);
        surfacesToMatch.AddRange(navObject.BottomSurfaces);
        surfacesToMatch.AddRange(navObject.VerticalSurfaces.Take(3)); // Limit vertical surfaces

        foreach (var surface in surfacesToMatch.Take(5)) // Limit total surfaces per object
        {
            var matches = _surfaceMatchingService.FindWMOMatchesBySurface(surface, maxMatches);
            allMatches.AddRange(matches);
        }

        // Return best unique matches
        return allMatches
            .GroupBy(m => m.WmoAsset.FileName)
            .Select(g => g.OrderByDescending(m => m.OverallConfidence).First())
            .OrderByDescending(m => m.OverallConfidence)
            .Take(maxMatches)
            .ToList();
    }
} 