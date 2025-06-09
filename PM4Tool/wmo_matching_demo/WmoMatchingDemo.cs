using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.Core.Models;
using WoWToolbox.PM4Parsing;
using WoWToolbox.PM4Parsing.BuildingExtraction;
using WoWToolbox.Core.Navigation.PM4.Models;

namespace PM4Tool
{
    /// <summary>
    /// COMPREHENSIVE PM4 → WMO Positioning and Orientation Analysis
    /// Loads ALL WMO assets and performs exhaustive spatial correlation analysis
    /// </summary>
    class WmoMatchingDemo
    {
        private const string WMO_OBJ_BASE_PATH = @"I:\parp-tools\parp-tools\PM4Tool\test_data\wmo_335-objs\wmo\World\wmo\";
        private const string PM4_TEST_DATA_PATH = @"../test_data/original_development/development/";
        private const string OUTPUT_LOG_PATH = @"../output/comprehensive_wmo_matching_log.txt";
        
        // Complete asset database - ALL WMO assets loaded
        private static List<WmoAsset> completeWmoDatabase = new List<WmoAsset>();
        private static StreamWriter? logWriter;
        
        static void Main(string[] args)
        {
            Console.WriteLine("🎯 PM4-WMO Comprehensive Positioning Analysis");
            Console.WriteLine("==============================================");
            
            try
            {
                // Initialize output directory and logging
                var outputDir = Path.GetDirectoryName(OUTPUT_LOG_PATH);
                if (!string.IsNullOrEmpty(outputDir))
                {
                    Directory.CreateDirectory(outputDir);
                }
                
                using (logWriter = new StreamWriter(OUTPUT_LOG_PATH))
                {
                    WriteLogHeader();
                    
                    Console.WriteLine("📦 Loading WMO asset database...");
                    LoadAllWmoAssets();
                    
                    Console.WriteLine("🗂️ Discovering PM4 files...");
                    var pm4Files = GetAllPM4Files();
                    
                    Console.WriteLine($"🧪 TESTING WALKABLE SURFACE EXTRACTION");
                    TestWalkableSurfaceExtraction();
                    
                    Console.WriteLine("🔍 Performing comprehensive positioning analysis...");
                    PerformExhaustivePositioningAnalysis(pm4Files);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Critical error: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            
            Console.WriteLine("\n✅ Analysis complete! Check output log for detailed results.");
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void LoadAllWmoAssets()
        {
            if (!Directory.Exists(WMO_OBJ_BASE_PATH))
            {
                Console.WriteLine($"❌ WMO OBJ data not found at: {WMO_OBJ_BASE_PATH}");
                return;
            }

            // Load ALL WMO files recursively (not limited to 100!)
            var allObjFiles = Directory.GetFiles(WMO_OBJ_BASE_PATH, "*.obj", SearchOption.AllDirectories);
            Console.WriteLine($"🔍 Processing ALL {allObjFiles.Length} WMO OBJ files...");

            int processed = 0;
            foreach (var filePath in allObjFiles)
            {
                try
                {
                    var asset = CreateDetailedWmoAsset(filePath);
                    completeWmoDatabase.Add(asset);
                    processed++;

                    if (processed % 200 == 0)
                    {
                        Console.WriteLine($"   📊 Processed {processed}/{allObjFiles.Length} WMO assets...");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ⚠️ Failed to process {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            Console.WriteLine($"✅ Loaded {completeWmoDatabase.Count} complete WMO assets");
            
            // Category summary
            var categories = completeWmoDatabase.GroupBy(a => a.Category).OrderByDescending(g => g.Count()).Take(10);
            foreach (var cat in categories)
            {
                Console.WriteLine($"   📂 {cat.Key}: {cat.Count()} assets");
            }

            // Write to log
            logWriter?.WriteLine($"COMPLETE WMO ASSET DATABASE: {completeWmoDatabase.Count} assets loaded");
            logWriter?.WriteLine($"Categories: {string.Join(", ", categories.Select(c => $"{c.Key}({c.Count()})"))}");
            logWriter?.WriteLine();
        }

        static List<string> GetAllPM4Files()
        {
            if (!Directory.Exists(PM4_TEST_DATA_PATH))
            {
                Console.WriteLine($"❌ PM4 test data not found at: {PM4_TEST_DATA_PATH}");
                return new List<string>();
            }

            // Process ALL PM4 files (not limited!)
            var pm4Files = Directory.GetFiles(PM4_TEST_DATA_PATH, "*.pm4")
                .Where(f => new FileInfo(f).Length > 10000) // Focus on substantial files
                .OrderBy(f => f) // Consistent ordering
                .ToList();

            Console.WriteLine($"📁 Found {pm4Files.Count} substantial PM4 files to process");
            return pm4Files;
        }

        static void PerformExhaustivePositioningAnalysis(List<string> pm4FilePaths)
        {
            foreach (var pm4FilePath in pm4FilePaths)
            {
                var fileName = Path.GetFileName(pm4FilePath);
                Console.WriteLine($"\n🔍 Analyzing: {fileName}");
                logWriter?.WriteLine($"\n{'='*80}");
                logWriter?.WriteLine($"ANALYZING PM4 FILE: {fileName}");
                logWriter?.WriteLine($"{'='*80}");

                try
                {
                    // Load PM4 file
                    var fileData = File.ReadAllBytes(pm4FilePath);
                    var pm4File = LoadPM4File(fileData);
                    
                    if (pm4File == null)
                    {
                        Console.WriteLine($"   ❌ Failed to load PM4 file");
                        continue;
                    }

                    // Extract PM4 objects with positioning data
                    var pm4Objects = ExtractPM4ObjectsWithPositioning(pm4FilePath);
                    Console.WriteLine($"   📊 Extracted {pm4Objects.Count} PM4 objects with positioning data");
                    
                    // Log to comprehensive file
                    logWriter.WriteLine($"\n🎯 PM4 FILE: {fileName}");
                    logWriter.WriteLine($"   📄 File Path: {pm4FilePath}");
                    logWriter.WriteLine($"   📊 PM4 Statistics:");
                    logWriter.WriteLine($"      - Objects Extracted: {pm4Objects.Count}");
                    logWriter.WriteLine($"      - Total Vertices: {pm4Objects.Sum(o => o.VertexCount)}");
                    logWriter.WriteLine($"      - Total Triangles: {pm4Objects.Sum(o => o.TriangleCount)}");
                    logWriter.WriteLine($"      - Total Volume: {pm4Objects.Sum(o => o.Volume):F0}");
                    logWriter.WriteLine($"      - Total Surface Area: {pm4Objects.Sum(o => o.SurfaceArea):F0}");
                    logWriter.WriteLine($"      - Average Complexity: {(pm4Objects.Count > 0 ? pm4Objects.Average(o => o.Complexity) : 0):F1}");
                    logWriter.WriteLine($"      - Max Hierarchy Depth: {(pm4Objects.Count > 0 ? pm4Objects.Max(o => o.HierarchyDepth) : 0)}");

                    for (int objIndex = 0; objIndex < pm4Objects.Count; objIndex++)
                    {
                        var pm4Object = pm4Objects[objIndex];
                        
                        // Enhanced rotation extraction
                        var rotation = ExtractMPRLRotationData(pm4File, objIndex);
                        
                        logWriter.WriteLine($"\n   🏗️ OBJECT {objIndex}:");
                        logWriter.WriteLine($"      📍 Position: ({pm4Object.Position.X:F2}, {pm4Object.Position.Y:F2}, {pm4Object.Position.Z:F2})");
                        logWriter.WriteLine($"      📏 Scale: ({pm4Object.Scale.X:F2}, {pm4Object.Scale.Y:F2}, {pm4Object.Scale.Z:F2})");
                        logWriter.WriteLine($"      🔢 Geometry: {pm4Object.VertexCount} vertices, {pm4Object.TriangleCount} triangles");
                        logWriter.WriteLine($"      📐 Volume: {pm4Object.Volume:F0}, Surface Area: {pm4Object.SurfaceArea:F0}");
                        logWriter.WriteLine($"      🧮 Complexity: {pm4Object.Complexity:F1}, Hierarchy Depth: {pm4Object.HierarchyDepth}");
                        
                        // Extract key metadata
                        var extractionMethod = pm4Object.Metadata.GetValueOrDefault("ExtractionMethod", "Unknown");
                        var volumeToSurfaceRatio = pm4Object.Metadata.GetValueOrDefault("VolumeToSurfaceRatio", 0);
                        var vertexDensity = pm4Object.Metadata.GetValueOrDefault("VertexDensity", 0);
                        
                        logWriter.WriteLine($"      🎯 Metadata: Method={extractionMethod}, V/S Ratio={volumeToSurfaceRatio:F3}, Vertex Density={vertexDensity:F1}");
                        logWriter.WriteLine($"      🔄 MPRL Rotation: ({rotation.X:F2}°, {rotation.Y:F2}°, {rotation.Z:F2}°)");
                        
                        // EXHAUSTIVE matching against ALL WMO assets
                        var bestMatches = FindBestWmoMatches(pm4Object, completeWmoDatabase);
                        
                        if (bestMatches.Count > 0)
                        {
                            logWriter.WriteLine($"      🎉 FOUND {bestMatches.Count} POTENTIAL MATCHES:");
                            
                            // Show top 5 matches with enhanced details
                            for (int matchIndex = 0; matchIndex < Math.Min(bestMatches.Count, 5); matchIndex++)
                            {
                                var match = bestMatches[matchIndex];
                                
                                logWriter.WriteLine($"\n      🏆 MATCH #{matchIndex + 1} (Score: {match.MatchScore:F3})");
                                logWriter.WriteLine($"         📄 Asset: {match.AssetName}");
                                logWriter.WriteLine($"         📂 Path: {match.AssetPath}");
                                logWriter.WriteLine($"         🏷️ Category: {match.Category}");
                                logWriter.WriteLine($"         📍 WMO Position: ({match.WmoAsset.Position.X:F2}, {match.WmoAsset.Position.Y:F2}, {match.WmoAsset.Position.Z:F2})");
                                logWriter.WriteLine($"         📏 WMO Dimensions: ({match.WmoAsset.Dimensions.X:F2}, {match.WmoAsset.Dimensions.Y:F2}, {match.WmoAsset.Dimensions.Z:F2}) [NOT scale!]");
                                logWriter.WriteLine($"         🔄 WMO Rotation: ({match.WmoAsset.Rotation.X:F2}°, {match.WmoAsset.Rotation.Y:F2}°, {match.WmoAsset.Rotation.Z:F2}°)");
                                logWriter.WriteLine($"         ↗️ WMO Tilt: ({match.WmoAsset.Tilt.X:F2}°, {match.WmoAsset.Tilt.Y:F2}°, {match.WmoAsset.Tilt.Z:F2}°)");
                                logWriter.WriteLine($"         🔢 WMO Geometry: {match.WmoAsset.VertexCount} vertices, {match.WmoAsset.TriangleCount} triangles");
                                logWriter.WriteLine($"         📐 WMO Volume: {match.WmoAsset.Volume:F0}, Surface Area: {match.WmoAsset.SurfaceArea:F0}");
                                logWriter.WriteLine($"         🧮 WMO Complexity: {match.WmoAsset.Complexity:F1}");
                                logWriter.WriteLine($"         ⚖️ Position Offset: ({match.PositionOffset.X:F2}, {match.PositionOffset.Y:F2}, {match.PositionOffset.Z:F2})");
                                logWriter.WriteLine($"         📊 Dimensional Similarity: {match.ScaleRatio:F3} [CORRECTED - not scale!]");
                                logWriter.WriteLine($"         🔄 Rotation Delta: ({match.RotationDelta.X:F2}°, {match.RotationDelta.Y:F2}°, {match.RotationDelta.Z:F2}°)");
                                logWriter.WriteLine($"         📈 Enhanced Analysis:");
                                
                                // Format the detailed analysis with proper indentation
                                var analysisLines = match.Analysis.Split('\n');
                                foreach (var line in analysisLines)
                                {
                                    if (!string.IsNullOrWhiteSpace(line))
                                    {
                                        logWriter.WriteLine($"            {line.Trim()}");
                                    }
                                }
                            }
                        }
                        else
                        {
                            logWriter.WriteLine($"      ❌ NO MATCHES FOUND (all scores below threshold)");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ Error processing {fileName}: {ex.Message}");
                    logWriter.WriteLine($"ERROR: {ex.Message}");
                }
            }
        }

        /// <summary>
        /// ENHANCED PM4 object extraction utilizing ALL available PM4 chunk data
        /// </summary>
        static List<PM4ObjectWithPositioning> ExtractPM4ObjectsWithPositioning(string filePath)
        {
            var results = new List<PM4ObjectWithPositioning>();
            
            try
            {
                Console.WriteLine($"      🧪 TESTING CORE.V2 FIX: Comparing extraction methods");
                
                // Load PM4 file
                var pm4File = PM4File.FromFile(filePath);
                Console.WriteLine($"      📦 PM4File loaded successfully with {pm4File.MSLK?.Entries.Count ?? 0} MSLK entries");
                
                // TEST 1: Core.v2 FIXED ExtractBuildings (should now have full geometry)
                Console.WriteLine($"      🔧 Testing FIXED Core.v2 PM4File.ExtractBuildings()...");
                var coreV2Buildings = pm4File.ExtractBuildings();
                Console.WriteLine($"      📊 Core.v2 ExtractBuildings returned {coreV2Buildings.Count} buildings");
                
                if (coreV2Buildings.Count > 0)
                {
                    var firstBuilding = coreV2Buildings[0];
                    Console.WriteLine($"      🎯 First Core.v2 building: {firstBuilding.Vertices.Count} vertices, {firstBuilding.TriangleIndices.Count / 3} faces");
                    
                    if (firstBuilding.Vertices.Count <= 4)
                    {
                        Console.WriteLine($"      ❌ CORE.V2 STILL BROKEN: Only {firstBuilding.Vertices.Count} vertices (collision hull)");
                    }
                    else
                    {
                        Console.WriteLine($"      ✅ CORE.V2 FIXED: {firstBuilding.Vertices.Count} vertices (full geometry!)");
                    }
                }
                
                // TEST 2: PM4BuildingExtractionService (known working method)
                Console.WriteLine($"      📈 Testing PM4BuildingExtractionService (reference method)...");
                var extractionService = new WoWToolbox.PM4Parsing.PM4BuildingExtractionService();
                var tempOutputDir = Path.GetTempPath();
                var extractionResult = extractionService.ExtractAndExportBuildings(filePath, tempOutputDir);
                var serviceBuildings = extractionResult.Buildings;
                Console.WriteLine($"      📊 PM4BuildingExtractionService returned {serviceBuildings.Count} buildings");
                
                if (serviceBuildings.Count > 0)
                {
                    var firstServiceBuilding = serviceBuildings[0];
                    Console.WriteLine($"      📈 First service building: {firstServiceBuilding.Vertices.Count} vertices, {firstServiceBuilding.TriangleIndices.Count / 3} faces");
                }
                
                // COMPARISON
                if (coreV2Buildings.Count > 0 && serviceBuildings.Count > 0)
                {
                    var coreVertices = coreV2Buildings[0].Vertices.Count;
                    var serviceVertices = serviceBuildings[0].Vertices.Count;
                    
                    Console.WriteLine($"      🔍 COMPARISON:");
                    Console.WriteLine($"         Core.v2:     {coreVertices} vertices");
                    Console.WriteLine($"         Service:     {serviceVertices} vertices");
                    
                    if (coreVertices > 100 && serviceVertices > 100)
                    {
                        Console.WriteLine($"      🎉 SUCCESS: Both methods now return full geometry!");
                    }
                    else if (coreVertices <= 4)
                    {
                        Console.WriteLine($"      ⚠️  Core.v2 still needs more work (only {coreVertices} vertices)");
                    }
                    else
                    {
                        Console.WriteLine($"      ✅ Core.v2 fix working, different extraction strategy");
                    }
                }
                
                // Convert first building to normalized format for matching
                if (coreV2Buildings.Count > 0)
                {
                    var building = coreV2Buildings[0];
                    results.Add(new PM4ObjectWithPositioning
                    {
                        ObjectId = 0,
                        Vertices = building.Vertices.ToList(),
                                                 BoundingBox = CalculateBoundingBox(building.Vertices),
                         ComplexityClass = PM4ComplexityClass.FullGeometry, // Will be > 4 vertices if our fix works
                        VertexCount = building.Vertices.Count,
                        Source = "Core.v2_Fixed"
                    });
                }
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ❌ Error in PM4 extraction: {ex.Message}");
            }
            
            return results;
        }

        /// <summary>
        /// Extract MSLK hierarchy data for building relationship analysis
        /// </summary>
        static MSLKHierarchyData ExtractMSLKHierarchyData(PM4File pm4File)
        {
            var hierarchyData = new MSLKHierarchyData();
            
            try
            {
                if (pm4File.MSLK?.Entries != null)
                {
                    hierarchyData.TotalEntries = pm4File.MSLK.Entries.Count;
                    
                    // Analyze hierarchy structure
                    for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
                    {
                        var entry = pm4File.MSLK.Entries[i];
                        
                        if (entry.Unknown_0x04 == i) // Self-referencing = root
                        {
                            hierarchyData.RootNodes.Add(i);
                        }
                        
                        if (entry.MspiIndexCount > 0)
                        {
                            hierarchyData.NodesWithGeometry++;
                        }
                        
                        hierarchyData.ParentChildRelations[i] = (int)entry.Unknown_0x04;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️ MSLK analysis error: {ex.Message}");
            }
            
            return hierarchyData;
        }

        /// <summary>
        /// Extract MSUR surface data for render correlation
        /// </summary>
        static MSURSurfaceData ExtractMSURSurfaceData(PM4File pm4File)
        {
            var surfaceData = new MSURSurfaceData();
            
            try
            {
                if (pm4File.MSUR?.Entries != null)
                {
                    surfaceData.SurfaceCount = pm4File.MSUR.Entries.Count;
                    surfaceData.TotalTriangles = pm4File.MSUR.Entries.Sum(s => (int)s.IndexCount / 3);
                    surfaceData.AverageTrianglesPerSurface = surfaceData.SurfaceCount > 0 
                        ? (float)surfaceData.TotalTriangles / surfaceData.SurfaceCount : 0;
                    
                    // Analyze surface complexity distribution
                    var triangleCounts = pm4File.MSUR.Entries.Select(s => (int)s.IndexCount / 3).Where(c => c > 0).ToList();
                    if (triangleCounts.Count > 0)
                    {
                        surfaceData.MinTrianglesPerSurface = triangleCounts.Min();
                        surfaceData.MaxTrianglesPerSurface = triangleCounts.Max();
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️ MSUR analysis error: {ex.Message}");
            }
            
            return surfaceData;
        }

        /// <summary>
        /// Extract detailed vertex information
        /// </summary>
        static VertexDetailData ExtractVertexDetailData(PM4File pm4File)
        {
            var vertexData = new VertexDetailData();
            
            try
            {
                vertexData.MSVTVertices = pm4File.MSVT?.Vertices?.Count ?? 0;
                vertexData.MSPVVertices = pm4File.MSPV?.Vertices?.Count ?? 0;
                vertexData.TotalVertices = vertexData.MSVTVertices + vertexData.MSPVVertices;
                
                // Calculate vertex distribution and density
                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var vertices = pm4File.MSVT.Vertices.Select(v => new Vector3(v.X, v.Y, v.Z)).ToList();
                    var bounds = CalculateBoundingBox(vertices);
                    vertexData.MSVTBounds = bounds;
                    vertexData.MSVTDensity = bounds.GetVolume() > 0 ? vertices.Count / bounds.GetVolume() : 0;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️ Vertex analysis error: {ex.Message}");
            }
            
            return vertexData;
        }

        /// <summary>
        /// Investigates MPRL chunk for rotation/tilt data for specific object
        /// </summary>
        static Vector3 ExtractMPRLRotationData(PM4File pm4File, int objIndex)
        {
            try
            {
                // TODO: Implement MPRL chunk parsing for rotation data
                // For now, return zero rotation
                return Vector3.Zero;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"      ⚠️ MPRL rotation extraction error: {ex.Message}");
                return Vector3.Zero;
            }
        }

        /// <summary>
        /// FIXED comprehensive matching with aggressive 4-vertex filtering
        /// </summary>
        static List<WmoPositionalMatch> FindBestWmoMatches(PM4ObjectWithPositioning pm4Object, List<WmoAsset> wmoAssets)
        {
            var matches = new List<WmoPositionalMatch>();

            // CRITICAL FIX: Skip 4-vertex collision hulls entirely - they create false positives
            if (pm4Object.VertexCount <= 4)
            {
                Console.WriteLine($"        ⏭️ SKIPPING 4-vertex collision hull - no meaningful WMO correlation possible");
                return matches; // Return empty list
            }

            // ENHANCED: Classify PM4 object complexity for remaining objects
            var pm4ComplexityClass = ClassifyPM4Complexity(pm4Object);
            
            Console.WriteLine($"        🔍 PM4 Object Complexity: {pm4ComplexityClass} ({pm4Object.VertexCount} vertices)");

            foreach (var wmo in wmoAssets)
            {
                // ENHANCED: Apply complexity-appropriate matching algorithm
                var matchResult = pm4ComplexityClass switch
                {
                    PM4ComplexityClass.CollisionHull => MatchCollisionHullUltraStrict(pm4Object, wmo), // Ultra-strict for remaining low-vertex objects
                    PM4ComplexityClass.SimplifiedGeometry => MatchSimplifiedGeometry(pm4Object, wmo),
                    PM4ComplexityClass.FullGeometry => MatchFullGeometry(pm4Object, wmo),
                    _ => null
                };

                if (matchResult != null && matchResult.Score > GetMinimumScoreThreshold(pm4ComplexityClass))
                {
                    matches.Add(new WmoPositionalMatch
                    {
                        WmoAsset = wmo,
                        PM4Object = pm4Object,
                        MatchScore = matchResult.Score,
                        PositionOffset = pm4Object.Position - wmo.Position,
                        ScaleRatio = matchResult.DimensionalSimilarity,
                        RotationDelta = wmo.Rotation,
                        Analysis = $"COMPLEXITY-AWARE Analysis ({pm4ComplexityClass}):\n{matchResult.Analysis}"
                    });
                }
            }

            return matches.OrderByDescending(m => m.MatchScore).Take(3).ToList(); // Reduced to top 3 to minimize noise
        }

        /// <summary>
        /// Classify PM4 object complexity (updated thresholds)
        /// </summary>
        static PM4ComplexityClass ClassifyPM4Complexity(PM4ObjectWithPositioning pm4Object)
        {
            if (pm4Object.VertexCount <= 12)        // Raised from 9 to 12
                return PM4ComplexityClass.CollisionHull;
            else if (pm4Object.VertexCount <= 50)
                return PM4ComplexityClass.SimplifiedGeometry;
            else
                return PM4ComplexityClass.FullGeometry;
        }

        /// <summary>
        /// Ultra-strict scoring thresholds to reduce false positives
        /// </summary>
        static float GetMinimumScoreThreshold(PM4ComplexityClass complexityClass)
        {
            return complexityClass switch
            {
                PM4ComplexityClass.CollisionHull => 0.85f,       // ULTRA-HIGH threshold for collision hulls
                PM4ComplexityClass.SimplifiedGeometry => 0.5f,   // Raised threshold
                PM4ComplexityClass.FullGeometry => 0.3f,         // Slightly raised threshold
                _ => 0.1f
            };
        }

        /// <summary>
        /// ULTRA-STRICT COLLISION HULL MATCHING: Nearly perfect overlap required (5-12 vertices)
        /// </summary>
        static MatchResult? MatchCollisionHullUltraStrict(PM4ObjectWithPositioning pm4Object, WmoAsset wmo)
        {
            // ULTRA-STRICT: Require very close position match
            var positionMatch = CalculatePositionMatch(pm4Object.Position, wmo.Position);
            if (positionMatch < 0.8f) return null; // Early exit if position doesn't closely match
            
            var boundsMatch = CalculateBoundsContainment(pm4Object.Scale, wmo.Dimensions);
            var volumeRatio = CalculateVolumeCompatibility(pm4Object.Volume, wmo.Volume);
            
            // ULTRA-STRICT scoring for collision hulls - require near-perfect spatial alignment
            var score = positionMatch * 0.60f +      // Position is CRITICAL (60%)
                       boundsMatch * 0.30f +         // Bounds containment (30%)
                       volumeRatio * 0.10f;          // Volume compatibility (10%)
            
            // MASSIVE PENALTY for extreme vertex disparity (much harsher than before)
            var vertexDisparity = (float)Math.Abs(pm4Object.VertexCount - wmo.VertexCount) / Math.Max(pm4Object.VertexCount, wmo.VertexCount);
            var vertexPenalty = vertexDisparity > 0.95f ? 0.2f :  // 80% penalty for extreme disparity
                               vertexDisparity > 0.9f ? 0.4f :   // 60% penalty for high disparity  
                               vertexDisparity > 0.8f ? 0.7f :   // 30% penalty for medium disparity
                               1.0f;                              // No penalty for reasonable disparity
            
            score *= vertexPenalty;
            
            // ADDITIONAL PENALTY: If WMO has >1000 vertices and PM4 has <10, apply massive penalty
            if (wmo.VertexCount > 1000 && pm4Object.VertexCount < 10)
            {
                score *= 0.1f; // 90% penalty for comparing tiny collision hull to complex WMO
            }
            
            var analysis = $"  🔸 ULTRA-STRICT COLLISION HULL Analysis (PM4:{pm4Object.VertexCount}v vs WMO:{wmo.VertexCount}v):\n" +
                          $"    • Position Match: {positionMatch:F3} (weight 60%, min required: 0.8)\n" +
                          $"    • Bounds Containment: {boundsMatch:F3} (weight 30%)\n" +
                          $"    • Volume Compatibility: {volumeRatio:F3} (weight 10%)\n" +
                          $"    • Vertex Disparity Penalty: {vertexPenalty:F3}\n" +
                          $"    • Complex WMO Penalty: {(wmo.VertexCount > 1000 && pm4Object.VertexCount < 10 ? "0.1 (90% penalty)" : "none")}\n" +
                          $"    • Final Score: {score:F3} (threshold: 0.85)";
            
            return new MatchResult { Score = score, DimensionalSimilarity = boundsMatch, Analysis = analysis };
        }

        /// <summary>
        /// COLLISION HULL MATCHING: Focus on bounds and position only (4-9 vertices)
        /// </summary>
        static MatchResult? MatchCollisionHull(PM4ObjectWithPositioning pm4Object, WmoAsset wmo)
        {
            // For collision hulls, only meaningful comparisons are bounds and position
            var positionMatch = CalculatePositionMatch(pm4Object.Position, wmo.Position);
            var boundsMatch = CalculateBoundsContainment(pm4Object.Scale, wmo.Dimensions);
            var volumeRatio = CalculateVolumeCompatibility(pm4Object.Volume, wmo.Volume);
            
            // STRICT scoring for collision hulls - they must be spatially related
            var score = positionMatch * 0.50f +      // Position is critical (50%)
                       boundsMatch * 0.35f +         // Bounds containment (35%)
                       volumeRatio * 0.15f;          // Volume compatibility (15%)
            
            // PENALTY for extreme vertex disparity
            var vertexDisparity = (float)Math.Abs(pm4Object.VertexCount - wmo.VertexCount) / Math.Max(pm4Object.VertexCount, wmo.VertexCount);
            var vertexPenalty = vertexDisparity > 0.9f ? 0.5f : 1.0f; // 50% penalty for extreme disparity
            score *= vertexPenalty;
            
            var analysis = $"  🔸 COLLISION HULL Analysis (PM4:{pm4Object.VertexCount}v vs WMO:{wmo.VertexCount}v):\n" +
                          $"    • Position Match: {positionMatch:F3} (weight 50%)\n" +
                          $"    • Bounds Containment: {boundsMatch:F3} (weight 35%)\n" +
                          $"    • Volume Compatibility: {volumeRatio:F3} (weight 15%)\n" +
                          $"    • Vertex Disparity Penalty: {vertexPenalty:F3}\n" +
                          $"    • Final Score: {score:F3}";
            
            return new MatchResult { Score = score, DimensionalSimilarity = boundsMatch, Analysis = analysis };
        }

        /// <summary>
        /// SIMPLIFIED GEOMETRY MATCHING: Balanced approach (10-50 vertices)
        /// </summary>
        static MatchResult? MatchSimplifiedGeometry(PM4ObjectWithPositioning pm4Object, WmoAsset wmo)
        {
            var positionMatch = CalculatePositionMatch(pm4Object.Position, wmo.Position);
            var dimensionalSimilarity = CalculateDimensionalSimilarity(pm4Object.Scale, wmo.Dimensions);
            var complexityRatio = CalculateComplexityRatio(pm4Object.Complexity, wmo.Complexity);
            var volumeRatio = wmo.Volume > 0 ? Math.Min(pm4Object.Volume / wmo.Volume, wmo.Volume / pm4Object.Volume) : 0;
            
            // Balanced scoring for simplified geometry
            var score = positionMatch * 0.35f +           // Position (35%)
                       dimensionalSimilarity * 0.30f +   // Dimensions (30%)
                       complexityRatio * 0.20f +         // Complexity (20%)
                       volumeRatio * 0.15f;              // Volume (15%)
            
            var analysis = $"  🔹 SIMPLIFIED GEOMETRY Analysis (PM4:{pm4Object.VertexCount}v vs WMO:{wmo.VertexCount}v):\n" +
                          $"    • Position Match: {positionMatch:F3} (weight 35%)\n" +
                          $"    • Dimensional Similarity: {dimensionalSimilarity:F3} (weight 30%)\n" +
                          $"    • Complexity Ratio: {complexityRatio:F3} (weight 20%)\n" +
                          $"    • Volume Ratio: {volumeRatio:F3} (weight 15%)";
            
            return new MatchResult { Score = score, DimensionalSimilarity = dimensionalSimilarity, Analysis = analysis };
        }

        /// <summary>
        /// FULL GEOMETRY MATCHING: Comprehensive analysis (50+ vertices)
        /// </summary>
        static MatchResult? MatchFullGeometry(PM4ObjectWithPositioning pm4Object, WmoAsset wmo)
        {
            // Full comprehensive analysis for complex geometry
            var positionMatch = CalculatePositionMatch(pm4Object.Position, wmo.Position);
            var dimensionalSimilarity = CalculateDimensionalSimilarity(pm4Object.Scale, wmo.Dimensions);
            var proportionMatch = CalculateProportionMatch(pm4Object.Scale, wmo.Dimensions);
            var volumeRatio = wmo.Volume > 0 ? Math.Min(pm4Object.Volume / wmo.Volume, wmo.Volume / pm4Object.Volume) : 0;
            var complexityRatio = CalculateComplexityRatio(pm4Object.Complexity, wmo.Complexity);
            var vertexDensityRatio = CalculateVertexDensityRatio(pm4Object.VertexCount, wmo.VertexCount);
            var surfaceAreaRatio = wmo.SurfaceArea > 0 ? Math.Min(pm4Object.SurfaceArea / wmo.SurfaceArea, wmo.SurfaceArea / pm4Object.SurfaceArea) : 0;
            
            // Comprehensive scoring for full geometry
            var score = positionMatch * 0.25f +           // Position (25%)
                       dimensionalSimilarity * 0.20f +   // Dimensions (20%)
                       proportionMatch * 0.15f +         // Proportions (15%)
                       volumeRatio * 0.15f +            // Volume (15%)
                       complexityRatio * 0.10f +        // Complexity (10%)
                       vertexDensityRatio * 0.10f +     // Vertex density (10%)
                       surfaceAreaRatio * 0.05f;        // Surface area (5%)
            
            var analysis = $"  🔸 FULL GEOMETRY Analysis (PM4:{pm4Object.VertexCount}v vs WMO:{wmo.VertexCount}v):\n" +
                          $"    • Position Match: {positionMatch:F3} (weight 25%)\n" +
                          $"    • Dimensional Similarity: {dimensionalSimilarity:F3} (weight 20%)\n" +
                          $"    • Proportion Match: {proportionMatch:F3} (weight 15%)\n" +
                          $"    • Volume Ratio: {volumeRatio:F3} (weight 15%)\n" +
                          $"    • Complexity Ratio: {complexityRatio:F3} (weight 10%)\n" +
                          $"    • Vertex Density Ratio: {vertexDensityRatio:F3} (weight 10%)\n" +
                          $"    • Surface Area Ratio: {surfaceAreaRatio:F3} (weight 5%)";
            
            return new MatchResult { Score = score, DimensionalSimilarity = dimensionalSimilarity, Analysis = analysis };
        }

        // HELPER METHODS FOR COMPLEXITY-AWARE MATCHING

        static float CalculateBoundsContainment(Vector3 pm4Bounds, Vector3 wmoBounds)
        {
            // Check if PM4 bounds could reasonably fit within WMO bounds
            var xContainment = pm4Bounds.X <= wmoBounds.X ? 1.0f : wmoBounds.X / pm4Bounds.X;
            var yContainment = pm4Bounds.Y <= wmoBounds.Y ? 1.0f : wmoBounds.Y / pm4Bounds.Y;
            var zContainment = pm4Bounds.Z <= wmoBounds.Z ? 1.0f : wmoBounds.Z / pm4Bounds.Z;
            
            return (xContainment + yContainment + zContainment) / 3.0f;
        }

        public static float CalculateVolumeCompatibility(float pm4Volume, float wmoVolume)
        {
            if (pm4Volume == 0 || wmoVolume == 0) return 0f;
            
            // For collision hulls, PM4 volume should be smaller than WMO volume
            return pm4Volume <= wmoVolume ? 1.0f : Math.Max(0f, wmoVolume / pm4Volume);
        }

        static float CalculateComplexityRatio(float pm4Complexity, float wmoComplexity)
        {
            if (pm4Complexity == 0 && wmoComplexity == 0) return 1.0f;
            if (pm4Complexity == 0 || wmoComplexity == 0) return 0f;
            
            return Math.Min(pm4Complexity, wmoComplexity) / Math.Max(pm4Complexity, wmoComplexity);
        }

        static float CalculateVertexDensityRatio(int pm4Vertices, int wmoVertices)
        {
            if (pm4Vertices == 0 || wmoVertices == 0) return 0f;
            
            return Math.Min(pm4Vertices, wmoVertices) / (float)Math.Max(pm4Vertices, wmoVertices);
        }

        /// <summary>
        /// CORRECTED: Calculate dimensional similarity (not scale ratios!)
        /// </summary>
        public static float CalculateDimensionalSimilarity(Vector3 pm4Dimensions, Vector3 wmoDimensions)
        {
            if (pm4Dimensions == Vector3.Zero || wmoDimensions == Vector3.Zero) return 0f;
            
            var ratios = new[]
            {
                Math.Min(pm4Dimensions.X, wmoDimensions.X) / Math.Max(pm4Dimensions.X, wmoDimensions.X),
                Math.Min(pm4Dimensions.Y, wmoDimensions.Y) / Math.Max(pm4Dimensions.Y, wmoDimensions.Y),
                Math.Min(pm4Dimensions.Z, wmoDimensions.Z) / Math.Max(pm4Dimensions.Z, wmoDimensions.Z)
            };
            
            return ratios.Average();
        }

        // REAL PM4 LOADING using actual WoWToolbox classes
        static PM4File? LoadPM4File(byte[] fileData)
        {
            try
            {
                return new PM4File(fileData);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ⚠️ PM4 loading error: {ex.Message}");
                return null;
            }
        }

        // Helper methods for realistic data generation (temporary until real integration)
        static List<Vector3> GenerateRealisticVertexData(int fileSize, int objectIndex)
        {
            var random = new Random(fileSize + objectIndex);
            var vertexCount = random.Next(200, 2000);
            var vertices = new List<Vector3>();
            
            var centerX = random.Next(-100, 100);
            var centerZ = random.Next(-100, 100);
            var scale = random.Next(10, 50);
            
            for (int i = 0; i < vertexCount; i++)
            {
                vertices.Add(new Vector3(
                    centerX + (float)(random.NextDouble() - 0.5) * scale,
                    (float)random.NextDouble() * scale,
                    centerZ + (float)(random.NextDouble() - 0.5) * scale
                ));
            }
            
            return vertices;
        }

        static List<Triangle> GenerateRealisticTriangleData(List<Vector3> vertices)
        {
            var triangles = new List<Triangle>();
            var triangleCount = Math.Max(1, vertices.Count / 3);
            
            for (int i = 0; i < triangleCount && i + 2 < vertices.Count; i += 3)
            {
                triangles.Add(new Triangle
                {
                    V1 = vertices[i],
                    V2 = vertices[i + 1],
                    V3 = vertices[i + 2]
                });
            }
            
            return triangles;
        }

        /// <summary>
        /// Creates detailed WMO asset with comprehensive geometric analysis
        /// </summary>
        static WmoAsset CreateDetailedWmoAsset(string filePath)
        {
            var fileInfo = new FileInfo(filePath);
            var relativePath = Path.GetRelativePath(WMO_OBJ_BASE_PATH, filePath);
            var category = GetAssetCategory(relativePath);
            
            // Enhanced analysis of vertices for positioning data
            var vertices = new List<Vector3>();
            var lines = File.ReadLines(filePath).Take(1000); // Sample more for better analysis
            
            foreach (var line in lines)
            {
                if (line.StartsWith("v "))
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 4 &&
                        float.TryParse(parts[1], out float x) &&
                        float.TryParse(parts[2], out float y) &&
                        float.TryParse(parts[3], out float z))
                    {
                        vertices.Add(new Vector3(x, y, z));
                    }
                }
            }

            var bounds = vertices.Count > 0 ? CalculateBoundingBox(vertices) : 
                             new BoundingBox3D(Vector3.Zero, Vector3.Zero);

            return new WmoAsset
            {
                FileName = fileInfo.Name,
                FilePath = filePath,
                RelativePath = relativePath,
                Category = category,
                FileSizeBytes = fileInfo.Length,
                BoundingBox = bounds,
                Volume = bounds.GetVolume(),
                SurfaceArea = EstimateBuildingSurfaceArea(vertices),
                Complexity = CalculateBuildingComplexity(vertices, new MSURSurfaceData()),
                VertexCount = vertices.Count,
                TriangleCount = vertices.Count / 3,
                Position = bounds.GetCenter(),
                Rotation = Vector3.Zero,
                Tilt = Vector3.Zero,
                Dimensions = bounds.GetScale()
            };
        }

        static string GetAssetCategory(string relativePath)
        {
            var pathParts = relativePath.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            return pathParts.Length > 0 ? pathParts[0] : "Uncategorized";
        }

        public static BoundingBox3D CalculateBoundingBox(List<Vector3> vertices)
        {
            if (vertices.Count == 0)
                return new BoundingBox3D(Vector3.Zero, Vector3.Zero);

            var min = vertices[0];
            var max = vertices[0];

            foreach (var vertex in vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            return new BoundingBox3D(min, max);
        }
        
        /// <summary>
        /// Helper method to calculate bounding box from vertices (for walkable surface extraction)
        /// </summary>
        static BoundingBox3D CalculateBoundingBoxFromVertices(List<Vector3> vertices)
        {
            return CalculateBoundingBox(vertices);
        }
        
        /// <summary>
        /// Normalizes a Vector3 (creates unit vector)
        /// </summary>
        static Vector3 NormalizeVector3(Vector3 vector)
        {
            var length = vector.Length();
            if (length > 0.000001f)
            {
                return vector / length;
            }
            return Vector3.UnitZ; // Default to up vector if zero length
        }

        static void WriteLogHeader()
        {
            logWriter?.WriteLine("=== COMPREHENSIVE PM4 → WMO Positioning Analysis ===");
            logWriter?.WriteLine(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
            logWriter?.WriteLine();
        }

        // POSITIONING AND SCALING ANALYSIS METHODS

        static Vector3 CalculateScaleRatio(Vector3 pm4Scale, Vector3 wmoScale)
        {
            return new Vector3(
                pm4Scale.X > 0 ? wmoScale.X / pm4Scale.X : 1.0f,
                pm4Scale.Y > 0 ? wmoScale.Y / pm4Scale.Y : 1.0f,
                pm4Scale.Z > 0 ? wmoScale.Z / pm4Scale.Z : 1.0f
            );
        }

        static float CalculatePositionMatch(Vector3 pm4Position, Vector3 wmoPosition)
        {
            var distance = Vector3.Distance(pm4Position, wmoPosition);
            // Normalize distance to 0-1 scale (closer = higher match)
            var maxReasonableDistance = 1000.0f; // Adjust based on world scale
            return Math.Max(0f, 1.0f - (distance / maxReasonableDistance));
        }

        static float CalculateScaleMatch(Vector3 pm4Scale, Vector3 wmoScale)
        {
            if (pm4Scale == Vector3.Zero || wmoScale == Vector3.Zero) return 0f;
            
            var ratios = new[]
            {
                Math.Min(pm4Scale.X, wmoScale.X) / Math.Max(pm4Scale.X, wmoScale.X),
                Math.Min(pm4Scale.Y, wmoScale.Y) / Math.Max(pm4Scale.Y, wmoScale.Y),
                Math.Min(pm4Scale.Z, wmoScale.Z) / Math.Max(pm4Scale.Z, wmoScale.Z)
            };
            
            return ratios.Average();
        }

        static float CalculateProportionMatch(Vector3 pm4Scale, Vector3 wmoScale)
        {
            // Calculate aspect ratios for both
            var pm4Ratios = new Vector3(
                pm4Scale.Y > 0 ? pm4Scale.X / pm4Scale.Y : 1.0f,
                pm4Scale.Z > 0 ? pm4Scale.X / pm4Scale.Z : 1.0f,
                pm4Scale.Z > 0 ? pm4Scale.Y / pm4Scale.Z : 1.0f
            );
            
            var wmoRatios = new Vector3(
                wmoScale.Y > 0 ? wmoScale.X / wmoScale.Y : 1.0f,
                wmoScale.Z > 0 ? wmoScale.X / wmoScale.Z : 1.0f,
                wmoScale.Z > 0 ? wmoScale.Y / wmoScale.Z : 1.0f
            );
            
            var diff = Vector3.Abs(pm4Ratios - wmoRatios);
            var avgDiff = (diff.X + diff.Y + diff.Z) / 3.0f;
            return Math.Max(0f, 1.0f - avgDiff);
        }

        // ENHANCED GEOMETRIC CALCULATIONS

        static float EstimateBuildingSurfaceArea(List<Vector3> vertices)
        {
            // Estimate surface area from vertex count and bounding box
            if (vertices.Count < 3) return 0f;
            
            var bounds = CalculateBoundingBox(vertices);
            var dimensions = bounds.GetScale();
            
            // Approximate surface area as bounding box surface area
            return 2 * (dimensions.X * dimensions.Y + dimensions.Y * dimensions.Z + dimensions.X * dimensions.Z);
        }

        static float CalculateBuildingComplexity(List<Vector3> vertices, MSURSurfaceData surfaceData)
        {
            // Simple complexity based on vertex density
            var bounds = CalculateBoundingBox(vertices);
            var volume = bounds.GetVolume();
            
            return volume > 0 ? vertices.Count / volume : vertices.Count;
        }

        static int GetBuildingHierarchyDepth(int buildingIndex, MSLKHierarchyData hierarchyData)
        {
            // Calculate depth in MSLK hierarchy
            if (hierarchyData.ParentChildRelations.ContainsKey(buildingIndex))
            {
                var parent = hierarchyData.ParentChildRelations[buildingIndex];
                var depth = 0;
                var visited = new HashSet<int>();
                
                while (parent != buildingIndex && !visited.Contains(parent) && depth < 10)
                {
                    visited.Add(parent);
                    if (hierarchyData.ParentChildRelations.ContainsKey(parent))
                    {
                        parent = hierarchyData.ParentChildRelations[parent];
                        depth++;
                    }
                    else break;
                }
                
                return depth;
            }
            
            return 0;
        }

        static void TestWalkableSurfaceExtraction()
        {
            Console.WriteLine("🚶 Testing Walkable Surface Extraction on Sample WMO Files");
            Console.WriteLine("=========================================================");
            
            // Test on first few WMO assets
            var testAssets = completeWmoDatabase.Take(3).ToList();
            
            foreach (var asset in testAssets)
            {
                Console.WriteLine($"\n📂 Testing: {asset.FileName}");
                Console.WriteLine($"   Category: {asset.Category}");
                Console.WriteLine($"   File Size: {asset.FileSizeBytes:N0} bytes");
                
                // Extract walkable surfaces
                var walkableSegments = WalkableSurfaceExtractor.ExtractWalkableSurfaces(asset);
                
                if (walkableSegments.Count > 0)
                {
                    Console.WriteLine($"   ✅ Found {walkableSegments.Count} walkable surface segments:");
                    
                    foreach (var segment in walkableSegments.Take(3)) // Show first 3 segments
                    {
                        Console.WriteLine($"      🔸 Segment {segment.SegmentId}: {segment.Vertices.Count} vertices, " +
                                        $"{segment.SurfaceArea:F1} area, {segment.WalkableType}, height {segment.AverageHeight:F1}");
                    }
                    
                    // Demonstrate improved matching potential
                    var largestSegment = walkableSegments.OrderByDescending(s => s.SurfaceArea).FirstOrDefault();
                    if (largestSegment != null)
                    {
                        Console.WriteLine($"   🎯 Largest walkable segment: {largestSegment.SurfaceArea:F1} area, " +
                                        $"{largestSegment.WalkableType}");
                        Console.WriteLine($"      This segment would be ideal for matching PM4 navigation data!");
                    }
                }
                else
                {
                    Console.WriteLine($"   ⚠️  No walkable surfaces extracted (likely OBJ file not found or parsing issue)");
                }
            }
            
            Console.WriteLine("\n🎯 WALKABLE SURFACE EXTRACTION BENEFITS:");
            Console.WriteLine("   • Focus on actual navigable areas instead of complete models");
            Console.WriteLine("   • Better spatial correlation with PM4 navigation data");
            Console.WriteLine("   • Segment complex buildings into matchable components");
            Console.WriteLine("   • Classify walkable areas by type and importance");
            Console.WriteLine("   • Multi-factor scoring for navigation-aware matching");
        }
    }

    /// <summary>
    /// Real PM4 Object extracted from actual PM4 files
    /// </summary>
    public class RealPM4Object
    {
        public int ObjectId { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Triangle> Triangles { get; set; } = new List<Triangle>();
        public BoundingBox3D BoundingBox { get; set; }
        public Vector3 Scale { get; set; }
        public float Volume { get; set; }
    }

    public struct Triangle
    {
        public Vector3 V1 { get; set; }
        public Vector3 V2 { get; set; }
        public Vector3 V3 { get; set; }
    }

    /// <summary>
    /// WMO Asset representation with CORRECT WMO properties (no scale!)
    /// </summary>
    public class WmoAsset
    {
        public string FileName { get; set; } = "";
        public string FilePath { get; set; } = "";
        public string RelativePath { get; set; } = "";
        public string Category { get; set; } = "";
        public long FileSizeBytes { get; set; }
        
        // CORRECTED: WMO actual properties
        public Vector3 Position { get; set; }          // World position
        public Vector3 Rotation { get; set; }          // Rotation angles (degrees)
        public Vector3 Tilt { get; set; }             // Tilt angles (degrees)
        public Vector3 Dimensions { get; set; }        // Bounding box dimensions (NOT scale!)
        
        // Geometric analysis
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public float Volume { get; set; }
        public float SurfaceArea { get; set; }
        public float Complexity { get; set; }
        public BoundingBox3D BoundingBox { get; set; }
        
        // Legacy compatibility (DEPRECATED - these were wrong!)
        [Obsolete("WMOs don't have scale - use Dimensions for bounding box size")]
        public Vector3 Scale => Dimensions;
    }

    /// <summary>
    /// 3D bounding box for real geometric analysis
    /// </summary>
    public struct BoundingBox3D
    {
        public Vector3 Min { get; }
        public Vector3 Max { get; }

        public BoundingBox3D(Vector3 min, Vector3 max)
        {
            Min = min;
            Max = max;
        }

        public Vector3 GetScale() => Max - Min;
        
        public Vector3 GetCenter() => (Min + Max) * 0.5f;
        
        public float GetVolume()
        {
            var scale = GetScale();
            return scale.X * scale.Y * scale.Z;
        }
    }

    /// <summary>
    /// Normalized geometry representation for shape comparison
    /// </summary>
    public class NormalizedGeometry
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Triangle> Triangles { get; set; } = new List<Triangle>();
        public Vector3 OriginalCentroid { get; set; }
        public Vector3 OriginalScale { get; set; }
        public float NormalizationFactor { get; set; }
    }

    /// <summary>
    /// Scale-invariant geometric features for shape comparison
    /// </summary>
    public class GeometricFeatures
    {
        public Vector3 AspectRatio { get; set; }        // X/Y, X/Z, Y/Z ratios
        public Vector3 Proportions { get; set; }        // Normalized dimensions (0-1)
        public float VolumeRatio { get; set; }          // Volume to surface area ratio
        public float Complexity { get; set; }          // Vertices per unit volume
        public float NormalizedVolume { get; set; }    // Volume of normalized object
    }

    /// <summary>
    /// Building cluster representing multiple 4-vertex PM4 objects grouped into a compound building
    /// </summary>
    public class BuildingCluster
    {
        public List<RealPM4Object> Objects { get; set; } = new List<RealPM4Object>();
        public Vector3 Footprint { get; set; }         // Overall building dimensions
        public Vector3 AspectRatio { get; set; }       // Building proportions
        public float TotalVolume { get; set; }         // Combined volume
        public BoundingBox3D CompoundBounds { get; set; }

        /// <summary>
        /// Calculates compound building characteristics from clustered objects
        /// </summary>
        public void CalculateCompoundCharacteristics()
        {
            if (Objects.Count == 0) return;

            // Calculate compound bounding box
            var allVertices = Objects.SelectMany(obj => obj.Vertices).ToList();
            CompoundBounds = CalculateBoundingBox(allVertices);
            
            // Building footprint (overall dimensions)
            Footprint = CompoundBounds.GetScale();
            
            // Total volume (sum of all collision objects)
            TotalVolume = Objects.Sum(obj => obj.Volume);
            
            // Building aspect ratios
            AspectRatio = new Vector3(
                Footprint.Y > 0 ? Footprint.X / Footprint.Y : 1.0f,
                Footprint.Z > 0 ? Footprint.X / Footprint.Z : 1.0f,
                Footprint.Z > 0 ? Footprint.Y / Footprint.Z : 1.0f
            );
        }

        static BoundingBox3D CalculateBoundingBox(List<Vector3> vertices)
        {
            if (vertices.Count == 0)
                return new BoundingBox3D(Vector3.Zero, Vector3.Zero);

            var min = vertices[0];
            var max = vertices[0];

            foreach (var vertex in vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            return new BoundingBox3D(min, max);
        }
    }

    /// <summary>
    /// PM4 object with complete positioning and orientation data
    /// </summary>
    public class PM4ObjectWithPositioning
    {
        public int ObjectId { get; set; }
        public Vector3 Position { get; set; }
        public Vector3 Scale { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public float Volume { get; set; }
        public float SurfaceArea { get; set; }
        public float Complexity { get; set; }
        public int HierarchyDepth { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
        public List<Vector3> Vertices { get; set; } = new();
        public BoundingBox3D BoundingBox { get; set; }
        public string Source { get; set; } = "";
        public PM4ComplexityClass ComplexityClass { get; set; }
    }

    /// <summary>
    /// Detailed WMO positional match result with comprehensive data
    /// </summary>
    public class WmoPositionalMatch
    {
        public required WmoAsset WmoAsset { get; set; }
        public required PM4ObjectWithPositioning PM4Object { get; set; }
        public float MatchScore { get; set; }
        public Vector3 PositionOffset { get; set; }
        public float ScaleRatio { get; set; }
        public Vector3 RotationDelta { get; set; }
        public string Analysis { get; set; } = "";
        
        // Legacy properties for backward compatibility
        public string AssetName => WmoAsset?.FileName ?? "";
        public string AssetPath => WmoAsset?.FilePath ?? "";
        public string Category => WmoAsset?.Category ?? "";
        public float Confidence => MatchScore;
    }

    // ENHANCED DATA STRUCTURES FOR COMPREHENSIVE PM4 ANALYSIS

    public class MSLKHierarchyData
    {
        public int TotalEntries { get; set; }
        public List<int> RootNodes { get; set; } = new();
        public int NodesWithGeometry { get; set; }
        public Dictionary<int, int> ParentChildRelations { get; set; } = new();
    }

    public class MSURSurfaceData
    {
        public int SurfaceCount { get; set; }
        public int TotalTriangles { get; set; }
        public float AverageTrianglesPerSurface { get; set; }
        public int MinTrianglesPerSurface { get; set; }
        public int MaxTrianglesPerSurface { get; set; }
    }

    public class VertexDetailData
    {
        public int MSVTVertices { get; set; }
        public int MSPVVertices { get; set; }
        public int TotalVertices { get; set; }
        public BoundingBox3D? MSVTBounds { get; set; }
        public float MSVTDensity { get; set; }
    }

    // COMPLEXITY-AWARE MATCHING SYSTEM

    public enum PM4ComplexityClass
    {
        CollisionHull,       // 4-9 vertices (basic collision hulls)
        SimplifiedGeometry,  // 10-50 vertices (simplified models)
        FullGeometry        // 50+ vertices (complex geometry)
    }

    public class MatchResult
    {
        public float Score { get; set; }
        public float DimensionalSimilarity { get; set; }
        public string Analysis { get; set; } = "";
    }

    // NORMALIZED OBJECT STRUCTURES FOR COORDINATE-AWARE MATCHING

    public class NormalizedPM4Object
    {
        public Vector3 OriginalPosition { get; set; }
        public Vector3 NormalizedPosition { get; set; }
        public Vector3 RelativeDimensions { get; set; }
        public Vector3 BoundingBoxSize { get; set; }
        public Vector3 LocalCoordinateSystem { get; set; }
        public int VertexCount { get; set; }
        public float Volume { get; set; }
        public float SurfaceArea { get; set; }
        public float Complexity { get; set; }
    }

    public class NormalizedWMOAsset
    {
        public required WmoAsset OriginalAsset { get; set; }
        public Vector3 OriginalPosition { get; set; }
        public Vector3 NormalizedPosition { get; set; }
        public Vector3 RelativeDimensions { get; set; }
        public Vector3 BoundingBoxSize { get; set; }
        public Vector3 LocalCoordinateSystem { get; set; }
        public int VertexCount { get; set; }
        public float Volume { get; set; }
        public float SurfaceArea { get; set; }
        public float Complexity { get; set; }
    }

    /// <summary>
    /// ENHANCED WALKABLE SURFACE EXTRACTION SYSTEM
    /// Extracts only the navigable/walkable surfaces from WMO models for proper PM4 matching
    /// </summary>
    public class WalkableSurfaceExtractor
    {
        private const float WALKABLE_ANGLE_THRESHOLD = 45.0f; // Maximum slope for walkable surface (degrees)
        private const float SURFACE_SEGMENT_DISTANCE = 10.0f; // Distance to group nearby walkable surfaces
        private const int MIN_WALKABLE_VERTICES = 3; // Minimum vertices to consider a valid walkable surface
        
        /// <summary>
        /// Extracts walkable surface segments from a WMO asset
        /// </summary>
        public static List<WalkableSurfaceSegment> ExtractWalkableSurfaces(WmoAsset wmoAsset)
        {
            var walkableSegments = new List<WalkableSurfaceSegment>();
            
            try
            {
                // Load WMO geometry (this would need to be implemented based on your WMO parsing)
                var wmoGeometry = LoadWmoGeometry(wmoAsset.FilePath);
                if (wmoGeometry == null) return walkableSegments;
                
                // Step 1: Identify all walkable faces (horizontal or near-horizontal)
                var walkableFaces = IdentifyWalkableFaces(wmoGeometry);
                Console.WriteLine($"         🚶 Found {walkableFaces.Count} potentially walkable faces in {wmoAsset.FileName}");
                
                // Step 2: Group nearby walkable faces into coherent surface segments
                var surfaceSegments = GroupWalkableFacesIntoSegments(walkableFaces);
                Console.WriteLine($"         📐 Grouped into {surfaceSegments.Count} walkable surface segments");
                
                                 // Step 3: Create walkable surface segments with proper metadata
                 foreach (var segment in surfaceSegments)
                 {
                     // Populate vertices from indices
                     segment.PopulateVertices(wmoGeometry);
                     
                     if (segment.Vertices.Count >= MIN_WALKABLE_VERTICES)
                     {
                         var walkableSegment = new WalkableSurfaceSegment
                         {
                             OriginalWmoAsset = wmoAsset,
                             Vertices = segment.Vertices,
                             Faces = segment.Faces,
                             BoundingBox = CalculateBoundingBoxFromVertices(segment.Vertices),
                             SurfaceArea = CalculateSurfaceArea(segment.Faces),
                             AverageHeight = segment.Vertices.Average(v => v.Z),
                             WalkableType = ClassifyWalkableType(segment),
                             SegmentId = walkableSegments.Count
                         };
                         
                         walkableSegments.Add(walkableSegment);
                     }
                 }
                
                Console.WriteLine($"         ✅ Extracted {walkableSegments.Count} walkable segments from {wmoAsset.FileName}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"         ❌ Error extracting walkable surfaces from {wmoAsset.FileName}: {ex.Message}");
            }
            
            return walkableSegments;
        }
        
        /// <summary>
        /// Identifies faces that are walkable (horizontal or near-horizontal surfaces)
        /// </summary>
        private static List<WalkableFace> IdentifyWalkableFaces(WmoGeometry geometry)
        {
            var walkableFaces = new List<WalkableFace>();
            var upVector = Vector3.UnitZ; // Assuming Z is up
            
            foreach (var face in geometry.Faces)
            {
                // Calculate face normal
                var normal = CalculateFaceNormal(face, geometry.Vertices);
                
                // Check if face is mostly horizontal (angle with up vector < threshold)
                var angleWithUp = MathF.Acos(Vector3.Dot(NormalizeVector3(normal), upVector)) * (180.0f / MathF.PI);
                
                if (angleWithUp <= WALKABLE_ANGLE_THRESHOLD)
                {
                    walkableFaces.Add(new WalkableFace
                    {
                        Face = face,
                        Normal = normal,
                        AngleWithUp = angleWithUp,
                        CenterPoint = CalculateFaceCenter(face, geometry.Vertices),
                        SurfaceArea = CalculateFaceArea(face, geometry.Vertices)
                    });
                }
            }
            
            return walkableFaces;
        }
        
        /// <summary>
        /// Groups nearby walkable faces into coherent surface segments
        /// </summary>
        private static List<SurfaceSegment> GroupWalkableFacesIntoSegments(List<WalkableFace> walkableFaces)
        {
            var segments = new List<SurfaceSegment>();
            var processed = new HashSet<int>();
            
            for (int i = 0; i < walkableFaces.Count; i++)
            {
                if (processed.Contains(i)) continue;
                
                var segment = new SurfaceSegment();
                var faceQueue = new Queue<int>();
                faceQueue.Enqueue(i);
                processed.Add(i);
                
                // Flood fill to find connected walkable faces
                while (faceQueue.Count > 0)
                {
                    var currentFaceIndex = faceQueue.Dequeue();
                    var currentFace = walkableFaces[currentFaceIndex];
                    segment.Faces.Add(currentFace.Face);
                    
                    // Add vertices from this face
                    foreach (var vertexIndex in currentFace.Face.VertexIndices)
                    {
                        if (!segment.VertexIndices.Contains(vertexIndex))
                        {
                            segment.VertexIndices.Add(vertexIndex);
                        }
                    }
                    
                    // Find nearby unprocessed faces
                    for (int j = 0; j < walkableFaces.Count; j++)
                    {
                        if (processed.Contains(j)) continue;
                        
                        var distance = Vector3.Distance(currentFace.CenterPoint, walkableFaces[j].CenterPoint);
                        if (distance <= SURFACE_SEGMENT_DISTANCE)
                        {
                            faceQueue.Enqueue(j);
                            processed.Add(j);
                        }
                    }
                }
                
                segments.Add(segment);
            }
            
            return segments;
        }
        
        /// <summary>
        /// Classifies the type of walkable surface
        /// </summary>
        private static WalkableSurfaceType ClassifyWalkableType(SurfaceSegment segment)
        {
            var area = segment.Faces.Sum(f => f.SurfaceArea);
            var avgHeight = segment.Vertices.Average(v => v.Z);
            
            if (area > 100.0f) return WalkableSurfaceType.LargeFloor;
            if (area > 20.0f) return WalkableSurfaceType.Platform;
            if (avgHeight > 50.0f) return WalkableSurfaceType.Elevated;
            return WalkableSurfaceType.SmallSurface;
        }
        
        // Helper methods (these would need to be implemented based on your WMO format)
        private static WmoGeometry? LoadWmoGeometry(string filePath) 
        {
            try
            {
                // Convert WMO file path to corresponding OBJ file path
                var objPath = ConvertWmoPathToObjPath(filePath);
                if (!File.Exists(objPath))
                {
                    Console.WriteLine($"         ⚠️  OBJ file not found: {objPath}");
                    return null;
                }
                
                Console.WriteLine($"         📂 Loading OBJ geometry from: {Path.GetFileName(objPath)}");
                
                var geometry = new WmoGeometry();
                var lines = File.ReadAllLines(objPath);
                
                foreach (var line in lines)
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length == 0) continue;
                    
                    switch (parts[0].ToLower())
                    {
                        case "v": // Vertex
                            if (parts.Length >= 4 &&
                                float.TryParse(parts[1], out float x) &&
                                float.TryParse(parts[2], out float y) &&
                                float.TryParse(parts[3], out float z))
                            {
                                // Note: OBJ files use +Z as up, might need coordinate transformation
                                geometry.Vertices.Add(new Vector3(x, y, z));
                            }
                            break;
                            
                        case "f": // Face
                            if (parts.Length >= 4) // At least triangle
                            {
                                var face = new Face();
                                for (int i = 1; i < parts.Length; i++)
                                {
                                    // OBJ indices are 1-based, convert to 0-based
                                    var indexStr = parts[i].Split('/')[0]; // Handle "vertex/texture/normal" format
                                    if (int.TryParse(indexStr, out int index) && index > 0)
                                    {
                                        face.VertexIndices.Add(index - 1);
                                    }
                                }
                                
                                if (face.VertexIndices.Count >= 3)
                                {
                                    face.SurfaceArea = CalculateFaceArea(face, geometry.Vertices);
                                    geometry.Faces.Add(face);
                                }
                            }
                            break;
                    }
                }
                
                Console.WriteLine($"         ✅ Loaded {geometry.Vertices.Count} vertices, {geometry.Faces.Count} faces");
                return geometry;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"         ❌ Error loading WMO geometry: {ex.Message}");
                return null;
            }
        }
        
        /// <summary>
        /// Converts WMO file path to corresponding OBJ file path in the wmo_335-objs directory
        /// </summary>
        private static string ConvertWmoPathToObjPath(string wmoPath)
        {
            // Example: Convert "I:\...\test_data\wmo_335-objs\wmo\World\wmo\Azeroth\Buildings\Human_Buildings\StormwindCity_Objects.wmo"
            // to corresponding OBJ path
            var fileName = Path.GetFileNameWithoutExtension(wmoPath);
            var directory = Path.GetDirectoryName(wmoPath) ?? "";
            
            // Replace .wmo extension with .obj
            var objFileName = fileName + ".obj";
            var objPath = Path.Combine(directory, objFileName);
            
            return objPath;
        }
        
        private static Vector3 CalculateFaceNormal(Face face, List<Vector3> vertices)
        {
            if (face.VertexIndices.Count < 3) return Vector3.UnitZ;
            
            var v1 = vertices[face.VertexIndices[1]] - vertices[face.VertexIndices[0]];
            var v2 = vertices[face.VertexIndices[2]] - vertices[face.VertexIndices[0]];
            return NormalizeVector3(Vector3.Cross(v1, v2));
        }
        
        private static Vector3 CalculateFaceCenter(Face face, List<Vector3> vertices)
        {
            var center = Vector3.Zero;
            foreach (var index in face.VertexIndices)
            {
                center += vertices[index];
            }
            return center / face.VertexIndices.Count;
        }
        
        private static float CalculateFaceArea(Face face, List<Vector3> vertices)
        {
            if (face.VertexIndices.Count < 3) return 0.0f;
            
            // Simple triangle area calculation (for more complex faces, would need triangulation)
            var v1 = vertices[face.VertexIndices[1]] - vertices[face.VertexIndices[0]];
            var v2 = vertices[face.VertexIndices[2]] - vertices[face.VertexIndices[0]];
            return Vector3.Cross(v1, v2).Length() * 0.5f;
        }
        
        private static float CalculateSurfaceArea(List<Face> faces)
        {
            return faces.Sum(f => f.SurfaceArea);
        }
        
        /// <summary>
        /// Helper method to calculate bounding box from vertices (for walkable surface extraction)
        /// </summary>
        private static BoundingBox3D CalculateBoundingBoxFromVertices(List<Vector3> vertices)
        {
            if (vertices.Count == 0)
                return new BoundingBox3D(Vector3.Zero, Vector3.Zero);

            var min = vertices[0];
            var max = vertices[0];

            foreach (var vertex in vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }

            return new BoundingBox3D(min, max);
        }
        
        /// <summary>
        /// Normalizes a Vector3 (creates unit vector)
        /// </summary>
        private static Vector3 NormalizeVector3(Vector3 vector)
        {
            var length = vector.Length();
            if (length > 0.000001f)
            {
                return vector / length;
            }
            return Vector3.UnitZ; // Default to up vector if zero length
        }
    }
    
    // WALKABLE SURFACE DATA STRUCTURES
    
    /// <summary>
    /// Represents a walkable surface segment extracted from a WMO model
    /// </summary>
    public class WalkableSurfaceSegment
    {
        public required WmoAsset OriginalWmoAsset { get; set; }
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Face> Faces { get; set; } = new List<Face>();
        public BoundingBox3D BoundingBox { get; set; }
        public float SurfaceArea { get; set; }
        public float AverageHeight { get; set; }
        public WalkableSurfaceType WalkableType { get; set; }
        public int SegmentId { get; set; }
        
        /// <summary>
        /// Matches this walkable surface against a PM4 object
        /// </summary>
        public float CalculateMatchScore(PM4ObjectWithPositioning pm4Object)
        {
            var score = 0.0f;
            
            // Spatial proximity (30%)
            var distance = Vector3.Distance(BoundingBox.GetCenter(), pm4Object.Position);
            var proximityScore = MathF.Exp(-distance / 50.0f); // Exponential decay
            score += proximityScore * 0.3f;
            
            // Surface area compatibility (25%)
            var areaRatio = MathF.Min(SurfaceArea, pm4Object.SurfaceArea) / MathF.Max(SurfaceArea, pm4Object.SurfaceArea);
            score += areaRatio * 0.25f;
            
            // Height compatibility (20%)
            var heightDiff = MathF.Abs(AverageHeight - pm4Object.Position.Z);
            var heightScore = MathF.Exp(-heightDiff / 10.0f);
            score += heightScore * 0.2f;
            
            // Vertex count reasonableness (15%)
            var vertexRatio = MathF.Min(Vertices.Count, pm4Object.VertexCount) / MathF.Max(Vertices.Count, pm4Object.VertexCount);
            score += vertexRatio * 0.15f;
            
            // Walkable surface type bonus (10%)
            var typeBonus = WalkableType == WalkableSurfaceType.LargeFloor ? 1.0f : 0.5f;
            score += typeBonus * 0.1f;
            
            return score;
        }
    }
    
    public enum WalkableSurfaceType
    {
        LargeFloor,      // Large walkable floor areas
        Platform,        // Medium-sized platforms
        Elevated,        // Elevated walkable surfaces
        SmallSurface     // Small walkable areas
    }
    
    public class WalkableFace
    {
        public required Face Face { get; set; }
        public Vector3 Normal { get; set; }
        public float AngleWithUp { get; set; }
        public Vector3 CenterPoint { get; set; }
        public float SurfaceArea { get; set; }
    }
    
         public class SurfaceSegment
     {
         public List<Face> Faces { get; set; } = new List<Face>();
         public List<int> VertexIndices { get; set; } = new List<int>();
         public List<Vector3> Vertices { get; set; } = new List<Vector3>();
         
         /// <summary>
         /// Populates vertices from vertex indices using the provided geometry
         /// </summary>
         public void PopulateVertices(WmoGeometry geometry)
         {
             Vertices.Clear();
             foreach (var index in VertexIndices)
             {
                 if (index >= 0 && index < geometry.Vertices.Count)
                 {
                     Vertices.Add(geometry.Vertices[index]);
                 }
             }
         }
     }
    
    public class Face
    {
        public List<int> VertexIndices { get; set; } = new List<int>();
        public float SurfaceArea { get; set; }
    }
    
    public class WmoGeometry
    {
        public List<Vector3> Vertices { get; set; } = new List<Vector3>();
        public List<Face> Faces { get; set; } = new List<Face>();
    }

    /// <summary>
    /// ORIENTATION-AWARE MATCHING SYSTEM
    /// Tests different orientations and transformations to find proper PM4-WMO alignment
    /// </summary>
    public class OrientationMatcher
    {
        private static readonly Vector3[] CommonRotations = new[]
        {
            new Vector3(0, 0, 0),       // No rotation
            new Vector3(0, 90, 0),      // 90° Y rotation
            new Vector3(0, 180, 0),     // 180° Y rotation  
            new Vector3(0, 270, 0),     // 270° Y rotation
            new Vector3(90, 0, 0),      // 90° X rotation
            new Vector3(180, 0, 0),     // 180° X rotation (upside down)
            new Vector3(270, 0, 0),     // 270° X rotation
            new Vector3(0, 0, 90),      // 90° Z rotation
            new Vector3(0, 0, 180),     // 180° Z rotation
            new Vector3(0, 0, 270),     // 270° Z rotation
        };
        
        /// <summary>
        /// Tests multiple orientations and coordinate transformations to find best match
        /// </summary>
        public static OrientationMatchResult FindBestOrientation(PM4ObjectWithPositioning pm4Object, WmoAsset wmoAsset)
        {
            var bestResult = new OrientationMatchResult();
            
            foreach (var rotation in CommonRotations)
            {
                // Test normal coordinates
                var normalResult = TestOrientation(pm4Object, wmoAsset, rotation, false, false, false);
                if (normalResult.Score > bestResult.Score)
                {
                    bestResult = normalResult;
                }
                
                // Test with X-Y swap
                var swapXYResult = TestOrientation(pm4Object, wmoAsset, rotation, true, false, false);
                if (swapXYResult.Score > bestResult.Score)
                {
                    bestResult = swapXYResult;
                }
                
                // Test with Z inversion (flip vertical)
                var flipZResult = TestOrientation(pm4Object, wmoAsset, rotation, false, false, true);
                if (flipZResult.Score > bestResult.Score)
                {
                    bestResult = flipZResult;
                }
                
                // Test with Y-Z swap (different up axis)
                var swapYZResult = TestOrientation(pm4Object, wmoAsset, rotation, false, true, false);
                if (swapYZResult.Score > bestResult.Score)
                {
                    bestResult = swapYZResult;
                }
            }
            
            return bestResult;
        }
        
        private static OrientationMatchResult TestOrientation(PM4ObjectWithPositioning pm4Object, WmoAsset wmoAsset, 
            Vector3 rotation, bool swapXY, bool swapYZ, bool flipZ)
        {
            // Transform PM4 object vertices with the given orientation
            var transformedVertices = new List<Vector3>();
            foreach (var vertex in pm4Object.Vertices)
            {
                var transformed = ApplyTransformation(vertex, rotation, swapXY, swapYZ, flipZ);
                transformedVertices.Add(transformed);
            }
            
            // Calculate transformed bounding box
            var transformedBounds = WmoMatchingDemo.CalculateBoundingBox(transformedVertices);
            var transformedScale = transformedBounds.GetScale();
            
            // Calculate match score with transformed geometry
            var score = CalculateOrientationScore(transformedScale, transformedBounds.GetCenter(), wmoAsset);
            
            return new OrientationMatchResult
            {
                Score = score,
                Rotation = rotation,
                SwapXY = swapXY,
                SwapYZ = swapYZ,
                FlipZ = flipZ,
                TransformedVertices = transformedVertices,
                TransformedBounds = transformedBounds,
                Description = GetTransformationDescription(rotation, swapXY, swapYZ, flipZ)
            };
        }
        
        private static Vector3 ApplyTransformation(Vector3 vertex, Vector3 rotation, bool swapXY, bool swapYZ, bool flipZ)
        {
            var result = vertex;
            
            // Apply coordinate swaps
            if (swapXY)
            {
                result = new Vector3(result.Y, result.X, result.Z);
            }
            if (swapYZ)
            {
                result = new Vector3(result.X, result.Z, result.Y);
            }
            if (flipZ)
            {
                result = new Vector3(result.X, result.Y, -result.Z);
            }
            
            // Apply rotations (simplified - could use proper rotation matrices for more accuracy)
            if (rotation.Y != 0) // Y rotation (most common for buildings)
            {
                var radians = rotation.Y * MathF.PI / 180.0f;
                var cos = MathF.Cos(radians);
                var sin = MathF.Sin(radians);
                result = new Vector3(
                    result.X * cos - result.Z * sin,
                    result.Y,
                    result.X * sin + result.Z * cos
                );
            }
            
            return result;
        }
        
        private static float CalculateOrientationScore(Vector3 pm4Scale, Vector3 pm4Center, WmoAsset wmoAsset)
        {
            var score = 0.0f;
            
            // Dimensional similarity (40%)
            var dimSimilarity = WmoMatchingDemo.CalculateDimensionalSimilarity(pm4Scale, wmoAsset.Dimensions);
            score += dimSimilarity * 0.4f;
            
            // Position proximity (30%)
            var distance = Vector3.Distance(pm4Center, wmoAsset.Position);
            var proximityScore = MathF.Exp(-distance / 100.0f);
            score += proximityScore * 0.3f;
            
            // Volume compatibility (20%)
            var pm4Volume = pm4Scale.X * pm4Scale.Y * pm4Scale.Z;
            var volumeCompat = WmoMatchingDemo.CalculateVolumeCompatibility(pm4Volume, wmoAsset.Volume);
            score += volumeCompat * 0.2f;
            
            // Aspect ratio match (10%)
            var pm4Ratio = CalculateAspectRatio(pm4Scale);
            var wmoRatio = CalculateAspectRatio(wmoAsset.Dimensions);
            var ratioMatch = 1.0f - MathF.Abs(pm4Ratio - wmoRatio) / MathF.Max(pm4Ratio, wmoRatio);
            score += ratioMatch * 0.1f;
            
            return score;
        }
        
        private static float CalculateAspectRatio(Vector3 scale)
        {
            if (scale.Y > 0 && scale.Z > 0)
            {
                return (scale.X / scale.Y) + (scale.X / scale.Z) + (scale.Y / scale.Z);
            }
            return 1.0f;
        }
        
        private static string GetTransformationDescription(Vector3 rotation, bool swapXY, bool swapYZ, bool flipZ)
        {
            var parts = new List<string>();
            
            if (rotation != Vector3.Zero)
            {
                parts.Add($"Rot({rotation.X}°,{rotation.Y}°,{rotation.Z}°)");
            }
            if (swapXY) parts.Add("SwapXY");
            if (swapYZ) parts.Add("SwapYZ");
            if (flipZ) parts.Add("FlipZ");
            
            return parts.Count > 0 ? string.Join("+", parts) : "Identity";
        }
    }
    
    public class OrientationMatchResult
    {
        public float Score { get; set; }
        public Vector3 Rotation { get; set; }
        public bool SwapXY { get; set; }
        public bool SwapYZ { get; set; }
        public bool FlipZ { get; set; }
        public List<Vector3> TransformedVertices { get; set; } = new();
        public BoundingBox3D TransformedBounds { get; set; }
        public string Description { get; set; } = "";
    }
    
    /// <summary>
    /// HIERARCHY-AWARE EXTRACTION SYSTEM
    /// Prevents merging of unrelated geometry by analyzing MSLK hierarchy properly
    /// </summary>
    public class HierarchyAnalyzer
    {
        /// <summary>
        /// Extracts individual objects without merging unrelated geometry
        /// </summary>
        public static List<PM4ObjectWithPositioning> ExtractIndividualObjects(PM4File pm4File, string filePath)
        {
            var objects = new List<PM4ObjectWithPositioning>();
            
            if (pm4File.MSLK?.Entries == null) return objects;
            
            Console.WriteLine($"      🔍 HIERARCHY ANALYSIS: {pm4File.MSLK.Entries.Count} MSLK entries");
            
            // Analyze hierarchy relationships
            var hierarchyMap = BuildHierarchyMap(pm4File.MSLK.Entries);
            
            // Find root objects (objects that are not children of others)
            var rootObjects = FindRootObjects(hierarchyMap);
            Console.WriteLine($"      📊 Found {rootObjects.Count} root objects");
            
            // Extract each root object individually
            foreach (var rootIndex in rootObjects)
            {
                var individualObject = ExtractSingleObject(pm4File, rootIndex, hierarchyMap);
                if (individualObject != null && individualObject.VertexCount > 0)
                {
                    objects.Add(individualObject);
                    Console.WriteLine($"      ✅ Extracted object {rootIndex}: {individualObject.VertexCount} vertices");
                }
            }
            
            return objects;
        }
        
        private static Dictionary<int, HashSet<int>> BuildHierarchyMap(List<WoWToolbox.Core.v2.Models.PM4.Chunks.MSLKEntry> entries)
        {
            var hierarchy = new Dictionary<int, HashSet<int>>();
            
            for (int i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                var parentIndex = (int)entry.Unknown_0x04;
                
                if (parentIndex != i) // Not self-referencing
                {
                    if (!hierarchy.ContainsKey(parentIndex))
                    {
                        hierarchy[parentIndex] = new HashSet<int>();
                    }
                    hierarchy[parentIndex].Add(i);
                }
            }
            
            return hierarchy;
        }
        
        private static List<int> FindRootObjects(Dictionary<int, HashSet<int>> hierarchyMap)
        {
            var allChildren = new HashSet<int>();
            foreach (var children in hierarchyMap.Values)
            {
                foreach (var child in children)
                {
                    allChildren.Add(child);
                }
            }
            
            var roots = new List<int>();
            foreach (var parent in hierarchyMap.Keys)
            {
                if (!allChildren.Contains(parent))
                {
                    roots.Add(parent);
                }
            }
            
            return roots;
        }
        
        private static PM4ObjectWithPositioning? ExtractSingleObject(PM4File pm4File, int rootIndex, Dictionary<int, HashSet<int>> hierarchyMap)
        {
            // Extract only this specific object and its direct children (not distant relatives)
            var objectIndices = new HashSet<int> { rootIndex };
            
            // Add direct children only (avoid merging unrelated geometry)
            if (hierarchyMap.ContainsKey(rootIndex))
            {
                foreach (var child in hierarchyMap[rootIndex])
                {
                    objectIndices.Add(child);
                    
                    // Only add grandchildren if they're small decorative elements
                    if (hierarchyMap.ContainsKey(child) && hierarchyMap[child].Count <= 2)
                    {
                        foreach (var grandchild in hierarchyMap[child])
                        {
                            objectIndices.Add(grandchild);
                        }
                    }
                }
            }
            
            // Now extract vertices only from these specific indices
            var vertices = new List<Vector3>();
            foreach (var index in objectIndices)
            {
                if (index < pm4File.MSLK.Entries.Count)
                {
                    var entry = pm4File.MSLK.Entries[index];
                    ExtractVerticesFromEntry(pm4File, entry, vertices);
                }
            }
            
            if (vertices.Count == 0) return null;
            
            var bounds = WmoMatchingDemo.CalculateBoundingBox(vertices);
            return new PM4ObjectWithPositioning
            {
                ObjectId = rootIndex,
                Vertices = vertices,
                BoundingBox = bounds,
                VertexCount = vertices.Count,
                Position = bounds.GetCenter(),
                Scale = bounds.GetScale(),
                Volume = bounds.GetVolume(),
                Source = "HierarchyAware",
                ComplexityClass = vertices.Count <= 4 ? PM4ComplexityClass.CollisionHull : 
                                vertices.Count <= 50 ? PM4ComplexityClass.SimplifiedGeometry : PM4ComplexityClass.FullGeometry
            };
        }
        
        private static void ExtractVerticesFromEntry(PM4File pm4File, WoWToolbox.Core.v2.Models.PM4.Chunks.MSLKEntry entry, List<Vector3> vertices)
        {
            if (pm4File.MSPI?.Indices == null || pm4File.MSPV?.Vertices == null) return;
            
            for (int i = entry.MspiFirstIndex; i < entry.MspiFirstIndex + entry.MspiIndexCount && i < pm4File.MSPI.Indices.Count; i++)
            {
                var vertexIndex = pm4File.MSPI.Indices[i];
                if (vertexIndex < pm4File.MSPV.Vertices.Count)
                {
                    var vertex = pm4File.MSPV.Vertices[(int)vertexIndex];
                    vertices.Add(new Vector3(vertex.X, vertex.Y, vertex.Z));
                }
            }
        }
    }
} 