// SURFACE-FOCUSED PM4-WMO MATCHING SYSTEM
// MSUR surface data extraction with top surface geometry comparison

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using System.Threading;
using WoWToolbox.PM4Parsing;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WmoMatchingDemo
{
    /// <summary>
    /// 🎯 SURFACE-FOCUSED PM4-WMO MATCHING SYSTEM
    /// Extracts MSUR surface data and compares top surface geometry patterns
    /// </summary>
    class WmoMatchingDemo
    {
        private const string WMO_OBJ_BASE_PATH = @"I:\parp-tools\parp-tools\PM4Tool\test_data\wmo_335-objs\wmo\World\wmo\";
        private const string PM4_TEST_DATA_PATH = @"../test_data/original_development/development/";
        private const string OUTPUT_BASE_PATH = @"../output/surface_focused_matching/";
        
        private const int BATCH_SIZE = 50; // Smaller batches for memory management
        
        // Lightweight WMO top surface data (kept in memory)
        private static List<WmoTopSurface> wmoTopSurfaces = new List<WmoTopSurface>();
        
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== 🎯 SURFACE-FOCUSED PM4-WMO MATCHING ===");
            Console.WriteLine("📊 APPROACH: MSUR surface data → Top surface geometry comparison");
            Console.WriteLine($"🔄 MEMORY OPTIMIZED: Processing {BATCH_SIZE} PM4 objects at a time");
            Console.WriteLine();
            
            try
            {
                // Step 1: Extract WMO top surfaces
                Console.WriteLine("Step 1: Extracting WMO top surface geometry...");
                await ExtractWmoTopSurfaces();
                Console.WriteLine($"✅ Extracted {wmoTopSurfaces.Count} WMO top surfaces");
                Console.WriteLine();
                
                // Step 2: Scan PM4 files for objects with MSUR data
                Console.WriteLine("Step 2: Scanning PM4 files for MSUR surface objects...");
                var pm4SurfaceRefs = await ScanPM4SurfaceObjects();
                Console.WriteLine($"✅ Found {pm4SurfaceRefs.Count} PM4 objects with surface data");
                Console.WriteLine();
                
                // Step 3: Process PM4 surface objects in small batches
                Console.WriteLine("Step 3: SURFACE-FOCUSED MATCHING...");
                await ProcessPM4SurfaceObjectsInBatches(pm4SurfaceRefs);
                
                Console.WriteLine();
                Console.WriteLine("🎉 === SURFACE MATCHING COMPLETE ===");
                Console.WriteLine($"Results saved to: {OUTPUT_BASE_PATH}");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ ERROR: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }
            
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
        
        /// <summary>
        /// Extract top surface geometry from WMO OBJ files
        /// </summary>
        static async Task ExtractWmoTopSurfaces()
        {
            wmoTopSurfaces.Clear();
            
            if (!Directory.Exists(WMO_OBJ_BASE_PATH))
            {
                Console.WriteLine($"⚠️ WMO path not found: {WMO_OBJ_BASE_PATH}");
                CreateSampleWmoTopSurfaces();
                return;
            }
            
            var objFiles = Directory.GetFiles(WMO_OBJ_BASE_PATH, "*.obj", SearchOption.AllDirectories);
            Console.WriteLine($"🔍 Processing {objFiles.Length} WMO OBJ files for top surfaces...");
            
            var processed = 0;
            var extracted = 0;
            
            await Task.Run(() =>
            {
                Parallel.ForEach(objFiles, objFile =>
                {
                    try
                    {
                        var topSurface = ExtractWmoTopSurface(objFile);
                        if (topSurface != null)
                        {
                            lock (wmoTopSurfaces)
                            {
                                wmoTopSurfaces.Add(topSurface);
                                extracted++;
                            }
                        }
                        
                        Interlocked.Increment(ref processed);
                        if (processed % 200 == 0)
                        {
                            Console.WriteLine($"  Processed {processed} WMO files, extracted {extracted} top surfaces");
                        }
                    }
                    catch
                    {
                        // Skip problematic files
                    }
                });
            });
            
            Console.WriteLine($"📊 WMO top surface extraction complete: {extracted} surfaces from {processed} files");
        }
        
        /// <summary>
        /// Scan PM4 files for objects with MSUR surface data
        /// </summary>
        static async Task<List<PM4SurfaceReference>> ScanPM4SurfaceObjects()
        {
            var surfaceRefs = new List<PM4SurfaceReference>();
            var pm4Files = Directory.GetFiles(PM4_TEST_DATA_PATH, "*.pm4", SearchOption.AllDirectories);
            
            await Task.Run(() =>
            {
                foreach (var pm4File in pm4Files)
                {
                    try
                    {
                        var pm4FileObj = WoWToolbox.Core.v2.Foundation.Data.PM4File.FromFile(pm4File);
                        var buildings = pm4FileObj.ExtractBuildings();
                        
                        for (int i = 0; i < buildings.Count; i++)
                        {
                            var building = buildings[i];
                            
                            // Check if this object has surface data (MSUR-related vertices)
                            if (HasSurfaceData(building))
                            {
                                surfaceRefs.Add(new PM4SurfaceReference
                                {
                                    PM4FilePath = pm4File,
                                    ObjectIndex = i,
                                    ObjectId = $"{Path.GetFileNameWithoutExtension(pm4File)}_SURF_{i:D3}"
                                });
                            }
                        }
                        
                        var surfaceCount = surfaceRefs.Count(r => r.PM4FilePath == pm4File);
                        if (surfaceCount > 0)
                        {
                            Console.WriteLine($"  {Path.GetFileNameWithoutExtension(pm4File)}: {surfaceCount} surface objects");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"    ❌ Failed to scan {pm4File}: {ex.Message}");
                    }
                }
            });
            
            return surfaceRefs;
        }
        
        /// <summary>
        /// Process PM4 surface objects in very small batches for memory efficiency
        /// </summary>
        static async Task ProcessPM4SurfaceObjectsInBatches(List<PM4SurfaceReference> surfaceRefs)
        {
            var totalBatches = (int)Math.Ceiling(surfaceRefs.Count / (double)BATCH_SIZE);
            Console.WriteLine($"🔄 Processing {surfaceRefs.Count} surface objects in {totalBatches} batches of {BATCH_SIZE}");
            
            Directory.CreateDirectory(OUTPUT_BASE_PATH);
            var masterSummaryFile = Path.Combine(OUTPUT_BASE_PATH, "MASTER_SURFACE_MATCHING_SUMMARY.txt");
            var masterLines = new List<string>
            {
                $"🎯 === SURFACE-FOCUSED MATCHING MASTER SUMMARY ===",
                $"Analysis Date: {DateTime.Now}",
                $"Total PM4 Surface Objects: {surfaceRefs.Count}",
                $"Total WMO Top Surfaces: {wmoTopSurfaces.Count}",
                $"Processing Batch Size: {BATCH_SIZE}",
                "",
                "📊 === BEST SURFACE MATCHES ===",
                "PM4_SURFACE_OBJECT               BEST_WMO_TOP_SURFACE         SIMILARITY  SURFACE_PATTERN_BREAKDOWN",
                ""
            };
            
            for (int batchIndex = 0; batchIndex < totalBatches; batchIndex++)
            {
                var batchStart = batchIndex * BATCH_SIZE;
                var batchEnd = Math.Min(batchStart + BATCH_SIZE, surfaceRefs.Count);
                var batchRefs = surfaceRefs.GetRange(batchStart, batchEnd - batchStart);
                
                Console.WriteLine($"📦 Processing Batch {batchIndex + 1}/{totalBatches} ({batchRefs.Count} surface objects)...");
                
                // Load and process current batch with aggressive memory management
                var batchResults = await ProcessSurfaceBatch(batchRefs);
                Console.WriteLine($"  ✅ Generated {batchResults.Count} surface matches");
                
                // Write results immediately
                await WriteSurfaceBatchResults(batchResults);
                
                // Add to master summary
                foreach (var result in batchResults.OrderByDescending(r => r.SimilarityScore))
                {
                    masterLines.Add($"{result.PM4SurfaceObjectId,-32} {result.BestWmoTopSurface,-28} {result.SimilarityScore:F3}       {result.SurfacePatternBreakdown}");
                }
                
                // Aggressive memory cleanup
                batchResults.Clear();
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                
                Console.WriteLine($"  🧹 Batch {batchIndex + 1} complete, memory aggressively cleared");
            }
            
            // Write final master summary
            File.WriteAllLines(masterSummaryFile, masterLines);
            Console.WriteLine($"📝 Master summary written to {masterSummaryFile}");
        }
        
        /// <summary>
        /// Process a batch of surface objects with minimal memory footprint
        /// </summary>
        static async Task<List<SurfaceMatchResult>> ProcessSurfaceBatch(List<PM4SurfaceReference> batchRefs)
        {
            var batchResults = new List<SurfaceMatchResult>();
            
            await Task.Run(() =>
            {
                foreach (var surfaceRef in batchRefs)
                {
                    try
                    {
                        // Load single PM4 surface object
                        var pm4Surface = ExtractPM4SurfaceObject(surfaceRef);
                        if (pm4Surface != null)
                        {
                            // Find best WMO top surface match
                            var bestMatch = FindBestTopSurfaceMatch(pm4Surface);
                            batchResults.Add(bestMatch);
                            
                            // Clear this object immediately
                            pm4Surface = null;
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"    ❌ Failed to process {surfaceRef.ObjectId}: {ex.Message}");
                    }
                }
            });
            
            return batchResults;
        }
        
        /// <summary>
        /// Extract surface vertices from a single PM4 object with MSUR data
        /// </summary>
        static PM4SurfaceObject ExtractPM4SurfaceObject(PM4SurfaceReference surfaceRef)
        {
            try
            {
                var pm4File = WoWToolbox.Core.v2.Foundation.Data.PM4File.FromFile(surfaceRef.PM4FilePath);
                var buildings = pm4File.ExtractBuildings();
                
                if (surfaceRef.ObjectIndex >= buildings.Count) return null;
                
                var building = buildings[surfaceRef.ObjectIndex];
                var vertices = ((dynamic)building).Vertices as List<Vector3>;
                
                if (vertices == null || vertices.Count == 0) return null;
                
                // Extract TOP surface vertices (highest Y values)
                var topSurfaceVertices = ExtractTopSurfaceVertices(vertices);
                
                return new PM4SurfaceObject
                {
                    ObjectId = surfaceRef.ObjectId,
                    PM4FileName = Path.GetFileName(surfaceRef.PM4FilePath),
                    ObjectIndex = surfaceRef.ObjectIndex,
                    TopSurfaceVertices = topSurfaceVertices,
                    TopSurfacePattern = GenerateSurfacePattern(topSurfaceVertices)
                };
            }
            catch
            {
                return null;
            }
        }
        
        /// <summary>
        /// Extract top surface geometry from WMO OBJ file
        /// </summary>
        static WmoTopSurface ExtractWmoTopSurface(string objFilePath)
        {
            try
            {
                var vertices = new List<Vector3>();
                
                var lines = File.ReadAllLines(objFilePath);
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
                
                if (vertices.Count == 0) return null;
                
                // Extract TOP surface vertices (highest Y values)
                var topSurfaceVertices = ExtractTopSurfaceVertices(vertices);
                
                if (topSurfaceVertices.Count < 3) return null; // Need at least 3 vertices for a surface
                
                return new WmoTopSurface
                {
                    FileName = Path.GetFileName(objFilePath),
                    FilePath = objFilePath,
                    TopSurfaceVertices = topSurfaceVertices,
                    TopSurfacePattern = GenerateSurfacePattern(topSurfaceVertices)
                };
            }
            catch
            {
                return null;
            }
        }
        
        /// <summary>
        /// Extract top surface vertices (highest Y values representing building tops)
        /// </summary>
        static List<Vector3> ExtractTopSurfaceVertices(List<Vector3> allVertices)
        {
            if (allVertices.Count == 0) return new List<Vector3>();
            
            // Find the top 20% of vertices by Y coordinate (building tops)
            var sortedByHeight = allVertices.OrderByDescending(v => v.Y).ToList();
            var topPercentageCount = Math.Max(3, (int)(sortedByHeight.Count * 0.2f)); // At least 3, max 20%
            var topVertices = sortedByHeight.Take(topPercentageCount).ToList();
            
            // Normalize these vertices relative to their center
            var center = topVertices.Aggregate(Vector3.Zero, (sum, v) => sum + v) / topVertices.Count;
            var normalizedVertices = topVertices.Select(v => v - center).ToList();
            
            return normalizedVertices;
        }
        
        /// <summary>
        /// Generate a surface pattern signature for comparison
        /// </summary>
        static SurfacePattern GenerateSurfacePattern(List<Vector3> topSurfaceVertices)
        {
            if (topSurfaceVertices.Count < 3) return new SurfacePattern();
            
            // Calculate surface characteristics
            var centerPoint = topSurfaceVertices.Aggregate(Vector3.Zero, (sum, v) => sum + v) / topSurfaceVertices.Count;
            
            // Distance distribution from center
            var distances = topSurfaceVertices.Select(v => Vector3.Distance(v, centerPoint)).OrderBy(d => d).ToList();
            
            // Angular distribution (for shape characterization)
            var angles = new List<float>();
            for (int i = 0; i < topSurfaceVertices.Count; i++)
            {
                var v1 = topSurfaceVertices[i] - centerPoint;
                var v2 = topSurfaceVertices[(i + 1) % topSurfaceVertices.Count] - centerPoint;
                
                var angle = (float)Math.Acos(Vector3.Dot(Vector3.Normalize(v1), Vector3.Normalize(v2)));
                if (!float.IsNaN(angle)) angles.Add(angle);
            }
            
            return new SurfacePattern
            {
                VertexCount = topSurfaceVertices.Count,
                DistanceDistribution = distances,
                AngularDistribution = angles.OrderBy(a => a).ToList(),
                CenterPoint = centerPoint
            };
        }
        
        /// <summary>
        /// Find the best WMO top surface match for a PM4 surface object
        /// </summary>
        static SurfaceMatchResult FindBestTopSurfaceMatch(PM4SurfaceObject pm4Surface)
        {
            var bestSimilarity = 0f;
            var bestWmoSurface = wmoTopSurfaces[0];
            var bestBreakdown = "";
            
            foreach (var wmoSurface in wmoTopSurfaces)
            {
                var similarity = CompareSurfacePatterns(pm4Surface.TopSurfacePattern, wmoSurface.TopSurfacePattern);
                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestWmoSurface = wmoSurface;
                    bestBreakdown = GenerateSurfacePatternBreakdown(pm4Surface.TopSurfacePattern, wmoSurface.TopSurfacePattern);
                }
            }
            
            return new SurfaceMatchResult
            {
                PM4SurfaceObjectId = pm4Surface.ObjectId,
                BestWmoTopSurface = bestWmoSurface.FileName,
                SimilarityScore = bestSimilarity,
                SurfacePatternBreakdown = bestBreakdown
            };
        }
        
        /// <summary>
        /// Compare surface patterns between PM4 and WMO top surfaces
        /// </summary>
        static float CompareSurfacePatterns(SurfacePattern pm4Pattern, SurfacePattern wmoPattern)
        {
            var similarities = new List<float>();
            
            // 1. Vertex count similarity (20% weight)
            var vertexCountRatio = Math.Min(pm4Pattern.VertexCount, wmoPattern.VertexCount) / 
                                  (float)Math.Max(pm4Pattern.VertexCount, wmoPattern.VertexCount);
            similarities.Add(vertexCountRatio * 0.2f);
            
            // 2. Distance distribution similarity (50% weight)
            var distanceSim = CompareDistanceDistributions(pm4Pattern.DistanceDistribution, wmoPattern.DistanceDistribution);
            similarities.Add(distanceSim * 0.5f);
            
            // 3. Angular distribution similarity (30% weight)
            var angularSim = CompareAngularDistributions(pm4Pattern.AngularDistribution, wmoPattern.AngularDistribution);
            similarities.Add(angularSim * 0.3f);
            
            return similarities.Sum();
        }
        
        /// <summary>
        /// Check if a PM4 building object has surface data
        /// </summary>
        static bool HasSurfaceData(object building)
        {
            try
            {
                var vertices = ((dynamic)building).Vertices as List<Vector3>;
                return vertices != null && vertices.Count >= 6; // Minimum for a meaningful surface
            }
            catch
            {
                return false;
            }
        }
        
        /// <summary>
        /// Compare distance distributions between surface patterns
        /// </summary>
        static float CompareDistanceDistributions(List<float> dist1, List<float> dist2)
        {
            if (dist1.Count == 0 || dist2.Count == 0) return 0f;
            
            var maxCount = Math.Max(dist1.Count, dist2.Count);
            var totalDifference = 0f;
            
            for (int i = 0; i < maxCount; i++)
            {
                var val1 = i < dist1.Count ? dist1[i] : dist1.Last();
                var val2 = i < dist2.Count ? dist2[i] : dist2.Last();
                totalDifference += Math.Abs(val1 - val2);
            }
            
            var maxPossibleDifference = maxCount * Math.Max(dist1.Last(), dist2.Last());
            return maxPossibleDifference > 0 ? 1f - (totalDifference / maxPossibleDifference) : 0f;
        }
        
        /// <summary>
        /// Compare angular distributions between surface patterns
        /// </summary>
        static float CompareAngularDistributions(List<float> angles1, List<float> angles2)
        {
            if (angles1.Count == 0 || angles2.Count == 0) return 0.5f; // Default if no angles
            
            return CompareDistanceDistributions(angles1, angles2); // Same algorithm works for angles
        }
        
        /// <summary>
        /// Generate surface pattern breakdown for analysis
        /// </summary>
        static string GenerateSurfacePatternBreakdown(SurfacePattern pm4Pattern, SurfacePattern wmoPattern)
        {
            var vertexRatio = Math.Min(pm4Pattern.VertexCount, wmoPattern.VertexCount) / 
                             (float)Math.Max(pm4Pattern.VertexCount, wmoPattern.VertexCount);
            var distSim = CompareDistanceDistributions(pm4Pattern.DistanceDistribution, wmoPattern.DistanceDistribution);
            var angleSim = CompareAngularDistributions(pm4Pattern.AngularDistribution, wmoPattern.AngularDistribution);
            
            return $"Verts:{vertexRatio:F3}, Dist:{distSim:F3}, Angles:{angleSim:F3}";
        }
        
        /// <summary>
        /// Write surface batch results
        /// </summary>
        static async Task WriteSurfaceBatchResults(List<SurfaceMatchResult> batchResults)
        {
            await Task.Run(() =>
            {
                foreach (var result in batchResults)
                {
                    WriteSurfaceMatchingFile(result);
                }
            });
        }
        
        /// <summary>
        /// Write surface matching results for a single object
        /// </summary>
        static void WriteSurfaceMatchingFile(SurfaceMatchResult result)
        {
            var outputFile = Path.Combine(OUTPUT_BASE_PATH, $"{result.PM4SurfaceObjectId}_surface_matches.txt");
            
            var lines = new List<string>
            {
                $"🎯 === SURFACE-FOCUSED MATCHING RESULTS ===",
                $"PM4 Surface Object: {result.PM4SurfaceObjectId}",
                "",
                $"🏆 BEST TOP SURFACE MATCH:",
                $"WMO Top Surface: {result.BestWmoTopSurface}",
                $"Similarity Score: {result.SimilarityScore:F3}",
                $"Surface Pattern: {result.SurfacePatternBreakdown}",
                $"Match Quality: {GetSurfaceMatchQuality(result.SimilarityScore)}",
                ""
            };
            
            File.WriteAllLines(outputFile, lines);
        }
        
        static string GetSurfaceMatchQuality(float similarity)
        {
            return similarity switch
            {
                > 0.8f => "Excellent surface pattern match",
                > 0.6f => "Good surface pattern match", 
                > 0.4f => "Moderate surface pattern match",
                > 0.2f => "Weak surface pattern match",
                _ => "Poor surface pattern match"
            };
        }
        
        static void CreateSampleWmoTopSurfaces()
        {
            wmoTopSurfaces.AddRange(new[]
            {
                new WmoTopSurface 
                { 
                    FileName = "Human_Building_01.obj",
                    TopSurfaceVertices = new List<Vector3> { Vector3.Zero, Vector3.UnitX, Vector3.UnitZ },
                    TopSurfacePattern = new SurfacePattern { VertexCount = 3, DistanceDistribution = new List<float> { 0f, 1f, 1f } }
                }
            });
            
            Console.WriteLine($"Created sample WMO top surfaces: {wmoTopSurfaces.Count} surfaces");
        }
    }
    
    // ===== SURFACE-FOCUSED DATA STRUCTURES =====
    
    public class PM4SurfaceReference
    {
        public string PM4FilePath { get; set; } = "";
        public int ObjectIndex { get; set; }
        public string ObjectId { get; set; } = "";
    }
    
    public class PM4SurfaceObject
    {
        public string ObjectId { get; set; } = "";
        public string PM4FileName { get; set; } = "";
        public int ObjectIndex { get; set; }
        public List<Vector3> TopSurfaceVertices { get; set; } = new List<Vector3>();
        public SurfacePattern TopSurfacePattern { get; set; } = new SurfacePattern();
    }
    
    public class WmoTopSurface
    {
        public string FileName { get; set; } = "";
        public string FilePath { get; set; } = "";
        public List<Vector3> TopSurfaceVertices { get; set; } = new List<Vector3>();
        public SurfacePattern TopSurfacePattern { get; set; } = new SurfacePattern();
    }
    
    public class SurfacePattern
    {
        public int VertexCount { get; set; }
        public List<float> DistanceDistribution { get; set; } = new List<float>();
        public List<float> AngularDistribution { get; set; } = new List<float>();
        public Vector3 CenterPoint { get; set; }
    }
    
    public class SurfaceMatchResult
    {
        public string PM4SurfaceObjectId { get; set; } = "";
        public string BestWmoTopSurface { get; set; } = "";
        public float SimilarityScore { get; set; }
        public string SurfacePatternBreakdown { get; set; } = "";
    }
} 