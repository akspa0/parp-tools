using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Threading.Tasks;
using WoWToolbox.PM4Parsing;
using WoWToolbox.Core.v2.Foundation.Data;

namespace WmoMatchingDemo
{
    /// <summary>
    /// 🚀 HIGH-PERFORMANCE STREAMING PM4-WMO MATCHING
    /// One PM4 object vs streaming WMO files with simple vertex comparison
    /// </summary>
    class WmoMatchingDemo
    {
        private const string WMO_OBJ_BASE_PATH = @"I:\parp-tools\parp-tools\PM4Tool\test_data\wmo_335-objs\wmo\World\wmo\";
        private const string PM4_TEST_DATA_PATH = @"../test_data/original_development/development/";
        private const string OUTPUT_BASE_PATH = @"../output/streaming_vertex_matching/";
        
        private const int TOP_MATCHES_PER_OBJECT = 5; // Keep only top 5 matches per PM4 object
        
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== 🚀 HIGH-PERFORMANCE STREAMING PM4-WMO MATCHING ===");
            Console.WriteLine("📊 APPROACH: One PM4 object vs streaming WMO files");
            Console.WriteLine("⚡ PERFORMANCE: Simple vertex-only comparison");
            Console.WriteLine();
            
            var startTime = DateTime.Now;
            
            try
            {
                // Step 1: Get WMO file list (no pre-loading)
                Console.WriteLine("Step 1: Scanning WMO files...");
                var wmoFiles = GetWmoFileList();
                Console.WriteLine($"✅ Found {wmoFiles.Count} WMO OBJ files");
                Console.WriteLine();
                
                // Step 2: Get PM4 object references
                Console.WriteLine("Step 2: Scanning PM4 files...");
                var pm4ObjectRefs = await GetPM4ObjectReferences();
                Console.WriteLine($"✅ Found {pm4ObjectRefs.Count} PM4 objects");
                Console.WriteLine();
                
                // Step 3: Stream processing - one PM4 object at a time
                Console.WriteLine("Step 3: STREAMING VERTEX MATCHING...");
                await ProcessPM4ObjectsStreaming(pm4ObjectRefs, wmoFiles);
                
                var elapsed = DateTime.Now - startTime;
                Console.WriteLine();
                Console.WriteLine("🎉 === STREAMING MATCHING COMPLETE ===");
                Console.WriteLine($"⏱️ Total Time: {elapsed.TotalMinutes:F1} minutes");
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
        /// Get list of WMO files without loading them
        /// </summary>
        static List<string> GetWmoFileList()
        {
            if (!Directory.Exists(WMO_OBJ_BASE_PATH))
            {
                Console.WriteLine($"⚠️ WMO path not found: {WMO_OBJ_BASE_PATH}");
                return new List<string>();
            }
            
            return Directory.GetFiles(WMO_OBJ_BASE_PATH, "*.obj", SearchOption.AllDirectories).ToList();
        }
        
        /// <summary>
        /// Get PM4 object references without loading vertex data
        /// </summary>
        static async Task<List<PM4ObjectReference>> GetPM4ObjectReferences()
        {
            var objectRefs = new List<PM4ObjectReference>();
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
                            objectRefs.Add(new PM4ObjectReference
                            {
                                PM4FilePath = pm4File,
                                ObjectIndex = i,
                                ObjectId = $"{Path.GetFileNameWithoutExtension(pm4File)}_OBJ_{i:D3}"
                            });
                        }
                        
                        if (buildings.Count > 0)
                        {
                            Console.WriteLine($"  {Path.GetFileNameWithoutExtension(pm4File)}: {buildings.Count} objects");
                        }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"    ❌ Failed to scan {pm4File}: {ex.Message}");
                    }
                }
            });
            
            return objectRefs;
        }
        
        /// <summary>
        /// Process PM4 objects one at a time with streaming WMO comparison
        /// </summary>
        static async Task ProcessPM4ObjectsStreaming(List<PM4ObjectReference> pm4ObjectRefs, List<string> wmoFiles)
        {
            Console.WriteLine($"🔄 Processing {pm4ObjectRefs.Count} PM4 objects vs {wmoFiles.Count} WMO files");
            
            Directory.CreateDirectory(OUTPUT_BASE_PATH);
            var allResults = new List<StreamingMatchResult>();
            
            var processed = 0;
            foreach (var pm4Ref in pm4ObjectRefs)
            {
                processed++;
                Console.WriteLine($"📦 Processing {processed}/{pm4ObjectRefs.Count}: {pm4Ref.ObjectId}");
                
                // Load single PM4 object
                var pm4Object = LoadSinglePM4Object(pm4Ref);
                if (pm4Object == null)
                {
                    Console.WriteLine($"  ❌ Failed to load {pm4Ref.ObjectId}");
                    continue;
                }
                
                // Stream through all WMO files and find top matches
                var topMatches = await FindTopMatchesStreaming(pm4Object, wmoFiles);
                
                if (topMatches.Any())
                {
                    allResults.AddRange(topMatches);
                    Console.WriteLine($"  ✅ Found {topMatches.Count} matches, best: {topMatches.First().Similarity:F3}");
                }
                else
                {
                    Console.WriteLine($"  ⚠️ No matches found");
                }
                
                // Clear this object immediately
                pm4Object = null;
                GC.Collect();
                
                // Progress update every 50 objects
                if (processed % 50 == 0)
                {
                    Console.WriteLine($"  📊 Progress: {processed}/{pm4ObjectRefs.Count} ({processed * 100.0 / pm4ObjectRefs.Count:F1}%)");
                }
            }
            
            // Write consolidated results
            await WriteStreamingResults(allResults);
            Console.WriteLine($"📝 Written {allResults.Count} total matches");
        }
        
        /// <summary>
        /// Load a single PM4 object with normalized vertices
        /// </summary>
        static SimplePM4Object LoadSinglePM4Object(PM4ObjectReference pm4Ref)
        {
            try
            {
                var pm4File = WoWToolbox.Core.v2.Foundation.Data.PM4File.FromFile(pm4Ref.PM4FilePath);
                var buildings = pm4File.ExtractBuildings();
                
                if (pm4Ref.ObjectIndex >= buildings.Count) return null;
                
                var building = buildings[pm4Ref.ObjectIndex];
                var vertices = ((dynamic)building).Vertices as List<Vector3>;
                
                if (vertices == null || vertices.Count == 0) return null;
                
                // Simple normalization: translate to origin and scale to unit size
                var normalizedVertices = NormalizeVerticesSimple(vertices);
                
                return new SimplePM4Object
                {
                    ObjectId = pm4Ref.ObjectId,
                    VertexCount = vertices.Count,
                    NormalizedVertices = normalizedVertices
                };
            }
            catch
            {
                return null;
            }
        }
        
        /// <summary>
        /// Find top matches for PM4 object by streaming through WMO files
        /// </summary>
        static async Task<List<StreamingMatchResult>> FindTopMatchesStreaming(SimplePM4Object pm4Object, List<string> wmoFiles)
        {
            var topMatches = new List<StreamingMatchResult>();
            
            await Task.Run(() =>
            {
                foreach (var wmoFile in wmoFiles)
                {
                    try
                    {
                        // Load and normalize WMO vertices on the fly
                        var wmoVertices = LoadWmoVerticesSimple(wmoFile);
                        if (wmoVertices.Count == 0) continue;
                        
                        var normalizedWmoVertices = NormalizeVerticesSimple(wmoVertices);
                        
                        // Simple vertex similarity calculation
                        var similarity = CalculateSimpleVertexSimilarity(pm4Object.NormalizedVertices, normalizedWmoVertices);
                        
                        // Keep only if it's good enough and we have room, or it's better than our worst
                        if (similarity > 0.1f) // Minimum threshold
                        {
                            var match = new StreamingMatchResult
                            {
                                PM4ObjectId = pm4Object.ObjectId,
                                WmoFileName = Path.GetFileName(wmoFile),
                                Similarity = similarity,
                                PM4VertexCount = pm4Object.VertexCount,
                                WmoVertexCount = wmoVertices.Count
                            };
                            
                            topMatches.Add(match);
                        }
                        
                        // Clear WMO data immediately
                        wmoVertices.Clear();
                        normalizedWmoVertices.Clear();
                    }
                    catch
                    {
                        // Skip problematic WMO files
                    }
                }
                
                // Keep only top matches
                topMatches = topMatches.OrderByDescending(m => m.Similarity)
                                      .Take(TOP_MATCHES_PER_OBJECT)
                                      .ToList();
            });
            
            return topMatches;
        }
        
        /// <summary>
        /// Load WMO vertices simple and fast
        /// </summary>
        static List<Vector3> LoadWmoVerticesSimple(string objFilePath)
        {
            var vertices = new List<Vector3>();
            
            try
            {
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
            }
            catch
            {
                // Return empty list for problematic files
            }
            
            return vertices;
        }
        
        /// <summary>
        /// Simple and fast vertex normalization
        /// </summary>
        static List<Vector3> NormalizeVerticesSimple(List<Vector3> vertices)
        {
            if (vertices.Count == 0) return new List<Vector3>();
            
            // 1. Find center
            var center = Vector3.Zero;
            foreach (var v in vertices) center += v;
            center /= vertices.Count;
            
            // 2. Translate to origin
            var centered = vertices.Select(v => v - center).ToList();
            
            // 3. Find max distance from center
            var maxDistance = centered.Max(v => v.Length());
            if (maxDistance == 0) return centered;
            
            // 4. Scale to unit size
            var scale = 1.0f / maxDistance;
            return centered.Select(v => v * scale).ToList();
        }
        
        /// <summary>
        /// Simple vertex similarity calculation - fast and effective
        /// </summary>
        static float CalculateSimpleVertexSimilarity(List<Vector3> vertices1, List<Vector3> vertices2)
        {
            if (vertices1.Count == 0 || vertices2.Count == 0) return 0f;
            
            // 1. Vertex count similarity (30% weight)
            var countRatio = Math.Min(vertices1.Count, vertices2.Count) / (float)Math.Max(vertices1.Count, vertices2.Count);
            var countSimilarity = countRatio * 0.3f;
            
            // 2. Distance distribution similarity (70% weight)
            var sample1 = SampleVertices(vertices1, 50); // Sample for performance
            var sample2 = SampleVertices(vertices2, 50);
            
            var distances1 = sample1.Select(v => v.Length()).OrderBy(d => d).ToArray();
            var distances2 = sample2.Select(v => v.Length()).OrderBy(d => d).ToArray();
            
            var distanceSimilarity = CompareDistributionsSimple(distances1, distances2) * 0.7f;
            
            return countSimilarity + distanceSimilarity;
        }
        
        /// <summary>
        /// Sample vertices for performance
        /// </summary>
        static List<Vector3> SampleVertices(List<Vector3> vertices, int maxSamples)
        {
            if (vertices.Count <= maxSamples) return vertices;
            
            var step = vertices.Count / maxSamples;
            var sampled = new List<Vector3>();
            for (int i = 0; i < vertices.Count; i += step)
            {
                sampled.Add(vertices[i]);
                if (sampled.Count >= maxSamples) break;
            }
            return sampled;
        }
        
        /// <summary>
        /// Simple and fast distribution comparison
        /// </summary>
        static float CompareDistributionsSimple(float[] dist1, float[] dist2)
        {
            var minLength = Math.Min(dist1.Length, dist2.Length);
            if (minLength == 0) return 0f;
            
            var totalDifference = 0f;
            for (int i = 0; i < minLength; i++)
            {
                totalDifference += Math.Abs(dist1[i] - dist2[i]);
            }
            
            var avgRange = (dist1.Max() + dist2.Max()) / 2f;
            if (avgRange == 0) return 1f;
            
            return Math.Max(0f, 1f - (totalDifference / (minLength * avgRange)));
        }
        
        /// <summary>
        /// Write streaming results to consolidated files
        /// </summary>
        static async Task WriteStreamingResults(List<StreamingMatchResult> allResults)
        {
            await Task.Run(() =>
            {
                WriteMasterStreamingSummary(allResults);
                WriteQualityStreamingSummaries(allResults);
                WriteStreamingStatistics(allResults);
            });
        }
        
        /// <summary>
        /// Write master streaming summary
        /// </summary>
        static void WriteMasterStreamingSummary(List<StreamingMatchResult> allResults)
        {
            var outputFile = Path.Combine(OUTPUT_BASE_PATH, "MASTER_STREAMING_MATCHES.txt");
            var lines = new List<string>
            {
                "🚀 === STREAMING VERTEX MATCHING RESULTS ===",
                $"Analysis Date: {DateTime.Now}",
                $"Total Matches: {allResults.Count}",
                $"Top Matches Per Object: {TOP_MATCHES_PER_OBJECT}",
                "",
                "PM4_OBJECT                        WMO_MATCH                    SIMILARITY  PM4_VERTS  WMO_VERTS",
                "============================================================================================",
                ""
            };
            
            foreach (var result in allResults.OrderByDescending(r => r.Similarity))
            {
                lines.Add($"{result.PM4ObjectId,-32} {result.WmoFileName,-28} {result.Similarity:F3}       {result.PM4VertexCount,-9} {result.WmoVertexCount}");
            }
            
            File.WriteAllLines(outputFile, lines);
        }
        
        /// <summary>
        /// Write quality-based streaming summaries
        /// </summary>
        static void WriteQualityStreamingSummaries(List<StreamingMatchResult> allResults)
        {
            var excellent = allResults.Where(r => r.Similarity > 0.6f).ToList();
            var good = allResults.Where(r => r.Similarity > 0.4f && r.Similarity <= 0.6f).ToList();
            var moderate = allResults.Where(r => r.Similarity > 0.2f && r.Similarity <= 0.4f).ToList();
            
            WriteStreamingQualityFile(excellent, "EXCELLENT_STREAMING_MATCHES.txt", "🏆 EXCELLENT MATCHES (>0.6)");
            WriteStreamingQualityFile(good, "GOOD_STREAMING_MATCHES.txt", "✅ GOOD MATCHES (0.4-0.6)");
            WriteStreamingQualityFile(moderate, "MODERATE_STREAMING_MATCHES.txt", "🔶 MODERATE MATCHES (0.2-0.4)");
        }
        
        /// <summary>
        /// Write streaming quality file
        /// </summary>
        static void WriteStreamingQualityFile(List<StreamingMatchResult> results, string fileName, string title)
        {
            if (results.Count == 0) return;
            
            var outputFile = Path.Combine(OUTPUT_BASE_PATH, fileName);
            var lines = new List<string>
            {
                title,
                $"Found {results.Count} matches in this quality tier",
                "",
                "PM4_OBJECT                        WMO_MATCH                    SIMILARITY  PM4_VERTS  WMO_VERTS",
                "============================================================================================",
                ""
            };
            
            foreach (var result in results.OrderByDescending(r => r.Similarity))
            {
                lines.Add($"{result.PM4ObjectId,-32} {result.WmoFileName,-28} {result.Similarity:F3}       {result.PM4VertexCount,-9} {result.WmoVertexCount}");
            }
            
            File.WriteAllLines(outputFile, lines);
        }
        
        /// <summary>
        /// Write streaming statistics
        /// </summary>
        static void WriteStreamingStatistics(List<StreamingMatchResult> allResults)
        {
            var outputFile = Path.Combine(OUTPUT_BASE_PATH, "STREAMING_STATISTICS.txt");
            
            var excellent = allResults.Count(r => r.Similarity > 0.6f);
            var good = allResults.Count(r => r.Similarity > 0.4f && r.Similarity <= 0.6f);
            var moderate = allResults.Count(r => r.Similarity > 0.2f && r.Similarity <= 0.4f);
            var poor = allResults.Count(r => r.Similarity <= 0.2f);
            
            var lines = new List<string>
            {
                "📊 === STREAMING MATCHING STATISTICS ===",
                $"Analysis Date: {DateTime.Now}",
                "",
                "=== PERFORMANCE ===",
                $"Approach: One PM4 object vs streaming WMO files",
                $"Top matches kept per object: {TOP_MATCHES_PER_OBJECT}",
                $"Total matches generated: {allResults.Count}",
                "",
                "=== QUALITY DISTRIBUTION ===",
                $"🏆 Excellent (>0.6):     {excellent,6} ({excellent * 100.0 / allResults.Count:F1}%)",
                $"✅ Good (0.4-0.6):       {good,6} ({good * 100.0 / allResults.Count:F1}%)",
                $"🔶 Moderate (0.2-0.4):   {moderate,6} ({moderate * 100.0 / allResults.Count:F1}%)",
                $"❌ Poor (<0.2):          {poor,6} ({poor * 100.0 / allResults.Count:F1}%)",
                "",
                "=== SIMILARITY STATS ===",
                $"Average Similarity: {allResults.Average(r => r.Similarity):F3}",
                $"Best Match: {allResults.Max(r => r.Similarity):F3}",
                $"Worst Match: {allResults.Min(r => r.Similarity):F3}",
                "",
                "=== RECOMMENDATIONS ===",
                $"• Focus on {excellent + good} high-quality matches first",
                $"• System is optimized for speed and memory efficiency",
                $"• Each PM4 object processed independently for maximum performance"
            };
            
            File.WriteAllLines(outputFile, lines);
        }
    }
    
    // ===== SIMPLE STREAMING DATA STRUCTURES =====
    
    /// <summary>
    /// Reference to PM4 object for streaming
    /// </summary>
    public class PM4ObjectReference
    {
        public string PM4FilePath { get; set; } = "";
        public int ObjectIndex { get; set; }
        public string ObjectId { get; set; } = "";
    }
    
    /// <summary>
    /// Simple PM4 object for streaming processing
    /// </summary>
    public class SimplePM4Object
    {
        public string ObjectId { get; set; } = "";
        public int VertexCount { get; set; }
        public List<Vector3> NormalizedVertices { get; set; } = new List<Vector3>();
    }
    
    /// <summary>
    /// Streaming match result
    /// </summary>
    public class StreamingMatchResult
    {
        public string PM4ObjectId { get; set; } = "";
        public string WmoFileName { get; set; } = "";
        public float Similarity { get; set; }
        public int PM4VertexCount { get; set; }
        public int WmoVertexCount { get; set; }
    }
} 