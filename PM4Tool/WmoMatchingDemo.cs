using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace PM4Tool
{
    /// <summary>
    /// WMO Matching Demonstration - Key Algorithmic Concepts
    /// Shows how to perform geometric correlation between PM4 objects and WMO geometry
    /// </summary>
    class WmoMatchingDemo
    {
        private const string WMO_OBJ_BASE_PATH = @"I:\parp-tools\parp-tools\PM4Tool\test_data\wmo_335-objs\wmo\World\wmo\";
        
        static void Main(string[] args)
        {
            Console.WriteLine("=== Enhanced PM4 → WMO Geometric Matching Demo ===");
            Console.WriteLine("Demonstrates advanced spatial correlation algorithms\n");

            try
            {
                // Step 1: Verify WMO OBJ data exists
                if (!Directory.Exists(WMO_OBJ_BASE_PATH))
                {
                    Console.WriteLine($"❌ WMO OBJ data not found at: {WMO_OBJ_BASE_PATH}");
                    Console.WriteLine("Please ensure the pre-converted WMO OBJ files are available");
                    return;
                }

                Console.WriteLine($"✅ Found WMO OBJ data at: {WMO_OBJ_BASE_PATH}");
                
                // Step 2: List available WMO OBJ files
                string buildingsPath = Path.Combine(WMO_OBJ_BASE_PATH, "buildings");
                if (Directory.Exists(buildingsPath))
                {
                    var objFiles = Directory.GetFiles(buildingsPath, "*.obj");
                    Console.WriteLine($"✅ Found {objFiles.Length} WMO OBJ files in buildings directory");
                    
                    // Show first few files as examples
                    var sampleFiles = objFiles.Take(5);
                    foreach (var file in sampleFiles)
                    {
                        var fileName = Path.GetFileName(file);
                        var fileInfo = new FileInfo(file);
                        Console.WriteLine($"   📄 {fileName} ({fileInfo.Length:N0} bytes)");
                    }
                    
                    if (objFiles.Length > 5)
                    {
                        Console.WriteLine($"   ... and {objFiles.Length - 5} more files");
                    }
                }

                // Step 3: Demonstrate coordinate system conversion
                Console.WriteLine("\n🔄 Coordinate System Conversion:");
                DemonstrateCoordinateConversion();

                // Step 4: Show matching algorithm concepts
                Console.WriteLine("\n🎯 Matching Algorithm Concepts:");
                DemonstrateMatchingAlgorithms();

                // Step 5: Analyze a sample WMO OBJ file
                Console.WriteLine("\n📊 Sample WMO Analysis:");
                AnalyzeSampleWmoFile();

                Console.WriteLine("\n✅ Demo completed successfully!");
                Console.WriteLine("\nNext steps:");
                Console.WriteLine("1. Integrate PM4 object extraction");
                Console.WriteLine("2. Implement geometric comparison");
                Console.WriteLine("3. Add confidence scoring");
                Console.WriteLine("4. Generate matching reports");

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }

        static void DemonstrateCoordinateConversion()
        {
            // WMO OBJ coordinate system (+Z up) vs PM4 coordinate system (Y up)
            var wmoVertex = new Vector3(100.0f, 50.0f, 200.0f); // WMO: X, Y, Z(up)
            var pm4Vertex = ConvertWmoToPm4Coordinates(wmoVertex);
            
            Console.WriteLine($"   WMO Coordinates: X={wmoVertex.X:F1}, Y={wmoVertex.Y:F1}, Z={wmoVertex.Z:F1}");
            Console.WriteLine($"   PM4 Coordinates: X={pm4Vertex.X:F1}, Y={pm4Vertex.Y:F1}, Z={pm4Vertex.Z:F1}");
            Console.WriteLine("   ✅ Coordinate conversion applied (Z->Y axis swap)");
        }

        static Vector3 ConvertWmoToPm4Coordinates(Vector3 wmoVertex)
        {
            // Convert from WMO OBJ coordinate system (+Z up) to PM4 system (Y up)
            return new Vector3(
                wmoVertex.X,   // X remains the same
                wmoVertex.Z,   // Z becomes Y (up axis)
                -wmoVertex.Y   // Y becomes -Z (depth axis)
            );
        }

        static void DemonstrateMatchingAlgorithms()
        {
            Console.WriteLine("   🔍 Vertex Count Similarity:");
            DemonstrateVertexCountSimilarity(1500, 1520);  // Very similar
            DemonstrateVertexCountSimilarity(1500, 2000);  // Different
            
            Console.WriteLine("\n   📏 Scale Analysis:");
            DemonstrateScaleAnalysis();
            
            Console.WriteLine("\n   🎯 Confidence Scoring:");
            DemonstrateConfidenceScoring();
        }

        static void DemonstrateVertexCountSimilarity(int count1, int count2)
        {
            float ratio = Math.Min(count1, count2) / (float)Math.Max(count1, count2);
            float similarity = ratio * 100f;
            Console.WriteLine($"      {count1} vs {count2} vertices → {similarity:F1}% similarity");
        }

        static void DemonstrateScaleAnalysis()
        {
            var bbox1 = new BoundingBox3D(new Vector3(-50, 0, -30), new Vector3(50, 80, 30));
            var bbox2 = new BoundingBox3D(new Vector3(-48, 0, -28), new Vector3(52, 82, 32));
            
            var scale1 = bbox1.GetScale();
            var scale2 = bbox2.GetScale();
            
            Console.WriteLine($"      Object 1 scale: ({scale1.X:F1}, {scale1.Y:F1}, {scale1.Z:F1})");
            Console.WriteLine($"      Object 2 scale: ({scale2.X:F1}, {scale2.Y:F1}, {scale2.Z:F1})");
            
            float scaleConsistency = CalculateScaleConsistency(scale1, scale2);
            Console.WriteLine($"      Scale consistency: {scaleConsistency:F1}%");
        }

        static float CalculateScaleConsistency(Vector3 scale1, Vector3 scale2)
        {
            var ratios = new[]
            {
                Math.Min(scale1.X, scale2.X) / Math.Max(scale1.X, scale2.X),
                Math.Min(scale1.Y, scale2.Y) / Math.Max(scale1.Y, scale2.Y),
                Math.Min(scale1.Z, scale2.Z) / Math.Max(scale1.Z, scale2.Z)
            };
            
            return ratios.Average() * 100f;
        }

        static void DemonstrateConfidenceScoring()
        {
            // Simulate matching scores for different factors
            var scaleConsistency = 0.92f;      // 92% - very good
            var relativeSizeSimilarity = 0.85f; // 85% - good
            var vertexCountSimilarity = 0.78f;  // 78% - decent
            var geometricComplexity = 0.71f;    // 71% - moderate
            
            // Weighted confidence calculation
            var confidence = (scaleConsistency * 0.40f) +      // 40% weight
                           (relativeSizeSimilarity * 0.30f) +  // 30% weight
                           (vertexCountSimilarity * 0.20f) +   // 20% weight
                           (geometricComplexity * 0.10f);      // 10% weight
            
            Console.WriteLine($"      Scale Consistency:    {scaleConsistency:P1} (40% weight)");
            Console.WriteLine($"      Size Similarity:      {relativeSizeSimilarity:P1} (30% weight)");
            Console.WriteLine($"      Vertex Count Match:   {vertexCountSimilarity:P1} (20% weight)");
            Console.WriteLine($"      Geometric Complexity: {geometricComplexity:P1} (10% weight)");
            Console.WriteLine($"      → Overall Confidence: {confidence:P1}");
        }

        static void AnalyzeSampleWmoFile()
        {
            string buildingsPath = Path.Combine(WMO_OBJ_BASE_PATH, "buildings");
            if (!Directory.Exists(buildingsPath))
            {
                Console.WriteLine("   ❌ Buildings directory not found");
                return;
            }

            var objFiles = Directory.GetFiles(buildingsPath, "*.obj");
            if (objFiles.Length == 0)
            {
                Console.WriteLine("   ❌ No OBJ files found");
                return;
            }

            // Analyze the first available file
            string sampleFile = objFiles[0];
            string fileName = Path.GetFileName(sampleFile);
            
            try
            {
                var vertices = new List<Vector3>();
                var lines = File.ReadAllLines(sampleFile);
                
                foreach (var line in lines.Take(1000)) // Limit for demo
                {
                    if (line.StartsWith("v "))
                    {
                        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length >= 4)
                        {
                            if (float.TryParse(parts[1], out float x) &&
                                float.TryParse(parts[2], out float y) &&
                                float.TryParse(parts[3], out float z))
                            {
                                vertices.Add(new Vector3(x, y, z));
                            }
                        }
                    }
                }

                if (vertices.Count > 0)
                {
                    var bbox = CalculateBoundingBox(vertices);
                    var scale = bbox.GetScale();
                    
                    Console.WriteLine($"   📄 File: {fileName}");
                    Console.WriteLine($"   📊 Analyzed {vertices.Count} vertices");
                    Console.WriteLine($"   📦 Bounding Box: {bbox.Min:F1} to {bbox.Max:F1}");
                    Console.WriteLine($"   📏 Scale: ({scale.X:F1}, {scale.Y:F1}, {scale.Z:F1})");
                    Console.WriteLine($"   🎯 Ready for PM4 object comparison");
                }
                else
                {
                    Console.WriteLine($"   ❌ No vertices found in {fileName}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"   ❌ Error analyzing {fileName}: {ex.Message}");
            }
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
    /// Simple 3D bounding box for geometric analysis
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
} 