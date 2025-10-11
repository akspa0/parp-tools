using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Services.PM4;
using WoWToolbox.Core.v2.Foundation.Data;

namespace PM4Tool.Tests
{
    /// <summary>
    /// Demonstration of the new surface-oriented PM4 processing architecture.
    /// Shows how to use the Core.v2 services for advanced PM4 analysis.
    /// </summary>
    public class SurfaceOrientedDemo
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("🎯 === PM4 SURFACE-ORIENTED PROCESSING DEMO ===");
            Console.WriteLine("Demonstrating the new Core.v2 surface-oriented architecture");
            Console.WriteLine();

            // Demo file (using development_00_00 as it has MSUR data)
            var pm4FilePath = @"test_data\development\development_00_00.pm4";
            
            if (!File.Exists(pm4FilePath))
            {
                Console.WriteLine($"❌ Demo file not found: {pm4FilePath}");
                Console.WriteLine("   Please ensure the test data is available.");
                return;
            }

            try
            {
                // 1. Initialize the new surface-oriented services
                var buildingExtractionService = new PM4BuildingExtractionService();
                
                Console.WriteLine("🔍 Step 1: Detecting PM4 Format Version");
                var formatVersion = buildingExtractionService.DetectPM4FormatVersion(pm4FilePath);
                Console.WriteLine($"   Detected Format: {formatVersion}");
                Console.WriteLine();

                // 2. Extract surface-based navigation objects
                Console.WriteLine("🏗️  Step 2: Extracting Surface-Based Navigation Objects");
                var navObjects = buildingExtractionService.ExtractSurfaceBasedNavigationObjects(pm4FilePath);
                Console.WriteLine($"   Found {navObjects.Count} surface-based objects");
                Console.WriteLine();

                // 3. Analyze each object by surface orientation
                Console.WriteLine("📊 Step 3: Surface Orientation Analysis");
                foreach (var obj in navObjects.Take(5)) // Show first 5 objects
                {
                    Console.WriteLine($"   Object {obj.ObjectId} ({obj.EstimatedObjectType}):");
                    Console.WriteLine($"     - Top Surfaces: {obj.TopSurfaces.Count}");
                    Console.WriteLine($"     - Bottom Surfaces: {obj.BottomSurfaces.Count}");
                    Console.WriteLine($"     - Vertical Surfaces: {obj.VerticalSurfaces.Count}");
                    Console.WriteLine($"     - Total Vertices: {obj.TotalVertexCount:N0}");
                    Console.WriteLine($"     - Total Surface Area: {obj.TotalSurfaceArea:F1}");
                    
                    if (obj.ObjectBounds.HasValue)
                    {
                        var bounds = obj.ObjectBounds.Value;
                        Console.WriteLine($"     - Bounds: {bounds.Size.X:F1} × {bounds.Size.Y:F1} × {bounds.Size.Z:F1}");
                    }
                    Console.WriteLine();
                }

                // 4. Demonstrate individual surface analysis
                if (navObjects.Any())
                {
                    Console.WriteLine("🔬 Step 4: Individual Surface Analysis");
                    var firstObject = navObjects.First();
                    
                    // Analyze top surfaces
                    if (firstObject.TopSurfaces.Any())
                    {
                        Console.WriteLine("   Top Surfaces (Roofs/Upper Geometry):");
                        foreach (var surface in firstObject.TopSurfaces.Take(3))
                        {
                            Console.WriteLine($"     - Surface {surface.SurfaceIndex}: " +
                                            $"{surface.Vertices.Count} vertices, " +
                                            $"Area: {surface.SurfaceArea:F1}, " +
                                            $"Normal: ({surface.SurfaceNormal.X:F2}, {surface.SurfaceNormal.Y:F2}, {surface.SurfaceNormal.Z:F2})");
                        }
                        Console.WriteLine();
                    }

                    // Analyze bottom surfaces  
                    if (firstObject.BottomSurfaces.Any())
                    {
                        Console.WriteLine("   Bottom Surfaces (Foundations/Walkable Areas):");
                        foreach (var surface in firstObject.BottomSurfaces.Take(3))
                        {
                            Console.WriteLine($"     - Surface {surface.SurfaceIndex}: " +
                                            $"{surface.Vertices.Count} vertices, " +
                                            $"Area: {surface.SurfaceArea:F1}, " +
                                            $"Normal: ({surface.SurfaceNormal.X:F2}, {surface.SurfaceNormal.Y:F2}, {surface.SurfaceNormal.Z:F2})");
                        }
                        Console.WriteLine();
                    }
                }

                // 5. Show the advantage over blob-based matching
                Console.WriteLine("💡 Step 5: Blob vs Surface-Oriented Comparison");
                Console.WriteLine("   OLD APPROACH (Blob-based):");
                Console.WriteLine($"     - Would treat all {navObjects.Count} objects as single blobs");
                Console.WriteLine($"     - No surface orientation awareness");
                Console.WriteLine($"     - No separation of roofs vs foundations");
                Console.WriteLine();
                Console.WriteLine("   NEW APPROACH (Surface-oriented):");
                var totalTopSurfaces = navObjects.Sum(o => o.TopSurfaces.Count);
                var totalBottomSurfaces = navObjects.Sum(o => o.BottomSurfaces.Count);
                var totalVerticalSurfaces = navObjects.Sum(o => o.VerticalSurfaces.Count);
                
                Console.WriteLine($"     - {totalTopSurfaces} individual top surfaces for roof matching");
                Console.WriteLine($"     - {totalBottomSurfaces} individual bottom surfaces for foundation matching");
                Console.WriteLine($"     - {totalVerticalSurfaces} individual vertical surfaces for wall matching");
                Console.WriteLine($"     - Orientation-aware matching strategies");
                Console.WriteLine($"     - Surface normal compatibility analysis");
                Console.WriteLine();

                // 6. Demo the legacy format fallback
                Console.WriteLine("🔄 Step 6: Format Adaptation Demo");
                if (formatVersion == PM4BuildingExtractionService.PM4FormatVersion.NewerWithMDOS)
                {
                    Console.WriteLine("   ✅ Using MSUR surface extraction (newer format)");
                    Console.WriteLine("   ✅ Full surface orientation analysis available");
                    Console.WriteLine("   ✅ Precise geometric matching enabled");
                }
                else
                {
                    Console.WriteLine("   📋 Using legacy extraction fallback");
                    Console.WriteLine("   📋 Limited surface data available");
                    Console.WriteLine("   📋 Graceful degradation to MSLK-based extraction");
                }
                Console.WriteLine();

                Console.WriteLine("🎉 === DEMO COMPLETE ===");
                Console.WriteLine("This demonstrates the revolutionary surface-oriented approach:");
                Console.WriteLine("- Individual surface extraction and analysis");
                Console.WriteLine("- Orientation-aware matching capabilities");  
                Console.WriteLine("- Dual-format support with adaptive processing");
                Console.WriteLine("- Foundation for precise WMO matching");
                Console.WriteLine();
                Console.WriteLine("🚀 Ready for integration with WMO database and M2 models!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during demo: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
            }

            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
} 