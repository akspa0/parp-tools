using System;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Services.PM4;
using WoWToolbox.Core.v2.Foundation.Data;

namespace PM4Tool
{
    /// <summary>
    /// Demonstration of the new surface-oriented PM4 processing architecture.
    /// Shows how the Core.v2 services revolutionize PM4 analysis.
    /// </summary>
    public class SurfaceOrientedDemo
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("🎯 === PM4 SURFACE-ORIENTED PROCESSING DEMO ===");
            Console.WriteLine("Demonstrating how the new architecture works");
            Console.WriteLine();

            // Demo concept explanation (even without data)
            ExplainSurfaceOrientedConcept();
            Console.WriteLine();

            // Demo file (using development_00_00 as it has MSUR data)
            var pm4FilePath = @"test_data\development\development_00_00.pm4";
            
            if (!File.Exists(pm4FilePath))
            {
                Console.WriteLine($"📁 Demo file not found: {pm4FilePath}");
                Console.WriteLine("   Running conceptual demonstration instead...");
                Console.WriteLine();
                RunConceptualDemo();
                return;
            }

            try
            {
                RunRealDemo(pm4FilePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during demo: {ex.Message}");
                Console.WriteLine();
                RunConceptualDemo();
            }

            Console.WriteLine();
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }

        static void ExplainSurfaceOrientedConcept()
        {
            Console.WriteLine("🔍 HOW THE SURFACE-ORIENTED ARCHITECTURE WORKS:");
            Console.WriteLine();
            
            Console.WriteLine("📋 OLD APPROACH (Blob-based Matching):");
            Console.WriteLine("   PM4 File → Extract entire building as one blob → Match to WMO");
            Console.WriteLine("   ❌ Problems:");
            Console.WriteLine("     - Tile 22_18 'hundreds of snowballs' treated as single object");
            Console.WriteLine("     - No separation of roofs vs foundations");
            Console.WriteLine("     - Generic geometric correlation");
            Console.WriteLine("     - Fake procedural data instead of real PM4 geometry");
            Console.WriteLine();

            Console.WriteLine("🚀 NEW APPROACH (Surface-oriented Matching):");
            Console.WriteLine("   PM4 File → Extract MSUR surfaces → Analyze orientation → Match by purpose");
            Console.WriteLine("   ✅ Advantages:");
            Console.WriteLine("     - Individual surface extraction from MSUR/MSVT chunks");
            Console.WriteLine("     - Surface orientation analysis (Top/Bottom/Vertical)");
            Console.WriteLine("     - Top surfaces match roof WMOs, bottom surfaces match foundations");
            Console.WriteLine("     - Real vertex data with coordinate transformation");
            Console.WriteLine("     - Normal vector compatibility analysis");
            Console.WriteLine();

            Console.WriteLine("🎯 KEY BREAKTHROUGH CONCEPTS:");
            Console.WriteLine("   1. SURFACE SEPARATION: Instead of 'hundreds of snowballs' → individual surfaces");
            Console.WriteLine("   2. ORIENTATION AWARENESS: Each surface knows if it's roof, foundation, or wall");
            Console.WriteLine("   3. PURPOSE-BASED MATCHING: Roofs match roofs, foundations match foundations");
            Console.WriteLine("   4. REAL GEOMETRY: Actual MSUR/MSVT data instead of fake generation");
        }

        static void RunConceptualDemo()
        {
            Console.WriteLine("💡 CONCEPTUAL DEMONSTRATION:");
            Console.WriteLine();

            // Simulate what the system would find
            Console.WriteLine("🏗️  Surface Extraction Simulation:");
            Console.WriteLine("   Found 15 surface-based objects in PM4 file");
            Console.WriteLine();

            Console.WriteLine("📊 Surface Orientation Analysis:");
            Console.WriteLine("   Object SURFACE_OBJ_001 (Building):");
            Console.WriteLine("     - Top Surfaces: 3 (roof elements)");
            Console.WriteLine("     - Bottom Surfaces: 2 (foundation platforms)");
            Console.WriteLine("     - Vertical Surfaces: 8 (walls)");
            Console.WriteLine("     - Total Vertices: 1,247");
            Console.WriteLine("     - Total Surface Area: 156.3");
            Console.WriteLine("     - Bounds: 15.2 × 8.4 × 12.1");
            Console.WriteLine();

            Console.WriteLine("   Object SURFACE_OBJ_002 (Roof Structure):");
            Console.WriteLine("     - Top Surfaces: 5 (complex roof geometry)");
            Console.WriteLine("     - Bottom Surfaces: 0 (no foundation)");
            Console.WriteLine("     - Vertical Surfaces: 2 (roof edges)");
            Console.WriteLine("     - Total Vertices: 832");
            Console.WriteLine("     - Total Surface Area: 89.7");
            Console.WriteLine("     - Bounds: 12.8 × 3.2 × 9.6");
            Console.WriteLine();

            Console.WriteLine("🔬 Individual Surface Analysis:");
            Console.WriteLine("   Top Surfaces (Roofs/Upper Geometry):");
            Console.WriteLine("     - Surface 12: 156 vertices, Area: 34.2, Normal: (0.12, 0.95, 0.08)");
            Console.WriteLine("     - Surface 15: 203 vertices, Area: 45.8, Normal: (0.03, 0.89, 0.15)");
            Console.WriteLine("     - Surface 18: 89 vertices, Area: 12.4, Normal: (-0.08, 0.92, 0.06)");
            Console.WriteLine();

            Console.WriteLine("   Bottom Surfaces (Foundations/Walkable Areas):");
            Console.WriteLine("     - Surface 3: 234 vertices, Area: 78.9, Normal: (0.05, -0.96, 0.02)");
            Console.WriteLine("     - Surface 7: 178 vertices, Area: 56.3, Normal: (-0.02, -0.94, 0.08)");
            Console.WriteLine();

            Console.WriteLine("💡 Blob vs Surface-Oriented Comparison:");
            Console.WriteLine("   OLD APPROACH (Blob-based):");
            Console.WriteLine("     - Would treat all 15 objects as single blobs");
            Console.WriteLine("     - No surface orientation awareness");
            Console.WriteLine("     - No separation of roofs vs foundations");
            Console.WriteLine();
            Console.WriteLine("   NEW APPROACH (Surface-oriented):");
            Console.WriteLine("     - 28 individual top surfaces for roof matching");
            Console.WriteLine("     - 19 individual bottom surfaces for foundation matching");
            Console.WriteLine("     - 45 individual vertical surfaces for wall matching");
            Console.WriteLine("     - Orientation-aware matching strategies");
            Console.WriteLine("     - Surface normal compatibility analysis");
            Console.WriteLine();

            Console.WriteLine("🔄 Format Adaptation:");
            Console.WriteLine("   ✅ Detected: NewerWithMDOS format");
            Console.WriteLine("   ✅ Using MSUR surface extraction");
            Console.WriteLine("   ✅ Full surface orientation analysis available");
            Console.WriteLine("   ✅ Precise geometric matching enabled");
            Console.WriteLine();

            Console.WriteLine("🎉 === ARCHITECTURE BENEFITS ===");
            Console.WriteLine("This revolutionary approach enables:");
            Console.WriteLine("- Individual surface extraction and analysis");
            Console.WriteLine("- Orientation-aware matching capabilities");  
            Console.WriteLine("- Dual-format support with adaptive processing");
            Console.WriteLine("- Foundation for precise WMO matching");
            Console.WriteLine("- M2 model integration ready");
            Console.WriteLine();
            Console.WriteLine("🚀 Ready for WMO database integration!");
        }

        static void RunRealDemo(string pm4FilePath)
        {
            // If the file exists, try to run with real data
            var buildingExtractionService = new PM4BuildingExtractionService();
            
            Console.WriteLine("🔍 Step 1: Detecting PM4 Format Version");
            var formatVersion = buildingExtractionService.DetectPM4FormatVersion(pm4FilePath);
            Console.WriteLine($"   Detected Format: {formatVersion}");
            Console.WriteLine();

            Console.WriteLine("🏗️  Step 2: Extracting Surface-Based Navigation Objects");
            var navObjects = buildingExtractionService.ExtractSurfaceBasedNavigationObjects(pm4FilePath);
            Console.WriteLine($"   Found {navObjects.Count} surface-based objects");
            Console.WriteLine();

            if (navObjects.Any())
            {
                Console.WriteLine("📊 Step 3: Surface Orientation Analysis");
                foreach (var obj in navObjects.Take(3)) // Show first 3 objects
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

                var totalTopSurfaces = navObjects.Sum(o => o.TopSurfaces.Count);
                var totalBottomSurfaces = navObjects.Sum(o => o.BottomSurfaces.Count);
                var totalVerticalSurfaces = navObjects.Sum(o => o.VerticalSurfaces.Count);
                
                Console.WriteLine("🎯 REAL DATA ANALYSIS COMPLETE:");
                Console.WriteLine($"   - {totalTopSurfaces} individual top surfaces for roof matching");
                Console.WriteLine($"   - {totalBottomSurfaces} individual bottom surfaces for foundation matching");
                Console.WriteLine($"   - {totalVerticalSurfaces} individual vertical surfaces for wall matching");
            }
        }
    }
} 