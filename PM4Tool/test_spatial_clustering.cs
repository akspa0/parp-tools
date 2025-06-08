using System;
using WoWToolbox.Core.Navigation.PM4;

class TestSpatialClustering
{
    static void Main(string[] args)
    {
        Console.WriteLine("🚀 TESTING SPATIAL CLUSTERING FUNCTIONALITY");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        
        try
        {
            // Call the spatial clustering demo directly
            MslkHierarchyDemo.RunHierarchyAnalysis();
            
            Console.WriteLine("\n✅ Spatial clustering test completed successfully!");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }
        
        Console.WriteLine("\nPress any key to exit...");
        Console.ReadKey();
    }
} 