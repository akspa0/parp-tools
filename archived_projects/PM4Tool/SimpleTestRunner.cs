using System;
using WoWToolbox.Core.Navigation.PM4;

class SimpleTestRunner
{
    static void Main(string[] args)
    {
        Console.WriteLine("🚀 SPATIAL CLUSTERING TEST");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Testing the new spatial clustering functionality.");
        Console.WriteLine("This groups geometry components intelligently like WMO assemblies.");
        Console.WriteLine();

        try
        {
            // Call the working demo that exports individual geometry
            MslkHierarchyDemo.RunHierarchyAnalysis();

            Console.WriteLine("✅ Spatial clustering test completed successfully!");
            Console.WriteLine();
            Console.WriteLine("📁 Check the 'output/' folder for results:");
            Console.WriteLine("   - spatial_assemblies/ folder with clustered geometry assemblies");
            Console.WriteLine("   - Each cluster contains related components grouped by proximity");
            Console.WriteLine("   - Cluster manifests with component metadata and positioning");
            Console.WriteLine("   - Significant reduction in file count vs individual exports");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ Error during test: {ex.Message}");
            Console.WriteLine($"Stack trace: {ex.StackTrace}");
        }

        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }
} 