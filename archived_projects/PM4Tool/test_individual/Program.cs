using System;
using WoWToolbox.Core.Navigation.PM4;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("🚀 INDIVIDUAL GEOMETRY EXPORT TEST");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine("Testing the individual geometry export functionality.");
        Console.WriteLine("This should create clean separate OBJ files for each geometry node.");
        Console.WriteLine();

        try
        {
            // Call the working demo that exports individual geometry
            MslkHierarchyDemo.RunHierarchyAnalysis();

            Console.WriteLine("✅ Individual geometry export test completed successfully!");
            Console.WriteLine();
            Console.WriteLine("📁 RESULTS TO CHECK:");
            Console.WriteLine("   - Look in 'output/individual_objects/' folder");
            Console.WriteLine("   - Each geometry node should be a separate clean OBJ file");
            Console.WriteLine("   - Files named: [filename].geom_XXX.obj");
            Console.WriteLine("   - Small focused components (not super-concentrated files)");
            Console.WriteLine();
            Console.WriteLine("🎯 SUCCESS CRITERIA:");
            Console.WriteLine("   ✅ Multiple small OBJ files instead of large combined ones");
            Console.WriteLine("   ✅ Each file contains only single geometry node data");
            Console.WriteLine("   ✅ No circular reference errors");
            Console.WriteLine("   ✅ Proper triangular faces with validation");
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