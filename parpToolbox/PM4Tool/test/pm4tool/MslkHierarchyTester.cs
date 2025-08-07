using System;
using WoWToolbox.Core.Navigation.PM4;

class MslkHierarchyTester
{
    static void Main(string[] args)
    {
        Console.WriteLine("🔗 MSLK HIERARCHY ANALYSIS TESTER");
        Console.WriteLine("═══════════════════════════════════════════════════════════════");
        Console.WriteLine();

        if (args.Length > 0)
        {
            // Analyze specific file if provided
            Console.WriteLine($"Analyzing specific file: {args[0]}");
            MslkHierarchyDemo.AnalyzeSingleFile(args[0]);
        }
        else
        {
            // Run general analysis
            MslkHierarchyDemo.RunHierarchyAnalysis();
        }

        Console.WriteLine();
        Console.WriteLine("Press any key to exit...");
        Console.ReadKey();
    }
} 