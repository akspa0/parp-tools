using System;
using System.IO;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.SpatialAnalyzer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== WoWToolbox PM4 Spatial Analyzer ===\n");
            
            // Use the standard test file
            var testDataRoot = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
            var testFile = Path.Combine(testDataRoot, "original_development", "development", "development_00_00.pm4");
            var outputDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "output", "spatial_analysis"));
            
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"❌ Test file not found: {testFile}");
                Console.WriteLine($"📁 Looking in: {Path.GetDirectoryName(testFile)}");
                
                if (Directory.Exists(Path.GetDirectoryName(testFile)))
                {
                    var files = Directory.GetFiles(Path.GetDirectoryName(testFile), "*.pm4");
                    Console.WriteLine($"Available PM4 files: {files.Length}");
                    if (files.Length > 0)
                    {
                        testFile = files[0];
                        Console.WriteLine($"Using first available file: {Path.GetFileName(testFile)}");
                    }
                    else
                    {
                        Console.WriteLine("No PM4 files found in directory.");
                        return;
                    }
                }
                else
                {
                    Console.WriteLine("Test data directory does not exist.");
                    return;
                }
            }
            
            try
            {
                SpatialAnalysisUtility.AnalyzeChunkSpatialDistribution(testFile, outputDir);
                
                Console.WriteLine("\n🎉 Analysis completed successfully!");
                Console.WriteLine($"📂 Output files written to: {outputDir}");
                Console.WriteLine("\n💡 Load the individual OBJ files in MeshLab to identify which chunk type is causing spatial separation.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ Error: {ex.Message}");
                Console.WriteLine($"📋 Stack trace: {ex.StackTrace}");
            }
            
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
} 