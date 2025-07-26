using System;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Unified test command for PM4/PD4 validation and regression testing.
    /// </summary>
    public static class TestCommand
    {
        /// <summary>
        /// Executes test validation logic. Returns process exit code (0 = OK, 1 = error).
        /// </summary>
        /// <param name="args">Full command-line args array.</param>
        /// <param name="inputPath">Resolved absolute input file path.</param>
        public static int Run(string[] args, string inputPath)
        {
            try
            {
                ConsoleLogger.WriteLine($"Running PM4 validation tests on: {inputPath}");
                
                var adapter = new Pm4Adapter();
                var scene = adapter.LoadRegion(inputPath);
                
                // Validate basic scene integrity
                if (scene == null)
                {
                    ConsoleLogger.WriteLine("ERROR: Failed to load scene");
                    return 1;
                }

                // Check for MSUR data
                var msurCount = scene.Surfaces?.Count ?? 0;
                ConsoleLogger.WriteLine($"✓ Loaded {msurCount} MSUR entries");
                
                // Check for geometry data
                var vertices = scene.Vertices?.Count ?? 0;
                var indices = scene.Triangles?.Count ?? 0;
                ConsoleLogger.WriteLine($"✓ Loaded {vertices} vertices");
                ConsoleLogger.WriteLine($"✓ Loaded {indices} triangles");
                
                // Basic validation
                bool isValid = true;
                var errors = new System.Collections.Generic.List<string>();
                
                if (msurCount == 0)
                {
                    errors.Add("No MSUR entries found");
                    isValid = false;
                }
                
                if (vertices == 0)
                {
                    errors.Add("No vertex data found");
                    isValid = false;
                }
                
                if (indices == 0)
                {
                    errors.Add("No triangle data found");
                    isValid = false;
                }
                
                if (isValid)
                {
                    ConsoleLogger.WriteLine("✓ All basic validation tests passed");
                    return 0;
                }
                else
                {
                    ConsoleLogger.WriteLine("✗ Validation failed:");
                    foreach (var error in errors)
                    {
                        ConsoleLogger.WriteLine($"  {error}");
                    }
                    return 1;
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: Test validation failed: {ex.Message}");
                return 1;
            }
        }
    }
}
