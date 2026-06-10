using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

public static class Pm4PerObjectExporterTests
{
    public static void RunRegressionTest()
    {
        var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        var pm4Path = Path.Combine(projectRoot, "test_data", "original_development", "development_00_00.pm4");
        var expectedHash = "PLACEHOLDER_HASH"; // This will be replaced later

        var outputDir = Path.Combine(projectRoot, "project_output", $"regression_test_{DateTime.Now:yyyyMMdd_HHmmss}");
        Directory.CreateDirectory(outputDir);

        ConsoleLogger.WriteLine("[RegressionTest] Running PM4 Per-Object Exporter test...");

        if (!File.Exists(pm4Path))
        {
            ConsoleLogger.WriteLine($"[RegressionTest] Error: Input file not found at {pm4Path}");
            return;
        }

        // Correctly load the scene using the adapter
        var adapter = new Pm4Adapter();
        var scene = adapter.Load(pm4Path);
        var exporter = new Pm4PerObjectExporter();
        
        string exportPath = Path.Combine(outputDir, "regression_test_output");
        exporter.ExportObjects(scene, exportPath);

        // Find the largest exported OBJ file, which should be the main building
        var largestObj = new DirectoryInfo(exportPath)
            .GetFiles("*.obj")
            .OrderByDescending(f => f.Length)
            .FirstOrDefault();

        if (largestObj == null)
        {
            ConsoleLogger.WriteLine("[RegressionTest] Error: No OBJ files were exported.");
            return;
        }

        // Calculate SHA256 hash of the largest file
        using var sha256 = SHA256.Create();
        using var fileStream = largestObj.OpenRead();
        byte[] hashBytes = sha256.ComputeHash(fileStream);
        var hash = new StringBuilder(hashBytes.Length * 2);
        foreach (byte b in hashBytes)
        {
            hash.Append(b.ToString("x2"));
        }

        string actualHash = hash.ToString();
        // The expected hash is defined at the top of the method.

        ConsoleLogger.WriteLine($"[RegressionTest] Exported object file: {largestObj.Name}");
        ConsoleLogger.WriteLine($"[RegressionTest] Expected hash: {expectedHash}");
        ConsoleLogger.WriteLine($"[RegressionTest] Actual hash:   {actualHash}");

        if (actualHash == expectedHash)
        {
            ConsoleLogger.WriteLine("[RegressionTest] Result: SUCCESS - Hashes match!");
        }
        else
        {
            ConsoleLogger.WriteLine("[RegressionTest] Result: FAILED - Hash mismatch!");
        }
    }
}
