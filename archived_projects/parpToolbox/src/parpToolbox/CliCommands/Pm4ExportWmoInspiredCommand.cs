using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    public class Pm4ExportWmoInspiredOptions
    {
        public string InputFile { get; set; } = "";
        public string? OutputDirectory { get; set; }
    }

    public class Pm4ExportWmoInspiredCommand
    {
        private readonly Pm4Adapter _pm4Adapter;
        private readonly Pm4WmoInspiredObjectAssembler _assembler;

        public Pm4ExportWmoInspiredCommand(Pm4Adapter pm4Adapter, Pm4WmoInspiredObjectAssembler assembler)
        {
            _pm4Adapter = pm4Adapter;
            _assembler = assembler;
        }

        public void Execute(Pm4ExportWmoInspiredOptions opts)
        {
            try
            {
                ConsoleLogger.WriteLine("=== PM4 WMO-Inspired Export ===");
                ConsoleLogger.WriteLine($"Input: {opts.InputFile}");

                // Validate input file
                if (!File.Exists(opts.InputFile))
                {
                    ConsoleLogger.WriteLine($"ERROR: Input file not found: {opts.InputFile}");
                    return;
                }

                // Setup output directory
                var outputDir = opts.OutputDirectory ?? Path.Combine("project_output", $"pm4_wmo_export_{DateTime.Now:yyyyMMdd_HHmmss}");
                Directory.CreateDirectory(outputDir);
                ConsoleLogger.WriteLine($"Output: {outputDir}");

                // Load PM4 scene
                ConsoleLogger.WriteLine("Loading PM4 scene...");
                var scene = _pm4Adapter.Load(opts.InputFile);
                
                ConsoleLogger.WriteLine($"Scene loaded: {scene.Placements.Count} placements, {scene.Links.Count} links, {scene.Surfaces.Count} surfaces");
                ConsoleLogger.WriteLine($"Total vertices: {scene.Vertices.Count}, indices: {scene.Indices.Count}");

                // Assemble objects using WMO-inspired logic
                ConsoleLogger.WriteLine("\nAssembling objects using WMO organizational logic...");
                var objects = _assembler.AssembleObjects(scene);

                if (objects.Count == 0)
                {
                    ConsoleLogger.WriteLine("WARNING: No objects assembled! Check PM4 data structure.");
                    return;
                }

                // Display assembly statistics
                ConsoleLogger.WriteLine($"\n=== Assembly Statistics ===");
                ConsoleLogger.WriteLine($"Total objects assembled: {objects.Count}");
                ConsoleLogger.WriteLine($"Expected objects (~458 unique MPRL.Unknown4 values): 458");
                ConsoleLogger.WriteLine($"Match ratio: {(objects.Count / 458.0):P1}");

                var totalTriangles = objects.Sum(o => o.TotalTriangles);
                var totalVertices = objects.Sum(o => o.TotalVertices);
                ConsoleLogger.WriteLine($"Total triangles across all objects: {totalTriangles:N0}");
                ConsoleLogger.WriteLine($"Total vertices across all objects: {totalVertices:N0}");

                // Show size distribution
                var sizeGroups = objects.GroupBy(o => 
                    o.TotalTriangles < 10 ? "Tiny (< 10)" :
                    o.TotalTriangles < 100 ? "Small (10-99)" :
                    o.TotalTriangles < 1000 ? "Medium (100-999)" :
                    o.TotalTriangles < 10000 ? "Large (1K-9K)" : "Huge (10K+)")
                    .OrderBy(g => g.Key);

                ConsoleLogger.WriteLine("\nObject size distribution:");
                foreach (var group in sizeGroups)
                {
                    ConsoleLogger.WriteLine($"  {group.Key}: {group.Count()} objects");
                }

                // Export objects to OBJ files
                ConsoleLogger.WriteLine("\nExporting objects to OBJ files...");
                _assembler.ExportObjects(objects, outputDir);

                // Write summary report
                var reportPath = Path.Combine(outputDir, "assembly_report.txt");
                WriteSummaryReport(objects, scene, reportPath);

                ConsoleLogger.WriteLine($"\n=== Export Complete ===");
                ConsoleLogger.WriteLine($"Exported {objects.Count} objects to: {outputDir}");
                ConsoleLogger.WriteLine($"Assembly report: {reportPath}");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"ERROR: {ex.Message}");
                ConsoleLogger.WriteLine($"Stack trace: {ex.StackTrace}");
            }
        }

        private void WriteSummaryReport(List<Pm4WmoInspiredObjectAssembler.WmoInspiredObject> objects, Pm4Scene scene, string reportPath)
        {
            using (var writer = new StreamWriter(reportPath))
            {
                writer.WriteLine("PM4 WMO-Inspired Assembly Report");
                writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                writer.WriteLine();

                writer.WriteLine("=== Scene Statistics ===");
                writer.WriteLine($"Placements (MPRL): {scene.Placements.Count}");
                writer.WriteLine($"Links (MSLK): {scene.Links.Count}");
                writer.WriteLine($"Surfaces (MSUR): {scene.Surfaces.Count}");
                writer.WriteLine($"Total vertices: {scene.Vertices.Count:N0}");
                writer.WriteLine($"Total indices: {scene.Indices.Count:N0}");
                writer.WriteLine();

                writer.WriteLine("=== Assembly Results ===");
                writer.WriteLine($"Objects assembled: {objects.Count}");
                writer.WriteLine($"Expected objects (unique MPRL.Unknown4): 458");
                writer.WriteLine($"Success ratio: {(objects.Count / 458.0):P1}");
                writer.WriteLine();

                writer.WriteLine("=== Object Details ===");
                writer.WriteLine("ObjectID\tTriangles\tVertices\tBoundingBox");
                
                foreach (var obj in objects.OrderByDescending(o => o.TotalTriangles))
                {
                    var bbox = $"({obj.BoundingBoxMin.X:F1},{obj.BoundingBoxMin.Y:F1},{obj.BoundingBoxMin.Z:F1}) to ({obj.BoundingBoxMax.X:F1},{obj.BoundingBoxMax.Y:F1},{obj.BoundingBoxMax.Z:F1})";
                    writer.WriteLine($"{obj.ObjectId}\t{obj.TotalTriangles}\t{obj.TotalVertices}\t{bbox}");
                }

                writer.WriteLine();
                writer.WriteLine("=== WMO Organizational Logic Applied ===");
                writer.WriteLine("1. Grouped by MPRL.Unknown4 (like WMO group IDs)");
                writer.WriteLine("2. Found MSLK links for each object (like MOGP batch references)");
                writer.WriteLine("3. Extracted geometry from MSUR surfaces (like MOBA render batches)");
                writer.WriteLine("4. Applied WMO coordinate system (X-axis inversion)");
                writer.WriteLine("5. Grouped triangles by material ID (like WMO material batching)");
            }
        }
    }
}
