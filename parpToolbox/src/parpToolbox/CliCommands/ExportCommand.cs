using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// Handles the legacy *pm4* export functionality in a single cohesive place. This class is a straight extraction of
    /// the inline logic that previously lived in Program.cs.  No behavioural changes are introduced – it only consolidates
    /// the code to keep Program.cs clean.
    /// </summary>
    internal static class ExportCommand
    {
        /// <summary>
        /// Executes a PM4 export run.
        /// </summary>
        /// <param name="args">Full CLI args array (including the command token at index 0).</param>
        /// <param name="inputPath">Resolved absolute path to the PM4 file specified by the user.</param>
        /// <returns>Exit code (0 = success, non-zero = failure).</returns>
        public static int Run(string[] args, string inputPath)
        {
            // Flag parsing (mirrors former Program.cs logic)
            bool exportFaces   = args.Contains("--exportfaces");
            bool exportChunks  = args.Contains("--exportchunks");
            bool bulkDump      = args.Contains("--bulk-dump");
            bool csvDump       = args.Contains("--csv-dump");
            bool exportObjects = args.Contains("--objects") || args.Contains("--indexcount");
            bool selectorGrouping = args.Contains("--selector-grouping") || args.Contains("--selector");
            bool useSingleTile = args.Contains("--single-tile");
            bool useNewExporter = args.Contains("--new-exporter");
            bool enableCrossTile = args.Contains("--cross-tile");
            bool enableMprlTransforms = args.Contains("--mprl-transforms");
            bool includeM2Objects = args.Contains("--include-m2");
            bool disableXAxisInversion = args.Contains("--no-x-inversion");
            bool exportAsWmo = args.Contains("--wmo");

            // Load scene (region loader by default)
            Pm4Scene scene;
            var adapter = new Pm4Adapter();
            if (useSingleTile)
            {
                ConsoleLogger.WriteLine("Single-tile mode active (--single-tile flag detected)");
                scene = adapter.Load(inputPath);
            }
            else
            {
                ConsoleLogger.WriteLine("Region mode active (default) – loading cross-tile references...");
                scene = adapter.LoadRegion(inputPath);
            }

            var outputDir = ProjectOutput.CreateOutputDirectory(Path.GetFileNameWithoutExtension(inputPath));

            // Use NewPm4Exporter if requested
            if (useNewExporter)
            {
                ConsoleLogger.WriteLine("Using NewPm4Exporter...");
                
                // Configure export options
                var options = new NewPm4Exporter.ExportOptions
                {
                    Format = exportAsWmo ? NewPm4Exporter.ExportFormat.Wmo : NewPm4Exporter.ExportFormat.Obj,
                    MinTriangles = 10, // Default value, could be made configurable
                    ApplyXAxisInversion = !disableXAxisInversion,
                    IncludeM2Objects = includeM2Objects,
                    EnableMprlTransformations = enableMprlTransforms,
                    EnableCrossTileResolution = enableCrossTile
                };
                
                // Create exporter
                NewPm4Exporter exporter;
                if (enableCrossTile)
                {
                    // For cross-tile resolution, we need to pass the directory path
                    var directoryPath = Directory.Exists(inputPath) ? inputPath : Path.GetDirectoryName(inputPath);
                    exporter = new NewPm4Exporter(directoryPath, options);
                }
                else
                {
                    exporter = new NewPm4Exporter(scene, options);
                }
                
                // Export
                var exportedCount = exporter.Export(outputDir);
                ConsoleLogger.WriteLine($"Export complete – wrote {exportedCount} objects to '{outputDir}'.");
                return 0;
            }

            // Early-exit paths ------------------------------------------------------
            if (bulkDump)
            {
                var bulkDir = Path.Combine(outputDir, "bulk_dump");
                ConsoleLogger.WriteLine($"Running bulk dump to {bulkDir} ...");
                Pm4BulkDumper.Dump(scene, bulkDir, exportFaces, Pm4Adapter.LastRawMsvtData);
                ConsoleLogger.WriteLine("Bulk dump complete!");
                return 0;
            }

            if (csvDump)
            {
                // Clean previous timestamped dumps
                foreach (var dir in Directory.GetDirectories(outputDir, "csv_dump_*"))
                {
                    try { Directory.Delete(dir, recursive: true); }
                    catch (Exception ex) { ConsoleLogger.WriteLine($"Warning: Failed to delete old dump {dir}: {ex.Message}"); }
                }

                var csvDir = Path.Combine(outputDir, "csv_dump");
                ConsoleLogger.WriteLine($"Running CSV dump to {csvDir} ...");
                Pm4CsvDumper.DumpAllChunks(scene, csvDir);
                return 0;
            }

            if (selectorGrouping)
            {
                ConsoleLogger.WriteLine("Exporting objects by selector key (XX/YY) ...");
                var selectorObjs = Pm4MsurObjectAssembler.AssembleObjectsBySelectorKey(scene);
                Pm4MsurObjectAssembler.ExportMsurObjects(selectorObjs, scene, outputDir);
                ConsoleLogger.WriteLine($"Selector object export complete – wrote {selectorObjs.Count} objects to '{outputDir}'.");
                return 0;
            }

            if (exportObjects)
            {
                ConsoleLogger.WriteLine("Exporting assembled objects by MSUR.IndexCount ...");
                var assembled = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
                Pm4MsurObjectAssembler.ExportMsurObjects(assembled, scene, outputDir);
                ConsoleLogger.WriteLine("Assembled object export complete!");
                return 0;
            }
            // ---------------------------------------------------------------------

            // Default path: assemble objects by MSUR.IndexCount (proof-of-concept logic)
            ConsoleLogger.WriteLine("=== PM4 EXPORT (assembled objects default) ===");
            var assembledObjects = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
            Pm4MsurObjectAssembler.ExportMsurObjects(assembledObjects, scene, outputDir);
            ConsoleLogger.WriteLine($"Export complete – wrote {assembledObjects.Count} assembled objects to '{outputDir}'.");
            return 0;
        }
    }
}
