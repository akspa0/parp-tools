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
            bool useSingleTile = args.Contains("--single-tile");

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

            if (exportObjects)
            {
                ConsoleLogger.WriteLine("Exporting assembled objects by MSUR.IndexCount ...");
                var assembled = Pm4MsurObjectAssembler.AssembleObjectsByMsurIndex(scene);
                Pm4MsurObjectAssembler.ExportMsurObjects(assembled, scene, outputDir);
                ConsoleLogger.WriteLine("Assembled object export complete!");
                return 0;
            }
            // ---------------------------------------------------------------------

            // Default path: export whole scene
            var outputFile = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(inputPath) + ".obj");
            if (exportChunks && scene.Groups.Count > 0)
            {
                ConsoleLogger.WriteLine($"Exporting {scene.Groups.Count} groups to {outputDir}...");
                Pm4GroupObjExporter.Export(scene, outputDir, exportFaces);
            }
            else
            {
                ConsoleLogger.WriteLine($"Exporting OBJ to {outputFile}...");
                Pm4ObjExporter.Export(scene, outputFile, exportFaces);
            }
            ConsoleLogger.WriteLine("Export complete!");
            return 0;
        }
    }
}
