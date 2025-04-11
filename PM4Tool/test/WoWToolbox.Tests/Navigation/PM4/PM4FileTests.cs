using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Xunit;
using WoWToolbox.Core.Navigation.PM4;
using System.Numerics;
using WoWToolbox.Core.Vectors;
using WoWToolbox.Core;
using Microsoft.Extensions.Logging;
using System.Reflection;
// using WoWToolbox.Tests.Analysis; // REMOVED
using System.Text.Json; // ADDED for JSON Serialization
// REMOVED: using WoWToolbox.AnalysisTool; 

namespace WoWToolbox.Tests.Navigation.PM4
{
    // --- DTOs for MSLK JSON Hierarchy ---
    public class MslkNodeEntryDto
    {
        public int EntryIndex { get; set; }
        public byte Unk00 { get; set; }
        public byte Unk01 { get; set; }
        public ushort Unk02 { get; set; }
        public uint Unk0C { get; set; }
        public ushort Unk10 { get; set; }
        public ushort Unk12 { get; set; }
    }

    public class MslkGeometryEntryDto
    {
        public int EntryIndex { get; set; }
        public int MspiFirstIndex { get; set; }
        public byte MspiIndexCount { get; set; }
        public byte Unk00 { get; set; }
        public byte Unk01 { get; set; }
        public ushort Unk02 { get; set; }
        public uint Unk0C { get; set; }
        public ushort Unk10 { get; set; }
        public ushort Unk12 { get; set; }
    }

    public class MslkGroupDto
    {
        public List<MslkNodeEntryDto> Nodes { get; set; } = new List<MslkNodeEntryDto>();
        public List<MslkGeometryEntryDto> Geometry { get; set; } = new List<MslkGeometryEntryDto>();
    }
    // --- End DTOs ---

    public class PM4FileTests
    {
        private const string TestDataPath = "test_data/development/development_22_18.pm4"; // Keep for potential reference, but not used directly in main test
        private const float ScaleFactor = 36.0f; // Common scaling factor
        private const float CoordinateOffset = 17066.666f; // From MsvtVertex documentation/constants

        // Removed ApplyMprlTransform helper function

        [Fact]
        // public void LoadPM4File_ShouldLoadChunks() // Renamed/Replaced
        public void LoadAndProcessPm4FilesInDirectory_ShouldGenerateOutputs()
        {
            Console.WriteLine("--- LoadAndProcessPm4FilesInDirectory START ---");

            // --- Path Construction (Input Directory) ---
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            // Use Path.Combine for robust path handling across OS
            // Corrected path: Go up 5 levels from bin/Debug/netX.0 to reach solution root
            var testDataRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "test_data"));
            var inputDirectoryPath = Path.Combine(testDataRoot, "development");

            // --- Path Construction (Output Directory) ---
            // Place output relative to the test execution directory for clarity
            var outputSubDir = Path.Combine("output", "development");
            var outputDir = Path.Combine(baseDir, outputSubDir);
            Directory.CreateDirectory(outputDir); // Ensure the output directory exists

            Console.WriteLine($"Input Directory: {inputDirectoryPath}");
            Console.WriteLine($"Output Directory: {outputDir}");

             if (!Directory.Exists(inputDirectoryPath))
            {
                Console.WriteLine($"ERROR: Input directory not found: {inputDirectoryPath}");
                Assert.Fail($"Input directory not found: {inputDirectoryPath}"); // Fail the test if input dir is missing
                return;
            }

            // --- Get PM4 Files ---
            var pm4Files = Directory.EnumerateFiles(inputDirectoryPath, "*.pm4", SearchOption.TopDirectoryOnly).ToList(); // ToList() to get count easily
            int processedCount = 0;
            int errorCount = 0;

            Console.WriteLine($"Found {pm4Files.Count} PM4 files to process.");

            if (!pm4Files.Any())
            {
                Console.WriteLine("WARNING: No PM4 files found in the input directory.");
                // Assert.True(pm4Files.Any(), "No PM4 files found in the input directory."); // Optional: Fail if no files found
            }

            // --- Loop Through Files ---
            foreach (var inputFilePath in pm4Files)
            {
                var fileName = Path.GetFileName(inputFilePath);
                Console.WriteLine($"\n==================== Processing File: {fileName} ====================");
                try
                {
                    ProcessSinglePm4File(inputFilePath, outputDir); // Call the helper method
                    processedCount++;
                    Console.WriteLine($"-------------------- Successfully processed: {fileName} --------------------");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"!!!!!!!!!!!!!!!!!!!! ERROR processing file {fileName} !!!!!!!!!!!!!!!!!!!!");
                    Console.WriteLine($"Error Message: {ex.Message}");
                    Console.WriteLine($"Stack Trace:\n{ex.StackTrace}");
                    errorCount++;
                    // Optionally write error to a main error log file here
                    // File.AppendAllText(Path.Combine(outputDir, "_ERROR_LOG.txt"), $"Error processing {fileName}: {ex.ToString()}\n\n");
                }
                Console.WriteLine($"============================================================================");

            }

            Console.WriteLine($"\n--- LoadAndProcessPm4FilesInDirectory FINISHED ---");
            Console.WriteLine($"Successfully Processed: {processedCount} files");
            Console.WriteLine($"Encountered Errors:   {errorCount} files");

            // Assertions can be adjusted based on desired test outcome.
            // For example, fail if ANY file caused an error:
            Assert.True(errorCount == 0, $"Encountered {errorCount} errors during batch processing. Check console/log output.");
            // Or, assert that at least one file was processed successfully if the directory wasn't empty:
            if (pm4Files.Any())
            {
                Assert.True(processedCount > 0, "No PM4 files were successfully processed, although files were found.");
            }
        }


        // --- Helper Method for Single File Processing ---
        private void ProcessSinglePm4File(string inputFilePath, string outputDir)
        {
             // 1. Derive baseOutputName from inputFilePath
            var baseOutputName = Path.GetFileNameWithoutExtension(inputFilePath);
            var baseOutputPath = Path.Combine(outputDir, baseOutputName);

            // --- Path Construction (Relative to Test Output Directory) ---
            var baseDir = AppDomain.CurrentDomain.BaseDirectory; // Needed? No, use outputDir directly.

            // 2. Define all output file paths using baseOutputPath (as before)
            var outputMspvFilePath = baseOutputPath + "_mspv.obj";
            var outputMprlFilePath = baseOutputPath + "_mprl.obj";
            var outputMslkFilePath = baseOutputPath + "_mslk.obj";
            var outputMslkJsonPath = baseOutputPath + "_mslk_hierarchy.json";
            var outputSkippedMslkLogPath = baseOutputPath + "_skipped_mslk.log";
            var outputPm4MslkNodesFilePath = baseOutputPath + "_pm4_mslk_nodes.obj";
            var outputRenderMeshPath = baseOutputPath + "_render_mesh.obj";
            string debugLogPath = baseOutputPath + ".debug.log";
            string summaryLogPath = baseOutputPath + ".summary.log";
            var outputBuildingIdsPath = baseOutputPath + "_building_ids.log";

            // Console logging for paths specific to this file
            Console.WriteLine($"  Input File Path: {inputFilePath}");
            Console.WriteLine($"  Output Base Path: {baseOutputPath}");
            Console.WriteLine($"  Output MSPV OBJ: {outputMspvFilePath}");
            Console.WriteLine($"  Output MPRL OBJ: {outputMprlFilePath}");
            Console.WriteLine($"  Output MSLK OBJ: {outputMslkFilePath}");
            Console.WriteLine($"  Output MSLK JSON: {outputMslkJsonPath}");
            Console.WriteLine($"  Output Skipped MSLK Log: {outputSkippedMslkLogPath}");
            Console.WriteLine($"  Output PM4 MSLK Nodes OBJ: {outputPm4MslkNodesFilePath}");
            Console.WriteLine($"  Output RENDER MESH OBJ: {outputRenderMeshPath}");
            Console.WriteLine($"  Debug Log: {debugLogPath}");
            Console.WriteLine($"  Summary Log: {summaryLogPath}");
            Console.WriteLine($"  Building IDs Log: {outputBuildingIdsPath}");

            // 3. Initialize HashSet for unique building IDs (scoped per file)
            var uniqueBuildingIds = new HashSet<uint>();

            // 4. Load the PM4 file
            var pm4File = PM4File.FromFile(inputFilePath);
            Assert.NotNull(pm4File); // Basic assertion per file

            // Log first few MPRR entries for debugging index issue
            Console.WriteLine("\n  --- Dumping MPRR Entries 60-75 (if they exist) ---");
            if (pm4File.MPRR != null && pm4File.MPRR.Entries.Count > 60)
            {
                int startIndex = 60;
                int endIndex = Math.Min(startIndex + 16, pm4File.MPRR.Entries.Count); // Log around 16 entries
                for (int i = startIndex; i < endIndex; i++)
                {
                    var entry = pm4File.MPRR.Entries[i];
                    // Update to use new MprrEntry fields (ushort Unknown_0x00, ushort Unknown_0x02)
                    Console.WriteLine($"    Entry {i}: Unk0=0x{entry.Unknown_0x00:X4}({entry.Unknown_0x00}), Unk2=0x{entry.Unknown_0x02:X4}({entry.Unknown_0x02})");
                }
            } else {
                 Console.WriteLine("    MPRR chunk missing or has less than 60 entries.");
            }
            Console.WriteLine("  --- End MPRR Dump ---\n");


            // --- Export Configuration Flags ---
            // These could potentially be parameters to the helper method if needed
            bool exportMspvVertices = true; // Keep for structure
            bool exportMsvtVertices = true; // Need vertices for render mesh
            bool exportMprlPoints = true; // Keep as separate point cloud
            bool exportMslkPaths = true; // Keep for structure/Doodads
            bool exportOnlyFirstMslk = false;
            bool processMsurEntries = true; // Need MSUR for render mesh faces
            bool exportOnlyFirstMsur = false;
            bool logMdsfLinks = true;
            bool exportMscnPoints = false; // Disable MSCN point export

            // --- Initialize Writers ---
            // Use 'using' statements for automatic disposal
            using var debugWriter = new StreamWriter(debugLogPath, false);
            using var summaryWriter = new StreamWriter(summaryLogPath, false);
            using var buildingIdWriter = new StreamWriter(outputBuildingIdsPath, false);
            using var renderMeshWriter = new StreamWriter(outputRenderMeshPath, false);
            using var mspvWriter = new StreamWriter(outputMspvFilePath);
            using var mprlWriter = new StreamWriter(outputMprlFilePath);
            using var mslkWriter = new StreamWriter(outputMslkFilePath);
            using var skippedMslkWriter = new StreamWriter(outputSkippedMslkLogPath);
            using var mslkNodesWriter = new StreamWriter(outputPm4MslkNodesFilePath, false);
            // using var mscnWriter = exportMscnPoints ? new StreamWriter(baseOutputPath + "_mscn.obj") : null; // Conditional writer example

            try // Keep inner try for logging/resource cleanup context if needed, though outer one catches errors
            {
                // Log MDOS Entry Count for verification (MOVED HERE)
                int mdosEntryCount = pm4File.MDOS?.Entries.Count ?? -1;
                Console.WriteLine($"  INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");
                debugWriter.WriteLine($"INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");

                // Log counts of other potentially relevant chunks
                debugWriter.WriteLine($"INFO: MSVT Vertices: {pm4File.MSVT?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSPV Vertices: {pm4File.MSPV?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MPRL Entries: {pm4File.MPRL?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSPI Indices: {pm4File.MSPI?.Indices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSVI Indices: {pm4File.MSVI?.Indices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MPRR Entries: {pm4File.MPRR?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSUR Entries: {pm4File.MSUR?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MDBH Entries: {pm4File.MDBH?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MDSF Entries: {pm4File.MDSF?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSLK Entries: {pm4File.MSLK?.Entries.Count ?? -1}");


                mslkNodesWriter.WriteLine($"# PM4 MSLK Node Anchor Points (from Unk10 -> MSVI -> MSVT)");
                mslkNodesWriter.WriteLine($"# Transform: Y, X, Z");

                // Write headers for non-MPRL files
                renderMeshWriter.WriteLine($"# PM4 Render Mesh (MSVT/MSVI/MSUR) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                renderMeshWriter.WriteLine("# Vertices Transform: Y, X, Z");
                mspvWriter.WriteLine($"# PM4 MSPV/MSLK Geometry (X, Y, Z) - File: {Path.GetFileName(inputFilePath)}");
                mslkWriter.WriteLine($"# PM4 MSLK Geometry (Points 'p' and Lines 'l') (Exported: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                skippedMslkWriter.WriteLine($"# PM4 Skipped/Invalid MSLK Entries Log (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                buildingIdWriter.WriteLine($"# Unique Building IDs from MDOS (via MDSF/MSUR link) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");

                // --- Moved Index Validation Logging Inside Try Block ---
                int mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");
                if (pm4File.MPRR != null)
                {
                    // Log MPRR validation attempts - don't assert for batch processing
                    bool mprrIndicesValid = true;
                    for(int i = 0; i < pm4File.MPRR.Entries.Count; i++) {
                         var entry = pm4File.MPRR.Entries[i];
                         // Example validation logic (adjust if fields change)
                         // if (entry.SomeIndexField >= mprlVertexCount) {
                         //    debugWriter.WriteLine($"  WARN: MPRR Entry {i} has invalid index {entry.SomeIndexField} (MPRL Count: {mprlVertexCount})");
                         //    mprrIndicesValid = false;
                         // }
                    }
                     debugWriter.WriteLine(mprrIndicesValid ? "MPRR Indices appear valid within MPRL bounds (basic check)." : "MPRR Indices validation logged potential issues.");
                } else {
                     debugWriter.WriteLine("MPRR Chunk not present, skipping validation.");
                }


                int msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSVI Indices against MSVT Vertex Count: {msvtVertexCount} ---");
                if (pm4File.MSVI != null && msvtVertexCount > 0)
                {
                    bool msviIndicesValid = true;
                    for(int i = 0; i < pm4File.MSVI.Indices.Count; i++) {
                         var msviIndex = pm4File.MSVI.Indices[i];
                        if (msviIndex >= msvtVertexCount) {
                            debugWriter.WriteLine($"  ERROR: MSVI index {msviIndex} at pos {i} is out of bounds for MSVT vertex count {msvtVertexCount}");
                            msviIndicesValid = false;
                             // Optionally throw an exception here if this should halt processing for this file
                             // throw new IndexOutOfRangeException($"Critical Error: MSVI index {msviIndex} out of bounds.");
                        }
                    }
                    debugWriter.WriteLine(msviIndicesValid ? "MSVI Indices validated successfully." : "MSVI Indices validation FAILED.");
                     Assert.True(msviIndicesValid, "One or more MSVI indices were out of bounds for the MSVT vertex count."); // Keep assertion for this critical link
                } else {
                    debugWriter.WriteLine("MSVI Chunk not present or MSVT empty, skipping validation.");
                }


                int mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSPI Indices against MSPV Vertex Count: {mspvVertexCount} ---");
                if (pm4File.MSPI != null && mspvVertexCount > 0)
                {
                    bool mspiIndicesValid = true;
                     for(int i = 0; i < pm4File.MSPI.Indices.Count; i++) {
                         var mspiIndex = pm4File.MSPI.Indices[i];
                        if (mspiIndex >= mspvVertexCount) {
                            debugWriter.WriteLine($"  WARN: MSPI index {mspiIndex} at pos {i} is out of bounds for MSPV vertex count {mspvVertexCount}");
                            mspiIndicesValid = false;
                        }
                     }
                    debugWriter.WriteLine(mspiIndicesValid ? "MSPI Indices appear valid within MSPV bounds (basic check)." : "MSPI Indices validation logged potential issues.");
                } else {
                     debugWriter.WriteLine("MSPI Chunk not present or MSPV empty, skipping validation.");
                }


                int totalMspiIndices = pm4File.MSPI?.Indices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSUR Index Ranges against MSVI Index Count: {pm4File.MSVI?.Indices.Count ?? 0} ---");
                // MSUR validation (No assertions needed here, just checking ranges during processing)
                debugWriter.WriteLine("MSUR Index Range validation will occur during MSUR processing.");

                // Counters for exported vertices (can be useful for verification)
                int msvtFileVertexCount = 0;
                int mspvFileVertexCount = 0;
                int mprlFileVertexCount = 0; // For the single MPRL file
                int facesWrittenToRenderMesh = 0; // ADDED: Counter for faces written to the render mesh

                // --- 1. Export MSPV vertices -> mspvWriter ONLY ---
                if (exportMspvVertices)
                {
                    mspvWriter.WriteLine("o MSPV_Vertices");
                    if (pm4File.MSPV != null && pm4File.MSPV.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine("\n--- Exporting MSPV Vertices (X, Y, Z) -> _mspv.obj ---");
                        summaryWriter.WriteLine($"\n--- MSPV Vertices ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10) ---");
                        int logCounterMspv = 0;
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            mspvFileVertexCount++;
                            float worldX = vertex.X;
                            float worldY = vertex.Y;
                            float worldZ = vertex.Z;
                            // Reduced verbosity for batch processing debug log
                            // debugWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount - 1}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                            if (logCounterMspv < 10)
                            {
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount - 1}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                                debugWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount - 1}: Exp=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); // Shorter debug log
                            }
                            mspvWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            logCounterMspv++;
                        }
                        if (pm4File.MSPV.Vertices.Count > 10)
                        {
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                            debugWriter.WriteLine("  ... (Debug log limited to first 10) ...");
                        }
                        debugWriter.WriteLine($"Wrote {mspvFileVertexCount} MSPV vertices to _mspv.obj.");
                        mspvWriter.WriteLine();
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nNo MSPV vertex data found."); }
                }
                else
                {
                    debugWriter.WriteLine("\nSkipping MSPV Vertex export (Flag False).");
                }


                // --- 2. Export MSVT vertices (v) -> renderMeshWriter ONLY ---
                if (exportMsvtVertices)
                {
                    renderMeshWriter.WriteLine("o Render_Mesh"); // ADDED: Object name for the combined mesh
                    if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine("\n--- Exporting MSVT Vertices (Y, X, Z) -> _render_mesh.obj ---");
                        summaryWriter.WriteLine($"\n--- MSVT Vertices ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10) -> Render Mesh ---");
                        int logCounterMsvt = 0;
                        foreach (var vertex in pm4File.MSVT.Vertices)
                        {
                            msvtFileVertexCount++;
                            // Apply Y, X, Z transform
                            float worldX = vertex.Y;
                            float worldY = vertex.X;
                            float worldZ = vertex.Z;

                            // Reduced verbosity for batch processing debug log
                            // debugWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount - 1}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported v ({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                            if (logCounterMsvt < 10)
                            {
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount - 1}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported v ({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                                debugWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount - 1}: Exp=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); // Shorter debug log
                            }
                            renderMeshWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            logCounterMsvt++;
                        }
                        if (pm4File.MSVT.Vertices.Count > 10)
                        {
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10 Vertices) ...");
                            debugWriter.WriteLine("  ... (Debug log limited to first 10 Vertices) ...");
                        }
                        debugWriter.WriteLine($"Wrote {msvtFileVertexCount} MSVT vertices (v) to _render_mesh.obj.");
                        renderMeshWriter.WriteLine(); // ADDED: Blank line after vertices in render mesh

                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nNo MSVT vertex data found."); }
                }
                else
                {
                    debugWriter.WriteLine("\nSkipping MSVT Vertex export (Flag False).");
                }


                // --- 3. Export MPRL points with correct (X, -Z, Y) transformation -> mprlWriter ONLY ---
                if (exportMprlPoints)
                {
                    mprlWriter.WriteLine($"# PM4 MPRL Points (X, -Z, Y) - File: {Path.GetFileName(inputFilePath)}"); // Set correct header
                    if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                    {
                        debugWriter.WriteLine($"\n--- Exporting MPRL Vertices with transform (X, -Z, Y) -> _mprl.obj ---");
                        summaryWriter.WriteLine($"\n--- MPRL Vertices ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10) ---");
                        mprlWriter.WriteLine("o MPRL_Points"); // Added object group name
                        mprlFileVertexCount = 0;

                        int logCounterMprl = 0;
                        foreach (var entry in pm4File.MPRL.Entries)
                        {
                            // Apply the confirmed correct transformation: X, -Z, Y
                            float worldX = entry.Position.X;
                            float worldY = -entry.Position.Z; // Use Negated Raw Z for World Y
                            float worldZ = entry.Position.Y;  // Use Raw Y for World Z

                            // Reduced verbosity for batch processing debug log
                            // debugWriter.WriteLine(FormattableString.Invariant(
                            //    $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                            // ));
                            if (logCounterMprl < 10) {
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                                ));
                                debugWriter.WriteLine(FormattableString.Invariant($"  MPRL Vertex {mprlFileVertexCount}: Exp=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); // Shorter debug log
                            }
                            mprlWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            mprlFileVertexCount++;
                            logCounterMprl++;
                        }
                        if (pm4File.MPRL.Entries.Count > 10) {
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                            debugWriter.WriteLine("  ... (Debug log limited to first 10) ...");
                        }
                        debugWriter.WriteLine($"Wrote {mprlFileVertexCount} MPRL vertices to _mprl.obj file.");
                        summaryWriter.WriteLine($"--- Finished MPRL Processing (Exported: {mprlFileVertexCount}) ---");
                        mprlWriter.WriteLine();
                    }
                    else { debugWriter.WriteLine("No MPRL vertex data found. Skipping export."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MPRL export (Flag 'exportMprlPoints' is False).");
                }


                 // --- 4. Process MSCN Data (Output as Points to separate file) ---
                 if (exportMscnPoints) // Use the re-added flag
                 {
                      // Keep disabled code commented out
                      /* ... existing commented MSCN export code ... */
                       debugWriter.WriteLine("\n--- Skipping MSCN point export (Flag 'exportMscnPoints' is currently False in code) ---");
                 } else { // Flag is false
                      debugWriter.WriteLine("\n--- Skipping MSCN point export (Flag 'exportMscnPoints' is False) ---");
                  }


                 // --- 5. Export MSLK paths/points -> mslkWriter ONLY, log skipped to skippedMslkWriter ---
                var mslkHierarchy = new Dictionary<uint, MslkGroupDto>(); // For JSON export

                if (exportMslkPaths)
                {
                    // Ensure mspvFileVertexCount reflects the COUNT written for THIS file
                    // Corrected condition to check MSPV chunk presence and vertex count directly
                    if (pm4File.MSLK != null && pm4File.MSPI != null && pm4File.MSPV != null && pm4File.MSPV.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSLK -> MSPI -> MSPV Chain -> _mslk.obj (Skipped -> _skipped_mslk.log) ---");
                        summaryWriter.WriteLine($"\n--- MSLK Processing ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10 Entries) ---");

                        int entriesToProcessMslk = exportOnlyFirstMslk ? Math.Min(1, pm4File.MSLK.Entries.Count) : pm4File.MSLK.Entries.Count;
                        int skippedMslkCount = 0;
                        int exportedPaths = 0;
                        int exportedPoints = 0;
                        int processedMslkNodes = 0;
                        int mslkLogCounter = 0; // Counter for summary log

                        var msviIndices = pm4File.MSVI?.Indices;
                        var msvtVertices = pm4File.MSVT?.Vertices;
                        int msviCount = msviIndices?.Count ?? 0;
                        int msvtCount = msvtVertices?.Count ?? 0; // Use the loaded count, not exported count here
                        debugWriter.WriteLine($"INFO: Cached MSVI ({msviCount}) and MSVT ({msvtCount}) for node processing.");

                        int currentMspvVertexCount = pm4File.MSPV.Vertices.Count; // Use the actual count from the loaded chunk

                        for (int entryIndex = 0; entryIndex < entriesToProcessMslk; entryIndex++)
                        {
                            var mslkEntry = pm4File.MSLK.Entries[entryIndex];
                            uint groupKey = mslkEntry.Unknown_0x04; // Use Unk04 for grouping

                            if (!mslkHierarchy.ContainsKey(groupKey))
                            {
                                mslkHierarchy[groupKey] = new MslkGroupDto();
                            }

                            bool logSummaryThisEntry = mslkLogCounter < 10;
                            if (logSummaryThisEntry) {
                                summaryWriter.WriteLine($"  Processing MSLK Entry {entryIndex}: GroupKey=0x{groupKey:X8}, FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}, Unk10=0x{mslkEntry.Unknown_0x10:X4}");
                            }
                            // Reduce verbosity in debug log for batch
                            // debugWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            //      $"Processing MSLK Entry {entryIndex}: GroupKey=0x{groupKey:X8}, FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}, Unk00=0x{mslkEntry.Unknown_0x00:X2}, Unk01=0x{mslkEntry.Unknown_0x01:X2}, Unk04=0x{mslkEntry.Unknown_0x04:X8}, Unk10=0x{mslkEntry.Unknown_0x10:X4}, Unk12=0x{mslkEntry.Unknown_0x12:X4}"));

                            if (mslkEntry.MspiIndexCount > 0 && mslkEntry.MspiFirstIndex >= 0) // Geometry Path/Point
                            {
                                int mspiStart = mslkEntry.MspiFirstIndex;
                                int mspiEndExclusive = mspiStart + mslkEntry.MspiIndexCount;

                                if (mspiEndExclusive <= totalMspiIndices)
                                {
                                    List<int> validMspvIndices = new List<int>();
                                    bool invalidIndexDetected = false;
                                    for (int mspiIndex = mspiStart; mspiIndex < mspiEndExclusive; mspiIndex++)
                                    {
                                        uint mspvIndex = pm4File.MSPI!.Indices[mspiIndex];
                                        // Use currentMspvVertexCount for validation
                                        if (mspvIndex < currentMspvVertexCount)
                                        {
                                            validMspvIndices.Add((int)mspvIndex + 1);
                                        }
                                        else
                                        {
                                            if (!invalidIndexDetected) { // Log only first invalid index per entry in debug
                                                 debugWriter.WriteLine($"    WARNING: MSLK Entry {entryIndex}, MSPI index {mspiIndex} points to invalid MSPV index {mspvIndex} (Max: {currentMspvVertexCount - 1}). Skipping vertex.");
                                                 invalidIndexDetected = true;
                                            }
                                            if(logSummaryThisEntry) summaryWriter.WriteLine($"    WARNING: Invalid MSPV index {mspvIndex}. Skipping vertex.");
                                        }
                                    }

                                    if (validMspvIndices.Count >= 2)
                                    {
                                        mslkWriter!.WriteLine($"g MSLK_Path_{entryIndex}_Grp{groupKey:X8}");
                                        mslkWriter!.WriteLine("l " + string.Join(" ", validMspvIndices));
                                        if (logSummaryThisEntry) { // Reduce debug log verbosity
                                            debugWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices.");
                                            summaryWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices.");
                                        }
                                        exportedPaths++;

                                        // Add to JSON hierarchy
                                         mslkHierarchy[groupKey].Geometry.Add(new MslkGeometryEntryDto { /* ... DTO properties ... */ });
                                    }
                                    else if (validMspvIndices.Count == 1)
                                    {
                                        mslkWriter!.WriteLine($"g MSLK_Point_{entryIndex}_Grp{groupKey:X8}");
                                        mslkWriter!.WriteLine($"p {validMspvIndices[0]}");
                                        if (logSummaryThisEntry) { // Reduce debug log verbosity
                                            debugWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]}.");
                                            summaryWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]}. ");
                                        }
                                        exportedPoints++;

                                        // Add to JSON hierarchy
                                        mslkHierarchy[groupKey].Geometry.Add(new MslkGeometryEntryDto { /* ... DTO properties ... */ });
                                    }
                                    else // No valid vertices found
                                    {
                                        if (!invalidIndexDetected) { // If no invalid index was detected, it means the count was 0 or <2 after filtering
                                             debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} resulted in {validMspvIndices.Count} valid MSPV indices after range check. Skipping geometry output.");
                                             if (logSummaryThisEntry) { summaryWriter.WriteLine($"    INFO: Resulted in {validMspvIndices.Count} valid MSPV indices. Skipping."); }
                                        } else {
                                             // If invalid index was already logged, just note skipping
                                             debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} skipped due to invalid MSPV indices.");
                                        }
                                        skippedMslkWriter!.WriteLine($"Skipped ({validMspvIndices.Count} Valid MSPV): {mslkEntry.ToString()}");
                                        skippedMslkCount++;
                                    }
                                }
                                else // Invalid MSPI Range
                                {
                                    debugWriter.WriteLine($"    ERROR: MSLK Entry {entryIndex} defines invalid MSPI range [First:{mspiStart}, Count:{mslkEntry.MspiIndexCount}] (Max MSPI Index: {totalMspiIndices - 1}). Skipping entry.");
                                    if (logSummaryThisEntry) { summaryWriter.WriteLine($"    ERROR: Invalid MSPI range. Skipping entry."); }
                                    skippedMslkWriter!.WriteLine($"Skipped (Invalid MSPI Range): {mslkEntry.ToString()}");
                                    skippedMslkCount++;
                                }
                            }
                            else if (mslkEntry.MspiFirstIndex == -1) // Node Entry (using Unk10 -> MSVI -> MSVT)
                            {
                                 // Add to JSON hierarchy
                                mslkHierarchy[groupKey].Nodes.Add(new MslkNodeEntryDto { /* ... DTO properties ... */ });

                                if (msviIndices != null && msvtVertices != null && msviCount > 0 && msvtCount > 0)
                                {
                                    ushort msviLookupIndex = mslkEntry.Unknown_0x10;
                                    if (msviLookupIndex < msviCount)
                                    {
                                        uint msvtLookupIndex = msviIndices[msviLookupIndex];
                                        if (msvtLookupIndex < msvtCount) // Check against loaded msvt count
                                        {
                                            var msvtVertex = msvtVertices[(int)msvtLookupIndex];

                                            // Apply the MSVT transformation (Y, X, Z) to the anchor point
                                            float worldX = msvtVertex.Y;
                                            float worldY = msvtVertex.X;
                                            float worldZ = msvtVertex.Z;

                                            mslkNodesWriter!.WriteLine($"v {worldX.ToString(CultureInfo.InvariantCulture)} {worldY.ToString(CultureInfo.InvariantCulture)} {worldZ.ToString(CultureInfo.InvariantCulture)} # Node Idx={entryIndex} Grp=0x{groupKey:X8} Unk00=0x{mslkEntry.Unknown_0x00:X2} Unk01=0x{mslkEntry.Unknown_0x01:X2} Unk10={mslkEntry.Unknown_0x10}");
                                            processedMslkNodes++;

                                            if (logSummaryThisEntry) { // Reduce debug log verbosity
                                                 debugWriter.WriteLine($"  MSLK Node Entry {entryIndex}: Unk10={mslkEntry.Unknown_0x10} -> MSVI[{mslkEntry.Unknown_0x10}]={msvtLookupIndex} -> World=({worldX:F3},{worldY:F3},{worldZ:F3}) Grp=0x{groupKey:X8}");
                                                 summaryWriter.WriteLine($"    Processed Node Entry {entryIndex} -> Vertex {processedMslkNodes} in _pm4_mslk_nodes.obj");
                                            }
                                        }
                                        else // Invalid MSVT index
                                        {
                                            debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: MSVI index {msvtLookupIndex} (from MSVI[{mslkEntry.Unknown_0x10}]) is out of bounds for MSVT ({msvtCount}). Skipping node export.");
                                            skippedMslkWriter!.WriteLine($"Node Entry {entryIndex}: Invalid MSVT Index {msvtLookupIndex} (from MSVI[{mslkEntry.Unknown_0x10}]) for MSVT Count {msvtCount}. Grp=0x{groupKey:X8} Unk10={mslkEntry.Unknown_0x10}");
                                            skippedMslkCount++; // Also count this as skipped
                                        }
                                    }
                                    else // Invalid MSVI index
                                    {
                                        debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: Unk10 index {mslkEntry.Unknown_0x10} is out of bounds for MSVI ({msviCount}). Skipping node export.");
                                        skippedMslkWriter!.WriteLine($"Node Entry {entryIndex}: Invalid MSVI Index {mslkEntry.Unknown_0x10} for MSVI Count {msviCount}. Grp=0x{groupKey:X8}");
                                        skippedMslkCount++; // Also count this as skipped
                                    }
                                }
                                else // MSVI/MSVT data missing
                                {
                                    debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: MSVI or MSVT data is missing or empty. Cannot process node anchor.");
                                    skippedMslkWriter!.WriteLine($"Node Entry {entryIndex}: Missing MSVI/MSVT data. Cannot process node. Grp=0x{groupKey:X8}");
                                     skippedMslkCount++; // Also count this as skipped
                                }
                            }
                            else // Neither Geometry nor Node (based on MspiFirstIndex/Count)
                            {
                                debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} has MSPICount={mslkEntry.MspiIndexCount}, MSPIFirst={mslkEntry.MspiFirstIndex}. Skipping geometry/node export.");
                                if (logSummaryThisEntry) { summaryWriter.WriteLine($"    INFO: Not a geometry or node entry. Skipping."); }
                                skippedMslkWriter!.WriteLine($"Skipped (Not Geometry or Node): {mslkEntry.ToString()}");
                                skippedMslkCount++;
                            }
                             mslkLogCounter++; // Increment summary log counter
                        } // End MSLK entry loop

                        if (mslkLogCounter > 10) { // Add ellipsis if summary was truncated
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10 MSLK entries) ...");
                        }

                        debugWriter.WriteLine($"Finished processing {entriesToProcessMslk} MSLK entries. Exported {exportedPaths} paths, {exportedPoints} points, {processedMslkNodes} node anchors. Skipped {skippedMslkCount} entries.");
                        if (exportOnlyFirstMslk && pm4File.MSLK.Entries.Count > 1)
                        {
                            debugWriter.WriteLine("Note: MSLK processing was limited to the first entry by 'exportOnlyFirstMslk' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSLK Processing (Exported Paths: {exportedPaths}, Points: {exportedPoints}, Nodes: {processedMslkNodes}, Skipped: {skippedMslkCount}) ---");
                        mslkWriter!.WriteLine();
                        debugWriter.Flush();
                        skippedMslkWriter!.Flush();
                        mslkNodesWriter!.Flush(); // Ensure nodes file is written

                        // --- Export MSLK Hierarchy to JSON ---
                        debugWriter.WriteLine($"\n--- Exporting MSLK Hierarchy to {outputMslkJsonPath} ---");
                        try
                        {
                            var options = new JsonSerializerOptions { WriteIndented = true };
                            string jsonString = JsonSerializer.Serialize(mslkHierarchy, options);
                            File.WriteAllText(outputMslkJsonPath, jsonString);
                            debugWriter.WriteLine($"Successfully exported MSLK hierarchy for {mslkHierarchy.Count} groups.");
                            summaryWriter.WriteLine($"--- Exported MSLK hierarchy for {mslkHierarchy.Count} groups to JSON. ---");
                        }
                        catch (Exception ex)
                        {
                            debugWriter.WriteLine($"ERROR exporting MSLK hierarchy to JSON: {ex.Message}");
                            summaryWriter.WriteLine($"ERROR exporting MSLK hierarchy to JSON: {ex.Message}");
                        }
                    }
                    else // Required chunks missing
                    {
                        debugWriter.WriteLine("Skipping MSLK path/node export (MSLK, MSPI, or MSPV data missing or invalid).");
                    }
                } else { // Flag false
                    debugWriter.WriteLine("\nSkipping MSLK Path/Node/JSON export (Flag 'exportMslkPaths' is False).");
                }



                // --- 6. Export MSUR surfaces as faces -> renderMeshWriter ONLY ---
                if (processMsurEntries)
                {
                     // Ensure msvtFileVertexCount reflects the count written for THIS file
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null && pm4File.MDOS != null && pm4File.MDSF != null && msvtFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSUR -> MDSF -> MDOS Links (Adding faces to _render_mesh.obj) ---");
                        summaryWriter.WriteLine($"\n--- MSUR -> MDSF -> MDOS Links ({Path.GetFileNameWithoutExtension(inputFilePath)}) (Summary Log - First 20 Entries) -> Render Mesh ---");

                        int msurEntriesProcessed = 0;
                        int entriesToProcess = exportOnlyFirstMsur ? Math.Min(1, pm4File.MSUR.Entries.Count) : pm4File.MSUR.Entries.Count;
                        int msurLogCounter = 0; // Counter for summary log

                        // Pre-build a lookup for faster MDSF searching (if MDSF is not null)
                        var mdsfLookup = pm4File.MDSF.Entries?.ToDictionary(e => e.msur_index, e => e.mdos_index)
                                         ?? new Dictionary<uint, uint>(); // Empty dict if MDSF is null
                        debugWriter.WriteLine($"  Built MDSF lookup dictionary with {mdsfLookup.Count} entries.");

                        int currentMsviCount = pm4File.MSVI?.Indices.Count ?? 0; // Get current MSVI count

                        for (uint msurIndex = 0; msurIndex < entriesToProcess; msurIndex++) // Use uint for lookup consistency
                        {
                            var msurEntry = pm4File.MSUR.Entries[(int)msurIndex];
                            bool logSummary = msurLogCounter < 20; // Limit summary logging
                            string groupName = string.Empty; // Declare groupName once here

                            // Reduced verbosity in debug log
                            // debugWriter.WriteLine($"  Processing MSUR Entry {msurIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, FlagsOrUnk0=0x{msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}");
                            if (logSummary) {
                                 summaryWriter.WriteLine($"  Processing MSUR Entry {msurIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, FlagsOrUnk0=0x{msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}");
                                 debugWriter.WriteLine($"  Processing MSUR Entry {msurIndex}: MsviFirst={msurEntry.MsviFirstIndex}, Count={msurEntry.IndexCount}"); // Shorter debug
                            }


                            int firstIndex = (int)msurEntry.MsviFirstIndex;
                            int indexCount = msurEntry.IndexCount;


                            // Find corresponding MDOS index via MDSF
                            if (!mdsfLookup.TryGetValue(msurIndex, out uint linkedMdosIndex))
                            {
                                // CASE 1: No MDSF Link Found - Assume default state (0) and include face
                                if (logSummary) { // Reduce debug verbosity
                                     debugWriter.WriteLine($"    WARN: No MDSF entry found linking to MSUR index {msurIndex}. Assuming default state (0).");
                                     summaryWriter.WriteLine($"    WARN: No MDSF link found. Assuming default state (0) and including face.");
                                }


                                // *** BEGIN: Write face logic for unlinked MSUR ***
                                if (pm4File.MSVI == null) {
                                     if (logSummary) debugWriter.WriteLine("    Skipping face (unlinked): MSVI chunk missing.");
                                     continue;
                                }

                                // Check MSVI range validity using currentMsviCount
                                if (firstIndex >= 0 && firstIndex + indexCount <= currentMsviCount)
                                {
                                     if (indexCount < 3) { // Need at least 3 vertices for a face
                                          if (logSummary) {
                                              debugWriter.WriteLine($"    Skipping face (unlinked): Not enough indices (Count={indexCount} < 3).");
                                              summaryWriter.WriteLine($"    Skipping face (unlinked): Not enough indices (Count={indexCount} < 3).");
                                          }
                                         continue;
                                     }

                                    List<uint> msviIndicesForFace = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                                    // Validate MSVI indices point within the *EXPORTED* MSVT vertex bounds (msvtFileVertexCount)
                                    List<int> objFaceIndices = new List<int>();
                                    bool invalidMsvtIndex = false;
                                    foreach (uint msviIdx in msviIndicesForFace)
                                    {
                                        if (msviIdx >= msvtFileVertexCount) {
                                             if (logSummary) debugWriter.WriteLine($"    ERROR (unlinked): MSUR Entry {msurIndex} -> MSVI index {msviIdx} points beyond exported MSVT vertex count ({msvtFileVertexCount}). Skipping face.");
                                             if (logSummary) summaryWriter.WriteLine($"    ERROR (unlinked): Invalid MSVT index {msviIdx} detected. Skipping face.");
                                             invalidMsvtIndex = true;
                                             break; // Stop processing this face
                                        }
                                         objFaceIndices.Add((int)msviIdx + 1); // Convert to 1-based OBJ index
                                    }

                                     if (invalidMsvtIndex) continue; // Skip this face

                                    // Debug logging for indices (reduced verbosity)
                                    // debugWriter.WriteLine($"    Fetched {msviIndicesForFace.Count} MSVI Indices (Expected {indexCount}) for unlinked face.");
                                    // debugWriter.WriteLine($"      Debug 0-based MSVI Indices (unlinked): [{string.Join(", ", msviIndicesForFace)}]");
                                    // debugWriter.WriteLine($"      Debug 1-based OBJ Indices (unlinked): [{string.Join(", ", objFaceIndices)}]");


                                    // Write the face since it's assumed to be state 0
                                    groupName = $"MSUR{msurIndex}_UnlinkedState0"; // Assign specific group name
                                    string faceLine = "f " + string.Join(" ", objFaceIndices);
                                    renderMeshWriter!.WriteLine($"g {groupName}");
                                    renderMeshWriter!.WriteLine(faceLine);
                                    facesWrittenToRenderMesh++;
                                }
                                else // Invalid MSVI range
                                {
                                     if (logSummary) { // Reduce debug verbosity
                                         debugWriter.WriteLine($"    ERROR (unlinked): MSUR Entry {msurIndex} defines invalid MSVI range [First:{firstIndex}, Count:{indexCount}] (Max MSVI Index: {currentMsviCount - 1}). Skipping face.");
                                         summaryWriter.WriteLine($"    ERROR (unlinked): Invalid MSVI range. Skipping face.");
                                     }
                                }
                                // *** END: Write face logic for unlinked MSUR ***
                                 msurEntriesProcessed++; // Count processed entry
                                 msurLogCounter++; // Increment summary log counter
                                continue; // Skip the linked MDOS processing below
                            }

                            // CASE 2: MDSF Link Found - Proceed with original logic
                            // Reduce debug verbosity
                            // debugWriter.WriteLine($"    Found MDSF Link: MSUR[{msurIndex}] -> MDOS Index {linkedMdosIndex}");

                            // Get the correctly linked MDOS entry
                            // Ensure MDOS chunk and entries are not null
                            if (pm4File.MDOS?.Entries == null || linkedMdosIndex >= pm4File.MDOS.Entries.Count)
                            {
                                 if (logSummary) { // Reduce debug verbosity
                                    debugWriter.WriteLine($"    ERROR: MDSF links to invalid MDOS index {linkedMdosIndex} (Max: {mdosEntryCount - 1}) or MDOS missing. Skipping face.");
                                    summaryWriter.WriteLine($"    ERROR: Invalid MDOS link via MDSF ({linkedMdosIndex}) or MDOS missing. Skipping face.");
                                 }
                                 msurEntriesProcessed++; // Count processed entry
                                 msurLogCounter++; // Increment summary log counter
                                continue;
                            }
                            var linkedMdosEntry = pm4File.MDOS!.Entries[(int)linkedMdosIndex];
                            uint buildingId = linkedMdosEntry.m_destructible_building_index;
                            groupName = $"MSUR{msurIndex}_MDSF_MDOS{linkedMdosIndex}_BldID{buildingId:X8}_State{linkedMdosEntry.destruction_state}"; // Assign group name here

                            if (logSummary) { // Reduce debug verbosity
                                 // debugWriter.WriteLine($"      -> Linked MDOS Entry: {linkedMdosEntry}");
                                 summaryWriter.WriteLine($"      -> Linked MDOS Entry: State={linkedMdosEntry.destruction_state}, BldID=0x{buildingId:X8}");
                                 debugWriter.WriteLine($"      -> Linked MDOS: State={linkedMdosEntry.destruction_state}, BldID=0x{buildingId:X8}"); // Shorter debug
                            }

                            if (buildingId != 0) { uniqueBuildingIds.Add(buildingId); } // Collect unique IDs

                             // Check MSVI presence again (belt and suspenders)
                            if (pm4File.MSVI == null) {
                                 if (logSummary) debugWriter.WriteLine("    Skipping face (linked): MSVI chunk missing.");
                                continue;
                            }

                            // Check MSVI range validity using currentMsviCount
                            if (firstIndex >= 0 && firstIndex + indexCount <= currentMsviCount)
                            {
                                 if (indexCount < 3) { // Need at least 3 vertices for a face
                                     if (logSummary) { // Reduce debug verbosity
                                         debugWriter.WriteLine($"    Skipping face generation: Not enough indices (Count={indexCount} < 3).");
                                         summaryWriter.WriteLine($"    Skipping face: Not enough indices (Count={indexCount} < 3).");
                                     }
                                     continue;
                                 }

                                List<uint> msviIndicesForFace = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                                // Validate MSVI indices point within the *EXPORTED* MSVT vertex bounds (msvtFileVertexCount)
                                List<int> objFaceIndices = new List<int>();
                                bool invalidMsvtIndex = false;
                                foreach (uint msviIdx in msviIndicesForFace)
                                {
                                    if (msviIdx >= msvtFileVertexCount) {
                                         if (logSummary) debugWriter.WriteLine($"    ERROR (linked): MSUR Entry {msurIndex} -> MSVI index {msviIdx} points beyond exported MSVT vertex count ({msvtFileVertexCount}). Skipping face.");
                                         if (logSummary) summaryWriter.WriteLine($"    ERROR (linked): Invalid MSVT index {msviIdx} detected. Skipping face.");
                                         invalidMsvtIndex = true;
                                         break; // Stop processing this face
                                    }
                                    objFaceIndices.Add((int)msviIdx + 1); // Convert to 1-based OBJ index
                                }

                                if (invalidMsvtIndex) continue; // Skip this face

                                // Reduced verbosity
                                // debugWriter.WriteLine($"    Fetched {msviIndicesForFace.Count} MSVI Indices (Expected {indexCount}).");
                                // debugWriter.WriteLine($"      Debug 0-based MSVI Indices: [{string.Join(", ", msviIndicesForFace)}]");
                                // debugWriter.WriteLine($"      Debug 1-based OBJ Indices: [{string.Join(", ", objFaceIndices)}]");


                                // Filter based on the correctly linked MDOS state
                                if (linkedMdosEntry.destruction_state == 0)
                                {
                                    // Construct the face line using only vertex indices
                                    string faceLine = "f " + string.Join(" ", objFaceIndices);
                                    renderMeshWriter!.WriteLine($"g {groupName}"); // Group by MSUR index and linked MDOS info via MDSF
                                    renderMeshWriter!.WriteLine(faceLine);
                                    facesWrittenToRenderMesh++; // Ensure this is INSIDE the if(State == 0) block
                                }
                                else // State != 0
                                {
                                    if (logSummary) { // Reduce debug verbosity
                                        debugWriter.WriteLine($"    -> SKIPPING Face (MDOS State != 0)");
                                        summaryWriter.WriteLine($"    -> SKIPPING Face (MDOS State != 0)");
                                    }
                                }
                            }
                            else // Invalid MSVI range
                            {
                                if (logSummary) { // Reduce debug verbosity
                                    debugWriter.WriteLine($"    ERROR: MSUR Entry {msurIndex} defines invalid MSVI range [First:{firstIndex}, Count:{indexCount}] (Max MSVI Index: {currentMsviCount - 1}). Skipping face.");
                                    summaryWriter.WriteLine($"    ERROR: Invalid MSVI range. Skipping face.");
                                }
                            }
                             msurEntriesProcessed++;
                             msurLogCounter++; // Increment summary log counter
                        } // End MSUR loop

                        if (msurLogCounter > 20) { // Add ellipsis if summary truncated
                             summaryWriter.WriteLine("  ... (Summary log limited to first 20 MSUR entries) ...");
                        }

                        // CORRECTED LOG MESSAGES
                        debugWriter.WriteLine($"Finished processing {msurEntriesProcessed} MSUR entries. Wrote {facesWrittenToRenderMesh} faces to _render_mesh.obj.");
                        if (exportOnlyFirstMsur && pm4File.MSUR.Entries.Count > 1) {
                             debugWriter.WriteLine("Note: MSUR processing was limited to the first entry by 'exportOnlyFirstMsur' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSUR Processing (Processed: {msurEntriesProcessed}, Faces Written: {facesWrittenToRenderMesh}) ---");
                    }
                    else // Required chunks missing or no vertices exported
                    {
                        debugWriter.WriteLine("Skipping MSUR face export (MSUR, MSVI, MSVT, MDOS, or MDSF data missing or no MSVT vertices exported).");
                    }
                } else { // Flag false
                     debugWriter.WriteLine("\nSkipping MSUR Face processing (Flag 'processMsurEntries' is False).");
                }


                // --- 7. Log Unique Building IDs ---
                if (uniqueBuildingIds.Count > 0)
                {
                    debugWriter.WriteLine($"\n--- Logging {uniqueBuildingIds.Count} Unique Building IDs ---");
                    summaryWriter.WriteLine($"\n--- Found {uniqueBuildingIds.Count} Unique Building IDs (See {Path.GetFileName(outputBuildingIdsPath)}) ---");
                    buildingIdWriter!.WriteLine($"Found {uniqueBuildingIds.Count} unique non-zero building IDs linked via MDOS for file {Path.GetFileName(inputFilePath)}:");
                    foreach (uint id in uniqueBuildingIds.OrderBy(id => id))
                    {
                        buildingIdWriter.WriteLine($"0x{id:X8} ({id})");
                    }
                    debugWriter.WriteLine($"Logged {uniqueBuildingIds.Count} unique building IDs to {Path.GetFileName(outputBuildingIdsPath)}.");
                }
                else
                {
                     debugWriter.WriteLine("\n--- No unique non-zero building IDs found or logged. ---");
                     summaryWriter.WriteLine("\n--- No unique non-zero building IDs found. ---");
                }


                // --- 8. Log MDBH and MDSF Link Info ---
                 if (logMdsfLinks && pm4File.MDSF?.Entries != null && pm4File.MDOS?.Entries != null) // Null checks added
                 {
                     debugWriter.WriteLine("\n--- Logging MDSF -> MDOS Link Information ---");
                     summaryWriter.WriteLine($"\n--- MDSF -> MDOS Link Summary ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 20) ---");
                     int mdsfCount = 0;
                     int mdsfLogCounter = 0;
                     foreach(var mdsfEntry in pm4File.MDSF.Entries)
                     {
                         uint mdosIdx = mdsfEntry.mdos_index;
                         string linkInfo;
                         if (mdosIdx < (pm4File.MDOS?.Entries.Count ?? 0)) { // Check against actual count
                             var linkedMdosEntry = pm4File.MDOS!.Entries[(int)mdosIdx];
                             linkInfo = $"MDSF Entry (MSUR:{mdsfEntry.msur_index}) -> MDOS Index {mdosIdx} -> MDOS Entry [ID:0x{linkedMdosEntry.m_destructible_building_index:X8}, State:{linkedMdosEntry.destruction_state}]";
                         } else {
                              linkInfo = $"MDSF Entry (MSUR:{mdsfEntry.msur_index}) -> Invalid MDOS Index {mdosIdx}";
                         }

                         if (mdsfLogCounter < 20) { // Reduce debug verbosity
                             debugWriter.WriteLine($"  {linkInfo}");
                             summaryWriter.WriteLine($"  {linkInfo}");
                             mdsfLogCounter++;
                         }
                         mdsfCount++;
                     }
                      if (mdsfCount > 20) {
                         summaryWriter.WriteLine("  ... (Summary log limited to first 20 MDSF entries) ...");
                         debugWriter.WriteLine($"  ... (Debug log limited to first 20 MDSF entries) ...");
                     }
                     debugWriter.WriteLine($"Logged info for {mdsfCount} MDSF entries.");
                     summaryWriter.WriteLine($"--- Finished logging MDSF links ({mdsfCount} total). ---");
                 } else {
                      debugWriter.WriteLine("\n--- Skipping MDSF/MDOS Link Logging (Flag 'logMdsfLinks' is False or data missing) ---");
                 }


                 // MDBH Logging (Optional - if needed for context)
                 if (pm4File.MDBH?.Entries != null) { // Null check added
                      debugWriter.WriteLine("\n--- Logging MDBH Entries (First 10) ---");
                      summaryWriter.WriteLine($"\n--- MDBH Entries ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10) ---");
                      int mdbhCount = 0;
                      int mdbhLogCounter = 0;
                      foreach(var mdbhEntry in pm4File.MDBH.Entries) {
                           if (mdbhLogCounter < 10) { // Reduce debug verbosity
                               // Use correct property names from MdbhEntry class: Index, Filename
                               debugWriter.WriteLine($"  MDBH Entry {mdbhCount}: Index={mdbhEntry.Index}, Filename=\"{mdbhEntry.Filename}\"");
                               summaryWriter.WriteLine($"  MDBH Entry {mdbhCount}: Index={mdbhEntry.Index}, Filename=\"{mdbhEntry.Filename}\"");
                               mdbhLogCounter++;
                          }
                          mdbhCount++;
                      }
                       if (mdbhCount > 10) {
                         summaryWriter.WriteLine("  ... (Summary log limited to first 10 MDBH entries) ...");
                         debugWriter.WriteLine("  ... (Debug log limited to first 10 MDBH entries) ...");
                     }
                 } else {
                      debugWriter.WriteLine("\n--- Skipping MDBH Logging (Data missing) ---");
                 }

                summaryWriter.WriteLine($"--- End PM4 File Summary for {Path.GetFileName(inputFilePath)} ---");
                debugWriter.WriteLine($"--- End PM4 File Debug Log for {Path.GetFileName(inputFilePath)} ---");
            }
            catch (Exception ex) // Catch exceptions within the using block
            {
                 Console.WriteLine($"ERROR during processing logic for {Path.GetFileName(inputFilePath)}: {ex.Message}");
                 debugWriter?.WriteLine($"\n!!!!!! ERROR during processing: {ex.ToString()} !!!!!!"); // Log to file-specific debug log
                 summaryWriter?.WriteLine($"\n!!!!!! ERROR during processing: {ex.Message} !!!!!!"); // Log to file-specific summary log
                // Re-throw the exception to be caught by the outer loop's handler
                throw;
            }
            // No finally needed here as 'using' handles disposal

        } // End ProcessSinglePm4File

        // Original LoadPM4File_ShouldLoadChunks can be removed or commented out
        /*
        [Fact]
        public void LoadPM4File_ShouldLoadChunks_OLD()
        {
            // ... original single-file test code ...
        }
        */

    } // End PM4FileTests class
} // End namespace
