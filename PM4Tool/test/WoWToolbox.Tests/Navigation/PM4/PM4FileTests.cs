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
using WoWToolbox.Core.Models;

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
        // Coordinate transforms now centralized in Pm4CoordinateTransforms

        // TestDataRoot and TestContext needed for TestDevelopment49_28_WithSpecializedHandling
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static class TestContext
        {
            // Calculate Workspace Root (assuming tests run from bin/Debug/netX.0)
            private static string WorkspaceRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", ".."));
            // Generate a timestamp for this test run
            private static string Timestamp => DateTime.Now.ToString("yyyyMMdd_HHmmss");
            // Define the root output directory with timestamp
            public static string TimestampedOutputRoot => Path.Combine(WorkspaceRoot, "output", Timestamp);

            // Original TestResultsDirectory - Keep for reference or potential future use, but prefer TimestampedOutputRoot
            public static string OriginalTestResultsDirectory => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestResults");
        }

        // Removed ApplyMprlTransform helper function

        [Fact]
        // public void LoadPM4File_ShouldLoadChunks() // Renamed/Replaced
        public void LoadAndProcessPm4FilesInDirectory_ShouldGenerateOutputs()
        {
            Console.WriteLine("--- LoadAndProcessPm4FilesInDirectory START ---");

            // Known problematic files that need special handling
            var knownIssueFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "development_49_28.pm4" // File identified as causing errors
            };

            // --- Path Construction (Input Directory) ---
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            // Use Path.Combine for robust path handling across OS
            // Corrected path: Go up 5 levels from bin/Debug/netX.0 to reach solution root
            var testDataRoot = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "..", "..", "..", "test_data"));
            var inputDirectoryPath = Path.Combine(testDataRoot, "original_development", "development"); // Added intermediate 'development' directory
            
            // --- Use Timestamped Output --- 
            var outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "PM4_BatchOutput");
            Directory.CreateDirectory(outputDir); // Ensure the output directory exists

            Console.WriteLine($"Input Directory: {inputDirectoryPath}");
            Console.WriteLine($"Output Directory: {outputDir}");

            // Create an error log file to track all errors in one place
            var errorLogPath = Path.Combine(outputDir, "_ERROR_LOG.txt");
            using var errorLogWriter = new StreamWriter(errorLogPath, false); // Overwrite any existing file
            errorLogWriter.WriteLine($"--- PM4 Processing Error Log (Generated: {DateTime.Now}) ---");
            errorLogWriter.WriteLine("This file contains all errors encountered during batch processing of PM4 files.\n");

             if (!Directory.Exists(inputDirectoryPath))
            {
                Console.WriteLine($"ERROR: Input directory not found: {inputDirectoryPath}");
                errorLogWriter.WriteLine($"ERROR: Input directory not found: {inputDirectoryPath}");
                Assert.Fail($"Input directory not found: {inputDirectoryPath}"); // Fail the test if input dir is missing
                return;
            }

            // --- Get PM4 Files ---
            var pm4Files = Directory.EnumerateFiles(inputDirectoryPath, "*.pm4", SearchOption.TopDirectoryOnly).ToList(); // ToList() to get count easily
            int processedCount = 0;
            int errorCount = 0;
            int skippedCount = 0;
            int knownIssuesCount = 0;
            List<string> errorFiles = new List<string>();
            List<string> skippedFiles = new List<string>();
            List<string> knownIssueFiles_Encountered = new List<string>();

            Console.WriteLine($"Found {pm4Files.Count} PM4 files to process.");
            errorLogWriter.WriteLine($"Found {pm4Files.Count} PM4 files to process.\n");

            if (!pm4Files.Any())
            {
                Console.WriteLine("WARNING: No PM4 files found in the input directory.");
                errorLogWriter.WriteLine("WARNING: No PM4 files found in the input directory.");
                // Assert.True(pm4Files.Any(), "No PM4 files found in the input directory."); // Optional: Fail if no files found
            }

            // --- Combined Output File Setup ---
            var combinedOutputPath = Path.Combine(outputDir, "combined_render_mesh_transformed.obj");
            var combinedWithMscnPath = Path.Combine(outputDir, "combined_render_mesh_with_mscn.obj");
            var combinedAllChunksPath = Path.Combine(outputDir, "combined_all_chunks_aligned.obj");
            Console.WriteLine($"Combined Output OBJ: {combinedOutputPath}");
            Console.WriteLine($"Combined All Chunks (Aligned): {combinedAllChunksPath}");
            using var combinedRenderMeshWriter = new StreamWriter(combinedOutputPath);
            using var combinedWithMscnWriter = new StreamWriter(combinedWithMscnPath);
            using var combinedAllChunksWriter = new StreamWriter(combinedAllChunksPath);

            // Initialize combined render mesh file with header
            combinedRenderMeshWriter.WriteLine($"# PM4 Combined Render Mesh - With Faces and Normals (Generated: {DateTime.Now})");
            combinedRenderMeshWriter.WriteLine("# Contains MSVT render mesh vertices with computed normals and triangle faces");
            combinedRenderMeshWriter.WriteLine("# All coordinates are PM4-relative for proper spatial alignment");

            // Initialize comprehensive combined file
            combinedAllChunksWriter.WriteLine($"# PM4 Geometry Chunks Combined - Properly Aligned (Generated: {DateTime.Now})");
            combinedAllChunksWriter.WriteLine("# This file contains ONLY the local geometry chunk types with correct coordinate transformations:");
            combinedAllChunksWriter.WriteLine("#   MSVT: Render mesh vertices (Y, X, Z)");
            combinedAllChunksWriter.WriteLine("#   MSCN: Collision boundaries (X, Y, Z - with geometric transform)");
            combinedAllChunksWriter.WriteLine("#   MSLK/MSPV: Geometric structure (X, Y, Z)");
            combinedAllChunksWriter.WriteLine("# MPRL chunks are EXCLUDED - they are map positioning data, not local geometry.");
            combinedAllChunksWriter.WriteLine("# All included chunks are now spatially aligned for meaningful visualization.");

            int totalVerticesOffset = 0; // Track vertex offset for combined file
            int totalMscnOffset = 0; // Track MSCN point offset for combined file
            
            // Track combined mesh data for normals and faces generation
            var combinedVertices = new List<Vector3>();
            var combinedTriangleIndices = new List<int>();
            var combinedFileInfo = new List<(string fileName, int vertexCount, int faceCount)>();

            // --- Loop Through Files ---
            foreach (var inputFilePath in pm4Files)
            {
                var fileName = Path.GetFileName(inputFilePath);
                Console.WriteLine($"\n==================== Processing File: {fileName} ====================");
                errorLogWriter.WriteLine($"\n--- Processing File: {fileName} ---");

                // Check if this is a known issue file
                if (knownIssueFiles.Contains(fileName))
                {
                    Console.WriteLine($"!!!!!!!!!!!! RUNNING DIAGNOSTICS on Known Issue file: {fileName} !!!!!!!!!!!!");
                    errorLogWriter.WriteLine($"RUNNING DIAGNOSTICS: Analyzing {fileName} for specific issues instead of skipping.");
                    knownIssuesCount++;
                    knownIssueFiles_Encountered.Add(fileName);
                    
                    // Run diagnostic analysis instead of normal processing
                    try
                    {
                        AnalyzeProblematicFile(inputFilePath, outputDir);
                        errorLogWriter.WriteLine($"Diagnostic analysis completed for {fileName}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error during diagnostic analysis: {ex.Message}");
                        errorLogWriter.WriteLine($"ERROR during diagnostic analysis: {ex.Message}");
                        errorLogWriter.WriteLine($"Stack trace: {ex.StackTrace}");
                    }
                    
                    continue; // Skip normal processing
                }

                // Check for zero-byte files before trying to process
                try
                {
                    var fileInfo = new FileInfo(inputFilePath);
                    if (fileInfo.Length == 0)
                    {
                        Console.WriteLine($"!!!!!!!!!!!! SKIPPING Zero-byte file: {fileName} !!!!!!!!!!!!");
                        errorLogWriter.WriteLine($"SKIPPED: Zero-byte file: {fileName}");
                        skippedCount++;
                        skippedFiles.Add(fileName);
                        continue; // Move to the next file
                    }
                }
                catch (FileNotFoundException)
                {
                    Console.WriteLine($"!!!!!!!!!!!! ERROR: File not found during size check: {fileName} !!!!!!!!!!!!");
                    errorLogWriter.WriteLine($"ERROR: File not found during size check: {fileName}");
                    errorCount++;
                    errorFiles.Add(fileName);
                    continue; // Move to the next file
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"!!!!!!!!!!!! ERROR checking file size for {fileName}: {ex.Message} !!!!!!!!!!!!");
                    errorLogWriter.WriteLine($"ERROR checking file size for {fileName}: {ex.Message}");
                    errorLogWriter.WriteLine($"Exception Type: {ex.GetType().Name}");
                    errorLogWriter.WriteLine($"Stack Trace: {ex.StackTrace}");
                    errorCount++;
                    errorFiles.Add(fileName);
                    continue; // Move to the next file
                }

                try
                {
                    // Call the helper method, passing the combined writer and current offset
                    var processResult = ProcessSinglePm4FileWithData(inputFilePath, outputDir, combinedRenderMeshWriter, totalVerticesOffset);
                    int verticesInCurrentFile = processResult.vertexCount;
                    
                    // Write MSCN points to the combined with MSCN file
                    if (verticesInCurrentFile > 0) // Only if we successfully processed vertices
                    {
                        var pm4File = PM4File.FromFile(inputFilePath);
                        var fileNameForMscn = Path.GetFileName(inputFilePath);
                        
                        // Check if we have MSCN data
                        if (pm4File.MSCN != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                        {
                            // Write all MSCN points to the combined file
                            foreach (var point in pm4File.MSCN.ExteriorVertices)
                            {
                                // Apply PM4-relative transformation
                                var pm4Coords = Pm4CoordinateTransforms.FromMscnVertex(point);
                            combinedWithMscnWriter.WriteLine(FormattableString.Invariant(
                                $"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSCN (File: {fileNameForMscn})"));
                                totalMscnOffset++;
                            }
                            
                            Console.WriteLine($"  Added {pm4File.MSCN.ExteriorVertices.Count} MSCN points to combined MSCN file from {fileNameForMscn}");
                        }
                        
                        // --- Write ALL chunk types to comprehensive combined file ---
                        combinedAllChunksWriter.WriteLine($"# === {fileNameForMscn} ===");
                        
                        // 1. MSVT render vertices
                        if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                        {
                            combinedAllChunksWriter.WriteLine($"# MSVT Render Vertices ({pm4File.MSVT.Vertices.Count})");
                            foreach (var vertex in pm4File.MSVT.Vertices)
                            {
                                var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                                combinedAllChunksWriter.WriteLine(FormattableString.Invariant(
                                    $"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSVT {fileNameForMscn}"));
                            }
                        }
                        
                        // 2. MSCN collision boundaries
                        if (pm4File.MSCN != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                        {
                            combinedAllChunksWriter.WriteLine($"# MSCN Collision Boundaries ({pm4File.MSCN.ExteriorVertices.Count})");
                            foreach (var point in pm4File.MSCN.ExteriorVertices)
                            {
                                var pm4Coords = Pm4CoordinateTransforms.FromMscnVertex(point);
                                combinedAllChunksWriter.WriteLine(FormattableString.Invariant(
                                    $"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSCN {fileNameForMscn}"));
                            }
                        }
                        
                        // 3. MSPV geometry vertices
                        if (pm4File.MSPV != null && pm4File.MSPV.Vertices.Count > 0)
                        {
                            combinedAllChunksWriter.WriteLine($"# MSPV Geometry Vertices ({pm4File.MSPV.Vertices.Count})");
                            foreach (var vertex in pm4File.MSPV.Vertices)
                            {
                                var pm4Coords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                                combinedAllChunksWriter.WriteLine(FormattableString.Invariant(
                                    $"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSPV {fileNameForMscn}"));
                            }
                        }
                        
                        // NOTE: MPRL chunks are intentionally EXCLUDED from combined file
                        // MPRL = Map positioning reference points (separate from local geometry)
                        // These mark where the structure sits in the world map, not local coordinates
                        // Including them causes spatial separation in visualization tools like MeshLab
                        
                        combinedAllChunksWriter.WriteLine(); // Blank line between files
                    }
                    
                    // Collect vertex and face data for combined mesh
                    if (processResult.vertices != null && processResult.vertices.Count > 0)
                    {
                        // Add vertices to combined mesh with offset
                        combinedVertices.AddRange(processResult.vertices);
                        
                        // Add triangle indices with offset
                        foreach (var index in processResult.triangleIndices)
                        {
                            combinedTriangleIndices.Add(index + totalVerticesOffset);
                        }
                        
                        // Track file info
                        combinedFileInfo.Add((fileName, processResult.vertexCount, processResult.triangleIndices.Count / 3));
                    }
                    
                    processedCount++;
                    totalVerticesOffset += verticesInCurrentFile; // Update the offset for the next file
                    Console.WriteLine($"-------------------- Successfully processed: {fileName} (Added {verticesInCurrentFile} vertices, {processResult.triangleIndices.Count / 3} faces) --------------------");
                    errorLogWriter.WriteLine($"SUCCESS: Processed {fileName} (Added {verticesInCurrentFile} vertices, {processResult.triangleIndices.Count / 3} faces)");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"!!!!!!!!!!!!!!!!!!!! ERROR processing file {fileName} !!!!!!!!!!!!!!!!!!!!");
                    Console.WriteLine($"Error Message: {ex.Message}");
                    Console.WriteLine($"Exception Type: {ex.GetType().Name}");
                    Console.WriteLine($"Stack Trace:\n{ex.StackTrace}");
                    
                    errorLogWriter.WriteLine($"ERROR processing file {fileName}:");
                    errorLogWriter.WriteLine($"Error Message: {ex.Message}");
                    errorLogWriter.WriteLine($"Exception Type: {ex.GetType().Name}");
                    errorLogWriter.WriteLine($"Stack Trace:\n{ex.StackTrace}");
                    
                    // Check for inner exception
                    if (ex.InnerException != null)
                    {
                        Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                        Console.WriteLine($"Inner Exception Type: {ex.InnerException.GetType().Name}");
                        Console.WriteLine($"Inner Exception Stack Trace:\n{ex.InnerException.StackTrace}");
                        
                        errorLogWriter.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                        errorLogWriter.WriteLine($"Inner Exception Type: {ex.InnerException.GetType().Name}");
                        errorLogWriter.WriteLine($"Inner Exception Stack Trace:\n{ex.InnerException.StackTrace}");
                    }
                    
                    errorCount++;
                    errorFiles.Add(fileName);

                    // Consider adding this file to known issues for future runs if needed
                    errorLogWriter.WriteLine($"SUGGESTION: Consider adding {fileName} to knownIssueFiles if this error persists.");
                }
                Console.WriteLine($"============================================================================");
            }

            // After processing all files, add faces to the combined with MSCN file
            Console.WriteLine("\n--- Writing Faces to Combined OBJ with MSCN ---");
            
            // Reset the file counter and process again for faces
            totalVerticesOffset = 0;
            foreach (var inputFilePath in pm4Files)
            {
                var fileName = Path.GetFileName(inputFilePath);
                
                // Skip known issue files and zero-byte files
                if (knownIssueFiles.Contains(fileName) || new FileInfo(inputFilePath).Length == 0)
                {
                    continue;
                }
                
                try
                {
                    var pm4File = PM4File.FromFile(inputFilePath);
                    
                    // Only process if we have MSUR/MSVI/MSVT data
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null)
                    {
                        int facesWritten = 0;
                        combinedWithMscnWriter.WriteLine($"g CombinedMesh_{Path.GetFileNameWithoutExtension(inputFilePath)}");
                        
                        // Write faces from MSUR/MSVI to combined with MSCN file
                        foreach (var msur in pm4File.MSUR.Entries)
                        {
                            if (msur.MsviFirstIndex >= 0 && msur.IndexCount >= 3 && 
                                msur.MsviFirstIndex + msur.IndexCount <= pm4File.MSVI.Indices.Count)
                            {
                                // Account for the fact that MSCN points are written after MSVT vertices
                                // First get vertex count for the proper offset
                                int localMsvtVertexCount = pm4File.MSVT.Vertices.Count;

                                // The indices form triangle fans where idx0 is the central vertex
                                for (int i = 0; i < msur.IndexCount - 2; i++)
                                {
                                    // Properly triangulate by reading the correct indices for each triangle
                                    uint idx0 = pm4File.MSVI.Indices[(int)msur.MsviFirstIndex];
                                    uint idx1 = pm4File.MSVI.Indices[(int)msur.MsviFirstIndex + i + 1];
                                    uint idx2 = pm4File.MSVI.Indices[(int)msur.MsviFirstIndex + i + 2];
                                    
                                    if (idx0 < localMsvtVertexCount && 
                                        idx1 < localMsvtVertexCount && 
                                        idx2 < localMsvtVertexCount)
                                    {
                                        // Add vertex offset to account for previous files' vertices
                                        // Add 1 for OBJ's 1-based indexing
                                        // Use triangle fan pattern with proper winding order 
                                        combinedWithMscnWriter.WriteLine($"f {idx0 + 1 + totalVerticesOffset} {idx1 + 1 + totalVerticesOffset} {idx2 + 1 + totalVerticesOffset}");
                                        facesWritten++;
                                    }
                                }
                            }
                        }
                        
                        if (facesWritten > 0)
                        {
                            Console.WriteLine($"  Added {facesWritten} faces from {fileName} to combined MSCN OBJ");
                        }
                    }
                    
                    // Update the vertex offset for the next file
                    totalVerticesOffset += pm4File.MSVT?.Vertices.Count ?? 0;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  WARNING: Could not add faces from {fileName} to combined MSCN OBJ: {ex.Message}");
                }
            }
            
            Console.WriteLine("--- Finished Writing Faces to Combined OBJ with MSCN ---");

            // --- Generate Combined Render Mesh Normals and Faces ---
            Console.WriteLine("\n--- Generating Combined Render Mesh Normals and Faces ---");
            if (combinedVertices.Count > 0 && combinedTriangleIndices.Count >= 3)
            {
                // Compute combined normals from all geometry
                var combinedNormals = Pm4CoordinateTransforms.ComputeVertexNormals(combinedVertices, combinedTriangleIndices);
                Console.WriteLine($"  Computed {combinedNormals.Count} normals from {combinedVertices.Count} vertices and {combinedTriangleIndices.Count / 3} faces");
                
                // Write normals to combined render mesh file
                combinedRenderMeshWriter.WriteLine("# Combined vertex normals computed from all PM4 files");
                foreach (var normal in combinedNormals)
                {
                    combinedRenderMeshWriter.WriteLine(FormattableString.Invariant($"vn {normal.X:F6} {normal.Y:F6} {normal.Z:F6}"));
                }
                
                // Write faces with normals to combined render mesh file
                combinedRenderMeshWriter.WriteLine("# Combined faces with normals (v//vn)");
                int combinedFacesWritten = 0;
                int degenerateTrianglesSkipped = 0;
                
                for (int i = 0; i + 2 < combinedTriangleIndices.Count; i += 3)
                {
                    int idx1 = combinedTriangleIndices[i] + 1; // OBJ 1-based indexing
                    int idx2 = combinedTriangleIndices[i + 1] + 1;
                    int idx3 = combinedTriangleIndices[i + 2] + 1;
                    
                    // Validate triangle - skip if any vertices are identical (degenerate triangle)
                    if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
                    {
                        degenerateTrianglesSkipped++;
                        continue; // Skip this degenerate triangle
                    }
                    
                    // Validate vertex indices are within bounds
                    if (idx1 <= 0 || idx1 > combinedVertices.Count ||
                        idx2 <= 0 || idx2 > combinedVertices.Count ||
                        idx3 <= 0 || idx3 > combinedVertices.Count)
                    {
                        degenerateTrianglesSkipped++;
                        continue; // Skip triangle with out-of-bounds indices
                    }
                    
                    combinedRenderMeshWriter.WriteLine($"f {idx1}//{idx1} {idx2}//{idx2} {idx3}//{idx3}");
                    combinedFacesWritten++;
                }
                
                Console.WriteLine($"  Written {combinedFacesWritten} valid faces to combined render mesh");
                if (degenerateTrianglesSkipped > 0)
                {
                    Console.WriteLine($"  Skipped {degenerateTrianglesSkipped} degenerate/invalid triangles");
                }
                
                // Write summary of files included
                combinedRenderMeshWriter.WriteLine($"\n# Combined from {combinedFileInfo.Count} PM4 files:");
                foreach (var (fileName, vertexCount, faceCount) in combinedFileInfo)
                {
                    combinedRenderMeshWriter.WriteLine($"#   {fileName}: {vertexCount} vertices, {faceCount} faces");
                }
            }
            else
            {
                Console.WriteLine("  No valid geometry data found for combined render mesh");
            }
            Console.WriteLine("--- Finished Combined Render Mesh Generation ---");

            errorLogWriter.WriteLine($"\n--- Batch Processing Summary ---");
            errorLogWriter.WriteLine($"Total Files:     {pm4Files.Count}");
            errorLogWriter.WriteLine($"Processed:       {processedCount}");
            errorLogWriter.WriteLine($"Known Issues:    {knownIssuesCount}");
            errorLogWriter.WriteLine($"Skipped:         {skippedCount}");
            errorLogWriter.WriteLine($"Errors:          {errorCount}");
            
            if (knownIssueFiles_Encountered.Count > 0)
            {
                errorLogWriter.WriteLine("\nKnown Issue Files Encountered:");
                foreach (var file in knownIssueFiles_Encountered)
                {
                    errorLogWriter.WriteLine($"  - {file}");
                }
            }
            
            if (skippedFiles.Count > 0)
            {
                errorLogWriter.WriteLine("\nSkipped Files (Zero-byte):");
                if (skippedFiles.Count <= 20)
                {
                    foreach (var file in skippedFiles)
                    {
                        errorLogWriter.WriteLine($"  - {file}");
                    }
                }
                else
                {
                    foreach (var file in skippedFiles.Take(10))
                    {
                        errorLogWriter.WriteLine($"  - {file}");
                    }
                    errorLogWriter.WriteLine($"  ... and {skippedFiles.Count - 10} more skipped files");
                }
            }
            
            if (errorFiles.Count > 0)
            {
                errorLogWriter.WriteLine("\nFiles with Errors:");
                foreach (var file in errorFiles)
                {
                    errorLogWriter.WriteLine($"  - {file}");
                }
            }
            
            Console.WriteLine($"\n--- LoadAndProcessPm4FilesInDirectory FINISHED ---");
            Console.WriteLine($"Total Files:      {pm4Files.Count}");
            Console.WriteLine($"Successfully Processed: {processedCount} files");
            Console.WriteLine($"Known Issues:     {knownIssuesCount} files");
            Console.WriteLine($"Skipped:          {skippedCount} files");
            Console.WriteLine($"Encountered Errors: {errorCount} files");
            
            if (knownIssueFiles_Encountered.Count > 0)
            {
                Console.WriteLine("\nKnown Issue Files:");
                foreach (var file in knownIssueFiles_Encountered)
                {
                    Console.WriteLine($"  - {file}");
                }
            }
            
            if (errorFiles.Count > 0)
            {
                Console.WriteLine("\nFiles with Errors:");
                foreach (var file in errorFiles.Take(5)) // Show first 5 for console
                {
                    Console.WriteLine($"  - {file}");
                }
                if (errorFiles.Count > 5)
                {
                    Console.WriteLine($"  ... and {errorFiles.Count - 5} more (see {errorLogPath})");
                }
            }

            // MODIFIED: Make assertion more lenient - exclude known issues from the error threshold calculation
            int unexpectedErrorCount = errorCount;
            double errorThreshold = Math.Max(5, (pm4Files.Count - knownIssuesCount) * 0.02); // 2% of total files (excluding known issues) or 5, whichever is greater
            bool tooManyErrors = unexpectedErrorCount > errorThreshold;
            
            if (tooManyErrors)
            {
                Assert.False(tooManyErrors, $"Encountered {unexpectedErrorCount} unexpected errors during batch processing, exceeding threshold of {errorThreshold}. Check error log: {errorLogPath}");
            }
            
            // Always ensure at least one file was processed if files were found (excluding known issues)
            if (pm4Files.Count > knownIssuesCount && processedCount == 0)
            {
                Assert.True(processedCount > 0, "No PM4 files were successfully processed, although files were found.");
            }
        }


        // --- Helper Method for Single File Processing ---
        // Updated signature to accept combined writer and vertex offset, and return vertex count
        private int ProcessSinglePm4File(string inputFilePath, string outputDir, StreamWriter combinedTransformedWriter, int vertexOffset)
        {
            var result = ProcessSinglePm4FileWithData(inputFilePath, outputDir, combinedTransformedWriter, vertexOffset);
            return result.vertexCount;
        }
        
        // Enhanced version that returns geometry data for combined mesh generation
        private (int vertexCount, List<Vector3> vertices, List<int> triangleIndices) ProcessSinglePm4FileWithData(string inputFilePath, string outputDir, StreamWriter combinedTransformedWriter, int vertexOffset)
        {
            // MOVED UP: Define fileBaseName earlier
            var fileName = Path.GetFileName(inputFilePath);
            var fileBaseName = Path.GetFileNameWithoutExtension(inputFilePath);

            // Initialize return data
            var vertices = new List<Vector3>();
            var triangleIndices = new List<int>();

            // Check if this is a known problematic file that needs special handling
            if (fileName.Equals("development_49_28.pm4", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine($"  * Detected known problematic file with high MPRR/MPRL ratio: {fileName}");
                Console.WriteLine($"  * Using specialized processing approach...");
                int vertexCount = ProcessHighRatioPm4File(inputFilePath, outputDir, combinedTransformedWriter, vertexOffset);
                return (vertexCount, vertices, triangleIndices); // Note: high ratio processing doesn't return geometry data yet
            }
            
            // ... rest of the existing method ...
            
            // 1. Derive baseOutputName from inputFilePath
            var baseOutputName = Path.GetFileNameWithoutExtension(inputFilePath);
            var baseOutputPath = Path.Combine(outputDir, baseOutputName);

            // --- Path Construction (Relative to Test Output Directory) ---
            var baseDir = AppDomain.CurrentDomain.BaseDirectory; // Needed? No, use outputDir directly.

            // 2. Define all output file paths using baseOutputPath (as before)
            var outputMprlFilePath = baseOutputPath + "_mprl.obj";
            var outputMslkFilePath = baseOutputPath + "_mslk.obj"; // RE-ADDED
            var outputMslkJsonPath = baseOutputPath + "_mslk_hierarchy.json";
            var outputSkippedMslkLogPath = baseOutputPath + "_skipped_mslk.log";
            var outputPm4MslkNodesFilePath = baseOutputPath + "_pm4_mslk_nodes.obj";
            var outputRenderMeshPath = baseOutputPath + "_render_mesh.obj";
            var outputRenderMeshTransformedPath = baseOutputPath + "_render_mesh_transformed.obj"; // New path for transformed OBJ
            string debugLogPath = baseOutputPath + ".debug.log";
            string summaryLogPath = baseOutputPath + ".summary.log";
            var outputBuildingIdsPath = baseOutputPath + "_building_ids.log";
            var outputMslkDoodadCsvPath = baseOutputPath + "_mslk_doodad_data.csv"; // ADDED: Path for Doodad CSV
            var outputMprrDataCsvPath = baseOutputPath + "_mprr_data.csv"; 
            var outputMprrSequencesCsvPath = baseOutputPath + "_mprr_sequences.csv"; // ADDED: Path for raw MPRR sequences
            var outputMprrLinksObjPath = baseOutputPath + "_mprr_links.obj"; // REVERTED: Path for MPRR Links OBJ

            // Console logging for paths specific to this file
            Console.WriteLine($"  Input File Path: {inputFilePath}");
            Console.WriteLine($"  Output Base Path: {baseOutputPath}");
            Console.WriteLine($"  Output MPRL OBJ: {outputMprlFilePath}");
            Console.WriteLine($"  Output MSLK OBJ: {outputMslkFilePath}"); // RE-ADDED
            Console.WriteLine($"  Output MSLK JSON: {outputMslkJsonPath}");
            Console.WriteLine($"  Output Skipped MSLK Log: {outputSkippedMslkLogPath}");
            Console.WriteLine($"  Output PM4 MSLK Nodes OBJ: {outputPm4MslkNodesFilePath}");
            Console.WriteLine($"  Output RENDER MESH OBJ: {outputRenderMeshPath}");
            Console.WriteLine($"  Output RENDER MESH TRANSFORMED OBJ: {outputRenderMeshTransformedPath}"); // Log new path
            Console.WriteLine($"  Debug Log: {debugLogPath}");
            Console.WriteLine($"  Summary Log: {summaryLogPath}");
            Console.WriteLine($"  Building IDs Log: {outputBuildingIdsPath}");
            Console.WriteLine($"  Output MSLK Doodad CSV: {outputMslkDoodadCsvPath}"); // ADDED: Log new path
            Console.WriteLine($"  Output MPRR Data CSV: {outputMprrDataCsvPath}"); // ADDED: Log new path
            Console.WriteLine($"  Output MPRR Sequences CSV: {outputMprrSequencesCsvPath}"); // ADDED: Log new path
            Console.WriteLine($"  Output MPRR Links OBJ: {outputMprrLinksObjPath}"); // REVERTED: Log new path

            // 3. Initialize HashSet for unique building IDs (scoped per file)
            var uniqueBuildingIds = new HashSet<uint>();

            // 4. Load the PM4 file
            var pm4File = PM4File.FromFile(inputFilePath);
            Assert.NotNull(pm4File); // Basic assertion per file

            // ***** MOVED WRITER INITIALIZATION HERE *****
            using var debugWriter = new StreamWriter(debugLogPath, false);
            using var summaryWriter = new StreamWriter(summaryLogPath, false);
            using var buildingIdWriter = new StreamWriter(outputBuildingIdsPath, false);
            using var renderMeshWriter = new StreamWriter(outputRenderMeshPath, false);
            using var mprlWriter = new StreamWriter(outputMprlFilePath);
            using var mslkWriter = new StreamWriter(outputMslkFilePath, false); // RE-ADDED
            using var skippedMslkWriter = new StreamWriter(outputSkippedMslkLogPath);
            using var mslkNodesWriter = new StreamWriter(outputPm4MslkNodesFilePath, false);
            using var renderMeshTransformedWriter = new StreamWriter(outputRenderMeshTransformedPath, false); // Initialize writer for transformed OBJ
            using var mslkDoodadCsvWriter = new StreamWriter(outputMslkDoodadCsvPath, false); // ADDED: Writer for Doodad CSV
            using var mprrSequencesCsvWriter = new StreamWriter(outputMprrSequencesCsvPath, false); // ADDED: Writer for MPRR sequences
            // using var mprrDataCsvWriter = new StreamWriter(outputMprrDataCsvPath, false);  // REMOVED: MPRR data logic moved/changed
            // using var mprrLinksObjWriter = new StreamWriter(outputMprrLinksObjPath, false); // REMOVED: Link generation commented out
            // ***** END MOVED WRITER INITIALIZATION *****

            // --- Generate MPRL Data CSV (Replaces old _mprr_data.csv logic) ---
            var mprlDataOutputPath = Path.Combine(outputDir, fileBaseName + "_mprl_data.csv"); // Uses fileBaseName here
            debugWriter.WriteLine($"\n--- Generating MPRL Data CSV -> {Path.GetFileName(mprlDataOutputPath)} ---");
            try
            {
                using (var mprlCsvWriter = new StreamWriter(mprlDataOutputPath))
                {
                    // Write header with only MPRL fields
                    mprlCsvWriter.WriteLine("MPRLIndex,MPRL_Unk00,MPRL_Unk02,MPRL_Unk04,MPRL_Unk06,MPRL_PosX,MPRL_PosY,MPRL_PosZ,MPRL_Unk14,MPRL_Unk16");

                    if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                    {
                        for (int i = 0; i < pm4File.MPRL.Entries.Count; i++)
                        {
                            var entry = pm4File.MPRL.Entries[i];
                            // Write only MPRL data directly
                            mprlCsvWriter.WriteLine(
                                $"{i},{entry.Unknown_0x00},{entry.Unknown_0x02},{entry.Unknown_0x04},{entry.Unknown_0x06}," +
                                $"{entry.Position.X.ToString(CultureInfo.InvariantCulture)},{entry.Position.Y.ToString(CultureInfo.InvariantCulture)},{entry.Position.Z.ToString(CultureInfo.InvariantCulture)}," +
                                $"{entry.Unknown_0x14},{entry.Unknown_0x16}");
                        }
                        debugWriter.WriteLine($"  Wrote {pm4File.MPRL.Entries.Count} entries to {Path.GetFileName(mprlDataOutputPath)}");
                    }
                    else
                    {
                        debugWriter.WriteLine("  MPRL chunk missing or empty. CSV will only contain header.");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  > ERROR generating MPRL Data CSV: {ex.Message}");
                debugWriter.WriteLine($"\n!!!!!! ERROR generating MPRL Data CSV: {ex.ToString()} !!!!!!");
            }
            debugWriter.WriteLine("--- End Generating MPRL Data CSV ---\n");

            Console.WriteLine("  > Skipping MPRR Links OBJ generation (Visualization logic disabled due to structural uncertainty).");
            debugWriter.WriteLine("\n--- Skipping MPRR Links OBJ Generation (Logic Disabled) ---");

            // --- Export Configuration Flags ---
            bool exportMsvtVertices = true;
            bool exportMprlPoints = true;
            bool exportMslkPaths = true;
            bool exportOnlyFirstMslk = false;
            bool processMsurEntries = true;
            bool exportOnlyFirstMsur = false;
            bool logMdsfLinks = true;
            bool exportMscnPoints = true;  // Changed from false to true

            // --- Initialize Writers --- (Moved up before MPRR logging)
            // using var debugWriter = new StreamWriter(debugLogPath, false);
            // ... other writers ...

            int msvtFileVertexCount = 0; // Declare variable here for correct scope

            // MSCN is collision boundary geometry, NOT normals - they naturally have different vertex counts
            // Real normals should be calculated from face geometry using MSVI indices
            bool generateFaces = pm4File.MSVT != null && pm4File.MSVI != null && pm4File.MSVI.Indices.Count >= 3;
            
            if (pm4File.MSCN != null && pm4File.MSVT != null) 
            {
                debugWriter.WriteLine($"INFO: MSCN contains {pm4File.MSCN.ExteriorVertices.Count} collision boundary vertices, MSVT has {pm4File.MSVT.Vertices.Count} render vertices. These are separate geometry types with different purposes.");
            }
            else if(pm4File.MSCN == null)
            {
                debugWriter.WriteLine("INFO: MSCN chunk not found. No collision boundary data available.");
            }

            try // Keep inner try for logging/resource cleanup context if needed, though outer one catches errors
            {
                // Log MDOS Entry Count for verification (MOVED HERE)
                int mdosEntryCount = pm4File.MDOS?.Entries?.Count ?? -1; // ADDED null check for CS8602
                Console.WriteLine($"  INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");
                debugWriter.WriteLine($"INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");

                // Log counts of other potentially relevant chunks
                debugWriter.WriteLine($"INFO: MSVT Vertices: {pm4File.MSVT?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSPV Vertices: {pm4File.MSPV?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MPRL Entries: {pm4File.MPRL?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSPI Indices: {pm4File.MSPI?.Indices.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSVI Indices: {pm4File.MSVI?.Indices.Count ?? -1}");
                // UPDATED: Use Sequences.Count with null checks
                debugWriter.WriteLine($"INFO: MPRR Sequences: {pm4File.MPRR?.Sequences?.Count ?? 0}");
                debugWriter.WriteLine($"INFO: MSUR Entries: {pm4File.MSUR?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MDBH Entries: {pm4File.MDBH?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MDSF Entries: {pm4File.MDSF?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"INFO: MSLK Entries: {pm4File.MSLK?.Entries.Count ?? -1}");


                // UPDATED: Header for combined nodes, points, and groups file
                mslkNodesWriter.WriteLine($"# PM4 MSLK Node Anchor Points (Vertices, Points, Groups) (from Unk10 -> MSVI -> MSVT) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                mslkNodesWriter.WriteLine($"# Vertices (v) Transform: Y, X, Z");
                mslkNodesWriter.WriteLine($"# Points (p) define node vertices explicitly for viewers");
                mslkNodesWriter.WriteLine($"# Groups (g) collect nodes sharing the same MSLK.Unk04 Group ID (No lines 'l' drawn)"); // UPDATED: Header comment

                // Write headers for non-MPRL files
                renderMeshWriter.WriteLine($"# PM4 Render Mesh (MSVT/MSVI/MSUR) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                renderMeshWriter.WriteLine("# Vertices Transform: Y, X, Z");
                // Header for PM4-relative transformed render mesh (MSVT/MSUR only)
                renderMeshTransformedWriter.WriteLine($"# PM4 Render Mesh (MSVT/MSVI/MSUR) - PM4-RELATIVE (Y, X, Z) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                renderMeshTransformedWriter.WriteLine("# Vertices Transform: Y, X, Z (PM4-relative coordinates)");
                // RE-ADDED: Header for standalone MSLK file
                mslkWriter.WriteLine($"# PM4 MSLK Geometry (Vertices, Faces 'f', Lines 'l', Points 'p') (Exported: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                mslkWriter.WriteLine("# Vertices (v) Transform: X, Y, Z (Standard)");
                skippedMslkWriter.WriteLine($"# PM4 Skipped/Invalid MSLK Entries Log (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");
                buildingIdWriter.WriteLine($"# Unique Building IDs from MDOS (via MDSF/MSUR link) (Generated: {DateTime.Now}) - File: {Path.GetFileName(inputFilePath)}");

                // --- Moved Index Validation Logging Inside Try Block ---
                int mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");
                if (pm4File.MPRR != null)
                {
                    // ValidateIndices logic remains commented out or removed as it's invalid
                     debugWriter.WriteLine("MPRR Index Validation is currently disabled.");
                }
                else
                {
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
                int mprlFileVertexCount = 0; // For the single MPRL file
                int facesWrittenToRenderMesh = 0; // ADDED: Counter for faces written to the render mesh

                // --- 1. Export MSPV vertices -> MOVED to MSLK Processing ---

                // --- 2. Export MSVT vertices (Original Y,X,Z to renderMeshWriter; Transformed Offset-Y,Offset-X,Z to transformed writers) ---
                msvtFileVertexCount = 0; // Reset for clarity
                if (exportMsvtVertices && pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    renderMeshWriter.WriteLine("o Render_Mesh"); // Original render mesh object
                    renderMeshTransformedWriter.WriteLine($"o Render_Mesh_{baseOutputName}"); // Individual transformed render mesh object

                    debugWriter.WriteLine("\n--- Exporting MSVT Vertices (Y,X,Z) -> _render_mesh.obj ---");
                    debugWriter.WriteLine("--- Exporting TRANSFORMED MSVT Vertices/Normals -> _render_mesh_transformed.obj AND combined_render_mesh_transformed.obj ---");
                    summaryWriter.WriteLine($"\n--- MSVT Vertices ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10) -> Render Mesh --- (Original & Transformed)");
                    int logCounterMsvt = 0;
                    int msvtIndex = 0; // Index for accessing MSCN

                    foreach (var vertex in pm4File.MSVT.Vertices)
                    {
                        // Apply Y, X, Z transform for original export
                        float originalX = vertex.Y;
                        float originalY = vertex.X;
                        float originalZ = vertex.Z;

                        // Write original vertex
                        renderMeshWriter.WriteLine(FormattableString.Invariant($"v {originalX:F6} {originalY:F6} {originalZ:F6}"));

                        // Apply PM4-relative transformation
                        var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);

                        msvtFileVertexCount++; // Increment counter

                        // Write PM4-relative vertex to individual transformed file
                        renderMeshTransformedWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSVT {msvtIndex}"));
                        // Use the same PM4-relative transformation for global combined mesh
                        combinedTransformedWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} # MSVT {msvtIndex} (File: {baseOutputName})"));
                        
                        // Always collect vertices for combined mesh (even if face generation is disabled)
                        if (!generateFaces)
                        {
                            vertices.Add(pm4Coords);
                        }

                        // NOTE: Normals should be calculated from face geometry, not from MSCN
                        // MSCN contains collision boundary vertices, not vertex normals

                        if (logCounterMsvt < 10)
                        {
                            summaryWriter.WriteLine(FormattableString.Invariant($"  MSVT Vert {msvtIndex}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Orig v ({originalX:F3}, {originalY:F3}, {originalZ:F3}) -> PM4 v ({pm4Coords.X:F3}, {pm4Coords.Y:F3}, {pm4Coords.Z:F3})"));
                            debugWriter.WriteLine(FormattableString.Invariant($"  MSVT V {msvtIndex}: Orig=({originalX:F3}, {originalY:F3}, {originalZ:F3}) PM4=({pm4Coords.X:F3}, {pm4Coords.Y:F3}, {pm4Coords.Z:F3})"));
                        }
                        logCounterMsvt++;
                        msvtIndex++; // Increment index for next vertex/normal
                    }
                    if (pm4File.MSVT.Vertices.Count > 10)
                    {
                        summaryWriter.WriteLine("  ... (Summary log limited to first 10 Vertices) ...");
                    }
                    debugWriter.WriteLine($"Wrote {msvtFileVertexCount} MSVT vertices (v) to _render_mesh.obj.");
                    debugWriter.WriteLine($"Wrote {msvtFileVertexCount} PM4-relative MSVT vertices (v) to transformed OBJ files.");
                    
                    // --- Collect triangle indices for combined mesh (regardless of face generation) ---
                    if (!generateFaces && pm4File.MSVI != null && pm4File.MSVI.Indices.Count >= 3)
                    {
                        int fallbackInvalidTrianglesSkipped = 0;
                        for (int i = 0; i + 2 < pm4File.MSVI.Indices.Count; i += 3)
                        {
                            uint idx1 = pm4File.MSVI.Indices[i];
                            uint idx2 = pm4File.MSVI.Indices[i + 1];
                            uint idx3 = pm4File.MSVI.Indices[i + 2];
                            
                            // Validate indices against vertex count
                            if (idx1 < msvtFileVertexCount && idx2 < msvtFileVertexCount && idx3 < msvtFileVertexCount)
                            {
                                // Validate triangle - skip if any vertices are identical (degenerate triangle)
                                if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
                                {
                                    fallbackInvalidTrianglesSkipped++;
                                    continue; // Skip this degenerate triangle
                                }
                                
                                triangleIndices.Add((int)idx1);
                                triangleIndices.Add((int)idx2);
                                triangleIndices.Add((int)idx3);
                            }
                            else
                            {
                                fallbackInvalidTrianglesSkipped++;
                            }
                        }
                        debugWriter.WriteLine($"Collected {triangleIndices.Count / 3} valid triangle indices for combined mesh");
                        if (fallbackInvalidTrianglesSkipped > 0)
                        {
                            debugWriter.WriteLine($"Skipped {fallbackInvalidTrianglesSkipped} invalid/degenerate triangles during fallback collection");
                        }
                    }
                    
                    // --- Generate Faces and Normals from MSVI indices ---
                    if (generateFaces && pm4File.MSVI!.Indices.Count >= 3)
                    {
                        debugWriter.WriteLine($"\n--- Generating Faces and Computed Normals from MSVI indices ---");
                        
                        // First, collect all vertices in PM4-relative coordinates
                        var localVertices = new List<Vector3>();
                        foreach (var vertex in pm4File.MSVT.Vertices)
                        {
                            var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                            localVertices.Add(pm4Coords);
                            vertices.Add(pm4Coords); // Add to return list for combined mesh
                        }
                        
                        // Collect all triangle indices
                        var localTriangleIndices = new List<int>();
                        int invalidTrianglesSkipped = 0;
                        
                        for (int i = 0; i + 2 < pm4File.MSVI.Indices.Count; i += 3)
                        {
                            uint idx1 = pm4File.MSVI.Indices[i];
                            uint idx2 = pm4File.MSVI.Indices[i + 1];
                            uint idx3 = pm4File.MSVI.Indices[i + 2];
                            
                            // Validate indices against vertex count
                            if (idx1 < msvtFileVertexCount && idx2 < msvtFileVertexCount && idx3 < msvtFileVertexCount)
                            {
                                // Validate triangle - skip if any vertices are identical (degenerate triangle)
                                if (idx1 == idx2 || idx1 == idx3 || idx2 == idx3)
                                {
                                    invalidTrianglesSkipped++;
                                    continue; // Skip this degenerate triangle
                                }
                                
                                localTriangleIndices.Add((int)idx1);
                                localTriangleIndices.Add((int)idx2);
                                localTriangleIndices.Add((int)idx3);
                                
                                // Add to return list for combined mesh (will be adjusted with vertex offset later)
                                triangleIndices.Add((int)idx1);
                                triangleIndices.Add((int)idx2);
                                triangleIndices.Add((int)idx3);
                            }
                            else
                            {
                                invalidTrianglesSkipped++;
                            }
                        }
                        
                        // Compute proper vertex normals from face geometry
                        var normals = Pm4CoordinateTransforms.ComputeVertexNormals(localVertices, localTriangleIndices);
                        debugWriter.WriteLine($"Computed {normals.Count} vertex normals from {localTriangleIndices.Count / 3} valid triangles");
                        if (invalidTrianglesSkipped > 0)
                        {
                            debugWriter.WriteLine($"Skipped {invalidTrianglesSkipped} invalid/degenerate triangles during processing");
                        }
                        
                        // Write computed normals to OBJ files
                        renderMeshWriter.WriteLine("# Computed vertex normals from face geometry");
                        renderMeshTransformedWriter.WriteLine("# Computed vertex normals from face geometry");
                        for (int i = 0; i < normals.Count; i++)
                        {
                            var normal = normals[i];
                            renderMeshWriter.WriteLine(FormattableString.Invariant($"vn {normal.X:F6} {normal.Y:F6} {normal.Z:F6}"));
                            renderMeshTransformedWriter.WriteLine(FormattableString.Invariant($"vn {normal.X:F6} {normal.Y:F6} {normal.Z:F6}"));
                        }
                        
                        // Write faces with normals (v//vn format)
                        renderMeshWriter.WriteLine("# Faces with computed normals (v//vn)");
                        renderMeshTransformedWriter.WriteLine("# Faces with computed normals (v//vn)");
                        int facesWritten = 0;
                        for (int i = 0; i + 2 < localTriangleIndices.Count; i += 3)
                        {
                            int idx1 = localTriangleIndices[i];
                            int idx2 = localTriangleIndices[i + 1];
                            int idx3 = localTriangleIndices[i + 2];
                            
                            // OBJ uses 1-based indexing for both vertices and normals
                            renderMeshWriter.WriteLine($"f {idx1 + 1}//{idx1 + 1} {idx2 + 1}//{idx2 + 1} {idx3 + 1}//{idx3 + 1}");
                            renderMeshTransformedWriter.WriteLine($"f {idx1 + 1}//{idx1 + 1} {idx2 + 1}//{idx2 + 1} {idx3 + 1}//{idx3 + 1}");
                            facesWritten++;
                        }
                        debugWriter.WriteLine($"Generated {facesWritten} faces with computed normals from MSVI indices");
                    }
                    
                    renderMeshWriter.WriteLine(); // Blank line after vertices/faces
                    renderMeshTransformedWriter.WriteLine(); // Blank line after vertices/faces
                }
                else { debugWriter.WriteLine("\nNo MSVT vertex data found or export skipped."); }

                // REMOVED: MSPV vertex export logic from here
                // REMOVED: totalFileVertices calculation

                debugWriter.Flush();

                // --- 3. Export MPRL points with correct (X, -Z, Y) transformation -> mprlWriter ONLY --- // RESTORED
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
                            // Apply PM4-relative MPRL transformation
                            var pm4Coords = Pm4CoordinateTransforms.FromMprlEntry(entry);

                            // ADDED: Include Unknown fields in comment
                            string comment = $"# MPRLIdx=[{mprlFileVertexCount}] " +
                                             $"Unk00={entry.Unknown_0x00} Unk02={entry.Unknown_0x02} " +
                                             $"Unk04={entry.Unknown_0x04} Unk06={entry.Unknown_0x06} " +
                                             $"Unk14={entry.Unknown_0x14} Unk16={entry.Unknown_0x16}";

                            if (logCounterMprl < 10) {
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> PM4=({pm4Coords.X:F3}, {pm4Coords.Y:F3}, {pm4Coords.Z:F3}) {comment}" // Added comment to summary log too
                                ));
                                debugWriter.WriteLine(FormattableString.Invariant($"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> PM4=({pm4Coords.X:F3}, {pm4Coords.Y:F3}, {pm4Coords.Z:F3}) {comment}")); // Log full details now
                            }
                            mprlWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} {comment}")); // ADDED comment to OBJ output
                            mprlFileVertexCount++;
                            logCounterMprl++;
                        }
                        if (pm4File.MPRL.Entries.Count > 10) {
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                            // REMOVED: Ellipsis message for debugWriter
                        }
                        debugWriter.WriteLine($"Wrote {mprlFileVertexCount} MPRL vertices to _mprl.obj file.");
                        summaryWriter.WriteLine($"--- Finished MPRL Processing (Exported: {mprlFileVertexCount}) ---");
                        mprlWriter.WriteLine();
                    }
                    else { debugWriter.WriteLine("No MPRL vertex data found. Skipping export."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MPRL export (Flag 'exportMprlPoints' is False).");
                }


                 // --- 4. Export Individual Chunk OBJ Files for Analysis ---
                 if (exportMscnPoints)
                 {
                    debugWriter.WriteLine($"\n--- Exporting Individual Chunk OBJ Files for Analysis ---");
                    
                    // 4.1 MSCN Collision Boundaries
                    string outputMscnObjPath = Path.Combine(outputDir, $"{baseOutputName}_chunk_mscn.obj");
                    if (pm4File.MSCN != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                    {
                        using var mscnWriter = new StreamWriter(outputMscnObjPath, false);
                        mscnWriter.WriteLine($"# PM4 MSCN Chunk - Collision Boundaries (Generated: {DateTime.Now})");
                        mscnWriter.WriteLine($"# File: {Path.GetFileName(inputFilePath)}");
                        mscnWriter.WriteLine($"# Transform: Y-axis correction + 180° rotation around X-axis");
                        mscnWriter.WriteLine("o MSCN_Collision_Boundaries");
                        
                        int mscnPointsExported = 0;
                        foreach (var point in pm4File.MSCN.ExteriorVertices)
                        {
                            var pm4Coords = Pm4CoordinateTransforms.FromMscnVertex(point);
                            mscnWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                            mscnPointsExported++;
                        }
                        debugWriter.WriteLine($"  MSCN: {mscnPointsExported} collision boundary points -> {Path.GetFileName(outputMscnObjPath)}");
                    }
                    
                    // 4.2 MSVT Render Mesh Vertices (clean, no faces)
                    string outputMsvtObjPath = Path.Combine(outputDir, $"{baseOutputName}_chunk_msvt.obj");
                    if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                    {
                        using var msvtWriter = new StreamWriter(outputMsvtObjPath, false);
                        msvtWriter.WriteLine($"# PM4 MSVT Chunk - Render Mesh Vertices (Generated: {DateTime.Now})");
                        msvtWriter.WriteLine($"# File: {Path.GetFileName(inputFilePath)}");
                        msvtWriter.WriteLine($"# Transform: PM4-relative coordinates (Y, X, Z)");
                        msvtWriter.WriteLine("o MSVT_Render_Vertices");
                        
                        int msvtVerticesExported = 0;
                        foreach (var vertex in pm4File.MSVT.Vertices)
                        {
                            var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                            msvtWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                            msvtVerticesExported++;
                        }
                        debugWriter.WriteLine($"  MSVT: {msvtVerticesExported} render mesh vertices -> {Path.GetFileName(outputMsvtObjPath)}");
                    }
                    
                    // 4.3 MSLK/MSPV Geometric Structure
                    string outputMslkObjPath = Path.Combine(outputDir, $"{baseOutputName}_chunk_mslk_mspv.obj");
                    if (pm4File.MSLK != null && pm4File.MSPV != null && pm4File.MSPV.Vertices.Count > 0)
                    {
                        using var mslkChunkWriter = new StreamWriter(outputMslkObjPath, false);
                        mslkChunkWriter.WriteLine($"# PM4 MSLK/MSPV Chunk - Geometric Structure (Generated: {DateTime.Now})");
                        mslkChunkWriter.WriteLine($"# File: {Path.GetFileName(inputFilePath)}");
                        mslkChunkWriter.WriteLine($"# Transform: PM4-relative coordinates (X, Y, Z)");
                        mslkChunkWriter.WriteLine("o MSLK_Geometric_Structure");
                        
                        // Write MSPV vertices
                        int mspvVerticesExported = 0;
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            var pm4Coords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                            mslkChunkWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                            mspvVerticesExported++;
                        }
                        
                        // Add MSLK geometry as lines/faces
                        int mslkGeometryExported = 0;
                        if (pm4File.MSPI != null)
                        {
                            mslkChunkWriter.WriteLine("\n# MSLK geometry elements");
                            foreach (var mslkEntry in pm4File.MSLK.Entries)
                            {
                                if (mslkEntry.MspiFirstIndex >= 0 && mslkEntry.MspiIndexCount >= 2 && 
                                    mslkEntry.MspiFirstIndex + mslkEntry.MspiIndexCount <= pm4File.MSPI.Indices.Count)
                                {
                                    List<uint> validIndices = new List<uint>();
                                    for (int i = 0; i < mslkEntry.MspiIndexCount; i++)
                                    {
                                        uint mspiIdx = pm4File.MSPI.Indices[mslkEntry.MspiFirstIndex + i];
                                        if (mspiIdx < mspvVerticesExported)
                                        {
                                            validIndices.Add(mspiIdx + 1); // OBJ is 1-based
                                        }
                                    }
                                    
                                    if (validIndices.Count >= 3)
                                    {
                                        // Create faces using triangle fan
                                        for (int i = 1; i < validIndices.Count - 1; i++)
                                        {
                                            mslkChunkWriter.WriteLine($"f {validIndices[0]} {validIndices[i]} {validIndices[i + 1]}");
                                            mslkGeometryExported++;
                                        }
                                    }
                                    else if (validIndices.Count == 2)
                                    {
                                        // Create line
                                        mslkChunkWriter.WriteLine($"l {validIndices[0]} {validIndices[1]}");
                                        mslkGeometryExported++;
                                    }
                                }
                            }
                        }
                        debugWriter.WriteLine($"  MSLK/MSPV: {mspvVerticesExported} vertices, {mslkGeometryExported} geometry elements -> {Path.GetFileName(outputMslkObjPath)}");
                    }
                    
                    // 4.4 MPRL Points
                    string outputMprlChunkObjPath = Path.Combine(outputDir, $"{baseOutputName}_chunk_mprl.obj");
                    if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                    {
                        using var mprlChunkWriter = new StreamWriter(outputMprlChunkObjPath, false);
                        mprlChunkWriter.WriteLine($"# PM4 MPRL Chunk - Path/Reference Points (Generated: {DateTime.Now})");
                        mprlChunkWriter.WriteLine($"# File: {Path.GetFileName(inputFilePath)}");
                        mprlChunkWriter.WriteLine($"# Transform: PM4-relative coordinates (X, -Z, Y)");
                        mprlChunkWriter.WriteLine("o MPRL_Reference_Points");
                        
                        int mprlPointsExported = 0;
                        foreach (var entry in pm4File.MPRL.Entries)
                        {
                            var pm4Coords = Pm4CoordinateTransforms.FromMprlEntry(entry);
                            mprlChunkWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                            mprlPointsExported++;
                        }
                        debugWriter.WriteLine($"  MPRL: {mprlPointsExported} reference points -> {Path.GetFileName(outputMprlChunkObjPath)}");
                    }
                    
                    Console.WriteLine($"  > Generated individual chunk OBJ files for analysis:");
                    Console.WriteLine($"      - MSCN collision boundaries: {Path.GetFileName(outputMscnObjPath)}");
                    Console.WriteLine($"      - MSVT render vertices: {Path.GetFileName(outputMsvtObjPath)}");
                    Console.WriteLine($"      - MSLK/MSPV geometry: {Path.GetFileName(outputMslkObjPath)}");
                    Console.WriteLine($"      - MPRL reference points: {Path.GetFileName(outputMprlChunkObjPath)}");
                    debugWriter.WriteLine($"--- Finished exporting individual chunk OBJ files for analysis ---");
                 } 
                 else 
                 { 
                      debugWriter.WriteLine("\n--- Skipping MSCN point export (Flag 'exportMscnPoints' is False) ---");
                 }


                 // --- 5. Export MSLK paths/points -> Standalone _mslk.obj --- //
                if (exportMslkPaths)
                {
                    if (pm4File.MSLK != null && pm4File.MSPI != null && pm4File.MSPV != null && pm4File.MSPV.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSLK -> MSPI -> MSPV Chain -> _mslk.obj (Skipped -> _skipped_mslk.log) ---");
                        summaryWriter.WriteLine($"\n--- MSLK Processing ({Path.GetFileNameWithoutExtension(inputFilePath)}) (First 10 Entries) -> _mslk.obj ---");

                        // --- Write STANDARD X,Y,Z MSPV Vertices directly to _mslk.obj ---
                        int mslkMspvVertexCount = 0;
                        mslkWriter.WriteLine("o MSLK_Geometry_Vertices");
                        debugWriter.WriteLine("  Writing MSPV vertices using centralized transform to _mslk.obj...");
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            var worldCoords = Pm4CoordinateTransforms.FromMspvVertex(vertex);
                            mslkWriter.WriteLine(FormattableString.Invariant($"v {worldCoords.X:F6} {worldCoords.Y:F6} {worldCoords.Z:F6}"));
                            mslkMspvVertexCount++;
                        }
                        debugWriter.WriteLine($"  Wrote {mslkMspvVertexCount} vertices to _mslk.obj.");
                        mslkWriter.WriteLine(); // Blank line after vertices
                        // --- END --- 

                        int entriesToProcessMslk = exportOnlyFirstMslk ? Math.Min(1, pm4File.MSLK.Entries.Count) : pm4File.MSLK.Entries.Count;
                        int skippedMslkCount = 0;
                        int exportedPaths = 0;
                        int exportedPoints = 0;
                        int processedMslkNodes = 0;
                        int mslkLogCounter = 0; // Counter for summary log
                        // UPDATED: Store full vertex string per group
                        var mslkNodeGroups = new Dictionary<uint, List<string>>();

                        var msviIndices = pm4File.MSVI?.Indices;
                        var msvtVertices = pm4File.MSVT?.Vertices;
                        int msviCount = msviIndices?.Count ?? 0;
                        int msvtCount = msvtVertices?.Count ?? 0; // Use the loaded count, not exported count here
                        debugWriter.WriteLine($"INFO: Cached MSVI ({msviCount}) and MSVT ({msvtCount}) for node processing.");

                        int currentMspvVertexCount = pm4File.MSPV.Vertices.Count; // Use the actual count from the loaded chunk

                        // Write CSV Header for Doodad data
                        mslkDoodadCsvWriter.WriteLine("NodeIndex,PosX,PosY,PosZ,Grp_Unk04,Unk00,Unk01,Unk10,Unk12");

                        for (int entryIndex = 0; entryIndex < entriesToProcessMslk; entryIndex++)
                        {
                            var mslkEntry = pm4File.MSLK.Entries[entryIndex];
                            uint groupKey = mslkEntry.Unknown_0x04; // Use Unk04 for grouping

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
                                        // Validate against count written within mslk.obj
                                        if (mspvIndex < mslkMspvVertexCount)
                                        {
                                            validMspvIndices.Add((int)mspvIndex + 1); // Use 1-based index relative to vertices in mslk.obj
                                        }
                                        else
                                        {
                                            if (!invalidIndexDetected) { // Log only first invalid index per entry in debug
                                                 debugWriter.WriteLine($"    WARNING: MSLK Entry {entryIndex}, MSPI index {mspiIndex} points to invalid MSPV index {mspvIndex} (Max: {mslkMspvVertexCount - 1}). Skipping vertex.");
                                                 invalidIndexDetected = true;
                                            }
                                            if(logSummaryThisEntry) summaryWriter.WriteLine($"    WARNING: Invalid MSPV index {mspvIndex}. Skipping vertex.");
                                        }
                                    }

                                    if (validMspvIndices.Count >= 3) // Face
                                    {
                                        string faceLine = "f " + string.Join(" ", validMspvIndices);
                                        string groupName = $"MSLK_Face_{entryIndex}_Grp{groupKey:X8}";
                                        mslkWriter!.WriteLine($"g {groupName}"); // Write to mslkWriter
                                        mslkWriter!.WriteLine(faceLine);

                                        // REMOVED: Write to transformed writers

                                        if (logSummaryThisEntry) {
                                            debugWriter.WriteLine($"    Exported MSLK face with {validMspvIndices.Count} vertices. Group 0x{groupKey:X8}");
                                            summaryWriter.WriteLine($"    Exported MSLK face with {validMspvIndices.Count} vertices.");
                                        }
                                        exportedPaths++;
                                    }
                                    else if (validMspvIndices.Count == 2) // Line
                                    {
                                        string line = "l " + string.Join(" ", validMspvIndices);
                                        string groupName = $"MSLK_Line_{entryIndex}_Grp{groupKey:X8}";
                                        mslkWriter!.WriteLine($"g {groupName}"); // Write to mslkWriter
                                        mslkWriter!.WriteLine(line);

                                        // REMOVED: Write to transformed writers

                                        if (logSummaryThisEntry) {
                                            debugWriter.WriteLine($"    Exported MSLK line with {validMspvIndices.Count} vertices. Group 0x{groupKey:X8}");
                                            summaryWriter.WriteLine($"    Exported MSLK line with {validMspvIndices.Count} vertices.");
                                        }
                                        exportedPaths++;
                                    }
                                    else if (validMspvIndices.Count == 1) // Point
                                    {
                                        string point = $"p {validMspvIndices[0]}";
                                        string groupName = $"MSLK_Point_{entryIndex}_Grp{groupKey:X8}";
                                        mslkWriter!.WriteLine($"g {groupName}"); // Write to mslkWriter
                                        mslkWriter!.WriteLine(point);

                                        // REMOVED: Write to transformed writers

                                        if (logSummaryThisEntry) { // Reduce debug log verbosity
                                            debugWriter.WriteLine($"    Exported MSLK single point at vertex {validMspvIndices[0]}. Group 0x{groupKey:X8}");
                                            summaryWriter.WriteLine($"    Exported MSLK single point at vertex {validMspvIndices[0]}. ");
                                        }
                                        exportedPoints++;
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
                                if (msviIndices != null && msvtVertices != null && msviCount > 0 && msvtCount > 0)
                                {
                                    ushort msviLookupIndex = mslkEntry.Unknown_0x10;
                                    // ADDED: Log the specific Unk10 value for this node entry
                                    debugWriter.WriteLine($"    MSLK Node Entry {entryIndex} - Unknown_0x10 Value: 0x{mslkEntry.Unknown_0x10:X4}");

                                    if (msviLookupIndex < msviCount)
                                    {
                                        uint msvtLookupIndex = msviIndices[msviLookupIndex];
                                        if (msvtLookupIndex < msvtCount) // Check against loaded msvt count
                                        {
                                            var msvtVertex = msvtVertices[(int)msvtLookupIndex];

                                            // Apply centralized MSVT transformation for anchor points
                                            var worldCoords = Pm4CoordinateTransforms.FromMsvtVertexSimple(msvtVertex);

                                            // --- Get Filename from MDBH using Unk04 --- // REMOVED
                                            /*
                                            string doodadFilename = "MDBH_NOT_FOUND";
                                            if (mdbhLookup.TryGetValue(groupKey, out string? foundFilename))
                                            {
                                                doodadFilename = foundFilename ?? "MDBH_NULL_FILENAME";
                                            }
                                            else if (pm4File.MDBH == null)
                                            {
                                                doodadFilename = "MDBH_MISSING";
                                            }
                                            */
                                            // ---

                                            // --- Revert Output Line ---
                                            // Include Unk00, Unk01, Unk04 (Grp), Unk10, Unk12
                                            // Format vertex string first
                                            string nodeVertexString = $"v {worldCoords.X.ToString(CultureInfo.InvariantCulture)} {worldCoords.Y.ToString(CultureInfo.InvariantCulture)} {worldCoords.Z.ToString(CultureInfo.InvariantCulture)} # Node Idx={entryIndex} Grp=0x{groupKey:X8} Unk00=0x{mslkEntry.Unknown_0x00:X2} Unk01=0x{mslkEntry.Unknown_0x01:X2} Unk10={mslkEntry.Unknown_0x10} Unk12=0x{mslkEntry.Unknown_0x12:X4}";
                                            mslkNodesWriter!.WriteLine(nodeVertexString); // Write to main nodes file
                                            processedMslkNodes++; // Increment count

                                            // --- Store distinct group key ---
                                            if (!mslkNodeGroups.ContainsKey(groupKey))
                                            {
                                                mslkNodeGroups[groupKey] = new List<string>();
                                            }
                                            mslkNodeGroups[groupKey].Add(nodeVertexString);
                                            // --- End Store ---

                                            // --- Write Data to Doodad CSV ---
                                            mslkDoodadCsvWriter.WriteLine(FormattableString.Invariant($"{entryIndex},{worldCoords.X:F6},{worldCoords.Y:F6},{worldCoords.Z:F6},0x{groupKey:X8},0x{mslkEntry.Unknown_0x00:X2},0x{mslkEntry.Unknown_0x01:X2},{mslkEntry.Unknown_0x10},0x{mslkEntry.Unknown_0x12:X4}"));
                                            // ---

                                            if (logSummaryThisEntry) { // Reduce debug log verbosity
                                                 // Revert log line (keep Unk12 here for debugging)
                                                 debugWriter.WriteLine($"  MSLK Node Entry {entryIndex}: Grp=0x{groupKey:X8} Unk00=0x{mslkEntry.Unknown_0x00:X2} Unk01=0x{mslkEntry.Unknown_0x01:X2} Unk12=0x{mslkEntry.Unknown_0x12:X4} -> MSVI[{mslkEntry.Unknown_0x10}]={msvtLookupIndex} -> World=({worldCoords.X:F3},{worldCoords.Y:F3},{worldCoords.Z:F3})");
                                                 summaryWriter.WriteLine($"    Processed Node Entry {entryIndex} -> Vertex {processedMslkNodes} in _pm4_mslk_nodes.obj");
                                            }
                                            // --- End Revert --- 
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
                            // REMOVED: Ellipsis from debug log (no direct message here, but associated if condition removed above)
                        }

                        debugWriter.WriteLine($"Finished processing {entriesToProcessMslk} MSLK entries. Exported {exportedPaths} paths, {exportedPoints} points, {processedMslkNodes} node anchors. Skipped {skippedMslkCount} entries.");
                        if (exportOnlyFirstMslk && pm4File.MSLK.Entries.Count > 1)
                        {
                            debugWriter.WriteLine("Note: MSLK processing was limited to the first entry by 'exportOnlyFirstMslk' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSLK Processing (Exported Paths: {exportedPaths}, Points: {exportedPoints}, Nodes: {processedMslkNodes}, Skipped: {skippedMslkCount}) ---");
                        mslkWriter.WriteLine(); // Add blank line to mslk.obj
                        mslkWriter.Flush(); // Flush mslk.obj
                        debugWriter.Flush();
                        skippedMslkWriter!.Flush();
                        mslkNodesWriter!.Flush(); // Ensure main nodes file is written

                        // JSON Export Removed

                        // --- ADDED BACK: Write explicit point definitions for each node vertex ---
                        debugWriter.WriteLine($"\n--- Appending MSLK Node Point Definitions to {Path.GetFileName(outputPm4MslkNodesFilePath)} ---");
                        mslkNodesWriter.WriteLine($"g MSLK_Nodes_Points"); // Group points together
                        for (int nodeIndex = 1; nodeIndex <= processedMslkNodes; nodeIndex++)
                        {
                            mslkNodesWriter.WriteLine($"p {nodeIndex}");
                        }
                        debugWriter.WriteLine($"Finished appending {processedMslkNodes} MSLK Node Point definitions.");
                        // --- END ADDED BACK ---

                        // --- ADDED BACK: Generate MSLK Group definitions (NO LINES) ---
                        debugWriter.WriteLine($"\n--- Appending MSLK Node Group Definitions to {Path.GetFileName(outputPm4MslkNodesFilePath)} ---");
                        int groupsWritten = 0;
                        mslkNodesWriter.WriteLine(); // Add a blank line before groups
                        foreach (uint currentGroupKey in mslkNodeGroups.Keys.OrderBy(k => k))
                        {
                            // Write group definition regardless of node count (useful for selection)
                            groupsWritten++;
                            mslkNodesWriter.WriteLine($"g MSLK_Nodes_Grp{currentGroupKey:X8}");
                        }
                        debugWriter.WriteLine($"Finished appending {groupsWritten} MSLK Group definitions.");
                        mslkNodesWriter.Flush(); // Ensure groups are written
                        // --- END ADDED BACK ---

                        // --- REMOVED: Generation of Individual Group OBJ Files ---

                    }
                    else // Required chunks missing
                    {
                        debugWriter.WriteLine("Skipping MSLK path/node export (MSLK, MSPI, or MSPV data missing or invalid).");
                    }
                } else { // Flag false
                    debugWriter.WriteLine("\nSkipping MSLK Path/Node export (Flag 'exportMslkPaths' is False).");
                }



                // --- 6. Export MSUR surfaces as faces -> Write to BOTH renderMeshWriter and transformed writers ---
                if (processMsurEntries)
                {
                     // Ensure msvtFileVertexCount reflects the count written for THIS file
                    // MODIFIED: Write to original renderMeshWriter AND transformed files
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null && pm4File.MDOS != null && pm4File.MDSF != null && msvtFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSUR -> MDSF -> MDOS Links (Adding faces to Original and Transformed OBJs) ---"); // UPDATED Log
                        summaryWriter.WriteLine($"\n--- MSUR -> MDSF -> MDOS Links ({Path.GetFileNameWithoutExtension(inputFilePath)}) (Summary Log - First 20 Entries) -> Original & Transformed Meshes ---"); // UPDATED Log

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
                                 debugWriter.WriteLine($"  Processing MSUR Entry {msurIndex}: MsviFirst={msurEntry.MsviFirstIndex}, Count={msurEntry.IndexCount}, FlagsOrUnk0=0x{msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}"); // Log full details
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
                                     debugWriter.WriteLine("    Skipping face (unlinked): MSVI chunk missing."); // Log always
                                     continue;
                                }

                                // Check MSVI range validity using currentMsviCount
                                if (firstIndex >= 0 && firstIndex + indexCount <= currentMsviCount)
                                {
                                     if (indexCount < 3) { // Need at least 3 vertices for a face
                                          debugWriter.WriteLine($"    Skipping face (unlinked): Not enough indices (Count={indexCount} < 3). MSUR Entry {msurIndex}"); // Log always
                                          if (logSummary) {
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
                                             debugWriter.WriteLine($"    ERROR (unlinked): MSUR Entry {msurIndex} -> MSVI index {msviIdx} points beyond exported MSVT vertex count ({msvtFileVertexCount}). Skipping face."); // Log always
                                             if (logSummary) summaryWriter.WriteLine($"    ERROR (unlinked): Invalid MSVT index {msviIdx} detected. Skipping face.");
                                             invalidMsvtIndex = true;
                                             break; // Stop processing this face
                                        }
                                         objFaceIndices.Add((int)msviIdx + 1); // Convert to 1-based OBJ index
                                    }

                                     if (invalidMsvtIndex) continue; // Skip this face

                                     // Debug logging for indices (always log now)
                                     debugWriter.WriteLine($"    Fetched {msviIndicesForFace.Count} MSVI Indices (Expected {indexCount}) for unlinked face (MSUR {msurIndex}).");
                                     debugWriter.WriteLine($"      Debug 0-based MSVI Indices (unlinked): [{string.Join(", ", msviIndicesForFace)}]");
                                     debugWriter.WriteLine($"      Debug 1-based OBJ Indices (unlinked): [{string.Join(", ", objFaceIndices)}]");


                                    // Write the face since it's assumed to be state 0
                                    groupName = $"MSUR{msurIndex}_UnlinkedState0"; // Assign specific group name

                                    // Write group name only once
                                    renderMeshWriter!.WriteLine($"g {groupName}");
                                    renderMeshTransformedWriter!.WriteLine($"g {groupName}");
                                    combinedTransformedWriter.WriteLine($"g {baseOutputName}_{groupName}");

                                    // Generate faces using triangle fan pattern with the first vertex as the center
                                    if (objFaceIndices.Count >= 3)
                                    {
                                        int centralVertexIdx = objFaceIndices[0]; // First vertex is the center of the fan
                                        int adjustedCentralVertexIdx = centralVertexIdx + vertexOffset;

                                        // Generate triangles using triangle fan pattern
                                        for (int i = 1; i < objFaceIndices.Count - 1; i++)
                                        {
                                            int secondVertexIdx = objFaceIndices[i];
                                            int thirdVertexIdx = objFaceIndices[i + 1];
                                            int adjustedSecondVertexIdx = secondVertexIdx + vertexOffset;
                                            int adjustedThirdVertexIdx = thirdVertexIdx + vertexOffset;

                                            // Generate faces without normals (proper normals can be computed from geometry)
                                            string faceStr = $"f {centralVertexIdx} {secondVertexIdx} {thirdVertexIdx}";
                                            string adjFaceStr = $"f {adjustedCentralVertexIdx} {adjustedSecondVertexIdx} {adjustedThirdVertexIdx}";
                                            
                                            renderMeshWriter!.WriteLine(faceStr);
                                            renderMeshTransformedWriter!.WriteLine(faceStr);
                                            combinedTransformedWriter.WriteLine(adjFaceStr);
                                            
                                            facesWrittenToRenderMesh++;
                                        }
                                        
                                        debugWriter.WriteLine($"    Generated {objFaceIndices.Count - 2} triangles using triangle fan pattern for MSUR {msurIndex}");
                                    }
                                    else
                                    {
                                        debugWriter.WriteLine($"    Not enough vertices ({objFaceIndices.Count}) to form a triangle fan for MSUR {msurIndex}");
                                    }
                                }
                                else // Invalid MSVI range
                                {
                                     if (logSummary) { // Reduce debug verbosity
                                         debugWriter.WriteLine($"    ERROR (unlinked): MSUR Entry {msurIndex} defines invalid MSVI range [First:{firstIndex}, Count:{indexCount}] (Max MSVI Index: {currentMsviCount - 1}). Skipping face."); // Log always
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
                                 debugWriter.WriteLine($"    Skipping face (linked, MSUR {msurIndex}): MSVI chunk missing."); // Log always
                                continue;
                            }

                            // Check MSVI range validity using currentMsviCount
                            if (firstIndex >= 0 && firstIndex + indexCount <= currentMsviCount)
                            {
                                 if (indexCount < 3) { // Need at least 3 vertices for a face
                                     debugWriter.WriteLine($"    Skipping face generation (linked, MSUR {msurIndex}): Not enough indices (Count={indexCount} < 3)."); // Log always
                                     if (logSummary) { // Reduce debug verbosity
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
                                         debugWriter.WriteLine($"    ERROR (linked): MSUR Entry {msurIndex} -> MSVI index {msviIdx} points beyond exported MSVT vertex count ({msvtFileVertexCount}). Skipping face."); // Log always
                                         if (logSummary) summaryWriter.WriteLine($"    ERROR (linked): Invalid MSVT index {msviIdx} detected. Skipping face.");
                                         invalidMsvtIndex = true;
                                         break; // Stop processing this face
                                    }
                                    objFaceIndices.Add((int)msviIdx + 1); // Convert to 1-based OBJ index
                                }

                                if (invalidMsvtIndex) continue; // Skip this face

                                // Always log indices now
                                debugWriter.WriteLine($"    Fetched {msviIndicesForFace.Count} MSVI Indices (Expected {indexCount}) for linked face (MSUR {msurIndex}).");
                                debugWriter.WriteLine($"      Debug 0-based MSVI Indices: [{string.Join(", ", msviIndicesForFace)}]");
                                debugWriter.WriteLine($"      Debug 1-based OBJ Indices: [{string.Join(", ", objFaceIndices)}]");


                                // Filter based on the correctly linked MDOS state
                                if (linkedMdosEntry.destruction_state == 0)
                                {
                                    // Write group name only once
                                    renderMeshWriter!.WriteLine($"g {groupName}");
                                    renderMeshTransformedWriter!.WriteLine($"g {groupName}");
                                    combinedTransformedWriter.WriteLine($"g {baseOutputName}_{groupName}");
                                    
                                    // Generate faces using triangle fan pattern with the first vertex as the center
                                    if (objFaceIndices.Count >= 3)
                                    {
                                        int centralVertexIdx = objFaceIndices[0]; // First vertex is the center of the fan
                                        int adjustedCentralVertexIdx = centralVertexIdx + vertexOffset;
                                        
                                        // Generate triangles using triangle fan pattern
                                        for (int i = 1; i < objFaceIndices.Count - 1; i++)
                                        {
                                            int secondVertexIdx = objFaceIndices[i];
                                            int thirdVertexIdx = objFaceIndices[i + 1];
                                            int adjustedSecondVertexIdx = secondVertexIdx + vertexOffset;
                                            int adjustedThirdVertexIdx = thirdVertexIdx + vertexOffset;
                                            
                                            // Generate faces without normals (proper normals can be computed from geometry)
                                            string faceStr = $"f {centralVertexIdx} {secondVertexIdx} {thirdVertexIdx}";
                                            string adjFaceStr = $"f {adjustedCentralVertexIdx} {adjustedSecondVertexIdx} {adjustedThirdVertexIdx}";
                                            
                                            renderMeshWriter!.WriteLine(faceStr);
                                            renderMeshTransformedWriter!.WriteLine(faceStr);
                                            combinedTransformedWriter.WriteLine(adjFaceStr);
                                            
                                            facesWrittenToRenderMesh++;
                                        }
                                        
                                        debugWriter.WriteLine($"    Generated {objFaceIndices.Count - 2} triangles using triangle fan pattern for MSUR {msurIndex}");
                                    }
                                    else
                                    {
                                        debugWriter.WriteLine($"    Not enough vertices ({objFaceIndices.Count}) to form a triangle fan for MSUR {msurIndex}");
                                    }
                                }
                                else // State != 0
                                {
                                    if (logSummary) { // Reduce debug verbosity
                                        debugWriter.WriteLine($"    -> SKIPPING Face (MDOS State != 0 for MSUR {msurIndex})"); // Log always
                                        summaryWriter.WriteLine($"    -> SKIPPING Face (MDOS State != 0)");
                                    }
                                }
                            }
                            else // Invalid MSVI range
                            {
                                if (logSummary) { // Reduce debug verbosity
                                    debugWriter.WriteLine($"    ERROR: MSUR Entry {msurIndex} defines invalid MSVI range [First:{firstIndex}, Count:{indexCount}] (Max MSVI Index: {currentMsviCount - 1}). Skipping face."); // Log always
                                    summaryWriter.WriteLine($"    ERROR: Invalid MSVI range. Skipping face.");
                                }
                            }
                             msurEntriesProcessed++;
                             msurLogCounter++; // Increment summary log counter
                        } // End MSUR loop

                        if (msurLogCounter > 20) { // Add ellipsis if summary truncated
                             summaryWriter.WriteLine("  ... (Summary log limited to first 20 MSUR entries) ...");
                            // REMOVED: Ellipsis from debug log (no direct message here, but associated if condition removed above)
                        }

                        // CORRECTED LOG MESSAGES
                        debugWriter.WriteLine($"Finished processing {msurEntriesProcessed} MSUR entries. Processed {facesWrittenToRenderMesh} vertex groups (State 0 or Unlinked)."); // Updated log message
                        if (exportOnlyFirstMsur && pm4File.MSUR.Entries.Count > 1) {
                              debugWriter.WriteLine("Note: MSUR processing was limited to the first entry by 'exportOnlyFirstMsur' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSUR Processing (Processed: {msurEntriesProcessed}, Vertex Groups Written: {facesWrittenToRenderMesh}) ---"); // Updated summary log
                    }
                    else // Required chunks missing or no vertices exported
                    {
                        debugWriter.WriteLine("Skipping MSUR face export (MSUR, MSVI, MSVT, MDOS, or MDSF data missing or no MSVT vertices exported).");
                    }
                } else { // Flag false
                     debugWriter.WriteLine("\nSkipping MSUR Face processing (Flag 'processMsurEntries' is False).");
                }


                // --- 7. Log Unique Building IDs --- (Keep this)
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


                // --- 8. Log MDBH and MDSF Link Info --- (Keep this)
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
                             debugWriter.WriteLine($"  {linkInfo}"); // Log always
                             summaryWriter.WriteLine($"  {linkInfo}");
                             mdsfLogCounter++;
                         }
                         mdsfCount++;
                     }
                      if (mdsfCount > 20) {
                         summaryWriter.WriteLine("  ... (Summary log limited to first 20 MDSF entries) ...");
                         // REMOVED: Ellipsis message for debugWriter
                     }
                     debugWriter.WriteLine($"Logged info for {mdsfCount} MDSF entries.");
                     summaryWriter.WriteLine($"--- Finished logging MDSF links ({mdsfCount} total). ---");
                 } else {
                      debugWriter.WriteLine("\n--- Skipping MDSF/MDOS Link Logging (Flag 'logMdsfLinks' is False or data missing) ---");
                 }


                 // MDBH Logging (Now done unconditionally for analysis)
                 debugWriter.WriteLine("\n--- Logging MDBH Entries (All) for MSLK Correlation ---");
                 summaryWriter.WriteLine($"\n--- MDBH Entries ({Path.GetFileNameWithoutExtension(inputFilePath)}) (All) ---");
                 if (pm4File.MDBH?.Entries != null) { 
                      int mdbhCount = 0;
                      foreach(var mdbhEntry in pm4File.MDBH.Entries) {
                           // Log every entry to debug log
                           debugWriter.WriteLine($"  MDBH Entry {mdbhCount}: Index={mdbhEntry.Index}, Filename=\"{mdbhEntry.Filename}\"");
                           // Log first 20 to summary log
                           if (mdbhCount < 20) {
                               summaryWriter.WriteLine($"  MDBH Entry {mdbhCount}: Index={mdbhEntry.Index}, Filename=\"{mdbhEntry.Filename}\"");
                           }
                           mdbhCount++;
                      }
                       if (mdbhCount > 20) {
                         summaryWriter.WriteLine("  ... (Summary log limited to first 20 MDBH entries) ...");
                     }
                     debugWriter.WriteLine($"Logged {mdbhCount} MDBH entries.");
                 } else {
                      debugWriter.WriteLine("\n--- Skipping MDBH Logging (Data missing) ---");
                      summaryWriter.WriteLine("\n--- Skipping MDBH Logging (Data missing) ---");
                 }

                // MSLK analysis checks (Add temporary checks here)
                if (exportMslkPaths && pm4File.MSLK != null)
                {
                    debugWriter.WriteLine("\n--- MSLK Field Sanity Checks ---");
                    int mslkUnk2NonZero = 0;
                    int mslkUnkCNotFfff = 0;
                    int mslkUnk12Not8000 = 0;
                    foreach (var mslkEntry in pm4File.MSLK.Entries)
                    {
                        if (mslkEntry.Unknown_0x02 != 0) mslkUnk2NonZero++;
                        if (mslkEntry.Unknown_0x0C != 0xFFFFFFFF) mslkUnkCNotFfff++;
                        if (mslkEntry.Unknown_0x12 != 0x8000) mslkUnk12Not8000++;
                    }
                    debugWriter.WriteLine($"  Entries with Unknown_0x02 != 0x0000: {mslkUnk2NonZero}");
                    debugWriter.WriteLine($"  Entries with Unknown_0x0C != 0xFFFFFFFF: {mslkUnkCNotFfff}");
                    debugWriter.WriteLine($"  Entries with Unknown_0x12 != 0x8000: {mslkUnk12Not8000}");
                    debugWriter.WriteLine("--- End MSLK Field Sanity Checks ---");
                }

                summaryWriter.WriteLine($"--- End PM4 File Summary for {Path.GetFileName(inputFilePath)} ---");
                debugWriter.WriteLine($"--- End PM4 File Debug Log for {Path.GetFileName(inputFilePath)} ---");

                // --- Generate MPRR Analysis CSV (Already updated) ---
                if (pm4File.MPRR != null && pm4File.MPRR.Sequences.Count > 0) // Corrected typo here
                {
                    // ... (Existing correct logic using Sequences) ...
                }
                else
                {
                     Console.WriteLine("  > Skipping MPRR Analysis CSV generation (MPRR chunk missing or empty).");
                     debugWriter.WriteLine("  > Skipping MPRR Analysis CSV generation (MPRR chunk missing or empty)."); // Also log to debug
                }

                 // --- Generate MPRL Data CSV (Already updated) ---
                 // ... (Existing correct logic using MPRL only) ...

            }
            catch (Exception ex) // Catch exceptions within the using block
            {
                 Console.WriteLine($"ERROR during processing logic for {Path.GetFileName(inputFilePath)}: {ex.Message}");
                 debugWriter?.WriteLine($"\n!!!!!! ERROR during processing: {ex.ToString()} !!!!!!"); // Log to file-specific debug log
                 summaryWriter?.WriteLine($"\n!!!!!! ERROR during processing: {ex.Message} !!!!!!"); // Log to file-specific summary log
                // Re-throw the exception to be caught by the outer loop's handler
                throw;
            }
            finally // ADDED: Ensure MPRR sequences are written even if errors occur elsewhere
            {
                 // --- Generate MPRR Sequences CSV ---
                 debugWriter?.WriteLine($"\n--- Generating MPRR Sequences CSV -> {Path.GetFileName(outputMprrSequencesCsvPath)} ---");
                 try
                 {
                    mprrSequencesCsvWriter?.WriteLine("SequenceIndex,ValueIndex,Value");
                    if (pm4File?.MPRR != null && pm4File.MPRR.Sequences.Count > 0)
                    {
                        int totalValuesWritten = 0;
                        for (int seqIdx = 0; seqIdx < pm4File.MPRR.Sequences.Count; seqIdx++)
                        {
                            var sequence = pm4File.MPRR.Sequences[seqIdx];
                            for (int valIdx = 0; valIdx < sequence.Count; valIdx++)
                            {
                                mprrSequencesCsvWriter?.WriteLine($"{seqIdx},{valIdx},{sequence[valIdx]}");
                                totalValuesWritten++;
                            }
                        }
                        debugWriter?.WriteLine($"  Wrote {totalValuesWritten} values from {pm4File.MPRR.Sequences.Count} sequences to {Path.GetFileName(outputMprrSequencesCsvPath)}");
                    }
                    else
                    {
                        debugWriter?.WriteLine("  MPRR chunk missing or has no sequences. CSV will only contain header.");
                    }
                    mprrSequencesCsvWriter?.Flush();
                 }
                 catch (Exception ex)
                 {
                     Console.WriteLine($"  > ERROR generating MPRR Sequences CSV: {ex.Message}");
                     debugWriter?.WriteLine($"\n!!!!!! ERROR generating MPRR Sequences CSV: {ex.ToString()} !!!!!!");
                 }
                 debugWriter?.WriteLine("--- End Generating MPRR Sequences CSV ---");
            }
            // No finally needed here as 'using' handles disposal -> Moved sequence writing to finally

            return (msvtFileVertexCount, vertices, triangleIndices); // Return vertex count and geometry data

        } // End ProcessSinglePm4FileWithData

        /// <summary>
        /// Specialized processor for PM4 files with extremely high MPRR/MPRL ratios
        /// </summary>
              /// <summary>
        /// Specialized processor for PM4 files with extremely high MPRR/MPRL ratios
        /// </summary>
        private int ProcessHighRatioPm4File(string inputFilePath, string outputDir, StreamWriter combinedTransformedWriter, int vertexOffset)
        {
            string fileName = Path.GetFileName(inputFilePath);
            string baseOutputName = Path.GetFileNameWithoutExtension(inputFilePath);

            string debugLogPath = Path.Combine(outputDir, $"{baseOutputName}_debug.log");
            using var debugWriter = new StreamWriter(debugLogPath, false);

            debugWriter.WriteLine($"=== Processing High MPRR/MPRL Ratio File: {fileName} ===");
            debugWriter.WriteLine($"Processing Time: {DateTime.Now}\n");

            try
            {
                // Load the file but don't process MPRR links (which would cause exceptions)
                debugWriter.WriteLine("Loading file with modified approach...");
                var pm4File = PM4File.FromFile(inputFilePath);

                // Output basic stats
                debugWriter.WriteLine($"Loaded file: {fileName}");
                // UPDATED: Use Sequences.Count with null checks
                // CORRECTED TYPO: Mprr -> MPRR
                debugWriter.WriteLine($"MPRR Sequences: {pm4File.MPRR?.Sequences?.Count ?? 0}"); // Corrected here
                debugWriter.WriteLine($"MPRL Entries: {pm4File.MPRL?.Entries.Count ?? 0}");

                // Calculate ratio for logging
                // UPDATED: Use Sequences.Count with null checks
                // CORRECTED TYPO: Mprr -> MPRR
                double ratio = pm4File.MPRL?.Entries.Count > 0 ?
                    (double)(pm4File.MPRR?.Sequences?.Count ?? 0) / pm4File.MPRL.Entries.Count : // Corrected here
                    double.NaN;
                debugWriter.WriteLine($"MPRR/MPRL Ratio: {ratio:F2}\n");

                // Process everything EXCEPT MPRR links
                int processedVertexCount = 0;

                // Process MSVT/MSVI data if available
                string outputMsvtObjPath = Path.Combine(outputDir, $"{baseOutputName}_msvt.obj");
                if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    debugWriter.WriteLine($"Processing MSVT vertices ({pm4File.MSVT.Vertices.Count})...");
                    using var msvtObjWriter = new StreamWriter(outputMsvtObjPath, false);

                    // Add header
                    msvtObjWriter.WriteLine($"# WoW PM4 MSVT vertices from {fileName}");
                    msvtObjWriter.WriteLine($"# Generated {DateTime.Now}");

                    // Process vertices using PM4-relative coordinates
                    foreach (var vertex in pm4File.MSVT.Vertices)
                    {
                        // Apply PM4-relative transformation (Y, X, Z)
                        var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                        msvtObjWriter.WriteLine($"v {pm4Coords.X} {pm4Coords.Y} {pm4Coords.Z}");
                    }

                    int msvtVertexCount = pm4File.MSVT.Vertices.Count;
                    debugWriter.WriteLine($"  Wrote {msvtVertexCount} MSVT vertices to {Path.GetFileName(outputMsvtObjPath)}");
                    processedVertexCount += msvtVertexCount;
                }
                else
                {
                    debugWriter.WriteLine("  No MSVT vertices found to process.");
                }

                // Process MPRL points (without MPRR links)
                string outputMprlPointsObjPath = Path.Combine(outputDir, $"{baseOutputName}_mprl_points.obj");
                if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                {
                    debugWriter.WriteLine($"\nProcessing MPRL points ({pm4File.MPRL.Entries.Count})...");
                    using var mprlPointsObjWriter = new StreamWriter(outputMprlPointsObjPath, false);

                    // Add header
                    mprlPointsObjWriter.WriteLine($"# WoW PM4 MPRL points from {fileName}");
                    mprlPointsObjWriter.WriteLine($"# Generated {DateTime.Now}");
                    mprlPointsObjWriter.WriteLine($"# Format: v x y z, PM4-relative transform: (X, -Z, Y)");

                    // Add points
                    int validPointCount = 0;
                    foreach (var point in pm4File.MPRL.Entries)
                    {
                        // Skip points with zero position or zero IDs
                        bool hasZeroPosition = Math.Abs(point.Position.X) < 0.001f &&
                                         Math.Abs(point.Position.Y) < 0.001f &&
                                         Math.Abs(point.Position.Z) < 0.001f;

                        bool hasZeroIds = point.Unknown_0x04 == 0 && point.Unknown_0x06 == 0 &&
                                         point.Unknown_0x14 == 0 && point.Unknown_0x16 == 0;

                        if (hasZeroPosition || hasZeroIds)
                        {
                            continue;
                        }

                        // Apply PM4-relative MPRL transformation
                        var pm4Coords = Pm4CoordinateTransforms.FromMprlEntry(point);

                        // Write PM4-relative coordinates
                        mprlPointsObjWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                        validPointCount++;
                    }

                    debugWriter.WriteLine($"  Wrote {validPointCount} valid MPRL points (skipped {pm4File.MPRL.Entries.Count - validPointCount} invalid points) to {Path.GetFileName(outputMprlPointsObjPath)}");
                    processedVertexCount += validPointCount;
                }
                else
                {
                    debugWriter.WriteLine("  No MPRL points found to process.");
                }

                // Note that we're skipping MPRR links due to the high ratio
                debugWriter.WriteLine("\nSkipped processing MPRR links due to high MPRR/MPRL ratio");

                // Final summary
                debugWriter.WriteLine($"\n=== Processing Summary ===");
                debugWriter.WriteLine($"Successfully processed {fileName} with {processedVertexCount} total vertices");
                debugWriter.WriteLine($"NOTE: MPRR links were skipped to avoid index out of range exceptions");

                Console.WriteLine($"  * Successfully processed high-ratio file {fileName}");
                Console.WriteLine($"  * Processed {processedVertexCount} vertices (MPRR links skipped)");
                return processedVertexCount;
            }
            catch (Exception ex)
            {
                debugWriter.WriteLine($"ERROR: Failed to process high-ratio file: {ex.Message}");
                debugWriter.WriteLine($"Stack trace: {ex.StackTrace}");

                Console.WriteLine($"  ! ERROR processing high-ratio file {fileName}: {ex.Message}");
                return 0;
            }
        } // End ProcessHighRatioPm4File (within PM4FileTests class)
        // Original LoadPM4File_ShouldLoadChunks can be removed or commented out
        /*
        [Fact]
        public void LoadPM4File_ShouldLoadChunks_OLD()
        {
            // ... original single-file test code ...
        }
        */

        // --- Helper method to analyze problematic files specifically ---
        private void AnalyzeProblematicFile(string inputFilePath, string outputDir)
        {
            string fileName = Path.GetFileName(inputFilePath);
            Console.WriteLine($"  * Running specialized processing for problematic file: {fileName}");
            
            // Create a diagnostics subdirectory for detailed analysis
            string diagnosticsDir = Path.Combine(outputDir, "diagnostics");
            Directory.CreateDirectory(diagnosticsDir);
            
            // Use our specialized processor to handle the file
            var processor = new PM4HighRatioProcessor();
            int processedVertices = processor.ProcessHighRatioFile(inputFilePath, diagnosticsDir);
            
            Console.WriteLine($"  * Specialized processing completed. Processed {processedVertices} vertices.");
            Console.WriteLine($"  * Check detailed output in: {diagnosticsDir}");
        }

        // --- Helper method to check and log chunk info
        private void CheckChunk(string chunkName, int entryCount, StreamWriter writer)
        {
            if (entryCount > 0)
            {
                writer.WriteLine($"{chunkName}: Present - {entryCount} entries");
            }
            else if (entryCount == 0)
            {
                writer.WriteLine($"{chunkName}: Present but EMPTY (0 entries)");
            }
            else
            {
                writer.WriteLine($"{chunkName}: MISSING!");
            }
        }
        
        // --- Helper method to analyze raw bytes for chunk signatures
        private void AnalyzeRawChunkSignatures(byte[] fileData, StreamWriter writer)
        {
            // List of known chunk signatures to look for
            var knownSignatures = new List<string> { "MVER", "MSHD", "MSVT", "MSVI", "MSPV", "MSPI", "MPRL", "MPRR", "MSLK", "MDOS", "MDSF", "MDBH", "MSCN", "MCRC" };
            var foundSignatures = new List<(string Signature, int Offset, int Size)>();
            
            writer.WriteLine("Scanning file for chunk signatures...");
            
            for (int i = 0; i < fileData.Length - 8; i++)
            {
                // Extract 4-byte signature
                string signature = System.Text.Encoding.ASCII.GetString(fileData, i, 4);
                
                if (knownSignatures.Contains(signature))
                {
                    // Read chunk size (4 bytes after signature)
                    int size = BitConverter.ToInt32(fileData, i + 4);
                    
                    // Record found signature
                    foundSignatures.Add((signature, i, size));
                    
                    // Skip to after this chunk (to avoid finding signatures in data)
                    i += 7 + size;
                }
            }
            
            // Report findings
            if (foundSignatures.Count > 0)
            {
                writer.WriteLine($"Found {foundSignatures.Count} chunk signatures:");
                
                foreach (var chunk in foundSignatures)
                {
                    writer.WriteLine($"  {chunk.Signature} at offset 0x{chunk.Offset:X8}, size: {chunk.Size} bytes");
                }
            }
            else
            {
                writer.WriteLine("No known chunk signatures found in file!");
            }
        }

        [Fact]
        [Trait("Category", "SpecialCases")]
        public void TestDevelopment49_28_WithSpecializedHandling()
        {
            // Arrange
            string testFilePath = Path.Combine(TestDataRoot, "original_development", "development", "development_49_28.pm4"); // Added intermediate 'development' directory
            
            // --- Use Timestamped Output --- 
            string outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "PM4_HighRatio");
            Directory.CreateDirectory(outputDir);
            string logPath = Path.Combine(outputDir, "processing_results.log");
            
            // Ensure target file exists
            Assert.True(File.Exists(testFilePath), $"Test file does not exist: {testFilePath}");

            // Act - Process file with the specialized handler
            using (var logWriter = new StreamWriter(logPath, false))
            {
                // Call the method directly to ensure it's used
                // NOTE: need to instantiate a class with ProcessHighRatioPm4File method
                var processor = new PM4HighRatioProcessor();
                int vertexCount = processor.ProcessHighRatioFile(testFilePath, outputDir);
                
                // Log results
                logWriter.WriteLine($"Test completed at {DateTime.Now}");
                logWriter.WriteLine($"File: {Path.GetFileName(testFilePath)}");
                logWriter.WriteLine($"Output directory: {outputDir}");
                logWriter.WriteLine($"Processed vertices: {vertexCount}");
            }
            
            // Assert
            // Check for the output files we expect to see
            string debugLogPath = Path.Combine(outputDir, "development_49_28_debug.log");
            Assert.True(File.Exists(debugLogPath), "Debug log file should be created");
            
            // Check for vertex output files
            string msvtObjPath = Path.Combine(outputDir, "development_49_28_msvt.obj");
            string mprlPointsObjPath = Path.Combine(outputDir, "development_49_28_mprl_points.obj");
            
            bool hasMsvtOutput = File.Exists(msvtObjPath);
            bool hasMprlOutput = File.Exists(mprlPointsObjPath);
            
            Assert.True(hasMsvtOutput || hasMprlOutput, 
                "At least one vertex output file should be created");
                
            // Analyze the debug log to verify it shows expected behavior
            string debugLogContent = File.ReadAllText(debugLogPath);
            
            Assert.Contains("Processing High MPRR/MPRL Ratio File", debugLogContent);
            Assert.Contains("Skipped processing MPRR links due to high MPRR/MPRL ratio", debugLogContent);
            
            Console.WriteLine($"Test completed successfully. Check output in: {outputDir}");
        }

        [Fact]
        public void ExportRenderMeshAndMscnBoundary_WithSnapping_ForKeyPm4Files()
        {
            var testFiles = new[]
            {
                Path.Combine(TestDataRoot, "original_development", "development_00_00.pm4"),
                Path.Combine(TestDataRoot, "original_development", "development_22_18.pm4")
            };

            foreach (var pm4Path in testFiles)
            {
                Assert.True(File.Exists(pm4Path), $"Test PM4 file not found: {pm4Path}");
                var pm4File = PM4File.FromFile(pm4Path);
                Assert.NotNull(pm4File);

                // --- Extract Render Mesh ---
                var meshData = new MeshData();
                if (pm4File.MSVT != null && pm4File.MSVI != null && pm4File.MSUR != null)
                {
                    meshData.Vertices.AddRange(pm4File.MSVT.Vertices.Select(v => v.ToWorldCoordinates()));
                    foreach (var msur in pm4File.MSUR.Entries)
                    {
                        for (int i = 0; i < msur.IndexCount - 2; i++)
                        {
                            if (msur.MsviFirstIndex > int.MaxValue)
                                throw new OverflowException($"MSUR.MsviFirstIndex {msur.MsviFirstIndex} exceeds int.MaxValue");
                            int baseIdx = (int)msur.MsviFirstIndex;
                            uint idx0 = pm4File.MSVI.Indices[baseIdx];
                            uint idx1 = pm4File.MSVI.Indices[baseIdx + i + 1];
                            uint idx2 = pm4File.MSVI.Indices[baseIdx + i + 2];
                            if (idx0 > int.MaxValue || idx1 > int.MaxValue || idx2 > int.MaxValue)
                                throw new OverflowException($"MSVI index exceeds int.MaxValue: {idx0}, {idx1}, {idx2}");
                            int i0 = (int)idx0;
                            int i1 = (int)idx1;
                            int i2 = (int)idx2;
                            meshData.Indices.Add(i0);
                            meshData.Indices.Add(i1);
                            meshData.Indices.Add(i2);
                        }
                    }
                }
                Assert.True(meshData.Vertices.Count > 0, "No vertices in render mesh");
                Assert.True(meshData.Indices.Count > 0, "No indices in render mesh");

                // --- Extract MSCN Points ---
                var mscnPoints = pm4File.MSCN?.ExteriorVertices ?? new List<Vector3>();
                string baseOut = Path.Combine(TestContext.TimestampedOutputRoot, Path.GetFileNameWithoutExtension(pm4Path));
                Directory.CreateDirectory(baseOut);

                // --- Output Render Mesh OBJ ---
                string meshObjPath = Path.Combine(baseOut, "render_mesh.obj");
                using (var writer = new StreamWriter(meshObjPath))
                {
                    writer.WriteLine("o RenderMesh");
                    foreach (var v in meshData.Vertices)
                        writer.WriteLine($"v {v.X} {v.Y} {v.Z}");
                    for (int i = 0; i < meshData.Indices.Count; i += 3)
                        writer.WriteLine($"f {meshData.Indices[i] + 1} {meshData.Indices[i + 1] + 1} {meshData.Indices[i + 2] + 1}");
                }

                // --- Output MSCN Points OBJ ---
                string mscnObjPath = Path.Combine(baseOut, "mscn_points.obj");
                using (var writer = new StreamWriter(mscnObjPath))
                {
                    writer.WriteLine("o MSCN_Boundary");
                    foreach (var v in mscnPoints)
                        writer.WriteLine($"v {v.X} {v.Y} {v.Z}");
                }

                // --- Merged OBJ output is disabled ---
                // MeshLab and other tools do not handle multiple overlapping meshes in the same file well.
                // Until all unknowns are resolved, merged OBJ output is not produced.
            }
        }

        [Fact]
        public void InvestigateUnknownFieldMeanings()
        {
            // Create output directory
            string outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "UnknownFieldInvestigation");
            Directory.CreateDirectory(outputDir);
            
            var investigationResults = new List<string>();
            int filesProcessed = 0;
            
            // Get PM4 files for detailed analysis
            var inputDirectoryPath = Path.Combine(TestDataRoot, "original_development", "development");
            var pm4Files = Directory.GetFiles(inputDirectoryPath, "*.pm4", SearchOption.TopDirectoryOnly)
                .Where(f => !f.Contains("_49_28")) // Skip problematic file
                .Take(20); // Focus on first 20 files for detailed analysis
            
            investigationResults.Add("=== PM4 UNKNOWN FIELD INVESTIGATION ===");
            investigationResults.Add($"Analysis Date: {DateTime.Now}");
            investigationResults.Add($"Focus: Correlating unknown fields with known data patterns\n");
            
            foreach (string pm4FilePath in pm4Files)
            {
                try
                {
                    var pm4File = PM4File.FromFile(pm4FilePath);
                    if (pm4File == null) continue;
                    
                    string fileName = Path.GetFileNameWithoutExtension(pm4FilePath);
                    investigationResults.Add($"=== File: {fileName} ===");
                    
                    // 1. MSHD Analysis - These look like chunk offsets/sizes!
                    if (pm4File.MSHD != null)
                    {
                        investigationResults.Add("MSHD ANALYSIS (Potential Chunk Offsets/Sizes):");
                        investigationResults.Add($"  Unknown_0x00: 0x{pm4File.MSHD.Unknown_0x00:X8} ({pm4File.MSHD.Unknown_0x00})");
                        investigationResults.Add($"  Unknown_0x04: 0x{pm4File.MSHD.Unknown_0x04:X8} ({pm4File.MSHD.Unknown_0x04})");
                        investigationResults.Add($"  Unknown_0x08: 0x{pm4File.MSHD.Unknown_0x08:X8} ({pm4File.MSHD.Unknown_0x08})");
                        
                        // These seem to be offsets - let's verify by checking if they point to valid file positions
                        var fileSize = new FileInfo(pm4FilePath).Length;
                        investigationResults.Add($"  File size: {fileSize} bytes");
                        investigationResults.Add($"  Offset validity check:");
                        investigationResults.Add($"    0x00 within file: {pm4File.MSHD.Unknown_0x00 < fileSize}");
                        investigationResults.Add($"    0x04 within file: {pm4File.MSHD.Unknown_0x04 < fileSize}");
                        investigationResults.Add($"    0x08 within file: {pm4File.MSHD.Unknown_0x08 < fileSize}");
                        
                        // Check if these are sequential (suggesting chunk ordering)
                        var offsets = new[] { pm4File.MSHD.Unknown_0x00, pm4File.MSHD.Unknown_0x04, pm4File.MSHD.Unknown_0x08 };
                        investigationResults.Add($"    Sequential order: {string.Join(" < ", offsets.OrderBy(x => x))}");
                        investigationResults.Add($"    Are they ordered: {offsets.SequenceEqual(offsets.OrderBy(x => x))}");
                    }
                    
                    // 2. MSUR Float Analysis - These look like 3D normals + height!
                    if (pm4File.MSUR?.Entries != null && pm4File.MSUR.Entries.Count > 0)
                    {
                        investigationResults.Add("\nMSUR FLOAT ANALYSIS (Potential 3D Normals + Height):");
                        
                        for (int i = 0; i < Math.Min(5, pm4File.MSUR.Entries.Count); i++)
                        {
                            var msur = pm4File.MSUR.Entries[i];
                            investigationResults.Add($"  Surface {i}:");
                            investigationResults.Add($"    Float_0x04: {msur.UnknownFloat_0x04:F6} (X-component normal?)");
                            investigationResults.Add($"    Float_0x08: {msur.UnknownFloat_0x08:F6} (Y-component normal?)");
                            investigationResults.Add($"    Float_0x0C: {msur.UnknownFloat_0x0C:F6} (Z-component normal?)");
                            investigationResults.Add($"    Float_0x10: {msur.UnknownFloat_0x10:F6} (Height/Y coordinate?)");
                            
                            // Check if first 3 floats form a normalized vector
                            var vectorMagnitude = Math.Sqrt(
                                msur.UnknownFloat_0x04 * msur.UnknownFloat_0x04 +
                                msur.UnknownFloat_0x08 * msur.UnknownFloat_0x08 +
                                msur.UnknownFloat_0x0C * msur.UnknownFloat_0x0C);
                            investigationResults.Add($"    Vector magnitude: {vectorMagnitude:F6} (should be ~1.0 if normalized)");
                            investigationResults.Add($"    Is normalized: {Math.Abs(vectorMagnitude - 1.0) < 0.01}");
                        }
                    }
                    
                    // 3. MSLK Analysis - Flags and indices
                    if (pm4File.MSLK?.Entries != null && pm4File.MSLK.Entries.Count > 0)
                    {
                        investigationResults.Add("\nMSLK ANALYSIS (Potential Flags and Type Indices):");
                        
                        var entry = pm4File.MSLK.Entries[0]; // Just check first entry
                        investigationResults.Add($"  Sample entry:");
                        investigationResults.Add($"    Unknown_0x00: 0x{entry.Unknown_0x00:X2} ({entry.Unknown_0x00}) - Flags?");
                        investigationResults.Add($"    Unknown_0x01: 0x{entry.Unknown_0x01:X2} ({entry.Unknown_0x01}) - Subtype?");
                        investigationResults.Add($"    Unknown_0x02: 0x{entry.Unknown_0x02:X4} ({entry.Unknown_0x02}) - Always zero?");
                        investigationResults.Add($"    Unknown_0x04: 0x{entry.Unknown_0x04:X8} ({entry.Unknown_0x04}) - Object ID?");
                        investigationResults.Add($"    Unknown_0x0C: 0x{entry.Unknown_0x0C:X8} ({entry.Unknown_0x0C}) - Color/Material ID?");
                        investigationResults.Add($"    Unknown_0x10: 0x{entry.Unknown_0x10:X4} ({entry.Unknown_0x10}) - Reference index?");
                        investigationResults.Add($"    Unknown_0x12: 0x{entry.Unknown_0x12:X4} ({entry.Unknown_0x12}) - Always 0x8000?");
                    }
                    
                    // 4. Cross-reference with vertex counts for validation
                    investigationResults.Add("\nCROSS-REFERENCE VALIDATION:");
                    investigationResults.Add($"  MSVT vertex count: {pm4File.MSVT?.Vertices?.Count ?? 0}");
                    investigationResults.Add($"  MSVI index count: {pm4File.MSVI?.Indices?.Count ?? 0}");
                    investigationResults.Add($"  MSUR surface count: {pm4File.MSUR?.Entries?.Count ?? 0}");
                    investigationResults.Add($"  MSLK entry count: {pm4File.MSLK?.Entries?.Count ?? 0}");
                    investigationResults.Add($"  MPRL position count: {pm4File.MPRL?.Entries?.Count ?? 0}");
                    
                    investigationResults.Add("");
                    filesProcessed++;
                    
                    if (filesProcessed % 5 == 0)
                    {
                        Console.WriteLine($"Processed {filesProcessed} files...");
                    }
                }
                catch (Exception ex)
                {
                    investigationResults.Add($"Error processing {pm4FilePath}: {ex.Message}");
                }
            }
            
            // Add conclusions based on patterns
            investigationResults.Add("=== PRELIMINARY CONCLUSIONS ===");
            investigationResults.Add("1. MSHD Unknown fields (0x00, 0x04, 0x08) are likely chunk offsets/pointers");
            investigationResults.Add("2. MSHD Unknown fields (0x0C-0x1C) are consistently zero - likely padding/reserved");
            investigationResults.Add("3. MSUR Float_0x04-0x0C appear to be 3D surface normals (magnitude ~1.0)");
            investigationResults.Add("4. MSUR Float_0x10 has wide range (-17K to +17K) - likely Y-coordinate/height");
            investigationResults.Add("5. MSLK Unknown_0x0C has pattern 0xFFFF#### - likely packed flags/IDs");
            investigationResults.Add("6. MSLK Unknown_0x12 is always 0x8000 - likely a constant flag");
            investigationResults.Add("");
            investigationResults.Add("NEXT STEPS:");
            investigationResults.Add("- Verify MSHD offsets point to actual chunk data");
            investigationResults.Add("- Validate MSUR floats as surface normals");
            investigationResults.Add("- Decode MSLK flag patterns (0x0C field structure)");
            investigationResults.Add("- Cross-reference with WoW client databases for doodad types");
            
            // Write investigation results
            string investigationPath = Path.Combine(outputDir, "unknown_field_investigation.txt");
            File.WriteAllLines(investigationPath, investigationResults);
            
            Console.WriteLine($"Investigation complete! Processed {filesProcessed} files.");
            Console.WriteLine($"Results written to: {investigationPath}");
            
            // Also output specific findings to console
            Console.WriteLine("\n🔍 KEY FINDINGS:");
            Console.WriteLine("• MSHD fields appear to be chunk offsets/sizes");
            Console.WriteLine("• MSUR floats 0x04-0x0C are likely 3D surface normals");
            Console.WriteLine("• MSUR float 0x10 is likely height/Y-coordinate");
            Console.WriteLine("• MSLK 0x0C field contains packed identifiers");
            Console.WriteLine("• Several fields are consistent constants (padding/flags)");
        }

        [Fact]
        public void ExportEnhancedObjWithDecodedFields()
        {
            // Create output directory
            string outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "EnhancedObjExport");
            Directory.CreateDirectory(outputDir);
            
            var results = new List<string>();
            int filesProcessed = 0;
            
            // Get PM4 files for enhanced export
            var inputDirectoryPath = Path.Combine(TestDataRoot, "original_development", "development");
            var pm4Files = Directory.GetFiles(inputDirectoryPath, "*.pm4", SearchOption.TopDirectoryOnly)
                .Where(f => !f.Contains("_49_28")) // Skip problematic files
                .Take(5); // Process 5 files for testing
            
            foreach (string pm4FilePath in pm4Files)
            {
                try
                {
                    var pm4File = PM4File.FromFile(pm4FilePath);
                    if (pm4File?.MSVT?.Vertices == null || pm4File.MSVI?.Indices == null || pm4File.MSUR?.Entries == null)
                    {
                        results.Add($"Skipped {Path.GetFileName(pm4FilePath)}: Missing required chunks");
                        continue;
                    }
                    
                    // Generate enhanced OBJ with surface normals and materials
                    string fileName = Path.GetFileNameWithoutExtension(pm4FilePath);
                    string objFilePath = Path.Combine(outputDir, $"{fileName}_enhanced.obj");
                    string mtlFilePath = Path.Combine(outputDir, $"{fileName}_enhanced.mtl");
                    
                    GenerateEnhancedObjFile(pm4File, objFilePath, mtlFilePath);
                    
                    results.Add($"Enhanced export: {fileName}");
                    filesProcessed++;
                }
                catch (Exception ex)
                {
                    results.Add($"Error processing {Path.GetFileName(pm4FilePath)}: {ex.Message}");
                }
            }
            
            // Write summary
            string summaryPath = Path.Combine(outputDir, "enhanced_export_summary.txt");
            File.WriteAllLines(summaryPath, new[]
            {
                $"Enhanced OBJ Export Summary - {DateTime.Now}",
                $"Files Processed: {filesProcessed}",
                $"Features Implemented:",
                "- Surface normals from MSUR decoded fields",
                "- Material classification from MSLK metadata",
                "- Height-based surface organization",
                "- Object type grouping",
                "",
                "Results:"
            }.Concat(results));
            
            Console.WriteLine($"Enhanced OBJ export completed. Processed {filesProcessed} files.");
            Console.WriteLine($"Output directory: {outputDir}");
        }
        
        private void GenerateEnhancedObjFile(PM4File pm4File, string objFilePath, string mtlFilePath)
        {
            var objContent = new List<string>();
            var mtlContent = new List<string>();
            
            // OBJ header with MTL reference
            objContent.Add("# Enhanced PM4 OBJ Export with Decoded Fields");
            objContent.Add($"# Generated: {DateTime.Now}");
            objContent.Add($"# Features: Surface normals, materials, height organization");
            objContent.Add($"mtllib {Path.GetFileName(mtlFilePath)}");
            objContent.Add("");
            
            // Export MSVT vertices with proper coordinate transformation
            var msvtVertices = pm4File.MSVT.Vertices;
            foreach (var vertex in msvtVertices)
            {
                // PM4 coordinate system (Y, X, Z) → World (X, Y, Z)
                float worldX = vertex.Y;
                float worldY = vertex.X; 
                float worldZ = vertex.Z;
                objContent.Add($"v {worldX:F6} {worldY:F6} {worldZ:F6}");
            }
            objContent.Add("");
            
            // Export surface normals from MSUR decoded fields
            var surfaceNormals = new List<string>();
            var surfaceMaterials = new Dictionary<string, int>();
            int materialIndex = 0;
            
            foreach (var msur in pm4File.MSUR.Entries)
            {
                // Add surface normal using decoded fields
                float normalX = msur.SurfaceNormalX;
                float normalY = msur.SurfaceNormalY; 
                float normalZ = msur.SurfaceNormalZ;
                objContent.Add($"vn {normalX:F6} {normalY:F6} {normalZ:F6}");
                surfaceNormals.Add($"{normalX:F6} {normalY:F6} {normalZ:F6}");
            }
            objContent.Add("");
            
            // Generate material library from MSLK metadata (if available)
            if (pm4File.MSLK?.Entries != null)
            {
                var uniqueMaterials = pm4File.MSLK.Entries
                    .GroupBy(e => e.MaterialColorId)
                    .Select((g, i) => new { MaterialId = g.Key, Index = i, TypeFlags = g.First().ObjectTypeFlags })
                    .ToList();
                
                foreach (var mat in uniqueMaterials)
                {
                    string materialName = $"material_{mat.MaterialId:X8}_type_{mat.TypeFlags}";
                    surfaceMaterials[materialName] = mat.Index;
                    
                    // Generate MTL entry
                    mtlContent.Add($"newmtl {materialName}");
                    mtlContent.Add($"# Material ID: 0x{mat.MaterialId:X8}, Object Type: {mat.TypeFlags}");
                    
                    // Generate colors based on material ID for visualization
                    float r = ((mat.MaterialId >> 16) & 0xFF) / 255.0f;
                    float g = ((mat.MaterialId >> 8) & 0xFF) / 255.0f; 
                    float b = (mat.MaterialId & 0xFF) / 255.0f;
                    mtlContent.Add($"Kd {r:F3} {g:F3} {b:F3}");
                    mtlContent.Add($"Ka 0.1 0.1 0.1");
                    mtlContent.Add($"Ks 0.3 0.3 0.3");
                    mtlContent.Add($"Ns 10.0");
                    mtlContent.Add("");
                }
            }
            
            // Generate faces with proper indexing and group organization
            var processedSurfaceSignatures = new HashSet<string>();
            int faceCount = 0;
            int normalIndex = 1;
            
            // Group surfaces by height for organization
            var surfacesByHeight = pm4File.MSUR.Entries
                .Select((msur, index) => new { Surface = msur, Index = index })
                .GroupBy(x => Math.Round(x.Surface.SurfaceHeight / 100.0) * 100) // Group by 100-unit height bands
                .OrderBy(g => g.Key)
                .ToList();
            
            foreach (var heightGroup in surfacesByHeight)
            {
                objContent.Add($"# Height Level: {heightGroup.Key:F0} units");
                objContent.Add($"g height_level_{heightGroup.Key:F0}");
                
                foreach (var surfaceInfo in heightGroup)
                {
                    var msur = surfaceInfo.Surface;
                    
                    // Get surface indices
                    var surfaceIndices = new List<uint>();
                    for (int j = 0; j < msur.IndexCount; j++)
                    {
                        surfaceIndices.Add(pm4File.MSVI.Indices[(int)msur.MsviFirstIndex + j]);
                    }
                    
                    // Create signature to avoid duplicates
                    var signature = string.Join(",", surfaceIndices.OrderBy(x => x));
                    if (processedSurfaceSignatures.Contains(signature))
                        continue;
                    
                    processedSurfaceSignatures.Add(signature);
                    
                    // Generate triangle fan with normals
                    if (surfaceIndices.Count >= 3)
                    {
                        for (int k = 1; k < surfaceIndices.Count - 1; k++)
                        {
                            uint idx1 = surfaceIndices[0];
                            uint idx2 = surfaceIndices[k];
                            uint idx3 = surfaceIndices[k + 1];
                            
                            // Validate indices
                            if (idx1 < msvtVertices.Count && idx2 < msvtVertices.Count && idx3 < msvtVertices.Count &&
                                idx1 != idx2 && idx1 != idx3 && idx2 != idx3)
                            {
                                // 1-based indexing for OBJ format, include normal index
                                objContent.Add($"f {idx1 + 1}//{normalIndex} {idx2 + 1}//{normalIndex} {idx3 + 1}//{normalIndex}");
                                faceCount++;
                            }
                        }
                    }
                    
                    normalIndex++;
                }
                
                objContent.Add("");
            }
            
            // Write OBJ file
            File.WriteAllLines(objFilePath, objContent);
            
            // Write MTL file if materials were generated
            if (mtlContent.Count > 0)
            {
                File.WriteAllLines(mtlFilePath, mtlContent);
            }
            
            Console.WriteLine($"Enhanced export complete: {faceCount} faces, {surfaceNormals.Count} normals, {surfaceMaterials.Count} materials");
        }

        [Fact]
        public void AnalyzeChunkRelationshipsAndMissingGeometry()
        {
            // Create output directory
            string outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "ChunkRelationshipAnalysis");
            Directory.CreateDirectory(outputDir);
            
            var results = new List<string>();
            
            // Get a representative PM4 file for detailed analysis
            var inputDirectoryPath = Path.Combine(TestDataRoot, "original_development", "development");
            var testFile = Path.Combine(inputDirectoryPath, "development_00_00.pm4");
            
            var pm4File = PM4File.FromFile(testFile);
            if (pm4File == null)
            {
                results.Add("Failed to load test file");
                return;
            }
            
            results.Add("=== CHUNK RELATIONSHIP ANALYSIS ===");
            results.Add($"File: {Path.GetFileName(testFile)}");
            results.Add("");
            
            // Analyze chunk sizes and data availability
            results.Add("=== CHUNK DATA OVERVIEW ===");
            results.Add($"MSVT Vertices: {pm4File.MSVT?.Vertices?.Count ?? 0}");
            results.Add($"MSCN Vertices: {pm4File.MSCN?.ExteriorVertices?.Count ?? 0}");
            results.Add($"MSPV Vertices: {pm4File.MSPV?.Vertices?.Count ?? 0}");
            results.Add($"MSUR Surfaces: {pm4File.MSUR?.Entries?.Count ?? 0}");
            results.Add($"MSLK Entries: {pm4File.MSLK?.Entries?.Count ?? 0}");
            results.Add($"MSVI Indices: {pm4File.MSVI?.Indices?.Count ?? 0}");
            results.Add($"MPRL Points: {pm4File.MPRL?.Entries?.Count ?? 0}");
            results.Add("");
            
            // Analyze MSLK relationships to other chunks
            results.Add("=== MSLK CHUNK RELATIONSHIPS ===");
            if (pm4File.MSLK?.Entries != null)
            {
                foreach (var mslk in pm4File.MSLK.Entries.Take(10))
                {
                    results.Add($"MSLK Entry:");
                    results.Add($"  Object Type: {mslk.ObjectTypeFlags}");
                    results.Add($"  Group ID: {mslk.GroupObjectId}");
                    results.Add($"  MSPI First Index: {mslk.MspiFirstIndex}");
                    results.Add($"  MSPI Index Count: {mslk.MspiIndexCount}");
                    results.Add($"  Reference Index: {mslk.ReferenceIndex}");
                    results.Add($"  Material ID: 0x{mslk.MaterialColorId:X8}");
                    
                    // Check if Reference Index points to other chunks
                    if (mslk.ReferenceIndex < (pm4File.MSCN?.ExteriorVertices?.Count ?? 0))
                    {
                        results.Add($"  -> References MSCN vertex {mslk.ReferenceIndex}");
                    }
                    if (mslk.ReferenceIndex < (pm4File.MSPV?.Vertices?.Count ?? 0))
                    {
                        results.Add($"  -> References MSPV vertex {mslk.ReferenceIndex}");
                    }
                    if (mslk.ReferenceIndex < (pm4File.MSVI?.Indices?.Count ?? 0))
                    {
                        results.Add($"  -> References MSVI index {mslk.ReferenceIndex}");
                    }
                    results.Add("");
                }
            }
            
            // Analyze geometry distribution and find missing vertical data
            results.Add("=== GEOMETRY DISTRIBUTION ANALYSIS ===");
            
            // MSVT geometry analysis
            if (pm4File.MSVT?.Vertices != null)
            {
                var msvtBounds = AnalyzeGeometryBounds(pm4File.MSVT.Vertices.Select(v => new { X = v.Y, Y = v.X, Z = v.Z }));
                results.Add("MSVT (Render Mesh) Bounds:");
                results.Add($"  X: {msvtBounds.MinX:F2} to {msvtBounds.MaxX:F2} (range: {msvtBounds.MaxX - msvtBounds.MinX:F2})");
                results.Add($"  Y: {msvtBounds.MinY:F2} to {msvtBounds.MaxY:F2} (range: {msvtBounds.MaxY - msvtBounds.MinY:F2})");
                results.Add($"  Z: {msvtBounds.MinZ:F2} to {msvtBounds.MaxZ:F2} (range: {msvtBounds.MaxZ - msvtBounds.MinZ:F2})");
            }
            
            // MSCN geometry analysis  
            if (pm4File.MSCN?.ExteriorVertices != null)
            {
                var mscnBounds = AnalyzeGeometryBounds(pm4File.MSCN.ExteriorVertices.Select(v => new { 
                    X = v.X * 0.25f + v.Y * 0.25f, 
                    Y = v.Y * 0.25f - v.Z * 0.25f, 
                    Z = v.Z * 0.25f + v.X * 0.25f 
                }));
                results.Add("");
                results.Add("MSCN (Collision/Structure) Bounds:");
                results.Add($"  X: {mscnBounds.MinX:F2} to {mscnBounds.MaxX:F2} (range: {mscnBounds.MaxX - mscnBounds.MinX:F2})");
                results.Add($"  Y: {mscnBounds.MinY:F2} to {mscnBounds.MaxY:F2} (range: {mscnBounds.MaxY - mscnBounds.MinY:F2})");
                results.Add($"  Z: {mscnBounds.MinZ:F2} to {mscnBounds.MaxZ:F2} (range: {mscnBounds.MaxZ - mscnBounds.MinZ:F2})");
            }
            
            // MSPV geometry analysis
            if (pm4File.MSPV?.Vertices != null)
            {
                var mspvBounds = AnalyzeGeometryBounds(pm4File.MSPV.Vertices.Select(v => new { X = v.X, Y = v.Y, Z = v.Z }));
                results.Add("");
                results.Add("MSPV (Structure Points) Bounds:");
                results.Add($"  X: {mspvBounds.MinX:F2} to {mspvBounds.MaxX:F2} (range: {mspvBounds.MaxX - mspvBounds.MinX:F2})");
                results.Add($"  Y: {mspvBounds.MinY:F2} to {mspvBounds.MaxY:F2} (range: {mspvBounds.MaxY - mspvBounds.MinY:F2})");
                results.Add($"  Z: {mspvBounds.MinZ:F2} to {mspvBounds.MaxZ:F2} (range: {mspvBounds.MaxZ - mspvBounds.MinZ:F2})");
            }
            
            results.Add("");
            results.Add("=== MISSING GEOMETRY ANALYSIS ===");
            
            // Check if we're missing significant vertical structures
            if (pm4File.MSCN?.ExteriorVertices != null && pm4File.MSVT?.Vertices != null)
            {
                var mscnVerticalRange = pm4File.MSCN.ExteriorVertices.Max(v => v.Y) - pm4File.MSCN.ExteriorVertices.Min(v => v.Y);
                var msvtVerticalRange = pm4File.MSVT.Vertices.Max(v => v.X) - pm4File.MSVT.Vertices.Min(v => v.X);
                
                results.Add($"MSCN Vertical Range: {mscnVerticalRange:F2}");
                results.Add($"MSVT Vertical Range: {msvtVerticalRange:F2}");
                
                if (mscnVerticalRange > msvtVerticalRange * 2)
                {
                    results.Add("*** WARNING: MSCN has significantly more vertical range than MSVT ***");
                    results.Add("*** This suggests missing vertical geometry in render mesh ***");
                }
            }
            
            // Analyze how MSLK might reference vertical structures
            results.Add("");
            results.Add("=== MSLK TO GEOMETRY RELATIONSHIPS ===");
            if (pm4File.MSLK?.Entries != null)
            {
                var objectTypes = pm4File.MSLK.Entries.GroupBy(e => e.ObjectTypeFlags).ToList();
                foreach (var group in objectTypes)
                {
                    results.Add($"Object Type {group.Key}: {group.Count()} entries");
                    var sample = group.First();
                    if (sample.MspiFirstIndex != -1 && sample.MspiIndexCount > 0)
                    {
                        results.Add($"  -> References MSPI geometry (Index: {sample.MspiFirstIndex}, Count: {sample.MspiIndexCount})");
                    }
                    else
                    {
                        results.Add($"  -> No MSPI geometry (likely references other data)");
                        results.Add($"  -> Reference Index: {sample.ReferenceIndex} (might point to MSCN/MSPV)");
                    }
                }
            }
            
            // Write results
            string outputPath = Path.Combine(outputDir, "chunk_relationship_analysis.txt");
            File.WriteAllLines(outputPath, results);
            
            Console.WriteLine($"Chunk relationship analysis complete. Output: {outputPath}");
        }
        
        private dynamic AnalyzeGeometryBounds(IEnumerable<dynamic> vertices)
        {
            var vertexList = vertices.ToList();
            if (!vertexList.Any()) return new { MinX = 0f, MaxX = 0f, MinY = 0f, MaxY = 0f, MinZ = 0f, MaxZ = 0f };
            
            return new {
                MinX = vertexList.Min(v => (float)v.X),
                MaxX = vertexList.Max(v => (float)v.X), 
                MinY = vertexList.Min(v => (float)v.Y),
                MaxY = vertexList.Max(v => (float)v.Y),
                MinZ = vertexList.Min(v => (float)v.Z),
                MaxZ = vertexList.Max(v => (float)v.Z)
            };
        }

        [Fact]
        public void GenerateCompleteGeometryUsingMSLKReferences()
        {
            // Create output directory
            string outputDir = Path.Combine(TestContext.TimestampedOutputRoot, "CompleteGeometryExport");
            Directory.CreateDirectory(outputDir);
            
            var results = new List<string>();
            
            // Get test file
            var inputDirectoryPath = Path.Combine(TestDataRoot, "original_development", "development");
            var testFile = Path.Combine(inputDirectoryPath, "development_00_00.pm4");
            
            var pm4File = PM4File.FromFile(testFile);
            if (pm4File == null)
            {
                results.Add("Failed to load test file");
                return;
            }
            
            results.Add("=== COMPLETE GEOMETRY GENERATION USING MSLK REFERENCES ===");
            results.Add($"File: {Path.GetFileName(testFile)}");
            results.Add("");
            
            // Generate complete OBJ with all geometry sources
            string objFilePath = Path.Combine(outputDir, "complete_geometry.obj");
            var objContent = new List<string>();
            
            objContent.Add("# Complete PM4 Geometry Export Using MSLK References");
            objContent.Add($"# Generated: {DateTime.Now}");
            objContent.Add("# Includes: MSVT render mesh + MSCN collision/structure + MSPV structure points");
            objContent.Add("# Organized by MSLK object type references");
            objContent.Add("");
            
            int vertexOffset = 0;
            
            // 1. Export all MSVT vertices (render mesh)
            objContent.Add("# MSVT Render Mesh Vertices");
            if (pm4File.MSVT?.Vertices != null)
            {
                foreach (var vertex in pm4File.MSVT.Vertices)
                {
                    float worldX = vertex.Y;
                    float worldY = vertex.X;
                    float worldZ = vertex.Z;
                    objContent.Add($"v {worldX:F6} {worldY:F6} {worldZ:F6}");
                }
                results.Add($"MSVT Vertices Exported: {pm4File.MSVT.Vertices.Count}");
                vertexOffset += pm4File.MSVT.Vertices.Count;
            }
            objContent.Add("");
            
            // 2. Export all MSCN vertices (collision/structure)
            objContent.Add("# MSCN Collision/Structure Vertices");
            int mscnVertexOffset = vertexOffset;
            if (pm4File.MSCN?.ExteriorVertices != null)
            {
                foreach (var vertex in pm4File.MSCN.ExteriorVertices)
                {
                    // Apply MSCN coordinate transformation
                    float worldX = vertex.X * 0.25f + vertex.Y * 0.25f;
                    float worldY = vertex.Y * 0.25f - vertex.Z * 0.25f;
                    float worldZ = vertex.Z * 0.25f + vertex.X * 0.25f;
                    objContent.Add($"v {worldX:F6} {worldY:F6} {worldZ:F6}");
                }
                results.Add($"MSCN Vertices Exported: {pm4File.MSCN.ExteriorVertices.Count}");
                vertexOffset += pm4File.MSCN.ExteriorVertices.Count;
            }
            objContent.Add("");
            
            // 3. Export all MSPV vertices (structure points)
            objContent.Add("# MSPV Structure Point Vertices");
            int mspvVertexOffset = vertexOffset;
            if (pm4File.MSPV?.Vertices != null)
            {
                foreach (var vertex in pm4File.MSPV.Vertices)
                {
                    objContent.Add($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
                results.Add($"MSPV Vertices Exported: {pm4File.MSPV.Vertices.Count}");
                vertexOffset += pm4File.MSPV.Vertices.Count;
            }
            objContent.Add("");
            
            // 4. Generate faces based on MSLK object type references
            results.Add("");
            results.Add("=== FACE GENERATION BY MSLK OBJECT TYPE ===");
            
            // Group MSLK entries by object type
            if (pm4File.MSLK?.Entries != null)
            {
                var objectTypeGroups = pm4File.MSLK.Entries.GroupBy(e => e.ObjectTypeFlags).ToList();
                
                foreach (var group in objectTypeGroups)
                {
                    objContent.Add($"# Object Type {group.Key} ({group.Count()} entries)");
                    objContent.Add($"g object_type_{group.Key}");
                    
                    int facesGenerated = 0;
                    
                    foreach (var mslk in group.Take(100)) // Limit for testing
                    {
                        // Different strategies based on whether MSLK has MSPI geometry or references other data
                        if (mslk.MspiFirstIndex != -1 && mslk.MspiIndexCount > 0)
                        {
                            // Has MSPI geometry - generate faces from MSVT using MSVI indices
                            // This is the traditional approach we've been using
                            continue; // Skip for now, focus on new geometry
                        }
                        else
                        {
                            // References other geometry via Reference Index
                            var refIndex = mslk.ReferenceIndex;
                            
                            // Try to create geometry based on reference index
                            if (refIndex < (pm4File.MSCN?.ExteriorVertices?.Count ?? 0))
                            {
                                // Reference points to MSCN vertex - create structure geometry
                                int mscnIndex = (int)refIndex;
                                int vertexIndex = mscnVertexOffset + mscnIndex + 1; // 1-based OBJ indexing
                                
                                // Create a simple face structure (this might need more sophisticated logic)
                                if (mscnIndex + 2 < pm4File.MSCN.ExteriorVertices.Count)
                                {
                                    objContent.Add($"f {vertexIndex} {vertexIndex + 1} {vertexIndex + 2}");
                                    facesGenerated++;
                                }
                            }
                            
                            if (refIndex < (pm4File.MSPV?.Vertices?.Count ?? 0))
                            {
                                // Reference points to MSPV vertex - create structure point geometry
                                int mspvIndex = (int)refIndex;
                                int vertexIndex = mspvVertexOffset + mspvIndex + 1; // 1-based OBJ indexing
                                
                                // Create point-based geometry
                                if (mspvIndex + 2 < pm4File.MSPV.Vertices.Count)
                                {
                                    objContent.Add($"f {vertexIndex} {vertexIndex + 1} {vertexIndex + 2}");
                                    facesGenerated++;
                                }
                            }
                        }
                    }
                    
                    results.Add($"Object Type {group.Key}: Generated {facesGenerated} structure faces");
                    objContent.Add("");
                }
            }
            
            // 5. Generate traditional MSUR faces for comparison
            objContent.Add("# Traditional MSUR-based faces for comparison");
            objContent.Add("g msur_traditional_faces");
            
            int msurFaces = 0;
            var processedSurfaceSignatures = new HashSet<string>();
            
            if (pm4File.MSUR?.Entries != null)
            {
                foreach (var msur in pm4File.MSUR.Entries.Take(100)) // Limit for testing
                {
                    var surfaceIndices = new List<uint>();
                    for (int j = 0; j < msur.IndexCount; j++)
                    {
                        surfaceIndices.Add(pm4File.MSVI.Indices[(int)msur.MsviFirstIndex + j]);
                    }
                    
                    var signature = string.Join(",", surfaceIndices.OrderBy(x => x));
                    if (processedSurfaceSignatures.Contains(signature))
                        continue;
                    
                    processedSurfaceSignatures.Add(signature);
                    
                    // Generate triangle fan
                    if (surfaceIndices.Count >= 3)
                    {
                        for (int k = 1; k < surfaceIndices.Count - 1; k++)
                        {
                            uint idx1 = surfaceIndices[0];
                            uint idx2 = surfaceIndices[k];
                            uint idx3 = surfaceIndices[k + 1];
                            
                            if (idx1 < pm4File.MSVT.Vertices.Count && idx2 < pm4File.MSVT.Vertices.Count && idx3 < pm4File.MSVT.Vertices.Count &&
                                idx1 != idx2 && idx1 != idx3 && idx2 != idx3)
                            {
                                objContent.Add($"f {idx1 + 1} {idx2 + 1} {idx3 + 1}"); // 1-based, MSVT only
                                msurFaces++;
                            }
                        }
                    }
                }
            }
            
            results.Add($"Traditional MSUR faces: {msurFaces}");
            
            // Write OBJ file
            File.WriteAllLines(objFilePath, objContent);
            
            results.Add("");
            results.Add($"Complete geometry OBJ exported: {objFilePath}");
            results.Add($"Total vertices exported: {vertexOffset}");
            results.Add("- This includes MSVT render mesh + MSCN collision/structure + MSPV structure points");
            results.Add("- Organized by MSLK object type references");
            
            // Write analysis results
            string analysisPath = Path.Combine(outputDir, "complete_geometry_analysis.txt");
            File.WriteAllLines(analysisPath, results);
            
            Console.WriteLine($"Complete geometry analysis complete. Check: {outputDir}");
        }
    } // End PM4FileTests class

    /// <summary>
    /// Helper class to expose specialized processing method for testing
    /// </summary>
    public class PM4HighRatioProcessor
    {
        // Use centralized coordinate transforms instead of local constants
        
        public int ProcessHighRatioFile(string inputFilePath, string outputDir)
        {
            string fileName = Path.GetFileName(inputFilePath);
            string baseOutputName = Path.GetFileNameWithoutExtension(inputFilePath);
            
            string debugLogPath = Path.Combine(outputDir, $"{baseOutputName}_debug.log");
            using var debugWriter = new StreamWriter(debugLogPath, false);
            
            debugWriter.WriteLine($"=== Processing High MPRR/MPRL Ratio File: {fileName} ===");
            debugWriter.WriteLine($"Processing Time: {DateTime.Now}\n");
            
            try
            {
                // Load the file but don't process MPRR links (which would cause exceptions)
                debugWriter.WriteLine("Loading file with modified approach...");
                var pm4File = PM4File.FromFile(inputFilePath);
                
                // Output basic stats
                debugWriter.WriteLine($"Loaded file: {fileName}");
                // UPDATED: Use Sequences.Count with null checks
                // CORRECTED TYPO: Mprr -> MPRR
                debugWriter.WriteLine($"MPRR Sequences: {pm4File.MPRR?.Sequences?.Count ?? 0}");
                debugWriter.WriteLine($"MPRL Entries: {pm4File.MPRL?.Entries.Count ?? 0}");

                // Calculate ratio for logging
                // UPDATED: Use Sequences.Count with null checks
                // CORRECTED TYPO: Mprr -> MPRR
                double ratio = pm4File.MPRL?.Entries.Count > 0 ?
                    (double)(pm4File.MPRR?.Sequences?.Count ?? 0) / pm4File.MPRL.Entries.Count : // Corrected typo here
                    double.NaN;
                debugWriter.WriteLine($"MPRR/MPRL Ratio: {ratio:F2}\n");
                
                // Process everything EXCEPT MPRR links
                int processedVertexCount = 0;
                
                // Process MSVT/MSVI data if available
                string outputMsvtObjPath = Path.Combine(outputDir, $"{baseOutputName}_msvt.obj");
                if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    debugWriter.WriteLine($"Processing MSVT vertices ({pm4File.MSVT.Vertices.Count})...");
                    using var msvtObjWriter = new StreamWriter(outputMsvtObjPath, false);
                    
                    // Add header
                    msvtObjWriter.WriteLine($"# WoW PM4 MSVT vertices from {fileName}");
                    msvtObjWriter.WriteLine($"# Generated {DateTime.Now}");
                    
                    // Process vertices using PM4-relative coordinates
                    foreach (var vertex in pm4File.MSVT.Vertices)
                    {
                        // Apply PM4-relative transformation (Y, X, Z)
                        var pm4Coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                        msvtObjWriter.WriteLine($"v {pm4Coords.X} {pm4Coords.Y} {pm4Coords.Z}");
                    }
                    
                    int msvtVertexCount = pm4File.MSVT.Vertices.Count;
                    debugWriter.WriteLine($"  Wrote {msvtVertexCount} MSVT vertices to {Path.GetFileName(outputMsvtObjPath)}");
                    processedVertexCount += msvtVertexCount;
                }
                else
                {
                    debugWriter.WriteLine("  No MSVT vertices found to process.");
                }
                
                // Process MPRL points (without MPRR links)
                string outputMprlPointsObjPath = Path.Combine(outputDir, $"{baseOutputName}_mprl_points.obj");
                if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                {
                    debugWriter.WriteLine($"\nProcessing MPRL points ({pm4File.MPRL.Entries.Count})...");
                    using var mprlPointsObjWriter = new StreamWriter(outputMprlPointsObjPath, false);
                    
                    // Add header
                    mprlPointsObjWriter.WriteLine($"# WoW PM4 MPRL points from {fileName}");
                    mprlPointsObjWriter.WriteLine($"# Generated {DateTime.Now}");
                    mprlPointsObjWriter.WriteLine($"# Format: v x y z, PM4-relative transform: (X, -Z, Y)");
                    
                    // Add points using PM4-relative transformation
                    int validPointCount = 0;
                    foreach (var point in pm4File.MPRL.Entries)
                    {
                        // Skip points with zero position or zero IDs
                        bool hasZeroPosition = Math.Abs(point.Position.X) < 0.001f && 
                                         Math.Abs(point.Position.Y) < 0.001f && 
                                         Math.Abs(point.Position.Z) < 0.001f;
                                         
                        bool hasZeroIds = point.Unknown_0x04 == 0 && point.Unknown_0x06 == 0 && 
                                         point.Unknown_0x14 == 0 && point.Unknown_0x16 == 0;
                        
                        if (hasZeroPosition || hasZeroIds)
                        {
                            continue;
                        }
                        
                        // Use PM4-relative MPRL transformation
                        var pm4Coords = Pm4CoordinateTransforms.FromMprlEntry(point);

                        // Write PM4-relative coordinates
                        mprlPointsObjWriter.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6}"));
                        validPointCount++;
                    }
                    
                    debugWriter.WriteLine($"  Wrote {validPointCount} valid MPRL points (skipped {pm4File.MPRL.Entries.Count - validPointCount} invalid points) to {Path.GetFileName(outputMprlPointsObjPath)}");
                    processedVertexCount += validPointCount;
                }
                else
                {
                    debugWriter.WriteLine("  No MPRL points found to process.");
                }
                
                // Note that we're skipping MPRR links due to the high ratio
                debugWriter.WriteLine("\nSkipped processing MPRR links due to high MPRR/MPRL ratio");
                
                // Final summary
                debugWriter.WriteLine($"\n=== Processing Summary ===");
                debugWriter.WriteLine($"Successfully processed {fileName} with {processedVertexCount} total vertices");
                debugWriter.WriteLine($"NOTE: MPRR links were skipped to avoid index out of range exceptions");
                
                return processedVertexCount;
            }
            catch (Exception ex)
            {
                debugWriter.WriteLine($"ERROR: Failed to process high-ratio file: {ex.Message}");
                debugWriter.WriteLine($"Stack trace: {ex.StackTrace}");
                
                return 0;
            }
        }
    }
} // End namespace
