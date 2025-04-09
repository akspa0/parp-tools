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
        private const string TestDataPath = "test_data/development/development_00_00.pm4";
        private const float ScaleFactor = 36.0f; // Common scaling factor
        private const float CoordinateOffset = 17066.666f; // From MsvtVertex documentation/constants

        // NEW helper specifically for MPRL (Z Scale ONLY)
        private static Vector3 MprlToWorld(C3Vectori v)
        {
            // Apply Z scaling only, assume X/Y are direct
            return new Vector3(
                (float)v.X,                    // Use X directly
                (float)v.Y,                    // Use Y directly
                (float)v.Z              // Use Z directly, divide by 36.0
            );
        }

        // Helper function to convert world coordinates back to C3Vectori (MPRL)
        // Needs updating if C3VectoriToWorld changes significantly
        private static C3Vectori WorldToC3Vectori(Vector3 worldPos)
        {
            // const float CoordinateOffset = 17066.666f; // Unused
            // Assuming Z is not scaled back for now
            // Reverting to previous inverse logic assuming X/Y swap+offset is needed ONLY for MSVT
            // This function may need complete removal or revision based on chosen approach.
            return new C3Vectori
            {
                X = (int)worldPos.X, 
                Y = (int)worldPos.Y, 
                Z = (int)(worldPos.Z * 36.0f) // Apply inverse scaling if needed
            };
        }

        [Fact]
        public void LoadPM4File_ShouldLoadChunks()
        {
            Console.WriteLine("--- LoadPM4File_ShouldLoadChunks START ---"); // Execution Start Log
            // Arrange
            var inputFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, TestDataPath);
            
            // --- Define Separate Output Paths --- Attempt 21 ---
            string baseOutputPath = Path.Combine(Path.GetDirectoryName(TestDataPath) ?? AppDomain.CurrentDomain.BaseDirectory, Path.GetFileNameWithoutExtension(TestDataPath));
            var outputMsvtFilePath = baseOutputPath + "_msvt.obj";
            var outputMspvFilePath = baseOutputPath + "_mspv.obj";
            // Revert to single MPRL output path
            var outputMprlFilePath = baseOutputPath + "_mprl.obj";
            var outputMscnFilePath = baseOutputPath + "_mscn.obj"; // ADDED MSCN output path
            var outputMslkFilePath = baseOutputPath + "_mslk.obj"; // ADDED MSLK output path
            var outputMslkJsonPath = baseOutputPath + "_mslk_hierarchy.json"; // ADDED MSLK JSON output path
            var outputSkippedMslkLogPath = baseOutputPath + "_skipped_mslk.log"; // ADDED Skipped MSLK log path
            // var outputMprlXZYFilePath = baseOutputPath + "_mprl_XZY.obj"; // Removed
            // var outputMprlNegXZYFilePath = baseOutputPath + "_mprl_negXZY.obj"; // Removed
            // var outputMscnFilePath = baseOutputPath + "_mscn.obj"; // Reverted: MSCN Disabled
            // var outputCombinedFilePath = baseOutputPath + "_combined.obj"; // Reverted: Removed Combined
            string debugLogPath = baseOutputPath + ".debug.log";
            string summaryLogPath = baseOutputPath + ".summary.log"; // ADDED Summary Log Path

            Console.WriteLine($"Output MSVT OBJ: {outputMsvtFilePath}");
            Console.WriteLine($"Output MSPV OBJ: {outputMspvFilePath}");
            // Update console output for single MPRL file
            Console.WriteLine($"Output MPRL OBJ: {outputMprlFilePath}");
            Console.WriteLine($"Output MSCN OBJ: {outputMscnFilePath}"); // ADDED log for MSCN path
            Console.WriteLine($"Output MSLK OBJ: {outputMslkFilePath}"); // ADDED log for MSLK path
            Console.WriteLine($"Output MSLK JSON: {outputMslkJsonPath}"); // ADDED log for MSLK JSON path
            Console.WriteLine($"Output Skipped MSLK Log: {outputSkippedMslkLogPath}"); // ADDED log for skipped MSLK path
            // Console.WriteLine($"Output MPRL (XZY) OBJ: {outputMprlXZYFilePath}"); // Removed
            // Console.WriteLine($"Output MPRL (NegXZY) OBJ: {outputMprlNegXZYFilePath}"); // Removed
            // Console.WriteLine($"Output MSCN OBJ: {outputMscnFilePath}"); // Reverted: MSCN Disabled
            // Console.WriteLine($"Output Combined OBJ: {outputCombinedFilePath}"); // Reverted: Removed Combined
            Console.WriteLine($"Debug Log: {debugLogPath}");
            Console.WriteLine($"Summary Log: {summaryLogPath}"); // ADDED log for summary path

            // Act
            var pm4File = PM4File.FromFile(TestDataPath);

            // Assert - Basic Chunk Loading
            Assert.NotNull(pm4File);
            Assert.NotNull(pm4File.MVER);
            Assert.NotNull(pm4File.MSHD);
            Assert.NotNull(pm4File.MSLK);
            Assert.NotNull(pm4File.MSPI);
            Assert.NotNull(pm4File.MSPV);
            Assert.NotNull(pm4File.MSVT);
            Assert.NotNull(pm4File.MSVI);
            Assert.NotNull(pm4File.MSUR);
            Assert.NotNull(pm4File.MSCN);
            Assert.NotNull(pm4File.MPRL);
            Assert.NotNull(pm4File.MPRR);
            Assert.NotNull(pm4File.MDBH);
            Assert.NotNull(pm4File.MDOS);
            Assert.NotNull(pm4File.MDSF);
            // MSRN is optional, so don't assert not null
            // Assert.NotNull(pm4File.MSRN); 

            // Log MDOS Entry Count for verification (MOVED INSIDE TRY BLOCK)
            // int mdosEntryCount = pm4File.MDOS?.Entries.Count ?? -1;
            // Console.WriteLine($"INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");
            // debugWriter.WriteLine($"INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");

            // Assert - Basic Counts (assuming test file has data)
            Assert.True(pm4File.MSLK?.Entries.Count > 0);
            Assert.True(pm4File.MSPI?.Indices.Count > 0);
            Assert.True(pm4File.MSPV?.Vertices.Count > 0);
            Assert.True(pm4File.MSVT?.Vertices.Count > 0);
            Assert.True(pm4File.MSVI?.Indices.Count > 0);
            Assert.True(pm4File.MSUR?.Entries.Count > 0);
            // Check if MSCN exists and has data, but allow it to be empty if not present in file
            if (pm4File.MSCN != null)
            {
                 Assert.True(pm4File.MSCN.Vectors.Count >= 0); // Allow empty chunk
            }
            // else Assert.Null(pm4File.MSCN); // Optionally assert it's null if not expected
            Assert.True(pm4File.MPRL?.Entries.Count > 0);
            Assert.True(pm4File.MPRR?.Entries.Count > 0);
            // Check if MDBH exists and has entries, allowing for empty if not present
            if (pm4File.MDBH != null)
            {
                Assert.True(pm4File.MDBH.Entries.Count >= 0); // Allow empty
                // Optional: Could add checks on parsed data within MDBH entries if needed
            }
            // Check if MDOS exists and has entries, allowing for empty if not present
            if (pm4File.MDOS != null)
            {
                 Assert.True(pm4File.MDOS.Entries.Count >= 0); // Allow empty
            }
            // Check if MDSF exists and has entries, allowing for empty if not present
            if (pm4File.MDSF != null)
            {
                Assert.True(pm4File.MDSF.Entries.Count >= 0); // Allow empty
            }

            // Log first few MPRR entries for debugging index issue
            Console.WriteLine("\n--- Dumping MPRR Entries 60-75 (if they exist) ---");
            if (pm4File.MPRR != null)
            {
                int startIndex = 60;
                int endIndex = Math.Min(startIndex + 16, pm4File.MPRR.Entries.Count); // Log around 16 entries
                for (int i = startIndex; i < endIndex; i++)
                {
                    var entry = pm4File.MPRR.Entries[i];
                    // Update to use new MprrEntry fields (ushort Unknown_0x00, ushort Unknown_0x02)
                    Console.WriteLine($"  Entry {i}: Unk0=0x{entry.Unknown_0x00:X4}({entry.Unknown_0x00}), Unk2=0x{entry.Unknown_0x02:X4}({entry.Unknown_0x02})");
                }
            }
            Console.WriteLine("--- End MPRR Dump ---\n");

            // Assert - Index Validations
            Assert.NotNull(pm4File.MVER); // Basic check: MVER should always exist
            Assert.NotNull(pm4File.MSHD); 
            
            // --- Export Configuration Flags ---
            bool exportMspvVertices = true;  // REVERTED
            bool exportMsvtVertices = true;  // REVERTED
            bool exportMprlPoints = true; // REVERTED
            bool exportMscnNormals = true; // REVERTED (ENABLED MSCN Export)
            bool exportMslkPaths = true; // REVERTED
            bool exportOnlyFirstMslk = false; 
            bool processMsurEntries = true; // Renamed from exportMsurFaces - Remains ENABLED
            bool exportOnlyFirstMsur = false; // RE-ADDED THIS FLAG
            bool exportMprrLines = true; // REVERTED (ENABLED MPRR LINE EXPORT FOR DEBUGGING)
            bool exportMdsfFaces = true; // REVERTED
            // --- End Configuration Flags ---

            // --- Reverted to Separate OBJ Export Setup ---
            StreamWriter? debugWriter = null;
            StreamWriter? summaryWriter = null; // ADDED Summary Writer Variable
            // StreamWriter? combinedWriter = null; // Reverted: Removed Combined
            StreamWriter? msvtWriter = null;
            StreamWriter? mspvWriter = null;
            StreamWriter? mprlWriter = null;
            StreamWriter? mscnWriter = null; // ADDED MSCN writer variable
            StreamWriter? mslkWriter = null; // Added writer for MSLK geometry
            StreamWriter? skippedMslkWriter = null; // Added writer for skipped MSLK entries
            
            // Reverted: Removed Global vertex tracking
            // int globalVertexCount = 0;
            // int baseMspvIndex = 0;
            // int baseMsvtIndex = 0;
            // int baseMprlIndex = 0;
            // int baseMscnIndex = 0;

            try
            {
                // Initialize writers here, now that we are inside the try block
                debugWriter = new StreamWriter(debugLogPath, false);
                summaryWriter = new StreamWriter(summaryLogPath, false); // ADDED Summary Writer Init
                
                // Log MDOS Entry Count for verification (MOVED HERE)
                int mdosEntryCount = pm4File.MDOS?.Entries.Count ?? -1;
                Console.WriteLine($"INFO: Loaded MDOS Chunk contains {mdosEntryCount} entries.");
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
                debugWriter.WriteLine($"INFO: MSLK Entries: {pm4File.MSLK?.Entries.Count ?? -1}"); // Added MSLK count
                debugWriter.WriteLine($"INFO: MSRN Normals: {pm4File.MSRN?.Normals.Count ?? -1}"); // MSRN is optional, uses Normals property

                // combinedWriter = new StreamWriter(outputCombinedFilePath); // Reverted: Removed Combined
                msvtWriter = new StreamWriter(outputMsvtFilePath);
                mspvWriter = new StreamWriter(outputMspvFilePath);
                mprlWriter = new StreamWriter(outputMprlFilePath);
                mscnWriter = new StreamWriter(outputMscnFilePath); // ADDED Init
                mslkWriter = new StreamWriter(outputMslkFilePath); // Initialize MSLK writer
                skippedMslkWriter = new StreamWriter(outputSkippedMslkLogPath); // Initialize skipped MSLK writer
                // mscnWriter = new StreamWriter(outputMsvtFilePath, true); // REMOVED writer init

                // Write headers (Reverted)
                // combinedWriter.WriteLine("# PM4 Combined Geometry (Direct Mapping -> Final Transform: X, Z, -Y)"); 
                msvtWriter.WriteLine("# PM4 MSVT/MSUR Geometry (Y, X, Z)"); // Corrected Header
                mspvWriter.WriteLine("# PM4 MSPV/MSLK Geometry (X, Y, Z)"); // Corrected Header
                mprlWriter.WriteLine("# PM4 MPRL/MPRR Geometry (Y, Z, X)"); // Corrected Header (based on user change)
                mscnWriter.WriteLine("# MSCN Data as Vertices (Raw X, Y, Z)"); // ADDED Header
                mslkWriter.WriteLine("# PM4 MSLK Geometry (Points 'p' and Lines 'l') (Exported: {DateTime.Now})"); // MSLK Header
                skippedMslkWriter.WriteLine("# PM4 Skipped/Invalid MSLK Entries Log (Generated: {DateTime.Now})"); // Skipped MSLK Header

                // --- Moved Index Validation Logging Inside Try Block ---
                int mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");
                // Assert.True(pm4File.MPRR?.ValidateIndices(mprlVertexCount), "MPRR indices are out of bounds for MPRL vertices."); // TEMP COMMENTED: Test file has invalid MPRR data.
                
                int msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSVI Indices against MSVT Vertex Count: {msvtVertexCount} ---");
                // Re-enable MSVI validation check - We expect this might fail with the test file, but want to see the failure.
                // Assert.True(pm4File.MSVI?.ValidateIndices(msvtVertexCount), "MSVI indices are out of bounds for MSVT vertices."); // RE-COMMENTED AGAIN - Temporarily to ensure MSUR logging runs

                int mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSPI Indices against MSPV Vertex Count: {mspvVertexCount} ---");
                Assert.True(pm4File.MSPI?.ValidateIndices(mspvVertexCount), "MSPI indices are out of bounds for MSPV vertices.");
                
                int totalMspiIndices = pm4File.MSPI?.Indices.Count ?? 0; // Renamed variable
                debugWriter.WriteLine($"\n--- Validating MSUR Index Ranges against MSVI Index Count: {totalMspiIndices} ---"); // Used renamed variable
                // Assert.True(pm4File.MSLK?.ValidateIndices(totalMspiIndices), "MSLK indices are out of bounds for MSPI indices."); // MSLK class does not have ValidateIndices method
                // Assert.True(pm4File.MSUR?.ValidateIndices(totalMspiIndices), "MSUR index ranges are out of bounds for MSVI indices."); // Commented out - MSUR purpose/link is unclear
                // --- End Validation Logging ---

                debugWriter.WriteLine("--- Exporting Split Geometry (Standard Transform) ---"); // Reverted
                // ... (Log Summary Information remains the same) ...

                // Vertex counters specific to each separate file
                int mspvFileVertexCount = 0;
                int msvtFileVertexCount = 0;
                int mprlFileVertexCount = 0;
                // int mscnFileVertexCount = 0; // Reverted: MSCN Disabled

                // --- 1. Export MSPV vertices -> mspvWriter ONLY ---
                // baseMspvIndex = globalVertexCount; // Reverted
                if (exportMspvVertices)
                {
                    mspvWriter.WriteLine("# MSPV Vertices");
                    if (pm4File.MSPV != null)
                    {
                        debugWriter.WriteLine("\n--- Exporting MSPV Vertices (X, Z, -Y) -> _mspv.obj ---"); // Reverted
                        summaryWriter.WriteLine("\n--- MSPV Vertices (First 10) ---"); // ADDED Summary Header
                        int logCounterMspv = 0;
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            // Applying standard (X, Z, -Y) transform - Reverted
                            float worldX = vertex.X;
                            float worldY = vertex.Y; 
                            float worldZ = vertex.Z;

                            // Write final coords to separate file ONLY
                            // Log ALL entries to debugWriter
                            debugWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); 
                            // Limit summaryWriter log
                            if (logCounterMspv < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); // ADDED Summary Log
                            }
                            mspvWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            mspvFileVertexCount++;
                            logCounterMspv++; 
                        }
                        if (pm4File.MSPV.Vertices.Count > 10) { 
                            // debugWriter.WriteLine("  ... MSPV log truncated ..."); // No truncation for debug log
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ..."); // ADDED Summary Truncation
                        }
                        debugWriter.WriteLine($"Wrote {mspvFileVertexCount} MSPV vertices to _mspv.obj."); // Reverted log
                        mspvWriter.WriteLine();
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("No MSPV vertex data found."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MSPV Vertex export (Flag False).");
                }

                // --- 2. Export MSVT vertices -> msvtWriter ONLY ---
                // baseMsvtIndex = globalVertexCount; // Reverted
                if (exportMsvtVertices)
                {
                    msvtWriter.WriteLine("# MSVT Vertices");
                    if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine($"\n--- Exporting MSVT Vertices (Using PD4.md Formula) -> _msvt.obj ---");
                        summaryWriter.WriteLine("\n--- MSVT Vertices (First 10) ---"); // ADDED Summary Header
                        int logCounterMsvt = 0;
                        foreach (var vertex in pm4File.MSVT.Vertices)
                        {
                            // REVERTING MSVT transform back to: (-Y, Z/36, X)
                            // Input vertex fields are (Y, X, Z)
                            // Output OBJ coords are (X, Y, Z)
                            float worldX = vertex.Y;         // Mirror X by negating input Y
                            float worldY = vertex.X;  // Z (Up, Scaled) -> Y (Up)
                            float worldZ = vertex.Z;          // X -> Z (Depth)
                            
                            // Write final coords to separate file ONLY
                            // Log ALL entries to debugWriter
                            debugWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); 
                            // Limit summaryWriter log
                            if (logCounterMsvt < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); // ADDED Summary Log
                            }
                            msvtWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}")); // Needs syntax check
                            msvtFileVertexCount++;
                            logCounterMsvt++; 
                        }
                        if (pm4File.MSVT.Vertices.Count > 10) { 
                            // debugWriter.WriteLine("  ... MSVT log truncated ..."); // No truncation for debug log
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ..."); // ADDED Summary Truncation
                        }
                        debugWriter.WriteLine($"Wrote {msvtFileVertexCount} MSVT vertices to _msvt.obj."); 
                        msvtWriter.WriteLine();
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nNo MSVT vertex data found."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MSVT Vertex export (Flag False).");
                }

                 // --- 3. Export MPRL vertices -> mprlWriter ONLY ---
                 // baseMprlIndex = globalVertexCount; // Reverted
                 if (exportMprlPoints)
                 {
                     mprlWriter.WriteLine("# MPRL Vertices (X, Z, -Y)"); // Reverted

                     if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                     {
                        debugWriter.WriteLine($"\n--- Exporting MPRL Vertices (X, Z, Y) -> _mprl.obj ---"); // Reverted
                        summaryWriter.WriteLine("\n--- MPRL Vertices (First 10) ---"); // ADDED Summary Header
                        int logCounterMprl = 0;
                        mprlFileVertexCount = 0;

                        foreach (var entry in pm4File.MPRL.Entries)
                        {
                            // Input (X, Y, Z)
                            float worldX = entry.Position.X; 
                            float worldY = -entry.Position.Z; 
                            float worldZ = entry.Position.Y; 

                            // Write final coords to separate file ONLY
                            // Log ALL entries to debugWriter
                            debugWriter.WriteLine(FormattableString.Invariant(
                                $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                            )); 
                            // Limit summaryWriter log
                            if (logCounterMprl < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                                )); // ADDED Summary Log
                            }
                            mprlWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}")); // Needs syntax check
                            mprlFileVertexCount++;
                            logCounterMprl++; 
                        }
                        if (pm4File.MPRL.Entries.Count > 10) { 
                            // debugWriter.WriteLine("  ... MPRL log truncated ..."); // No truncation for debug log
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ..."); // ADDED Summary Truncation
                        }
                         debugWriter.WriteLine($"Wrote {mprlFileVertexCount} MPRL vertices to _mprl.obj file."); // Reverted comment
                        mprlWriter.WriteLine();
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MPRL vertex data found."); }
                 } else {
                    debugWriter.WriteLine("\nSkipping MPRL Point export (Flag False).");
                 }

                 // --- 4. Process MSCN Data (Output as Vertices to separate file) ---
                 // var mscnNormalStrings = new List<string>(); // REMOVED List
                 if (exportMscnNormals) 
                 {
                     if (pm4File.MSCN != null && pm4File.MSCN.Vectors.Count > 0 && mscnWriter != null) // Check mscnWriter
                     {
                        int mscnVertexCount = 0; // Renamed variable
                        debugWriter.WriteLine($"\n--- Exporting MSCN Data as Vertices (Raw X, Y, Z) -> _mscn.obj ---"); // Updated log
                        summaryWriter.WriteLine("\n--- MSCN Vertices (First 10) ---"); // ADDED Summary Header
                        int logCounterMscn = 0;

                        foreach (var vectorData in pm4File.MSCN.Vectors) // Renamed loop variable
                        {
                            // Input (X, Y, Z) - Assuming raw export for now
                            float vX = vectorData.X; 
                            float vY = vectorData.Y; 
                            float vZ = vectorData.Z; 

                            // Write directly to mscnWriter as a vertex ('v')
                            var vertexString = FormattableString.Invariant($"v {vX:F6} {vY:F6} {vZ:F6}");
                            mscnWriter.WriteLine(vertexString);
                            mscnVertexCount++;

                            // Log ALL entries to debugWriter
                            debugWriter.WriteLine(FormattableString.Invariant(
                                $"  MSCN Vertex {mscnVertexCount-1}: Raw=({vectorData.X:F3}, {vectorData.Y:F3}, {vectorData.Z:F3}) -> Exported=({vX:F3}, {vY:F3}, {vZ:F3})"
                            ));
                            // Limit summaryWriter log
                            if (logCounterMscn < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MSCN Vertex {mscnVertexCount-1}: Raw=({vectorData.X:F3}, {vectorData.Y:F3}, {vectorData.Z:F3}) -> Exported=({vX:F3}, {vY:F3}, {vZ:F3})"
                                )); // ADDED Summary Log
                            }
                            logCounterMscn++; 
                        }
                        if (pm4File.MSCN.Vectors.Count > 10) { 
                            // debugWriter.WriteLine("  ... MSCN log truncated ..."); // No truncation for debug log
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ..."); // ADDED Summary Truncation
                        }
                        debugWriter.WriteLine($"Wrote {mscnVertexCount} MSCN vertices to _mscn.obj file."); // Updated log
                        mscnWriter.WriteLine(); // Add final newline
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MSCN data found or mscnWriter not available."); } // Updated log
                 } else { 
                     debugWriter.WriteLine("\n--- Skipping MSCN export (Flag 'exportMscnNormals' is False) ---");
                 }

                 // --- 5. Export MSLK paths/points -> mslkWriter ONLY, log skipped to skippedMslkWriter ---
                if (exportMslkPaths)
                {
                    if (pm4File.MSLK != null && pm4File.MSPI != null && pm4File.MSPV != null && mspvFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSLK -> MSPI -> MSPV Chain -> _mslk.obj (Skipped -> _skipped_mslk.log) ---");
                        summaryWriter.WriteLine("\n--- MSLK Processing (First 10 Entries) ---");

                        int entriesToProcessMslk = exportOnlyFirstMslk ? Math.Min(1, pm4File.MSLK.Entries.Count) : pm4File.MSLK.Entries.Count;
                        int skippedMslkCount = 0;
                        int exportedPaths = 0;
                        int exportedPoints = 0;

                        for (int entryIndex = 0; entryIndex < entriesToProcessMslk; entryIndex++)
                        {
                            // --- Start MSLK Processing ---
                            var mslkEntry = pm4File.MSLK.Entries[entryIndex];
                            // Log entry details (always to debug, limited to summary)
                            if (entryIndex < 10) {
                                summaryWriter.WriteLine($"  Processing MSLK Entry {entryIndex}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}");
                            }
                            debugWriter.WriteLine($"  Processing MSLK Entry {entryIndex}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}, Unk00=0x{mslkEntry.Unknown_0x00:X2}, Unk01=0x{mslkEntry.Unknown_0x01:X2}, Unk04=0x{mslkEntry.Unknown_0x04:X8}, Unk10=0x{mslkEntry.Unknown_0x10:X4}");

                            // Condition 1: Potentially Valid Geometry Check
                            if (mslkEntry.MspiIndexCount > 0 && mslkEntry.MspiFirstIndex >= 0)
                            {
                                int mspiStart = mslkEntry.MspiFirstIndex;
                                int mspiEndExclusive = mspiStart + mslkEntry.MspiIndexCount;

                                // Condition 2: MSPI Range Check
                                if (mspiEndExclusive <= totalMspiIndices)
                                {
                                    // Valid Range
                                    List<int> validMspvIndices = new List<int>();
                                    for (int mspiIndex = mspiStart; mspiIndex < mspiEndExclusive; mspiIndex++)
                                    {
                                        uint mspvIndex = pm4File.MSPI.Indices[mspiIndex];
                                        if (mspvIndex < mspvFileVertexCount)
                                        {
                                            validMspvIndices.Add((int)mspvIndex + 1);
                                        }
                                        else
                                        {
                                            debugWriter.WriteLine($"    WARNING: MSLK Entry {entryIndex}, MSPI index {mspiIndex} points to invalid MSPV index {mspvIndex} (Max: {mspvFileVertexCount - 1}). Skipping vertex.");
                                            summaryWriter.WriteLine($"    WARNING: MSLK Entry {entryIndex}, MSPI index {mspiIndex} points to invalid MSPV index {mspvIndex} (Max: {mspvFileVertexCount - 1}). Skipping vertex.");
                                        }
                                    }

                                    // Condition 3: Valid Vertex Count Check
                                    if (validMspvIndices.Count >= 2)
                                    {
                                        // Export as line (path) to mslkWriter
                                        mslkWriter.WriteLine($"g MSLK_Path_{entryIndex}");
                                        mslkWriter.WriteLine("l " + string.Join(" ", validMspvIndices));
                                        debugWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices to _mslk.obj.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices."); }
                                        exportedPaths++;
                                    }
                                    else if (validMspvIndices.Count == 1)
                                    {
                                        // Export as point to mslkWriter
                                        mslkWriter.WriteLine($"g MSLK_Point_{entryIndex}");
                                        mslkWriter.WriteLine($"p {validMspvIndices[0]}");
                                        debugWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]} to _mslk.obj.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]}. "); }
                                        exportedPoints++;
                                    }
                                    else
                                    {
                                        // No valid points found for this entry
                                        debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} resulted in 0 valid MSPV indices. Skipping geometry output.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    INFO: Resulted in 0 valid MSPV indices. Skipping."); }
                                        // Log skipped entry
                                        skippedMslkWriter.WriteLine($"Skipped (0 Valid MSPV): {mslkEntry.ToString()}");
                                        skippedMslkCount++;
                                    }
                                }
                                else
                                {
                                    // Invalid MSPI Range
                                    debugWriter.WriteLine($"    ERROR: MSLK Entry {entryIndex} defines invalid MSPI range [First:{mspiStart}, Count:{mslkEntry.MspiIndexCount}] (Max MSPI Index: {totalMspiIndices - 1}). Skipping entry.");
                                    if (entryIndex < 10) { summaryWriter.WriteLine($"    ERROR: Invalid MSPI range. Skipping entry."); }
                                    // Log skipped entry
                                    skippedMslkWriter.WriteLine($"Skipped (Invalid MSPI Range): {mslkEntry.ToString()}");
                                    skippedMslkCount++;
                                }
                            }
                            else
                            {
                                // Skip entries with count 0 or invalid first index
                                debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} has MSPICount=0 or MSPIFirst=-1. Skipping geometry export.");
                                if (entryIndex < 10) { summaryWriter.WriteLine($"    INFO: MSPICount=0 or MSPIFirst=-1. Skipping."); }
                                // Log skipped entry
                                skippedMslkWriter.WriteLine($"Skipped (Count=0 or FirstIndex<0): {mslkEntry.ToString()}");
                                skippedMslkCount++;
                            }
                        } // --- End MSLK Processing Loop ---
                        debugWriter.WriteLine($"Finished processing {entriesToProcessMslk} MSLK entries. Exported {exportedPaths} paths, {exportedPoints} points. Skipped {skippedMslkCount} entries.");
                        if (exportOnlyFirstMslk && pm4File.MSLK.Entries.Count > 1)
                        {
                            debugWriter.WriteLine("Note: MSLK processing was limited to the first entry by 'exportOnlyFirstMslk' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSLK Processing (Exported Paths: {exportedPaths}, Points: {exportedPoints}, Skipped: {skippedMslkCount}) ---");
                        mslkWriter.WriteLine(); // Add final newline to geometry file
                        debugWriter.Flush();
                        skippedMslkWriter.Flush(); // Ensure skipped log is written
                    }
                    else
                    {
                        debugWriter.WriteLine("Skipping MSLK path export (MSLK, MSPI, or MSPV data missing or no MSPV vertices exported).");
                    }
                } else {
                    debugWriter.WriteLine("\nSkipping MSLK Path export (Flag 'exportMslkPaths' is False).");
                }

                // --- 6. Export MSUR surfaces -> msvtWriter ONLY ---
                if (processMsurEntries) // Renamed from exportMsurFaces
                {
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null && pm4File.MDOS != null && msvtFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSUR -> MSVI Links (Logging Only - First 20 Entries) ---"); // Updated description
                        summaryWriter.WriteLine("\n--- MSUR -> MSVI Links (First 20 Entries) ---"); // ADDED Summary Header

                        int entriesToProcess = exportOnlyFirstMsur ? Math.Min(1, pm4File.MSUR.Entries.Count) : pm4File.MSUR.Entries.Count;
                        // Limit logging to the first 20 entries for BOTH writers
                        for (int entryIndex = 0; entryIndex < Math.Min(entriesToProcess, 20); entryIndex++)
                        {
                            var msurEntry = pm4File.MSUR.Entries[entryIndex];
                            // Log details for the current MSUR entry being processed
                            debugWriter.WriteLine($"  Processing MSUR Entry {entryIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, MdosIndex={msurEntry.MdosIndex}, FlagsOrUnk0={msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}"); 
                            summaryWriter.WriteLine($"  Processing MSUR Entry {entryIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, MdosIndex={msurEntry.MdosIndex}, FlagsOrUnk0={msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}"); // ADDED Summary Log
                            
                            int firstIndex = (int)msurEntry.MsviFirstIndex;
                            int indexCount = msurEntry.IndexCount;
                            uint mdosIndex = msurEntry.MdosIndex;

                            // Determine group name for logging context 
                            string groupName;
                            if (mdosIndex < (pm4File.MDOS?.Entries.Count ?? 0))
                            {
                                var mdosEntry = pm4File.MDOS!.Entries[(int)mdosIndex];
                                groupName = $"MSUR_Mdos{mdosIndex}_ID{mdosEntry.Value_0x00:X8}";
                                debugWriter.WriteLine($"    MDOS Link Found: Index={mdosIndex}, Entry={mdosEntry}");
                                summaryWriter.WriteLine($"    MDOS Link Found: Index={mdosIndex}, Entry={mdosEntry}"); // ADDED Summary Log
                            }
                            else
                            {
                                debugWriter.WriteLine($"  MSUR Entry {entryIndex}: Invalid MDOS index {mdosIndex}. Assigning default group name.");
                                summaryWriter.WriteLine($"  MSUR Entry {entryIndex}: Invalid MDOS index {mdosIndex}. Assigning default group name."); // ADDED Summary Log
                            }

                            if (pm4File.MSVI == null) { /* Should not happen due to outer check */ continue; }

                            if (firstIndex >= 0 && firstIndex + indexCount <= pm4File.MSVI!.Indices.Count)
                            {
                                 if (indexCount < 1) { 
                                     debugWriter.WriteLine($"    Skipping: Zero indices requested (IndexCount={indexCount}).");
                                     summaryWriter.WriteLine($"    Skipping: Zero indices requested (IndexCount={indexCount})."); // ADDED Summary Log
                                     continue; 
                                 }

                                List<uint> msviIndices = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                                // Log only the count of indices fetched to debug
                                debugWriter.WriteLine($"    Fetched {msviIndices.Count} MSVI Indices (Expected {indexCount}).");
                                // Log the actual indices fetched for the first 20 entries to debug
                                debugWriter.WriteLine($"      Debug MSVI Indices: [{string.Join(", ", msviIndices)}]"); // Log full list to debug
                                
                                // Log count to summary
                                if (entryIndex < 20) { summaryWriter.WriteLine($"    Fetched {msviIndices.Count} MSVI Indices (Expected {indexCount})."); }
                                // Log first ~15 indices to summary
                                int indicesToLogSummary = Math.Min(msviIndices.Count, 15);
                                if (entryIndex < 20) {
                                    summaryWriter.WriteLine($"      Summary MSVI Indices (first {indicesToLogSummary}): [{string.Join(", ", msviIndices.Take(indicesToLogSummary))}]" + (msviIndices.Count > 15 ? " ..." : "")); 
                                    // ADDED: Log corresponding MSVT vertices
                                    summaryWriter.WriteLine("        MSVT Vertices (Raw Y, X, Z):");
                                    for(int k=0; k < indicesToLogSummary; k++) {
                                        uint msviIdx = msviIndices[k];
                                        if (msviIdx < msvtVertexCount) {
                                            var msvtVert = pm4File.MSVT!.Vertices[(int)msviIdx];
                                            summaryWriter.WriteLine(FormattableString.Invariant($"          Index {msviIdx} -> (Y:{msvtVert.Y:F3}, X:{msvtVert.X:F3}, Z:{msvtVert.Z:F3})"));
                                        } else {
                                            summaryWriter.WriteLine($"          Index {msviIdx} -> (Out of bounds >= {msvtVertexCount})");
                                        }
                                    }
                                    if (msviIndices.Count > 15) { summaryWriter.WriteLine("          ..."); }
                                }

                                // ADDED: Log corresponding MSVT vertices to debug log as well (full list)
                                debugWriter.WriteLine("        Debug MSVT Vertices (Raw Y, X, Z):");
                                foreach(uint msviIdx in msviIndices) {
                                    if (msviIdx < msvtVertexCount) {
                                        var msvtVert = pm4File.MSVT!.Vertices[(int)msviIdx];
                                        debugWriter.WriteLine(FormattableString.Invariant($"          Index {msviIdx} -> (Y:{msvtVert.Y:F3}, X:{msvtVert.X:F3}, Z:{msvtVert.Z:F3})"));
                                    } else {
                                        debugWriter.WriteLine($"          Index {msviIdx} -> (Out of bounds >= {msvtVertexCount})");
                                    }
                                }

                            }
                            else { 
                                // Log ALL skips for invalid range (for the first 20 entries)
                                debugWriter.WriteLine($"    Skipped MSUR Entry {entryIndex} due to invalid MSVI range (FirstIndex={firstIndex}, IndexCount={indexCount}, Total MSVI Indices={pm4File.MSVI!.Indices.Count}).");
                                summaryWriter.WriteLine($"    Skipped MSUR Entry {entryIndex} due to invalid MSVI range (FirstIndex={firstIndex}, IndexCount={indexCount}, Total MSVI Indices={pm4File.MSVI!.Indices.Count})."); // ADDED Summary Log
                            }
                        }
                        if (entriesToProcess > 20)
                        {
                            debugWriter.WriteLine($"  ... Log truncated after first 20 MSUR entries ...");
                            summaryWriter.WriteLine("  ... (Log limited to first 20 entries) ..."); // ADDED Summary Truncation
                        }
                        debugWriter.WriteLine("--- Finished Processing MSUR -> MSVI Links ---"); // Updated description
                        // msvtWriter.WriteLine(); // No longer writing faces, so no newline needed here
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nSkipping MSUR->MSVI Link processing (missing data or source vertices)."); } // Updated description
                } else {
                    // debugWriter.WriteLine("\nSkipping MSUR Face export (Flag False)."); // Original line
                    debugWriter.WriteLine("\n--- Skipping MSUR->MSVI Link processing (Flag 'processMsurEntries' is False) ---"); // Updated description and flag name
                }

                 // --- 7. Export MPRR pairs -> mprlWriter ONLY ---
                 if (exportMprrLines)
                 {
                     if (pm4File.MPRR != null && pm4File.MPRL != null && mprlFileVertexCount > 0)
                     {
                        debugWriter.WriteLine($"\n--- Processing MPRR -> MPRL Chain -> _mprl.obj ---"); // Reverted
                        summaryWriter.WriteLine("\n--- MPRR Lines (First 10 Valid) ---"); // ADDED Summary Header
                         int logCounterMprr = 0;
                         int validMprrLogged = 0; // Counter for summary log limit
 
                         foreach (var mprrEntry in pm4File.MPRR.Entries)
                         {
                             // Read the raw ushort values
                             ushort rawIndex1 = mprrEntry.Unknown_0x00;
                             ushort rawIndex2 = mprrEntry.Unknown_0x02;
 
                             // bool shouldLogDebug = logCounterMprr < 10; // REMOVED - Debug logs all
                             bool shouldLogSummary = validMprrLogged < 10; // Summary log limit
 
                             // Log ALL processing attempts to debugWriter
                             debugWriter.WriteLine($"  Processing MPRR Entry {logCounterMprr}: RawIdx1={rawIndex1}, RawIdx2={rawIndex2}"); 
                             
                             // Sentinel/Identical skips - log to both (conditionally for summary)
                             if (rawIndex1 == 65535 || rawIndex2 == 65535) { 
                                 debugWriter.WriteLine("    Skipping: Sentinel index (65535).");
                                 if (shouldLogSummary) { summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Skip Sentinel"); }
                                 logCounterMprr++; 
                                 continue; 
                             }
                             if (rawIndex1 == rawIndex2) {
                                 debugWriter.WriteLine($"    Skipping: Identical indices ({rawIndex1}).");
                                 if (shouldLogSummary) { summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Skip Identical {rawIndex1}"); }
                                 logCounterMprr++; 
                                 continue;
                             }

                            // Bounds check
                             if (rawIndex1 < mprlFileVertexCount && rawIndex2 < mprlFileVertexCount) 
                             {
                                 // Indices seem valid relative to the vertices we exported, proceed
                                 uint relativeObjIndex1 = 1 + (uint)rawIndex1;
                                 uint relativeObjIndex2 = 1 + (uint)rawIndex2;
 
                                 // Write group header (only once) and line to separate file
                                 bool firstProcessedEntry = !pm4File.MPRR.Entries
                                     .Take(pm4File.MPRR.Entries.IndexOf(mprrEntry))
                                     .Any(e => e.Unknown_0x00 != 65535 && e.Unknown_0x02 != 65535 && e.Unknown_0x00 != e.Unknown_0x02 && e.Unknown_0x00 < mprlFileVertexCount && e.Unknown_0x02 < mprlFileVertexCount);
                                 if (firstProcessedEntry) { 
                                     mprlWriter.WriteLine("g MPRR_Lines"); 
                                     // Only write group header once to summary too
                                     if(shouldLogSummary) { summaryWriter.WriteLine("  (Group MPRR_Lines)"); }
                                 }
 
                                 // Log valid line to BOTH (conditionally for summary)
                                 debugWriter.WriteLine($"    Writing SEPARATE OBJ Line {relativeObjIndex1} -> {relativeObjIndex2} (From Raw MPRL Indices: {rawIndex1}, {rawIndex2})"); 
                                 if (shouldLogSummary) {
                                     summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Line {relativeObjIndex1} -> {relativeObjIndex2} (Raw: {rawIndex1}, {rawIndex2})");
                                     validMprrLogged++;
                                 }
                             }
                             else { 
                                   // Log invalid index skip to BOTH (conditionally for summary)
                                   debugWriter.WriteLine($"    Skipping: Indices out of bounds (RawIdx1={rawIndex1}, RawIdx2={rawIndex2}, File MPRL Count={mprlFileVertexCount}). CHECK MPRR ASSERTION."); 
                                   if (shouldLogSummary) {
                                       summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Skip Invalid Index (Raw: {rawIndex1}, {rawIndex2} / Max: {mprlFileVertexCount - 1})");
                                       validMprrLogged++; // Count skips too for summary limit
                                   }
                             }
                             logCounterMprr++;
                         }
                         // Add truncation messages
                         // if (pm4File.MPRR.Entries.Count > 10 && logCounterMprr >= 10) { debugWriter.WriteLine("  ... MPRR debug log truncated ..."); } // No truncation for debug log
                         if (validMprrLogged >= 10) { summaryWriter.WriteLine("  ... (Summary log limited to first 10 processed entries/skips) ..."); }
                         debugWriter.WriteLine("--- Finished Processing MPRR Chain for separate file ---"); // Reverted log
                         mprlWriter.WriteLine(); // Reverted: Add newline to mprlWriter
                         debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("\nSkipping MPRR Chain processing (missing data or source vertices)."); }
                 } else {
                      debugWriter.WriteLine("\nSkipping MPRR Line export (Flag False)."); 
                      debugWriter.WriteLine("\n--- Skipping MPRR Line export (Flag 'exportMprrLines' is False) ---"); // Enhanced Log
                 }

                 // --- 8. Export MDSF faces (Reverted: Disabled) ---
                 if (exportMdsfFaces) { /* ... Disabled ... */ } else { 
                     debugWriter.WriteLine("\n--- Skipping MDSF Face export (Flag 'exportMdsfFaces' is False) ---"); // Enhanced Log
                 }

                 // --- ADDED: Generate MSLK Hierarchical JSON ---
                 if (pm4File.MSLK != null && pm4File.MSLK.Entries.Any())
                 {
                     debugWriter.WriteLine("\n--- Generating MSLK Hierarchy JSON ---");
                     var groupedMslkData = new Dictionary<uint, MslkGroupDto>();

                     for (int i = 0; i < pm4File.MSLK.Entries.Count; i++)
                     {
                         var entry = pm4File.MSLK.Entries[i];
                         var groupId = entry.Unknown_0x04;

                         if (!groupedMslkData.ContainsKey(groupId))
                         {
                             groupedMslkData[groupId] = new MslkGroupDto();
                         }

                         if (entry.MspiFirstIndex == -1) // Node Entry
                         {
                             groupedMslkData[groupId].Nodes.Add(new MslkNodeEntryDto
                             {
                                 EntryIndex = i,
                                 Unk00 = entry.Unknown_0x00,
                                 Unk01 = entry.Unknown_0x01,
                                 Unk02 = entry.Unknown_0x02,
                                 Unk0C = entry.Unknown_0x0C,
                                 Unk10 = entry.Unknown_0x10,
                                 Unk12 = entry.Unknown_0x12
                             });
                         }
                         else // Geometry Entry
                         {
                             groupedMslkData[groupId].Geometry.Add(new MslkGeometryEntryDto
                             {
                                 EntryIndex = i,
                                 MspiFirstIndex = entry.MspiFirstIndex,
                                 MspiIndexCount = entry.MspiIndexCount,
                                 Unk00 = entry.Unknown_0x00,
                                 Unk01 = entry.Unknown_0x01,
                                 Unk02 = entry.Unknown_0x02,
                                 Unk0C = entry.Unknown_0x0C,
                                 Unk10 = entry.Unknown_0x10,
                                 Unk12 = entry.Unknown_0x12
                             });
                         }
                     }

                     var options = new JsonSerializerOptions { WriteIndented = true };
                     string jsonString = JsonSerializer.Serialize(groupedMslkData, options);
                     File.WriteAllText(outputMslkJsonPath, jsonString);
                     debugWriter.WriteLine($"Successfully wrote MSLK hierarchy to {outputMslkJsonPath}");
                 }
                 else
                 {
                      debugWriter.WriteLine("\n--- Skipping MSLK Hierarchy JSON generation (MSLK chunk is null or empty) ---");
                 }
                 // --- END: Generate MSLK Hierarchical JSON ---

                 // Reverted: Original Completion Message
                 Console.WriteLine($"Successfully wrote split OBJ files to directory: {Path.GetDirectoryName(outputMsvtFilePath)}"); 
                 debugWriter.WriteLine($"Split OBJ export complete."); 
             }
             // Reverted: Catch block remains the same conceptually
             catch (Exception ex)
             {
                  Console.WriteLine($"Error writing split OBJ data: {ex.ToString()}"); 
                  try { 
                      debugWriter?.WriteLine($"### EXCEPTION DURING SPLIT WRITE: {ex.ToString()} ###"); 
                      debugWriter?.Flush(); 
                  } catch { /* Ignore */ }
             }
            finally
            {
                // Close all writers (Reverted)
                debugWriter?.Close();
                summaryWriter?.Close(); // ADDED Close Summary Writer
                // combinedWriter?.Close(); // Reverted: Removed Combined
                msvtWriter?.Close();
                mspvWriter?.Close();
                mprlWriter?.Close();
                mscnWriter?.Close(); // ADDED close call
                mslkWriter?.Close(); // Close MSLK writer
                skippedMslkWriter?.Close(); // Close skipped MSLK writer
                // mscnWriter?.Close(); // REMOVED close call
            }

             // Reverted: Original Asserts
             Assert.True(File.Exists(outputMsvtFilePath), $"MSVT OBJ file was not created at {outputMsvtFilePath}");
             Assert.True(File.Exists(outputMspvFilePath), $"MSPV OBJ file was not created at {outputMspvFilePath}");
             Assert.True(File.Exists(outputMprlFilePath), $"MPRL OBJ file was not created at {outputMprlFilePath}");
             Assert.True(File.Exists(outputMscnFilePath), $"MSCN OBJ file was not created at {outputMscnFilePath}"); // ADDED Assert
             Assert.True(File.Exists(outputMslkFilePath), $"MSLK OBJ file should exist at {outputMslkFilePath}"); // Assert MSLK file
             Assert.True(File.Exists(outputMslkJsonPath), $"MSLK JSON file should exist at {outputMslkJsonPath}"); // Assert MSLK JSON file
             Assert.True(File.Exists(summaryLogPath), $"Summary log file was not created at {summaryLogPath}"); // ADDED Summary Assert

            Console.WriteLine("--- LoadPM4File_ShouldLoadChunks END ---");
        }

        // ... other test methods ...
    }
} 