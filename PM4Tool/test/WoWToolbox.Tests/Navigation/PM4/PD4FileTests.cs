using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Numerics;
using Xunit;
using Warcraft.NET.Files.Interfaces;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using WoWToolbox.Core.Navigation.PD4;
using WoWToolbox.Core.Vectors;
using WoWToolbox.Core;
using Microsoft.Extensions.Logging;
using System.Reflection; // Added for reflection

namespace WoWToolbox.Tests.Navigation.PM4 // Keep the same namespace for now
{
    public class PD4FileTests // Renamed class
    {
        // Define paths for the two PD4 test files
        private const string TestDataPath1 = "test_data/development/6or_garrison_workshop_v3_snow.pd4";
        private const string TestDataPath2 = "test_data/development/6or_garrison_workshop_v3_snow_lod1.pd4";

        // Constants for MSVT transformation based on PD4.md
        private const float CoordinateOffset = 17066.666f;
        private const float ScaleFactor = 36.0f;

        // Helper function to transform PD4 MSVT coordinates to world space
        private static Vector3 MsvtToWorld_PD4(MsvtVertex v)
        {
            // Apply PD4 specific transform based on PD4.md and user correction (Z is NOT scaled)
            return new Vector3(
                CoordinateOffset - v.X,      // worldX = offset - vertex.X
                CoordinateOffset - v.Y,      // worldY = offset - vertex.Y
                v.Z                        // worldZ = vertex.Z (No scaling)
            );
        }

        [Fact]
        public void LoadPD4Files_ShouldLoadChunks()
        {
            Console.WriteLine("--- LoadPD4Files_ShouldLoadChunks START ---");

            // --- Process First PD4 File ---
            ProcessPd4File(TestDataPath1);

            // --- Process Second PD4 File ---
            ProcessPd4File(TestDataPath2);

            Console.WriteLine("--- LoadPD4Files_ShouldLoadChunks END ---");
        }

        private void ProcessPd4File(string relativeTestDataPath)
        {
            Console.WriteLine($"\n--- Processing PD4 File: {relativeTestDataPath} ---");
            // Arrange
            var inputFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativeTestDataPath);
            Console.WriteLine($"Full Path: {inputFilePath}");

            // Define Output Paths (relative to input file)
            string baseOutputPath = Path.Combine(Path.GetDirectoryName(relativeTestDataPath) ?? AppDomain.CurrentDomain.BaseDirectory, Path.GetFileNameWithoutExtension(relativeTestDataPath));
            string debugLogPath = baseOutputPath + ".debug.log";
            string summaryLogPath = baseOutputPath + ".summary.log";
            // Add OBJ output paths
            string outputMspvFilePath = baseOutputPath + "_mspv.obj";
            string outputMsvtFilePath = baseOutputPath + "_msvt.obj";
            string outputMscnFilePath = baseOutputPath + "_mscn.obj"; // Added MSCN path
            string outputMslkFilePath = baseOutputPath + "_mslk.obj"; // Added MSLK path
            string outputMslkNodesFilePath = baseOutputPath + "_mslk_nodes.obj"; // ADDED MSLK Node Anchors path

            Console.WriteLine($"Debug Log: {debugLogPath}");
            Console.WriteLine($"Summary Log: {summaryLogPath}");
            Console.WriteLine($"Output MSPV OBJ: {outputMspvFilePath}");
            Console.WriteLine($"Output MSVT OBJ: {outputMsvtFilePath}");
            Console.WriteLine($"Output MSCN OBJ: {outputMscnFilePath}"); // Added log for MSCN path
            Console.WriteLine($"Output MSLK OBJ: {outputMslkFilePath}"); // Added log for MSLK path
            Console.WriteLine($"Output MSLK Nodes OBJ: {outputMslkNodesFilePath}"); // ADDED log for MSLK Nodes path

            // Check if file exists before attempting to load
            if (!File.Exists(inputFilePath))
            {
                 Console.WriteLine($"ERROR: Test data file not found: {inputFilePath}");
                 Assert.Fail($"Test data file not found: {inputFilePath}"); // Use Assert.Fail instead
                 return; // Stop processing this file
            }

            // Act
            PD4File? pd4File = null;
            StreamWriter? debugWriter = null;
            StreamWriter? summaryWriter = null;
            // Add OBJ writers
            StreamWriter? mspvWriter = null;
            StreamWriter? msvtWriter = null;
            StreamWriter? mscnWriter = null; // Added MSCN writer
            StreamWriter? mslkWriter = null; // Added MSLK writer
            StreamWriter? skippedMslkWriter = null; // ADDED writer for skipped MSLK entries
            StreamWriter? mslkNodesWriter = null; // ADDED writer for MSLK nodes

            try
            {
                // Initialize writers
                debugWriter = new StreamWriter(debugLogPath, false);
                summaryWriter = new StreamWriter(summaryLogPath, false);
                // Initialize OBJ writers
                mspvWriter = new StreamWriter(outputMspvFilePath, false);
                msvtWriter = new StreamWriter(outputMsvtFilePath, false);
                mscnWriter = new StreamWriter(outputMscnFilePath, false);
                mslkWriter = new StreamWriter(outputMslkFilePath, false); // Initialized MSLK writer
                skippedMslkWriter = new StreamWriter(baseOutputPath + "_skipped_mslk.log", false); // ADDED skipped writer init
                mslkNodesWriter = new StreamWriter(outputMslkNodesFilePath, false); // ADDED init for nodes writer

                debugWriter.WriteLine($"--- DEBUG LOG FOR {relativeTestDataPath} ---");
                summaryWriter.WriteLine($"--- SUMMARY LOG FOR {relativeTestDataPath} ---");
                // Write header to skipped log
                skippedMslkWriter.WriteLine($"# PD4 Skipped/Invalid MSLK Entries Log ({DateTime.Now})");
                mslkNodesWriter.WriteLine($"# PD4 MSLK Node Anchor Points (from Unk10 -> MSVI -> MSVT)"); // ADDED header for nodes

                // Logging before FromFile call
                Console.WriteLine("DEBUG: About to call PD4File.FromFile...");
                debugWriter.WriteLine("DEBUG: About to call PD4File.FromFile...");

                pd4File = PD4File.FromFile(relativeTestDataPath);

                // Logging after FromFile call
                Console.WriteLine("DEBUG: PD4File.FromFile call completed.");
                debugWriter.WriteLine("DEBUG: PD4File.FromFile call completed.");

                // Assert - Basic Loading Check (Ensures constructor didn't hard crash)
                Assert.NotNull(pd4File);
                debugWriter.WriteLine("File loaded successfully (pd4File object created).");
                summaryWriter.WriteLine("Basic file load successful (object created).");
                Console.WriteLine("Basic file load assertion passed (pd4File object created).");

                // --- Log All Chunk Counts ---
                debugWriter.WriteLine("\n--- Logging Initial Chunk Counts ---");
                debugWriter.WriteLine($"MVER Version: {pd4File.MVER?.Version ?? 0u}");
                debugWriter.WriteLine($"MCRC Unknown0x00: {pd4File.MCRC?.Unknown0x00 ?? 0u}");
                debugWriter.WriteLine($"MSHD Fields: {pd4File.MSHD?.ToString() ?? "null"}");
                debugWriter.WriteLine($"MSPV Vertices: {pd4File.MSPV?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"MSPI Indices: {pd4File.MSPI?.Indices.Count ?? -1}");
                debugWriter.WriteLine($"MSCN Vectors: {pd4File.MSCN?.Vectors.Count ?? -1}");
                debugWriter.WriteLine($"MSLK Entries: {pd4File.MSLK?.Entries.Count ?? -1}");
                debugWriter.WriteLine($"MSVT Vertices: {pd4File.MSVT?.Vertices.Count ?? -1}");
                debugWriter.WriteLine($"MSVI Indices: {pd4File.MSVI?.Indices.Count ?? -1}");
                debugWriter.WriteLine($"MSUR Entries: {pd4File.MSUR?.Entries.Count ?? -1}");
                debugWriter.WriteLine("--- Finished Logging Chunk Counts ---");

                // --- Basic Index Validation Logging ---
                int mspvVertexCount = pd4File.MSPV?.Vertices.Count ?? 0;
                int msvtVertexCount = pd4File.MSVT?.Vertices.Count ?? 0;
                int mspiIndexCount = pd4File.MSPI?.Indices.Count ?? 0;
                int msviIndexCount = pd4File.MSVI?.Indices.Count ?? 0;

                debugWriter.WriteLine("\n--- Validating Indices (Logging Only) ---");
                debugWriter.WriteLine($"Validating MSPI ({mspiIndexCount}) against MSPV ({mspvVertexCount})...");
                // Actual validation logic (e.g., checking max index) could be added here
                bool mspiValid = pd4File.MSPI?.ValidateIndices(mspvVertexCount) ?? true; // Assume valid if null
                debugWriter.WriteLine($"MSPI indices appear valid for MSPV: {mspiValid}");

                debugWriter.WriteLine($"Validating MSVI ({msviIndexCount}) against MSVT ({msvtVertexCount})...");
                bool msviValid = pd4File.MSVI?.ValidateIndices(msvtVertexCount) ?? true; // Assume valid if null
                debugWriter.WriteLine($"MSVI indices appear valid for MSVT: {msviValid}");
                debugWriter.WriteLine("--- Finished Validating Indices ---");

                // --- Export OBJ Geometry ---
                debugWriter.WriteLine("\n--- Exporting OBJ Geometry --- (.NET Culture: " + CultureInfo.CurrentCulture.Name + ")" ); // Log culture
                summaryWriter.WriteLine("\n--- Exporting OBJ Geometry ---");

                // Write OBJ Headers
                mspvWriter.WriteLine("# PD4 MSPV Geometry (Direct X, Y, Z)");
                msvtWriter.WriteLine("# PD4 MSVT Geometry (Transformed)");
                mscnWriter.WriteLine("# PD4 MSCN Vectors (Direct X, Y, Z)");
                mslkWriter.WriteLine("# PD4 MSLK Paths referencing MSPV Vertices");
                mslkNodesWriter.WriteLine("# PD4 MSLK Node Anchor Points (from Unk10 -> MSVI -> MSVT)"); // ADDED header for nodes

                // --- Export MSPV (Vertices needed for MSLK) ---
                if (pd4File.MSPV?.Vertices != null && pd4File.MSPV.Vertices.Count > 0)
                {
                    debugWriter.WriteLine($"Exporting {pd4File.MSPV.Vertices.Count} MSPV vertices (for MSPV and MSLK OBJs)...");
                    // Write to both MSPV and MSLK obj files
                    for (int i = 0; i < pd4File.MSPV.Vertices.Count; i++)
                    {
                        var vertex = pd4File.MSPV.Vertices[i];
                        string line = $"v {vertex.X.ToString(CultureInfo.InvariantCulture)} {vertex.Y.ToString(CultureInfo.InvariantCulture)} {vertex.Z.ToString(CultureInfo.InvariantCulture)}";
                        mspvWriter.WriteLine(line);
                        mslkWriter.WriteLine(line); // Write vertices to MSLK file too
                    }
                    debugWriter.WriteLine("MSPV vertex export complete.");
                }
                else
                {
                    debugWriter.WriteLine("No MSPV vertices found to export.");
                }
                // --- End MSPV Export ---
                
                // --- Export MSVT --- 
                if (pd4File.MSVT?.Vertices != null)
                {
                    debugWriter.WriteLine($"Exporting {pd4File.MSVT.Vertices.Count} MSVT vertices with transformation...");
                    foreach (var vertex in pd4File.MSVT.Vertices)
                    {
                        // Apply PD4 specific transformation
                        Vector3 worldPos = MsvtToWorld_PD4(vertex);
                        // Write transformed vertex
                        msvtWriter.WriteLine($"v {worldPos.X.ToString(CultureInfo.InvariantCulture)} {worldPos.Y.ToString(CultureInfo.InvariantCulture)} {worldPos.Z.ToString(CultureInfo.InvariantCulture)}");
                    }
                    debugWriter.WriteLine("MSVT export complete.");
                }
                else
                {
                    debugWriter.WriteLine("No MSVT vertices to export.");
                }
                // --- End MSVT Export ---

                // --- Export MSCN --- 
                if (pd4File.MSCN?.Vectors != null)
                {
                    debugWriter.WriteLine($"Exporting {pd4File.MSCN.Vectors.Count} MSCN vectors...");
                    foreach (var vec in pd4File.MSCN.Vectors)
                    {
                        // Write vector directly (assuming X, Y, Z)
                        mscnWriter.WriteLine($"v {vec.X.ToString(CultureInfo.InvariantCulture)} {vec.Y.ToString(CultureInfo.InvariantCulture)} {vec.Z.ToString(CultureInfo.InvariantCulture)}");
                    }
                    debugWriter.WriteLine("MSCN export complete.");
                }
                else
                {
                    debugWriter.WriteLine("No MSCN vectors to export.");
                }
                // --- End MSCN Export ---
                
                // --- Export MSLK Paths ---
                debugWriter.WriteLine("\n--- Processing MSLK Path Entries ---");
                skippedMslkWriter.WriteLine("--- Processing MSLK Entries ---"); // Add section to skipped log
                int exportedMslkPaths = 0;
                int skippedMslkEntries = 0;

                if (pd4File.MSLK?.Entries != null && pd4File.MSPI?.Indices != null && mspvVertexCount > 0)
                {
                    int mslkEntryCount = pd4File.MSLK.Entries.Count;
                    debugWriter.WriteLine($"Processing {mslkEntryCount} MSLK entries...");
                    
                    // ADDED: Log message immediately before the loop starts
                    debugWriter.WriteLine($"  Attempting to start MSLK entry loop (i=0 to {mslkEntryCount - 1})...");
                    Console.WriteLine($"  Attempting to start MSLK entry loop (i=0 to {mslkEntryCount - 1})..."); // CONSOLE LOG

                    for(int i = 0; i < mslkEntryCount; i++)
                    {
                        var mslkEntry = pd4File.MSLK.Entries[i];
                        string mslkBaseInfo = string.Format(CultureInfo.InvariantCulture,
                            "Index={0}, FirstIndex={1}, Count={2}, Unk00=0x{3:X2}, Unk01=0x{4:X2}, Unk04=0x{5:X8}, Unk10=0x{6:X4}",
                            i, mslkEntry.MspiFirstIndex, mslkEntry.MspiIndexCount, mslkEntry.Unknown_0x00, mslkEntry.Unknown_0x01, mslkEntry.Unknown_0x04, mslkEntry.Unknown_0x10);

                        // Log entry details in PARSABLE format for AnalysisTool
                        debugWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                            "Processing MSLK Entry {0}: FirstIndex={1}, Count={2}, Unk00=0x{3:X2}, Unk01=0x{4:X2}, Unk04=0x{5:X8}, Unk10=0x{6:X4}",
                             i, mslkEntry.MspiFirstIndex, mslkEntry.MspiIndexCount, mslkEntry.Unknown_0x00, mslkEntry.Unknown_0x01, mslkEntry.Unknown_0x04, mslkEntry.Unknown_0x10));

                        // --- Start Skipped Entry Checks ---
                        bool skipped = false;
                        string skipReason = "";

                        // ADDED: Log values for the node condition check
                        debugWriter.WriteLine($"  Checking Node Condition for Entry {i}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}");
                        Console.WriteLine($"  Checking Node Condition for Entry {i}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}"); // CONSOLE LOG

                        if (mslkEntry.MspiIndexCount == 0 && mslkEntry.MspiFirstIndex == -1) 
                        { 
                            // This is a NODE entry (based on FirstIndex/Count), but double-check Unk00 too for consistency
                            if (mslkEntry.Unknown_0x00 != 0x01)
                            {
                                debugWriter.WriteLine($"  [WARN] Entry {i} looks like a Node (FirstIdx=-1, Count=0) but Unk00 is not 0x01 (Value=0x{mslkEntry.Unknown_0x00:X2}). Treating as Node anyway.");
                                Console.WriteLine($"  [WARN] Entry {i} looks like a Node (FirstIdx=-1, Count=0) but Unk00 is not 0x01 (Value=0x{mslkEntry.Unknown_0x00:X2}). Treating as Node anyway."); // CONSOLE LOG
                            }

                            // --- ADDED: Process Node Anchor Point ---
                            debugWriter.WriteLine($"  Attempting to process anchor point for Node {i}..."); // ADDED PRE-TRY LOG
                            Console.WriteLine($"  Attempting to process anchor point for Node {i}..."); // CONSOLE LOG
                            if (mslkNodesWriter != null && pd4File.MSVI?.Indices != null && pd4File.MSVT?.Vertices != null) 
                            {
                                try 
                                {
                                    ushort msviIndex = mslkEntry.Unknown_0x10; // Unk10 is the potential MSVI index
                                    if (msviIndex < msviIndexCount)
                                    {
                                        int msvtIndex = (int)pd4File.MSVI.Indices[msviIndex]; // EXPLICIT CAST uint -> int
                                        if (msvtIndex >= 0 && msvtIndex < msvtVertexCount)
                                        {
                                            MsvtVertex anchorVertex = pd4File.MSVT.Vertices[msvtIndex];
                                            Vector3 worldPos = MsvtToWorld_PD4(anchorVertex);
                                            mslkNodesWriter.WriteLine($"v {worldPos.X.ToString(CultureInfo.InvariantCulture)} {worldPos.Y.ToString(CultureInfo.InvariantCulture)} {worldPos.Z.ToString(CultureInfo.InvariantCulture)} # Node Idx:{i} Unk01:0x{mslkEntry.Unknown_0x01:X2} Unk10:0x{mslkEntry.Unknown_0x10:X4} MSVI:{msviIndex} MSVT:{msvtIndex}");
                                            debugWriter.WriteLine($"  Node {i} (Unk01=0x{mslkEntry.Unknown_0x01:X2}) anchored via Unk10=0x{mslkEntry.Unknown_0x10:X4} -> MSVI[{msviIndex}] -> MSVT[{msvtIndex}] -> World({worldPos.X:F3}, {worldPos.Y:F3}, {worldPos.Z:F3})");
                                            Console.WriteLine($"  Node {i} (Unk01=0x{mslkEntry.Unknown_0x01:X2}) anchored via Unk10=0x{mslkEntry.Unknown_0x10:X4} -> MSVI[{msviIndex}] -> MSVT[{msvtIndex}] -> World({worldPos.X:F3}, {worldPos.Y:F3}, {worldPos.Z:F3})"); // CONSOLE LOG
                                        }
                                        else 
                                        {
                                            debugWriter.WriteLine($"  [WARN] Node {i} MSVI index {msviIndex} yielded invalid MSVT index {msvtIndex} (>= {msvtVertexCount}). Cannot get anchor point.");
                                            Console.WriteLine($"  [WARN] Node {i} MSVI index {msviIndex} yielded invalid MSVT index {msvtIndex} (>= {msvtVertexCount}). Cannot get anchor point."); // CONSOLE LOG
                                        }
                                    }
                                    else 
                                    {
                                         debugWriter.WriteLine($"  [WARN] Node {i} Unk10 value 0x{mslkEntry.Unknown_0x10:X4} (={msviIndex}) is out of bounds for MSVI count {msviIndexCount}. Cannot get anchor point.");
                                         Console.WriteLine($"  [WARN] Node {i} Unk10 value 0x{mslkEntry.Unknown_0x10:X4} (={msviIndex}) is out of bounds for MSVI count {msviIndexCount}. Cannot get anchor point."); // CONSOLE LOG
                                    }
                                } catch (Exception ex) {
                                     debugWriter.WriteLine($"  [ERROR] Exception processing node {i} anchor point: {ex.Message}");
                                     Console.WriteLine($"  [ERROR] Exception processing node {i} anchor point: {ex.Message}"); // CONSOLE LOG
                                }
                            }
                            else
                            {
                                debugWriter.WriteLine($"  [WARN] Cannot process node {i} anchor point because MSVI or MSVT data is missing.");
                                Console.WriteLine($"  [WARN] Cannot process node {i} anchor point because MSVI or MSVT data is missing."); // CONSOLE LOG
                            }
                            // --- END: Process Node Anchor Point ---
                        } 
                        else if (mslkEntry.MspiIndexCount <= 0 || mslkEntry.MspiFirstIndex < 0) 
                        {
                            string reason = mslkEntry.MspiIndexCount <= 0 ? "Count<=0" : "FirstIndex<0";
                             debugWriter.WriteLine($"    Skipping Entry {i}: Reason={reason}.");
                             skippedMslkWriter.WriteLine($"Skipped ({reason}): {mslkBaseInfo}");
                             skippedMslkEntries++;
                             continue; // Skip to next entry
                        }

                        // Validate MSPI range
                        int rangeEndExclusive = mslkEntry.MspiFirstIndex + mslkEntry.MspiIndexCount;
                        if (rangeEndExclusive > mspiIndexCount) 
                        {
                             debugWriter.WriteLine($"    Skipping Entry {i}: Invalid MSPI Range [First:{mslkEntry.MspiFirstIndex}, Count:{mslkEntry.MspiIndexCount}] (Max MSPI Index: {mspiIndexCount - 1}).");
                             skippedMslkWriter.WriteLine($"Skipped (Invalid MSPI Range): {mslkBaseInfo}");
                             skippedMslkEntries++;
                             continue; // Skip to next entry
                        }
                        // --- End Skipped Entry Checks ---

                        // --- Process Valid Entry ---
                        List<int> validMspvIndices = new List<int>();
                        bool mspvIndexInvalid = false;
                        for (int j = 0; j < mslkEntry.MspiIndexCount; j++)
                        {
                            int currentMspiIndex = mslkEntry.MspiFirstIndex + j;
                            uint mspvIndex = pd4File.MSPI.Indices[currentMspiIndex]; // Get index into MSPV

                            // Validate MSPV index
                            if (mspvIndex < mspvVertexCount)
                            {
                                validMspvIndices.Add((int)mspvIndex + 1); // Store 1-based index for OBJ
                            }
                            else
                            {
                                debugWriter.WriteLine($"      WARNING: MSLK Entry {i}, MSPI index {currentMspiIndex} points to invalid MSPV index {mspvIndex} (Max: {mspvVertexCount - 1}). Skipping vertex.");
                                mspvIndexInvalid = true; // Flag that at least one index was bad
                            }
                        }

                        // Check if ANY valid vertices were found AFTER validation loop
                         if (validMspvIndices.Count == 0)
                         {
                             string reason = mspvIndexInvalid ? "All MSPV Indices Invalid" : "Processed but yielded 0 vertices";
                             debugWriter.WriteLine($"    Skipping Entry {i}: Reason={reason}. No valid MSPV indices found.");
                             skippedMslkWriter.WriteLine($"Skipped ({reason}): {mslkBaseInfo}");
                             skippedMslkEntries++;
                             continue; // Skip OBJ output for this entry
                         }

                        // Write OBJ geometry (only if valid vertices exist)
                        mslkWriter.WriteLine($"\ng MSLK_Entry_{i}"); // Start OBJ group
                        if (validMspvIndices.Count >= 2)
                        {
                             mslkWriter.WriteLine("l " + string.Join(" ", validMspvIndices)); // Write line
                             debugWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices.");
                             exportedMslkPaths++;
                        }
                        else // Exactly 1 valid vertex
                        {
                             mslkWriter.WriteLine($"p {validMspvIndices[0]}"); // Write point
                             debugWriter.WriteLine($"    Exported point for vertex {validMspvIndices[0]}.");
                             exportedMslkPaths++; // Count points as exported paths for simplicity here
                        }

                    } // End for loop iterating through MSLK entries

                    // ADDED: Log message immediately after the loop finishes
                    debugWriter.WriteLine($"  Finished MSLK entry loop.");
                    Console.WriteLine($"  Finished MSLK entry loop."); // CONSOLE LOG

                    summaryWriter.WriteLine($"MSLK Paths Exported: {exportedMslkPaths}");
                    summaryWriter.WriteLine($"MSLK Entries Skipped: {skippedMslkEntries}");

                }
                else
                {
                    debugWriter.WriteLine("Skipping MSLK processing (MSLK, MSPI, or MSPV data missing or empty).");
                    Console.WriteLine("Skipping MSLK processing (MSLK, MSPI, or MSPV data missing or empty)."); // CONSOLE LOG
                }
                // --- End MSLK Path Export ---

                // --- Log MSUR/MSVI Data ---
                debugWriter.WriteLine("\n--- Processing MSUR/MSVI Entries (Logging Only) ---");
                if (pd4File.MSUR?.Entries != null && pd4File.MSVI?.Indices != null && msvtVertexCount > 0)
                {
                    int msurEntryCount = pd4File.MSUR.Entries.Count;
                    debugWriter.WriteLine($"Processing {msurEntryCount} MSUR entries...");

                    for (int i = 0; i < msurEntryCount; i++)
                    {
                        var msurEntry = pd4File.MSUR.Entries[i];
                        debugWriter.WriteLine($"  MSUR Entry {i}: {msurEntry.ToString()}"); // Log full entry details
                        
                        int msviFirstIndex = (int)msurEntry.MsviFirstIndex; // Cast is safe if validation passes
                        int msviIndexCountLocal = msurEntry.IndexCount; // Corrected property name

                        // Validate MSVI range defined by MSUR entry
                        if (msviFirstIndex >= 0 && msviIndexCountLocal > 0)
                        {
                            int rangeEndExclusive = msviFirstIndex + msviIndexCountLocal;
                            if (rangeEndExclusive <= msviIndexCount)
                            {
                                // Range is valid within MSVI list
                                debugWriter.WriteLine($"    MSVI Range: [{msviFirstIndex}..{rangeEndExclusive - 1}] (Count: {msviIndexCountLocal}) - Valid Range.");
                                List<uint> msviIndicesFromRange = new List<uint>();
                                List<int> invalidMsvtIndices = new List<int>();

                                // Retrieve and validate indices within the range
                                for (int j = 0; j < msviIndexCountLocal; j++)
                                {
                                    int currentMsviListIndex = msviFirstIndex + j;
                                    uint msvtIndex = pd4File.MSVI.Indices[currentMsviListIndex]; // Get index into MSVT
                                    msviIndicesFromRange.Add(msvtIndex); // Store the index from MSVI

                                    // Validate the MSVT index itself
                                    if (msvtIndex >= msvtVertexCount)
                                    {
                                        invalidMsvtIndices.Add((int)msvtIndex);
                                    }
                                }
                                
                                // Log the retrieved MSVI indices (first few for brevity)
                                int maxLog = Math.Min(msviIndicesFromRange.Count, 10);
                                debugWriter.WriteLine($"      MSVI Indices Retrieved (first {maxLog}): [" + string.Join(", ", msviIndicesFromRange.Take(maxLog)) + (msviIndicesFromRange.Count > maxLog ? " ..." : "") + "]");
                                
                                // Log any invalid MSVT indices found within this MSUR entry's range
                                if (invalidMsvtIndices.Count > 0)
                                {
                                    debugWriter.WriteLine($"      WARNING: Found {invalidMsvtIndices.Count} indices in this range pointing outside MSVT bounds (Max: {msvtVertexCount - 1}). Invalid Indices: [" + string.Join(", ", invalidMsvtIndices.Distinct().Take(10)) + (invalidMsvtIndices.Distinct().Count() > 10 ? " ..." : "") + "]");
                                }
                            }
                            else
                            {
                                // MSVI range defined by MSUR is invalid
                                debugWriter.WriteLine($"    ERROR: MSUR Entry {i} defines invalid MSVI range [First:{msviFirstIndex}, Count:{msviIndexCountLocal}] (Max MSVI Index: {msviIndexCount - 1})");
                            }
                        }
                        else
                        {
                            debugWriter.WriteLine($"    INFO: MSUR Entry {i} has MsviIndexCount=0 or MsviFirstIndex invalid. Skipping index retrieval.");
                        }
                    }
                     debugWriter.WriteLine("Finished processing MSUR entries.");
                }
                else
                {
                    debugWriter.WriteLine("Skipping MSUR/MSVI processing (MSUR, MSVI, or MSVT data missing).");
                }
                debugWriter.WriteLine("--- End MSUR/MSVI Processing ---");
                // --- End MSUR/MSVI Logging ---

                debugWriter.WriteLine("--- Finished OBJ Export ---");
                summaryWriter.WriteLine("--- Finished OBJ Export ---");
                // --- End OBJ Export ---

                // Log presence of known PD4 chunks
                debugWriter.WriteLine("\n--- Checking for known PD4 chunks in loaded PD4 data ---");
                summaryWriter.WriteLine("\n--- Detected PD4 Chunks ---");
                Console.WriteLine("\n--- Checking for known PD4 chunks ---");

                // Use reflection to check all properties derived from IChunk
                PropertyInfo[] properties = typeof(PD4File).GetProperties()
                                            .Where(p => typeof(IIFFChunk).IsAssignableFrom(p.PropertyType))
                                            .OrderBy(p => p.Name) // Optional: Order alphabetically
                                            .ToArray();

                bool anyChunkFound = false;
                foreach (var prop in properties)
                {
                    var chunk = prop.GetValue(pd4File);
                    bool isPresent = chunk != null;
                    string status = isPresent ? "Present" : "MISSING";
                    string logLine = $"Chunk {prop.Name}: {status}";

                    debugWriter.WriteLine(logLine);
                    Console.WriteLine(logLine);
                    if (isPresent)
                    {
                        summaryWriter.WriteLine($"- {prop.Name}");
                        anyChunkFound = true;
                    }

                    // Optional: Add Assertions here later if specific chunks MUST exist in PD4
                    // if (prop.Name == "MVER" && !isPresent) { Assert.True(false, "MVER chunk is missing!"); }
                }
                debugWriter.WriteLine("--- Finished checking chunks ---");
                summaryWriter.WriteLine(anyChunkFound ? "--- End Detected Chunks ---" : "--- NO KNOWN PD4 CHUNKS DETECTED ---");
                Console.WriteLine("--- Finished checking chunks ---");

            }
            catch (Exception ex)
            {
                Console.WriteLine($">> ERROR processing {relativeTestDataPath}: {ex.Message}");
                // Console.WriteLine(ex.StackTrace); // Optionally uncomment for full stack
                debugWriter?.WriteLine($">> EXCEPTION during loading/processing: {ex}");
                summaryWriter?.WriteLine($">> EXCEPTION: {ex.Message}");
                // Flush streams after writing exception
                debugWriter?.Flush();
                summaryWriter?.Flush();
                 Console.WriteLine($">> Test will continue to next file if applicable, but this file failed.");
            }
            finally
            {
                // Close all writers
                debugWriter?.Close();
                summaryWriter?.Close();
                mspvWriter?.Close();
                msvtWriter?.Close();
                mscnWriter?.Close();
                mslkWriter?.Close();
                skippedMslkWriter?.Close(); // ADDED close skipped writer
                mslkNodesWriter?.Close(); // ADDED Dispose for nodes writer
                Console.WriteLine($"--- Finished processing {relativeTestDataPath} ---");
            }

            // Final Assertions (check if files were created)
            Assert.True(File.Exists(debugLogPath), $"Debug log file was not created at {debugLogPath}");
            Assert.True(File.Exists(summaryLogPath), $"Summary log file was not created at {summaryLogPath}");
            Assert.True(File.Exists(outputMspvFilePath), $"MSPV OBJ file was not created at {outputMspvFilePath}");
            Assert.True(File.Exists(outputMsvtFilePath), $"MSVT OBJ file was not created at {outputMsvtFilePath}");
            Assert.True(File.Exists(outputMscnFilePath), $"MSCN OBJ file was not created at {outputMscnFilePath}");
            Assert.True(File.Exists(outputMslkFilePath), $"MSLK OBJ file was not created at {outputMslkFilePath}");
            // Assert skipped log exists
             Assert.True(File.Exists(baseOutputPath + "_skipped_mslk.log"), $"Skipped MSLK log file was not created at {baseOutputPath}_skipped_mslk.log");
            Assert.True(File.Exists(outputMslkNodesFilePath), $"MSLK Nodes OBJ file should exist at {outputMslkNodesFilePath}"); // ADDED Assert for nodes OBJ

            Console.WriteLine($"--- Finished Processing: {relativeTestDataPath} ---");
        }

        // Add helper methods or specific chunk processing logic here later
    }
} 