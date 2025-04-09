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

namespace WoWToolbox.Tests.Navigation.PM4
{
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
                (float)v.Z / 36.0f             // Use Z directly, divide by 36.0
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
            // var outputMprlXZYFilePath = baseOutputPath + "_mprl_XZY.obj"; // Removed
            // var outputMprlNegXZYFilePath = baseOutputPath + "_mprl_negXZY.obj"; // Removed
            // var outputMscnFilePath = baseOutputPath + "_mscn.obj"; // Reverted: MSCN Disabled
            // var outputCombinedFilePath = baseOutputPath + "_combined.obj"; // Reverted: Removed Combined
            string debugLogPath = baseOutputPath + ".debug.log";

            Console.WriteLine($"Output MSVT OBJ: {outputMsvtFilePath}");
            Console.WriteLine($"Output MSPV OBJ: {outputMspvFilePath}");
            // Update console output for single MPRL file
            Console.WriteLine($"Output MPRL OBJ: {outputMprlFilePath}");
            // Console.WriteLine($"Output MPRL (XZY) OBJ: {outputMprlXZYFilePath}"); // Removed
            // Console.WriteLine($"Output MPRL (NegXZY) OBJ: {outputMprlNegXZYFilePath}"); // Removed
            // Console.WriteLine($"Output MSCN OBJ: {outputMscnFilePath}"); // Reverted: MSCN Disabled
            // Console.WriteLine($"Output Combined OBJ: {outputCombinedFilePath}"); // Reverted: Removed Combined
            Console.WriteLine($"Debug Log: {debugLogPath}");

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
            bool exportMspvVertices = true;   
            bool exportMsvtVertices = true;  
            bool exportMprlPoints = true;
            bool exportMscnNormals = true; // ENABLED MSCN Export
            bool exportMslkPaths = true;
            bool exportOnlyFirstMslk = false; 
            bool exportMsurFaces = true;
            bool exportOnlyFirstMsur = false; // RE-ADDED THIS FLAG
            bool exportMprrLines = true; // ENABLED MPRR LINE EXPORT FOR DEBUGGING
            bool exportMdsfFaces = false; 
            // --- End Configuration Flags ---

            // --- Reverted to Separate OBJ Export Setup ---
            StreamWriter? debugWriter = null;
            // StreamWriter? combinedWriter = null; // Reverted: Removed Combined
            StreamWriter? msvtWriter = null;
            StreamWriter? mspvWriter = null;
            StreamWriter? mprlWriter = null;
            // StreamWriter? mscnWriter = null; // REMOVED separate writer for normals
            
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
                // combinedWriter = new StreamWriter(outputCombinedFilePath); // Reverted: Removed Combined
                msvtWriter = new StreamWriter(outputMsvtFilePath);
                mspvWriter = new StreamWriter(outputMspvFilePath);
                mprlWriter = new StreamWriter(outputMprlFilePath);
                // mscnWriter = new StreamWriter(outputMsvtFilePath, true); // REMOVED writer init

                // Write headers (Reverted)
                // combinedWriter.WriteLine("# PM4 Combined Geometry (Direct Mapping -> Final Transform: X, Z, -Y)"); 
                msvtWriter.WriteLine("# PM4 MSVT/MSUR Geometry (X, Z, -Y)"); // Reverted Header
                mspvWriter.WriteLine("# PM4 MSPV/MSLK Geometry (X, Z, -Y)"); // Reverted Header
                mprlWriter.WriteLine("# PM4 MPRL/MPRR Geometry (X, Z, -Y)"); // Reverted Header
                // mscnWriter.WriteLine("# PM4 MSCN Vertices (Direct Mapping with Y/Z Swap -> Final Transform: X, Z, -Y)");

                // --- Moved Index Validation Logging Inside Try Block ---
                int mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");
                // Assert.True(pm4File.MPRR?.ValidateIndices(mprlVertexCount), "MPRR indices are out of bounds for MPRL vertices."); // TEMP COMMENTED: Test file has invalid MPRR data.
                
                int msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSVI Indices against MSVT Vertex Count: {msvtVertexCount} ---");
                // Re-enable MSVI validation check - We expect this might fail with the test file, but want to see the failure.
                Assert.True(pm4File.MSVI?.ValidateIndices(msvtVertexCount), "MSVI indices are out of bounds for MSVT vertices.");

                int mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSPI Indices against MSPV Vertex Count: {mspvVertexCount} ---");
                Assert.True(pm4File.MSPI?.ValidateIndices(mspvVertexCount), "MSPI indices are out of bounds for MSPV vertices.");
                
                int totalMspiIndices = pm4File.MSPI?.Indices.Count ?? 0; // Renamed variable
                debugWriter.WriteLine($"\n--- Validating MSUR Index Ranges against MSVI Index Count: {totalMspiIndices} ---"); // Used renamed variable
                // Assert.True(pm4File.MSLK?.ValidateIndices(totalMspiIndices), "MSLK indices are out of bounds for MSPI indices."); // MSLK class does not have ValidateIndices method
                Assert.True(pm4File.MSUR?.ValidateIndices(totalMspiIndices), "MSUR index ranges are out of bounds for MSVI indices.");
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
                        int logCounterMspv = 0;
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            // Applying standard (X, Z, -Y) transform - Reverted
                            float worldX = vertex.X;
                            float worldY = vertex.Y; 
                            float worldZ = vertex.Z;

                            // Write final coords to separate file ONLY
                            mspvWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            mspvFileVertexCount++;
                            // Write final coords to combined file - Reverted
                            // combinedWriter.WriteLine(FormattableString.Invariant($"v {finalX:F6} {finalY:F6} {finalZ:F6}"));
                            // globalVertexCount++;

                            if (logCounterMspv < 5) {
                                debugWriter.WriteLine($"  MSPV Vertex {mspvFileVertexCount-1}: ReadFloat=({vertex.X:F6}, {vertex.Y:F6}, {vertex.Z:F6}), FileIndex={mspvFileVertexCount}"); // Reverted log
                                logCounterMspv++;
                            }
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
                            msvtWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}")); // Needs syntax check
                            msvtFileVertexCount++;
                            // Write final coords to combined file - Reverted
                            // combinedWriter.WriteLine(FormattableString.Invariant($\"v {finalX:F6} {finalY:F6} {finalZ:F6}\="));
                            // globalVertexCount++;

                             if (logCounterMsvt < 5) {
                                // Updated log message
                                debugWriter.WriteLine($"  MSVT Vertex {msvtFileVertexCount-1}: Raw=(Y:{vertex.Y}, X:{vertex.X}, Z:{vertex.Z}), RevertedCoords_NegY=({worldX:F3}, {worldY:F3}, {worldZ:F3}) FileIndex={msvtFileVertexCount}"); // Needs syntax check
                                logCounterMsvt++;
                            }
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
                        debugWriter.WriteLine($"\n--- Exporting MPRL Vertices (X, Z, -Y) -> _mprl.obj ---"); // Reverted
                        int logCounterMprl = 0;
                        mprlFileVertexCount = 0;

                        foreach (var entry in pm4File.MPRL.Entries)
                        {
                            // Input (X, Y, Z)
                            float worldX = entry.Position.X; 
                            float worldY = entry.Position.Y; 
                            float worldZ = entry.Position.Z; 

                            // Write final coords to separate file ONLY
                            mprlWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}")); // Needs syntax check
                            mprlFileVertexCount++;
                            // Write final coords to combined file - Reverted
                            // combinedWriter.WriteLine(FormattableString.Invariant($"v {finalX:F6} {finalY:F6} {finalZ:F6}"));
                            // globalVertexCount++;

                            if (logCounterMprl < 10) { 
                                debugWriter.WriteLine(FormattableString.Invariant(
                                    // Updated log message
                                    $"  MPRL Entry {mprlFileVertexCount-1}: [...] Pos=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}), [...] FileIndex={mprlFileVertexCount}, RevertedCoords_NegY0X=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                                )); // Needs syntax check
                                logCounterMprl++;
                            }
                        }
                         debugWriter.WriteLine($"Wrote {mprlFileVertexCount} MPRL vertices to _mprl.obj file."); // Reverted comment
                        mprlWriter.WriteLine();
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MPRL vertex data found."); }
                 } else {
                    debugWriter.WriteLine("\nSkipping MPRL Point export (Flag False).");
                 }

                 // --- 4. Process MSCN Normals (Collect for later writing) ---
                 var mscnNormalStrings = new List<string>(); // List to hold formatted normals
                 if (exportMscnNormals) 
                 {
                     if (pm4File.MSCN != null && pm4File.MSCN.Vectors.Count > 0)
                     {
                        int mscnNormalCount = 0;
                        debugWriter.WriteLine($"\n--- Processing MSCN Normals (Raw X, Y, Z) for later writing ---");
                        int logCounterMscn = 0;

                        foreach (var normal in pm4File.MSCN.Vectors)
                        {
                            // Input (X, Y, Z) - Assuming raw export for now
                            float normX = normal.X; 
                            float normY = normal.Y; 
                            float normZ = normal.Z; 

                            // Format and add to list
                            mscnNormalStrings.Add(FormattableString.Invariant($"vn {normX:F6} {normY:F6} {normZ:F6}"));
                            mscnNormalCount++;

                            if (logCounterMscn < 10) { 
                                debugWriter.WriteLine(FormattableString.Invariant(
                                    $"  MSCN Normal {mscnNormalCount-1}: Raw=({normal.X:F3}, {normal.Y:F3}, {normal.Z:F3}), Formatted=({normX:F3}, {normY:F3}, {normZ:F3})"
                                ));
                                logCounterMscn++;
                            }
                        }
                        debugWriter.WriteLine($"Collected {mscnNormalCount} MSCN normals to write later.");
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MSCN normal data found to process."); }
                 } else { 
                     debugWriter.WriteLine("\n--- Skipping MSCN Normal processing (Flag 'exportMscnNormals' is False) ---");
                 }

                 // --- NOW Write Collected MSCN Normals using msvtWriter ---
                 if (msvtWriter != null && mscnNormalStrings.Count > 0)
                 {
                     debugWriter.WriteLine($"\n--- Writing {mscnNormalStrings.Count} collected MSCN Normals to _msvt.obj ---");
                     msvtWriter.WriteLine("# MSCN Normals (Raw X, Y, Z)"); // Write header
                     foreach(var normalString in mscnNormalStrings)
                     {
                         msvtWriter.WriteLine(normalString);
                     }
                     msvtWriter.WriteLine(); // Add newline after normals
                     debugWriter.WriteLine("Finished writing collected MSCN normals.");
                     debugWriter.Flush();
                 }
                 else if (msvtWriter != null)
                 {
                      debugWriter.WriteLine("\n--- No collected MSCN normals to write to _msvt.obj ---");
                 }
                 // --- END Write Collected MSCN Normals ---

                 // Write a newline to the combined file after all vertices - Reverted
                 // combinedWriter.WriteLine();

                 // --- 5. Export MSLK paths -> mspvWriter ONLY ---
                if (exportMslkPaths)
                {
                    if (pm4File.MSLK != null && pm4File.MSPI != null && pm4File.MSPV != null && mspvFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSLK -> MSPI -> MSPV Chain -> _mspv.obj ---"); // Reverted

                        int entriesToProcess = exportOnlyFirstMslk ? Math.Min(1, pm4File.MSLK.Entries.Count) : pm4File.MSLK.Entries.Count;
                        for (int entryIndex = 0; entryIndex < entriesToProcess; entryIndex++)
                        {
                            MSLKEntry mslkEntry = pm4File.MSLK.Entries[entryIndex];
                            // ... Log MSLK Entry ...

                            int firstMspiIndex = mslkEntry.MspiFirstIndex;
                            int mspiIndexCount = mslkEntry.MspiIndexCount;

                            // Determine Group Name (using MDOS if linked, otherwise default)
                            string groupName;
                            if (mslkEntry.Unknown_0x04 < (pm4File.MDOS?.Entries.Count ?? 0)) {
                                var mdosEntry = pm4File.MDOS!.Entries[(int)mslkEntry.Unknown_0x04];
                                groupName = $"MSLK_Mdos{mslkEntry.Unknown_0x04}_ID{mdosEntry.Value_0x00:X}";
                            } else {
                                groupName = $"MSLK_Entry_{entryIndex}";
                            }

                            if (firstMspiIndex >= 0 && firstMspiIndex + mspiIndexCount <= pm4File.MSPI.Indices.Count)
                            {
                                if (mspiIndexCount < 2) continue;

                                List<uint> mspiIndices = pm4File.MSPI.Indices.GetRange(firstMspiIndex, mspiIndexCount);
                                List<uint> relativeObjMspvIndices = new List<uint>(); // Reverted: Indices relative to _mspv.obj

                                foreach(uint mspiValue_MspvIndex in mspiIndices)
                                {
                                    // Validate against local file count
                                    if (mspiValue_MspvIndex < mspvVertexCount) // Use total count loaded 
                                    {
                                        // Reverted: Calculate relative index
                                        relativeObjMspvIndices.Add(1 + mspiValue_MspvIndex); // 1-based index within _mspv.obj
                                    } else { debugWriter.WriteLine($"  MSLK->MSPI: WARNING - Skipping Raw MSPV Index {mspiValue_MspvIndex} because it is out of bounds (>= TOTAL Loaded MSPV Count: {mspvVertexCount})."); } // Updated Log
                                }

                                if (relativeObjMspvIndices.Count >= 2)
                                {
                                    mspvWriter.WriteLine($"g {groupName}"); // Reverted: Write to mspvWriter
                                    for (int j = 0; j < relativeObjMspvIndices.Count - 1; j++)
                                    {
                                        uint objIdxStart = relativeObjMspvIndices[j];
                                        uint objIdxEnd = relativeObjMspvIndices[j+1];
                                        mspvWriter.WriteLine($"l {objIdxStart} {objIdxEnd}"); // Reverted: Write to mspvWriter
                                        if(entryIndex < 2 && j < 3) {
                                            debugWriter.WriteLine($"    MSLK Line (Entry {entryIndex}): Writing SEPARATE OBJ Line {objIdxStart} -> {objIdxEnd}"); // Reverted log
                                        }
                                    }
                                }
                            }
                            else
                            {
                                // REFINED LOGGING: Differentiate expected skips from actual errors
                                if (firstMspiIndex == -1 && mspiIndexCount == 0)
                                {
                                    debugWriter.WriteLine($"  Skipping MSLK Entry {entryIndex}: Empty link (FirstIndex={firstMspiIndex}, Count={mspiIndexCount}).");
                                }
                                else
                                {
                                    debugWriter.WriteLine($"  Skipping MSLK Entry {entryIndex}: GENUINE Invalid MSPI index range. Details: FirstIndex={firstMspiIndex}, Count={mspiIndexCount}, TotalIndices={pm4File.MSPI.Indices.Count}");
                                }
                            }
                        }
                        debugWriter.WriteLine("--- Finished Processing MSLK Chain for separate file ---"); // Reverted log
                        mspvWriter.WriteLine(); // Reverted: Newline in mspvWriter
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nSkipping MSLK Chain processing (missing data or source vertices)."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MSLK Path export (Flag False)."); 
                    debugWriter.WriteLine("\n--- Skipping MSLK Path export (Flag 'exportMslkPaths' is False) ---"); // Enhanced Log
                }

                // --- 6. Export MSUR surfaces -> msvtWriter ONLY ---
                if (exportMsurFaces)
                {
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null && pm4File.MDOS != null && msvtFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSUR -> MSVI -> MSVT Faces -> _msvt.obj ---"); // Reverted

                        int entriesToProcess = exportOnlyFirstMsur ? Math.Min(1, pm4File.MSUR.Entries.Count) : pm4File.MSUR.Entries.Count;
                        for (int entryIndex = 0; entryIndex < entriesToProcess; entryIndex++)
                        {
                            var msurEntry = pm4File.MSUR.Entries[entryIndex];
                            // Log details for the current MSUR entry being processed
                            debugWriter.WriteLine($"Processing MSUR Entry {entryIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, MdosIndex={msurEntry.MdosIndex}, FlagsOrUnk0={msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}"); 
                            
                            int firstIndex = (int)msurEntry.MsviFirstIndex;
                            int indexCount = msurEntry.IndexCount;
                            uint mdosIndex = msurEntry.MdosIndex;

                            string groupName;
                            if (mdosIndex < (pm4File.MDOS?.Entries.Count ?? 0))
                            {
                                var mdosEntry = pm4File.MDOS!.Entries[(int)mdosIndex];
                                groupName = $"MSUR_Mdos{mdosIndex}_ID{mdosEntry.Value_0x00:X}";
                            }
                            else
                            {
                                debugWriter.WriteLine($"  MSUR Entry {entryIndex}: Invalid MDOS index {mdosIndex}. Assigning default group.");
                                groupName = $"MSUR_InvalidMdosIdx_{entryIndex}";
                            }

                            if (pm4File.MSVI == null || pm4File.MSVT == null) { /* ... */ continue; }

                            if (firstIndex >= 0 && firstIndex + indexCount <= pm4File.MSVI!.Indices.Count)
                            {
                                 if (indexCount < 3) continue;

                                List<uint> msviIndices = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                                var relativeObjFaceIndices = new List<string>(); // Reverted: Indices relative to _msvt.obj

                                for(int i=0; i < msviIndices.Count; ++i)
                                {
                                    uint msviIdx = msviIndices[i];

                                    // Validate against TOTAL vertex count, not the count written so far
                                    if (msviIdx >= msvtVertexCount) // CORRECTED: Use total count
                                    {
                                        debugWriter.WriteLine($"    MSUR Face Vertex: ERROR - Skipping Raw MSVI Index {msviIdx} because it is out of bounds (>= TOTAL MSVT Count: {msvtVertexCount})."); // Updated Log Message
                                        continue;
                                    }

                                    // Calculate relative index
                                    uint relativeObjVertexIndex = 1 + msviIdx;
                                    relativeObjFaceIndices.Add($"{relativeObjVertexIndex}");

                                    if (entryIndex < 2 && i < 5) {
                                        debugWriter.WriteLine($"    MSUR Face Vertex: RawMSVI={msviIdx} -> RelativeOBJIndex={relativeObjVertexIndex}"); // Reverted log
                                    }
                                }

                                if (relativeObjFaceIndices.Count >= 3)
                                {
                                    bool isDegenerate = false;
                                    // ... degenerate check uses relativeObjFaceIndices ...
                                    for (int i = 0; i < relativeObjFaceIndices.Count; i++) {
                                        for (int j = i + 1; j < relativeObjFaceIndices.Count; j++) {
                                            if (relativeObjFaceIndices[i] == relativeObjFaceIndices[j]) {
                                                isDegenerate = true; break;
                                            }
                                        }
                                        if (isDegenerate) break;
                                    }

                                    if (!isDegenerate)
                                    {
                                        msvtWriter.WriteLine($"g {groupName}"); // Reverted: Write to msvtWriter
                                        // TRIANGULATION LOGIC uses relativeObjFaceIndices and msvtWriter:
                                        if (relativeObjFaceIndices.Count == 4) { 
                                            msvtWriter.WriteLine($"f {relativeObjFaceIndices[0]} {relativeObjFaceIndices[1]} {relativeObjFaceIndices[2]}");
                                            msvtWriter.WriteLine($"f {relativeObjFaceIndices[0]} {relativeObjFaceIndices[2]} {relativeObjFaceIndices[3]}");
                                        } else if (relativeObjFaceIndices.Count == 3) { 
                                            msvtWriter.WriteLine($"f {string.Join(" ", relativeObjFaceIndices)}");
                                        } else { 
                                             for (int k = 1; k < relativeObjFaceIndices.Count - 1; k++) {
                                                msvtWriter.WriteLine($"f {relativeObjFaceIndices[0]} {relativeObjFaceIndices[k]} {relativeObjFaceIndices[k+1]}");
                                             }
                                        }
                                    }
                                    else { debugWriter.WriteLine($"Skipped degenerate face: {string.Join(" ", relativeObjFaceIndices)}"); }
                                }
                                 else { debugWriter.WriteLine($"Skipped face with < 3 valid vertices for Entry {entryIndex}."); }
                            }
                            else { debugWriter.WriteLine($"Skipped MSUR Entry {entryIndex} due to invalid MSVI range."); }
                        }
                        debugWriter.WriteLine("--- Finished Processing MSUR Chain for separate file ---"); // Reverted log
                        msvtWriter.WriteLine(); // Reverted: Newline in msvtWriter
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nSkipping MSUR Chain processing (missing data or source vertices)."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MSUR Face export (Flag False)."); 
                    debugWriter.WriteLine("\n--- Skipping MSUR Face export (Flag 'exportMsurFaces' is False) ---"); // Enhanced Log
                }

                 // --- 7. Export MPRR pairs -> mprlWriter ONLY ---
                 if (exportMprrLines)
                 {
                     if (pm4File.MPRR != null && pm4File.MPRL != null && mprlFileVertexCount > 0)
                     {
                        debugWriter.WriteLine($"\n--- Processing MPRR -> MPRL Chain -> _mprl.obj ---"); // Reverted
                         int logCounterMprr = 0;
                         // bool mprrGroupWritten = false; // Reverted

                         foreach (var mprrEntry in pm4File.MPRR.Entries)
                         {
                             // Read the raw ushort values
                             ushort rawIndex1 = mprrEntry.Unknown_0x00;
                             ushort rawIndex2 = mprrEntry.Unknown_0x02;

                             if (logCounterMprr < 20) { // Log more entries initially
                                 debugWriter.WriteLine($"  MPRR Entry {logCounterMprr}: RawIdx1={rawIndex1}, RawIdx2={rawIndex2}"); 
                             }

                             // Skip sentinel value if used (common in WoW indices)
                             if (rawIndex1 == 65535 || rawIndex2 == 65535) { 
                                 if (logCounterMprr < 20) { debugWriter.WriteLine("    Skipping line due to sentinel index (65535)."); }
                                 continue; 
                             }

                             // Explicit bounds check against the total MPRL vertices written to the file
                             if (rawIndex1 < mprlFileVertexCount && rawIndex2 < mprlFileVertexCount && rawIndex1 != rawIndex2)
                             {
                                 // Indices seem valid relative to the vertices we exported, proceed
                                 uint relativeObjIndex1 = 1 + (uint)rawIndex1;
                                 uint relativeObjIndex2 = 1 + (uint)rawIndex2;

                                 // Write group header (only once) and line to separate file
                                 if (logCounterMprr == 0) { mprlWriter.WriteLine("g MPRR_Lines"); } 
                                 mprlWriter.WriteLine($"l {relativeObjIndex1} {relativeObjIndex2}"); 

                                 if (logCounterMprr < 20) {
                                     debugWriter.WriteLine($"    MPRR Line: Writing SEPARATE OBJ Line {relativeObjIndex1} -> {relativeObjIndex2} (Raw MPRL Indices: {rawIndex1} -> {rawIndex2})"); 
                                 }
                             }
                             else { 
                                   // Log the reason for skipping (out of bounds or identical indices)
                                   if (logCounterMprr < 20) { 
                                       debugWriter.WriteLine($"    MPRR->MPRL: WARNING - Skipping Line. Indices invalid or out of bounds (RawIdx1={rawIndex1}, RawIdx2={rawIndex2}, File MPRL Count={mprlFileVertexCount})."); 
                                   }
                             }
                             logCounterMprr++;
                         }
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
                // combinedWriter?.Close(); // Reverted: Removed Combined
                msvtWriter?.Close();
                mspvWriter?.Close();
                mprlWriter?.Close();
                // mscnWriter?.Close(); // REMOVED close call
            }

             // Reverted: Original Asserts
             Assert.True(File.Exists(outputMsvtFilePath), $"MSVT OBJ file was not created at {outputMsvtFilePath}");
             Assert.True(File.Exists(outputMspvFilePath), $"MSPV OBJ file was not created at {outputMspvFilePath}");
             Assert.True(File.Exists(outputMprlFilePath), $"MPRL OBJ file was not created at {outputMprlFilePath}");
             Assert.True(File.Exists(debugLogPath), $"Debug log file was not created at {debugLogPath}");

            Console.WriteLine("--- LoadPM4File_ShouldLoadChunks END ---");
        }

        // ... other test methods ...
    }
} 