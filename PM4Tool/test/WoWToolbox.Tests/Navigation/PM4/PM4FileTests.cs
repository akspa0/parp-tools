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
            Console.WriteLine("--- LoadPM4File_ShouldLoadChunks START ---");

            // --- Path Construction (Relative to Test Output Directory) ---
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var testDataSubDir = Path.Combine("test_data", "development"); // Relative path within output dir
            var inputFileName = "development_00_00.pm4";
            var inputFilePath = Path.Combine(baseDir, testDataSubDir, inputFileName);

            var outputSubDir = Path.Combine("output", "development"); // Relative path for output files
            var outputDir = Path.Combine(baseDir, outputSubDir);
            Directory.CreateDirectory(outputDir); // Ensure the output directory exists

            var baseOutputName = "development_00_00";
            var baseOutputPath = Path.Combine(outputDir, baseOutputName);
            // --- End Path Construction ---

            // --- Define Output File Paths (Using new baseOutputPath) ---
            var outputMsvtFilePath = baseOutputPath + "_msvt.obj";
            var outputMspvFilePath = baseOutputPath + "_mspv.obj";
            var outputMprlFilePath = baseOutputPath + "_mprl.obj";
            var outputMscnFilePath = baseOutputPath + "_mscn.obj";
            var outputMslkFilePath = baseOutputPath + "_mslk.obj";
            var outputMslkJsonPath = baseOutputPath + "_mslk_hierarchy.json";
            var outputSkippedMslkLogPath = baseOutputPath + "_skipped_mslk.log";
            var outputPm4MslkNodesFilePath = baseOutputPath + "_pm4_mslk_nodes.obj";
            string debugLogPath = baseOutputPath + ".debug.log";
            string summaryLogPath = baseOutputPath + ".summary.log";
            var outputBuildingIdsPath = baseOutputPath + "_building_ids.log";

            Console.WriteLine($"Base Directory: {baseDir}");
            Console.WriteLine($"Input File Path: {inputFilePath}");
            Console.WriteLine($"Output Directory: {outputDir}");
            Console.WriteLine($"Output MSVT OBJ: {outputMsvtFilePath}");
            Console.WriteLine($"Output MSPV OBJ: {outputMspvFilePath}");
            Console.WriteLine($"Output MPRL OBJ: {outputMprlFilePath}");
            Console.WriteLine($"Output MSCN OBJ: {outputMscnFilePath}");
            Console.WriteLine($"Output MSLK OBJ: {outputMslkFilePath}");
            Console.WriteLine($"Output MSLK JSON: {outputMslkJsonPath}");
            Console.WriteLine($"Output Skipped MSLK Log: {outputSkippedMslkLogPath}");
            Console.WriteLine($"Output PM4 MSLK Nodes OBJ: {outputPm4MslkNodesFilePath}");
            Console.WriteLine($"Debug Log: {debugLogPath}");
            Console.WriteLine($"Summary Log: {summaryLogPath}");
            Console.WriteLine($"Building IDs Log: {outputBuildingIdsPath}");

            // Initialize collection for unique building IDs
            var uniqueBuildingIds = new HashSet<uint>();

            // Act
            var pm4File = PM4File.FromFile(inputFilePath);

            // Assert - Basic Chunk Loading
            Assert.NotNull(pm4File);

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
            int mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
            Console.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");

            int msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
            Console.WriteLine($"\n--- Validating MSVI Indices against MSVT Vertex Count: {msvtVertexCount} ---");

            int mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
            Console.WriteLine($"\n--- Validating MSPI Indices against MSPV Vertex Count: {mspvVertexCount} ---");

            int totalMspiIndices = pm4File.MSPI?.Indices.Count ?? 0;
            Console.WriteLine($"\n--- Validating MSUR Index Ranges against MSVI Index Count: {totalMspiIndices} ---");

            // Assert - Export Configuration Flags
            bool exportMspvVertices = true;
            bool exportMsvtVertices = true;
            bool exportMprlPoints = true;
            bool exportMscnNormals = true;
            bool exportMslkPaths = true;
            bool exportOnlyFirstMslk = false;
            bool processMsurEntries = true;
            bool exportOnlyFirstMsur = false;
            bool exportMprrLines = true;
            bool logMdsfLinks = true;

            // --- Reverted to Separate OBJ Export Setup ---
            StreamWriter? debugWriter = null;
            StreamWriter? summaryWriter = null;
            StreamWriter? msvtWriter = null;
            StreamWriter? mspvWriter = null;
            StreamWriter? mprlWriter = null;
            StreamWriter? mscnWriter = null;
            StreamWriter? mslkWriter = null;
            StreamWriter? skippedMslkWriter = null;
            StreamWriter? mslkNodesWriter = null;
            
            try
            {
                // Initialize writers here, now that we are inside the try block
                debugWriter = new StreamWriter(debugLogPath, false);
                summaryWriter = new StreamWriter(summaryLogPath, false);
                
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
                debugWriter.WriteLine($"INFO: MSLK Entries: {pm4File.MSLK?.Entries.Count ?? -1}");

                // combinedWriter = new StreamWriter(outputCombinedFilePath);
                msvtWriter = new StreamWriter(outputMsvtFilePath);
                mspvWriter = new StreamWriter(outputMspvFilePath);
                mprlWriter = new StreamWriter(outputMprlFilePath);
                mscnWriter = new StreamWriter(outputMscnFilePath);
                mslkWriter = new StreamWriter(outputMslkFilePath);
                skippedMslkWriter = new StreamWriter(outputSkippedMslkLogPath);
                mslkNodesWriter = new StreamWriter(outputPm4MslkNodesFilePath, false);
                mslkNodesWriter.WriteLine($"# PM4 MSLK Node Anchor Points (from Unk10 -> MSVI -> MSVT)");
                mslkNodesWriter.WriteLine($"# Transform: Y, X, Z");

                // Write headers
                msvtWriter.WriteLine("# PM4 MSVT/MSUR Geometry (Y, X, Z)");
                mspvWriter.WriteLine("# PM4 MSPV/MSLK Geometry (X, Y, Z)");
                mprlWriter.WriteLine("# PM4 MPRL/MPRR Geometry (Y, Z, X)");
                mscnWriter.WriteLine("# MSCN Data as Vertices (Raw X, Y, Z)");
                mslkWriter.WriteLine("# PM4 MSLK Geometry (Points 'p' and Lines 'l') (Exported: {DateTime.Now})");
                skippedMslkWriter.WriteLine("# PM4 Skipped/Invalid MSLK Entries Log (Generated: {DateTime.Now})");

                // --- Moved Index Validation Logging Inside Try Block ---
                mprlVertexCount = pm4File.MPRL?.Entries.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MPRR Indices against MPRL Vertex Count: {mprlVertexCount} ---");
                
                msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSVI Indices against MSVT Vertex Count: {msvtVertexCount} ---");
                
                mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSPI Indices against MSPV Vertex Count: {mspvVertexCount} ---");
                
                totalMspiIndices = pm4File.MSPI?.Indices.Count ?? 0;
                debugWriter.WriteLine($"\n--- Validating MSUR Index Ranges against MSVI Index Count: {totalMspiIndices} ---");

                debugWriter.WriteLine("--- Exporting Split Geometry (Standard Transform) ---");

                // Vertex counters specific to each separate file
                int mspvFileVertexCount = 0;
                int msvtFileVertexCount = 0;
                int mprlFileVertexCount = 0;

                // --- 1. Export MSPV vertices -> mspvWriter ONLY ---
                if (exportMspvVertices)
                {
                    mspvWriter.WriteLine("# MSPV Vertices");
                    if (pm4File.MSPV != null)
                    {
                        debugWriter.WriteLine("\n--- Exporting MSPV Vertices (X, Z, -Y) -> _mspv.obj ---");
                        summaryWriter.WriteLine("\n--- MSPV Vertices (First 10) ---");
                        int logCounterMspv = 0;
                        foreach (var vertex in pm4File.MSPV.Vertices)
                        {
                            float worldX = vertex.X;
                            float worldY = vertex.Y; 
                            float worldZ = vertex.Z;

                            debugWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); 
                            if (logCounterMspv < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSPV Vertex {mspvFileVertexCount}: Raw=({vertex.X:F3}, {vertex.Y:F3}, {vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                            }
                            mspvWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            mspvFileVertexCount++;
                            logCounterMspv++; 
                        }
                        if (pm4File.MSPV.Vertices.Count > 10) { 
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                        }
                        debugWriter.WriteLine($"Wrote {mspvFileVertexCount} MSPV vertices to _mspv.obj.");
                        mspvWriter.WriteLine();
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("No MSPV vertex data found."); }
                } else {
                    debugWriter.WriteLine("\nSkipping MSPV Vertex export (Flag False).");
                }

                // --- 2. Export MSVT vertices -> msvtWriter ONLY ---
                if (exportMsvtVertices)
                {
                    msvtWriter.WriteLine("# MSVT Vertices");
                    if (pm4File.MSVT != null && pm4File.MSVT.Vertices.Count > 0)
                    {
                        debugWriter.WriteLine($"\n--- Exporting MSVT Vertices (Using PD4.md Formula) -> _msvt.obj ---");
                        summaryWriter.WriteLine("\n--- MSVT Vertices (First 10) ---");
                        int logCounterMsvt = 0;
                        msvtFileVertexCount = 0;
                        foreach (var vertex in pm4File.MSVT.Vertices)
                        {
                            msvtFileVertexCount++;
                            float worldX = vertex.Y;         // Mirror X by negating input Y
                            float worldY = vertex.X;  // Z (Up, Scaled) -> Y (Up)
                            float worldZ = vertex.Z;          // X -> Z (Depth)
                            
                            debugWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})")); 
                            if (logCounterMsvt < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant($"  MSVT Vertex {msvtFileVertexCount}: Raw=(Y:{vertex.Y:F3}, X:{vertex.X:F3}, Z:{vertex.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"));
                            }
                            msvtWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            logCounterMsvt++; 
                        }
                        if (pm4File.MSVT.Vertices.Count > 10) { 
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
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
                 if (exportMprlPoints)
                 {
                     mprlWriter.WriteLine("# MPRL Vertices (X, Z, -Y)");

                     if (pm4File.MPRL != null && pm4File.MPRL.Entries.Count > 0)
                     {
                        debugWriter.WriteLine($"\n--- Exporting MPRL Vertices (X, Z, Y) -> _mprl.obj ---");
                        summaryWriter.WriteLine("\n--- MPRL Vertices (First 10) ---");
                        int logCounterMprl = 0;
                        mprlFileVertexCount = 0;

                        foreach (var entry in pm4File.MPRL.Entries)
                        {
                            float worldX = entry.Position.Y; 
                            float worldY = entry.Position.Z; 
                            float worldZ = entry.Position.X; 

                            debugWriter.WriteLine(FormattableString.Invariant(
                                $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                            )); 
                            if (logCounterMprl < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MPRL Vertex {mprlFileVertexCount}: Raw=({entry.Position.X:F3}, {entry.Position.Y:F3}, {entry.Position.Z:F3}) -> Exported=({worldX:F3}, {worldY:F3}, {worldZ:F3})"
                                ));
                            }
                            mprlWriter.WriteLine(FormattableString.Invariant($"v {worldX:F6} {worldY:F6} {worldZ:F6}"));
                            mprlFileVertexCount++;
                            logCounterMprl++; 
                        }
                        if (pm4File.MPRL.Entries.Count > 10) { 
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                        }
                         debugWriter.WriteLine($"Wrote {mprlFileVertexCount} MPRL vertices to _mprl.obj file.");
                        mprlWriter.WriteLine();
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MPRL vertex data found."); }
                 } else {
                    debugWriter.WriteLine("\nSkipping MPRL Point export (Flag False).");
                 }

                 // --- 4. Process MSCN Data (Output as Vertices to separate file) ---
                 if (exportMscnNormals) 
                 {
                     if (pm4File.MSCN != null && pm4File.MSCN.Vectors.Count > 0 && mscnWriter != null)
                     {
                        int mscnVertexCount = 0;
                        debugWriter.WriteLine($"\n--- Exporting MSCN Data as Vertices (Raw X, Y, Z) -> _mscn.obj ---");
                        summaryWriter.WriteLine("\n--- MSCN Vertices (First 10) ---");
                        int logCounterMscn = 0;

                        foreach (var vectorData in pm4File.MSCN.Vectors)
                        {
                            float vX = vectorData.X; 
                            float vY = vectorData.Y; 
                            float vZ = vectorData.Z; 

                            var vertexString = FormattableString.Invariant($"v {vX:F6} {vY:F6} {vZ:F6}");
                            mscnWriter.WriteLine(vertexString);
                            mscnVertexCount++;

                            debugWriter.WriteLine(FormattableString.Invariant(
                                $"  MSCN Vertex {mscnVertexCount-1}: Raw=({vectorData.X:F3}, {vectorData.Y:F3}, {vectorData.Z:F3}) -> Exported=({vX:F3}, {vY:F3}, {vZ:F3})"
                            ));
                            if (logCounterMscn < 10) { 
                                summaryWriter.WriteLine(FormattableString.Invariant(
                                    $"  MSCN Vertex {mscnVertexCount-1}: Raw=({vectorData.X:F3}, {vectorData.Y:F3}, {vectorData.Z:F3}) -> Exported=({vX:F3}, {vY:F3}, {vZ:F3})"
                                ));
                            }
                            logCounterMscn++; 
                        }
                        if (pm4File.MSCN.Vectors.Count > 10) { 
                            summaryWriter.WriteLine("  ... (Summary log limited to first 10) ...");
                        }
                        debugWriter.WriteLine($"Wrote {mscnVertexCount} MSCN vertices to _mscn.obj file.");
                        mscnWriter.WriteLine();
                        debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("No MSCN data found or mscnWriter not available."); }
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
                        int processedMslkNodes = 0;

                        var msviIndices = pm4File.MSVI?.Indices;
                        var msvtVertices = pm4File.MSVT?.Vertices;
                        int msviCount = msviIndices?.Count ?? 0;
                        int msvtCount = msvtVertices?.Count ?? 0;
                        debugWriter.WriteLine($"INFO: Cached MSVI ({msviCount}) and MSVT ({msvtCount}) for node processing.");

                        for (int entryIndex = 0; entryIndex < entriesToProcessMslk; entryIndex++)
                        {
                            var mslkEntry = pm4File.MSLK.Entries[entryIndex];
                            if (entryIndex < 10) {
                                summaryWriter.WriteLine($"  Processing MSLK Entry {entryIndex}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}");
                            }
                            debugWriter.WriteLine(string.Format(CultureInfo.InvariantCulture,
                                 $"Processing MSLK Entry {entryIndex}: FirstIndex={mslkEntry.MspiFirstIndex}, Count={mslkEntry.MspiIndexCount}, Unk00=0x{mslkEntry.Unknown_0x00:X2}, Unk01=0x{mslkEntry.Unknown_0x01:X2}, Unk04=0x{mslkEntry.Unknown_0x04:X8}, Unk10=0x{mslkEntry.Unknown_0x10:X4}, Unk12=0x{mslkEntry.Unknown_0x12:X4}"));

                            if (mslkEntry.MspiIndexCount > 0 && mslkEntry.MspiFirstIndex >= 0)
                            {
                                int mspiStart = mslkEntry.MspiFirstIndex;
                                int mspiEndExclusive = mspiStart + mslkEntry.MspiIndexCount;

                                if (mspiEndExclusive <= totalMspiIndices)
                                {
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

                                    if (validMspvIndices.Count >= 2)
                                    {
                                        mslkWriter.WriteLine($"g MSLK_Path_{entryIndex}");
                                        mslkWriter.WriteLine("l " + string.Join(" ", validMspvIndices));
                                        debugWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices to _mslk.obj.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    Exported line with {validMspvIndices.Count} vertices."); }
                                        exportedPaths++;
                                    }
                                    else if (validMspvIndices.Count == 1)
                                    {
                                        mslkWriter.WriteLine($"g MSLK_Point_{entryIndex}");
                                        mslkWriter.WriteLine($"p {validMspvIndices[0]}");
                                        debugWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]} to _mslk.obj.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    Exported single point at vertex {validMspvIndices[0]}. "); }
                                        exportedPoints++;
                                    }
                                    else
                                    {
                                        debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} resulted in 0 valid MSPV indices. Skipping geometry output.");
                                        if (entryIndex < 10) { summaryWriter.WriteLine($"    INFO: Resulted in 0 valid MSPV indices. Skipping."); }
                                        skippedMslkWriter.WriteLine($"Skipped (0 Valid MSPV): {mslkEntry.ToString()}");
                                        skippedMslkCount++;
                                    }
                                }
                                else
                                {
                                    debugWriter.WriteLine($"    ERROR: MSLK Entry {entryIndex} defines invalid MSPI range [First:{mspiStart}, Count:{mslkEntry.MspiIndexCount}] (Max MSPI Index: {totalMspiIndices - 1}). Skipping entry.");
                                    if (entryIndex < 10) { summaryWriter.WriteLine($"    ERROR: Invalid MSPI range. Skipping entry."); }
                                    skippedMslkWriter.WriteLine($"Skipped (Invalid MSPI Range): {mslkEntry.ToString()}");
                                    skippedMslkCount++;
                                }
                            }
                            else
                            {
                                debugWriter.WriteLine($"    INFO: MSLK Entry {entryIndex} has MSPICount=0 or MSPIFirst=-1. Skipping geometry export.");
                                if (entryIndex < 10) { summaryWriter.WriteLine($"    INFO: MSPICount=0 or MSPIFirst=-1. Skipping."); }
                                skippedMslkWriter.WriteLine($"Skipped (Count=0 or FirstIndex<0): {mslkEntry.ToString()}");
                                skippedMslkCount++;
                            }

                            if (mslkEntry.MspiFirstIndex == -1)
                            {
                                if (msviIndices != null && msvtVertices != null && msviCount > 0 && msvtCount > 0)
                                {
                                    ushort msviLookupIndex = mslkEntry.Unknown_0x10;
                                    if (msviLookupIndex < msviCount)
                                    {
                                        uint msvtLookupIndex = msviIndices[msviLookupIndex];
                                        if (msvtLookupIndex < msvtCount)
                                        {
                                            var msvtVertex = msvtVertices[(int)msvtLookupIndex];

                                            float worldX = msvtVertex.Y;
                                            float worldY = msvtVertex.X;
                                            float worldZ = msvtVertex.Z;

                                            mslkNodesWriter.WriteLine($"v {worldX.ToString(CultureInfo.InvariantCulture)} {worldY.ToString(CultureInfo.InvariantCulture)} {worldZ.ToString(CultureInfo.InvariantCulture)} # Node Idx={entryIndex} Unk00=0x{mslkEntry.Unknown_0x00:X2} Unk01=0x{mslkEntry.Unknown_0x01:X2} Unk10={mslkEntry.Unknown_0x10}");
                                            processedMslkNodes++;

                                            debugWriter.WriteLine($"  MSLK Node Entry {entryIndex}: Unk10={mslkEntry.Unknown_0x10} -> MSVI[{mslkEntry.Unknown_0x10}]={msvtLookupIndex} -> MSVT[{msvtLookupIndex}]=({msvtVertex.X},{msvtVertex.Y},{msvtVertex.Z}) -> World=({worldX},{worldY},{worldZ}) Unk00=0x{mslkEntry.Unknown_0x00:X2} Unk01=0x{mslkEntry.Unknown_0x01:X2}");
                                        }
                                        else
                                        {
                                            debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: MSVI index {msvtLookupIndex} (from MSVI[{mslkEntry.Unknown_0x10}]) is out of bounds for MSVT ({msvtCount}). Skipping node export.");
                                            skippedMslkWriter.WriteLine($"Node Entry {entryIndex}: Invalid MSVT Index {msvtLookupIndex} (from MSVI[{mslkEntry.Unknown_0x10}]) for MSVT Count {msvtCount}. Unk10={mslkEntry.Unknown_0x10}");
                                        }
                                    }
                                    else
                                    {
                                        debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: Unk10 index {mslkEntry.Unknown_0x10} is out of bounds for MSVI ({msviCount}). Skipping node export.");
                                        skippedMslkWriter.WriteLine($"Node Entry {entryIndex}: Invalid MSVI Index {mslkEntry.Unknown_0x10} for MSVI Count {msviCount}.");
                                    }
                                }
                                else
                                {
                                    debugWriter.WriteLine($"  WARN: MSLK Node Entry {entryIndex}: MSVI or MSVT data is missing or empty. Cannot process node anchor.");
                                    skippedMslkWriter.WriteLine($"Node Entry {entryIndex}: Missing MSVI/MSVT data. Cannot process node.");
                                }
                            }
                        }
                        debugWriter.WriteLine($"Finished processing {entriesToProcessMslk} MSLK entries. Exported {exportedPaths} paths, {exportedPoints} points. Skipped {skippedMslkCount} entries.");
                        if (exportOnlyFirstMslk && pm4File.MSLK.Entries.Count > 1)
                        {
                            debugWriter.WriteLine("Note: MSLK processing was limited to the first entry by 'exportOnlyFirstMslk' flag.");
                        }
                        summaryWriter.WriteLine($"--- Finished MSLK Processing (Exported Paths: {exportedPaths}, Points: {exportedPoints}, Skipped: {skippedMslkCount}) ---");
                        mslkWriter.WriteLine();
                        debugWriter.Flush();
                        skippedMslkWriter.Flush();
                    }
                    else
                    {
                        debugWriter.WriteLine("Skipping MSLK path export (MSLK, MSPI, or MSPV data missing or no MSPV vertices exported).");
                    }
                } else {
                    debugWriter.WriteLine("\nSkipping MSLK Path export (Flag 'exportMslkPaths' is False).");
                }

                // --- 6. Export MSUR surfaces -> msvtWriter ONLY ---
                if (processMsurEntries)
                {
                    if (pm4File.MSUR != null && pm4File.MSVI != null && pm4File.MSVT != null && pm4File.MDOS != null && msvtFileVertexCount > 0)
                    {
                        debugWriter.WriteLine($"\n--- Processing MSUR -> MSVI Links (Logging Only - First 20 Entries) ---");
                        summaryWriter.WriteLine("\n--- MSUR -> MSVI Links (First 20 Entries) ---");

                        int entriesToProcess = exportOnlyFirstMsur ? Math.Min(1, pm4File.MSUR.Entries.Count) : pm4File.MSUR.Entries.Count;
                        for (int entryIndex = 0; entryIndex < Math.Min(entriesToProcess, 20); entryIndex++)
                        {
                            var msurEntry = pm4File.MSUR.Entries[entryIndex];
                            debugWriter.WriteLine($"  Processing MSUR Entry {entryIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, MdosIndex={msurEntry.MdosIndex}, FlagsOrUnk0={msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}"); 
                            summaryWriter.WriteLine($"  Processing MSUR Entry {entryIndex}: MsviFirstIndex={msurEntry.MsviFirstIndex}, IndexCount={msurEntry.IndexCount}, MdosIndex={msurEntry.MdosIndex}, FlagsOrUnk0={msurEntry.FlagsOrUnknown_0x00:X2}, Unk02={msurEntry.Unknown_0x02}");
                            
                            int firstIndex = (int)msurEntry.MsviFirstIndex;
                            int indexCount = msurEntry.IndexCount;
                            uint mdosIndex = msurEntry.MdosIndex;

                            string groupName;
                            if (mdosIndex < (pm4File.MDOS?.Entries.Count ?? 0))
                            {
                                var mdosEntry = pm4File.MDOS!.Entries[(int)mdosIndex];
                                groupName = $"MSUR_Mdos{mdosIndex}_ID{mdosEntry.m_destructible_building_index:X8}";
                                debugWriter.WriteLine($"    MDOS Link Found: Index={mdosIndex}, Entry={mdosEntry}");
                                summaryWriter.WriteLine($"    MDOS Link Found: Index={mdosIndex}, Entry={mdosEntry}");
                            }
                            else
                            {
                                debugWriter.WriteLine($"  MSUR Entry {entryIndex}: Invalid MDOS index {mdosIndex}. Assigning default group name.");
                                summaryWriter.WriteLine($"  MSUR Entry {entryIndex}: Invalid MDOS index {mdosIndex}. Assigning default group name.");
                            }

                            if (pm4File.MSVI == null) { continue; }

                            if (firstIndex >= 0 && firstIndex + indexCount <= pm4File.MSVI!.Indices.Count)
                            {
                                 if (indexCount < 1) { 
                                     debugWriter.WriteLine($"    Skipping: Zero indices requested (IndexCount={indexCount}).");
                                     summaryWriter.WriteLine($"    Skipping: Zero indices requested (IndexCount={indexCount}).");
                                     continue; 
                                 }

                                List<uint> msviIndices = pm4File.MSVI.Indices.GetRange(firstIndex, indexCount);
                                debugWriter.WriteLine($"    Fetched {msviIndices.Count} MSVI Indices (Expected {indexCount}).");
                                debugWriter.WriteLine($"      Debug MSVI Indices: [{string.Join(", ", msviIndices)}]");
                                
                                if (entryIndex < 20) { summaryWriter.WriteLine($"    Fetched {msviIndices.Count} MSVI Indices (Expected {indexCount})."); }
                                
                                bool faceIsValid = true;
                                List<int> objFaceIndices = new List<int>(indexCount);
                                foreach (uint msviIdx in msviIndices)
                                {
                                    if (msviIdx < msvtFileVertexCount) 
                                    {
                                        objFaceIndices.Add((int)msviIdx + 1);
                                    }
                                    else
                                    {
                                        debugWriter.WriteLine($"    ERROR: MSUR Entry {entryIndex} references invalid MSVT index {msviIdx} (via MSVI). Max exported MSVT index: {msvtFileVertexCount - 1}. Skipping face.");
                                        if (entryIndex < 20) { summaryWriter.WriteLine($"    ERROR: MSUR Entry {entryIndex} references invalid MSVT index {msviIdx} (via MSVI). Max exported: {msvtFileVertexCount - 1}. Skipping face."); }
                                        faceIsValid = false;
                                        break;
                                    }
                                }

                                if (faceIsValid && objFaceIndices.Count >= 3)
                                {
                                    msvtWriter.WriteLine("f " + string.Join(" ", objFaceIndices));
                                    debugWriter.WriteLine($"    Wrote face with {objFaceIndices.Count} vertices to _msvt.obj.");
                                    if (entryIndex < 20) { summaryWriter.WriteLine($"    Wrote face with {objFaceIndices.Count} vertices to _msvt.obj."); }
                                }
                                else if (faceIsValid)
                                {
                                    debugWriter.WriteLine($"    Skipping face for MSUR Entry {entryIndex}: Not enough valid vertices ({objFaceIndices.Count}). Minimum required: 3.");
                                    if (entryIndex < 20) { summaryWriter.WriteLine($"    Skipping face: Not enough valid vertices ({objFaceIndices.Count})."); }
                                }

                                int indicesToLogSummary = Math.Min(msviIndices.Count, 15);
                                if (entryIndex < 20) {
                                    summaryWriter.WriteLine($"      Summary MSVI Indices (first {indicesToLogSummary}): [{string.Join(", ", msviIndices.Take(indicesToLogSummary))}]" + (msviIndices.Count > 15 ? " ..." : "")); 
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
                                debugWriter.WriteLine($"    Skipped MSUR Entry {entryIndex} due to invalid MSVI range (FirstIndex={firstIndex}, IndexCount={indexCount}, Total MSVI Indices={pm4File.MSVI!.Indices.Count}).");
                                summaryWriter.WriteLine($"    Skipped MSUR Entry {entryIndex} due to invalid MSVI range (FirstIndex={firstIndex}, IndexCount={indexCount}, Total MSVI Indices={pm4File.MSVI!.Indices.Count}).");
                            }
                        }
                        if (entriesToProcess > 20)
                        {
                            debugWriter.WriteLine($"  ... Log truncated after first 20 MSUR entries ...");
                            summaryWriter.WriteLine("  ... (Log limited to first 20 entries) ...");
                        }
                        debugWriter.WriteLine("--- Finished Processing MSUR -> MSVI Links ---");
                        debugWriter.Flush();
                    }
                    else { debugWriter.WriteLine("\nSkipping MSUR->MSVI Link processing (missing data or source vertices)."); }
                } else {
                    debugWriter.WriteLine("\n--- Skipping MSUR->MSVI Link processing (Flag 'processMsurEntries' is False) ---");
                }

                 // --- 7. Export MPRR pairs -> mprlWriter ONLY ---
                 if (exportMprrLines)
                 {
                     if (pm4File.MPRR != null && pm4File.MPRL != null && mprlFileVertexCount > 0)
                     {
                        debugWriter.WriteLine($"\n--- Processing MPRR -> MPRL Chain -> _mprl.obj ---");
                        summaryWriter.WriteLine("\n--- MPRR Lines (First 10 Valid) ---");
                         int logCounterMprr = 0;
                         int validMprrLogged = 0;
 
                         foreach (var mprrEntry in pm4File.MPRR.Entries)
                         {
                             ushort rawIndex1 = mprrEntry.Unknown_0x00;
                             ushort rawIndex2 = mprrEntry.Unknown_0x02;
 
                             bool shouldLogSummary = validMprrLogged < 10;
 
                             debugWriter.WriteLine($"  Processing MPRR Entry {logCounterMprr}: RawIdx1={rawIndex1}, RawIdx2={rawIndex2}"); 
                             
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

                            if (rawIndex1 < mprlFileVertexCount && rawIndex2 < mprlFileVertexCount) 
                            {
                                uint relativeObjIndex1 = 1 + (uint)rawIndex1;
                                uint relativeObjIndex2 = 1 + (uint)rawIndex2;
 
                                bool firstProcessedEntry = !pm4File.MPRR.Entries
                                    .Take(pm4File.MPRR.Entries.IndexOf(mprrEntry))
                                    .Any(e => e.Unknown_0x00 != 65535 && e.Unknown_0x02 != 65535 && e.Unknown_0x00 != e.Unknown_0x02 && e.Unknown_0x00 < mprlFileVertexCount && e.Unknown_0x02 < mprlFileVertexCount);
                                if (firstProcessedEntry) { 
                                    mprlWriter.WriteLine("g MPRR_Lines"); 
                                    if(shouldLogSummary) { summaryWriter.WriteLine("  (Group MPRR_Lines)"); }
                                }
 
                                debugWriter.WriteLine($"    Writing SEPARATE OBJ Line {relativeObjIndex1} -> {relativeObjIndex2} (From Raw MPRL Indices: {rawIndex1}, {rawIndex2})"); 
                                if (shouldLogSummary) {
                                    summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Line {relativeObjIndex1} -> {relativeObjIndex2} (Raw: {rawIndex1}, {rawIndex2})");
                                    validMprrLogged++;
                                }
                            }
                            else { 
                                  debugWriter.WriteLine($"    Skipping: Indices out of bounds (RawIdx1={rawIndex1}, RawIdx2={rawIndex2}, File MPRL Count={mprlFileVertexCount}). CHECK MPRR ASSERTION."); 
                                  if (shouldLogSummary) {
                                      summaryWriter.WriteLine($"  MPRR Entry {logCounterMprr}: Skip Invalid Index (Raw: {rawIndex1}, {rawIndex2} / Max: {mprlFileVertexCount - 1})");
                                      validMprrLogged++;
                                  }
                            }
                            logCounterMprr++;
                         }
                         if (validMprrLogged >= 10) { summaryWriter.WriteLine("  ... (Summary log limited to first 10 processed entries/skips) ..."); }
                         debugWriter.WriteLine("--- Finished Processing MPRR Chain for separate file ---");
                         mprlWriter.WriteLine();
                         debugWriter.Flush();
                     }
                     else { debugWriter.WriteLine("\nSkipping MPRR Chain processing (missing data or source vertices)."); }
                 } else {
                      debugWriter.WriteLine("\nSkipping MPRR Line export (Flag False)."); 
                      debugWriter.WriteLine("\n--- Skipping MPRR Line export (Flag 'exportMprrLines' is False) ---");
                 }

                 // --- 8. Log MDSF -> MDOS Links ---
                 if (logMdsfLinks)
                 {
                     if (pm4File.MDSF != null && pm4File.MDOS != null && pm4File.MDSF.Entries.Any())
                     {
                         debugWriter.WriteLine($"\n--- Logging MDSF -> MDOS Links and collecting Building IDs ---");
                         summaryWriter.WriteLine($"\n--- MDSF -> MDOS Links (First 20) ---");
                         int mdsfLoggedCount = 0;
                         int totalMdsfEntries = pm4File.MDSF.Entries.Count;

                         for (int mdsfIndex = 0; mdsfIndex < totalMdsfEntries; mdsfIndex++)
                         {
                             var mdsfEntry = pm4File.MDSF.Entries[mdsfIndex];
                             string logLineBase = $"  MDSF[{mdsfIndex}]: MSUR_Idx={mdsfEntry.msur_index}, MDOS_Idx={mdsfEntry.mdos_index}";
                             string logLineDetails = "";

                             if (mdsfEntry.mdos_index < mdosEntryCount)
                             {
                                 var mdosEntry = pm4File.MDOS.Entries[(int)mdsfEntry.mdos_index];
                                 logLineDetails = $" -> MDOS[DestState={mdosEntry.destruction_state}, BldgIdx={mdosEntry.m_destructible_building_index}]";
                                 uniqueBuildingIds.Add(mdosEntry.m_destructible_building_index);
                             }
                             else
                             {
                                 logLineDetails = $" -> MDOS Index {mdsfEntry.mdos_index} OUT OF BOUNDS (Count: {mdosEntryCount})";
                             }

                             debugWriter.WriteLine(logLineBase + logLineDetails);

                             if (mdsfLoggedCount < 20)
                             {
                                 summaryWriter.WriteLine(logLineBase + logLineDetails);
                                 mdsfLoggedCount++;
                             }
                         }

                         if (totalMdsfEntries > 20)
                         {
                             summaryWriter.WriteLine("  ... (Summary log limited to first 20 entries) ...");
                         }
                         debugWriter.WriteLine($"--- Finished logging MDSF Links ({totalMdsfEntries} total) ---");
                         debugWriter.Flush();
                     }
                     else
                     {
                         debugWriter.WriteLine("\n--- Skipping MDSF Link logging (MDSF or MDOS chunk missing or empty) ---");
                     }
                 }
                 else
                 {
                     debugWriter.WriteLine("\n--- Skipping MDSF Link logging (Flag 'logMdsfLinks' is False) ---");
                 }

                 // --- 9. Generate MSLK Hierarchical JSON ---
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

                         if (entry.MspiFirstIndex == -1)
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
                         else
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

                 Console.WriteLine($"Successfully prepared data for writing split files.");
                 debugWriter.WriteLine($"Split data preparation complete.");

                 // Log count before closing writers in finally block
                 debugWriter.WriteLine($"\nTotal unique building IDs collected: {uniqueBuildingIds.Count}");
                 summaryWriter.WriteLine($"\nTotal unique building IDs collected: {uniqueBuildingIds.Count}");
             }
             catch (Exception ex)
             {
                  Console.WriteLine($"Error during data processing/preparation: {ex.ToString()}");
                  try { debugWriter?.WriteLine($"### EXCEPTION DURING DATA PROCESSING: {ex.ToString()} ###"); } catch { /* Ignore */ }
             }
            finally
            {
                // Close all writers
                debugWriter?.Close();
                summaryWriter?.Close();
                msvtWriter?.Close();
                mspvWriter?.Close();
                mprlWriter?.Close();
                mscnWriter?.Close();
                mslkWriter?.Close();
                skippedMslkWriter?.Close();
                mslkNodesWriter?.Close();
            }

            // --- Write Unique Building IDs Log (Moved outside main try/finally) ---
            Console.WriteLine($"\nAttempting to write {uniqueBuildingIds.Count} unique building IDs to {outputBuildingIdsPath}");

            if (uniqueBuildingIds.Any())
            {
                StreamWriter? buildingIdWriter = null;
                try
                {
                    buildingIdWriter = new StreamWriter(outputBuildingIdsPath, false);
                    buildingIdWriter.WriteLine("# Unique Building IDs (m_destructible_building_index from MDOS via MDSF)");
                    buildingIdWriter.WriteLine($"# Total Found: {uniqueBuildingIds.Count}");
                    foreach (uint id in uniqueBuildingIds.OrderBy(id => id))
                    {
                        buildingIdWriter.WriteLine(id);
                    }
                    Console.WriteLine($"Successfully wrote unique building IDs to {outputBuildingIdsPath}");
                }
                catch (Exception ex)
                {
                    // Log detailed exception
                    Console.WriteLine($"Error writing unique building IDs log to {outputBuildingIdsPath}: {ex.ToString()}");
                    // Attempt to log to debug log if possible (might fail if path issue exists)
                    try { File.AppendAllText(debugLogPath, $"\n### ERROR WRITING BUILDING IDS LOG: {ex.ToString()} ###\n"); }
                    catch { /* Best effort */ }
                }
                finally
                {
                    buildingIdWriter?.Close();
                }
            }
            else
            {
                Console.WriteLine("No unique building IDs found to write.");
                 try { File.AppendAllText(debugLogPath, "\nNo unique building IDs found to write.\n"); } catch { /* Best effort */ }
            }
            // --- END: Write Unique Building IDs Log ---

            Assert.True(File.Exists(outputMsvtFilePath), $"MSVT OBJ file was not created at {outputMsvtFilePath}");
            Assert.True(File.Exists(outputMspvFilePath), $"MSPV OBJ file was not created at {outputMspvFilePath}");
            Assert.True(File.Exists(outputMprlFilePath), $"MPRL OBJ file was not created at {outputMprlFilePath}");
            Assert.True(File.Exists(outputMscnFilePath), $"MSCN OBJ file was not created at {outputMscnFilePath}");
            Assert.True(File.Exists(outputMslkFilePath), $"MSLK OBJ file should exist at {outputMslkFilePath}");
            Assert.True(File.Exists(outputMslkJsonPath), $"MSLK JSON file should exist at {outputMslkJsonPath}");
            Assert.True(File.Exists(outputSkippedMslkLogPath), $"Skipped MSLK log file should exist at {outputSkippedMslkLogPath}");
            Assert.True(File.Exists(outputPm4MslkNodesFilePath), $"PM4 MSLK Nodes OBJ file should exist at {outputPm4MslkNodesFilePath}");
            Assert.True(File.Exists(summaryLogPath), $"Summary log file was not created at {summaryLogPath}");
            Assert.True(File.Exists(outputBuildingIdsPath), $"Building IDs log file was not created at {outputBuildingIdsPath}");

            Console.WriteLine("--- LoadPM4File_ShouldLoadChunks END ---");
        }

        // ... other test methods ...
    }
} 