using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Chunks; // Add missing using
using Warcraft.NET.Extensions; // For SeekChunk and ReadBinarySignature

namespace WoWToolbox.AnalysisTool
{
    public class MprrHypothesisAnalyzer
    {
        // Define constants for the field checks based on hypotheses
        private const int MSLK_UNK00_MAX = 32;
        private const int MSLK_UNK01_MAX = 12;
        private const ushort MSLK_UNK02_EXPECTED = 0x0000;
        private const ushort MSLK_UNK10_EXPECTED = 0xFFFF; // Hypothesis: Should always be FFFF
        private const ushort MSLK_UNK12_EXPECTED = 0x8000;
        private const short MPRL_UNK02_EXPECTED = -1; // Corrected type to short and value to -1 (== 0xFFFF as ushort)
        private const ushort MPRL_UNK06_EXPECTED = 0x8000;

        // Structure to hold aggregate statistics
        private class FieldStats
        {
            public HashSet<object> DistinctValues { get; } = new HashSet<object>();
            public long Count { get; set; } = 0;
            public long ViolationCount { get; set; } = 0;
            public object MinValue { get; set; } = null;
            public object MaxValue { get; set; } = null;

            public void Update<T>(T value, bool isViolation = false) where T : IComparable
            {
                Count++;
                DistinctValues.Add(value);
                if (isViolation) ViolationCount++;

                if (MinValue == null || value.CompareTo(MinValue) < 0)
                {
                    MinValue = value;
                }
                if (MaxValue == null || value.CompareTo(MaxValue) > 0)
                {
                    MaxValue = value;
                }
            }
        }


        public void Analyze(string inputDirectory, string outputDirectory)
        {
            Console.WriteLine($"Starting Analysis...");
            Console.WriteLine($"Input Directory: {inputDirectory}");
            Console.WriteLine($"Output Directory: {outputDirectory}");

            if (!Directory.Exists(inputDirectory))
            {
                Console.WriteLine($"Error: Input directory not found: {inputDirectory}");
                return;
            }

            Directory.CreateDirectory(outputDirectory); // Ensure output directory exists

            var pm4Files = Directory.GetFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly);

            if (pm4Files.Length == 0)
            {
                Console.WriteLine($"Warning: No .pm4 files found in {inputDirectory}");
                return;
            }

            // --- Output Files ---
            string summaryLogPath = Path.Combine(outputDirectory, "_analysis_summary.log");
            string fieldValidationLogPath = Path.Combine(outputDirectory, "_field_validation.log");
            string mprrAnalysisCsvPath = Path.Combine(outputDirectory, "_mprr_analysis.csv");
            string mslkUnk10LogPath = Path.Combine(outputDirectory, "_mslk_unk10_values.log"); // For checking Unk10 hypothesis

            using var summaryWriter = new StreamWriter(summaryLogPath);
            using var validationWriter = new StreamWriter(fieldValidationLogPath);
            using var mprrWriter = new StreamWriter(mprrAnalysisCsvPath);
            using var mslkUnk10Writer = new StreamWriter(mslkUnk10LogPath);

            summaryWriter.WriteLine($"--- Analysis Summary ({DateTime.Now}) ---");
            summaryWriter.WriteLine($"Input Directory: {inputDirectory}");
            summaryWriter.WriteLine($"Processed {pm4Files.Length} files.");
            validationWriter.WriteLine($"--- Field Validation Log ({DateTime.Now}) ---");
            mprrWriter.WriteLine("FileName,SequenceIndex,FlagValueHex,IndicesCount,MinIndex,MaxIndex,Indices");
            mslkUnk10Writer.WriteLine($"--- MSLK Unknown_0x10 Values Log ({DateTime.Now}) ---");

            // --- Aggregate Statistics ---
            var stats = new Dictionary<string, FieldStats>
            {
                // MSLK
                { "MSLK.Unk00", new FieldStats() },
                { "MSLK.Unk01", new FieldStats() },
                { "MSLK.Unk02", new FieldStats() },
                { "MSLK.Unk0C", new FieldStats() },
                { "MSLK.Unk10", new FieldStats() },
                { "MSLK.Unk12", new FieldStats() },
                // MPRL
                { "MPRL.Unk00", new FieldStats() },
                { "MPRL.Unk02", new FieldStats() },
                { "MPRL.Unk04", new FieldStats() }, // Include for context
                { "MPRL.Unk06", new FieldStats() },
                { "MPRL.Unk14", new FieldStats() }, // Include for context
                { "MPRL.Unk16", new FieldStats() }, // Include for context
                // MPRR (Re-interpreted)
                { "MPRR.FlagValue", new FieldStats() },
                { "MPRR.IndexValue", new FieldStats() }
            };
            long totalMslkUnk0cMprlUnk00Matches = 0;
            long totalMslkMprlComparisons = 0;

            // --- File Loop ---
            foreach (var filePath in pm4Files)
            {
                string fileName = Path.GetFileName(filePath);
                Console.WriteLine($"Processing: {fileName}");
                validationWriter.WriteLine($"\n--- {fileName} ---");
                mslkUnk10Writer.WriteLine($"\n--- {fileName} ---");

                PM4File pm4File = null;
                try
                {
                    pm4File = PM4File.FromFile(filePath);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error loading {fileName}: {ex.Message}");
                    validationWriter.WriteLine($"  ERROR loading file: {ex.Message}");
                    continue; // Skip to next file
                }

                // --- Get MPRR Data via Serialize() ---
                byte[]? mprrDataToAnalyze = null;
                if (pm4File.MPRR != null)
                {
                    try
                    {
                        mprrDataToAnalyze = pm4File.MPRR.Serialize();
                        validationWriter.WriteLine($"  MPRR chunk found and serialized ({mprrDataToAnalyze?.Length ?? 0} bytes).");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Error serializing MPRR chunk for {fileName}: {ex.Message}");
                        validationWriter.WriteLine($"  ERROR serializing existing MPRR chunk: {ex.Message}");
                    }
                }
                else
                {
                    validationWriter.WriteLine("  MPRR chunk was null after PM4File load.");
                }

                // --- Analyze Re-Serialized MPRR Data ---
                if (mprrDataToAnalyze != null)
                {
                    AnalyzeRawMprrData(fileName, mprrDataToAnalyze, mprrWriter, stats);
                }
                else
                {
                     // No need to log again, handled above
                }

                // --- Validate Fields & Check Correlation ---
                if (pm4File.MSLK != null)
                {
                    validationWriter.WriteLine("  MSLK Field Validation:");
                    int mslkNodeIndex = 0;
                    foreach (var entry in pm4File.MSLK.Entries)
                    {
                        bool unk00Violation = entry.Unknown_0x00 >= MSLK_UNK00_MAX;
                        stats["MSLK.Unk00"].Update(entry.Unknown_0x00, unk00Violation);
                        if (unk00Violation) validationWriter.WriteLine($"    Entry {mslkNodeIndex}: Unk00 violation ({entry.Unknown_0x00} >= {MSLK_UNK00_MAX})");

                        bool unk01Violation = !(entry.Unknown_0x01 == 0xFF || entry.Unknown_0x01 < MSLK_UNK01_MAX);
                        stats["MSLK.Unk01"].Update(entry.Unknown_0x01, unk01Violation);
                        if (unk01Violation) validationWriter.WriteLine($"    Entry {mslkNodeIndex}: Unk01 violation ({entry.Unknown_0x01} not in [0..{MSLK_UNK01_MAX-1}] or 0xFF)");

                        bool unk02Violation = entry.Unknown_0x02 != MSLK_UNK02_EXPECTED;
                        stats["MSLK.Unk02"].Update(entry.Unknown_0x02, unk02Violation);
                        if (unk02Violation) validationWriter.WriteLine($"    Entry {mslkNodeIndex}: Unk02 violation (0x{entry.Unknown_0x02:X4} != 0x{MSLK_UNK02_EXPECTED:X4})");

                        stats["MSLK.Unk0C"].Update(entry.Unknown_0x0C); // No violation check currently

                        bool unk10Violation = entry.Unknown_0x10 != MSLK_UNK10_EXPECTED; // Check hypothesis
                        stats["MSLK.Unk10"].Update(entry.Unknown_0x10, unk10Violation);
                        if (unk10Violation) validationWriter.WriteLine($"    Entry {mslkNodeIndex}: Unk10 violation (0x{entry.Unknown_0x10:X4} != 0x{MSLK_UNK10_EXPECTED:X4}) - Hypothesis Failed?");
                         // Log all Unk10 values for nodes regardless of violation
                        if(entry.MspiFirstIndex == -1) { // Only log for nodes
                            mslkUnk10Writer.WriteLine($"    Node Entry {mslkNodeIndex}: Unk10 = 0x{entry.Unknown_0x10:X4}");
                        }


                        bool unk12Violation = entry.Unknown_0x12 != MSLK_UNK12_EXPECTED;
                        stats["MSLK.Unk12"].Update(entry.Unknown_0x12, unk12Violation);
                        if (unk12Violation) validationWriter.WriteLine($"    Entry {mslkNodeIndex}: Unk12 violation (0x{entry.Unknown_0x12:X4} != 0x{MSLK_UNK12_EXPECTED:X4})");

                        mslkNodeIndex++;
                    }
                } else { validationWriter.WriteLine("  MSLK chunk not present."); }

                if (pm4File.MPRL != null)
                {
                    validationWriter.WriteLine("  MPRL Field Validation:");
                    int mprlIndex = 0;
                    foreach (var entry in pm4File.MPRL.Entries)
                    {
                        stats["MPRL.Unk00"].Update(entry.Unknown_0x00); // No violation check currently

                        bool unk02Violation = entry.Unknown_0x02 != MPRL_UNK02_EXPECTED; // Compare short to short
                        stats["MPRL.Unk02"].Update(entry.Unknown_0x02, unk02Violation);
                        if (unk02Violation) validationWriter.WriteLine($"    Entry {mprlIndex}: Unk02 violation ({entry.Unknown_0x02} != {MPRL_UNK02_EXPECTED})"); // Updated log message

                        stats["MPRL.Unk04"].Update(entry.Unknown_0x04); // Include for context

                        bool unk06Violation = entry.Unknown_0x06 != MPRL_UNK06_EXPECTED;
                        stats["MPRL.Unk06"].Update(entry.Unknown_0x06, unk06Violation);
                        if (unk06Violation) validationWriter.WriteLine($"    Entry {mprlIndex}: Unk06 violation (0x{entry.Unknown_0x06:X4} != 0x{MPRL_UNK06_EXPECTED:X4})");

                        stats["MPRL.Unk14"].Update(entry.Unknown_0x14); // Include for context
                        stats["MPRL.Unk16"].Update(entry.Unknown_0x16); // Include for context

                        mprlIndex++;
                    }
                } else { validationWriter.WriteLine("  MPRL chunk not present."); }

                // Check MSLK.Unk0C vs MPRL.Unk00 Correlation (Simple check: count matches)
                if (pm4File.MSLK != null && pm4File.MPRL != null)
                {
                     validationWriter.WriteLine("  MSLK.Unk0C vs MPRL.Unk00 Correlation Check:");
                     long fileComparisons = 0;
                     long fileMatches = 0;
                     // This is a naive check - assumes direct correspondence which might not be true
                     // A more sophisticated analysis might be needed
                     foreach(var mslkEntry in pm4File.MSLK.Entries)
                     {
                         foreach(var mprlEntry in pm4File.MPRL.Entries)
                         {
                             if(mslkEntry.Unknown_0x0C == mprlEntry.Unknown_0x00)
                             {
                                 fileMatches++;
                                 // Could log specific matches here if needed, but might be too verbose
                             }
                             fileComparisons++;
                         }
                     }
                     totalMslkUnk0cMprlUnk00Matches += fileMatches;
                     totalMslkMprlComparisons += fileComparisons;
                     validationWriter.WriteLine($"    Found {fileMatches} matches out of {fileComparisons} pairwise comparisons in this file.");
                }


            } // End file loop

            // --- Write Summary ---
            summaryWriter.WriteLine("\n--- Aggregate Field Statistics ---");
            foreach (var kvp in stats.OrderBy(kv => kv.Key))
            {
                var stat = kvp.Value;
                summaryWriter.WriteLine($"\nField: {kvp.Key}");
                summaryWriter.WriteLine($"  Total Entries Checked: {stat.Count}");
                summaryWriter.WriteLine($"  Violations (Based on Hypotheses): {stat.ViolationCount} ({((double)stat.ViolationCount / stat.Count * 100):F2}%)");
                summaryWriter.WriteLine($"  Distinct Values ({stat.DistinctValues.Count}): {(stat.DistinctValues.Count > 50 ? "(>50, see validation log/CSV)" : string.Join(", ", stat.DistinctValues.OrderBy(dv => dv).Select(dv => dv is ushort || dv is byte || dv is uint ? $"0x{dv:X}" : dv.ToString())))}");
                summaryWriter.WriteLine($"  Min Value: {(stat.MinValue is ushort || stat.MinValue is byte || stat.MinValue is uint ? $"0x{stat.MinValue:X}" : stat.MinValue?.ToString() ?? "N/A")}");
                summaryWriter.WriteLine($"  Max Value: {(stat.MaxValue is ushort || stat.MaxValue is byte || stat.MaxValue is uint ? $"0x{stat.MaxValue:X}" : stat.MaxValue?.ToString() ?? "N/A")}");
            }

            summaryWriter.WriteLine("\n--- MSLK.Unk0C vs MPRL.Unk00 Correlation Summary ---");
            summaryWriter.WriteLine($"  Total Pairwise Comparisons: {totalMslkMprlComparisons}");
            summaryWriter.WriteLine($"  Total Matches Found: {totalMslkUnk0cMprlUnk00Matches}");
             if(totalMslkMprlComparisons > 0)
                 summaryWriter.WriteLine($"  Match Percentage (Naive): {((double)totalMslkUnk0cMprlUnk00Matches / totalMslkMprlComparisons * 100):F4}%");


            Console.WriteLine($"Analysis complete. Results written to: {outputDirectory}");
        }

        // Helper to analyze raw MPRR data based on new hypothesis
        private void AnalyzeRawMprrData(string fileName, byte[] rawData, StreamWriter csvWriter, Dictionary<string, FieldStats> stats)
        {
             Console.WriteLine($"  DEBUG: Entering AnalyzeRawMprrData for {fileName}. Data length: {rawData?.Length ?? -1}");
             if (rawData == null || rawData.Length == 0) {
                 Console.WriteLine($"    DEBUG: MPRR raw data is null or empty for {fileName}. Skipping analysis.");
                 return;
             }
             if (rawData.Length % sizeof(ushort) != 0) {
                 Console.WriteLine($"  Warning: MPRR raw data length ({rawData.Length}) in {fileName} is not a multiple of {sizeof(ushort)}. Data might be corrupt.");
                 // Optionally try processing anyway? For now, skip.
                 return;
             }

            using var ms = new MemoryStream(rawData);
            using var br = new BinaryReader(ms);

            List<ushort> currentSequenceValues = new List<ushort>();
            int sequenceIndex = 0;

            while (br.BaseStream.Position < br.BaseStream.Length)
            {
                ushort value = br.ReadUInt16();
                currentSequenceValues.Add(value);

                if (value == 0xFFFF)
                {
                    // End of sequence
                    ushort? flagValue = null;
                    List<ushort> indices = new List<ushort>();

                    if (currentSequenceValues.Count > 1)
                    {
                        flagValue = currentSequenceValues[currentSequenceValues.Count - 2];
                        indices.AddRange(currentSequenceValues.Take(currentSequenceValues.Count - 2));
                    }
                    // else if Count == 1, it's just the terminator, flag is null, indices empty.

                    // Log to CSV
                    string flagHex = flagValue.HasValue ? $"0x{flagValue.Value:X4}" : "N/A";
                    int indicesCount = indices.Count;
                    ushort minIndex = indicesCount > 0 ? indices.Min() : (ushort)0;
                    ushort maxIndex = indicesCount > 0 ? indices.Max() : (ushort)0;
                    string indicesStr = indicesCount > 0 ? string.Join(";", indices.Select(i => $"0x{i:X4}")) : ""; // Use semicolon for CSV
                    csvWriter.WriteLine($"{fileName},{sequenceIndex},{flagHex},{indicesCount},{minIndex},{maxIndex},{indicesStr}");

                    // Update stats
                    if(flagValue.HasValue) stats["MPRR.FlagValue"].Update(flagValue.Value);
                    foreach(var idx in indices) {
                        stats["MPRR.IndexValue"].Update(idx);
                    }

                    currentSequenceValues.Clear();
                    sequenceIndex++;
                }
            }
             if (currentSequenceValues.Count > 0)
             {
                  Console.WriteLine($"  Warning: MPRR data for {fileName} ended with partial sequence: {string.Join(", ", currentSequenceValues.Select(v => $"0x{v:X4}"))}");
             }
        }
    }
} 