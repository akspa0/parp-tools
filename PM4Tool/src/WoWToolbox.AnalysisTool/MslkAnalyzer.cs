using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace WoWToolbox.AnalysisTool
{
    // Structure to hold parsed data from log lines
    public struct MslkEntryData
    {
        public int EntryIndex { get; set; }
        public int MspiFirstIndex { get; set; }
        public uint MspiIndexCount { get; set; }
        public byte Unknown_0x00 { get; set; }
        public byte Unknown_0x01 { get; set; }
        public uint Unknown_0x04 { get; set; }
        public ushort Unknown_0x10 { get; set; }
        public ushort Unknown_0x12 { get; set; }
        public string SourceLog { get; set; } // "debug" or "skipped"
        public string? SkipReason { get; set; } // Only for skipped log
    }

    public static class MslkAnalyzer
    {
        // Regex patterns to capture data from log lines
        // Debug Log: "Processing MSLK Entry 3: FirstIndex=-1, Count=0, Unk00=0x11, Unk01=0x00, Unk04=0x00000001, Unk10=0x004D, Unk12=0x8000"
        private static readonly Regex DebugLogRegex = new Regex(
            @"Processing MSLK Entry (?<index>\d+): FirstIndex=(?<first>-?\d+), Count=(?<count>\d+), Unk00=0x(?<unk00>[0-9A-Fa-f]{2}), Unk01=0x(?<unk01>[0-9A-Fa-f]{2}), Unk04=0x(?<unk04>[0-9A-Fa-f]{8}), Unk10=0x(?<unk10>[0-9A-Fa-f]{4}), Unk12=0x(?<unk12>[0-9A-Fa-f]{4})",
            RegexOptions.Compiled);

        // Skipped Log: "Skipped (Reason): Index=X, FirstIndex=Y, Count=Z, Unk00=..., Unk01=..., Unk04=..., Unk10=..., Unk12=..."
        // Assuming Unk12 might be added to the ToString() or log format
        private static readonly Regex SkippedLogRegex = new Regex(
             @"Skipped \((?<reason>[^)]+)\):.*Index=(?<index>\d+), FirstIndex=(?<first>-?\d+), Count=(?<count>\d+), Unk00=0x(?<unk00>[0-9A-Fa-f]+), Unk01=0x(?<unk01>[0-9A-Fa-f]+), Unk04=0x(?<unk04>[0-9A-Fa-f]+), Unk10=0x(?<unk10>[0-9A-Fa-f]+)(, Unk12=0x(?<unk12>[0-9A-Fa-f]+))?",
             RegexOptions.Compiled | RegexOptions.IgnoreCase);


        public static void AnalyzeMslkData(string skippedLogPath, string debugLogPath, string outputLogPath)
        {
            using StreamWriter writer = new StreamWriter(outputLogPath, false);
            
            Action<string> Log = (message) =>
            {
                Console.WriteLine(message);
                writer.WriteLine(message);
            };

            Log("--- Starting MSLK Analysis ---");
            Log($"Output Log: {outputLogPath}");
            var allEntries = new List<MslkEntryData>();

            // Parse Skipped Log
            Log($"Parsing skipped log: {skippedLogPath}");
            try
            {
                int skippedParsed = 0;
                foreach (var line in File.ReadLines(skippedLogPath))
                {
                    if (TryParseSkippedLogLine(line, out var entry))
                    {
                        allEntries.Add(entry);
                        skippedParsed++;
                    }
                    else if (!line.StartsWith("#") && !string.IsNullOrWhiteSpace(line)) // Ignore comments/empty
                    {
                       Log($"  [WARN] Could not parse skipped log line: {line}");
                    }
                }
                Log($"  Parsed {skippedParsed} entries from skipped log.");
            }
            catch (Exception ex)
            {
                Log($"  [ERROR] Failed to read or parse skipped log: {ex.Message}");
                // Optionally return or continue with only debug log data
            }

            int initialSkippedCount = allEntries.Count;

            // Parse Debug Log
            Log($"Parsing debug log: {debugLogPath}");
             try
            {
                 int debugParsedCount = 0;
                 foreach (var line in File.ReadLines(debugLogPath))
                 {
                     if (TryParseDebugLogLine(line, out var entry))
                     {
                         // Avoid duplicates if an entry was processed (and logged) AND also skipped (and logged)
                         // Use EntryIndex as a potential key - assumption: index is unique per file.
                         if (!allEntries.Any(e => e.EntryIndex == entry.EntryIndex))
                         {
                            allEntries.Add(entry);
                            debugParsedCount++;
                         }
                     }
                     // No warning for non-matching debug lines, as it's expected to contain other logs.
                 }
                 Log($"  Parsed {debugParsedCount} additional unique entries from debug log.");
            }
            catch (Exception ex)
            {
                Log($"  [ERROR] Failed to read or parse debug log: {ex.Message}");
            }

             Log($"Total unique entries parsed: {allEntries.Count}");
             if (allEntries.Count == 0)
             {
                 Log("[ERROR] No MSLK entries parsed. Cannot perform analysis.");
                 return;
             }

            // Group by Unknown_0x04
            Log("\n--- Grouping Entries by Unknown_0x04 ---");
            var groupedEntries = allEntries.GroupBy(e => e.Unknown_0x04)
                                           .ToDictionary(g => g.Key, g => g.ToList());
            Log($"Found {groupedEntries.Count} unique Unknown_0x04 groups.");

            // Analyze Groups
            Log("\n--- Analyzing Groups ---");
            int nodeOnlyGroups = 0;
            int geometryOnlyGroups = 0;
            int mixedGroups = 0;
            int totalNodesAnalyzed = 0;
            int totalGeomAnalyzed = 0;

            // Sort groups by ID for consistent output
            var sortedGroupKeys = groupedEntries.Keys.OrderBy(k => k);

            foreach (var groupId in sortedGroupKeys)
            {
                var group = groupedEntries[groupId];
                var nodeEntries = group.Where(e => e.MspiFirstIndex == -1).ToList();
                var geometryEntries = group.Where(e => e.MspiFirstIndex >= 0).ToList();

                totalNodesAnalyzed += nodeEntries.Count;
                totalGeomAnalyzed += geometryEntries.Count;

                Log($"\nGroup 0x{groupId:X8} (Count: {group.Count})");
                Log($"  Node Entries (-1): {nodeEntries.Count}");
                Log($"  Geom Entries (>=0): {geometryEntries.Count}");

                // --- ADDED: Log details for each entry in the group ---
                if (nodeEntries.Any())
                {
                    Log("    --- Node Entries ---");
                    foreach(var entry in nodeEntries)
                    {
                        string skipInfo = entry.SkipReason != null ? $" (Skipped: {entry.SkipReason})" : "";
                        Log($"      Entry {entry.EntryIndex}: Unk00=0x{entry.Unknown_0x00:X2}, Unk01=0x{entry.Unknown_0x01:X2}, Unk10=0x{entry.Unknown_0x10:X4}, Unk12=0x{entry.Unknown_0x12:X4} (Source: {entry.SourceLog}{skipInfo})");
                    }
                }
                if (geometryEntries.Any())
                {
                    Log("    --- Geometry Entries ---");
                    foreach(var entry in geometryEntries)
                    {
                        string skipInfo = entry.SkipReason != null ? $" (Skipped: {entry.SkipReason})" : "";
                        Log($"      Entry {entry.EntryIndex}: First={entry.MspiFirstIndex}, Cnt={entry.MspiIndexCount}, Unk00=0x{entry.Unknown_0x00:X2}, Unk01=0x{entry.Unknown_0x01:X2}, Unk10=0x{entry.Unknown_0x10:X4}, Unk12=0x{entry.Unknown_0x12:X4} (Source: {entry.SourceLog}{skipInfo})");
                    }
                }
                // --- END: Log details ---

                if (nodeEntries.Count > 0 && geometryEntries.Count > 0)
                {
                    mixedGroups++;
                    Log("  Type: Mixed");
                    AnalyzeNodeTypes(nodeEntries, writer);
                }
                else if (nodeEntries.Count > 0)
                {
                    nodeOnlyGroups++;
                    Log("  Type: Node Only");
                     // Optionally analyze node types here too if needed
                     // AnalyzeNodeTypes(nodeEntries, writer);
                }
                else if (geometryEntries.Count > 0)
                {
                    geometryOnlyGroups++;
                    Log("  Type: Geometry Only");
                }
                else
                {
                     Log("  Type: Empty?"); // Should not happen if parsing worked
                }

                // Optional: Print first few entries per group for detail
                // foreach(var entry in group.Take(5)) { Log($"    - Entry {entry.EntryIndex}: First={entry.MspiFirstIndex}, Cnt={entry.MspiIndexCount}, Unk00=0x{entry.Unknown_0x00:X2}, Unk01=0x{entry.Unknown_0x01:X2}, Unk10=0x{entry.Unknown_0x10:X4} ({entry.SourceLog})"); }
                // if(group.Count > 5) { Log("    ..."); }
            }

            // Print Summary
            Log("\n--- Analysis Summary ---");
            Log($"Total Entries Analyzed: {totalNodesAnalyzed + totalGeomAnalyzed} ({totalNodesAnalyzed} Nodes, {totalGeomAnalyzed} Geometry)");
            Log($"Total Groups Found: {groupedEntries.Count}");
            Log($"  Node Only Groups:   {nodeOnlyGroups}");
            Log($"  Geometry Only Groups: {geometryOnlyGroups}");
            Log($"  Mixed Groups:       {mixedGroups}");

            Log("\n--- End of Analysis ---");
        }

        private static void AnalyzeNodeTypes(List<MslkEntryData> nodeEntries, StreamWriter writer)
        {
             Action<string> Log = (message) =>
            {
                Console.WriteLine(message);
                writer.WriteLine(message);
            };

            if (!nodeEntries.Any()) return;

            var unk00Counts = nodeEntries.GroupBy(n => n.Unknown_0x00)
                                         .ToDictionary(g => g.Key, g => g.Count());
            var unk01Counts = nodeEntries.GroupBy(n => n.Unknown_0x01)
                                         .ToDictionary(g => g.Key, g => g.Count());

            Log("    Node Analysis (Mixed Group):");
            string unk00String = string.Join(", ", unk00Counts.Select(kvp => $"0x{kvp.Key:X2}={kvp.Value}"));
            Log($"      Unk00 Counts: {unk00String}");
            string unk01String = string.Join(", ", unk01Counts.Select(kvp => $"0x{kvp.Key:X2}={kvp.Value}"));
            Log($"      Unk01 Counts: {unk01String}");
        }

        private static bool TryParseDebugLogLine(string line, out MslkEntryData entry)
        {
            entry = default;
            var match = DebugLogRegex.Match(line);
            if (!match.Success)
                return false;

            try
            {
                entry = new MslkEntryData
                {
                    EntryIndex = int.Parse(match.Groups["index"].Value),
                    MspiFirstIndex = int.Parse(match.Groups["first"].Value),
                    MspiIndexCount = uint.Parse(match.Groups["count"].Value),
                    Unknown_0x00 = byte.Parse(match.Groups["unk00"].Value, NumberStyles.HexNumber),
                    Unknown_0x01 = byte.Parse(match.Groups["unk01"].Value, NumberStyles.HexNumber),
                    Unknown_0x04 = uint.Parse(match.Groups["unk04"].Value, NumberStyles.HexNumber),
                    Unknown_0x10 = ushort.Parse(match.Groups["unk10"].Value, NumberStyles.HexNumber),
                    Unknown_0x12 = ushort.Parse(match.Groups["unk12"].Value, NumberStyles.HexNumber),
                    SourceLog = "debug",
                    SkipReason = null
                };
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to parse debug log match: {line} - {ex.Message}");
                return false;
            }
        }

         private static bool TryParseSkippedLogLine(string line, out MslkEntryData entry)
        {
            entry = default;
            var match = SkippedLogRegex.Match(line);
            if (!match.Success)
                return false;

            try
            {
                // Parse mandatory fields
                entry = new MslkEntryData
                {
                    EntryIndex = int.Parse(match.Groups["index"].Value),
                    MspiFirstIndex = int.Parse(match.Groups["first"].Value),
                    MspiIndexCount = uint.Parse(match.Groups["count"].Value),
                    Unknown_0x00 = byte.Parse(match.Groups["unk00"].Value, NumberStyles.HexNumber),
                    Unknown_0x01 = byte.Parse(match.Groups["unk01"].Value, NumberStyles.HexNumber),
                    Unknown_0x04 = uint.Parse(match.Groups["unk04"].Value, NumberStyles.HexNumber),
                    Unknown_0x10 = ushort.Parse(match.Groups["unk10"].Value, NumberStyles.HexNumber),
                    SourceLog = "skipped",
                    SkipReason = match.Groups["reason"].Value.Trim()
                };

                // Parse optional Unk12
                if (match.Groups["unk12"].Success)
                {
                    entry.Unknown_0x12 = ushort.Parse(match.Groups["unk12"].Value, NumberStyles.HexNumber);
                }
                else
                {
                    entry.Unknown_0x12 = 0; // Default value if not found
                }

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ERROR] Failed to parse skipped log match: {line} - {ex.Message}");
                return false;
            }
        }
    }
}