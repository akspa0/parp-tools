using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Performs structural audits of MSLK linkage within a single PM4 file or across many files.
    /// Focuses on validating ReferenceIndex hierarchy inside each LinkIdRaw group.
    /// </summary>
    public static class MslkStructureAuditor
    {
        public sealed record AuditRow(string Tile, uint LinkId, int GroupSize, int RootCount, int InvalidRefCount, bool HasCycles);

        public static IEnumerable<AuditRow> Audit(PM4File file, string tileName)
        {
            if (file.MSLK == null || file.MSLK.Entries.Count == 0) yield break;

            // Build groups by LinkIdRaw
            var groups = file.MSLK.Entries
                .Select((e, idx) => (Entry: e, Index: idx))
                .GroupBy(t => t.Entry.LinkIdRaw);

            foreach (var g in groups)
            {
                var entries = g.ToList();
                int groupSize = entries.Count;
                int rootCount = entries.Count(t => IsRoot(t.Entry));

                // build quick lookup by local index
                var dict = entries.ToDictionary(t => t.Index, t => t.Entry);

                // invalid references
                int invalid = entries.Count(t => !IsRoot(t.Entry) && !dict.ContainsKey(t.Entry.ReferenceIndex));

                // cycle check via simple visited set DFS
                bool hasCycles = false;
                foreach (var node in entries)
                {
                    var visited = new HashSet<int>();
                    int current = node.Index;
                    while (true)
                    {
                        var entry = dict[current];
                        if (IsRoot(entry)) break;
                        int next = entry.ReferenceIndex;
                        if (!dict.ContainsKey(next)) break; // invalid already counted
                        if (!visited.Add(next)) { hasCycles = true; break; }
                        current = next;
                    }
                    if (hasCycles) break;
                }

                yield return new AuditRow(tileName, g.Key, groupSize, rootCount, invalid, hasCycles);
            }
        }

        private static bool IsRoot(MSLKEntry entry) => entry.ReferenceIndex == unchecked((ushort)0xFFFF);

        public static void WriteCsv(IEnumerable<AuditRow> rows, string csvPath)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(csvPath)!);
            using var sw = new StreamWriter(csvPath);
            sw.WriteLine("Tile,LinkIdHex,GroupSize,RootCount,InvalidRefCount,HasCycles");
            foreach (var r in rows)
            {
                sw.WriteLine($"{r.Tile},0x{r.LinkId:X8},{r.GroupSize},{r.RootCount},{r.InvalidRefCount},{r.HasCycles}");
            }
        }
    }
}
