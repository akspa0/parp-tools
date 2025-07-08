using System;
using System.Globalization;
using System.IO;
using System.Text;
using System.Numerics;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Helper for exploring relationships between MSLK, MSPV, MSCN and other PM4 chunks.
    /// Produces a CSV dump suitable for spreadsheet / statistical analysis.
    /// </summary>
    public static class MsLkDiagnostics
    {
        /// <summary>
        /// Writes a CSV file listing all <see cref="MSLKEntry"/> records and basic derived information.
        /// </summary>
        /// <param name="pm4File">Loaded <see cref="PM4File"/>.</param>
        /// <param name="outputCsvPath">Destination file path.</param>
        public static void DumpEntryCsv(PM4File pm4File, string outputCsvPath)
        {
            if (pm4File == null) throw new ArgumentNullException(nameof(pm4File));
            if (string.IsNullOrWhiteSpace(outputCsvPath))
                throw new ArgumentException("Output path must be provided", nameof(outputCsvPath));

            var mslk = pm4File.MSLK ?? throw new InvalidOperationException("PM4 file does not contain an MSLK chunk.");

            // If all entries have zero indices, skip writing – no actionable data
            if (mslk.Entries.All(ent => ent.MspiIndexCount == 0))
            {
                return; // nothing useful
            }

            // MSCN is optional – treat as zero if missing
            var mscn = pm4File.MSCN;
            var sb = new StringBuilder();

            // CSV header (renamed MaterialColorId→LinkId and added EntryType,HasMscnSlice)
            // CSV header: only raw fields from MSLK (plus decoded LinkId high/low words)
            sb.AppendLine("Index,ObjectTypeFlags,ObjectSubtype,MspiFirstIndex,MspiIndexCount,GroupObjectId,LinkIdHex,ReferenceIndex,Unknown12");

            for (int i = 0; i < mslk.Entries.Count; i++)
            {
                var e = mslk.Entries[i];
                int exteriorCount = 0;
                // Very rough heuristic: many files arrange MSCN verts sequentially per entry matching MspiIndexCount
                // but we don't know the offset mapping yet – leave zero and refine later.
                if (mscn != null)
                {
                    // TODO: find actual mapping – placeholder only
                    exteriorCount = mscn.ExteriorVertices.Count;
                }

                sb.AppendLine(string.Join(',',
                    i,
                    e.ObjectTypeFlags,
                    e.ObjectSubtype,
                    e.MspiFirstIndex,
                    e.MspiIndexCount,
                    e.GroupObjectId,
                    e.MaterialColorId.ToString("X8"),
                    e.ReferenceIndex,
                    e.Unk12));
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputCsvPath)!);
            File.WriteAllText(outputCsvPath, sb.ToString(), Encoding.UTF8);

            // ----- MPRR SUMMARY & CONNECTIVITY -----
            var mprr = pm4File.MPRR;
            int mprrSeqCount = mprr?.Sequences.Count ?? 0;
            int mprrEdgeCount = 0;
            int mprrValidEdgeCount = 0;
            if (mprr != null)
            {
                foreach (var seq in mprr.Sequences)
                {
                    if (seq.Count < 2) continue;
                    ushort from = seq[0];
                    for (int k = 1; k < seq.Count - 1; k++) // exclude terminator
                    {
                        ushort to = seq[k];
                        mprrEdgeCount++;
                        if (from < mslk.Entries.Count && to < mslk.Entries.Count)
                            mprrValidEdgeCount++;
                    }
                }
            }




        }
    }
}
