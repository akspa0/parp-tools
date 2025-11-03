using System;
using System.Globalization;
using System.IO;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Infrastructure;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Writes a human-readable log that maps each MSLK entry to its associated MSPI index range.
    /// Intended for debugging object assembly in a single PM4 file.
    /// </summary>
    public static class MslkMeshLinkageLogger
    {
        /// <summary>
        /// Creates a text log file enumerating all MSLK entries for <paramref name="pm4Path"/>.
        /// The log lives inside the current <see cref="ProjectOutput"/> run directory so it never
        /// contaminates original data folders.
        /// </summary>
        /// <param name="pm4Path">Path to the source PM4 file.</param>
        /// <param name="outputPath">Destination file.  If <c>null</c>, a path under project_output/linkage/ is auto-chosen.</param>
        public static void Log(string pm4Path, string? outputPath = null)
        {
            if (!File.Exists(pm4Path))
                throw new FileNotFoundException($"PM4 file not found: {pm4Path}");

            // Auto-select output path if caller omitted it
            if (string.IsNullOrEmpty(outputPath))
            {
                string fileStem = Path.GetFileNameWithoutExtension(pm4Path);
                outputPath = ProjectOutput.GetPath("linkage", fileStem + "_mslk_linkage.txt");
            }

            // Ensure folder exists
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);

            var pm4 = PM4File.FromFile(pm4Path);
            if (pm4.MSLK == null)
            {
                File.WriteAllText(outputPath!, "(No MSLK chunk present – nothing to log)");
                return;
            }

            using var writer = new StreamWriter(outputPath!, false);
            writer.WriteLine($"# MSLK linkage log for {Path.GetFileName(pm4Path)}");
            writer.WriteLine($"# Generated {DateTime.UtcNow:u}\n");
            writer.WriteLine("Idx | GroupKey(0x04) | FirstIndex | Count | Unk00 | Unk01 | Unk04 | Unk10 | Unk12 | FaceRange (triangles)");
            writer.WriteLine("----|----------|-----------:|------:|------:|------:|-------:|-------:|-------:|----------------------");

            for (int i = 0; i < pm4.MSLK.Entries.Count; i++)
            {
                var entry = pm4.MSLK.Entries[i];

                // The FirstIndex + Count refer to MSPI indices.  Three indices make one triangle.
                int first = entry.MspiFirstIndex;
                int count = entry.MspiIndexCount; // byte to int
                int triFirst = first >= 0 ? first / 3 : -1;
                int triLast = first >= 0 ? (first + count - 1) / 3 : -1;

                writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0,3} | 0x{1:X8} | {2,10} | {3,5} | 0x{4:X2} | 0x{5:X2} | 0x{6:X8} | 0x{7:X4} | 0x{8:X4} | {9}",
                    i,
                    entry.Unknown_0x04,
                    entry.MspiFirstIndex,
                    entry.MspiIndexCount,
                    entry.Unknown_0x00,
                    entry.Unknown_0x01,
                    entry.Unknown_0x04,
                    entry.Unknown_0x10,
                    entry.Unknown_0x12,
                    first >= 0 ? $"{triFirst}–{triLast}" : "(n/a)"));
            }

            writer.Flush();
        }
    }
}
