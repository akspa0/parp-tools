using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Diagnostic helper that dumps key identifier fields from MSLK entries and MSUR surfaces
    /// to a CSV so we can visually inspect potential linkage between render mesh chunks and the scene graph.
    /// </summary>
    public static class MslkLinkScanner
    {
        private const string CsvHeader = "Chunk,Index,Flag,GroupId,LinkPad,LinkSubHigh,LinkSubLow,RefHigh,RefLow,MsviFirst,IndexCount,Unk1CHigh,Unk1CLow";

        public static void Dump(string pm4Path, string csvOutPath)
        {
            var pm4 = PM4File.FromFile(pm4Path);
            using var writer = new StreamWriter(csvOutPath);
            writer.WriteLine(CsvHeader);

            // --- Dump MSLK
            if (pm4.MSLK?.Entries != null)
            {
                for (int i = 0; i < pm4.MSLK.Entries.Count; i++)
                {
                    var e = pm4.MSLK.Entries[i];
                    writer.WriteLine(string.Join(',',
                        "MSLK",
                        i,
                        e.Unknown_0x00.ToString("X2"),
                        e.Unknown_0x04.ToString("X8"),
                        e.LinkPadWord.ToString("X4"),
                        e.LinkSubHighByte.ToString("X2"),
                        e.LinkSubLowByte.ToString("X2"),
                        e.RefHighByte.ToString("X2"),
                        e.RefLowByte.ToString("X2"),
                        string.Empty,
                        string.Empty,
                        string.Empty));
                }
            }

            // --- Dump MSUR (render surfaces)
            if (pm4.MSUR?.Entries != null)
            {
                for (int i = 0; i < pm4.MSUR.Entries.Count; i++)
                {
                    var s = pm4.MSUR.Entries[i];
                    writer.WriteLine(string.Join(',',
                        "MSUR",
                        i,
                        s.FlagsOrUnknown_0x00.ToString("X2"),
                        string.Empty,
                        string.Empty,
                        string.Empty,
                        string.Empty,
                        string.Empty,
                        string.Empty,
                        s.MsviFirstIndex.ToString(),
                        s.IndexCount.ToString(),
                        ((s.Unknown_0x1C >> 16) & 0xFFFF).ToString("X4"),
                        (s.Unknown_0x1C & 0xFFFF).ToString("X4")));
                }
            }
            writer.Flush();
        }
    }
}
