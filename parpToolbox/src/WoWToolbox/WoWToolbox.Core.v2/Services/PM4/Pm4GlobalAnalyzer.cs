using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using CsvHelper;
using CsvHelper.Configuration;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Models.PM4.Chunks;
using WoWToolbox.Core.v2.Utilities;

namespace WoWToolbox.Core.v2.Services.PM4
{
    /// <summary>
    /// Aggregates data from all PM4 files under a root directory into a single CSV for
    /// holistic analysis. Currenly focuses on MSLK entries, exposing decoded LinkId fields.
    /// </summary>
    public class Pm4GlobalAnalyzer
    {
        private readonly IPm4FileLoader _loader;

        public Pm4GlobalAnalyzer(IPm4FileLoader loader)
        {
            _loader = loader ?? throw new ArgumentNullException(nameof(loader));
        }

        public void GenerateMonolithicMslkCsv(string rootFolder, string outputCsvPath)
        {
            if (string.IsNullOrWhiteSpace(rootFolder) || !Directory.Exists(rootFolder))
                throw new DirectoryNotFoundException($"Root folder '{rootFolder}' not found.");

            var pm4Paths = Directory.EnumerateFiles(rootFolder, "*.pm4", SearchOption.AllDirectories).ToList();
            Console.WriteLine($"GlobalAnalyzer: Found {pm4Paths.Count} PM4 files under '{rootFolder}'.");

            var records = new List<MslkCsvRow>(capacity: 500_000); // pre-size for speed

            foreach (var path in pm4Paths)
            {
                PM4File? pm4;
                try { pm4 = _loader.Load(path); }
                catch (Exception ex)
                {
                    Console.WriteLine($"Failed to load PM4 '{path}': {ex.Message}");
                    continue;
                }

                if (pm4?.MSLK == null) continue;

                for (int i = 0; i < pm4.MSLK.Entries.Count; i++)
                {
                    var e = pm4.MSLK.Entries[i];
                    ushort hi = (ushort)(e.MaterialColorId >> 16);
                    LinkIdDecoder.TryDecode(e.MaterialColorId, out int tx, out int ty);

                    records.Add(new MslkCsvRow
                    {
                        FileName = Path.GetFileName(path),
                        EntryIndex = i,
                        TileX = tx,
                        TileY = ty,
                        HighWord = $"0x{hi:X4}",
                        ObjectType = e.ObjectTypeFlags,
                        SubType = e.ObjectSubtype,
                        GroupId = e.GroupObjectId,
                        ReferenceIndex = e.ReferenceIndex,
                        FlagsHex = $"0x{e.Unknown_0x12:X4}",
                        HasSlice = e.MspiIndexCount > 0
                    });
                }
            }

            Console.WriteLine($"Writing {records.Count} aggregated rows to '{outputCsvPath}'.");
            var cfg = new CsvConfiguration(CultureInfo.InvariantCulture) { HasHeaderRecord = true };
            using var writer = new StreamWriter(outputCsvPath);
            using var csv = new CsvWriter(writer, cfg);
            csv.WriteRecords(records);
        }

        private sealed class MslkCsvRow
        {
            public string FileName { get; set; } = string.Empty;
            public int EntryIndex { get; set; }
            public int TileX { get; set; }
            public int TileY { get; set; }
            public string HighWord { get; set; } = string.Empty;
            public byte ObjectType { get; set; }
            public byte SubType { get; set; }
            public uint GroupId { get; set; }
            public ushort ReferenceIndex { get; set; }
            public string FlagsHex { get; set; } = string.Empty;
            public bool HasSlice { get; set; }
        }
    }
}
