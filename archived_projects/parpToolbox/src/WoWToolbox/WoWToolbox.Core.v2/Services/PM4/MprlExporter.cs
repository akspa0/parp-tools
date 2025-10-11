using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using WoWToolbox.Core.v2.Services.Export;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class MprlExporter : IMprlExporter
    {
        private readonly ICoordinateService _coordinateService;

        public MprlExporter(ICoordinateService coordinateService)
        {
            _coordinateService = coordinateService;
        }

        public void Export(PM4File pm4File, string inputFilePath, string outputPath)
        {
            if (pm4File.MPRL == null || pm4File.MPRL.Entries.Count == 0)
            {
                // Optionally, create an empty file with a header to indicate no data was present.
                File.WriteAllText(outputPath, $"# PM4 MPRL Points - No data found in file: {Path.GetFileName(inputFilePath)}\n");
                return;
            }

            using var writer = new StreamWriter(outputPath, false);

            writer.WriteLine($"# PM4 MPRL Points (X, -Z, Y) - File: {Path.GetFileName(inputFilePath)} (Generated: {DateTime.Now})");
            writer.WriteLine("o MPRL_Points");

            int mprlFileVertexCount = 0;
            foreach (var entry in pm4File.MPRL.Entries)
            {
                var pm4Coords = _coordinateService.FromMprlEntry(entry);

                string comment = $"# MPRLIdx=[{mprlFileVertexCount}] " +
                                 $"Unk00={entry.Unknown_0x00} Unk02={entry.Unknown_0x02} " +
                                 $"Unk04={entry.Unknown_0x04} Unk06={entry.Unknown_0x06} " +
                                 $"Unk14={entry.Unknown_0x14} Unk16={entry.Unknown_0x16}";

                writer.WriteLine(FormattableString.Invariant($"v {pm4Coords.X:F6} {pm4Coords.Y:F6} {pm4Coords.Z:F6} {comment}"));
                mprlFileVertexCount++;
            }

            writer.WriteLine(); // Final blank line for formatting
        }
    }
}
