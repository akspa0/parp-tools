using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration.Attributes;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class AnalyzeLinksTool
    {
        public void Analyze(PM4File pm4File, string outputDirectory, string baseFileName)
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            ProcessMslkChunk(pm4File, outputDirectory, baseFileName);
            ProcessMsurChunk(pm4File, outputDirectory, baseFileName);
            ProcessMprlChunk(pm4File, outputDirectory, baseFileName);
            ProcessMprrChunk(pm4File, outputDirectory, baseFileName);
        }

        private void ProcessMslkChunk(PM4File pm4File, string outputDirectory, string baseFileName)
        {
            if (pm4File.MSLK?.Entries == null || pm4File.MSLK.Entries.Count == 0) return;

            var records = pm4File.MSLK.Entries.Select((entry, index) => new MslkLinkDto
            {
                Index = index,
                Unknown_0x00 = $"0x{entry.Unknown_0x00:X2}",
                Unknown_0x04 = $"0x{entry.GroupObjectId:X8}",
                MspiFirstIndex = (uint)entry.MspiFirstIndex,
                Unknown_0x0C = $"0x{entry.Unknown_0x0C:X8}",
                Reference = $"{entry.RefHighByte:X2}{entry.RefLowByte:X2}",
                LinkId = $"0x{entry.LinkIdRaw:X8}"
            });

            var outputPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk.csv");
            WriteToCsv(records, outputPath);
        }

        private void ProcessMsurChunk(PM4File pm4File, string outputDirectory, string baseFileName)
        {
            if (pm4File.MSUR?.Entries == null || pm4File.MSUR.Entries.Count == 0) return;

            var records = pm4File.MSUR.Entries.Select((entry, index) => new MsurLinkDto
            {
                Index = index,
                FlagsOrUnknown_0x00 = $"0x{entry.FlagsOrUnknown_0x00:X2}",
                IndexCount = entry.IndexCount,
                Unknown_0x02 = $"0x{entry.Unknown_0x02:X2}",
                SurfaceNormalX = entry.SurfaceNormalX,
                SurfaceNormalY = entry.SurfaceNormalY,
                SurfaceNormalZ = entry.SurfaceNormalZ,
                SurfaceHeight = entry.SurfaceHeight,
                MsviFirstIndex = entry.MsviFirstIndex,
                MdosIndex = entry.MdosIndex,
                Unknown_0x1C = $"0x{entry.Unknown_0x1C:X8}"
            });

            var outputPath = Path.Combine(outputDirectory, $"{baseFileName}_msur.csv");
            WriteToCsv(records, outputPath);
        }

        private void ProcessMprlChunk(PM4File pm4File, string outputDirectory, string baseFileName)
        {
            if (pm4File.MPRL?.Entries == null || pm4File.MPRL.Entries.Count == 0) return;

            var records = pm4File.MPRL.Entries.Select((entry, index) => new MprlLinkDto
            {
                Index = index,
                Unknown_0x00 = entry.Unknown_0x00,
                Unknown_0x02 = entry.Unknown_0x02,
                Unknown_0x04 = entry.Unknown_0x04,
                Unknown_0x06 = entry.Unknown_0x06,
                PosX = entry.Position.X,
                PosY = entry.Position.Y,
                PosZ = entry.Position.Z,
                Unknown_0x14 = entry.Unknown_0x14,
                Unknown_0x16 = entry.Unknown_0x16
            });

            var outputPath = Path.Combine(outputDirectory, $"{baseFileName}_mprl.csv");
            WriteToCsv(records, outputPath);
        }

        private void ProcessMprrChunk(PM4File pm4File, string outputDirectory, string baseFileName)
        {
            if (pm4File.MPRR?.Sequences == null || pm4File.MPRR.Sequences.Count == 0) return;

            var records = new List<MprrLinkDto>();
            for (int i = 0; i < pm4File.MPRR.Sequences.Count; i++)
            {
                var sequence = pm4File.MPRR.Sequences[i];
                for (int j = 0; j < sequence.Count; j++)
                {
                    records.Add(new MprrLinkDto
                    {
                        SequenceIndex = i,
                        ValueIndex = j,
                        Value = $"0x{sequence[j]:X4}"
                    });
                }
            }

            var outputPath = Path.Combine(outputDirectory, $"{baseFileName}_mprr.csv");
            WriteToCsv(records, outputPath);
        }

        private void WriteToCsv<T>(IEnumerable<T> records, string filePath)
        {
            using var writer = new StreamWriter(filePath, false, new UTF8Encoding(true));
            using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
            csv.WriteRecords(records);
        }
    }

    public class MslkLinkDto
    {
        [Index(0)]
        public int Index { get; set; }
        [Index(1)]
        public string Unknown_0x00 { get; set; } = string.Empty;
        [Index(2)]
        public string Unknown_0x04 { get; set; } = string.Empty;
        [Index(3)]
        public uint MspiFirstIndex { get; set; }
        [Index(4)]
        public string Unknown_0x0C { get; set; } = string.Empty;
        [Index(5)]
        public string Reference { get; set; } = string.Empty;
        [Index(6)]
        public string LinkId { get; set; } = string.Empty;
    }

    public class MsurLinkDto
    {
        [Index(0)]
        public int Index { get; set; }
        [Index(1)]
        public string FlagsOrUnknown_0x00 { get; set; } = string.Empty;
        [Index(2)]
        public byte IndexCount { get; set; }
        [Index(3)]
        public string Unknown_0x02 { get; set; } = string.Empty;
        [Index(4)]
        public float SurfaceNormalX { get; set; }
        [Index(5)]
        public float SurfaceNormalY { get; set; }
        [Index(6)]
        public float SurfaceNormalZ { get; set; }
        [Index(7)]
        public float SurfaceHeight { get; set; }
        [Index(8)]
        public uint MsviFirstIndex { get; set; }
        [Index(9)]
        public uint MdosIndex { get; set; }
        [Index(10)]
        public string Unknown_0x1C { get; set; } = string.Empty;
    }

    public class MprlLinkDto
    {
        [Index(0)]
        public int Index { get; set; }
        [Index(1)]
        public ushort Unknown_0x00 { get; set; }
        [Index(2)]
        public short Unknown_0x02 { get; set; }
        [Index(3)]
        public ushort Unknown_0x04 { get; set; }
        [Index(4)]
        public ushort Unknown_0x06 { get; set; }
        [Index(5)]
        public float PosX { get; set; }
        [Index(6)]
        public float PosY { get; set; }
        [Index(7)]
        public float PosZ { get; set; }
        [Index(8)]
        public short Unknown_0x14 { get; set; }
        [Index(9)]
        public ushort Unknown_0x16 { get; set; }
    }

    public class MprrLinkDto
    {
        [Index(0)]
        public int SequenceIndex { get; set; }
        [Index(1)]
        public int ValueIndex { get; set; }
        [Index(2)]
        public string Value { get; set; } = string.Empty;
    }
}
