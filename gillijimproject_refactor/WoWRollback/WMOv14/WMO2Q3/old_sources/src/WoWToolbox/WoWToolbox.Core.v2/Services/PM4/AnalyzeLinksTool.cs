using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using CsvHelper;
using CsvHelper.Configuration.Attributes;
using WoWToolbox.Core.v2.Utilities;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;

namespace WoWToolbox.Core.v2.Services.PM4
{
    public class AnalyzeLinksTool
    {
        private readonly HashSet<ushort> _routeReferences = new();
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
                Unknown_0x00 = entry.Unknown_0x00,
                Unknown_0x04 = entry.GroupObjectId,
                MspiFirstIndex = (uint)entry.MspiFirstIndex,
                Unknown_0x0C = entry.Unknown_0x0C,
                Reference = (ushort)((entry.RefHighByte << 8) | entry.RefLowByte),
                LinkId = entry.LinkIdRaw
            }).ToList();

            var outputPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk.csv");
            WriteToCsv(records, outputPath);

            // treat MspiFirstIndex as additional reference candidates (adjacent geometry hypothesis)
            foreach (var rec in records)
            {
                if (rec.MspiFirstIndex != uint.MaxValue && rec.MspiFirstIndex <= ushort.MaxValue)
                {
                    _routeReferences.Add((ushort)rec.MspiFirstIndex);
                }
            }

            // exploratory summary by flag/group
            var summary = records.GroupBy(r => new { r.Unknown_0x00, r.Unknown_0x04 })
                                 .Select(g => new MslkSummaryDto
                                 {
                                     Unknown_0x00 = g.Key.Unknown_0x00,
                                     Unknown_0x04 = g.Key.Unknown_0x04,
                                     Count = g.Count()
                                 }).ToList();
            var summaryPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_summary.csv");
            WriteToCsv(summary, summaryPath);

            // derive list of links not present in any route (MPRR, special MPRL portal refs, or MSPI indices)
            var unused = records.Where(r => !_routeReferences.Contains(r.Reference)).ToList();

            // build missing-ref set for quick triage
            var mspiCount = pm4File.MSPI?.Indices?.Count ?? 0;
            // build MSUR index set
            var msurIndexSet = new HashSet<ushort>();
            if (pm4File.MSUR?.Entries != null)
            {
                foreach (var surf in pm4File.MSUR.Entries)
                {
                    uint start = surf.MsviFirstIndex;
                    for (uint i = 0; i < surf.IndexCount; i++)
                    {
                        uint idx = start + i;
                        if (idx <= ushort.MaxValue)
                            msurIndexSet.Add((ushort)idx);
                    }
                }
            }

            var enrichedMissing = unused.Select(u => u.Reference).Distinct().OrderBy(v => v)
                .Select(v => new EnrichedMissingRefDto
                {
                    Reference = v,
                    ExternalCandidate = v >= mspiCount,
                    InMsurRange = msurIndexSet.Contains(v)
                }).ToList();
            if (enrichedMissing.Any())
            {
                var missingPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_missing_refs_enriched.csv");
                WriteToCsv(enrichedMissing, missingPath);

                // produce non-benign distinct list (noise reduction)
                var nonBenign = enrichedMissing.Where(e => !(e.InMsurRange && !e.ExternalCandidate)).ToList();
                if (nonBenign.Any())
                {
                    var nbPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_missing_refs_distinct.csv");
                    WriteToCsv(nonBenign, nbPath);
                }
            }

            // filtered view: refs whose originating MSLK flag is NOT 2
            var filtered = unused.Where(r => r.Unknown_0x00 != 2)
                                  .Select(r => new MissingRefWithFlagDto
                                  {
                                      Reference = r.Reference,
                                      Flag = r.Unknown_0x00,
                                      GroupId = r.Unknown_0x04
                                  }).ToList();
            if (filtered.Any())
            {
                var filtPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_missing_refs_filtered.csv");
                WriteToCsv(filtered, filtPath);
            }

            // summary per reference/flag/group with occurrence count
            var missingSummary = unused.GroupBy(r => new { r.Reference, r.Unknown_0x00, r.Unknown_0x04 })
                                 .Select(g => new MissingRefSummaryDto
                                 {
                                     Reference = g.Key.Reference,
                                     Flag = g.Key.Unknown_0x00,
                                     GroupId = g.Key.Unknown_0x04,
                                     Occurrences = g.Count()
                                 }).OrderBy(s => s.Reference).ToList();
            if (missingSummary.Any())
            {
                var sumPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_missing_refs_summary.csv");
                WriteToCsv(missingSummary, sumPath);
            }

            // flag pivot counts
            var flagPivot = unused.GroupBy(r => r.Unknown_0x00)
                                   .Select(g => new FlagPivotDto { Flag = g.Key, Count = g.Count() })
                                   .OrderBy(p => p.Flag).ToList();
            if (flagPivot.Any())
            {
                var pivotPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_missing_refs_flagpivot.csv");
                WriteToCsv(flagPivot, pivotPath);
            }

            if (unused.Count > 0)
            {
                var unusedPath = Path.Combine(outputDirectory, $"{baseFileName}_mslk_unreferenced.csv");
                WriteToCsv(unused, unusedPath);
            }

            // ---------------- Cross-tile link analysis ----------------
            var srcCoords = LinkIdDecoder.TryExtractTileCoords(baseFileName);
            if (srcCoords != null)
            {
                var crossTiles = new List<CrossTileLinkDto>();
                foreach (var rec in records)
                {
                    if (LinkIdDecoder.TryDecode(rec.LinkId, out int tgtX, out int tgtY))
                    {
                        if (tgtX != srcCoords.Value.x || tgtY != srcCoords.Value.y)
                        {
                            crossTiles.Add(new CrossTileLinkDto
                            {
                                SourceTileX = srcCoords.Value.x,
                                SourceTileY = srcCoords.Value.y,
                                Reference = rec.Reference,
                                TargetTileX = tgtX,
                                TargetTileY = tgtY,
                                Flag = rec.Unknown_0x00,
                                GroupId = rec.Unknown_0x04
                            });
                        }
                    }
                }
                if (crossTiles.Any())
                {
                    var ctPath = Path.Combine(outputDirectory, $"{baseFileName}_cross_tile_links.csv");
                    WriteToCsv(crossTiles, ctPath);

                    // simple pivot for quick counts by source tile & flag
                    var ctPivot = crossTiles.GroupBy(c => new { c.SourceTileX, c.SourceTileY, c.Flag })
                                             .Select(g => new CrossTilePivotDto
                                             {
                                                 SourceTileX = g.Key.SourceTileX,
                                                 SourceTileY = g.Key.SourceTileY,
                                                 Flag = g.Key.Flag,
                                                 Count = g.Count()
                                             }).ToList();
                    var pivotPath = Path.Combine(outputDirectory, $"{baseFileName}_cross_tile_pivot.csv");
                    WriteToCsv(ctPivot, pivotPath);
                }
            }
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
            }).ToList();

            // collect candidate references where _0x02 == -1
            foreach (var e in pm4File.MPRL.Entries)
            {
                if (e.Unknown_0x02 == unchecked((short)0xFFFF))
                {
                    _routeReferences.Add((ushort)e.Unknown_0x04);
                }
            }

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
                    ushort val = sequence[j];
                    records.Add(new MprrLinkDto { SequenceIndex = i, ValueIndex = j, Value = val });
                    _routeReferences.Add(val);
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
        public byte Unknown_0x00 { get; set; }
        [Index(2)]
        public uint Unknown_0x04 { get; set; }
        [Index(3)]
        public uint MspiFirstIndex { get; set; }
        [Index(4)]
        public uint Unknown_0x0C { get; set; }
        [Index(5)]
        public ushort Reference { get; set; }
        [Index(6)]
        public uint LinkId { get; set; }
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

    public class MissingRefSummaryDto
    {
        [Index(0)] public ushort Reference { get; set; }
        [Index(1)] public byte Flag { get; set; }
        [Index(2)] public uint GroupId { get; set; }
        [Index(3)] public int Occurrences { get; set; }
    }

    public class FlagPivotDto
    {
        [Index(0)] public byte Flag { get; set; }
        [Index(1)] public int Count { get; set; }
    }

    public class MissingRefWithFlagDto
    {
        [Index(0)] public ushort Reference { get; set; }
        [Index(1)] public byte Flag { get; set; }
        [Index(2)] public uint GroupId { get; set; }
    }

    public class CrossTileLinkDto
    {
        [Index(0)] public int SourceTileX { get; set; }
        [Index(1)] public int SourceTileY { get; set; }
        [Index(2)] public ushort Reference { get; set; }
        [Index(3)] public int TargetTileX { get; set; }
        [Index(4)] public int TargetTileY { get; set; }
        [Index(5)] public byte Flag { get; set; }
        [Index(6)] public uint GroupId { get; set; }
    }

    public class CrossTilePivotDto
    {
        [Index(0)] public int SourceTileX { get; set; }
        [Index(1)] public int SourceTileY { get; set; }
        [Index(2)] public byte Flag { get; set; }
        [Index(3)] public int Count { get; set; }
    }

    public class EnrichedMissingRefDto
    {
        [Index(0)] public ushort Reference { get; set; }
        [Index(1)] public bool ExternalCandidate { get; set; }
        [Index(2)] public bool InMsurRange { get; set; }
    }

    // legacy simple DTO remains if needed
    public class MissingRefDto
    {
        [Index(0)] public ushort Reference { get; set; }
        [Index(1)] public bool ExternalCandidate { get; set; }
    }

    public class MslkSummaryDto
    {
        [Index(0)]
        public byte Unknown_0x00 { get; set; }
        [Index(1)]
        public uint Unknown_0x04 { get; set; }
        [Index(2)]
        public int Count { get; set; }
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
        public ushort Value { get; set; }
    }
}
