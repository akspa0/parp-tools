using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MFED chunk - Map Object Fog Extra Data
    /// Contains additional fog data, added in 9.0.1.33978
    /// </summary>
    public class MFED : IChunk
    {
        /// <summary>
        /// Gets the list of fog extra data entries.
        /// </summary>
        public List<FogExtraData> FogExtraDataList { get; } = new();

        public class FogExtraData
        {
            /// <summary>
            /// Gets or sets the doodad set ID.
            /// </summary>
            public ushort DoodadSetId { get; set; }

            /// <summary>
            /// Gets or sets the unknown data.
            /// </summary>
            public byte[] UnknownData { get; set; } = new byte[14];
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each fog extra data entry is 16 bytes:
            // - 1 ushort (doodad set id) = 2 bytes
            // - 14 bytes of unknown data
            var entryCount = (int)size / 16;

            // Clear existing data
            FogExtraDataList.Clear();

            // Read fog extra data entries
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new FogExtraData
                {
                    DoodadSetId = reader.ReadUInt16(),
                    UnknownData = reader.ReadBytes(14)
                };

                FogExtraDataList.Add(entry);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var entry in FogExtraDataList)
            {
                writer.Write(entry.DoodadSetId);
                writer.Write(entry.UnknownData);
            }
        }

        /// <summary>
        /// Gets fog extra data for a specific doodad set.
        /// </summary>
        /// <param name="doodadSetId">ID of the doodad set.</param>
        /// <returns>Fog extra data if found, null otherwise.</returns>
        public FogExtraData GetFogExtraDataByDoodadSet(ushort doodadSetId)
        {
            return FogExtraDataList.Find(e => e.DoodadSetId == doodadSetId);
        }

        /// <summary>
        /// Validates fog extra data count against MFOG.
        /// </summary>
        /// <param name="mfogCount">Number of fog entries in MFOG chunk.</param>
        /// <returns>True if counts match, false otherwise.</returns>
        public bool ValidateFogCount(int mfogCount)
        {
            return FogExtraDataList.Count == mfogCount;
        }

        /// <summary>
        /// Gets a validation report for the fog extra data.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MFED Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Fog Extra Data Entries: {FogExtraDataList.Count}");
            report.AppendLine();

            // Analyze entries
            var uniqueDoodadSets = new HashSet<ushort>();
            var uniqueUnknownDataPatterns = new HashSet<string>(System.StringComparer.Ordinal);

            foreach (var entry in FogExtraDataList)
            {
                uniqueDoodadSets.Add(entry.DoodadSetId);
                uniqueUnknownDataPatterns.Add(System.BitConverter.ToString(entry.UnknownData));
            }

            report.AppendLine($"Unique Doodad Sets: {uniqueDoodadSets.Count}");
            report.AppendLine($"Unique Unknown Data Patterns: {uniqueUnknownDataPatterns.Count}");

            if (uniqueUnknownDataPatterns.Count < 10)
            {
                report.AppendLine();
                report.AppendLine("Unknown Data Patterns:");
                foreach (var pattern in uniqueUnknownDataPatterns)
                {
                    var count = FogExtraDataList.Count(e => System.BitConverter.ToString(e.UnknownData) == pattern);
                    report.AppendLine($"  {pattern}: {count} entries");
                }
            }

            return report.ToString();
        }
    }
} 