using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MGI2 chunk - Map Object Group Info v2
    /// Contains additional group information, added in 9.0.1.33978
    /// </summary>
    public class MGI2 : IChunk
    {
        /// <summary>
        /// Gets the list of group info v2 entries.
        /// </summary>
        public List<GroupInfoV2> GroupInfoV2List { get; } = new();

        public class GroupInfoV2
        {
            /// <summary>
            /// Gets or sets the flags2 value.
            /// </summary>
            public uint Flags2 { get; set; }

            /// <summary>
            /// Gets or sets the LOD index.
            /// </summary>
            public uint LodIndex { get; set; }
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each group info v2 entry is 8 bytes:
            // - 1 uint (flags2) = 4 bytes
            // - 1 uint (lod index) = 4 bytes
            var entryCount = (int)size / 8;

            // Clear existing data
            GroupInfoV2List.Clear();

            // Read group info v2 entries
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new GroupInfoV2
                {
                    Flags2 = reader.ReadUInt32(),
                    LodIndex = reader.ReadUInt32()
                };

                GroupInfoV2List.Add(entry);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var entry in GroupInfoV2List)
            {
                writer.Write(entry.Flags2);
                writer.Write(entry.LodIndex);
            }
        }

        /// <summary>
        /// Gets group info v2 for a specific group index.
        /// </summary>
        /// <param name="groupIndex">Index of the group.</param>
        /// <returns>Group info v2 if found, null otherwise.</returns>
        public GroupInfoV2 GetGroupInfoV2(int groupIndex)
        {
            if (groupIndex < 0 || groupIndex >= GroupInfoV2List.Count)
                return null;

            return GroupInfoV2List[groupIndex];
        }

        /// <summary>
        /// Validates group info v2 count against MOGI.
        /// </summary>
        /// <param name="mogiCount">Number of groups in MOGI chunk.</param>
        /// <returns>True if counts match, false otherwise.</returns>
        public bool ValidateGroupCount(int mogiCount)
        {
            return GroupInfoV2List.Count == mogiCount;
        }

        /// <summary>
        /// Gets a validation report for the group info v2 entries.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MGI2 Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Group Info V2 Entries: {GroupInfoV2List.Count}");
            report.AppendLine();

            // Analyze entries
            var uniqueFlags2 = new HashSet<uint>();
            var uniqueLodIndices = new HashSet<uint>();
            var groupsWithFlags2 = 0;
            var groupsWithLod = 0;

            foreach (var entry in GroupInfoV2List)
            {
                if (entry.Flags2 != 0)
                {
                    uniqueFlags2.Add(entry.Flags2);
                    groupsWithFlags2++;
                }
                if (entry.LodIndex != 0)
                {
                    uniqueLodIndices.Add(entry.LodIndex);
                    groupsWithLod++;
                }
            }

            report.AppendLine($"Groups with Flags2: {groupsWithFlags2}");
            report.AppendLine($"Unique Flags2 Values: {uniqueFlags2.Count}");
            if (uniqueFlags2.Count > 0)
            {
                report.AppendLine();
                report.AppendLine("Flags2 Values:");
                foreach (var flag in uniqueFlags2)
                {
                    var count = GroupInfoV2List.Count(e => e.Flags2 == flag);
                    report.AppendLine($"  0x{flag:X8}: {count} groups");
                }
            }

            report.AppendLine();
            report.AppendLine($"Groups with LOD: {groupsWithLod}");
            report.AppendLine($"Unique LOD Indices: {uniqueLodIndices.Count}");
            if (uniqueLodIndices.Count > 0)
            {
                report.AppendLine();
                report.AppendLine("LOD Index Distribution:");
                foreach (var index in uniqueLodIndices.OrderBy(i => i))
                {
                    var count = GroupInfoV2List.Count(e => e.LodIndex == index);
                    report.AppendLine($"  LOD {index}: {count} groups");
                }
            }

            return report.ToString();
        }
    }
} 