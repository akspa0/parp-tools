using System;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Group information for WMO groups, 32 bytes per group.
    /// </summary>
    public class MOGI : IChunk
    {
        public const int GROUP_INFO_SIZE = 32;
        public const int V14_GROUP_INFO_SIZE = 40;

        [Flags]
        public enum GroupFlags : uint
        {
            HasBatch1 = 0x1,                    // Batch A. Regular geometry
            HasBatch2 = 0x2,                    // Batch B. Terrain-like geometry
            HasBatch3 = 0x4,                    // Batch C. Double-sided geometry
            HasExterior = 0x8,                  // Outdoor group
            HasInterior = 0x10,                 // Indoor group
            ShowSkybox = 0x20,                  // Show skybox if enabled
            HasMountable = 0x40,                // Mount support (e.g. benches, chairs)
            UseLightFromOtherGroup = 0x80,      // Use light of another group (antiportals)
            IsDoodadGroup = 0x100,              // Has no BSP tree
            HasLiquid = 0x200,                  // Contains liquid
            IsIndoorGroup = 0x400,              // Indoor
            HasDarkerAO = 0x800,                // Ambient occlusion is darker
            HasUnk1000 = 0x1000,                // Unknown flag
            HasUnk2000 = 0x2000,                // Unknown flag
            HasUnk4000 = 0x4000,                // Unknown flag
            AlwaysDraw = 0x8000,                // Always render group
            HasMOCV = 0x10000,                  // Has vertex colors (MOCV chunk)
            HasMOTV = 0x20000,                  // Has texture coordinates (MOTV chunk)
            HasUnk40000 = 0x40000,              // Unknown flag
            HasUnk80000 = 0x80000,              // Unknown flag
            HasUnk100000 = 0x100000,            // Unknown flag
            HasUnk200000 = 0x200000,            // Unknown flag
            HasMOLR = 0x400000,                 // Has light references (MOLR chunk)
            HasUnk800000 = 0x800000,            // Unknown flag
            HasUnk1000000 = 0x1000000,          // Unknown flag
            HasUnk2000000 = 0x2000000,          // Unknown flag
            HasUnk4000000 = 0x4000000,          // Unknown flag
            HasUnk8000000 = 0x8000000,          // Unknown flag
            HasUnk10000000 = 0x10000000,        // Unknown flag
            HasUnk20000000 = 0x20000000,        // Unknown flag
            HasUnk40000000 = 0x40000000,        // Unknown flag
            HasUnk80000000 = 0x80000000         // Unknown flag
        }

        public class GroupInfo
        {
            public uint Offset { get; set; }        // v14 only: Absolute address
            public uint Size { get; set; }          // v14 only: Includes IffChunk header
            public GroupFlags Flags { get; set; }
            public CAaBox BoundingBox { get; set; }
            public int NameOffset { get; set; }     // -1 for no name

            public GroupInfo()
            {
                BoundingBox = new CAaBox();
            }
        }

        public GroupInfo[] Groups { get; private set; }
        public bool IsV14 { get; private set; }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Determine version based on chunk size and group count
            var groupCount = (int)size / GROUP_INFO_SIZE;
            var v14GroupCount = (int)size / V14_GROUP_INFO_SIZE;

            // If the size perfectly divides by V14_GROUP_INFO_SIZE but not GROUP_INFO_SIZE, it's v14
            IsV14 = size % GROUP_INFO_SIZE != 0 && size % V14_GROUP_INFO_SIZE == 0;

            Groups = new GroupInfo[IsV14 ? v14GroupCount : groupCount];

            for (int i = 0; i < Groups.Length; i++)
            {
                var group = new GroupInfo();

                if (IsV14)
                {
                    group.Offset = reader.ReadUInt32();
                    group.Size = reader.ReadUInt32();
                }

                group.Flags = (GroupFlags)reader.ReadUInt32();
                group.BoundingBox = reader.ReadCAaBox();
                group.NameOffset = reader.ReadInt32();

                Groups[i] = group;
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var group in Groups)
            {
                if (IsV14)
                {
                    writer.Write(group.Offset);
                    writer.Write(group.Size);
                }

                writer.Write((uint)group.Flags);
                writer.Write(group.BoundingBox);
                writer.Write(group.NameOffset);
            }
        }

        /// <summary>
        /// Validates group name offsets against the MOGN chunk.
        /// </summary>
        public bool ValidateGroupNames(MOGN mogn)
        {
            if (mogn == null)
                return false;

            foreach (var group in Groups)
            {
                // Skip groups with no name
                if (group.NameOffset == -1)
                    continue;

                // Validate the name offset
                if (!mogn.ValidateOffset(group.NameOffset))
                    return false;

                // Special validation for antiportals
                var name = mogn.GetNameByOffset(group.NameOffset);
                if (name?.ToLowerInvariant() == "antiportal" && !group.Flags.HasFlag(GroupFlags.UseLightFromOtherGroup))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for all groups.
        /// </summary>
        public string GetValidationReport(MOGN mogn)
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOGI Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Groups: {Groups.Length}");
            report.AppendLine($"Version: {(IsV14 ? "14" : "17+")}");
            report.AppendLine();

            for (int i = 0; i < Groups.Length; i++)
            {
                var group = Groups[i];
                report.AppendLine($"Group {i}:");
                report.AppendLine($"  Flags: {group.Flags}");
                report.AppendLine($"  Bounding Box: {group.BoundingBox}");
                
                if (mogn != null)
                {
                    var name = group.NameOffset >= 0 ? mogn.GetNameByOffset(group.NameOffset) : null;
                    report.AppendLine($"  Name Offset: {group.NameOffset} ({(name ?? "No Name")})");

                    if (name?.ToLowerInvariant() == "antiportal" && !group.Flags.HasFlag(GroupFlags.UseLightFromOtherGroup))
                    {
                        report.AppendLine("  WARNING: Antiportal group missing UseLightFromOtherGroup flag");
                    }
                }

                if (IsV14)
                {
                    report.AppendLine($"  Offset: 0x{group.Offset:X8}");
                    report.AppendLine($"  Size: {group.Size}");
                }

                report.AppendLine();
            }

            return report.ToString();
        }

        /// <summary>
        /// Gets the number of groups that have a particular flag set.
        /// </summary>
        public int GetGroupCountWithFlag(GroupFlags flag)
        {
            int count = 0;
            foreach (var group in Groups)
            {
                if (group.Flags.HasFlag(flag))
                    count++;
            }
            return count;
        }

        /// <summary>
        /// Gets statistics about group types.
        /// </summary>
        public (int Exterior, int Interior, int Antiportal) GetGroupTypeStats(MOGN mogn)
        {
            int exterior = 0;
            int interior = 0;
            int antiportal = 0;

            foreach (var group in Groups)
            {
                if (group.Flags.HasFlag(GroupFlags.HasExterior))
                    exterior++;
                if (group.Flags.HasFlag(GroupFlags.HasInterior))
                    interior++;

                if (mogn != null && group.NameOffset >= 0)
                {
                    var name = mogn.GetNameByOffset(group.NameOffset);
                    if (name?.ToLowerInvariant() == "antiportal")
                        antiportal++;
                }
            }

            return (exterior, interior, antiportal);
        }
    }
} 