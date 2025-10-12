using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MAVG chunk - Map Object Global Ambient Volumes
    /// Contains global ambient volume data, added in 8.3.0.32044
    /// </summary>
    public class MAVG : IChunk
    {
        /// <summary>
        /// Gets the list of global ambient volumes.
        /// </summary>
        public List<GlobalAmbientVolume> GlobalAmbientVolumes { get; } = new();

        public class GlobalAmbientVolume
        {
            /// <summary>
            /// Gets or sets the position (always 0,0,0 for global ambient).
            /// </summary>
            public C3Vector Position { get; set; } = new();

            /// <summary>
            /// Gets or sets the start distance (always 0 for global ambient).
            /// </summary>
            public float Start { get; set; }

            /// <summary>
            /// Gets or sets the end distance (always 0 for global ambient).
            /// </summary>
            public float End { get; set; }

            /// <summary>
            /// Gets or sets the primary color.
            /// </summary>
            public CImVector Color1 { get; set; } = new();

            /// <summary>
            /// Gets or sets the secondary color.
            /// </summary>
            public CImVector Color2 { get; set; } = new();

            /// <summary>
            /// Gets or sets the tertiary color.
            /// </summary>
            public CImVector Color3 { get; set; } = new();

            /// <summary>
            /// Gets or sets the flags (1: use color1 and color3).
            /// </summary>
            public uint Flags { get; set; }

            /// <summary>
            /// Gets or sets the doodad set ID.
            /// </summary>
            public ushort DoodadSetId { get; set; }

            /// <summary>
            /// Gets or sets the padding bytes.
            /// </summary>
            public byte[] Padding { get; set; } = new byte[10];
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each global ambient volume is 54 bytes
            var volumeCount = (int)size / 54;

            // Clear existing data
            GlobalAmbientVolumes.Clear();

            // Read global ambient volumes
            for (int i = 0; i < volumeCount; i++)
            {
                var volume = new GlobalAmbientVolume
                {
                    Position = new C3Vector
                    {
                        X = reader.ReadSingle(),
                        Y = reader.ReadSingle(),
                        Z = reader.ReadSingle()
                    },
                    Start = reader.ReadSingle(),
                    End = reader.ReadSingle(),
                    Color1 = new CImVector
                    {
                        B = reader.ReadByte(),
                        G = reader.ReadByte(),
                        R = reader.ReadByte(),
                        A = reader.ReadByte()
                    },
                    Color2 = new CImVector
                    {
                        B = reader.ReadByte(),
                        G = reader.ReadByte(),
                        R = reader.ReadByte(),
                        A = reader.ReadByte()
                    },
                    Color3 = new CImVector
                    {
                        B = reader.ReadByte(),
                        G = reader.ReadByte(),
                        R = reader.ReadByte(),
                        A = reader.ReadByte()
                    },
                    Flags = reader.ReadUInt32(),
                    DoodadSetId = reader.ReadUInt16()
                };

                // Read padding bytes
                volume.Padding = reader.ReadBytes(10);

                GlobalAmbientVolumes.Add(volume);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var volume in GlobalAmbientVolumes)
            {
                writer.Write(volume.Position.X);
                writer.Write(volume.Position.Y);
                writer.Write(volume.Position.Z);
                writer.Write(volume.Start);
                writer.Write(volume.End);
                writer.Write(volume.Color1.B);
                writer.Write(volume.Color1.G);
                writer.Write(volume.Color1.R);
                writer.Write(volume.Color1.A);
                writer.Write(volume.Color2.B);
                writer.Write(volume.Color2.G);
                writer.Write(volume.Color2.R);
                writer.Write(volume.Color2.A);
                writer.Write(volume.Color3.B);
                writer.Write(volume.Color3.G);
                writer.Write(volume.Color3.R);
                writer.Write(volume.Color3.A);
                writer.Write(volume.Flags);
                writer.Write(volume.DoodadSetId);
                writer.Write(volume.Padding);
            }
        }

        /// <summary>
        /// Gets a global ambient volume by doodad set ID.
        /// </summary>
        /// <param name="doodadSetId">ID of the doodad set.</param>
        /// <returns>Global ambient volume if found, null otherwise.</returns>
        public GlobalAmbientVolume GetVolumeByDoodadSet(ushort doodadSetId)
        {
            return GlobalAmbientVolumes.Find(v => v.DoodadSetId == doodadSetId);
        }

        /// <summary>
        /// Gets a validation report for the global ambient volumes.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MAVG Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Global Ambient Volumes: {GlobalAmbientVolumes.Count}");
            report.AppendLine();

            // Analyze volumes
            var uniqueDoodadSets = new HashSet<ushort>();
            var volumesUsingColor1And3 = 0;

            foreach (var volume in GlobalAmbientVolumes)
            {
                uniqueDoodadSets.Add(volume.DoodadSetId);
                if ((volume.Flags & 1) != 0)
                    volumesUsingColor1And3++;

                // Verify that position, start, and end are 0 as they should be for global ambient
                if (volume.Position.X != 0 || volume.Position.Y != 0 || volume.Position.Z != 0 ||
                    volume.Start != 0 || volume.End != 0)
                {
                    report.AppendLine($"Warning: Volume for doodad set {volume.DoodadSetId} has non-zero position/range values");
                }
            }

            report.AppendLine($"Unique Doodad Sets: {uniqueDoodadSets.Count}");
            report.AppendLine($"Volumes Using Color1 and Color3: {volumesUsingColor1And3}");

            return report.ToString();
        }
    }
} 