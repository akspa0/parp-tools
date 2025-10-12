using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MAVD chunk - Map Object Ambient Volumes
    /// Contains ambient volume data, added in 8.3.0.32044
    /// </summary>
    public class MAVD : IChunk
    {
        /// <summary>
        /// Gets the list of ambient volumes.
        /// </summary>
        public List<AmbientVolume> AmbientVolumes { get; } = new();

        public class AmbientVolume
        {
            /// <summary>
            /// Gets or sets the position.
            /// </summary>
            public C3Vector Position { get; set; } = new();

            /// <summary>
            /// Gets or sets the start distance.
            /// </summary>
            public float Start { get; set; }

            /// <summary>
            /// Gets or sets the end distance.
            /// </summary>
            public float End { get; set; }

            /// <summary>
            /// Gets or sets the primary color (overrides MOHD.ambColor).
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
            /// Gets or sets the flags (1: use color2 and color3).
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

            // Each ambient volume is 54 bytes
            var volumeCount = (int)size / 54;

            // Clear existing data
            AmbientVolumes.Clear();

            // Read ambient volumes
            for (int i = 0; i < volumeCount; i++)
            {
                var volume = new AmbientVolume
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

                AmbientVolumes.Add(volume);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var volume in AmbientVolumes)
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
        /// Gets ambient volumes for a specific doodad set.
        /// </summary>
        /// <param name="doodadSetId">ID of the doodad set.</param>
        /// <returns>List of ambient volumes for the doodad set.</returns>
        public List<AmbientVolume> GetVolumesByDoodadSet(ushort doodadSetId)
        {
            return AmbientVolumes.FindAll(v => v.DoodadSetId == doodadSetId);
        }

        /// <summary>
        /// Gets a validation report for the ambient volumes.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MAVD Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Ambient Volumes: {AmbientVolumes.Count}");
            report.AppendLine();

            // Analyze volumes
            var uniqueDoodadSets = new HashSet<ushort>();
            var volumesUsingColor2And3 = 0;
            var volumesWithZeroRange = 0;

            foreach (var volume in AmbientVolumes)
            {
                uniqueDoodadSets.Add(volume.DoodadSetId);
                if ((volume.Flags & 1) != 0)
                    volumesUsingColor2And3++;
                if (volume.Start == volume.End)
                    volumesWithZeroRange++;
            }

            report.AppendLine($"Unique Doodad Sets: {uniqueDoodadSets.Count}");
            report.AppendLine($"Volumes Using Color2 and Color3: {volumesUsingColor2And3}");
            report.AppendLine($"Volumes with Zero Range: {volumesWithZeroRange}");

            // Analyze ranges
            if (AmbientVolumes.Count > 0)
            {
                var minStart = float.MaxValue;
                var maxEnd = float.MinValue;
                foreach (var volume in AmbientVolumes)
                {
                    minStart = System.Math.Min(minStart, volume.Start);
                    maxEnd = System.Math.Max(maxEnd, volume.End);
                }

                report.AppendLine();
                report.AppendLine("Volume Ranges:");
                report.AppendLine($"  Min Start: {minStart:F3}");
                report.AppendLine($"  Max End: {maxEnd:F3}");
            }

            return report.ToString();
        }
    }
} 