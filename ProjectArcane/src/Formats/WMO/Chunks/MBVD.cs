using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MBVD chunk - Map Object Box Volume Definitions
    /// Contains ambient box volume definitions.
    /// Added in v8.3.0.32044.
    /// </summary>
    public class MBVD : IChunk
    {
        /// <summary>
        /// Gets the list of ambient box volumes.
        /// </summary>
        public List<AmbientBoxVolume> AmbientBoxVolumes { get; } = new();

        public class AmbientBoxVolume
        {
            /// <summary>
            /// Gets or sets the planes defining the box volume.
            /// </summary>
            public C4Plane[] Planes { get; set; } = new C4Plane[6];

            /// <summary>
            /// Gets or sets the end distance.
            /// </summary>
            public float End { get; set; }

            /// <summary>
            /// Gets or sets the first color.
            /// </summary>
            public CImVector Color1 { get; set; }

            /// <summary>
            /// Gets or sets the second color.
            /// </summary>
            public CImVector Color2 { get; set; }

            /// <summary>
            /// Gets or sets the third color.
            /// </summary>
            public CImVector Color3 { get; set; }

            /// <summary>
            /// Gets or sets the flags. &1: use color2 + color3.
            /// </summary>
            public uint Flags { get; set; }

            /// <summary>
            /// Gets or sets the doodad set ID.
            /// </summary>
            public ushort DoodadSetId { get; set; }

            /// <summary>
            /// Gets or sets the unknown bytes (10 bytes).
            /// </summary>
            public byte[] Unknown { get; set; } = new byte[10];
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each ambient box volume is 134 bytes:
            // 6 * 16 bytes for planes = 96 bytes
            // 4 bytes for end
            // 3 * 4 bytes for colors = 12 bytes
            // 4 bytes for flags
            // 2 bytes for doodad set ID
            // 10 bytes for unknown data
            var volumeCount = (int)size / 134;

            // Clear existing data
            AmbientBoxVolumes.Clear();

            // Read volumes
            for (int i = 0; i < volumeCount; i++)
            {
                var volume = new AmbientBoxVolume();

                // Read planes
                for (int j = 0; j < 6; j++)
                {
                    volume.Planes[j] = new C4Plane
                    {
                        A = reader.ReadSingle(),
                        B = reader.ReadSingle(),
                        C = reader.ReadSingle(),
                        D = reader.ReadSingle()
                    };
                }

                volume.End = reader.ReadSingle();
                volume.Color1 = new CImVector { Value = reader.ReadUInt32() };
                volume.Color2 = new CImVector { Value = reader.ReadUInt32() };
                volume.Color3 = new CImVector { Value = reader.ReadUInt32() };
                volume.Flags = reader.ReadUInt32();
                volume.DoodadSetId = reader.ReadUInt16();
                volume.Unknown = reader.ReadBytes(10);

                AmbientBoxVolumes.Add(volume);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var volume in AmbientBoxVolumes)
            {
                // Write planes
                foreach (var plane in volume.Planes)
                {
                    writer.Write(plane.A);
                    writer.Write(plane.B);
                    writer.Write(plane.C);
                    writer.Write(plane.D);
                }

                writer.Write(volume.End);
                writer.Write(volume.Color1.Value);
                writer.Write(volume.Color2.Value);
                writer.Write(volume.Color3.Value);
                writer.Write(volume.Flags);
                writer.Write(volume.DoodadSetId);
                writer.Write(volume.Unknown);
            }
        }

        /// <summary>
        /// Gets a validation report for the ambient box volumes.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MBVD Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Ambient Box Volumes: {AmbientBoxVolumes.Count}");
            report.AppendLine();

            // Track statistics
            var uniqueDoodadSets = new HashSet<ushort>();
            var volumesUsingColorBlend = 0;
            var minEnd = float.MaxValue;
            var maxEnd = float.MinValue;

            foreach (var volume in AmbientBoxVolumes)
            {
                uniqueDoodadSets.Add(volume.DoodadSetId);
                if ((volume.Flags & 1) != 0) volumesUsingColorBlend++;
                minEnd = Math.Min(minEnd, volume.End);
                maxEnd = Math.Max(maxEnd, volume.End);
            }

            // Report statistics
            report.AppendLine("Volume Statistics:");
            report.AppendLine($"  Unique Doodad Sets: {uniqueDoodadSets.Count}");
            report.AppendLine($"  Volumes Using Color Blend: {volumesUsingColorBlend}");
            report.AppendLine($"  End Distance Range: {minEnd:F2} to {maxEnd:F2}");

            // Analyze plane configurations
            report.AppendLine();
            report.AppendLine("Plane Analysis:");
            for (int i = 0; i < AmbientBoxVolumes.Count; i++)
            {
                var volume = AmbientBoxVolumes[i];
                var invalidPlanes = volume.Planes.Where(p => 
                    float.IsNaN(p.A) || float.IsInfinity(p.A) ||
                    float.IsNaN(p.B) || float.IsInfinity(p.B) ||
                    float.IsNaN(p.C) || float.IsInfinity(p.C) ||
                    float.IsNaN(p.D) || float.IsInfinity(p.D)).ToList();

                if (invalidPlanes.Any())
                {
                    report.AppendLine($"  Volume {i} has {invalidPlanes.Count} invalid planes");
                }
            }

            // Report doodad set usage
            var doodadSetCounts = AmbientBoxVolumes.GroupBy(v => v.DoodadSetId)
                                                  .OrderByDescending(g => g.Count())
                                                  .Take(5);

            if (doodadSetCounts.Any())
            {
                report.AppendLine();
                report.AppendLine("Most Used Doodad Sets:");
                foreach (var doodadSet in doodadSetCounts)
                {
                    report.AppendLine($"  Set {doodadSet.Key}: {doodadSet.Count()} volumes");
                }
            }

            return report.ToString();
        }
    }

    public class C4Plane
    {
        public float A { get; set; }
        public float B { get; set; }
        public float C { get; set; }
        public float D { get; set; }
    }
} 