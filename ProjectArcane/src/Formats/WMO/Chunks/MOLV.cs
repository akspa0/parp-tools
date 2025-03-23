using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLV chunk - Map Object Light Vertices
    /// Extension to MOLT chunk, added in v9.1.0.39015.
    /// Present in file 3623016 (9.1 Broker Dungeon).
    /// </summary>
    public class MOLV : IChunk
    {
        /// <summary>
        /// Gets the list of light vertex entries.
        /// </summary>
        public List<LightVertexEntry> LightVertexEntries { get; } = new();

        public class LightVertexEntry
        {
            /// <summary>
            /// Gets or sets the array of light vertex directions and values.
            /// Usually either xy or z and the remainder 0.
            /// </summary>
            public LightVertexDirection[] Directions { get; set; } = new LightVertexDirection[6];

            /// <summary>
            /// Gets or sets the unknown values (only seen zeros).
            /// </summary>
            public byte[] Unknown { get; set; } = new byte[3];

            /// <summary>
            /// Gets or sets the MOLT index. Multiple MOLV may reference/extend the same MOLT.
            /// </summary>
            public byte MoltIndex { get; set; }
        }

        public class LightVertexDirection
        {
            /// <summary>
            /// Gets or sets the direction vector.
            /// Usually either xy or z and the remainder 0.
            /// </summary>
            public Vector3 Direction { get; set; }

            /// <summary>
            /// Gets or sets the value associated with this direction.
            /// </summary>
            public float Value { get; set; }
        }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each entry is 100 bytes:
            // 6 * (12 bytes for Vector3 + 4 bytes for float) = 96 bytes for directions
            // 3 bytes for unknown
            // 1 byte for MOLT index
            var entryCount = (int)size / 100;

            // Clear existing data
            LightVertexEntries.Clear();

            // Read entries
            for (int i = 0; i < entryCount; i++)
            {
                var entry = new LightVertexEntry();

                // Read 6 direction entries
                for (int j = 0; j < 6; j++)
                {
                    entry.Directions[j] = new LightVertexDirection
                    {
                        Direction = new Vector3
                        {
                            X = reader.ReadSingle(),
                            Y = reader.ReadSingle(),
                            Z = reader.ReadSingle()
                        },
                        Value = reader.ReadSingle()
                    };
                }

                // Read unknown bytes and MOLT index
                entry.Unknown = reader.ReadBytes(3);
                entry.MoltIndex = reader.ReadByte();

                LightVertexEntries.Add(entry);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var entry in LightVertexEntries)
            {
                // Write 6 direction entries
                foreach (var direction in entry.Directions)
                {
                    writer.Write(direction.Direction.X);
                    writer.Write(direction.Direction.Y);
                    writer.Write(direction.Direction.Z);
                    writer.Write(direction.Value);
                }

                // Write unknown bytes and MOLT index
                writer.Write(entry.Unknown);
                writer.Write(entry.MoltIndex);
            }
        }

        /// <summary>
        /// Gets a validation report for the light vertex entries.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOLV Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Light Vertex Entries: {LightVertexEntries.Count}");
            report.AppendLine();

            // Track MOLT index usage
            var uniqueMoltIndices = new HashSet<byte>();
            var moldIndexCounts = new Dictionary<byte, int>();

            // Track direction patterns
            var xyOnlyCount = 0;
            var zOnlyCount = 0;
            var otherCount = 0;

            // Track value ranges
            var minValue = float.MaxValue;
            var maxValue = float.MinValue;

            foreach (var entry in LightVertexEntries)
            {
                // Track MOLT index usage
                uniqueMoltIndices.Add(entry.MoltIndex);
                if (!moldIndexCounts.ContainsKey(entry.MoltIndex))
                    moldIndexCounts[entry.MoltIndex] = 0;
                moldIndexCounts[entry.MoltIndex]++;

                foreach (var direction in entry.Directions)
                {
                    // Track direction patterns
                    if (direction.Direction.Z == 0 && (direction.Direction.X != 0 || direction.Direction.Y != 0))
                        xyOnlyCount++;
                    else if (direction.Direction.X == 0 && direction.Direction.Y == 0 && direction.Direction.Z != 0)
                        zOnlyCount++;
                    else if (direction.Direction.X != 0 || direction.Direction.Y != 0 || direction.Direction.Z != 0)
                        otherCount++;

                    // Track value ranges
                    minValue = Math.Min(minValue, direction.Value);
                    maxValue = Math.Max(maxValue, direction.Value);
                }
            }

            // Report MOLT index statistics
            report.AppendLine("MOLT Index Usage:");
            report.AppendLine($"  Unique MOLT Indices: {uniqueMoltIndices.Count}");
            
            var topMoltIndices = moldIndexCounts.OrderByDescending(kvp => kvp.Value).Take(5);
            if (topMoltIndices.Any())
            {
                report.AppendLine("  Most Referenced MOLT Indices:");
                foreach (var kvp in topMoltIndices)
                {
                    report.AppendLine($"    Index {kvp.Key}: {kvp.Value} references");
                }
            }
            report.AppendLine();

            // Report direction patterns
            report.AppendLine("Direction Patterns:");
            report.AppendLine($"  XY-Only Directions: {xyOnlyCount}");
            report.AppendLine($"  Z-Only Directions: {zOnlyCount}");
            report.AppendLine($"  Other Directions: {otherCount}");
            report.AppendLine();

            // Report value ranges
            if (minValue != float.MaxValue)
            {
                report.AppendLine("Value Ranges:");
                report.AppendLine($"  Min Value: {minValue:F2}");
                report.AppendLine($"  Max Value: {maxValue:F2}");
            }

            return report.ToString();
        }
    }
} 