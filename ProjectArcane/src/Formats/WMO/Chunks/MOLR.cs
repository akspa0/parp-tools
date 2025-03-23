using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLR chunk - Map Object Light References
    /// Contains references to lights from MOLS and MOLP chunks.
    /// </summary>
    public class MOLR : IChunk
    {
        /// <summary>
        /// Gets the list of light references.
        /// Each reference is a 16-bit unsigned integer index into either MOLS or MOLP chunks.
        /// </summary>
        public List<ushort> LightReferences { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each reference is 2 bytes (ushort)
            var referenceCount = (int)size / 2;

            // Clear existing data
            LightReferences.Clear();

            // Read references
            for (int i = 0; i < referenceCount; i++)
            {
                LightReferences.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var reference in LightReferences)
            {
                writer.Write(reference);
            }
        }

        /// <summary>
        /// Gets a validation report for the light references.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOLR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Light References: {LightReferences.Count}");
            report.AppendLine();

            if (LightReferences.Count > 0)
            {
                // Track statistics
                var uniqueReferences = new HashSet<ushort>(LightReferences);
                var minReference = ushort.MaxValue;
                var maxReference = ushort.MinValue;

                foreach (var reference in LightReferences)
                {
                    minReference = Math.Min(minReference, reference);
                    maxReference = Math.Max(maxReference, reference);
                }

                // Report statistics
                report.AppendLine("Reference Statistics:");
                report.AppendLine($"  Unique References: {uniqueReferences.Count}");
                report.AppendLine($"  Reference Range: {minReference} to {maxReference}");

                // Analyze reference frequency
                var referenceFrequency = LightReferences
                    .GroupBy(r => r)
                    .OrderByDescending(g => g.Count())
                    .Take(5);

                report.AppendLine();
                report.AppendLine("Most Referenced Lights:");
                foreach (var group in referenceFrequency)
                {
                    report.AppendLine($"  Light {group.Key}: {group.Count()} references");
                }
            }

            return report.ToString();
        }
    }
} 