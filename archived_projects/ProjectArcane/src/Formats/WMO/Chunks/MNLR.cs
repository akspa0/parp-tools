using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MNLR chunk - Map Object New Light References
    /// Contains references to new light definitions (MNLD) for WMO groups.
    /// Added in v9.0.1.33978.
    /// </summary>
    public class MNLR : IChunk
    {
        /// <summary>
        /// Gets the list of new light references.
        /// </summary>
        public List<ushort> NewLightRefs { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each reference is a uint16
            var refCount = (int)size / 2;

            // Clear existing data
            NewLightRefs.Clear();

            // Read light references
            for (int i = 0; i < refCount; i++)
            {
                NewLightRefs.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var lightRef in NewLightRefs)
            {
                writer.Write(lightRef);
            }
        }

        /// <summary>
        /// Gets a validation report for the new light references.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MNLR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Light References: {NewLightRefs.Count}");
            report.AppendLine();

            // Analyze reference indices
            var uniqueRefs = new HashSet<ushort>(NewLightRefs);
            var minRef = NewLightRefs.Count > 0 ? NewLightRefs.Min() : 0;
            var maxRef = NewLightRefs.Count > 0 ? NewLightRefs.Max() : 0;

            report.AppendLine("Reference Statistics:");
            report.AppendLine($"  Unique References: {uniqueRefs.Count}");
            report.AppendLine($"  Index Range: {minRef} to {maxRef}");

            // Create histogram of reference usage
            var refCounts = NewLightRefs.GroupBy(r => r)
                                      .OrderByDescending(g => g.Count())
                                      .Take(5);

            if (refCounts.Any())
            {
                report.AppendLine();
                report.AppendLine("Most Referenced Lights:");
                foreach (var refCount in refCounts)
                {
                    report.AppendLine($"  Light {refCount.Key}: {refCount.Count()} references");
                }
            }

            return report.ToString();
        }
    }
} 