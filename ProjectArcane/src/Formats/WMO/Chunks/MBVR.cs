using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MBVR chunk - Map Object Box Volume References
    /// Contains references to box volumes from the MBVD chunk.
    /// Added in v8.3.0.32044.
    /// </summary>
    public class MBVR : IChunk
    {
        /// <summary>
        /// Gets the list of box volume references.
        /// Each reference is a 16-bit unsigned integer index into the MBVD chunk.
        /// </summary>
        public List<ushort> BoxVolumeReferences { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each reference is 2 bytes (ushort)
            var referenceCount = (int)size / 2;

            // Clear existing data
            BoxVolumeReferences.Clear();

            // Read references
            for (int i = 0; i < referenceCount; i++)
            {
                BoxVolumeReferences.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var reference in BoxVolumeReferences)
            {
                writer.Write(reference);
            }
        }

        /// <summary>
        /// Gets a validation report for the box volume references.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MBVR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Box Volume References: {BoxVolumeReferences.Count}");
            report.AppendLine();

            if (BoxVolumeReferences.Count > 0)
            {
                // Track statistics
                var uniqueReferences = new HashSet<ushort>(BoxVolumeReferences);
                var minReference = ushort.MaxValue;
                var maxReference = ushort.MinValue;

                foreach (var reference in BoxVolumeReferences)
                {
                    minReference = Math.Min(minReference, reference);
                    maxReference = Math.Max(maxReference, reference);
                }

                // Report statistics
                report.AppendLine("Reference Statistics:");
                report.AppendLine($"  Unique References: {uniqueReferences.Count}");
                report.AppendLine($"  Reference Range: {minReference} to {maxReference}");

                // Analyze reference frequency
                var referenceFrequency = BoxVolumeReferences
                    .GroupBy(r => r)
                    .OrderByDescending(g => g.Count())
                    .Take(5);

                report.AppendLine();
                report.AppendLine("Most Referenced Box Volumes:");
                foreach (var group in referenceFrequency)
                {
                    report.AppendLine($"  Volume {group.Key}: {group.Count()} references");
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Validates that all references point to valid indices in the MBVD chunk.
        /// </summary>
        /// <param name="mbvdChunk">The MBVD chunk containing box volume definitions.</param>
        /// <returns>True if all references are valid, false otherwise.</returns>
        public bool ValidateReferences(MBVD mbvdChunk)
        {
            if (mbvdChunk == null)
                return false;

            var volumeCount = mbvdChunk.AmbientBoxVolumes.Count;
            return BoxVolumeReferences.All(reference => reference < volumeCount);
        }
    }
} 