using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MAVR chunk - Map Object Ambient Volume References
    /// Contains references to ambient volumes from the MAVD and MAVG chunks.
    /// Added in v9.0.1.33978.
    /// </summary>
    public class MAVR : IChunk
    {
        /// <summary>
        /// Gets the list of ambient volume references.
        /// Each reference is a 16-bit unsigned integer index into the MAVD/MAVG chunks.
        /// </summary>
        public List<ushort> AmbientVolumeReferences { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each reference is 2 bytes (ushort)
            var referenceCount = (int)size / 2;

            // Clear existing data
            AmbientVolumeReferences.Clear();

            // Read references
            for (int i = 0; i < referenceCount; i++)
            {
                AmbientVolumeReferences.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var reference in AmbientVolumeReferences)
            {
                writer.Write(reference);
            }
        }

        /// <summary>
        /// Gets a validation report for the ambient volume references.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MAVR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Ambient Volume References: {AmbientVolumeReferences.Count}");
            report.AppendLine();

            if (AmbientVolumeReferences.Count > 0)
            {
                // Track statistics
                var uniqueReferences = new HashSet<ushort>(AmbientVolumeReferences);
                var minReference = ushort.MaxValue;
                var maxReference = ushort.MinValue;

                foreach (var reference in AmbientVolumeReferences)
                {
                    minReference = Math.Min(minReference, reference);
                    maxReference = Math.Max(maxReference, reference);
                }

                // Report statistics
                report.AppendLine("Reference Statistics:");
                report.AppendLine($"  Unique References: {uniqueReferences.Count}");
                report.AppendLine($"  Reference Range: {minReference} to {maxReference}");

                // Analyze reference frequency
                var referenceFrequency = AmbientVolumeReferences
                    .GroupBy(r => r)
                    .OrderByDescending(g => g.Count())
                    .Take(5);

                report.AppendLine();
                report.AppendLine("Most Referenced Ambient Volumes:");
                foreach (var group in referenceFrequency)
                {
                    report.AppendLine($"  Volume {group.Key}: {group.Count()} references");
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Validates that all references point to valid indices in the MAVD and MAVG chunks.
        /// </summary>
        /// <param name="mavdChunk">The MAVD chunk containing ambient volume definitions.</param>
        /// <param name="mavgChunk">The MAVG chunk containing ambient volume groups.</param>
        /// <returns>True if all references are valid, false otherwise.</returns>
        public bool ValidateReferences(MAVD mavdChunk, MAVG mavgChunk)
        {
            if (mavdChunk == null || mavgChunk == null)
                return false;

            var volumeCount = mavdChunk.AmbientVolumes.Count;
            var groupCount = mavgChunk.VolumeGroups.Count;

            // Each reference should be valid for both MAVD and MAVG
            return AmbientVolumeReferences.All(reference => 
                reference < volumeCount && reference < groupCount);
        }
    }
} 