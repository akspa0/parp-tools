using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MFVR chunk - Map Object Fog Volume References
    /// Contains references to fog volumes from the MFOG and MFED chunks.
    /// Added in v9.0.1.33978.
    /// </summary>
    public class MFVR : IChunk
    {
        /// <summary>
        /// Gets the list of fog volume references.
        /// Each reference is a 16-bit unsigned integer index into the MFOG/MFED chunks.
        /// </summary>
        public List<ushort> FogVolumeReferences { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each reference is 2 bytes (ushort)
            var referenceCount = (int)size / 2;

            // Clear existing data
            FogVolumeReferences.Clear();

            // Read references
            for (int i = 0; i < referenceCount; i++)
            {
                FogVolumeReferences.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var reference in FogVolumeReferences)
            {
                writer.Write(reference);
            }
        }

        /// <summary>
        /// Gets a validation report for the fog volume references.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MFVR Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Fog Volume References: {FogVolumeReferences.Count}");
            report.AppendLine();

            if (FogVolumeReferences.Count > 0)
            {
                // Track statistics
                var uniqueReferences = new HashSet<ushort>(FogVolumeReferences);
                var minReference = ushort.MaxValue;
                var maxReference = ushort.MinValue;

                foreach (var reference in FogVolumeReferences)
                {
                    minReference = Math.Min(minReference, reference);
                    maxReference = Math.Max(maxReference, reference);
                }

                // Report statistics
                report.AppendLine("Reference Statistics:");
                report.AppendLine($"  Unique References: {uniqueReferences.Count}");
                report.AppendLine($"  Reference Range: {minReference} to {maxReference}");

                // Analyze reference frequency
                var referenceFrequency = FogVolumeReferences
                    .GroupBy(r => r)
                    .OrderByDescending(g => g.Count())
                    .Take(5);

                report.AppendLine();
                report.AppendLine("Most Referenced Fog Volumes:");
                foreach (var group in referenceFrequency)
                {
                    report.AppendLine($"  Volume {group.Key}: {group.Count()} references");
                }
            }

            return report.ToString();
        }

        /// <summary>
        /// Validates that all references point to valid indices in the MFOG chunk.
        /// </summary>
        /// <param name="mfogChunk">The MFOG chunk containing fog definitions.</param>
        /// <param name="mfedChunk">The MFED chunk containing fog extra data.</param>
        /// <returns>True if all references are valid, false otherwise.</returns>
        public bool ValidateReferences(MFOG mfogChunk, MFED mfedChunk)
        {
            if (mfogChunk == null)
                return false;

            var fogCount = mfogChunk.FogData.Count;

            // If MFED is present, both chunks should have the same count
            if (mfedChunk != null && mfedChunk.FogExtraDataList.Count != fogCount)
                return false;

            return FogVolumeReferences.All(reference => reference < fogCount);
        }
    }
} 