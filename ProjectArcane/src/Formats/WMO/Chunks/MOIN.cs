using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOIN chunk - Index list for v14 (Alpha) WMO
    /// Contains indices for faces, typically incrementing from 0 to nFaces * 3
    /// </summary>
    public class MOIN : IChunk
    {
        /// <summary>
        /// Gets the list of indices.
        /// </summary>
        public List<ushort> Indices { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each index is 2 bytes
            var indexCount = (int)size / 2;

            // Clear existing data
            Indices.Clear();

            // Read indices
            for (int i = 0; i < indexCount; i++)
            {
                Indices.Add(reader.ReadUInt16());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var index in Indices)
            {
                writer.Write(index);
            }
        }

        /// <summary>
        /// Gets indices for a specific face.
        /// </summary>
        /// <param name="faceIndex">Index of the face.</param>
        /// <returns>Array of 3 indices for the face vertices.</returns>
        public ushort[] GetFaceIndices(int faceIndex)
        {
            var startIndex = faceIndex * 3;
            if (startIndex < 0 || startIndex + 3 > Indices.Count)
                return null;

            return new[] { Indices[startIndex], Indices[startIndex + 1], Indices[startIndex + 2] };
        }

        /// <summary>
        /// Validates that indices form proper triangles and are sequential.
        /// </summary>
        /// <returns>True if indices are valid, false otherwise.</returns>
        public bool ValidateIndices()
        {
            // Must have complete triangles
            if (Indices.Count % 3 != 0)
                return false;

            // In v14, indices should be sequential (0,1,2,3,4,5,...)
            for (int i = 0; i < Indices.Count; i++)
            {
                if (Indices[i] != i)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets a validation report for the indices.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOIN Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Indices: {Indices.Count}");
            report.AppendLine($"Face Count: {Indices.Count / 3}");
            report.AppendLine();

            // Check if indices are sequential
            bool isSequential = true;
            int firstNonSequential = -1;
            for (int i = 0; i < Indices.Count; i++)
            {
                if (Indices[i] != i)
                {
                    isSequential = false;
                    firstNonSequential = i;
                    break;
                }
            }

            report.AppendLine($"Sequential Indices: {isSequential}");
            if (!isSequential && firstNonSequential >= 0)
            {
                report.AppendLine($"First Non-Sequential Index at position {firstNonSequential}:");
                report.AppendLine($"  Expected: {firstNonSequential}");
                report.AppendLine($"  Found: {Indices[firstNonSequential]}");
            }

            // Check index range
            if (Indices.Count > 0)
            {
                var minIndex = ushort.MaxValue;
                var maxIndex = ushort.MinValue;
                foreach (var index in Indices)
                {
                    minIndex = System.Math.Min(minIndex, index);
                    maxIndex = System.Math.Max(maxIndex, index);
                }

                report.AppendLine();
                report.AppendLine("Index Range:");
                report.AppendLine($"  Min Index: {minIndex}");
                report.AppendLine($"  Max Index: {maxIndex}");
            }

            return report.ToString();
        }
    }
} 