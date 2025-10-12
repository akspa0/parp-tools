using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MPVD chunk - Map Object Particulate Volumes
    /// Contains particulate volume data, added in 8.3.0.32044
    /// </summary>
    public class MPVD : IChunk
    {
        /// <summary>
        /// Gets the raw particulate volume data.
        /// Structure is currently unknown, so we store raw bytes.
        /// </summary>
        public List<byte[]> ParticleVolumeData { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Clear existing data
            ParticleVolumeData.Clear();

            // Since the structure is unknown, we'll read the entire chunk as raw bytes
            var data = reader.ReadBytes((int)size);
            ParticleVolumeData.Add(data);
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var data in ParticleVolumeData)
            {
                writer.Write(data);
            }
        }

        /// <summary>
        /// Gets a validation report for the particulate volume data.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MPVD Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Data Blocks: {ParticleVolumeData.Count}");

            var totalBytes = 0;
            foreach (var data in ParticleVolumeData)
            {
                totalBytes += data.Length;
            }

            report.AppendLine($"Total Data Size: {totalBytes} bytes");

            return report.ToString();
        }
    }
} 