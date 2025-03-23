using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOSI chunk - Map Object Skybox Info
    /// Contains file data ID for skybox, added in 8.1.0.27826
    /// </summary>
    public class MOSI : IChunk
    {
        /// <summary>
        /// Gets or sets the skybox file data ID.
        /// </summary>
        public uint SkyboxFileDataId { get; set; }

        public void Read(BinaryReader reader)
        {
            SkyboxFileDataId = reader.ReadUInt32();
        }

        public void Write(BinaryWriter writer)
        {
            writer.Write(SkyboxFileDataId);
        }

        /// <summary>
        /// Gets a validation report for the skybox file data ID.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOSI Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Skybox File Data ID: {SkyboxFileDataId}");
            report.AppendLine($"Has Skybox: {SkyboxFileDataId != 0}");

            return report.ToString();
        }
    }
} 