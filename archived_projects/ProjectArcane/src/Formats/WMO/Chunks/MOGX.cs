using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOGX chunk - Map Object Group Query Face Start
    /// Contains the starting index for query faces, added in 10.0.0.46181
    /// Used in combination with MOQG chunk to determine per polygon groundType
    /// </summary>
    public class MOGX : IChunk
    {
        /// <summary>
        /// Gets or sets the query face start index.
        /// </summary>
        public uint QueryFaceStart { get; set; }

        public void Read(BinaryReader reader)
        {
            QueryFaceStart = reader.ReadUInt32();
        }

        public void Write(BinaryWriter writer)
        {
            writer.Write(QueryFaceStart);
        }

        /// <summary>
        /// Gets a validation report for the query face start.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MOGX Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Query Face Start Index: {QueryFaceStart}");

            return report.ToString();
        }
    }
}