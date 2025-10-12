using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MODI chunk - Map Object Doodad IDs
    /// Contains file data IDs for doodads, added in 8.1.0.27826
    /// Replaces filenames in MODN chunk
    /// </summary>
    public class MODI : IChunk
    {
        /// <summary>
        /// Gets the list of doodad file data IDs.
        /// </summary>
        public List<uint> DoodadFileDataIds { get; } = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Each file data ID is 4 bytes
            var idCount = (int)size / 4;

            // Clear existing data
            DoodadFileDataIds.Clear();

            // Read file data IDs
            for (int i = 0; i < idCount; i++)
            {
                DoodadFileDataIds.Add(reader.ReadUInt32());
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var id in DoodadFileDataIds)
            {
                writer.Write(id);
            }
        }

        /// <summary>
        /// Gets a doodad file data ID by index.
        /// </summary>
        /// <param name="index">Index of the doodad.</param>
        /// <returns>File data ID if found, 0 otherwise.</returns>
        public uint GetDoodadFileDataId(int index)
        {
            if (index < 0 || index >= DoodadFileDataIds.Count)
                return 0;

            return DoodadFileDataIds[index];
        }

        /// <summary>
        /// Validates doodad count against MOHD.
        /// </summary>
        /// <param name="mohdDoodadCount">Number of doodads specified in MOHD chunk.</param>
        /// <returns>True if counts match, false otherwise.</returns>
        public bool ValidateDoodadCount(int mohdDoodadCount)
        {
            return DoodadFileDataIds.Count == mohdDoodadCount;
        }

        /// <summary>
        /// Gets a validation report for the doodad file data IDs.
        /// </summary>
        public string GetValidationReport()
        {
            var report = new System.Text.StringBuilder();
            report.AppendLine("MODI Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Doodad IDs: {DoodadFileDataIds.Count}");
            report.AppendLine();

            // Analyze file data IDs
            var uniqueIds = new HashSet<uint>();
            var zeroIds = 0;

            foreach (var id in DoodadFileDataIds)
            {
                if (id == 0)
                    zeroIds++;
                else
                    uniqueIds.Add(id);
            }

            report.AppendLine($"Unique File Data IDs: {uniqueIds.Count}");
            report.AppendLine($"Zero/Invalid IDs: {zeroIds}");

            if (uniqueIds.Count > 0)
            {
                report.AppendLine();
                report.AppendLine("File Data ID Range:");
                var minId = uint.MaxValue;
                var maxId = uint.MinValue;
                foreach (var id in uniqueIds)
                {
                    minId = System.Math.Min(minId, id);
                    maxId = System.Math.Max(maxId, id);
                }
                report.AppendLine($"  Min ID: {minId}");
                report.AppendLine($"  Max ID: {maxId}");
            }

            return report.ToString();
        }
    }
} 