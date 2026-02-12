using System;
using System.Collections.Generic;
using System.Text;
using ArcaneFileParser.Core.Chunks;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// List of group names for the WMO. Referenced by offset in MOGI chunk.
    /// </summary>
    public class MOGN : IChunk
    {
        /// <summary>
        /// Gets the raw data of the chunk for offset calculations.
        /// </summary>
        public byte[] RawData { get; private set; }

        /// <summary>
        /// Gets the list of group names.
        /// </summary>
        public List<string> Names { get; } = new();

        /// <summary>
        /// Dictionary mapping names to their offsets in the chunk.
        /// </summary>
        private readonly Dictionary<string, int> _offsetLookup = new();

        /// <summary>
        /// Dictionary mapping offsets to their names.
        /// </summary>
        private readonly Dictionary<int, string> _nameByOffset = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Store raw data for offset calculations
            RawData = reader.ReadBytes((int)size);

            // Clear existing data
            Names.Clear();
            _offsetLookup.Clear();
            _nameByOffset.Clear();

            int offset = 0;
            while (offset < RawData.Length)
            {
                // Find the null terminator
                int stringStart = offset;
                int stringEnd = stringStart;

                while (stringEnd < RawData.Length && RawData[stringEnd] != 0)
                {
                    stringEnd++;
                }

                // Extract the name if we found one
                if (stringEnd > stringStart)
                {
                    string name = Encoding.ASCII.GetString(RawData, stringStart, stringEnd - stringStart);
                    
                    // Store the name and its offset
                    Names.Add(name);
                    _offsetLookup[name] = stringStart;
                    _nameByOffset[stringStart] = name;
                }

                // Move past the null terminator
                offset = stringEnd + 1;
            }
        }

        public void Write(BinaryWriter writer)
        {
            // Calculate total size needed
            int totalSize = 0;
            foreach (string name in Names)
            {
                totalSize += name.Length + 1; // +1 for null terminator
            }

            // Create buffer for the data
            byte[] buffer = new byte[totalSize];
            int offset = 0;

            // Reset offset lookups
            _offsetLookup.Clear();
            _nameByOffset.Clear();

            // Write each name to the buffer
            foreach (string name in Names)
            {
                _offsetLookup[name] = offset;
                _nameByOffset[offset] = name;

                if (!string.IsNullOrEmpty(name))
                {
                    byte[] nameBytes = Encoding.ASCII.GetBytes(name);
                    Buffer.BlockCopy(nameBytes, 0, buffer, offset, nameBytes.Length);
                    offset += nameBytes.Length;
                }

                // Add null terminator
                buffer[offset++] = 0;
            }

            // Write the buffer
            writer.Write(buffer);
            RawData = buffer;
        }

        /// <summary>
        /// Gets the offset of a group name in the chunk data.
        /// </summary>
        /// <param name="name">The group name to look up.</param>
        /// <returns>The offset of the name, or -1 if not found.</returns>
        public int GetOffset(string name)
        {
            return _offsetLookup.TryGetValue(name, out int offset) ? offset : -1;
        }

        /// <summary>
        /// Gets a group name by its offset in the chunk data.
        /// </summary>
        /// <param name="offset">The offset to look up.</param>
        /// <returns>The group name at the offset, or null if invalid.</returns>
        public string GetNameByOffset(int offset)
        {
            // First try the direct lookup
            if (_nameByOffset.TryGetValue(offset, out string name))
                return name;

            // If not found and offset is valid, try to read from raw data
            if (offset >= 0 && offset < RawData.Length)
            {
                // Find end of string
                int end = offset;
                while (end < RawData.Length && RawData[end] != 0)
                {
                    end++;
                }

                // Extract the string
                int length = end - offset;
                if (length > 0)
                {
                    name = Encoding.ASCII.GetString(RawData, offset, length);
                    // Cache the result
                    _nameByOffset[offset] = name;
                    return name;
                }
            }

            return null;
        }

        /// <summary>
        /// Validates that a group name offset points to a valid name.
        /// </summary>
        /// <param name="offset">The offset to validate.</param>
        /// <returns>True if the offset points to a valid group name.</returns>
        public bool ValidateOffset(int offset)
        {
            return GetNameByOffset(offset) != null;
        }

        /// <summary>
        /// Gets a validation report for all group names.
        /// </summary>
        /// <returns>A formatted report string.</returns>
        public string GetValidationReport()
        {
            var report = new StringBuilder();
            report.AppendLine("MOGN Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Groups: {Names.Count}");
            report.AppendLine();

            if (Names.Count > 0)
            {
                report.AppendLine("Group Names:");
                for (int i = 0; i < Names.Count; i++)
                {
                    var offset = GetOffset(Names[i]);
                    report.AppendLine($"  [{i}] Offset: {offset}, Name: {Names[i]}");
                }
            }

            return report.ToString();
        }
    }
} 