using System;
using System.Collections.Generic;
using System.Text;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Map Object TeXture names chunk. Contains a list of texture filenames used in the WMO.
    /// </summary>
    public class MOTX : IChunk
    {
        /// <summary>
        /// Gets the raw data of the chunk for offset calculations.
        /// </summary>
        public byte[] RawData { get; private set; }

        /// <summary>
        /// Gets the list of texture filenames.
        /// </summary>
        public List<string> Filenames { get; } = new();

        /// <summary>
        /// Gets the FileDataIDs for each texture path, if they exist in the listfile.
        /// </summary>
        public Dictionary<string, uint?> FileDataIds { get; } = new();

        /// <summary>
        /// Gets validation issues for each path.
        /// </summary>
        public Dictionary<string, List<string>> ValidationIssues { get; } = new();

        /// <summary>
        /// Gets pattern matching results for each path.
        /// </summary>
        public Dictionary<string, (bool IsValid, string? Pattern, string? Suggestion)> PatternResults { get; } = new();

        /// <summary>
        /// Gets suggested fixes for each path.
        /// </summary>
        public Dictionary<string, List<(string Description, string Fix)>> SuggestedFixes { get; } = new();

        /// <summary>
        /// Gets whether the listfile has been initialized.
        /// </summary>
        public bool HasListfile => ListfileManager.Instance._isInitialized;

        private readonly Dictionary<string, int> _offsetLookup = new();

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Store raw data for offset calculations
            RawData = reader.ReadBytes((int)size);

            // Extract and validate filenames
            Filenames.Clear();
            _offsetLookup.Clear();
            FileDataIds.Clear();
            ValidationIssues.Clear();
            PatternResults.Clear();
            SuggestedFixes.Clear();

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

                // Extract the filename
                if (stringEnd > stringStart)
                {
                    string filename = Encoding.ASCII.GetString(RawData, stringStart, stringEnd - stringStart);
                    
                    // Store the filename and its offset
                    Filenames.Add(filename);
                    _offsetLookup[filename] = stringStart;

                    // Validate the path
                    ValidatePath(filename);
                }

                // Move past the null terminator
                offset = stringEnd + 1;
            }
        }

        public void Write(BinaryWriter writer)
        {
            // Calculate total size needed
            int totalSize = 0;
            foreach (string filename in Filenames)
            {
                totalSize += filename.Length + 1; // +1 for null terminator
            }

            // Create buffer for the data
            byte[] buffer = new byte[totalSize];
            int offset = 0;

            // Reset offset lookup
            _offsetLookup.Clear();

            // Write each filename to the buffer
            foreach (string filename in Filenames)
            {
                _offsetLookup[filename] = offset;

                if (!string.IsNullOrEmpty(filename))
                {
                    byte[] filenameBytes = Encoding.ASCII.GetBytes(filename);
                    Buffer.BlockCopy(filenameBytes, 0, buffer, offset, filenameBytes.Length);
                    offset += filenameBytes.Length;
                }

                // Add null terminator
                buffer[offset++] = 0;
            }

            // Write the buffer
            writer.Write(buffer);
            RawData = buffer;
        }

        /// <summary>
        /// Gets the offset of a filename in the chunk data.
        /// </summary>
        /// <param name="filename">The filename to look up.</param>
        /// <returns>The offset of the filename, or -1 if not found.</returns>
        public int GetOffset(string filename)
        {
            return _offsetLookup.TryGetValue(filename, out int offset) ? offset : -1;
        }

        /// <summary>
        /// Gets a filename by its offset in the chunk data.
        /// </summary>
        /// <param name="offset">The offset to look up.</param>
        /// <returns>The filename at the offset, or null if invalid.</returns>
        public string? GetFilenameByOffset(int offset)
        {
            if (offset < 0 || offset >= RawData.Length)
                return null;

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
                return Encoding.ASCII.GetString(RawData, offset, length);
            }

            return null;
        }

        private void ValidatePath(string path)
        {
            // Get FileDataID if available
            var fileDataId = ListfileManager.Instance.GetFileDataId(path);
            FileDataIds[path] = fileDataId;

            // Check if file exists in listfile
            if (!ListfileManager.Instance.ValidatePath(path))
            {
                AssetPathValidator.LogMissingFile(path, fileDataId);
            }

            // Validate path format
            if (AssetPathValidator.ValidatePath(path, out var issues))
            {
                ValidationIssues[path] = issues;
            }

            // Check path pattern
            PatternResults[path] = AssetPathValidator.MatchPattern(path);

            // Get suggested fixes
            var fixes = AssetPathValidator.GetSuggestedFixes(path).ToList();
            if (fixes.Any())
            {
                SuggestedFixes[path] = fixes;
            }
        }

        /// <summary>
        /// Gets validation statistics for all texture paths.
        /// </summary>
        /// <returns>A tuple containing (total paths, valid paths, invalid paths).</returns>
        public (int Total, int Valid, int Invalid) GetValidationStats()
        {
            int valid = 0, invalid = 0;

            foreach (var path in Filenames)
            {
                if (ListfileManager.Instance.ValidatePath(path))
                    valid++;
                else
                    invalid++;
            }

            return (Filenames.Count, valid, invalid);
        }

        /// <summary>
        /// Gets a detailed validation report.
        /// </summary>
        /// <returns>A formatted report string.</returns>
        public string GetValidationReport()
        {
            var report = new StringBuilder();
            var stats = GetValidationStats();

            report.AppendLine("MOTX Validation Report");
            report.AppendLine("--------------------");
            report.AppendLine($"Total Textures: {stats.Total}");
            report.AppendLine($"Valid Paths: {stats.Valid}");
            report.AppendLine($"Invalid Paths: {stats.Invalid}");
            report.AppendLine();

            if (ValidationIssues.Any())
            {
                report.AppendLine("Validation Issues:");
                foreach (var (path, issues) in ValidationIssues)
                {
                    report.AppendLine($"  {path}:");
                    foreach (var issue in issues)
                    {
                        report.AppendLine($"    - {issue}");
                    }
                }
                report.AppendLine();
            }

            if (SuggestedFixes.Any())
            {
                report.AppendLine("Suggested Fixes:");
                foreach (var (path, fixes) in SuggestedFixes)
                {
                    report.AppendLine($"  {path}:");
                    foreach (var (description, fix) in fixes)
                    {
                        report.AppendLine($"    - {description}");
                        report.AppendLine($"      Fix: {fix}");
                    }
                }
            }

            return report.ToString();
        }
    }
} 