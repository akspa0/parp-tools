using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using WoWToolbox.Validation.Chunkvault.Models;

namespace WoWToolbox.Validation.Chunkvault.Parsers
{
    /// <summary>
    /// Parser for chunkvault markdown documentation
    /// </summary>
    public class MarkdownSpecParser
    {
        private static readonly Regex ChunkIdPattern = new(@"^# (\w{4})", RegexOptions.Compiled);
        private static readonly Regex VersionPattern = new(@"Version(?:s)?: (.+)$", RegexOptions.Compiled);
        private static readonly Regex FieldPattern = new(@"^\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(.+)\s*\|$", RegexOptions.Compiled);

        /// <summary>
        /// Parses a chunk specification from markdown content
        /// </summary>
        /// <param name="markdownContent">The markdown content to parse</param>
        /// <returns>The parsed chunk specification</returns>
        public ChunkSpecification ParseChunkSpec(string markdownContent)
        {
            var spec = new ChunkSpecification
            {
                Fields = new Dictionary<string, FieldSpecification>(),
                Rules = new ValidationRules
                {
                    FieldConstraints = new Dictionary<string, ValueConstraints>(),
                    Size = new SizeConstraints()
                }
            };

            using var reader = new StringReader(markdownContent);
            string line;
            var inFieldTable = false;

            while ((line = reader.ReadLine()) != null)
            {
                // Parse chunk ID
                var chunkMatch = ChunkIdPattern.Match(line);
                if (chunkMatch.Success)
                {
                    spec.ChunkId = chunkMatch.Groups[1].Value;
                    continue;
                }

                // Parse versions
                var versionMatch = VersionPattern.Match(line);
                if (versionMatch.Success)
                {
                    spec.SupportedVersions = ParseVersions(versionMatch.Groups[1].Value);
                    continue;
                }

                // Parse field table
                if (line.StartsWith("|"))
                {
                    if (line.Contains("Field") && line.Contains("Type") && line.Contains("Size"))
                    {
                        inFieldTable = true;
                        continue;
                    }

                    if (inFieldTable)
                    {
                        var fieldMatch = FieldPattern.Match(line);
                        if (fieldMatch.Success)
                        {
                            var field = new FieldSpecification
                            {
                                DataType = fieldMatch.Groups[2].Value,
                                Size = int.Parse(fieldMatch.Groups[3].Value),
                                Description = fieldMatch.Groups[4].Value.Trim(),
                                VersionNotes = new Dictionary<int, string>()
                            };

                            spec.Fields[fieldMatch.Groups[1].Value] = field;
                        }
                    }
                }
                else
                {
                    inFieldTable = false;
                }
            }

            return spec;
        }

        private static int[] ParseVersions(string versionString)
        {
            var versions = new List<int>();
            var parts = versionString.Split(',', StringSplitOptions.RemoveEmptyEntries);

            foreach (var part in parts)
            {
                var trimmed = part.Trim();
                if (trimmed.Contains("-"))
                {
                    var range = trimmed.Split('-');
                    var start = int.Parse(range[0].Trim());
                    var end = int.Parse(range[1].Trim());
                    versions.AddRange(Enumerable.Range(start, end - start + 1));
                }
                else
                {
                    versions.Add(int.Parse(trimmed));
                }
            }

            return versions.Distinct().OrderBy(v => v).ToArray();
        }
    }
} 