using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Validation;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map WMO Filenames chunk containing a list of WMO model filenames.
/// </summary>
public class MwmoChunk : ChunkBase
{
    public override string ChunkId => "MWMO";

    /// <summary>
    /// Gets the list of WMO model filenames.
    /// </summary>
    public List<string> WmoNames { get; } = new();

    /// <summary>
    /// Gets the list of validation issues found in WMO paths.
    /// </summary>
    public List<AssetValidationIssue> ValidationIssues { get; } = new();

    /// <summary>
    /// Gets whether all WMO paths are valid.
    /// </summary>
    public bool HasValidPaths => ValidationIssues.Count == 0;

    /// <summary>
    /// Gets the offset of a WMO name in the chunk data.
    /// </summary>
    /// <param name="index">Index of the WMO name.</param>
    /// <returns>Offset of the WMO name, or -1 if not found.</returns>
    public int GetNameOffset(int index)
    {
        if (index < 0 || index >= WmoNames.Count)
            return -1;

        int offset = 0;
        for (int i = 0; i < index; i++)
        {
            // Add 1 for null terminator
            offset += WmoNames[i].Length + 1;
        }
        return offset;
    }

    public override void Parse(BinaryReader reader, uint size)
    {
        var startPosition = reader.BaseStream.Position;
        var endPosition = startPosition + size;

        // Clear existing data
        WmoNames.Clear();
        ValidationIssues.Clear();

        // Read null-terminated strings until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var stringBuilder = new StringBuilder();
            char c;
            while ((c = (char)reader.ReadByte()) != '\0')
            {
                stringBuilder.Append(c);
            }

            if (stringBuilder.Length > 0)
            {
                var wmoPath = stringBuilder.ToString();
                WmoNames.Add(wmoPath);
                ValidateWmoPath(wmoPath);
            }
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each WMO name as a null-terminated string
        foreach (var name in WmoNames)
        {
            foreach (char c in name)
            {
                writer.Write((byte)c);
            }
            writer.Write((byte)0); // Null terminator
        }
    }

    /// <summary>
    /// Gets a WMO name by index.
    /// </summary>
    /// <param name="index">Index of the WMO.</param>
    /// <returns>The WMO name if found, null otherwise.</returns>
    public string? GetWmoName(int index)
    {
        if (index < 0 || index >= WmoNames.Count)
            return null;

        return WmoNames[index];
    }

    /// <summary>
    /// Validates a WMO path and adds any issues to the validation list.
    /// </summary>
    private void ValidateWmoPath(string wmoPath)
    {
        var validator = AssetPathValidator.Instance;
        var listfile = ListfileManager.Instance;

        // Check if path exists in listfile
        if (!listfile.HasPath(wmoPath))
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.MissingFile,
                $"WMO path not found in listfile: {wmoPath}",
                wmoPath
            ));
            return;
        }

        // Validate WMO path format
        var validationResult = validator.ValidateWmoPath(wmoPath);
        if (!validationResult.IsValid)
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.InvalidPath,
                $"Invalid WMO path format: {validationResult.Message}",
                wmoPath,
                validationResult.SuggestedFix
            ));
        }

        // Check for correct extension (.wmo)
        if (!wmoPath.EndsWith(".wmo", System.StringComparison.OrdinalIgnoreCase))
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.InvalidExtension,
                $"WMO path has incorrect extension (should be .wmo): {wmoPath}",
                wmoPath,
                wmoPath.Substring(0, wmoPath.LastIndexOf('.')) + ".wmo"
            ));
        }
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"WMO Count: {WmoNames.Count}");
        builder.AppendLine($"Valid Paths: {(HasValidPaths ? "Yes" : "No")}");
        
        if (ValidationIssues.Count > 0)
        {
            builder.AppendLine("\nValidation Issues:");
            foreach (var issue in ValidationIssues)
            {
                builder.AppendLine($"- [{issue.Type}] {issue.Message}");
                if (!string.IsNullOrEmpty(issue.SuggestedFix))
                {
                    builder.AppendLine($"  Suggested Fix: {issue.SuggestedFix}");
                }
            }
            builder.AppendLine();
        }

        builder.AppendLine("\nWMO List:");
        for (int i = 0; i < WmoNames.Count; i++)
        {
            builder.AppendLine($"[{i}] {WmoNames[i]} (Offset: 0x{GetNameOffset(i):X8})");
        }

        return builder.ToString();
    }
} 