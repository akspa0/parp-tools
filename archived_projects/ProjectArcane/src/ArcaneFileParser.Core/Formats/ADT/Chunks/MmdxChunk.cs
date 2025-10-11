using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Validation;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Model Filenames chunk containing a list of M2 model filenames.
/// </summary>
public class MmdxChunk : ChunkBase
{
    public override string ChunkId => "MMDX";

    /// <summary>
    /// Gets the list of M2 model filenames.
    /// </summary>
    public List<string> ModelNames { get; } = new();

    /// <summary>
    /// Gets the list of validation issues found in model paths.
    /// </summary>
    public List<AssetValidationIssue> ValidationIssues { get; } = new();

    /// <summary>
    /// Gets whether all model paths are valid.
    /// </summary>
    public bool HasValidPaths => ValidationIssues.Count == 0;

    /// <summary>
    /// Gets the offset of a model name in the chunk data.
    /// </summary>
    /// <param name="index">Index of the model name.</param>
    /// <returns>Offset of the model name, or -1 if not found.</returns>
    public int GetNameOffset(int index)
    {
        if (index < 0 || index >= ModelNames.Count)
            return -1;

        int offset = 0;
        for (int i = 0; i < index; i++)
        {
            // Add 1 for null terminator
            offset += ModelNames[i].Length + 1;
        }
        return offset;
    }

    public override void Parse(BinaryReader reader, uint size)
    {
        var startPosition = reader.BaseStream.Position;
        var endPosition = startPosition + size;

        // Clear existing data
        ModelNames.Clear();
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
                var modelPath = stringBuilder.ToString();
                ModelNames.Add(modelPath);
                ValidateModelPath(modelPath);
            }
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each model name as a null-terminated string
        foreach (var name in ModelNames)
        {
            foreach (char c in name)
            {
                writer.Write((byte)c);
            }
            writer.Write((byte)0); // Null terminator
        }
    }

    /// <summary>
    /// Gets a model name by index.
    /// </summary>
    /// <param name="index">Index of the model.</param>
    /// <returns>The model name if found, null otherwise.</returns>
    public string? GetModelName(int index)
    {
        if (index < 0 || index >= ModelNames.Count)
            return null;

        return ModelNames[index];
    }

    /// <summary>
    /// Validates a model path and adds any issues to the validation list.
    /// </summary>
    private void ValidateModelPath(string modelPath)
    {
        var validator = AssetPathValidator.Instance;
        var listfile = ListfileManager.Instance;

        // Check if path exists in listfile
        if (!listfile.HasPath(modelPath))
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.MissingFile,
                $"Model path not found in listfile: {modelPath}",
                modelPath
            ));
            return;
        }

        // Validate model path format
        var validationResult = validator.ValidateModelPath(modelPath);
        if (!validationResult.IsValid)
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.InvalidPath,
                $"Invalid model path format: {validationResult.Message}",
                modelPath,
                validationResult.SuggestedFix
            ));
        }

        // Check for correct extension (.m2)
        if (!modelPath.EndsWith(".m2", System.StringComparison.OrdinalIgnoreCase))
        {
            // Special handling for .mdx files - suggest converting to .m2
            if (modelPath.EndsWith(".mdx", System.StringComparison.OrdinalIgnoreCase))
            {
                ValidationIssues.Add(new AssetValidationIssue(
                    AssetValidationIssueType.LegacyFormat,
                    $"Model uses legacy MDX format: {modelPath}",
                    modelPath,
                    modelPath.Substring(0, modelPath.Length - 4) + ".m2"
                ));
            }
            else
            {
                ValidationIssues.Add(new AssetValidationIssue(
                    AssetValidationIssueType.InvalidExtension,
                    $"Model path has incorrect extension (should be .m2): {modelPath}",
                    modelPath,
                    modelPath.Substring(0, modelPath.LastIndexOf('.')) + ".m2"
                ));
            }
        }
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Model Count: {ModelNames.Count}");
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

        builder.AppendLine("\nModel List:");
        for (int i = 0; i < ModelNames.Count; i++)
        {
            builder.AppendLine($"[{i}] {ModelNames[i]} (Offset: 0x{GetNameOffset(i):X8})");
        }

        return builder.ToString();
    }
} 