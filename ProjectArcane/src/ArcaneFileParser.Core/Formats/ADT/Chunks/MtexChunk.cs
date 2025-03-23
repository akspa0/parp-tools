using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Validation;

namespace ArcaneFileParser.Core.Formats.ADT.Chunks;

/// <summary>
/// Map Texture chunk containing a list of texture filenames.
/// </summary>
public class MtexChunk : ChunkBase
{
    public override string ChunkId => "MTEX";

    /// <summary>
    /// Gets the list of texture filenames.
    /// </summary>
    public List<string> TextureNames { get; } = new();

    /// <summary>
    /// Gets the list of validation issues found in texture paths.
    /// </summary>
    public List<AssetValidationIssue> ValidationIssues { get; } = new();

    /// <summary>
    /// Gets whether all texture paths are valid.
    /// </summary>
    public bool HasValidPaths => ValidationIssues.Count == 0;

    public override void Parse(BinaryReader reader, uint size)
    {
        var startPosition = reader.BaseStream.Position;
        var endPosition = startPosition + size;

        // Clear existing data
        TextureNames.Clear();
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
                var texturePath = stringBuilder.ToString();
                TextureNames.Add(texturePath);
                ValidateTexturePath(texturePath);
            }
        }
    }

    protected override void WriteContent(BinaryWriter writer)
    {
        // Write each texture name as a null-terminated string
        foreach (var name in TextureNames)
        {
            foreach (char c in name)
            {
                writer.Write((byte)c);
            }
            writer.Write((byte)0); // Null terminator
        }
    }

    /// <summary>
    /// Gets a texture name by index.
    /// </summary>
    /// <param name="index">Index of the texture.</param>
    /// <returns>The texture name if found, null otherwise.</returns>
    public string? GetTextureName(int index)
    {
        if (index < 0 || index >= TextureNames.Count)
            return null;

        return TextureNames[index];
    }

    /// <summary>
    /// Validates a texture path and adds any issues to the validation list.
    /// </summary>
    private void ValidateTexturePath(string texturePath)
    {
        var validator = AssetPathValidator.Instance;
        var listfile = ListfileManager.Instance;

        // Check if path exists in listfile
        if (!listfile.HasPath(texturePath))
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.MissingFile,
                $"Texture path not found in listfile: {texturePath}",
                texturePath
            ));
            return;
        }

        // Validate texture path format
        var validationResult = validator.ValidateTexturePath(texturePath);
        if (!validationResult.IsValid)
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.InvalidPath,
                $"Invalid texture path format: {validationResult.Message}",
                texturePath,
                validationResult.SuggestedFix
            ));
        }

        // Check for correct extension (.blp)
        if (!texturePath.EndsWith(".blp", System.StringComparison.OrdinalIgnoreCase))
        {
            ValidationIssues.Add(new AssetValidationIssue(
                AssetValidationIssueType.InvalidExtension,
                $"Texture path has incorrect extension (should be .blp): {texturePath}",
                texturePath,
                texturePath.Substring(0, texturePath.LastIndexOf('.')) + ".blp"
            ));
        }
    }

    public override string ToHumanReadable()
    {
        var builder = new StringBuilder();
        builder.AppendLine($"Texture Count: {TextureNames.Count}");
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

        builder.AppendLine("\nTexture List:");
        for (int i = 0; i < TextureNames.Count; i++)
        {
            builder.AppendLine($"[{i}] {TextureNames[i]}");
        }

        return builder.ToString();
    }
} 