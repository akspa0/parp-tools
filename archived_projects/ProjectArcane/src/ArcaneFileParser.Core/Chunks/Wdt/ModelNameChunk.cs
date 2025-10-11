using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common;

namespace ArcaneFileParser.Core.Chunks.Wdt;

/// <summary>
/// Model name chunk containing WMO file paths for model placements.
/// </summary>
public class ModelNameChunk : ChunkBase
{
    /// <summary>
    /// The expected signature for MWID chunks.
    /// </summary>
    public const uint ExpectedSignature = 0x4449574D; // "MWID"

    /// <summary>
    /// Gets the array of model file paths.
    /// </summary>
    public string[] ModelPaths { get; }

    /// <summary>
    /// Gets the array of FileDataIDs for each model path, if they exist in the listfile.
    /// </summary>
    public uint?[] FileDataIds { get; }

    /// <summary>
    /// Gets whether the listfile has been initialized and is available for validation.
    /// </summary>
    public bool HasListfile => ListfileManager.Instance._isInitialized;

    /// <summary>
    /// Gets validation results for each path.
    /// </summary>
    public Dictionary<uint, List<string>> ValidationIssues { get; } = new();

    /// <summary>
    /// Gets pattern matching results for each path.
    /// </summary>
    public Dictionary<uint, (bool IsValid, string? Pattern, string? Suggestion)> PatternResults { get; } = new();

    /// <summary>
    /// Gets suggested fixes for each path.
    /// </summary>
    public Dictionary<uint, List<(string Description, string Fix)>> SuggestedFixes { get; } = new();

    /// <summary>
    /// Creates a new instance of the MWID chunk.
    /// </summary>
    /// <param name="reader">The binary reader to read from.</param>
    public ModelNameChunk(BinaryReader reader) : base(reader)
    {
        if (!ValidateSignature(ExpectedSignature))
        {
            return;
        }

        // Each entry is a null-terminated string
        var paths = new List<string>();
        var fileDataIds = new List<uint?>();
        long endPosition = reader.BaseStream.Position + Size;

        while (reader.BaseStream.Position < endPosition)
        {
            // Read until we find a null terminator or reach the end
            var pathBytes = new List<byte>();
            byte b;
            while ((b = reader.ReadByte()) != 0 && reader.BaseStream.Position < endPosition)
            {
                pathBytes.Add(b);
            }

            if (pathBytes.Count > 0)
            {
                // Convert the bytes to a string using ASCII encoding (WoW uses ASCII for paths)
                string path = Encoding.ASCII.GetString(pathBytes.ToArray());
                paths.Add(path);

                // Try to get the FileDataID if the listfile is available
                var fileDataId = ListfileManager.Instance.GetFileDataId(path);
                fileDataIds.Add(fileDataId);

                // Log missing files
                if (!ListfileManager.Instance.ValidatePath(path))
                {
                    AssetPathValidator.LogMissingFile(path, fileDataId);
                }

                // Validate the path
                uint index = (uint)paths.Count - 1;
                if (AssetPathValidator.ValidatePath(path, out var issues))
                {
                    ValidationIssues[index] = issues;
                }

                // Check path pattern
                var patternResult = AssetPathValidator.MatchPattern(path);
                PatternResults[index] = patternResult;

                // Get suggested fixes
                var fixes = AssetPathValidator.GetSuggestedFixes(path).ToList();
                if (fixes.Any())
                {
                    SuggestedFixes[index] = fixes;
                }
            }
        }

        ModelPaths = paths.ToArray();
        FileDataIds = fileDataIds.ToArray();
        EnsureAtEnd(reader);
    }

    /// <summary>
    /// Gets a model path by its index.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>The model path at the specified index, or null if the index is invalid.</returns>
    public string? GetPath(uint index)
    {
        if (!IsValid || index >= ModelPaths.Length)
            return null;

        return ModelPaths[index];
    }

    /// <summary>
    /// Gets the FileDataID for a model path at the specified index.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>The FileDataID if found in the listfile, or null if not found or index is invalid.</returns>
    public uint? GetFileDataId(uint index)
    {
        if (!IsValid || index >= FileDataIds.Length)
            return null;

        return FileDataIds[index];
    }

    /// <summary>
    /// Gets validation issues for a path at the specified index.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>List of validation issues, or null if no issues or index is invalid.</returns>
    public List<string>? GetValidationIssues(uint index)
    {
        return ValidationIssues.TryGetValue(index, out var issues) ? issues : null;
    }

    /// <summary>
    /// Gets pattern matching results for a path at the specified index.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>Pattern matching results, or null if index is invalid.</returns>
    public (bool IsValid, string? Pattern, string? Suggestion)? GetPatternResult(uint index)
    {
        return PatternResults.TryGetValue(index, out var result) ? result : null;
    }

    /// <summary>
    /// Gets suggested fixes for a path at the specified index.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>List of suggested fixes, or null if no fixes or index is invalid.</returns>
    public List<(string Description, string Fix)>? GetSuggestedFixes(uint index)
    {
        return SuggestedFixes.TryGetValue(index, out var fixes) ? fixes : null;
    }

    /// <summary>
    /// Validates if a model path at the specified index exists in the listfile.
    /// </summary>
    /// <param name="index">The index into the name table.</param>
    /// <returns>True if the path exists in the listfile, false otherwise.</returns>
    public bool ValidatePath(uint index)
    {
        var path = GetPath(index);
        if (path == null)
            return false;

        var isValid = ListfileManager.Instance.ValidatePath(path);
        if (!isValid)
        {
            AssetPathValidator.LogMissingFile(path, GetFileDataId(index));
        }
        return isValid;
    }

    /// <summary>
    /// Gets validation statistics for all model paths in this chunk.
    /// </summary>
    /// <returns>A tuple containing (total paths, valid paths, invalid paths).</returns>
    public (int Total, int Valid, int Invalid) GetValidationStats()
    {
        if (!IsValid)
            return (0, 0, 0);

        int valid = 0, invalid = 0;
        for (uint i = 0; i < ModelPaths.Length; i++)
        {
            if (ValidatePath(i))
                valid++;
            else
                invalid++;
        }

        return (ModelPaths.Length, valid, invalid);
    }

    /// <summary>
    /// Gets all invalid model paths in this chunk.
    /// </summary>
    /// <returns>An enumerable of invalid paths and their indices.</returns>
    public IEnumerable<(uint Index, string Path)> GetInvalidPaths()
    {
        if (!IsValid)
            yield break;

        for (uint i = 0; i < ModelPaths.Length; i++)
        {
            if (!ValidatePath(i))
                yield return (i, ModelPaths[i]);
        }
    }

    /// <summary>
    /// Creates a string representation of the chunk for debugging.
    /// </summary>
    public override string ToString()
    {
        var stats = GetValidationStats();
        var patternStats = PatternResults.Values.Count(r => r.IsValid);
        return $"MWID [Size: {Size}, Models: {ModelPaths.Length}, Valid: {IsValid}, Validated: {stats.Valid}/{stats.Total}, Pattern Matches: {patternStats}/{ModelPaths.Length}]";
    }

    /// <summary>
    /// Gets a detailed report of the model paths in this chunk.
    /// </summary>
    /// <returns>A formatted report string.</returns>
    public string GetDetailedReport()
    {
        var report = new StringBuilder();
        report.AppendLine($"MWID Chunk Report");
        report.AppendLine("---------------");
        report.AppendLine($"Total Models: {ModelPaths.Length}");
        report.AppendLine($"Chunk Valid: {IsValid}");

        var stats = GetValidationStats();
        report.AppendLine($"\nValidation Stats:");
        report.AppendLine($"  Valid Paths: {stats.Valid}/{stats.Total}");
        report.AppendLine($"  Invalid Paths: {stats.Invalid}/{stats.Total}");

        var patternStats = PatternResults.Values.Count(r => r.IsValid);
        report.AppendLine($"  Pattern Matches: {patternStats}/{ModelPaths.Length}");

        // Sample of valid paths
        var validPaths = ModelPaths.Select((path, i) => (Index: (uint)i, Path: path))
            .Where(x => ValidatePath(x.Index))
            .Take(5);

        if (validPaths.Any())
        {
            report.AppendLine("\nSample Valid Paths:");
            foreach (var (index, path) in validPaths)
            {
                var fileDataId = GetFileDataId(index);
                var pattern = GetPatternResult(index)?.Pattern;
                report.AppendLine($"  [{index}] {path}");
                report.AppendLine($"    FileDataID: {fileDataId?.ToString() ?? "Unknown"}");
                if (pattern != null)
                    report.AppendLine($"    Pattern: {pattern}");
            }
        }

        // Sample of invalid paths with issues
        var invalidPaths = GetInvalidPaths().Take(5);
        if (invalidPaths.Any())
        {
            report.AppendLine("\nSample Invalid Paths:");
            foreach (var (index, path) in invalidPaths)
            {
                report.AppendLine($"  [{index}] {path}");
                
                var issues = GetValidationIssues(index);
                if (issues != null)
                {
                    foreach (var issue in issues)
                        report.AppendLine($"    Issue: {issue}");
                }

                var fixes = GetSuggestedFixes(index);
                if (fixes != null)
                {
                    foreach (var (description, fix) in fixes)
                        report.AppendLine($"    Fix ({description}): {fix}");
                }
            }
        }

        return report.ToString();
    }
} 