using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace ArcaneFileParser.Core.Common;

/// <summary>
/// Provides advanced validation and pattern matching for World of Warcraft asset paths.
/// </summary>
public class AssetPathValidator
{
    private static readonly string[] ValidExtensions = new[]
    {
        ".wmo",      // World Map Object
        ".m2",       // Model
        ".mdx",      // Legacy Model (pre-WotLK M2)
        ".blp",      // Texture
        ".skin",     // Model Skin
        ".anim",     // Animation
        ".bone",     // Skeleton
        ".phys",     // Physics
        ".skel",     // Skeleton
    };

    private static readonly string[] ValidRootFolders = new[]
    {
        "world",
        "buildings",
        "creature",
        "character",
        "spells",
        "interface",
        "item",
        "environment",
        "dungeons",
        "doodads",
    };

    // Track missing files globally
    private static readonly HashSet<string> _missingFiles = new();
    private static readonly HashSet<uint> _missingFileDataIds = new();

    /// <summary>
    /// Gets all missing files encountered during validation.
    /// </summary>
    public static IReadOnlySet<string> MissingFiles => _missingFiles;

    /// <summary>
    /// Gets all missing FileDataIDs encountered during validation.
    /// </summary>
    public static IReadOnlySet<uint> MissingFileDataIds => _missingFileDataIds;

    /// <summary>
    /// Logs a missing file.
    /// </summary>
    /// <param name="path">The missing file path.</param>
    /// <param name="fileDataId">Optional FileDataID of the missing file.</param>
    public static void LogMissingFile(string path, uint? fileDataId = null)
    {
        _missingFiles.Add(path);
        if (fileDataId.HasValue)
            _missingFileDataIds.Add(fileDataId.Value);
    }

    /// <summary>
    /// Clears the missing files log.
    /// </summary>
    public static void ClearMissingFiles()
    {
        _missingFiles.Clear();
        _missingFileDataIds.Clear();
    }

    /// <summary>
    /// Gets a report of all missing files.
    /// </summary>
    /// <returns>A formatted report string.</returns>
    public static string GetMissingFilesReport()
    {
        var report = new System.Text.StringBuilder();
        report.AppendLine("Missing Files Report");
        report.AppendLine("------------------");
        report.AppendLine($"Total Missing Files: {_missingFiles.Count}");
        report.AppendLine($"Total Missing FileDataIDs: {_missingFileDataIds.Count}");
        
        if (_missingFiles.Any())
        {
            report.AppendLine("\nMissing File Paths:");
            foreach (var path in _missingFiles.OrderBy(p => p))
            {
                report.AppendLine($"  {path}");
            }
        }

        if (_missingFileDataIds.Any())
        {
            report.AppendLine("\nMissing FileDataIDs:");
            foreach (var id in _missingFileDataIds.OrderBy(id => id))
            {
                report.AppendLine($"  {id}");
            }
        }

        return report.ToString();
    }

    /// <summary>
    /// Validates a path against common World of Warcraft asset path patterns.
    /// </summary>
    /// <param name="path">The path to validate.</param>
    /// <param name="issues">List of validation issues found.</param>
    /// <returns>True if the path is valid, false if any issues were found.</returns>
    public static bool ValidatePath(string path, out List<string> issues)
    {
        issues = new List<string>();

        // Basic checks
        if (string.IsNullOrWhiteSpace(path))
        {
            issues.Add("Path is empty or whitespace");
            return false;
        }

        // Normalize path separators and trim
        path = path.Replace('\\', '/').Trim();

        // Check for invalid characters (beyond normal path characters)
        var invalidChars = path.Where(c => !char.IsLetterOrDigit(c) && c != '/' && c != '_' && c != '-' && c != '.').ToList();
        if (invalidChars.Any())
        {
            issues.Add($"Path contains invalid characters: {string.Join(", ", invalidChars.Select(c => $"'{c}'"))}");
        }

        // Check extension and handle legacy .mdx files
        var extension = Path.GetExtension(path).ToLowerInvariant();
        if (!ValidExtensions.Contains(extension))
        {
            issues.Add($"Invalid file extension: {extension}. Expected one of: {string.Join(", ", ValidExtensions)}");
        }
        else if (extension == ".mdx")
        {
            issues.Add("Warning: Using legacy .mdx extension, modern equivalent would be .m2");
        }

        // Check root folder
        var firstFolder = path.Split('/').FirstOrDefault()?.ToLowerInvariant();
        if (firstFolder == null || !ValidRootFolders.Contains(firstFolder))
        {
            issues.Add($"Invalid root folder: {firstFolder ?? "none"}. Expected one of: {string.Join(", ", ValidRootFolders)}");
        }

        // Check for double slashes
        if (path.Contains("//"))
        {
            issues.Add("Path contains double slashes");
        }

        // Check for proper casing patterns (e.g., no ALL CAPS files)
        if (path.Any(char.IsUpper))
        {
            var fileNameWithoutExt = Path.GetFileNameWithoutExtension(path);
            if (fileNameWithoutExt.All(c => char.IsUpper(c) || !char.IsLetter(c)))
            {
                issues.Add("File name should not be all uppercase");
            }
        }

        return !issues.Any();
    }

    /// <summary>
    /// Checks if a path matches common World of Warcraft asset naming patterns.
    /// </summary>
    /// <param name="path">The path to check.</param>
    /// <returns>A tuple indicating (isValid, matchedPattern, suggestedFix).</returns>
    public static (bool IsValid, string? MatchedPattern, string? SuggestedFix) MatchPattern(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return (false, null, null);

        path = path.Replace('\\', '/').Trim();

        // Common WMO patterns
        var wmoPatterns = new Dictionary<string, string>
        {
            { @"^world/wmo/([^/]+)/([^/]+)\.wmo$", "World WMO" },
            { @"^buildings/([^/]+)/([^/]+)\.wmo$", "Building WMO" },
            { @"^dungeons/([^/]+)/([^/]+)\.wmo$", "Dungeon WMO" }
        };

        // Common M2/MDX patterns (support both extensions)
        var modelPatterns = new Dictionary<string, string>
        {
            { @"^creature/([^/]+)/([^/]+)\.(m2|mdx)$", "Creature Model" },
            { @"^character/([^/]+)/([^/]+)\.(m2|mdx)$", "Character Model" },
            { @"^item/([^/]+)/([^/]+)\.(m2|mdx)$", "Item Model" }
        };

        // Check WMO patterns
        if (path.EndsWith(".wmo", StringComparison.OrdinalIgnoreCase))
        {
            foreach (var (pattern, name) in wmoPatterns)
            {
                if (Regex.IsMatch(path, pattern, RegexOptions.IgnoreCase))
                    return (true, name, null);
            }

            // Suggest a fix if it's close to a known pattern
            var suggestedPath = SuggestWmoPath(path);
            if (suggestedPath != null)
                return (false, "Invalid WMO Path", suggestedPath);
        }

        // Check M2/MDX patterns
        if (path.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) || 
            path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            foreach (var (pattern, name) in modelPatterns)
            {
                if (Regex.IsMatch(path, pattern, RegexOptions.IgnoreCase))
                {
                    var result = (true, name, (string?)null);
                    
                    // Suggest upgrading MDX to M2 if applicable
                    if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                    {
                        result.Item3 = path.Substring(0, path.Length - 4) + ".m2";
                    }
                    
                    return result;
                }
            }

            // Suggest a fix if it's close to a known pattern
            var suggestedPath = SuggestM2Path(path);
            if (suggestedPath != null)
                return (false, "Invalid Model Path", suggestedPath);
        }

        return (false, null, null);
    }

    /// <summary>
    /// Gets suggested fixes for common path issues.
    /// </summary>
    /// <param name="path">The path to analyze.</param>
    /// <returns>A list of suggested fixes and their descriptions.</returns>
    public static IEnumerable<(string Description, string Fix)> GetSuggestedFixes(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            yield break;

        path = path.Replace('\\', '/').Trim();

        // Check and fix casing
        var fileName = Path.GetFileName(path);
        if (fileName.All(c => char.IsUpper(c) || !char.IsLetter(c)))
        {
            yield return (
                "File name is all uppercase",
                path.Replace(fileName, fileName.ToLowerInvariant())
            );
        }

        // Check and fix double slashes
        if (path.Contains("//"))
        {
            yield return (
                "Path contains double slashes",
                path.Replace("//", "/")
            );
        }

        // Check and suggest root folder
        var firstFolder = path.Split('/').FirstOrDefault()?.ToLowerInvariant();
        if (firstFolder != null && !ValidRootFolders.Contains(firstFolder))
        {
            var bestMatch = ValidRootFolders
                .OrderBy(r => LevenshteinDistance(r, firstFolder))
                .First();

            yield return (
                $"Invalid root folder: {firstFolder}",
                path.Replace(firstFolder, bestMatch)
            );
        }

        // Check and fix extension casing
        var extension = Path.GetExtension(path);
        if (extension != extension.ToLowerInvariant())
        {
            yield return (
                "File extension should be lowercase",
                path.Replace(extension, extension.ToLowerInvariant())
            );
        }

        // Suggest upgrading MDX to M2
        if (path.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            yield return (
                "Legacy .mdx extension detected",
                path.Substring(0, path.Length - 4) + ".m2"
            );
        }
    }

    private static string? SuggestWmoPath(string path)
    {
        var fileName = Path.GetFileName(path);
        var folders = path.Split('/').ToList();

        if (folders.Count < 2)
            return $"world/wmo/general/{fileName}";

        if (!ValidRootFolders.Contains(folders[0].ToLowerInvariant()))
            return $"world/wmo/{folders[^2]}/{fileName}";

        return null;
    }

    private static string? SuggestM2Path(string path)
    {
        var fileName = Path.GetFileName(path);
        
        // Convert .mdx to .m2 if needed
        if (fileName.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
        {
            fileName = Path.ChangeExtension(fileName, ".m2");
        }
        
        var folders = path.Split('/').ToList();

        if (folders.Count < 2)
            return $"creature/generic/{fileName}";

        if (!ValidRootFolders.Contains(folders[0].ToLowerInvariant()))
        {
            var category = folders[^2].ToLowerInvariant();
            if (category.Contains("character"))
                return $"character/{folders[^2]}/{fileName}";
            if (category.Contains("creature"))
                return $"creature/{folders[^2]}/{fileName}";
            if (category.Contains("item"))
                return $"item/{folders[^2]}/{fileName}";
            return $"creature/{folders[^2]}/{fileName}";
        }

        return null;
    }

    private static int LevenshteinDistance(string s, string t)
    {
        var m = s.Length;
        var n = t.Length;
        var d = new int[m + 1, n + 1];

        for (var i = 0; i <= m; i++)
            d[i, 0] = i;
        for (var j = 0; j <= n; j++)
            d[0, j] = j;

        for (var j = 1; j <= n; j++)
        for (var i = 1; i <= m; i++)
            if (s[i - 1] == t[j - 1])
                d[i, j] = d[i - 1, j - 1];
            else
                d[i, j] = Math.Min(Math.Min(
                    d[i - 1, j] + 1,    // deletion
                    d[i, j - 1] + 1),   // insertion
                    d[i - 1, j - 1] + 1 // substitution
                );

        return d[m, n];
    }
} 