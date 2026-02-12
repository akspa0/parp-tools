using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

namespace ArcaneFileParser.Core.Common;

/// <summary>
/// Manages the community listfile data for World of Warcraft asset validation.
/// </summary>
public class ListfileManager
{
    private static ListfileManager? _instance;
    private static readonly object _lock = new();

    /// <summary>
    /// Gets the singleton instance of the ListfileManager.
    /// </summary>
    public static ListfileManager Instance
    {
        get
        {
            lock (_lock)
            {
                _instance ??= new ListfileManager();
                return _instance;
            }
        }
    }

    private readonly Dictionary<uint, string> _fileDataIdToPath = new();
    private readonly Dictionary<string, uint> _pathToFileDataId = new();
    private bool _isInitialized;

    private ListfileManager()
    {
    }

    /// <summary>
    /// Initializes the listfile manager by downloading and parsing the latest community listfile.
    /// </summary>
    /// <param name="localCachePath">Optional path to cache the downloaded listfile.</param>
    /// <returns>A task representing the initialization operation.</returns>
    public async Task Initialize(string? localCachePath = null)
    {
        if (_isInitialized)
            return;

        string listfileContent;
        if (localCachePath != null && File.Exists(localCachePath))
        {
            // Use cached file if available
            listfileContent = await File.ReadAllTextAsync(localCachePath);
        }
        else
        {
            // Download the latest listfile
            using var client = new HttpClient();
            listfileContent = await client.GetStringAsync("https://github.com/wowdev/wow-listfile/releases/latest/download/community-listfile-withcapitals.csv");

            // Cache the file if requested
            if (localCachePath != null)
            {
                await File.WriteAllTextAsync(localCachePath, listfileContent);
            }
        }

        // Parse the CSV content
        using var reader = new StringReader(listfileContent);
        string? line;
        while ((line = reader.ReadLine()) != null)
        {
            var parts = line.Split(',');
            if (parts.Length >= 2 && uint.TryParse(parts[0], out uint fileDataId))
            {
                var path = parts[1].Trim('"');
                _fileDataIdToPath[fileDataId] = path;
                _pathToFileDataId[path.ToLowerInvariant()] = fileDataId;
            }
        }

        _isInitialized = true;
    }

    /// <summary>
    /// Gets the file path associated with a FileDataID.
    /// </summary>
    /// <param name="fileDataId">The FileDataID to look up.</param>
    /// <returns>The associated file path, or null if not found.</returns>
    public string? GetPath(uint fileDataId)
    {
        return _fileDataIdToPath.TryGetValue(fileDataId, out var path) ? path : null;
    }

    /// <summary>
    /// Gets the FileDataID associated with a file path.
    /// </summary>
    /// <param name="path">The file path to look up.</param>
    /// <returns>The associated FileDataID, or null if not found.</returns>
    public uint? GetFileDataId(string path)
    {
        return _pathToFileDataId.TryGetValue(path.ToLowerInvariant(), out var fileDataId) ? fileDataId : null;
    }

    /// <summary>
    /// Validates if a given path exists in the listfile.
    /// </summary>
    /// <param name="path">The path to validate.</param>
    /// <returns>True if the path exists in the listfile, false otherwise.</returns>
    public bool ValidatePath(string path)
    {
        return _pathToFileDataId.ContainsKey(path.ToLowerInvariant());
    }

    /// <summary>
    /// Gets all paths that match a specific pattern.
    /// </summary>
    /// <param name="pattern">The pattern to match against (case-insensitive).</param>
    /// <returns>An enumerable of matching paths.</returns>
    public IEnumerable<string> FindPaths(string pattern)
    {
        var lowerPattern = pattern.ToLowerInvariant();
        foreach (var path in _fileDataIdToPath.Values)
        {
            if (path.ToLowerInvariant().Contains(lowerPattern))
                yield return path;
        }
    }

    /// <summary>
    /// Gets all paths for a specific file type.
    /// </summary>
    /// <param name="extension">The file extension to filter by (e.g., ".wmo").</param>
    /// <returns>An enumerable of matching paths.</returns>
    public IEnumerable<string> GetPathsByType(string extension)
    {
        var lowerExt = extension.ToLowerInvariant();
        foreach (var path in _fileDataIdToPath.Values)
        {
            if (Path.GetExtension(path).ToLowerInvariant() == lowerExt)
                yield return path;
        }
    }
} 