using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using WCAnalyzer.Core.Utilities;

namespace WCAnalyzer.Core.Services;

/// <summary>
/// Service for validating file references in ADT files.
/// </summary>
public class ReferenceValidator
{
    private readonly ILogger<ReferenceValidator> _logger;
    private Dictionary<string, uint> _pathToFileDataIdMap = new Dictionary<string, uint>(StringComparer.OrdinalIgnoreCase);
    private Dictionary<uint, string> _fileDataIdToPathMap = new Dictionary<uint, string>();
    private HashSet<string> _listfileAllCaps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Creates a new instance of the ReferenceValidator class.
    /// </summary>
    /// <param name="logger">The logger to use.</param>
    public ReferenceValidator(ILogger<ReferenceValidator> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Loads a listfile containing known good file references.
    /// </summary>
    /// <param name="listfilePath">The path to the listfile.</param>
    /// <returns>A set of normalized file paths from the listfile.</returns>
    public async Task<HashSet<string>> LoadListfileAsync(string? listfilePath)
    {
        var knownGoodFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        
        // Clear existing maps
        _pathToFileDataIdMap.Clear();
        _fileDataIdToPathMap.Clear();
        _listfileAllCaps.Clear();

        if (string.IsNullOrEmpty(listfilePath) || !File.Exists(listfilePath))
        {
            _logger.LogWarning("Listfile not provided or not found. Reference validation will be limited.");
            return knownGoodFiles;
        }

        try
        {
            foreach (var line in await File.ReadAllLinesAsync(listfilePath))
            {
                // Handle different listfile formats
                string path;
                uint fileDataId = 0;
                
                if (line.Contains(';'))
                {
                    // Format: ID;Path
                    var parts = line.Split(';', 2);
                    if (parts.Length < 2)
                        continue;

                    // Try to parse the FileDataID
                    if (uint.TryParse(parts[0], out fileDataId))
                    {
                        path = parts[1].Trim();
                    }
                    else
                    {
                        path = parts[1].Trim();
                        fileDataId = 0; // Invalid FileDataID
                    }
                }
                else
                {
                    // Format: Path only
                    path = line.Trim();
                    fileDataId = 0; // No FileDataID
                }

                if (!string.IsNullOrWhiteSpace(path))
                {
                    var normalizedPath = PathUtility.NormalizePath(path);
                    knownGoodFiles.Add(normalizedPath);
                    
                    // Store FileDataID mappings if available
                    if (fileDataId > 0)
                    {
                        _pathToFileDataIdMap[normalizedPath] = fileDataId;
                        _fileDataIdToPathMap[fileDataId] = normalizedPath;
                    }
                    
                    // Also keep a set with all capital letters for reference
                    _listfileAllCaps.Add(normalizedPath.ToUpperInvariant());
                }
            }

            _logger.LogInformation($"Loaded {knownGoodFiles.Count} entries from listfile. Found {_pathToFileDataIdMap.Count} FileDataID mappings.");
        }
        catch (Exception ex)
        {
            _logger.LogError($"Error loading listfile: {ex.Message}");
        }

        return knownGoodFiles;
    }

    /// <summary>
    /// Validates file references against a listfile.
    /// </summary>
    /// <param name="result">The ADT analysis result to validate.</param>
    /// <param name="knownGoodFiles">The set of known good files.</param>
    /// <returns>The number of invalid references found.</returns>
    public int ValidateReferences(AdtAnalysisResult result, HashSet<string> knownGoodFiles)
    {
        if (result == null)
            throw new ArgumentNullException(nameof(result));

        if (knownGoodFiles == null || knownGoodFiles.Count == 0)
        {
            // No listfile provided, mark all references as valid
            foreach (var reference in result.AllReferences)
            {
                reference.IsValid = true;
            }
            return 0;
        }

        int invalidCount = 0;

        // Validate texture references
        foreach (var reference in result.TextureReferences)
        {
            ValidateReference(reference, knownGoodFiles, ref invalidCount);
        }

        // Validate model references
        foreach (var reference in result.ModelReferences)
        {
            ValidateReference(reference, knownGoodFiles, ref invalidCount);
            
            // Special handling for m2/mdx compatibilities
            if (!reference.IsValid)
            {
                // Try alternate extension if this is a model file
                if (reference.NormalizedPath.EndsWith(".m2", StringComparison.OrdinalIgnoreCase))
                {
                    // Try .mdx extension
                    reference.AlternativeExtensionPath = reference.NormalizedPath.Substring(0, reference.NormalizedPath.Length - 3) + ".mdx";
                    reference.AlternativeExtensionFound = knownGoodFiles.Contains(reference.AlternativeExtensionPath);
                    
                    if (reference.AlternativeExtensionFound)
                    {
                        _logger.LogDebug($"Found .mdx alternative for {reference.OriginalPath}: {reference.AlternativeExtensionPath}");
                        reference.IsValid = true;
                        
                        // If the alternative path has a FileDataID, assign it to the reference
                        if (_pathToFileDataIdMap.TryGetValue(reference.AlternativeExtensionPath, out uint fileDataId))
                        {
                            reference.FileDataId = fileDataId;
                            _logger.LogDebug($"Assigned FileDataID {fileDataId} from alternative path to {reference.OriginalPath}");
                        }
                        
                        invalidCount--;
                    }
                }
                else if (reference.NormalizedPath.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase))
                {
                    // Try .m2 extension
                    reference.AlternativeExtensionPath = reference.NormalizedPath.Substring(0, reference.NormalizedPath.Length - 4) + ".m2";
                    reference.AlternativeExtensionFound = knownGoodFiles.Contains(reference.AlternativeExtensionPath);
                    
                    if (reference.AlternativeExtensionFound)
                    {
                        _logger.LogDebug($"Found .m2 alternative for {reference.OriginalPath}: {reference.AlternativeExtensionPath}");
                        reference.IsValid = true;
                        
                        // If the alternative path has a FileDataID, assign it to the reference
                        if (_pathToFileDataIdMap.TryGetValue(reference.AlternativeExtensionPath, out uint fileDataId))
                        {
                            reference.FileDataId = fileDataId;
                            _logger.LogDebug($"Assigned FileDataID {fileDataId} from alternative path to {reference.OriginalPath}");
                        }
                        
                        invalidCount--;
                    }
                }
                
                // Try all-caps version as a last resort
                if (!reference.IsValid && !string.IsNullOrEmpty(reference.NormalizedPath))
                {
                    string upperPath = reference.NormalizedPath.ToUpperInvariant();
                    if (_listfileAllCaps.Contains(upperPath))
                    {
                        _logger.LogDebug($"Found uppercase match for {reference.OriginalPath}");
                        reference.IsValid = true;
                        reference.ExistsInListfile = true;
                        
                        // Look for the FileDataID using the uppercase version
                        foreach (var entry in _pathToFileDataIdMap)
                        {
                            if (string.Equals(entry.Key, upperPath, StringComparison.OrdinalIgnoreCase))
                            {
                                reference.FileDataId = entry.Value;
                                _logger.LogDebug($"Assigned FileDataID {entry.Value} from uppercase match to {reference.OriginalPath}");
                                break;
                            }
                        }
                        
                        invalidCount--;
                    }
                }
            }
        }

        // Validate WMO references
        foreach (var reference in result.WmoReferences)
        {
            ValidateReference(reference, knownGoodFiles, ref invalidCount);
            
            // Try all-caps version as a last resort
            if (!reference.IsValid && !string.IsNullOrEmpty(reference.NormalizedPath))
            {
                string upperPath = reference.NormalizedPath.ToUpperInvariant();
                if (_listfileAllCaps.Contains(upperPath))
                {
                    _logger.LogDebug($"Found uppercase match for {reference.OriginalPath}");
                    reference.IsValid = true;
                    reference.ExistsInListfile = true;
                    
                    // Look for the FileDataID using the uppercase version
                    foreach (var entry in _pathToFileDataIdMap)
                    {
                        if (string.Equals(entry.Key, upperPath, StringComparison.OrdinalIgnoreCase))
                        {
                            reference.FileDataId = entry.Value;
                            _logger.LogDebug($"Assigned FileDataID {entry.Value} from uppercase match to {reference.OriginalPath}");
                            break;
                        }
                    }
                    
                    invalidCount--;
                }
            }
        }
        
        // Validate model placements to ensure they have FileDataIDs
        foreach (var placement in result.ModelPlacements)
        {
            if (!placement.UsesFileDataId && !string.IsNullOrEmpty(placement.Name))
            {
                // Try to find the FileDataID for this model
                var modelRef = result.ModelReferences.FirstOrDefault(r => 
                    string.Equals(r.OriginalPath, placement.Name, StringComparison.OrdinalIgnoreCase));
                
                if (modelRef != null && modelRef.FileDataId > 0)
                {
                    placement.FileDataId = modelRef.FileDataId;
                    _logger.LogDebug($"Assigned FileDataID {modelRef.FileDataId} to model placement {placement.UniqueId}");
                }
            }
        }
        
        // Validate WMO placements to ensure they have FileDataIDs
        foreach (var placement in result.WmoPlacements)
        {
            if (!placement.UsesFileDataId && !string.IsNullOrEmpty(placement.Name))
            {
                // Try to find the FileDataID for this WMO
                var wmoRef = result.WmoReferences.FirstOrDefault(r => 
                    string.Equals(r.OriginalPath, placement.Name, StringComparison.OrdinalIgnoreCase));
                
                if (wmoRef != null && wmoRef.FileDataId > 0)
                {
                    placement.FileDataId = wmoRef.FileDataId;
                    _logger.LogDebug($"Assigned FileDataID {wmoRef.FileDataId} to WMO placement {placement.UniqueId}");
                }
            }
        }

        return invalidCount;
    }
    
    /// <summary>
    /// Validates a single file reference against the listfile and FileDataID mappings.
    /// </summary>
    /// <param name="reference">The reference to validate.</param>
    /// <param name="knownGoodFiles">The set of known good files.</param>
    /// <param name="invalidCount">A reference to the running total of invalid references.</param>
    private void ValidateReference(FileReference reference, HashSet<string> knownGoodFiles, ref int invalidCount)
    {
        // Check if the file exists in the listfile by path (case-insensitive)
        bool existsInListfile = knownGoodFiles.Contains(reference.NormalizedPath);
        reference.ExistsInListfile = existsInListfile;
        
        // If the reference has a FileDataID, check if it maps to a known path
        if (reference.UsesFileDataId && reference.FileDataId > 0)
        {
            if (_fileDataIdToPathMap.TryGetValue(reference.FileDataId, out string? mappedPath))
            {
                reference.MatchedByFileDataId = true;
                reference.RepairedPath = mappedPath;
                
                // If the original path doesn't exist but we found it by FileDataID, consider it valid
                if (!existsInListfile)
                {
                    _logger.LogDebug($"Found by FileDataID {reference.FileDataId}: {mappedPath} (original: {reference.OriginalPath})");
                }
                
                reference.IsValid = true;
                return;
            }
        }
        // If the reference has a path but no FileDataID, check if we can find a FileDataID for it
        else if (!reference.UsesFileDataId && _pathToFileDataIdMap.TryGetValue(reference.NormalizedPath, out uint fileDataId))
        {
            reference.FileDataId = fileDataId;
            _logger.LogDebug($"Assigned FileDataID {fileDataId} to {reference.OriginalPath}");
        }
        
        // Determine validity based on listfile presence
        reference.IsValid = existsInListfile || reference.MatchedByFileDataId;
        
        if (!reference.IsValid)
        {
            invalidCount++;
        }
    }
}