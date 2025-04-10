using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for detecting and handling split ADT files.
    /// </summary>
    public class AdtSplitDetector
    {
        private readonly ILogger<AdtSplitDetector> _logger;
        private static readonly Regex SplitAdtPattern = new(@"^(.+?)_(\d+)_(\d+)_(tex|obj|lod)0\.adt$", RegexOptions.Compiled);

        public AdtSplitDetector(ILogger<AdtSplitDetector> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Checks if a given ADT file is part of a split ADT set.
        /// </summary>
        /// <param name="fileName">The name of the ADT file to check.</param>
        /// <returns>True if the file is part of a split ADT set; otherwise, false.</returns>
        public bool IsSplitAdt(string fileName)
        {
            return SplitAdtPattern.IsMatch(fileName);
        }

        /// <summary>
        /// Gets the base ADT file name from a split ADT file name.
        /// </summary>
        /// <param name="fileName">The name of the split ADT file.</param>
        /// <returns>The base ADT file name.</returns>
        public string GetBaseAdtName(string fileName)
        {
            var match = SplitAdtPattern.Match(fileName);
            if (!match.Success)
            {
                throw new ArgumentException("The file name is not a valid split ADT file name.", nameof(fileName));
            }

            return $"{match.Groups[1].Value}_{match.Groups[2].Value}_{match.Groups[3].Value}.adt";
        }

        /// <summary>
        /// Groups ADT files by their base name.
        /// </summary>
        /// <param name="files">The list of ADT files to group.</param>
        /// <returns>A dictionary mapping base names to their associated files.</returns>
        public Dictionary<string, List<string>> GroupSplitAdts(IEnumerable<string> files)
        {
            var groups = new Dictionary<string, List<string>>();

            foreach (var file in files)
            {
                if (IsSplitAdt(file))
                {
                    var baseName = GetBaseAdtName(file);
                    if (!groups.ContainsKey(baseName))
                    {
                        groups[baseName] = new List<string>();
                    }
                    groups[baseName].Add(file);
                }
                else
                {
                    // Handle monolithic ADT files
                    groups[file] = new List<string> { file };
                }
            }

            return groups;
        }

        /// <summary>
        /// Determines if a set of ADT files is split or monolithic.
        /// </summary>
        /// <param name="files">The list of ADT files to check.</param>
        /// <returns>True if the files are split ADTs; otherwise, false.</returns>
        public bool IsSplitAdtSet(IEnumerable<string> files)
        {
            return files.Any(IsSplitAdt);
        }

        /// <summary>
        /// Consolidates results from multiple ADT analysis results into a single result.
        /// </summary>
        /// <param name="results">The list of ADT analysis results to consolidate.</param>
        /// <returns>A consolidated ADT analysis result.</returns>
        public AdtAnalysisResult ConsolidateResults(List<AdtAnalysisResult> results)
        {
            if (results == null || !results.Any())
            {
                throw new ArgumentException("The results list cannot be null or empty.", nameof(results));
            }

            var baseResult = results[0];
            var consolidatedResult = new AdtAnalysisResult
            {
                FileName = baseResult.FileName,
                XCoord = baseResult.XCoord,
                YCoord = baseResult.YCoord,
                AdtVersion = baseResult.AdtVersion,
                TerrainChunks = baseResult.TerrainChunks,
                Errors = baseResult.Errors.ToList()
            };

            // Consolidate references, removing duplicates
            var texturePaths = new HashSet<string>();
            var modelPaths = new HashSet<string>();
            var wmoPaths = new HashSet<string>();

            foreach (var result in results)
            {
                // Add texture references
                foreach (var texture in result.TextureReferences)
                {
                    if (!texturePaths.Contains(texture.OriginalPath))
                    {
                        texturePaths.Add(texture.OriginalPath);
                        consolidatedResult.TextureReferences.Add(texture);
                    }
                }

                // Add model references
                foreach (var model in result.ModelReferences)
                {
                    if (!modelPaths.Contains(model.OriginalPath))
                    {
                        modelPaths.Add(model.OriginalPath);
                        consolidatedResult.ModelReferences.Add(model);
                    }
                }

                // Add WMO references
                foreach (var wmo in result.WmoReferences)
                {
                    if (!wmoPaths.Contains(wmo.OriginalPath))
                    {
                        wmoPaths.Add(wmo.OriginalPath);
                        consolidatedResult.WmoReferences.Add(wmo);
                    }
                }

                // Add placements
                consolidatedResult.ModelPlacements.AddRange(result.ModelPlacements);
                consolidatedResult.WmoPlacements.AddRange(result.WmoPlacements);

                // Add errors
                if (result != baseResult)
                {
                    consolidatedResult.Errors.AddRange(result.Errors);
                }
            }

            // Log consolidation information
            _logger.LogInformation(
                "Consolidated {Count} ADT files into {BaseFile}. Found {TextureCount} unique textures, {ModelCount} unique models, and {WmoCount} unique WMOs.",
                results.Count,
                consolidatedResult.FileName,
                consolidatedResult.TextureReferences.Count,
                consolidatedResult.ModelReferences.Count,
                consolidatedResult.WmoReferences.Count);

            return consolidatedResult;
        }
    }
} 