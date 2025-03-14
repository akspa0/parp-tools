using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using System.Text.Json.Serialization;

namespace WCAnalyzer.UniqueIdAnalysis
{
    /// <summary>
    /// Custom JsonConverter for FileReferenceType to handle case-insensitive values
    /// </summary>
    public class FileReferenceTypeConverter : JsonConverter<FileReferenceType>
    {
        public override FileReferenceType Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            string value = reader.GetString()?.ToLowerInvariant();
            return value switch
            {
                "texture" => FileReferenceType.Texture,
                "model" => FileReferenceType.Model,
                "worldmodel" => FileReferenceType.WorldModel,
                "wmo" => FileReferenceType.WorldModel,
                _ => throw new JsonException($"Invalid FileReferenceType value: {value}")
            };
        }

        public override void Write(Utf8JsonWriter writer, FileReferenceType value, JsonSerializerOptions options)
        {
            writer.WriteStringValue(value.ToString().ToLowerInvariant());
        }
    }

    /// <summary>
    /// Main class for analyzing unique IDs from ADT analysis results.
    /// </summary>
    public class UniqueIdAnalyzer
    {
        private readonly string _resultsDirectory;
        private readonly string _outputDirectory;
        private readonly int _clusterThreshold;
        private readonly int _clusterGapThreshold;
        private readonly ILogger<UniqueIdAnalyzer> _logger;
        
        private List<AdtInfo> _adtFiles = new List<AdtInfo>();
        private Dictionary<string, List<UniqueIdCluster>> _mapClusters = new Dictionary<string, List<UniqueIdCluster>>();
        private List<UniqueIdCluster> _globalClusters = new List<UniqueIdCluster>();
        private List<int> _nonClusteredIds = new List<int>();
        private Dictionary<int, List<AssetReference>> _nonClusteredAssets = new Dictionary<int, List<AssetReference>>();
        private bool _generateComprehensiveReport = true;
        
        /// <summary>
        /// Creates a new instance of the UniqueIdAnalyzer class.
        /// </summary>
        /// <param name="resultsDirectory">Directory containing JSON results from ADT analysis</param>
        /// <param name="outputDirectory">Directory to write analysis results to</param>
        /// <param name="logger">Logger instance</param>
        /// <param name="clusterThreshold">Minimum number of IDs to form a cluster</param>
        /// <param name="clusterGapThreshold">Maximum gap between IDs to be considered part of the same cluster</param>
        /// <param name="generateComprehensiveReport">Whether to generate a comprehensive report with all assets</param>
        public UniqueIdAnalyzer(
            string resultsDirectory,
            string outputDirectory,
            ILogger<UniqueIdAnalyzer> logger = null,
            int clusterThreshold = 10,
            int clusterGapThreshold = 1000,
            bool generateComprehensiveReport = true)
        {
            _resultsDirectory = resultsDirectory;
            _outputDirectory = outputDirectory;
            _logger = logger;
            _clusterThreshold = clusterThreshold;
            _clusterGapThreshold = clusterGapThreshold;
            _generateComprehensiveReport = generateComprehensiveReport;
            
            // Create output directory if it doesn't exist
            if (!Directory.Exists(_outputDirectory))
            {
                Directory.CreateDirectory(_outputDirectory);
            }
        }

        /// <summary>
        /// Gets the ADT files processed by the analyzer.
        /// </summary>
        /// <returns>The list of processed ADT files.</returns>
        public List<AdtInfo> GetAdtFiles()
        {
            return _adtFiles;
        }

        /// <summary>
        /// Runs the unique ID analysis.
        /// </summary>
        /// <param name="generateComprehensiveReport">Whether to generate a comprehensive report with all assets</param>
        /// <returns>The analysis result</returns>
        public async Task<UniqueIdAnalysisResult> AnalyzeAsync(bool generateComprehensiveReport = true)
        {
            _logger?.LogInformation("Starting UniqueID analysis...");
            
            // Load ADT data from JSON files
            await LoadAdtDataAsync();
            
            // Check if we have any data to analyze
            if (_adtFiles.Count == 0)
            {
                _logger?.LogWarning("No ADT files with UniqueIDs were found. Analysis cannot proceed.");
                return new UniqueIdAnalysisResult
                {
                    Clusters = new List<UniqueIdCluster>(),
                    AdtInfos = new Dictionary<string, AdtInfo>(),
                    TotalUniqueIds = 0,
                    MinUniqueId = 0,
                    MaxUniqueId = 0,
                    TotalAdtFiles = 0,
                    TotalAssets = 0,
                    AnalysisTime = DateTime.Now
                };
            }
            
            // Identify clusters per map
            IdentifyMapClusters();
            
            // Identify global clusters across all maps
            IdentifyGlobalClusters();
            
            // Identify non-clustered IDs if generating comprehensive report
            if (generateComprehensiveReport)
            {
                IdentifyNonClusteredIds();
            }
            
            // Get all unique IDs
            var allUniqueIds = _adtFiles.SelectMany(a => a.UniqueIds).Distinct().ToList();
            
            // Create analysis result
            var result = new UniqueIdAnalysisResult
            {
                Clusters = _globalClusters,
                AdtInfos = _adtFiles
                    .GroupBy(a => $"{a.MapName}_{a.FileName}".ToLowerInvariant(), StringComparer.OrdinalIgnoreCase)
                    .ToDictionary(
                        g => g.Key,
                        g => g.First(), // Take the first occurrence of each ADT file
                        StringComparer.OrdinalIgnoreCase),
                TotalUniqueIds = allUniqueIds.Count,
                MinUniqueId = allUniqueIds.Any() ? allUniqueIds.Min() : 0,
                MaxUniqueId = allUniqueIds.Any() ? allUniqueIds.Max() : 0,
                TotalAdtFiles = _adtFiles.Count,
                TotalAssets = _adtFiles.Sum(a => a.AssetsByUniqueId.Values.Sum(v => v.Count))
            };
            
            _logger?.LogInformation("UniqueID analysis complete!");
            
            return result;
        }
        
        /// <summary>
        /// Loads ADT data from JSON results files.
        /// </summary>
        private async Task LoadAdtDataAsync()
        {
            _logger?.LogInformation("Loading ADT data from JSON files...");
            
            // Search for JSON files in the results directory and its subdirectories
            var jsonFiles = Directory.GetFiles(_resultsDirectory, "*.json", SearchOption.AllDirectories);
            
            _logger?.LogInformation("Found {JsonFileCount} JSON files to process", jsonFiles.Length);
            
            foreach (var jsonFile in jsonFiles)
            {
                try
                {
                    // Extract map name from directory path
                    var dirName = Path.GetDirectoryName(jsonFile);
                    var mapName = dirName != null ? Path.GetFileName(dirName) : "Unknown";
                    
                    // Load JSON
                    var json = await File.ReadAllTextAsync(jsonFile);
                    
                    // Create JSON options with custom converters
                    var jsonOptions = new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true,
                        Converters =
                        {
                            new JsonStringEnumConverter(JsonNamingPolicy.CamelCase),
                            new FileReferenceTypeConverter()
                        }
                    };
                    
                    // Check if this is a single ADT result or a collection
                    if (json.StartsWith("["))
                    {
                        // This is a collection of ADT results
                        _logger?.LogDebug("Processing JSON array from {JsonFile}", jsonFile);
                        
                        try
                        {
                            var adtResults = JsonSerializer.Deserialize<List<AdtAnalysisResult>>(json, jsonOptions);
                            
                            if (adtResults != null && adtResults.Count > 0)
                            {
                                foreach (var adtResult in adtResults)
                                {
                                    ProcessAdtResult(adtResult, mapName);
                                }
                            }
                        }
                        catch (JsonException ex)
                        {
                            _logger?.LogWarning(ex, "Could not deserialize JSON array from {JsonFile}. The file may have a different format.", jsonFile);
                        }
                    }
                    else
                    {
                        // Try to deserialize as a single ADT result
                        _logger?.LogDebug("Processing single JSON object from {JsonFile}", jsonFile);
                        
                        try
                        {
                            var adtResult = JsonSerializer.Deserialize<AdtAnalysisResult>(json, jsonOptions);
                            
                            if (adtResult != null)
                            {
                                ProcessAdtResult(adtResult, mapName);
                            }
                        }
                        catch (JsonException ex)
                        {
                            _logger?.LogWarning(ex, "Could not deserialize JSON object from {JsonFile} as AdtAnalysisResult. The file may have a different format.", jsonFile);
                            
                            // Try to extract model and WMO placements directly from the JSON
                            try
                            {
                                using (JsonDocument doc = JsonDocument.Parse(json))
                                {
                                    var root = doc.RootElement;
                                    
                                    // Try to extract filename from the JSON
                                    string fileName = Path.GetFileNameWithoutExtension(jsonFile);
                                    if (root.TryGetProperty("FileName", out var fileNameElement))
                                    {
                                        fileName = fileNameElement.GetString() ?? fileName;
                                    }
                                    
                                    // Try to extract model placements
                                    List<ModelPlacement> modelPlacements = new List<ModelPlacement>();
                                    if (root.TryGetProperty("ModelPlacements", out var modelPlacementsElement) && 
                                        modelPlacementsElement.ValueKind == JsonValueKind.Array)
                                    {
                                        foreach (var element in modelPlacementsElement.EnumerateArray())
                                        {
                                            if (element.TryGetProperty("UniqueId", out var uniqueIdElement) && 
                                                uniqueIdElement.ValueKind == JsonValueKind.Number)
                                            {
                                                var placement = new ModelPlacement
                                                {
                                                    UniqueId = uniqueIdElement.GetInt32(),
                                                    Name = element.TryGetProperty("Name", out var nameElement) ? 
                                                        nameElement.GetString() ?? string.Empty : string.Empty
                                                };
                                                
                                                modelPlacements.Add(placement);
                                            }
                                        }
                                    }
                                    
                                    // Try to extract WMO placements
                                    List<WmoPlacement> wmoPlacements = new List<WmoPlacement>();
                                    if (root.TryGetProperty("WmoPlacements", out var wmoPlacementsElement) && 
                                        wmoPlacementsElement.ValueKind == JsonValueKind.Array)
                                    {
                                        foreach (var element in wmoPlacementsElement.EnumerateArray())
                                        {
                                            if (element.TryGetProperty("UniqueId", out var uniqueIdElement) && 
                                                uniqueIdElement.ValueKind == JsonValueKind.Number)
                                            {
                                                var placement = new WmoPlacement
                                                {
                                                    UniqueId = uniqueIdElement.GetInt32(),
                                                    Name = element.TryGetProperty("Name", out var nameElement) ? 
                                                        nameElement.GetString() ?? string.Empty : string.Empty
                                                };
                                                
                                                wmoPlacements.Add(placement);
                                            }
                                        }
                                    }
                                    
                                    // Create a synthetic ADT result if we found any placements
                                    if (modelPlacements.Count > 0 || wmoPlacements.Count > 0)
                                    {
                                        var syntheticResult = new AdtAnalysisResult
                                        {
                                            FileName = fileName,
                                            FilePath = jsonFile,
                                            ModelPlacements = modelPlacements,
                                            WmoPlacements = wmoPlacements
                                        };
                                        
                                        ProcessAdtResult(syntheticResult, mapName);
                                    }
                                }
                            }
                            catch (Exception jsonDocEx)
                            {
                                _logger?.LogError(jsonDocEx, "Failed to extract placements from {JsonFile}", jsonFile);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger?.LogError(ex, "Error processing {JsonFile}", jsonFile);
                }
            }
            
            _logger?.LogInformation("Loaded {AdtCount} ADT files with uniqueIDs.", _adtFiles.Count);
        }
        
        /// <summary>
        /// Processes an ADT analysis result and extracts UniqueID information.
        /// </summary>
        /// <param name="result">The ADT analysis result to process</param>
        /// <param name="mapName">The name of the map this ADT belongs to</param>
        private void ProcessAdtResult(AdtAnalysisResult result, string mapName)
        {
            // Extract unique IDs from model and WMO placements
            var uniqueIds = new List<int>();
            
            if (result.ModelPlacements != null)
            {
                uniqueIds.AddRange(result.ModelPlacements.Select(p => p.UniqueId).Where(id => id != 0));
            }
            
            if (result.WmoPlacements != null)
            {
                uniqueIds.AddRange(result.WmoPlacements.Select(p => p.UniqueId).Where(id => id != 0));
            }
            
            if (uniqueIds.Count == 0)
                return;
            
            // Create AdtInfo with full file path
            var adtInfo = new AdtInfo(result.FilePath ?? result.FileName, mapName, uniqueIds);
            
            // Process model placements and link them to uniqueIDs
            if (result.ModelPlacements != null)
            {
                foreach (var placement in result.ModelPlacements)
                {
                    if (placement.UniqueId == 0)
                        continue;
                        
                    string modelPath = placement.Name;
                    
                    // Create asset reference
                    var assetRef = new AssetReference(
                        modelPath, 
                        "Model", 
                        placement.UniqueId, 
                        result.FileName,
                        mapName,
                        placement.Position.X, 
                        placement.Position.Y, 
                        placement.Position.Z,
                        placement.Scale);
                    
                    // Add to the AdtInfo
                    if (!adtInfo.AssetsByUniqueId.TryGetValue(placement.UniqueId, out var assetList))
                    {
                        assetList = new List<AssetReference>();
                        adtInfo.AssetsByUniqueId[placement.UniqueId] = assetList;
                    }
                    
                    assetList.Add(assetRef);
                }
            }
            
            // Process WMO placements and link them to uniqueIDs
            if (result.WmoPlacements != null)
            {
                foreach (var placement in result.WmoPlacements)
                {
                    if (placement.UniqueId == 0)
                        continue;
                        
                    string wmoPath = placement.Name;
                    
                    // Create asset reference
                    var assetRef = new AssetReference(
                        wmoPath, 
                        "WMO", 
                        placement.UniqueId, 
                        result.FileName,
                        mapName,
                        placement.Position.X, 
                        placement.Position.Y, 
                        placement.Position.Z);
                    
                    // Add to the AdtInfo
                    if (!adtInfo.AssetsByUniqueId.TryGetValue(placement.UniqueId, out var assetList))
                    {
                        assetList = new List<AssetReference>();
                        adtInfo.AssetsByUniqueId[placement.UniqueId] = assetList;
                    }
                    
                    assetList.Add(assetRef);
                }
            }
            
            _adtFiles.Add(adtInfo);
        }
        
        /// <summary>
        /// Identifies clusters of unique IDs for each map.
        /// </summary>
        private void IdentifyMapClusters()
        {
            _logger?.LogInformation("Identifying clusters for each map...");
            
            // Group ADTs by map
            var adtsByMap = _adtFiles.GroupBy(a => a.MapName);
            
            foreach (var mapGroup in adtsByMap)
            {
                var mapName = mapGroup.Key;
                var allIds = new List<int>();
                
                // Collect all unique IDs for this map
                foreach (var adt in mapGroup)
                {
                    allIds.AddRange(adt.UniqueIds);
                }
                
                // Sort IDs
                allIds = allIds.Distinct().OrderBy(id => id).ToList();
                
                if (allIds.Count == 0)
                {
                    _logger?.LogWarning("Map {MapName} has no unique IDs to analyze", mapName);
                    continue;
                }
                
                // Find clusters
                var clusters = ClusterAnalyzer.FindClusters(allIds, _clusterThreshold, _clusterGapThreshold);
                
                // Associate ADTs and assets with each cluster
                foreach (var cluster in clusters)
                {
                    foreach (var adt in mapGroup)
                    {
                        // Check if this ADT has IDs in the cluster
                        var idsInCluster = adt.UniqueIds.Where(id => id >= cluster.MinId && id <= cluster.MaxId).ToList();
                        
                        if (idsInCluster.Count > 0)
                        {
                            cluster.AdtFiles.Add(adt.FileName);
                            cluster.IdCountsByAdt[adt.FileName] = idsInCluster.Count;
                            
                            // Add assets to the cluster
                            foreach (var id in idsInCluster)
                            {
                                if (adt.AssetsByUniqueId.TryGetValue(id, out var assets))
                                {
                                    foreach (var asset in assets)
                                    {
                                        cluster.Assets.Add(asset);
                                    }
                                }
                            }
                        }
                    }
                }
                
                _mapClusters[mapName] = clusters;
                
                _logger?.LogInformation("Map {MapName}: Found {ClusterCount} clusters with {IdCount} unique IDs", 
                    mapName, clusters.Count, allIds.Count);
            }
        }
        
        /// <summary>
        /// Identifies global clusters of unique IDs across all maps.
        /// </summary>
        private void IdentifyGlobalClusters()
        {
            _logger?.LogInformation("Identifying global clusters across all maps...");
            
            // Collect all unique IDs
            var allIds = _adtFiles.SelectMany(a => a.UniqueIds).Distinct().OrderBy(id => id).ToList();
            
            if (allIds.Count == 0)
            {
                _logger?.LogWarning("No unique IDs found across all maps");
                _globalClusters = new List<UniqueIdCluster>();
                return;
            }
            
            // Find clusters
            _globalClusters = ClusterAnalyzer.FindClusters(allIds, _clusterThreshold, _clusterGapThreshold);
            
            // Associate ADTs and assets with each cluster
            foreach (var cluster in _globalClusters)
            {
                foreach (var adt in _adtFiles)
                {
                    // Check if this ADT has IDs in the cluster
                    var idsInCluster = adt.UniqueIds.Where(id => id >= cluster.MinId && id <= cluster.MaxId).ToList();
                    
                    if (idsInCluster.Count > 0)
                    {
                        cluster.AdtFiles.Add(adt.FileName);
                        cluster.IdCountsByAdt[adt.FileName] = idsInCluster.Count;
                        
                        // Add assets to the cluster
                        foreach (var id in idsInCluster)
                        {
                            if (adt.AssetsByUniqueId.TryGetValue(id, out var assets))
                            {
                                foreach (var asset in assets)
                                {
                                    cluster.Assets.Add(asset);
                                }
                            }
                        }
                    }
                }
            }
            
            _logger?.LogInformation("Found {ClusterCount} global clusters with {IdCount} unique IDs", 
                _globalClusters.Count, allIds.Count);
        }
        
        /// <summary>
        /// Identifies IDs that are not part of any cluster.
        /// </summary>
        private void IdentifyNonClusteredIds()
        {
            _logger?.LogInformation("Identifying non-clustered IDs...");
            
            // Get all unique IDs
            var allIds = _adtFiles.SelectMany(a => a.UniqueIds).Distinct().OrderBy(id => id).ToList();
            
            if (allIds.Count == 0)
            {
                _logger?.LogWarning("No unique IDs found to identify non-clustered IDs");
                _nonClusteredIds = new List<int>();
                return;
            }
            
            // Get IDs in clusters
            var clusteredIds = new HashSet<int>();
            foreach (var cluster in _globalClusters)
            {
                for (int id = cluster.MinId; id <= cluster.MaxId; id++)
                {
                    clusteredIds.Add(id);
                }
            }
            
            // Find non-clustered IDs
            _nonClusteredIds = allIds.Where(id => !clusteredIds.Contains(id)).ToList();
            
            // Associate assets with non-clustered IDs
            foreach (var adt in _adtFiles)
            {
                foreach (var id in adt.UniqueIds)
                {
                    if (!clusteredIds.Contains(id) && adt.AssetsByUniqueId.TryGetValue(id, out var assets))
                    {
                        if (!_nonClusteredAssets.TryGetValue(id, out var assetList))
                        {
                            assetList = new List<AssetReference>();
                            _nonClusteredAssets[id] = assetList;
                        }
                        
                        assetList.AddRange(assets);
                    }
                }
            }
            
            _logger?.LogInformation("Found {NonClusteredCount} non-clustered IDs", _nonClusteredIds.Count);
        }
    }
} 