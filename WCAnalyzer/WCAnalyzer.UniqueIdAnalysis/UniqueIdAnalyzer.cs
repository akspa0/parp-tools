using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;
using System.Text.Json.Serialization;
using System.Numerics;

namespace WCAnalyzer.UniqueIdAnalysis
{
    /// <summary>
    /// Custom JsonConverter for FileReferenceType to handle case-insensitive values
    /// </summary>
    public class FileReferenceTypeConverter : JsonConverter<FileReferenceType>
    {
        public override FileReferenceType Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            string? value = reader.GetString();
            string normalizedValue = value?.ToLowerInvariant() ?? string.Empty;
            return normalizedValue switch
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
    /// Represents position and rotation data in a Vector3-compatible format
    /// </summary>
    public class Vector3Data
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Z { get; set; }

        public Vector3Data()
        {
            X = 0;
            Y = 0;
            Z = 0;
        }

        public Vector3Data(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public static Vector3Data FromVector3(Vector3 vector)
        {
            return new Vector3Data(vector.X, vector.Y, vector.Z);
        }

        public Vector3 ToVector3()
        {
            return new Vector3(X, Y, Z);
        }

        public static Vector3Data FromObject(object obj)
        {
            var result = new Vector3Data();

            if (obj == null)
                return result;

            // If it's already a Vector3Data, just return it
            if (obj is Vector3Data v3data)
                return v3data;

            // If it's a Vector3, convert it
            if (obj is Vector3 vector3)
                return FromVector3(vector3);

            // If it's a JsonElement, try to extract X, Y, Z properties
            if (obj is JsonElement jsonElement)
            {
                if (jsonElement.TryGetProperty("X", out var xElement) && xElement.ValueKind == JsonValueKind.Number)
                    result.X = xElement.GetSingle();
                
                if (jsonElement.TryGetProperty("Y", out var yElement) && yElement.ValueKind == JsonValueKind.Number)
                    result.Y = yElement.GetSingle();
                
                if (jsonElement.TryGetProperty("Z", out var zElement) && zElement.ValueKind == JsonValueKind.Number)
                    result.Z = zElement.GetSingle();
                
                return result;
            }

            // Try to use reflection to access X, Y, Z properties
            try
            {
                var type = obj.GetType();
                var xProp = type.GetProperty("X");
                var yProp = type.GetProperty("Y");
                var zProp = type.GetProperty("Z");

                if (xProp != null)
                    result.X = Convert.ToSingle(xProp.GetValue(obj));
                
                if (yProp != null)
                    result.Y = Convert.ToSingle(yProp.GetValue(obj));
                
                if (zProp != null)
                    result.Z = Convert.ToSingle(zProp.GetValue(obj));
            }
            catch (Exception)
            {
                // Ignore reflection errors
            }

            return result;
        }
    }

    /// <summary>
    /// Custom JsonConverter for Vector3 to handle different JSON formats
    /// </summary>
    public class Vector3Converter : JsonConverter<Vector3>
    {
        public override Vector3 Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
                throw new JsonException("Expected start of object");

            float x = 0, y = 0, z = 0;
            
            while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
            {
                if (reader.TokenType != JsonTokenType.PropertyName)
                    throw new JsonException("Expected property name");

                string? propertyName = reader.GetString()?.ToLowerInvariant();
                reader.Read();

                switch (propertyName)
                {
                    case "x":
                        x = reader.GetSingle();
                        break;
                    case "y":
                        y = reader.GetSingle();
                        break;
                    case "z":
                        z = reader.GetSingle();
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }

            return new Vector3(x, y, z);
        }

        public override void Write(Utf8JsonWriter writer, Vector3 value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteNumber("X", value.X);
            writer.WriteNumber("Y", value.Y);
            writer.WriteNumber("Z", value.Z);
            writer.WriteEndObject();
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
        private readonly ILogger<UniqueIdAnalyzer>? _logger;
        
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
            ILogger<UniqueIdAnalyzer>? logger = null,
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
                            new FileReferenceTypeConverter(),
                            new Vector3Converter()
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
                                                
                                                // Extract position data from the JSON
                                                Vector3Data positionData = new Vector3Data();
                                                Vector3Data rotationData = new Vector3Data();
                                                float scale = 1.0f;
                                                
                                                // Extract position
                                                if (element.TryGetProperty("Position", out var positionElement))
                                                {
                                                    positionData = Vector3Data.FromObject(positionElement);
                                                }
                                                
                                                // Extract rotation
                                                if (element.TryGetProperty("Rotation", out var rotationElement))
                                                {
                                                    rotationData = Vector3Data.FromObject(rotationElement);
                                                }
                                                
                                                // Extract scale
                                                if (element.TryGetProperty("Scale", out var scaleElement) && 
                                                    scaleElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    scale = scaleElement.GetSingle();
                                                }
                                                
                                                // Set the properties
                                                placement.Position = positionData.ToVector3();
                                                placement.Rotation = rotationData.ToVector3();
                                                placement.Scale = scale;
                                                
                                                // Check for FileDataId and flags
                                                if (element.TryGetProperty("FileDataId", out var fileDataIdElement) && 
                                                    fileDataIdElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.FileDataId = fileDataIdElement.GetUInt32();
                                                    placement.UsesFileDataId = true;
                                                }
                                                
                                                if (element.TryGetProperty("Flags", out var flagsElement) && 
                                                    flagsElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.Flags = (ushort)flagsElement.GetInt32();
                                                }
                                                
                                                // Add to the list
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
                                                
                                                // Extract position data from the JSON
                                                Vector3Data positionData = new Vector3Data();
                                                Vector3Data rotationData = new Vector3Data();
                                                
                                                // Extract position
                                                if (element.TryGetProperty("Position", out var positionElement))
                                                {
                                                    positionData = Vector3Data.FromObject(positionElement);
                                                }
                                                
                                                // Extract rotation
                                                if (element.TryGetProperty("Rotation", out var rotationElement))
                                                {
                                                    rotationData = Vector3Data.FromObject(rotationElement);
                                                }
                                                
                                                // Extract scale, DoodadSet, NameSet
                                                if (element.TryGetProperty("Scale", out var scaleElement) && 
                                                    scaleElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    // Convert from float to ushort by scaling
                                                    placement.Scale = (ushort)(scaleElement.GetSingle() * 1024);
                                                }

                                                if (element.TryGetProperty("DoodadSet", out var doodadSetElement) && 
                                                    doodadSetElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.DoodadSet = (ushort)doodadSetElement.GetInt32();
                                                }
                                                
                                                if (element.TryGetProperty("NameSet", out var nameSetElement) && 
                                                    nameSetElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.NameSet = (ushort)nameSetElement.GetInt32();
                                                }
                                                
                                                // Set the properties
                                                placement.Position = positionData.ToVector3();
                                                placement.Rotation = rotationData.ToVector3();
                                                
                                                // Check for FileDataId and flags
                                                if (element.TryGetProperty("FileDataId", out var fileDataIdElement) && 
                                                    fileDataIdElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.FileDataId = fileDataIdElement.GetUInt32();
                                                    placement.UsesFileDataId = true;
                                                }
                                                
                                                if (element.TryGetProperty("Flags", out var flagsElement) && 
                                                    flagsElement.ValueKind == JsonValueKind.Number)
                                                {
                                                    placement.Flags = (ushort)flagsElement.GetInt32();
                                                }
                                                
                                                // Add to the list
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
                                            WmoPlacements = wmoPlacements,
                                            UniqueIds = new HashSet<int>(
                                                modelPlacements.Select(p => p.UniqueId)
                                                .Concat(wmoPlacements.Select(p => p.UniqueId))
                                                .Where(id => id != 0)
                                            )
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
            
            // Extract the actual map name from the ADT filename
            string actualMapName = mapName;
            string fileName = result.FileName;
            
            // ADT filenames typically follow the pattern "MapName_X_Y.adt"
            if (fileName != null && fileName.Contains("_"))
            {
                // Extract the map name part (everything before the first underscore)
                int firstUnderscoreIndex = fileName.IndexOf('_');
                if (firstUnderscoreIndex > 0)
                {
                    actualMapName = fileName.Substring(0, firstUnderscoreIndex);
                }
            }
            
            // Create AdtInfo with full file path
            var adtInfo = new AdtInfo(result.FilePath ?? result.FileName, actualMapName, uniqueIds);
            
            // Process model placements and link them to uniqueIDs
            if (result.ModelPlacements != null)
            {
                foreach (var placement in result.ModelPlacements)
                {
                    if (placement.UniqueId == 0)
                        continue;
                    
                    // Extract position data safely
                    Vector3Data posData = Vector3Data.FromObject(placement.Position);
                    Vector3Data rotData = Vector3Data.FromObject(placement.Rotation);
                    float scale = placement.Scale;
                    
                    var assetRef = new AssetReference(
                        placement.Name, 
                        "Model", 
                        placement.UniqueId, 
                        result.FileName,
                        actualMapName,
                        posData.X, 
                        posData.Y, 
                        posData.Z,
                        rotData.X,
                        rotData.Y,
                        rotData.Z,
                        scale);
                    
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
                    
                    // Extract position data safely
                    Vector3Data posData = Vector3Data.FromObject(placement.Position);
                    Vector3Data rotData = Vector3Data.FromObject(placement.Rotation);
                    
                    // Convert from ushort to float using the same scaling as for models
                    float scale = placement.Scale / 1024.0f;
                    
                    var assetRef = new AssetReference(
                        placement.Name, 
                        "WMO", 
                        placement.UniqueId, 
                        result.FileName,
                        actualMapName,
                        posData.X, 
                        posData.Y, 
                        posData.Z,
                        rotData.X,
                        rotData.Y,
                        rotData.Z,
                        scale);
                    
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