using System;
using System.Collections.Generic;
using System.Numerics;
using WCAnalyzer.Core.Models;

namespace WCAnalyzer.UniqueIdAnalysis
{
    /// <summary>
    /// Represents a cluster of unique IDs that might correspond to a specific time period.
    /// </summary>
    public class UniqueIdCluster
    {
        public int MinId { get; set; }
        public int MaxId { get; set; }
        public int Count { get; set; }
        public HashSet<string> AdtFiles { get; set; } = new HashSet<string>();
        public Dictionary<string, int> IdCountsByAdt { get; set; } = new Dictionary<string, int>();
        
        // Track asset references in the cluster
        public HashSet<AssetReference> Assets { get; set; } = new HashSet<AssetReference>();
        
        public double Density => (double)Count / (MaxId - MinId + 1);
        
        public override string ToString()
        {
            return $"Cluster {MinId}-{MaxId} ({Count} IDs, {AdtFiles.Count} ADTs, {Assets.Count} assets, Density: {Density:F2})";
        }
    }

    /// <summary>
    /// Simple representation of an ADT file with its unique IDs and asset placements.
    /// </summary>
    public class AdtInfo
    {
        public string FileName { get; set; }
        public string MapName { get; set; }
        public List<int> UniqueIds { get; set; } = new List<int>();
        
        // Track uniqueID to asset mapping
        public Dictionary<int, List<AssetReference>> AssetsByUniqueId { get; set; } = new Dictionary<int, List<AssetReference>>();
        
        public AdtInfo(string fileName, string mapName, List<int> uniqueIds)
        {
            FileName = fileName;
            MapName = mapName;
            UniqueIds = uniqueIds;
            AssetsByUniqueId = new Dictionary<int, List<AssetReference>>();
        }
    }

    /// <summary>
    /// Represents an asset reference (model or WMO) associated with a unique ID.
    /// </summary>
    public class AssetReference : IEquatable<AssetReference>
    {
        public string AssetPath { get; set; }
        public string Type { get; set; } // "Model" or "WMO"
        public int UniqueId { get; set; }
        public string AdtFile { get; set; }
        public string MapName { get; set; }
        public double PositionX { get; set; }
        public double PositionY { get; set; }
        public double PositionZ { get; set; }
        public float Scale { get; set; }
        
        public AssetReference(string assetPath, string type, int uniqueId, string adtFile, string mapName)
        {
            AssetPath = assetPath;
            Type = type;
            UniqueId = uniqueId;
            AdtFile = adtFile;
            MapName = mapName;
        }

        public AssetReference(string assetPath, string type, int uniqueId, string adtFile, string mapName, 
            double posX, double posY, double posZ, float scale = 1.0f)
            : this(assetPath, type, uniqueId, adtFile, mapName)
        {
            PositionX = posX;
            PositionY = posY;
            PositionZ = posZ;
            Scale = scale;
        }
        
        // Override Equals and GetHashCode to ensure proper HashSet behavior based on asset path
        public bool Equals(AssetReference other)
        {
            if (other == null)
                return false;
                
            return AssetPath == other.AssetPath && Type == other.Type;
        }
        
        public override bool Equals(object obj)
        {
            return Equals(obj as AssetReference);
        }
        
        public override int GetHashCode()
        {
            return (AssetPath + Type).GetHashCode();
        }
        
        public override string ToString()
        {
            return $"{Type}: {AssetPath} (ID: {UniqueId})";
        }
    }

    /// <summary>
    /// Represents the results of a UniqueID analysis.
    /// </summary>
    public class UniqueIdAnalysisResult
    {
        public List<UniqueIdCluster> Clusters { get; set; } = new List<UniqueIdCluster>();
        public Dictionary<string, AdtInfo> AdtInfos { get; set; } = new Dictionary<string, AdtInfo>();
        public int TotalUniqueIds { get; set; }
        public int MinUniqueId { get; set; }
        public int MaxUniqueId { get; set; }
        public int TotalAdtFiles { get; set; }
        public int TotalAssets { get; set; }
        public DateTime AnalysisTime { get; set; } = DateTime.Now;
    }
} 