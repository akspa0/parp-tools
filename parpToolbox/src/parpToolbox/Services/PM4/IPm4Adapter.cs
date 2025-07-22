using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Converts a PM4 file on disk to an in-memory <see cref="Pm4Scene"/> domain model.
    /// Supports comprehensive analysis and validation of PM4 data structures.
    /// </summary>
    public interface IPm4Adapter
    {
        /// <summary>
        /// Loads a PM4 file with default options (basic loading, no analysis).
        /// </summary>
        Pm4Scene Load(string path);
        
        /// <summary>
        /// Loads a PM4 file with specified analysis options.
        /// </summary>
        Pm4Scene Load(string path, Pm4LoadOptions options);
        
        /// <summary>
        /// Analyzes a loaded PM4 scene for data integrity, patterns, and relationships.
        /// </summary>
        Pm4AnalysisReport Analyze(Pm4Scene scene, Pm4AnalysisOptions options);
    }
    
    /// <summary>
    /// Options for loading PM4 files with various analysis and validation features.
    /// </summary>
    public class Pm4LoadOptions
    {
        /// <summary>Whether to perform comprehensive validation during loading.</summary>
        public bool ValidateData { get; set; } = false;
        
        /// <summary>Whether to analyze index patterns and out-of-bounds references.</summary>
        public bool AnalyzeIndexPatterns { get; set; } = false;
        
        /// <summary>Whether to capture raw chunk data for analysis.</summary>
        public bool CaptureRawData { get; set; } = false;
        
        /// <summary>Whether to log detailed loading information.</summary>
        public bool VerboseLogging { get; set; } = false;
    }
    
    /// <summary>
    /// Options for analyzing PM4 scene data.
    /// </summary>
    public class Pm4AnalysisOptions
    {
        /// <summary>Whether to analyze data structure relationships.</summary>
        public bool AnalyzeDataStructure { get; set; } = true;
        
        /// <summary>Whether to analyze index patterns and out-of-bounds references.</summary>
        public bool AnalyzeIndexPatterns { get; set; } = true;
        
        /// <summary>Whether to analyze unknown fields and correlations.</summary>
        public bool AnalyzeUnknownFields { get; set; } = true;
        
        /// <summary>Whether to analyze chunk relationships (MPRL↔MSLK, etc.).</summary>
        public bool AnalyzeChunkRelationships { get; set; } = true;
        
        /// <summary>Whether to analyze MPRR sentinel patterns.</summary>
        public bool AnalyzeMprrSentinels { get; set; } = true;
        
        /// <summary>Whether to analyze MSUR grouping patterns.</summary>
        public bool AnalyzeMsurGroups { get; set; } = true;
        
        /// <summary>Output directory for analysis reports.</summary>
        public string? OutputDirectory { get; set; }
        
        /// <summary>Whether to generate detailed CSV reports.</summary>
        public bool GenerateCsvReports { get; set; } = false;
        
        /// <summary>Enable verbose logging during analysis.</summary>
        public bool VerboseLogging { get; set; } = false;
    }
    
    /// <summary>
    /// Comprehensive analysis report for PM4 scene data.
    /// </summary>
    public class Pm4AnalysisReport
    {
        /// <summary>Summary of the analysis results.</summary>
        public string Summary { get; set; } = string.Empty;
        
        /// <summary>Data structure analysis results.</summary>
        public DataStructureAnalysis? DataStructure { get; set; }
        
        /// <summary>Index pattern analysis results.</summary>
        public IndexPatternAnalysis? IndexPatterns { get; set; }
        
        /// <summary>Unknown field analysis results.</summary>
        public UnknownFieldAnalysis? UnknownFields { get; set; }
        
        /// <summary>Chunk relationship analysis results.</summary>
        public ChunkRelationshipAnalysis? ChunkRelationships { get; set; }
        
        /// <summary>List of issues or anomalies found during analysis.</summary>
        public List<AnalysisIssue> Issues { get; set; } = new();
    }
    
    /// <summary>
    /// Analysis issue or anomaly found in PM4 data.
    /// </summary>
    public class AnalysisIssue
    {
        public string Category { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string Severity { get; set; } = "Info";
        public Dictionary<string, object> Details { get; set; } = new();
    }
    
    /// <summary>
    /// Data structure analysis results.
    /// </summary>
    public class DataStructureAnalysis
    {
        public int VertexCount { get; set; }
        public int IndexCount { get; set; }
        public int TriangleCount { get; set; }
        public int SurfaceCount { get; set; }
        public int LinkCount { get; set; }
        public int PlacementCount { get; set; }
        public int PropertyCount { get; set; }
        public Dictionary<string, int> ChunkCounts { get; set; } = new();
    }
    
    /// <summary>
    /// Index pattern analysis results.
    /// </summary>
    public class IndexPatternAnalysis
    {
        public int MaxVertexIndex { get; set; }
        public int AvailableVertices { get; set; }
        public int OutOfBoundsCount { get; set; }
        public double DataLossPercentage { get; set; }
        public List<uint> SuspiciousIndices { get; set; } = new();
        public Dictionary<string, List<uint>> HighLowPairs { get; set; } = new();
    }
    
    /// <summary>
    /// Unknown field analysis results.
    /// </summary>
    public class UnknownFieldAnalysis
    {
        public Dictionary<string, Dictionary<uint, int>> FieldDistributions { get; set; } = new();
        public List<FieldCorrelation> Correlations { get; set; } = new();
    }
    
    /// <summary>
    /// Field correlation result.
    /// </summary>
    public class FieldCorrelation
    {
        public string Field1 { get; set; } = string.Empty;
        public string Field2 { get; set; } = string.Empty;
        public double CorrelationStrength { get; set; }
        public int MatchCount { get; set; }
    }
    
    /// <summary>
    /// Chunk relationship analysis results.
    /// </summary>
    public class ChunkRelationshipAnalysis
    {
        public int MprlMslkMatches { get; set; }
        public int MprrSentinelCount { get; set; }
        public Dictionary<uint, int> SurfaceGroupDistribution { get; set; } = new();
        public List<string> RelationshipInsights { get; set; } = new();
    }
}
