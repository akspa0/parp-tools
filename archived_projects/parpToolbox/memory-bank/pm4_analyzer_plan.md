# PM4 Analyzer Consolidation Plan

## Overview

This document outlines the plan to consolidate multiple fragmented PM4 analyzer tools into a single, unified `Pm4Analyzer` class. This consolidation is part of the larger parpToolbox refactoring effort to simplify the codebase and improve maintainability.

## Current State

The PM4 analysis functionality is currently spread across multiple fragmented classes:

1. `Pm4DataAnalyzer.cs` - General PM4 data structure analysis
2. `Pm4IndexPatternAnalyzer.cs` - Analyzes index patterns in PM4 geometry
3. `Pm4UnknownFieldAnalyzer.cs` - Analyzes unknown fields in PM4 chunks
4. `Pm4ChunkCombinationTester.cs` - Tests relationships between chunks
5. `Pm4BulkDumper.cs` - Dumps bulk information about PM4 files
6. `Pm4CsvDumper.cs` - Exports PM4 data to CSV format

These fragmented tools make it difficult to maintain the codebase and introduce new features. A unified approach would provide a more cohesive and maintainable solution.

## Design Goals

The unified `Pm4Analyzer` class should:

1. Provide a consistent interface for all PM4 analysis operations
2. Support configurable analysis options
3. Preserve all existing analysis capabilities
4. Generate structured reports in multiple formats (console, CSV, JSON)
5. Integrate key format insights and discoveries directly in the code
6. Support advanced analysis like cross-tile vertex reference validation
7. Be extensible for future analysis needs

## Class Structure

```csharp
namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Unified PM4 analyzer with support for multiple analysis types and reporting formats.
    /// </summary>
    public class Pm4Analyzer
    {
        // Configuration options
        public class AnalysisOptions { ... }
        
        // Analysis types
        public enum AnalysisType { ... }
        
        // Report formats
        public enum ReportFormat { ... }
        
        // Constructor
        public Pm4Analyzer(Pm4Scene scene, AnalysisOptions options = null) { ... }
        
        // Main analysis method
        public AnalysisReport Analyze() { ... }
        
        // Individual analysis methods
        private AnalysisReport AnalyzeDataStructure() { ... }
        private AnalysisReport AnalyzeIndexPatterns() { ... }
        private AnalysisReport AnalyzeUnknownFields() { ... }
        private AnalysisReport AnalyzeChunkRelationships() { ... }
        private AnalysisReport AnalyzeCrossTileReferences() { ... }
        
        // Report generation
        private void GenerateReport(AnalysisReport report, string outputPath) { ... }
    }
}
```

## Analysis Options

The `AnalysisOptions` class will provide configuration for the analysis process:

```csharp
public class AnalysisOptions
{
    // What types of analysis to perform
    public AnalysisType Types { get; set; } = AnalysisType.All;
    
    // Report format(s) to generate
    public ReportFormat Format { get; set; } = ReportFormat.Console;
    
    // Where to write output files
    public string OutputPath { get; set; }
    
    // Whether to include detailed information in reports
    public bool Detailed { get; set; } = false;
    
    // Whether to analyze cross-tile references
    public bool AnalyzeCrossTile { get; set; } = true;
    
    // Whether to analyze unknown fields
    public bool AnalyzeUnknownFields { get; set; } = true;
    
    // Whether to analyze MPRR sentinel patterns
    public bool AnalyzeMprrSentinels { get; set; } = true;
    
    // Whether to analyze MSUR grouping patterns
    public bool AnalyzeMsurGroups { get; set; } = true;
    
    // Whether to show progress in console
    public bool Verbose { get; set; } = true;
}
```

## Analysis Types

```csharp
[Flags]
public enum AnalysisType
{
    None = 0,
    DataStructure = 1 << 0,  // Basic data structure analysis
    IndexPatterns = 1 << 1,  // Index pattern analysis
    UnknownFields = 1 << 2,  // Unknown field analysis
    ChunkRelationships = 1 << 3,  // Relationship between chunks
    CrossTileReferences = 1 << 4,  // Cross-tile vertex reference analysis
    MprrSentinels = 1 << 5,  // MPRR sentinel analysis
    MsurGroups = 1 << 6,  // MSUR grouping analysis
    All = DataStructure | IndexPatterns | UnknownFields | ChunkRelationships | CrossTileReferences | MprrSentinels | MsurGroups
}
```

## Report Formats

```csharp
[Flags]
public enum ReportFormat
{
    None = 0,
    Console = 1 << 0,  // Output to console
    Csv = 1 << 1,  // Output to CSV files
    Json = 1 << 2,  // Output to JSON files
    All = Console | Csv | Json
}
```

## Analysis Report Structure

```csharp
public class AnalysisReport
{
    public string Title { get; set; }
    public AnalysisType Type { get; set; }
    public Dictionary<string, Dictionary<string, object>> Categories { get; set; } = new();
    public List<AnalysisIssue> Issues { get; set; } = new();
    public Dictionary<string, object> Statistics { get; set; } = new();
    public DateTime AnalysisTime { get; set; }
    
    // Add result to a category
    public void AddResult(string category, string key, object value, string description = null) { ... }
    
    // Add an issue
    public void AddIssue(string category, IssueSeverity severity, string message, string details = null) { ... }
    
    // Add a statistic
    public void AddStatistic(string key, object value) { ... }
}
```

## Implementation Plan

1. Create the `Pm4Analyzer` class with basic structure and configuration options
2. Implement the data structure analysis functionality from `Pm4DataAnalyzer`
3. Implement the index pattern analysis functionality from `Pm4IndexPatternAnalyzer`
4. Implement the unknown field analysis functionality from `Pm4UnknownFieldAnalyzer`
5. Implement the chunk relationship analysis functionality from `Pm4ChunkCombinationTester`
6. Implement the cross-tile reference validation functionality
7. Implement the report generation in multiple formats
8. Add documentation and examples

## Key Insights to Preserve

The following key insights about the PM4 format must be documented in the code:

1. **Cross-Tile Vertex References**: PM4 files contain vertex indices that reference vertices from adjacent tiles. This requires proper resolution to avoid data loss.

2. **MSUR-Based Grouping**: The `MSUR.SurfaceGroupKey` (raw field `FlagsOrUnknown_0x00`) provides semantically meaningful grouping for building objects.

3. **MPRR Sentinel Values**: The MPRR chunk contains sentinel values (`Value1=65535`) that define object boundaries and produce coherent building-scale objects.

4. **MSLK Link Relationships**: The `MSLK.ReferenceIndex` field may link parts of complex objects, and the high word (`FFFF`) of `LinkId` might be padding.

## Integration with CLI

The unified `Pm4Analyzer` will be exposed through a single `pm4-analyze` CLI command with options for different analysis types and reporting formats:

```
pm4-analyze <file> [--output-dir <dir>] [--format <format>] [--analysis-type <type>] [--detailed] [--verbose]
```

## Testing Strategy

Testing the unified analyzer will involve:

1. Creating unit tests that use real PM4 data files (not mocks)
2. Verifying that all analysis results match the output of the original analyzers
3. Testing each analysis type individually and in combination
4. Testing each report format
5. Testing with different configuration options

## Migration Plan

1. Create the unified `Pm4Analyzer` class
2. Migrate functionality from existing analyzers one by one
3. Update any dependent code to use the new unified interface
4. Mark old analyzers as obsolete with guidance to use the new analyzer
5. Remove old analyzers once all dependencies have been updated

## Future Enhancements

Future enhancements to the analyzer could include:

1. Interactive analysis mode with CLI prompts
2. Visual reports (charts, diagrams)
3. Integration with a web-based viewer
4. Machine learning-based pattern detection for unknown fields
5. Automated validation of format assumptions
