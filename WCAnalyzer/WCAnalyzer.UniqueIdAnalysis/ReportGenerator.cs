using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;

namespace WCAnalyzer.UniqueIdAnalysis
{
    /// <summary>
    /// Generates reports from UniqueID analysis results.
    /// </summary>
    public class ReportGenerator
    {
        private readonly ILogger<ReportGenerator> _logger;
        
        // Increase the default number of top assets to show
        private const int DEFAULT_TOP_ASSETS_COUNT = 50;
        
        public ReportGenerator(ILogger<ReportGenerator> logger = null)
        {
            _logger = logger;
        }
        
        /// <summary>
        /// Generates a summary report for the UniqueID analysis.
        /// </summary>
        /// <param name="result">The analysis result</param>
        /// <param name="outputPath">Path to save the report</param>
        public async Task GenerateSummaryReportAsync(UniqueIdAnalysisResult result, string outputPath)
        {
            _logger?.LogInformation("Generating summary report...");
            
            // Ensure the output directory exists
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
                _logger?.LogInformation("Created output directory: {OutputDir}", outputDir);
            }
            
            var sb = new StringBuilder();
            
            sb.AppendLine("# UniqueID Analysis Summary");
            sb.AppendLine();
            sb.AppendLine($"Analysis Date: {result.AnalysisTime:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine();
            sb.AppendLine("## Overview");
            sb.AppendLine();
            sb.AppendLine($"- Total ADT Files: {result.TotalAdtFiles}");
            sb.AppendLine($"- Total Unique IDs: {result.TotalUniqueIds}");
            sb.AppendLine($"- ID Range: {result.MinUniqueId} - {result.MaxUniqueId}");
            sb.AppendLine($"- Total Assets: {result.TotalAssets}");
            sb.AppendLine($"- Total Clusters: {result.Clusters.Count}");
            sb.AppendLine();
            
            sb.AppendLine("## Clusters");
            sb.AppendLine();
            sb.AppendLine("| Cluster | ID Range | Count | Density | ADTs | Assets |");
            sb.AppendLine("|---------|----------|-------|---------|------|--------|");
            
            int clusterIndex = 1;
            foreach (var cluster in result.Clusters.OrderBy(c => c.MinId))
            {
                sb.AppendLine($"| {clusterIndex++} | {cluster.MinId} - {cluster.MaxId} | {cluster.Count} | {cluster.Density:F2} | {cluster.AdtFiles.Count} | {cluster.Assets.Count} |");
            }
            
            sb.AppendLine();
            sb.AppendLine("## Top Assets by Cluster");
            sb.AppendLine();
            
            clusterIndex = 1;
            foreach (var cluster in result.Clusters.OrderBy(c => c.MinId))
            {
                sb.AppendLine($"### Cluster {clusterIndex++}: {cluster.MinId} - {cluster.MaxId}");
                sb.AppendLine();
                
                var topAssets = cluster.Assets
                    .GroupBy(a => a.AssetPath)
                    .Select(g => new { AssetPath = g.Key, Count = g.Count(), Type = g.First().Type })
                    .OrderByDescending(a => a.Count)
                    .Take(DEFAULT_TOP_ASSETS_COUNT);
                
                sb.AppendLine("| Asset | Type | Count |");
                sb.AppendLine("|-------|------|-------|");
                
                foreach (var asset in topAssets)
                {
                    sb.AppendLine($"| {asset.AssetPath} | {asset.Type} | {asset.Count} |");
                }
                
                sb.AppendLine();
                
                // Add a section for placement examples
                sb.AppendLine("### Placement Examples");
                sb.AppendLine();
                sb.AppendLine("Examples of asset placements in this cluster:");
                sb.AppendLine();
                
                // Group by asset path and take a few examples from each
                var placementExamples = cluster.Assets
                    .GroupBy(a => a.AssetPath)
                    .Take(10) // Take top 10 asset types
                    .SelectMany(g => g.Take(3)) // Take 3 examples of each asset type
                    .Take(30); // Limit to 30 total examples
                
                sb.AppendLine("| Asset | Type | UniqueID | Map | ADT File | Position (X,Y,Z) | Scale |");
                sb.AppendLine("|-------|------|----------|-----|----------|------------------|-------|");
                
                foreach (var asset in placementExamples)
                {
                    sb.AppendLine($"| {asset.AssetPath} | {asset.Type} | {asset.UniqueId} | {asset.MapName} | {asset.AdtFile} | ({asset.PositionX:F2}, {asset.PositionY:F2}, {asset.PositionZ:F2}) | {asset.Scale:F2} |");
                }
                
                sb.AppendLine();
            }
            
            await File.WriteAllTextAsync(outputPath, sb.ToString());
            
            _logger?.LogInformation("Summary report generated at {OutputPath}", outputPath);
        }
        
        /// <summary>
        /// Generates a detailed report for a specific cluster.
        /// </summary>
        /// <param name="cluster">The cluster to generate a report for</param>
        /// <param name="outputPath">Path to save the report</param>
        public async Task GenerateClusterReportAsync(UniqueIdCluster cluster, string outputPath)
        {
            _logger?.LogInformation("Generating cluster report for {MinId}-{MaxId}...", cluster.MinId, cluster.MaxId);
            
            // Ensure the output directory exists
            var outputDir = Path.GetDirectoryName(outputPath);
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
                _logger?.LogInformation("Created output directory: {OutputDir}", outputDir);
            }
            
            var sb = new StringBuilder();
            
            sb.AppendLine($"# Cluster Report: {cluster.MinId} - {cluster.MaxId}");
            sb.AppendLine();
            sb.AppendLine("## Overview");
            sb.AppendLine();
            sb.AppendLine($"- ID Range: {cluster.MinId} - {cluster.MaxId}");
            sb.AppendLine($"- Count: {cluster.Count}");
            sb.AppendLine($"- Density: {cluster.Density:F2}");
            sb.AppendLine($"- ADT Files: {cluster.AdtFiles.Count}");
            sb.AppendLine($"- Assets: {cluster.Assets.Count}");
            sb.AppendLine();
            
            // Add ID distribution analysis
            sb.AppendLine("## ID Distribution");
            sb.AppendLine();
            sb.AppendLine("Distribution of UniqueIDs within this cluster:");
            sb.AppendLine();
            
            // Create a histogram of IDs in the cluster
            var idCounts = new Dictionary<int, int>();
            foreach (var asset in cluster.Assets)
            {
                if (!idCounts.ContainsKey(asset.UniqueId))
                {
                    idCounts[asset.UniqueId] = 0;
                }
                idCounts[asset.UniqueId]++;
            }
            
            // Calculate some statistics
            var idCountsList = idCounts.Values.ToList();
            var avgAssetsPerID = idCountsList.Count > 0 ? idCountsList.Average() : 0;
            var maxAssetsPerID = idCountsList.Count > 0 ? idCountsList.Max() : 0;
            
            sb.AppendLine($"- Average assets per UniqueID: {avgAssetsPerID:F2}");
            sb.AppendLine($"- Maximum assets per UniqueID: {maxAssetsPerID}");
            sb.AppendLine();
            
            sb.AppendLine("## ADT Files");
            sb.AppendLine();
            sb.AppendLine("| ADT File | ID Count |");
            sb.AppendLine("|----------|----------|");
            
            foreach (var adt in cluster.IdCountsByAdt.OrderByDescending(kv => kv.Value))
            {
                sb.AppendLine($"| {adt.Key} | {adt.Value} |");
            }
            
            sb.AppendLine();
            sb.AppendLine("## Top Assets");
            sb.AppendLine();
            
            // Group assets by path and count occurrences
            var assetGroups = cluster.Assets
                .GroupBy(a => a.AssetPath)
                .Select(g => new { AssetPath = g.Key, Count = g.Count(), Type = g.First().Type })
                .OrderByDescending(a => a.Count)
                .Take(DEFAULT_TOP_ASSETS_COUNT);
            
            sb.AppendLine("| Asset | Type | Count |");
            sb.AppendLine("|-------|------|-------|");
            
            foreach (var asset in assetGroups)
            {
                sb.AppendLine($"| {asset.AssetPath} | {asset.Type} | {asset.Count} |");
            }
            
            sb.AppendLine();
            sb.AppendLine("## Detailed Asset Placements");
            sb.AppendLine();
            sb.AppendLine("| Asset | Type | UniqueID | Map | ADT File | Position (X,Y,Z) | Scale |");
            sb.AppendLine("|-------|------|----------|-----|----------|------------------|-------|");
            
            // Take a reasonable number of assets to avoid extremely large reports
            var detailedAssets = cluster.Assets
                .OrderBy(a => a.UniqueId)
                .ThenBy(a => a.AssetPath)
                .Take(500); // Limit to 500 detailed entries
            
            foreach (var asset in detailedAssets)
            {
                sb.AppendLine($"| {asset.AssetPath} | {asset.Type} | {asset.UniqueId} | {asset.MapName} | {asset.AdtFile} | ({asset.PositionX:F2}, {asset.PositionY:F2}, {asset.PositionZ:F2}) | {asset.Scale:F2} |");
            }
            
            // If there are more assets than we're showing, add a note
            if (cluster.Assets.Count > 500)
            {
                sb.AppendLine();
                sb.AppendLine($"*Note: Showing 500 of {cluster.Assets.Count} total asset placements. See the full data for complete details.*");
            }
            
            await File.WriteAllTextAsync(outputPath, sb.ToString());
            
            _logger?.LogInformation("Cluster report generated at {OutputPath}", outputPath);
        }
    }
} 