using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Generates Markdown reports from ADT analysis results.
    /// </summary>
    public class MarkdownReportGenerator
    {
        private readonly ILogger<MarkdownReportGenerator> _logger;

        /// <summary>
        /// Creates a new instance of the MarkdownReportGenerator class.
        /// </summary>
        /// <param name="logger">The logger to use.</param>
        public MarkdownReportGenerator(ILogger<MarkdownReportGenerator> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Generates all Markdown reports for the analysis results.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateReportsAsync(List<AdtAnalysisResult> results, AnalysisSummary summary, string outputDirectory)
        {
            if (results == null)
                throw new ArgumentNullException(nameof(results));
            if (summary == null)
                throw new ArgumentNullException(nameof(summary));
            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentException("Output directory cannot be null or empty.", nameof(outputDirectory));

            // Create Markdown directory
            var mdDirectory = Path.Combine(outputDirectory, "markdown");
            if (!Directory.Exists(mdDirectory))
            {
                Directory.CreateDirectory(mdDirectory);
            }

            _logger.LogInformation("Generating Markdown reports in {MdDirectory}", mdDirectory);

            // Generate summary report
            await GenerateSummaryReportAsync(summary, mdDirectory);

            // Generate ADT files report
            await GenerateAdtFilesReportAsync(results, mdDirectory);
            
            // Generate reference reports
            await GenerateReferenceReportsAsync(results, mdDirectory);
            
            // Generate placement reports - new!
            await GeneratePlacementReportsAsync(results, mdDirectory);
            
            // Generate files not in listfile report - new!
            await GenerateFilesNotInListfileReportAsync(summary, mdDirectory);

            _logger.LogInformation("Markdown report generation complete");
        }

        /// <summary>
        /// Generates a summary report in Markdown format.
        /// </summary>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateSummaryReportAsync(AnalysisSummary summary, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "summary.md");
            _logger.LogInformation("Generating summary report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# ADT Analysis Summary");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync($"**Date:** {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## Analysis Information");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync($"- **Start Time:** {summary.StartTime}");
                await writer.WriteLineAsync($"- **End Time:** {summary.EndTime}");
                await writer.WriteLineAsync($"- **Duration:** {summary.Duration.TotalSeconds:F2} seconds");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## Files Statistics");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync($"- **Total Files:** {summary.TotalFiles}");
                await writer.WriteLineAsync($"- **Processed Files:** {summary.ProcessedFiles}");
                await writer.WriteLineAsync($"- **Failed Files:** {summary.FailedFiles}");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## Content Statistics");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync($"- **Total Texture References:** {summary.TotalTextureReferences}");
                await writer.WriteLineAsync($"- **Total Model References:** {summary.TotalModelReferences}");
                await writer.WriteLineAsync($"- **Total WMO References:** {summary.TotalWmoReferences}");
                await writer.WriteLineAsync($"- **Total Terrain Chunks:** {summary.TotalTerrainChunks}");
                await writer.WriteLineAsync($"- **Total Model Placements:** {summary.TotalModelPlacements}");
                await writer.WriteLineAsync($"- **Total WMO Placements:** {summary.TotalWmoPlacements}");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## Issues");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync($"- **Missing References:** {summary.MissingReferences}");
                await writer.WriteLineAsync($"- **Files Not In Listfile:** {summary.FilesNotInListfile}");
                await writer.WriteLineAsync($"- **Duplicate IDs:** {summary.DuplicateIds}");
                await writer.WriteLineAsync($"- **Maximum Unique ID:** {summary.MaxUniqueId}");
            }
        }

        /// <summary>
        /// Generates a report of all ADT files in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateAdtFilesReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "adt_files.md");
            _logger.LogInformation("Generating ADT files report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# ADT Files Report");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| File Name | Coords | Version | Texture Refs | Model Refs | WMO Refs | Model Placements | WMO Placements | Terrain Chunks |");
                await writer.WriteLineAsync("|-----------|--------|---------|--------------|------------|----------|------------------|----------------|---------------|");
                
                foreach (var result in results.OrderBy(r => r.FileName))
                {
                    await writer.WriteLineAsync(
                        $"| {result.FileName} | ({result.XCoord}, {result.YCoord}) | {result.AdtVersion} | " +
                        $"{result.TextureReferences.Count} | {result.ModelReferences.Count} | {result.WmoReferences.Count} | " +
                        $"{result.ModelPlacements.Count} | {result.WmoPlacements.Count} | {result.TerrainChunks.Count} |");
                }
            }
        }

        /// <summary>
        /// Generates reference reports in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateReferenceReportsAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            // Generate texture references report
            await GenerateTextureReferencesReportAsync(results, outputDirectory);
            
            // Generate model references report
            await GenerateModelReferencesReportAsync(results, outputDirectory);
            
            // Generate WMO references report
            await GenerateWmoReferencesReportAsync(results, outputDirectory);
        }

        /// <summary>
        /// Generates a report of texture references in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateTextureReferencesReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "texture_references.md");
            _logger.LogInformation("Generating texture references report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# Texture References Report");
                await writer.WriteLineAsync("");
                
                // Group textures by original path
                var textureGroups = results
                    .SelectMany(r => r.TextureReferences.Select(t => new { Result = r, Texture = t }))
                    .GroupBy(t => t.Texture.OriginalPath)
                    .OrderBy(g => g.Key);
                
                await writer.WriteLineAsync($"Total unique textures: {textureGroups.Count()}");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| Texture Path | Valid | In Listfile | FileDataID | Used In Files | Repaired Path |");
                await writer.WriteLineAsync("|--------------|-------|-------------|------------|--------------|---------------|");
                
                foreach (var group in textureGroups)
                {
                    var texture = group.First().Texture;
                    var files = group.Select(t => t.Result.FileName).Distinct().ToList();
                    var fileCount = files.Count;
                    
                    var fileDataIdInfo = texture.UsesFileDataId || texture.FileDataId > 0 
                        ? texture.FileDataId.ToString() 
                        : "-";
                    
                    await writer.WriteLineAsync(
                        $"| {texture.OriginalPath} | {texture.IsValid} | {texture.ExistsInListfile} | {fileDataIdInfo} | {fileCount} | " +
                        $"{(string.IsNullOrEmpty(texture.RepairedPath) ? "-" : texture.RepairedPath)} |");
                }
            }
        }

        /// <summary>
        /// Generates a report of model references in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateModelReferencesReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "model_references.md");
            _logger.LogInformation("Generating model references report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# Model References Report");
                await writer.WriteLineAsync("");
                
                // Group models by original path
                var modelGroups = results
                    .SelectMany(r => r.ModelReferences.Select(m => new { Result = r, Model = m }))
                    .GroupBy(m => m.Model.OriginalPath)
                    .OrderBy(g => g.Key);
                
                await writer.WriteLineAsync($"Total unique models: {modelGroups.Count()}");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| Model Path | Valid | In Listfile | FileDataID | Alternative Extension | Used In Files | Repaired Path |");
                await writer.WriteLineAsync("|-----------|-------|-------------|------------|----------------------|--------------|---------------|");
                
                foreach (var group in modelGroups)
                {
                    var model = group.First().Model;
                    var files = group.Select(m => m.Result.FileName).Distinct().ToList();
                    var fileCount = files.Count;
                    
                    var fileDataIdInfo = model.UsesFileDataId || model.FileDataId > 0 
                        ? model.FileDataId.ToString() 
                        : "-";
                    
                    var alternativeInfo = model.AlternativeExtensionFound 
                        ? model.AlternativeExtensionPath 
                        : "-";
                    
                    await writer.WriteLineAsync(
                        $"| {model.OriginalPath} | {model.IsValid} | {model.ExistsInListfile} | {fileDataIdInfo} | {alternativeInfo} | {fileCount} | " +
                        $"{(string.IsNullOrEmpty(model.RepairedPath) ? "-" : model.RepairedPath)} |");
                }
            }
        }

        /// <summary>
        /// Generates a report of WMO references in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateWmoReferencesReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "wmo_references.md");
            _logger.LogInformation("Generating WMO references report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# WMO References Report");
                await writer.WriteLineAsync("");
                
                // Group WMOs by original path
                var wmoGroups = results
                    .SelectMany(r => r.WmoReferences.Select(w => new { Result = r, Wmo = w }))
                    .GroupBy(w => w.Wmo.OriginalPath)
                    .OrderBy(g => g.Key);
                
                await writer.WriteLineAsync($"Total unique WMOs: {wmoGroups.Count()}");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| WMO Path | Valid | In Listfile | FileDataID | Used In Files | Repaired Path |");
                await writer.WriteLineAsync("|----------|-------|-------------|------------|--------------|---------------|");
                
                foreach (var group in wmoGroups)
                {
                    var wmo = group.First().Wmo;
                    var files = group.Select(w => w.Result.FileName).Distinct().ToList();
                    var fileCount = files.Count;
                    
                    var fileDataIdInfo = wmo.UsesFileDataId || wmo.FileDataId > 0 
                        ? wmo.FileDataId.ToString() 
                        : "-";
                    
                    await writer.WriteLineAsync(
                        $"| {wmo.OriginalPath} | {wmo.IsValid} | {wmo.ExistsInListfile} | {fileDataIdInfo} | {fileCount} | " +
                        $"{(string.IsNullOrEmpty(wmo.RepairedPath) ? "-" : wmo.RepairedPath)} |");
                }
            }
        }

        // Add new methods for model and WMO placements reports
        
        /// <summary>
        /// Generates all placement reports in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GeneratePlacementReportsAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            // Generate model placements report
            await GenerateModelPlacementsReportAsync(results, outputDirectory);
            
            // Generate WMO placements report
            await GenerateWmoPlacementsReportAsync(results, outputDirectory);
        }
        
        /// <summary>
        /// Generates a report of model placements in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateModelPlacementsReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "model_placements.md");
            _logger.LogInformation("Generating model placements report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# Model Placements Report");
                await writer.WriteLineAsync("");
                
                // Get all model placements across all ADT files
                var allPlacements = results
                    .SelectMany(r => r.ModelPlacements.Select(p => new { AdtFile = r.FileName, Placement = p }))
                    .ToList();
                
                await writer.WriteLineAsync($"Total model placements: {allPlacements.Count}");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## Model Placement Counts by ADT File");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| ADT File | Model Placements |");
                await writer.WriteLineAsync("|----------|------------------|");
                
                foreach (var group in allPlacements.GroupBy(p => p.AdtFile).OrderBy(g => g.Key))
                {
                    await writer.WriteLineAsync($"| {group.Key} | {group.Count()} |");
                }
                
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("## Top 20 Most Used Models");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| Model Name | Times Placed |");
                await writer.WriteLineAsync("|------------|--------------|");
                
                var topModels = allPlacements
                    .GroupBy(p => p.Placement.Name)
                    .OrderByDescending(g => g.Count())
                    .Take(20);
                
                foreach (var group in topModels)
                {
                    string modelName = string.IsNullOrEmpty(group.Key) ? "<unknown>" : group.Key;
                    await writer.WriteLineAsync($"| {modelName} | {group.Count()} |");
                }
                
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("## Sample Model Placements (First 50)");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| ADT File | Model Name | UniqueID | Position | Rotation | Scale |");
                await writer.WriteLineAsync("|----------|------------|----------|----------|----------|-------|");
                
                foreach (var placement in allPlacements.Take(50))
                {
                    string modelName = string.IsNullOrEmpty(placement.Placement.Name) ? "<unknown>" : placement.Placement.Name;
                    string position = $"({placement.Placement.Position.X:F2}, {placement.Placement.Position.Y:F2}, {placement.Placement.Position.Z:F2})";
                    string rotation = $"({placement.Placement.Rotation.X:F2}, {placement.Placement.Rotation.Y:F2}, {placement.Placement.Rotation.Z:F2})";
                    
                    await writer.WriteLineAsync(
                        $"| {placement.AdtFile} | {modelName} | {placement.Placement.UniqueId} | {position} | {rotation} | {placement.Placement.Scale:F2} |");
                }
            }
        }
        
        /// <summary>
        /// Generates a report of WMO placements in Markdown format.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateWmoPlacementsReportAsync(List<AdtAnalysisResult> results, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "wmo_placements.md");
            _logger.LogInformation("Generating WMO placements report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# WMO Placements Report");
                await writer.WriteLineAsync("");
                
                // Get all WMO placements across all ADT files
                var allPlacements = results
                    .SelectMany(r => r.WmoPlacements.Select(p => new { AdtFile = r.FileName, Placement = p }))
                    .ToList();
                
                await writer.WriteLineAsync($"Total WMO placements: {allPlacements.Count}");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync("## WMO Placement Counts by ADT File");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| ADT File | WMO Placements |");
                await writer.WriteLineAsync("|----------|----------------|");
                
                foreach (var group in allPlacements.GroupBy(p => p.AdtFile).OrderBy(g => g.Key))
                {
                    await writer.WriteLineAsync($"| {group.Key} | {group.Count()} |");
                }
                
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("## Top 20 Most Used WMOs");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| WMO Name | Times Placed |");
                await writer.WriteLineAsync("|----------|--------------|");
                
                var topWmos = allPlacements
                    .GroupBy(p => p.Placement.Name)
                    .OrderByDescending(g => g.Count())
                    .Take(20);
                
                foreach (var group in topWmos)
                {
                    string wmoName = string.IsNullOrEmpty(group.Key) ? "<unknown>" : group.Key;
                    await writer.WriteLineAsync($"| {wmoName} | {group.Count()} |");
                }
                
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("## Sample WMO Placements (First 50)");
                await writer.WriteLineAsync("");
                await writer.WriteLineAsync("| ADT File | WMO Name | UniqueID | Position | Rotation | DoodadSet | NameSet |");
                await writer.WriteLineAsync("|----------|----------|----------|----------|----------|-----------|---------|");
                
                foreach (var placement in allPlacements.Take(50))
                {
                    string wmoName = string.IsNullOrEmpty(placement.Placement.Name) ? "<unknown>" : placement.Placement.Name;
                    string position = $"({placement.Placement.Position.X:F2}, {placement.Placement.Position.Y:F2}, {placement.Placement.Position.Z:F2})";
                    string rotation = $"({placement.Placement.Rotation.X:F2}, {placement.Placement.Rotation.Y:F2}, {placement.Placement.Rotation.Z:F2})";
                    
                    await writer.WriteLineAsync(
                        $"| {placement.AdtFile} | {wmoName} | {placement.Placement.UniqueId} | {position} | {rotation} | {placement.Placement.DoodadSet} | {placement.Placement.NameSet} |");
                }
            }
        }
        
        /// <summary>
        /// Generates a report of files not found in the listfile.
        /// </summary>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the report to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        private async Task GenerateFilesNotInListfileReportAsync(AnalysisSummary summary, string outputDirectory)
        {
            var filePath = Path.Combine(outputDirectory, "files_not_in_listfile.md");
            _logger.LogInformation("Generating files not in listfile report: {FilePath}", filePath);

            using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
            {
                await writer.WriteLineAsync("# Files Not In Listfile Report");
                await writer.WriteLineAsync("");
                
                await writer.WriteLineAsync($"Total files not in listfile: {summary.FilesNotInListfile}");
                await writer.WriteLineAsync("");
                
                if (summary.FilesNotInListfileMap != null && summary.FilesNotInListfileMap.Count > 0)
                {
                    await writer.WriteLineAsync("| File Path | Appears In ADT Files |");
                    await writer.WriteLineAsync("|-----------|---------------------|");
                    
                    foreach (var entry in summary.FilesNotInListfileMap.OrderBy(e => e.Key))
                    {
                        string adtFiles = string.Join(", ", entry.Value.Take(5));
                        if (entry.Value.Count > 5)
                        {
                            adtFiles += $" and {entry.Value.Count - 5} more";
                        }
                        
                        await writer.WriteLineAsync($"| {entry.Key} | {adtFiles} |");
                    }
                }
                else
                {
                    await writer.WriteLineAsync("No files found that are not in the listfile.");
                }
            }
        }
    }
} 