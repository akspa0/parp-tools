using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ClosedXML.Excel;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Service for generating Excel reports from ADT analysis results.
    /// </summary>
    public class ExcelReportGenerator
    {
        private readonly ILogger<ExcelReportGenerator> _logger;

        public ExcelReportGenerator(ILogger<ExcelReportGenerator> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generates Excel reports for the analysis results.
        /// </summary>
        /// <param name="results">The ADT analysis results.</param>
        /// <param name="summary">The analysis summary.</param>
        /// <param name="outputDirectory">The directory to write the reports to.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        public async Task GenerateReportsAsync(List<AdtAnalysisResult> results, AnalysisSummary summary, string outputDirectory)
        {
            // Create reports directory
            var reportsDir = Path.Combine(outputDirectory, "excel");
            if (!Directory.Exists(reportsDir))
            {
                Directory.CreateDirectory(reportsDir);
            }

            var filePath = Path.Combine(reportsDir, "adt_analysis.xlsx");
            _logger.LogInformation("Generating Excel report: {Path}", filePath);

            using var workbook = new XLWorkbook();

            // Generate summary sheet
            GenerateSummarySheet(workbook, summary);

            // Generate ADT files sheet
            GenerateAdtFilesSheet(workbook, results);

            // Generate references sheet
            GenerateReferencesSheet(workbook, results);

            // Generate placements sheet
            GeneratePlacements(workbook, results);

            // Generate terrain data sheet
            GenerateTerrainDataSheet(workbook, results);

            // Save the workbook
            workbook.SaveAs(filePath);
        }

        private void GenerateSummarySheet(XLWorkbook workbook, AnalysisSummary summary)
        {
            var worksheet = workbook.Worksheets.Add("Summary");

            // Title
            worksheet.Cell("A1").Value = "ADT Analysis Summary";
            worksheet.Cell("A1").Style.Font.Bold = true;
            worksheet.Cell("A1").Style.Font.FontSize = 14;

            // Analysis Information
            worksheet.Cell("A3").Value = "Analysis Information";
            worksheet.Cell("A3").Style.Font.Bold = true;
            AddStatRow(worksheet, 4, "Start Time", summary.StartTime.ToString("yyyy-MM-dd HH:mm:ss"));
            AddStatRow(worksheet, 5, "End Time", summary.EndTime.ToString("yyyy-MM-dd HH:mm:ss"));
            AddStatRow(worksheet, 6, "Duration", summary.Duration.ToString());

            // Statistics
            worksheet.Cell("A8").Value = "Statistics";
            worksheet.Cell("A8").Style.Font.Bold = true;

            // File Statistics
            worksheet.Cell("A10").Value = "File Statistics";
            worksheet.Cell("A10").Style.Font.Bold = true;
            AddStatRow(worksheet, 11, "Total Files", summary.TotalFiles);
            AddStatRow(worksheet, 12, "Processed Files", summary.ProcessedFiles);
            AddStatRow(worksheet, 13, "Failed Files", summary.FailedFiles);

            // Reference Statistics
            worksheet.Cell("A15").Value = "Reference Statistics";
            worksheet.Cell("A15").Style.Font.Bold = true;
            AddStatRow(worksheet, 16, "Total Texture References", summary.TotalTextureReferences);
            AddStatRow(worksheet, 17, "Total Model References", summary.TotalModelReferences);
            AddStatRow(worksheet, 18, "Total WMO References", summary.TotalWmoReferences);

            // Placement Statistics
            worksheet.Cell("A20").Value = "Placement Statistics";
            worksheet.Cell("A20").Style.Font.Bold = true;
            AddStatRow(worksheet, 21, "Total Model Placements", summary.TotalModelPlacements);
            AddStatRow(worksheet, 22, "Total WMO Placements", summary.TotalWmoPlacements);

            // Terrain Statistics
            worksheet.Cell("A24").Value = "Terrain Statistics";
            worksheet.Cell("A24").Style.Font.Bold = true;
            AddStatRow(worksheet, 25, "Total Terrain Chunks", summary.TotalTerrainChunks);
            AddStatRow(worksheet, 26, "Total Texture Layers", summary.TotalTextureLayers);
            AddStatRow(worksheet, 27, "Total Doodad References", summary.TotalDoodadReferences);

            // Issues
            worksheet.Cell("A29").Value = "Issues";
            worksheet.Cell("A29").Style.Font.Bold = true;
            AddStatRow(worksheet, 30, "Missing References", summary.MissingReferences);
            AddStatRow(worksheet, 31, "Files Not In Listfile", summary.FilesNotInListfile);
            AddStatRow(worksheet, 32, "Duplicate IDs", summary.DuplicateIds);
            AddStatRow(worksheet, 33, "Maximum Unique ID", summary.MaxUniqueId);
            AddStatRow(worksheet, 34, "Parsing Errors", summary.ParsingErrors);
            AddStatRow(worksheet, 35, "Unique Area IDs", summary.AreaIdMap.Count);

            // Area ID Summary
            worksheet.Cell("A37").Value = "Area ID Summary";
            worksheet.Cell("A37").Style.Font.Bold = true;

            var row = 38;
            foreach (var area in summary.AreaIdMap.OrderBy(a => a.Key))
            {
                worksheet.Cell(row, 1).Value = $"Area ID: {area.Key}";
                worksheet.Cell(row, 1).Style.Font.Bold = true;
                row++;

                foreach (var file in area.Value.OrderBy(f => f))
                {
                    worksheet.Cell(row, 2).Value = file;
                    row++;
                }
                row++;
            }

            worksheet.Columns().AdjustToContents();
        }

        private void GenerateAdtFilesSheet(XLWorkbook workbook, List<AdtAnalysisResult> results)
        {
            var worksheet = workbook.Worksheets.Add("ADT Files");

            // Headers
            worksheet.Cell("A1").Value = "File Name";
            worksheet.Cell("B1").Value = "Coordinates";
            worksheet.Cell("C1").Value = "Version";
            worksheet.Cell("D1").Value = "Terrain Chunks";
            worksheet.Cell("E1").Value = "Texture References";
            worksheet.Cell("F1").Value = "Model References";
            worksheet.Cell("G1").Value = "WMO References";
            worksheet.Cell("H1").Value = "Model Placements";
            worksheet.Cell("I1").Value = "WMO Placements";
            worksheet.Cell("J1").Value = "Errors";

            // Style headers
            var headerRange = worksheet.Range("A1:J1");
            headerRange.Style.Font.Bold = true;
            headerRange.Style.Fill.BackgroundColor = XLColor.LightGray;

            // Data
            var row = 2;
            foreach (var result in results.OrderBy(r => r.FileName))
            {
                worksheet.Cell(row, 1).Value = result.FileName;
                worksheet.Cell(row, 2).Value = $"({result.XCoord}, {result.YCoord})";
                worksheet.Cell(row, 3).Value = result.AdtVersion;
                worksheet.Cell(row, 4).Value = result.TerrainChunks.Count;
                worksheet.Cell(row, 5).Value = result.TextureReferences.Count;
                worksheet.Cell(row, 6).Value = result.ModelReferences.Count;
                worksheet.Cell(row, 7).Value = result.WmoReferences.Count;
                worksheet.Cell(row, 8).Value = result.ModelPlacements.Count;
                worksheet.Cell(row, 9).Value = result.WmoPlacements.Count;
                worksheet.Cell(row, 10).Value = result.Errors.Count;

                if (result.Errors.Any())
                {
                    worksheet.Cell(row, 10).Style.Fill.BackgroundColor = XLColor.LightPink;
                }

                row++;
            }

            worksheet.Columns().AdjustToContents();
        }

        private void GenerateReferencesSheet(XLWorkbook workbook, List<AdtAnalysisResult> results)
        {
            var worksheet = workbook.Worksheets.Add("References");

            // Headers
            worksheet.Cell("A1").Value = "File Name";
            worksheet.Cell("B1").Value = "Type";
            worksheet.Cell("C1").Value = "Path";
            worksheet.Cell("D1").Value = "Valid";
            worksheet.Cell("E1").Value = "In Listfile";
            worksheet.Cell("F1").Value = "Repaired Path";

            // Style headers
            var headerRange = worksheet.Range("A1:F1");
            headerRange.Style.Font.Bold = true;
            headerRange.Style.Fill.BackgroundColor = XLColor.LightGray;

            // Data
            var row = 2;
            foreach (var result in results.OrderBy(r => r.FileName))
            {
                // Texture References
                foreach (var texture in result.TextureReferences.OrderBy(t => t.OriginalPath))
                {
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = "Texture";
                    worksheet.Cell(row, 3).Value = texture.OriginalPath;
                    worksheet.Cell(row, 4).Value = texture.IsValid;
                    worksheet.Cell(row, 5).Value = texture.ExistsInListfile;
                    worksheet.Cell(row, 6).Value = texture.RepairedPath;

                    if (!texture.IsValid || !texture.ExistsInListfile)
                    {
                        worksheet.Row(row).Style.Fill.BackgroundColor = XLColor.LightPink;
                    }

                    row++;
                }

                // Model References
                foreach (var model in result.ModelReferences.OrderBy(m => m.OriginalPath))
                {
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = "Model";
                    worksheet.Cell(row, 3).Value = model.OriginalPath;
                    worksheet.Cell(row, 4).Value = model.IsValid;
                    worksheet.Cell(row, 5).Value = model.ExistsInListfile;
                    worksheet.Cell(row, 6).Value = model.RepairedPath;

                    if (!model.IsValid || !model.ExistsInListfile)
                    {
                        worksheet.Row(row).Style.Fill.BackgroundColor = XLColor.LightPink;
                    }

                    row++;
                }

                // WMO References
                foreach (var wmo in result.WmoReferences.OrderBy(w => w.OriginalPath))
                {
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = "WMO";
                    worksheet.Cell(row, 3).Value = wmo.OriginalPath;
                    worksheet.Cell(row, 4).Value = wmo.IsValid;
                    worksheet.Cell(row, 5).Value = wmo.ExistsInListfile;
                    worksheet.Cell(row, 6).Value = wmo.RepairedPath;

                    if (!wmo.IsValid || !wmo.ExistsInListfile)
                    {
                        worksheet.Row(row).Style.Fill.BackgroundColor = XLColor.LightPink;
                    }

                    row++;
                }

                row++;
            }

            worksheet.Columns().AdjustToContents();
        }

        private void GeneratePlacements(XLWorkbook workbook, List<AdtAnalysisResult> results)
        {
            var worksheet = workbook.Worksheets.Add("Placements");

            // Headers
            worksheet.Cell("A1").Value = "File Name";
            worksheet.Cell("B1").Value = "Type";
            worksheet.Cell("C1").Value = "Unique ID";
            worksheet.Cell("D1").Value = "Position X";
            worksheet.Cell("E1").Value = "Position Y";
            worksheet.Cell("F1").Value = "Position Z";
            worksheet.Cell("G1").Value = "Rotation X";
            worksheet.Cell("H1").Value = "Rotation Y";
            worksheet.Cell("I1").Value = "Rotation Z";
            worksheet.Cell("J1").Value = "Scale";

            // Style headers
            var headerRange = worksheet.Range("A1:J1");
            headerRange.Style.Font.Bold = true;
            headerRange.Style.Fill.BackgroundColor = XLColor.LightGray;

            // Data
            var row = 2;
            foreach (var result in results.OrderBy(r => r.FileName))
            {
                // Model Placements
                foreach (var placement in result.ModelPlacements.OrderBy(p => p.UniqueId))
                {
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = "Model";
                    worksheet.Cell(row, 3).Value = placement.UniqueId;
                    worksheet.Cell(row, 4).Value = placement.Position.X;
                    worksheet.Cell(row, 5).Value = placement.Position.Y;
                    worksheet.Cell(row, 6).Value = placement.Position.Z;
                    worksheet.Cell(row, 7).Value = placement.Rotation.X;
                    worksheet.Cell(row, 8).Value = placement.Rotation.Y;
                    worksheet.Cell(row, 9).Value = placement.Rotation.Z;
                    worksheet.Cell(row, 10).Value = placement.Scale;

                    row++;
                }

                // WMO Placements
                foreach (var placement in result.WmoPlacements.OrderBy(p => p.UniqueId))
                {
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = "WMO";
                    worksheet.Cell(row, 3).Value = placement.UniqueId;
                    worksheet.Cell(row, 4).Value = placement.Position.X;
                    worksheet.Cell(row, 5).Value = placement.Position.Y;
                    worksheet.Cell(row, 6).Value = placement.Position.Z;
                    worksheet.Cell(row, 7).Value = placement.Rotation.X;
                    worksheet.Cell(row, 8).Value = placement.Rotation.Y;
                    worksheet.Cell(row, 9).Value = placement.Rotation.Z;
                    worksheet.Cell(row, 10).Value = 1.0; // WMOs don't have scale

                    row++;
                }

                row++;
            }

            worksheet.Columns().AdjustToContents();
        }

        private void GenerateTerrainDataSheet(XLWorkbook workbook, List<AdtAnalysisResult> results)
        {
            var worksheet = workbook.Worksheets.Add("Terrain Data");

            // Headers
            worksheet.Cell("A1").Value = "File Name";
            worksheet.Cell("B1").Value = "Chunk Index";
            worksheet.Cell("C1").Value = "Area ID";
            worksheet.Cell("D1").Value = "Texture Layers";
            worksheet.Cell("E1").Value = "Doodads";
            worksheet.Cell("F1").Value = "Height Min";
            worksheet.Cell("G1").Value = "Height Max";
            worksheet.Cell("H1").Value = "Height Average";

            // Style headers
            var headerRange = worksheet.Range("A1:H1");
            headerRange.Style.Font.Bold = true;
            headerRange.Style.Fill.BackgroundColor = XLColor.LightGray;

            // Data
            var row = 2;
            foreach (var result in results.OrderBy(r => r.FileName))
            {
                for (int i = 0; i < result.TerrainChunks.Count; i++)
                {
                    var chunk = result.TerrainChunks[i];
                    worksheet.Cell(row, 1).Value = result.FileName;
                    worksheet.Cell(row, 2).Value = i;
                    worksheet.Cell(row, 3).Value = chunk.AreaId;
                    worksheet.Cell(row, 4).Value = chunk.TextureLayers.Count;
                    worksheet.Cell(row, 5).Value = chunk.DoodadRefs.Count;
                    worksheet.Cell(row, 6).Value = chunk.Heights.Min();
                    worksheet.Cell(row, 7).Value = chunk.Heights.Max();
                    worksheet.Cell(row, 8).Value = chunk.Heights.Average();

                    row++;
                }
                row++;
            }

            worksheet.Columns().AdjustToContents();
        }

        private void AddStatRow(IXLWorksheet worksheet, int row, string label, object value)
        {
            worksheet.Cell(row, 1).Value = label;
            worksheet.Cell(row, 2).Value = (ClosedXML.Excel.XLCellValue)value;
        }
    }
} 