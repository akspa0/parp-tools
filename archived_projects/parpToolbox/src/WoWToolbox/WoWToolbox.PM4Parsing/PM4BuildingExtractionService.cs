using WoWToolbox.Core;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Models;
using WoWToolbox.PM4Parsing.BuildingExtraction;
using WoWToolbox.PM4Parsing.NodeSystem;

namespace WoWToolbox.PM4Parsing
{
    /// <summary>
    /// High-level service for extracting buildings from PM4 navigation files.
    /// Provides complete functionality with analysis, extraction, and export capabilities.
    /// </summary>
    public class PM4BuildingExtractionService
    {
        private readonly PM4BuildingExtractor _buildingExtractor;
        private readonly MslkRootNodeDetector _rootNodeDetector;

        public PM4BuildingExtractionService()
        {
            _buildingExtractor = new PM4BuildingExtractor();
            _rootNodeDetector = new MslkRootNodeDetector();
        }

        /// <summary>
        /// Complete building extraction workflow: extract buildings and export to OBJ files.
        /// </summary>
        /// <param name="pm4FilePath">Path to the PM4 file</param>
        /// <param name="outputDirectory">Directory for output files</param>
        /// <returns>Extraction result with buildings and metadata</returns>
        public BuildingExtractionResult ExtractAndExportBuildings(string pm4FilePath, string outputDirectory)
        {
            // Load PM4 file
            var pm4File = PM4File.FromFile(pm4FilePath);
            var sourceFileName = Path.GetFileNameWithoutExtension(pm4FilePath);

            // Create output directory
            Directory.CreateDirectory(outputDirectory);

            // Analyze the PM4 structure
            var analysisResult = AnalyzePM4Structure(pm4File);

            // Extract buildings
            var buildings = _buildingExtractor.ExtractBuildings(pm4File, sourceFileName);

            // Export buildings to OBJ files
            var exportedFiles = new List<string>();
            for (int i = 0; i < buildings.Count; i++)
            {
                var building = buildings[i];
                var objPath = Path.Combine(outputDirectory, $"{sourceFileName}_Building_{i + 1:D2}.obj");
                CompleteWMOModelUtilities.ExportToOBJ(building, objPath);
                exportedFiles.Add(objPath);
            }

            // Generate summary report
            var summaryPath = Path.Combine(outputDirectory, $"{sourceFileName}_extraction_summary.txt");
            GenerateExtractionSummary(analysisResult, buildings, summaryPath);

            return new BuildingExtractionResult
            {
                SourceFile = pm4FilePath,
                AnalysisResult = analysisResult,
                Buildings = buildings,
                ExportedFiles = exportedFiles,
                SummaryReportPath = summaryPath
            };
        }

        /// <summary>
        /// Analyzes PM4 file structure for extraction planning.
        /// </summary>
        /// <param name="pm4File">The PM4 file to analyze</param>
        /// <returns>Analysis result with structure information</returns>
        public PM4StructureAnalysis AnalyzePM4Structure(PM4File pm4File)
        {
            var analysis = new PM4StructureAnalysis();

            // Basic chunk analysis
            analysis.HasMSLK = pm4File.MSLK?.Entries?.Count > 0;
            analysis.HasMSPV = pm4File.MSPV?.Vertices?.Count > 0;
            analysis.HasMSVT = pm4File.MSVT?.Vertices?.Count > 0;
            analysis.HasMSUR = pm4File.MSUR?.Entries?.Count > 0;
            analysis.HasMDSF = pm4File.MDSF?.Entries?.Count > 0;
            analysis.HasMDOS = pm4File.MDOS?.Entries?.Count > 0;

            // Counts
            analysis.MSLKCount = pm4File.MSLK?.Entries?.Count ?? 0;
            analysis.MSPVCount = pm4File.MSPV?.Vertices?.Count ?? 0;
            analysis.MSVTCount = pm4File.MSVT?.Vertices?.Count ?? 0;
            analysis.MSURCount = pm4File.MSUR?.Entries?.Count ?? 0;
            analysis.MDSFCount = pm4File.MDSF?.Entries?.Count ?? 0;
            analysis.MDOSCount = pm4File.MDOS?.Entries?.Count ?? 0;

            // Determine extraction strategy
            analysis.SupportsMdsfMdosExtraction = analysis.HasMDSF && analysis.HasMDOS;
            analysis.SupportsMslkRootNodeExtraction = analysis.HasMSLK;

            // MSLK hierarchy analysis
            if (analysis.HasMSLK)
            {
                analysis.HierarchyStatistics = _rootNodeDetector.GetHierarchyStatistics(pm4File);
                analysis.RootNodes = _rootNodeDetector.DetectRootNodes(pm4File);
            }

            // Determine recommended strategy
            if (analysis.SupportsMdsfMdosExtraction)
            {
                analysis.RecommendedStrategy = "MDSF/MDOS Building IDs";
                analysis.StrategyReason = "MDSF/MDOS chunks available for precise building separation";
            }
            else if (analysis.SupportsMslkRootNodeExtraction)
            {
                analysis.RecommendedStrategy = "MSLK Root Node Clustering";
                analysis.StrategyReason = "Fallback to MSLK hierarchy with spatial clustering";
            }
            else
            {
                analysis.RecommendedStrategy = "None";
                analysis.StrategyReason = "Missing required chunks for building extraction";
            }

            return analysis;
        }

        /// <summary>
        /// Generates a detailed extraction summary report.
        /// </summary>
        private void GenerateExtractionSummary(PM4StructureAnalysis analysis, List<CompleteWMOModel> buildings, string summaryPath)
        {
            using var writer = new StreamWriter(summaryPath);

            writer.WriteLine("PM4 BUILDING EXTRACTION SUMMARY");
            writer.WriteLine($"Generated: {DateTime.Now}");
            writer.WriteLine($"Source: {analysis.RecommendedStrategy}");
            writer.WriteLine(new string('=', 60));
            writer.WriteLine();

            // PM4 Structure Analysis
            writer.WriteLine("PM4 STRUCTURE ANALYSIS:");
            writer.WriteLine($"  MSLK Entries: {analysis.MSLKCount:N0}");
            writer.WriteLine($"  MSPV Vertices: {analysis.MSPVCount:N0}");
            writer.WriteLine($"  MSVT Vertices: {analysis.MSVTCount:N0}");
            writer.WriteLine($"  MSUR Surfaces: {analysis.MSURCount:N0}");
            writer.WriteLine($"  MDSF Links: {analysis.MDSFCount:N0}");
            writer.WriteLine($"  MDOS Buildings: {analysis.MDOSCount:N0}");
            writer.WriteLine();

            // Extraction Strategy
            writer.WriteLine("EXTRACTION STRATEGY:");
            writer.WriteLine($"  Method: {analysis.RecommendedStrategy}");
            writer.WriteLine($"  Reason: {analysis.StrategyReason}");
            writer.WriteLine();

            // Hierarchy Statistics (if available)
            if (analysis.HierarchyStatistics != null)
            {
                var stats = analysis.HierarchyStatistics;
                writer.WriteLine("MSLK HIERARCHY ANALYSIS:");
                writer.WriteLine($"  Total Nodes: {stats.TotalNodes:N0}");
                writer.WriteLine($"  Root Nodes: {stats.RootNodeCount:N0}");
                writer.WriteLine($"  Root Nodes with Geometry: {stats.RootNodesWithGeometry:N0}");
                writer.WriteLine($"  Child Nodes: {stats.TotalChildNodes:N0}");
                writer.WriteLine($"  Orphaned Nodes: {stats.OrphanedNodes:N0}");
                if (stats.RootNodeCount > 0)
                {
                    writer.WriteLine($"  Avg Child Nodes per Root: {stats.AverageChildNodesPerRoot:F1}");
                }
                writer.WriteLine();
            }

            // Building Results
            writer.WriteLine("EXTRACTION RESULTS:");
            writer.WriteLine($"  Buildings Extracted: {buildings.Count:N0}");
            writer.WriteLine();

            for (int i = 0; i < buildings.Count; i++)
            {
                var building = buildings[i];
                writer.WriteLine($"Building {i + 1:D2} ({building.Category}):");
                writer.WriteLine($"  Vertices: {building.VertexCount:N0}");
                writer.WriteLine($"  Faces: {building.FaceCount:N0}");
                writer.WriteLine($"  Material: {building.MaterialName}");

                // Write metadata
                foreach (var metadata in building.Metadata)
                {
                    writer.WriteLine($"  {metadata.Key}: {metadata.Value}");
                }
                writer.WriteLine();
            }

            // Totals
            var totalVertices = buildings.Sum(b => b.VertexCount);
            var totalFaces = buildings.Sum(b => b.FaceCount);
            writer.WriteLine("TOTALS:");
            writer.WriteLine($"  Combined Vertices: {totalVertices:N0}");
            writer.WriteLine($"  Combined Faces: {totalFaces:N0}");
        }

        /// <summary>
        /// Result of PM4 building extraction operation.
        /// </summary>
        public class BuildingExtractionResult
        {
            public string SourceFile { get; set; } = "";
            public PM4StructureAnalysis AnalysisResult { get; set; } = new();
            public List<CompleteWMOModel> Buildings { get; set; } = new();
            public List<string> ExportedFiles { get; set; } = new();
            public string SummaryReportPath { get; set; } = "";
        }

        /// <summary>
        /// Analysis of PM4 file structure for extraction planning.
        /// </summary>
        public class PM4StructureAnalysis
        {
            // Chunk availability
            public bool HasMSLK { get; set; }
            public bool HasMSPV { get; set; }
            public bool HasMSVT { get; set; }
            public bool HasMSUR { get; set; }
            public bool HasMDSF { get; set; }
            public bool HasMDOS { get; set; }

            // Chunk counts
            public int MSLKCount { get; set; }
            public int MSPVCount { get; set; }
            public int MSVTCount { get; set; }
            public int MSURCount { get; set; }
            public int MDSFCount { get; set; }
            public int MDOSCount { get; set; }

            // Extraction capabilities
            public bool SupportsMdsfMdosExtraction { get; set; }
            public bool SupportsMslkRootNodeExtraction { get; set; }

            // Strategy recommendation
            public string RecommendedStrategy { get; set; } = "";
            public string StrategyReason { get; set; } = "";

            // Hierarchy analysis
            public MslkRootNodeDetector.HierarchyStatistics? HierarchyStatistics { get; set; }
            public List<MslkRootNodeDetector.RootNodeInfo> RootNodes { get; set; } = new();
        }
    }
} 