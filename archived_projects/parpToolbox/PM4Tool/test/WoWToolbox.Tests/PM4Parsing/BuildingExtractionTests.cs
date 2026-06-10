using System.Numerics;
using WoWToolbox.Core;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.Navigation.PM4.Models;
using WoWToolbox.PM4Parsing;
using WoWToolbox.PM4Parsing.BuildingExtraction;
using WoWToolbox.PM4Parsing.NodeSystem;
using Xunit;

namespace WoWToolbox.Tests.PM4Parsing
{
    /// <summary>
    /// Tests for PM4 building extraction functionality.
    /// Focuses on the core building extraction algorithms and new clean architecture.
    /// </summary>
    public class BuildingExtractionTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static string OutputRoot => Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "TestResults", "BuildingExtraction");

        [Fact]
        public void PM4BuildingExtractor_ShouldExtractBuildingsFromValidFile()
        {
            // Arrange
            var inputFilePath = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            var outputDir = Path.Combine(OutputRoot, "FlexibleMethod");
            Directory.CreateDirectory(outputDir);

            if (!File.Exists(inputFilePath))
            {
                // Skip test if test file doesn't exist
                return;
            }

            var pm4File = PM4File.FromFile(inputFilePath);
            var sourceFileName = Path.GetFileNameWithoutExtension(inputFilePath);
            var extractor = new PM4BuildingExtractor();

            // Act
            var buildings = extractor.ExtractBuildings(pm4File, sourceFileName);

            // Assert
            Assert.NotNull(buildings);
            Assert.NotEmpty(buildings);

            foreach (var building in buildings)
            {
                Assert.NotNull(building.FileName);
                Assert.NotNull(building.Category);
                Assert.True(building.VertexCount > 0, "Building should have vertices");
                Assert.True(building.FaceCount > 0, "Building should have faces");
                Assert.Contains("ExtractionMethod", building.Metadata.Keys);
            }

            // Export buildings for verification
            for (int i = 0; i < buildings.Count; i++)
            {
                var building = buildings[i];
                var objPath = Path.Combine(outputDir, $"{sourceFileName}_Building_{i + 1:D2}.obj");
                CompleteWMOModelUtilities.ExportToOBJ(building, objPath);
                Assert.True(File.Exists(objPath), "OBJ file should be created");
                Assert.True(File.Exists(Path.ChangeExtension(objPath, ".mtl")), "MTL file should be created");
            }
        }

        [Fact]
        public void MslkRootNodeDetector_ShouldDetectRootNodes()
        {
            // Arrange
            var inputFilePath = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            
            if (!File.Exists(inputFilePath))
            {
                return; // Skip if test file doesn't exist
            }

            var pm4File = PM4File.FromFile(inputFilePath);
            var detector = new MslkRootNodeDetector();

            // Act
            var rootNodes = detector.DetectRootNodes(pm4File);
            var hierarchyStats = detector.GetHierarchyStatistics(pm4File);

            // Assert
            Assert.NotNull(rootNodes);
            Assert.NotNull(hierarchyStats);
            Assert.True(hierarchyStats.TotalNodes > 0, "Should have MSLK nodes");
            Assert.True(hierarchyStats.RootNodeCount > 0, "Should detect root nodes");

            // Verify root node properties
            foreach (var rootNode in rootNodes)
            {
                Assert.True(rootNode.NodeIndex >= 0);
                Assert.Equal(rootNode.NodeIndex, (int)rootNode.Entry.Unknown_0x04); // Self-referencing
                Assert.NotNull(rootNode.ChildNodes);
            }

            // Verify hierarchy statistics make sense
            Assert.Equal(hierarchyStats.TotalNodes, hierarchyStats.RootNodeCount + hierarchyStats.TotalChildNodes + hierarchyStats.OrphanedNodes);
        }

        [Fact]
        public void PM4BuildingExtractionService_ShouldProvideCompleteWorkflow()
        {
            // Arrange
            var inputFilePath = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            var outputDir = Path.Combine(OutputRoot, "CompleteWorkflow");

            if (!File.Exists(inputFilePath))
            {
                return; // Skip if test file doesn't exist
            }

            var service = new PM4BuildingExtractionService();

            // Act
            var result = service.ExtractAndExportBuildings(inputFilePath, outputDir);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(inputFilePath, result.SourceFile);
            Assert.NotNull(result.AnalysisResult);
            Assert.NotNull(result.Buildings);
            Assert.NotEmpty(result.ExportedFiles);
            Assert.True(File.Exists(result.SummaryReportPath), "Summary report should be created");

            // Verify analysis result
            var analysis = result.AnalysisResult;
            Assert.True(analysis.HasMSLK, "Should detect MSLK chunk");
            Assert.True(analysis.MSLKCount > 0, "Should count MSLK entries");
            Assert.NotEmpty(analysis.RecommendedStrategy);
            Assert.NotEmpty(analysis.StrategyReason);

            // Verify buildings were extracted and exported
            Assert.True(result.Buildings.Count > 0, "Should extract buildings");
            Assert.Equal(result.Buildings.Count, result.ExportedFiles.Count);

            foreach (var exportedFile in result.ExportedFiles)
            {
                Assert.True(File.Exists(exportedFile), $"Exported file should exist: {exportedFile}");
            }
        }

        [Fact]
        public void CompleteWMOModelUtilities_ShouldGenerateNormalsCorrectly()
        {
            // Arrange
            var model = new CompleteWMOModel
            {
                FileName = "TestModel",
                Category = "Test"
            };

            // Create a simple triangle
            model.Vertices.Add(new Vector3(0, 0, 0));
            model.Vertices.Add(new Vector3(1, 0, 0));
            model.Vertices.Add(new Vector3(0, 1, 0));
            model.TriangleIndices.AddRange(new[] { 0, 1, 2 });

            // Act
            CompleteWMOModelUtilities.GenerateNormals(model);

            // Assert
            Assert.Equal(3, model.Normals.Count);
            foreach (var normal in model.Normals)
            {
                Assert.True(Math.Abs(normal.Length() - 1.0f) < 0.001f, "Normal should be unit length");
            }
        }

        [Fact]
        public void CompleteWMOModelUtilities_ShouldCalculateBoundingBox()
        {
            // Arrange
            var model = new CompleteWMOModel();
            model.Vertices.Add(new Vector3(-1, -2, -3));
            model.Vertices.Add(new Vector3(4, 5, 6));
            model.Vertices.Add(new Vector3(0, 0, 0));

            // Act
            var bbox = CompleteWMOModelUtilities.CalculateBoundingBox(model);

            // Assert
            Assert.NotNull(bbox);
            Assert.Equal(new Vector3(-1, -2, -3), bbox.Value.Min);
            Assert.Equal(new Vector3(4, 5, 6), bbox.Value.Max);
            Assert.Equal(new Vector3(1.5f, 1.5f, 1.5f), bbox.Value.Center);
            Assert.Equal(new Vector3(5, 7, 9), bbox.Value.Size);
        }

        [Fact]
        public void CompleteWMOModelUtilities_ShouldExportOBJFormat()
        {
            // Arrange
            var outputDir = Path.Combine(OutputRoot, "OBJExport");
            Directory.CreateDirectory(outputDir);

            var model = new CompleteWMOModel
            {
                FileName = "TestExport",
                Category = "UnitTest",
                MaterialName = "TestMaterial"
            };

            // Add simple geometry
            model.Vertices.AddRange(new[]
            {
                new Vector3(0, 0, 0),
                new Vector3(1, 0, 0),
                new Vector3(0, 1, 0)
            });

            model.TriangleIndices.AddRange(new[] { 0, 1, 2 });
            CompleteWMOModelUtilities.GenerateNormals(model);

            var objPath = Path.Combine(outputDir, "test_export.obj");

            // Act
            CompleteWMOModelUtilities.ExportToOBJ(model, objPath);

            // Assert
            Assert.True(File.Exists(objPath), "OBJ file should be created");
            Assert.True(File.Exists(Path.ChangeExtension(objPath, ".mtl")), "MTL file should be created");

            var objContent = File.ReadAllText(objPath);
            Assert.Contains("v ", objContent); // Vertices
            Assert.Contains("vn ", objContent); // Normals
            Assert.Contains("f ", objContent); // Faces
            Assert.Contains("usemtl TestMaterial", objContent); // Material reference

            var mtlContent = File.ReadAllText(Path.ChangeExtension(objPath, ".mtl"));
            Assert.Contains("newmtl TestMaterial", mtlContent); // Material definition
        }

        [Theory]
        [InlineData("development_00_00.pm4")]
        [InlineData("development_01_01.pm4")]
        public void BuildingExtraction_ShouldHandleMultipleFiles(string fileName)
        {
            // Arrange
            var inputFilePath = Path.Combine(TestDataRoot, "original_development", "development", fileName);
            
            if (!File.Exists(inputFilePath))
            {
                return; // Skip if test file doesn't exist
            }

            var outputDir = Path.Combine(OutputRoot, "MultipleFiles", Path.GetFileNameWithoutExtension(fileName));
            Directory.CreateDirectory(outputDir);

            var service = new PM4BuildingExtractionService();

            // Act & Assert - Should not throw
            var result = service.ExtractAndExportBuildings(inputFilePath, outputDir);
            
            Assert.NotNull(result);
            Assert.True(File.Exists(result.SummaryReportPath));
        }
    }
} 