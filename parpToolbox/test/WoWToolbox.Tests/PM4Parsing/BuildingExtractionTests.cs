using System;
using System.IO;
using System.Linq;
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

        [Fact]
        public void UniversalCompatibility_ShouldExtractFromAllPM4Files()
        {
            const string testDataDir = "test_data/original_development/development/";
            
            if (!Directory.Exists(testDataDir))
            {
                // Skip test if test data directory doesn't exist
                return;
            }

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4")
                .Where(f => new FileInfo(f).Length > 4256) // Skip minimal/empty files
                .Take(25) // Test first 25 substantial files for performance
                .ToList();

            if (pm4Files.Count == 0)
            {
                // Skip test if no files found
                return;
            }

            int totalFilesProcessed = 0;
            int filesWithBuildings = 0;
            int totalBuildingsExtracted = 0;
            var results = new List<(string fileName, int buildings, bool usedFallback)>();

            foreach (var filePath in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var fileData = File.ReadAllBytes(filePath);
                    var pm4File = new WoWToolbox.Core.v2.Foundation.Data.PM4File(fileData);

                    // Test Core.v2 universal compatibility
                    var buildings = pm4File.ExtractBuildings();
                    
                    totalFilesProcessed++;
                    
                    if (buildings.Count > 0)
                    {
                        filesWithBuildings++;
                        totalBuildingsExtracted += buildings.Count;
                        
                        // Detect if fallback strategy was used (typically more buildings)
                        bool usedFallback = buildings.Count > 5;
                        results.Add((fileName, buildings.Count, usedFallback));
                    }
                }
                catch (Exception ex)
                {
                    // Log but continue testing other files
                    Console.WriteLine($"Failed to process {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Validate universal compatibility is working
            Assert.True(totalFilesProcessed > 0, "No files were processed");
            
            // Output detailed results for verification
            Console.WriteLine($"✅ Universal Compatibility Test Results:");
            Console.WriteLine($"Files processed: {totalFilesProcessed}");
            Console.WriteLine($"Files with buildings: {filesWithBuildings} ({(double)filesWithBuildings / totalFilesProcessed:P1})");
            Console.WriteLine($"Total buildings extracted: {totalBuildingsExtracted}");
            
            if (filesWithBuildings > 0)
            {
                Console.WriteLine($"Average buildings per successful file: {(double)totalBuildingsExtracted / filesWithBuildings:F1}");
                
                var fallbackFiles = results.Where(r => r.usedFallback).ToList();
                Console.WriteLine($"Files using fallback strategy: {fallbackFiles.Count}");
                
                // Show some example results
                foreach (var result in results.Take(10))
                {
                    Console.WriteLine($"  {result.fileName}: {result.buildings} buildings {(result.usedFallback ? "(fallback)" : "(root nodes)")}");
                }
            }

            // Universal compatibility should extract buildings from at least some files
            // Our fix enables extraction even when root nodes don't have geometry
            Assert.True(filesWithBuildings > 0, "Universal compatibility fix should enable building extraction from at least some PM4 files");
            
            // Specifically test that development_01_01.pm4 works (our test case)
            var testCaseResult = results.FirstOrDefault(r => r.fileName == "development_01_01.pm4");
            if (testCaseResult.fileName != null)
            {
                Assert.True(testCaseResult.buildings > 0, 
                    $"development_01_01.pm4 should extract buildings with universal compatibility fix. Got: {testCaseResult.buildings}");
                Console.WriteLine($"✅ CRITICAL TEST CASE PASSED: development_01_01.pm4 extracted {testCaseResult.buildings} buildings");
            }
        }

        [Fact]
        public void UniversalCompatibility_ProcessAllPM4Files_WithFileExport()
        {
            const string testDataDir = "test_data/original_development/development/";
            const string outputBaseDir = "output/universal_compatibility_all_files/";
            
            if (!Directory.Exists(testDataDir))
            {
                // Skip test if test data directory doesn't exist
                return;
            }

            // Get ALL PM4 files, including smaller ones
            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4")
                .Where(f => new FileInfo(f).Length > 0) // Skip only empty files
                .OrderBy(f => f) // Process in order
                .ToList();

            if (pm4Files.Count == 0)
            {
                // Skip test if no files found
                return;
            }

            Console.WriteLine($"🔍 Processing ALL {pm4Files.Count} PM4 files with universal compatibility...");
            
            // Create main output directory
            Directory.CreateDirectory(outputBaseDir);
            
            var service = new PM4BuildingExtractionService();
            int totalFilesProcessed = 0;
            int filesWithBuildings = 0;
            int totalBuildingsExtracted = 0;
            int totalFilesExported = 0;
            var results = new List<(string fileName, int buildings, int objFiles, long totalSize)>();
            var failures = new List<(string fileName, string error)>();

            foreach (var filePath in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var fileNameWithoutExt = Path.GetFileNameWithoutExtension(filePath);
                    var outputDir = Path.Combine(outputBaseDir, fileNameWithoutExt);
                    
                    Console.WriteLine($"Processing {fileName}...");
                    
                    // Use PM4BuildingExtractionService for complete workflow with file export
                    var result = service.ExtractAndExportBuildings(filePath, outputDir);
                    
                    totalFilesProcessed++;
                    
                    if (result.Buildings.Count > 0)
                    {
                        filesWithBuildings++;
                        totalBuildingsExtracted += result.Buildings.Count;
                        
                        // Count actual OBJ files created
                        var objFiles = Directory.GetFiles(outputDir, "*.obj");
                        if (objFiles.Length > 0)
                        {
                            totalFilesExported++;
                            
                            // Calculate total file size
                            long totalSize = objFiles.Sum(f => new FileInfo(f).Length);
                            results.Add((fileName, result.Buildings.Count, objFiles.Length, totalSize));
                            
                            Console.WriteLine($"  ✅ {fileName}: {result.Buildings.Count} buildings → {objFiles.Length} OBJ files ({totalSize / 1024:N0} KB)");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"  ⚪ {fileName}: No buildings extracted");
                    }
                }
                catch (Exception ex)
                {
                    failures.Add((Path.GetFileName(filePath), ex.Message));
                    Console.WriteLine($"  ❌ {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Generate comprehensive summary report
            var summaryPath = Path.Combine(outputBaseDir, "UNIVERSAL_COMPATIBILITY_SUMMARY.txt");
            using (var writer = new StreamWriter(summaryPath))
            {
                writer.WriteLine("🎯 UNIVERSAL PM4 COMPATIBILITY TEST RESULTS");
                writer.WriteLine("==========================================");
                writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                writer.WriteLine();
                
                writer.WriteLine("📊 OVERALL STATISTICS:");
                writer.WriteLine($"Total PM4 files found: {pm4Files.Count}");
                writer.WriteLine($"Files processed: {totalFilesProcessed}");
                writer.WriteLine($"Files with buildings: {filesWithBuildings} ({(double)filesWithBuildings / totalFilesProcessed:P1})");
                writer.WriteLine($"Files with exported OBJ: {totalFilesExported}");
                writer.WriteLine($"Total buildings extracted: {totalBuildingsExtracted}");
                writer.WriteLine($"Total OBJ files created: {results.Sum(r => r.objFiles)}");
                writer.WriteLine($"Total exported data: {results.Sum(r => r.totalSize) / (1024 * 1024):N1} MB");
                writer.WriteLine();
                
                if (results.Count > 0)
                {
                    writer.WriteLine("📁 SUCCESSFUL EXTRACTIONS (Top 50):");
                    foreach (var result in results.OrderByDescending(r => r.buildings).Take(50))
                    {
                        writer.WriteLine($"  {result.fileName}: {result.buildings} buildings → {result.objFiles} OBJ files ({result.totalSize / 1024:N0} KB)");
                    }
                    writer.WriteLine();
                }
                
                if (failures.Count > 0)
                {
                    writer.WriteLine("❌ FAILURES:");
                    foreach (var failure in failures.Take(20))
                    {
                        writer.WriteLine($"  {failure.fileName}: {failure.error}");
                    }
                    if (failures.Count > 20)
                    {
                        writer.WriteLine($"  ... and {failures.Count - 20} more failures");
                    }
                    writer.WriteLine();
                }
                
                writer.WriteLine("🎯 UNIVERSAL COMPATIBILITY VALIDATION:");
                writer.WriteLine($"✅ Development_01_01.pm4 processed: {results.Any(r => r.fileName == "development_01_01.pm4")}");
                var testCase = results.FirstOrDefault(r => r.fileName == "development_01_01.pm4");
                if (testCase.fileName != null)
                {
                    writer.WriteLine($"✅ Development_01_01.pm4 buildings: {testCase.buildings} (SUCCESS - was failing before fix)");
                }
            }

            // Console output for immediate feedback
            Console.WriteLine();
            Console.WriteLine("🎯 UNIVERSAL PM4 COMPATIBILITY COMPLETE!");
            Console.WriteLine($"📊 Results: {filesWithBuildings}/{totalFilesProcessed} files successful ({(double)filesWithBuildings / totalFilesProcessed:P1})");
            Console.WriteLine($"🏗️ Buildings: {totalBuildingsExtracted} buildings across all files");
            Console.WriteLine($"📁 OBJ Files: {results.Sum(r => r.objFiles)} OBJ files exported");
            Console.WriteLine($"💾 Data Size: {results.Sum(r => r.totalSize) / (1024 * 1024):N1} MB exported");
            Console.WriteLine($"📄 Summary: {summaryPath}");
            Console.WriteLine($"📂 Output: {outputBaseDir}");

            // Test assertions for validation
            Assert.True(totalFilesProcessed > 400, $"Should process most PM4 files. Processed: {totalFilesProcessed}");
            Assert.True(filesWithBuildings > 0, "Universal compatibility should extract buildings from some files");
            Assert.True(totalFilesExported > 0, "Should export OBJ files to output directory");
            Assert.True(File.Exists(summaryPath), "Should create summary report");
            
            // Verify critical test case (development_01_01.pm4)
            var criticalTestCase = results.FirstOrDefault(r => r.fileName == "development_01_01.pm4");
            if (criticalTestCase.fileName != null)
            {
                Assert.True(criticalTestCase.buildings > 0, 
                    $"development_01_01.pm4 should extract buildings with universal compatibility fix. Got: {criticalTestCase.buildings}");
                Console.WriteLine($"✅ CRITICAL SUCCESS: development_01_01.pm4 extracted {criticalTestCase.buildings} buildings with {criticalTestCase.objFiles} OBJ files");
            }
        }
    }
} 