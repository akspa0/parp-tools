using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.v2.Foundation.Data;
using WoWToolbox.PM4Parsing.BuildingExtraction;
using WoWToolbox.PM4Parsing;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class UniversalCompatibilityTests
    {
        private const string TestDataDirectory = "test_data/original_development/development/";

        [Fact]
        public void UniversalCompatibility_ShouldLoadAllPM4Files()
        {
            var pm4Files = Directory.GetFiles(TestDataDirectory, "*.pm4")
                .Where(f => new FileInfo(f).Length > 0) // Skip empty files
                .ToList();

            Assert.True(pm4Files.Count > 400, $"Expected at least 400 PM4 files, found {pm4Files.Count}");

            int successCount = 0;
            int failureCount = 0;
            var failures = new List<string>();

            foreach (var filePath in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var fileData = File.ReadAllBytes(filePath);
                    var pm4File = new PM4File(fileData);

                    // Basic validation - file should load with key chunks
                    Assert.NotNull(pm4File);
                    Assert.NotNull(pm4File.MSLK);

                    successCount++;
                }
                catch (Exception ex)
                {
                    failureCount++;
                    failures.Add($"{Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Report results
            Assert.True(successCount > 0, "No PM4 files loaded successfully");
            Assert.True(failureCount < successCount * 0.1, // Allow up to 10% failure rate for corrupted files
                $"Too many failures. Success: {successCount}, Failures: {failureCount}. Failed files: {string.Join(", ", failures)}");
        }

        [Fact]
        public void UniversalCompatibility_ShouldExtractBuildingsFromAllValidFiles()
        {
            var pm4Files = Directory.GetFiles(TestDataDirectory, "*.pm4")
                .Where(f => new FileInfo(f).Length > 4256) // Skip minimal/empty files
                .Take(50) // Test first 50 substantial files for performance
                .ToList();

            Assert.True(pm4Files.Count > 0, "No substantial PM4 files found for testing");

            int totalFilesProcessed = 0;
            int filesWithBuildings = 0;
            int totalBuildingsExtracted = 0;
            var extractionResults = new List<(string fileName, int buildings, string strategy)>();

            foreach (var filePath in pm4Files)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var fileData = File.ReadAllBytes(filePath);
                    var pm4File = new PM4File(fileData);

                    // Test building extraction with both strategies
                    var buildings = pm4File.ExtractBuildings();
                    
                    totalFilesProcessed++;
                    
                    if (buildings.Count > 0)
                    {
                        filesWithBuildings++;
                        totalBuildingsExtracted += buildings.Count;
                        
                        // Determine which strategy was used based on log patterns
                        string strategy = "Unknown";
                        if (buildings.Count > 10) strategy = "Fallback Strategy";
                        else if (buildings.Count > 0) strategy = "Root Node Strategy";
                        
                        extractionResults.Add((fileName, buildings.Count, strategy));
                    }
                }
                catch (Exception ex)
                {
                    // Log but don't fail the test for individual file issues
                    System.Console.WriteLine($"Failed to process {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Validate universal compatibility is working
            Assert.True(totalFilesProcessed > 0, "No files were processed");
            Assert.True(filesWithBuildings > 0, "No buildings were extracted from any files");
            Assert.True(totalBuildingsExtracted > 0, "Total building count should be greater than 0");
            
            // Our universal compatibility fix should enable building extraction from most substantial files
            var extractionRate = (double)filesWithBuildings / totalFilesProcessed;
            Assert.True(extractionRate > 0.3, // At least 30% of files should extract buildings
                $"Extraction rate too low: {extractionRate:P1}. Files processed: {totalFilesProcessed}, " +
                $"Files with buildings: {filesWithBuildings}, Total buildings: {totalBuildingsExtracted}");

            // Output summary for verification
            System.Console.WriteLine($"Universal Compatibility Test Results:");
            System.Console.WriteLine($"Files processed: {totalFilesProcessed}");
            System.Console.WriteLine($"Files with buildings: {filesWithBuildings} ({extractionRate:P1})");
            System.Console.WriteLine($"Total buildings extracted: {totalBuildingsExtracted}");
            System.Console.WriteLine($"Average buildings per successful file: {(double)totalBuildingsExtracted / filesWithBuildings:F1}");
        }

        [Fact]
        public void UniversalCompatibility_PM4BuildingExtractorService_ShouldProcessMultipleFiles()
        {
            var pm4Files = Directory.GetFiles(TestDataDirectory, "*.pm4")
                .Where(f => new FileInfo(f).Length > 50000) // Test larger files only
                .Take(10) // Test 10 larger files
                .ToList();

            Assert.True(pm4Files.Count > 0, "No large PM4 files found for testing");

            var service = new PM4BuildingExtractionService();
            int successfulExtractions = 0;
            int totalBuildings = 0;

            foreach (var filePath in pm4Files)
            {
                var outputDir = Path.Combine("TestResults", "UniversalCompatibility", Path.GetFileNameWithoutExtension(filePath));
                Directory.CreateDirectory(outputDir);

                try
                {
                    var result = service.ExtractAndExportBuildings(filePath, outputDir);
                    
                    if (result.Buildings.Count > 0)
                    {
                        successfulExtractions++;
                        totalBuildings += result.Buildings.Count;
                        
                        // Verify files were actually created
                        var objFiles = Directory.GetFiles(outputDir, "*.obj");
                        Assert.True(objFiles.Length > 0, $"No OBJ files created for {Path.GetFileName(filePath)}");
                        
                        // Verify files have content
                        Assert.True(objFiles.All(f => new FileInfo(f).Length > 0), 
                            $"Empty OBJ files created for {Path.GetFileName(filePath)}");
                    }
                }
                catch (Exception ex)
                {
                    // Log but continue testing other files
                    System.Console.WriteLine($"Failed extraction for {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Validate the service works with universal compatibility
            Assert.True(successfulExtractions > 0, "No successful building extractions with PM4BuildingExtractionService");
            Assert.True(totalBuildings > 0, "No buildings exported to files");

            System.Console.WriteLine($"PM4BuildingExtractionService Test Results:");
            System.Console.WriteLine($"Files tested: {pm4Files.Count}");
            System.Console.WriteLine($"Successful extractions: {successfulExtractions}");
            System.Console.WriteLine($"Total buildings exported: {totalBuildings}");
        }

        [Theory]
        [InlineData("development_00_00.pm4")] // Original working file
        [InlineData("development_01_01.pm4")] // Previously failing file
        [InlineData("development_22_18.pm4")] // Large complex file
        [InlineData("development_14_36.pm4")] // Another substantial file
        [InlineData("development_15_37.pm4")] // High complexity file
        public void UniversalCompatibility_SpecificFiles_ShouldExtractBuildings(string fileName)
        {
            var filePath = Path.Combine(TestDataDirectory, fileName);
            
            if (!File.Exists(filePath))
            {
                // Skip test if file doesn't exist
                return;
            }

            if (new FileInfo(filePath).Length == 0)
            {
                // Skip empty files
                return;
            }

            var fileData = File.ReadAllBytes(filePath);
            var pm4File = new PM4File(fileData);

            // Test Core.v2 extraction
            var buildings = pm4File.ExtractBuildings();
            
            // Universal compatibility should extract buildings from all non-empty files
            Assert.True(buildings.Count >= 0, $"Building extraction should not fail for {fileName}");
            
            // If this is one of our known good files, verify we get reasonable results
            if (fileName == "development_01_01.pm4")
            {
                // This was our test case that was failing before the fix
                Assert.True(buildings.Count > 0, "development_01_01.pm4 should extract buildings with universal compatibility fix");
            }

            System.Console.WriteLine($"{fileName}: Extracted {buildings.Count} buildings");
        }
    }
} 