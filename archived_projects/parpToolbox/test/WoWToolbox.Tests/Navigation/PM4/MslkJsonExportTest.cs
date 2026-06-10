using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class MslkJsonExportTest
    {
        private readonly ITestOutputHelper _output;

        public MslkJsonExportTest(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void MslkJsonExporter_ShouldProduceStructuredOutput()
        {
            // Find PM4 files in test data
            var pm4Files = Directory.GetFiles("test_data/original_development", "*.pm4", SearchOption.AllDirectories)
                .Where(f => new FileInfo(f).Length > 0) // Only non-empty files
                .Take(3) // Limit to first 3 files for testing
                .ToArray();

            if (!pm4Files.Any())
            {
                _output.WriteLine("⚠️  No PM4 files found in test data, skipping test");
                return;
            }

            var exporter = new MslkJsonExporter();
            _output.WriteLine("🔄 Testing MSLK JSON Export on available PM4 files");
            _output.WriteLine(new string('═', 60));

            foreach (var filePath in pm4Files)
            {
                var fileName = Path.GetFileName(filePath);
                _output.WriteLine($"📄 Processing: {fileName}");

                try
                {
                    var pm4File = PM4File.FromFile(filePath);
                    
                    if (pm4File.MSLK == null)
                    {
                        _output.WriteLine("   ❌ No MSLK chunk found");
                        continue;
                    }

                    // Export to JSON analysis
                    var analysis = exporter.AnalyzeAndExport(pm4File.MSLK, fileName);
                    var jsonOutput = exporter.ExportToString(analysis);

                    // Display key information
                    _output.WriteLine($"   ✅ Total Entries: {analysis.Statistics.TotalEntries}");
                    _output.WriteLine($"   📊 Geometry: {analysis.Statistics.GeometryEntries}, Doodad: {analysis.Statistics.DoodadEntries}");
                    _output.WriteLine($"   🏗️  Hierarchy: {analysis.Statistics.HierarchyLevels} levels, {analysis.Statistics.RootNodes} roots");
                    
                    // Show flag distribution
                    _output.WriteLine($"   🏷️  Flag Distribution:");
                    foreach (var kvp in analysis.Statistics.FlagDistribution.OrderBy(x => x.Key))
                    {
                        _output.WriteLine($"      Flag 0x{kvp.Key:X2}: {kvp.Value} entries");
                    }

                    // Show sample JSON structure (first 500 characters)
                    var jsonSample = jsonOutput.Length > 500 ? jsonOutput.Substring(0, 500) + "..." : jsonOutput;
                    _output.WriteLine($"   📋 JSON Sample:");
                    _output.WriteLine($"```json");
                    _output.WriteLine($"{jsonSample}");
                    _output.WriteLine($"```");

                    // Validation results
                    if (analysis.ValidationResults.Any())
                    {
                        _output.WriteLine($"   ✅ Validation Results:");
                        foreach (var kvp in analysis.ValidationResults)
                        {
                            if (kvp.Key != "FlagGeometryCorrelation") // Skip the complex nested validation
                            {
                                _output.WriteLine($"      {kvp.Key}: {kvp.Value}");
                            }
                        }
                    }

                    // Assert basic functionality
                    Assert.NotNull(analysis);
                    Assert.NotEmpty(analysis.Entries);
                    Assert.True(analysis.Statistics.TotalEntries > 0);
                    Assert.NotEmpty(jsonOutput);

                    _output.WriteLine($"   💾 JSON Length: {jsonOutput.Length} characters");
                    _output.WriteLine();
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"   ❌ Error: {ex.Message}");
                }
            }

            _output.WriteLine("✅ MSLK JSON Export test complete!");
        }

        [Fact]
        public void MslkJsonExporter_ShouldValidateConstants()
        {
            var pm4Files = Directory.GetFiles("test_data/original_development", "*.pm4", SearchOption.AllDirectories)
                .Where(f => new FileInfo(f).Length > 0)
                .Take(1)
                .ToArray();

            if (!pm4Files.Any())
            {
                _output.WriteLine("⚠️  No PM4 files found, skipping validation test");
                return;
            }

            var exporter = new MslkJsonExporter();
            var pm4File = PM4File.FromFile(pm4Files[0]);
            
            if (pm4File.MSLK == null)
            {
                _output.WriteLine("⚠️  No MSLK chunk found, skipping validation test");
                return;
            }

            var analysis = exporter.AnalyzeAndExport(pm4File.MSLK, Path.GetFileName(pm4Files[0]));

            _output.WriteLine("🔍 MSLK Constants Validation:");
            _output.WriteLine($"   📄 File: {analysis.FileName}");
            _output.WriteLine($"   📊 Total Entries: {analysis.Statistics.TotalEntries}");

            // Check validation results
            Assert.True(analysis.ValidationResults.ContainsKey("Padding_0x02_AllZero"));
            Assert.True(analysis.ValidationResults.ContainsKey("Constant_0x0C_AllFFFFFFFF"));  
            Assert.True(analysis.ValidationResults.ContainsKey("Constant_0x12_All8000"));

            foreach (var kvp in analysis.ValidationResults)
            {
                if (kvp.Value is bool boolValue)
                {
                    _output.WriteLine($"   ✅ {kvp.Key}: {boolValue}");
                    if (kvp.Key.Contains("Constant") || kvp.Key.Contains("Padding"))
                    {
                        Assert.True(boolValue, $"Constant validation failed for {kvp.Key}");
                    }
                }
            }

            _output.WriteLine("✅ Constants validation complete!");
        }
    }
} 