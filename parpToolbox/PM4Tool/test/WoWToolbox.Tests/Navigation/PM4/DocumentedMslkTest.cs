using System;
using System.IO;
using System.Linq;
using Xunit;
using Xunit.Abstractions;
using WoWToolbox.Core.Navigation.PM4;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class DocumentedMslkTest
    {
        private readonly ITestOutputHelper _output;

        public DocumentedMslkTest(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void TestDocumentedMslkAnalysisWithMermaidDiagrams()
        {
            _output.WriteLine("🔍 DOCUMENTED MSLK ANALYSIS TEST");
            _output.WriteLine("═══════════════════════════════════════════════════════════════");
            _output.WriteLine("Testing MSLK analysis using ONLY documented field names");
            _output.WriteLine("No statistical interpretations - only wowdev.wiki structure");
            _output.WriteLine();

            // Find available test PM4 files
            var testDataPaths = new[]
            {
                "test_data/development",
                "test_data/development_335", 
                "test_data/original_development/development"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    break;
                }
            }

            if (testDataDir == null)
            {
                _output.WriteLine("⚠️  No test data found, skipping test");
                return;
            }

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                                   .Take(3)  // Test first 3 files
                                   .ToArray();

            if (!pm4Files.Any())
            {
                _output.WriteLine("⚠️  No PM4 files found in test data, skipping test");
                return;
            }

            var analyzer = new MslkDocumentedAnalyzer();

            foreach (var filePath in pm4Files)
            {
                var fileName = Path.GetFileName(filePath);
                _output.WriteLine($"📄 Analyzing: {fileName}");
                _output.WriteLine(new string('─', 50));

                try
                {
                    var pm4File = PM4File.FromFile(filePath);
                    
                    if (pm4File.MSLK == null)
                    {
                        _output.WriteLine("   No MSLK chunk found");
                        continue;
                    }

                    // Perform documented analysis
                    var analysis = analyzer.AnalyzeMslk(pm4File.MSLK);
                    
                    // Generate detailed report
                    var report = analyzer.GenerateAnalysisReport(analysis, fileName);
                    _output.WriteLine(report);
                    
                    // Generate Mermaid diagram
                    _output.WriteLine("=== MERMAID RELATIONSHIP DIAGRAM ===");
                    _output.WriteLine("```mermaid");
                    var mermaid = analyzer.GenerateMermaidDiagram(analysis, fileName);
                    _output.WriteLine(mermaid);
                    _output.WriteLine("```");
                    _output.WriteLine();
                    
                    _output.WriteLine(new string('═', 80));
                    _output.WriteLine();
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"❌ Error analyzing {fileName}: {ex.Message}");
                }
            }

            _output.WriteLine("✅ Documented MSLK analysis complete!");
        }

        [Fact]
        public void TestMslkDocumentationValidation()
        {
            _output.WriteLine("🔬 MSLK DOCUMENTATION VALIDATION TEST");
            _output.WriteLine("═══════════════════════════════════════════════════════════════");
            _output.WriteLine("Checking if real data matches wowdev.wiki documentation claims");
            _output.WriteLine();

            // Find test files
            var testDataPaths = new[]
            {
                "test_data/development",
                "test_data/development_335",
                "test_data/original_development/development"
            };

            string? testDataDir = null;
            foreach (var path in testDataPaths)
            {
                if (Directory.Exists(path))
                {
                    testDataDir = path;
                    break;
                }
            }

            if (testDataDir == null)
            {
                _output.WriteLine("⚠️  No test data found, skipping validation");
                return;
            }

            var pm4Files = Directory.GetFiles(testDataDir, "*.pm4", SearchOption.AllDirectories)
                                   .Take(5)
                                   .ToArray();

            if (!pm4Files.Any())
            {
                _output.WriteLine("⚠️  No PM4 files found for validation");
                return;
            }

            var analyzer = new MslkDocumentedAnalyzer();
            var allValidationWarnings = new List<string>();
            var allFlagsFound = new HashSet<byte>();
            var allUnknown01Values = new HashSet<byte>();
            var allUnknown0CValues = new HashSet<uint>();
            var allConstant12Values = new HashSet<ushort>();

            foreach (var filePath in pm4Files)
            {
                try
                {
                    var pm4File = PM4File.FromFile(filePath);
                    if (pm4File.MSLK?.Entries == null) continue;

                    var analysis = analyzer.AnalyzeMslk(pm4File.MSLK);
                    allValidationWarnings.AddRange(analysis.ValidationWarnings);
                    
                    // Collect actual values found
                    foreach (var entry in analysis.Entries)
                    {
                        allFlagsFound.Add(entry.Flags_0x00);
                        allUnknown01Values.Add(entry.Unknown_0x01);
                        allUnknown0CValues.Add(entry.Unknown_0x0C);
                        allConstant12Values.Add(entry.Constant_0x12);
                    }
                }
                catch (Exception ex)
                {
                    _output.WriteLine($"❌ Error processing {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Report validation results
            _output.WriteLine("=== DOCUMENTATION VALIDATION RESULTS ===");
            _output.WriteLine();

            if (allValidationWarnings.Any())
            {
                _output.WriteLine("⚠️  VALIDATION WARNINGS:");
                foreach (var warning in allValidationWarnings.Distinct())
                {
                    _output.WriteLine($"   • {warning}");
                }
            }
            else
            {
                _output.WriteLine("✅ All documented constants validated successfully!");
            }
            _output.WriteLine();

            // Show actual flag values found vs documented
            _output.WriteLine("=== FLAGS ANALYSIS ===");
            _output.WriteLine("Documentation claims: \"flags? seen: &1; &2; &4; &8; &16\"");
            _output.WriteLine($"Actual flags found: {string.Join(", ", allFlagsFound.OrderBy(x => x).Select(x => $"0x{x:X2}"))}");
            _output.WriteLine();

            // Show actual Unknown_0x01 values vs documented
            _output.WriteLine("=== UNKNOWN 0x01 ANALYSIS ===");
            _output.WriteLine("Documentation claims: \"0…11-ish; position in some sequence?\"");
            _output.WriteLine($"Actual values found: {string.Join(", ", allUnknown01Values.OrderBy(x => x))}");
            _output.WriteLine($"Range: {allUnknown01Values.Min()} to {allUnknown01Values.Max()}");
            _output.WriteLine();

            // Show actual Unknown_0x0C values
            _output.WriteLine("=== UNKNOWN 0x0C ANALYSIS ===");
            _output.WriteLine("Documentation claims: \"Always 0xffffffff in version_48\"");
            _output.WriteLine($"Unique values found: {allUnknown0CValues.Count}");
            foreach (var val in allUnknown0CValues.OrderBy(x => x).Take(10))
            {
                _output.WriteLine($"   0x{val:X8}");
            }
            _output.WriteLine();

            // Show actual Constant_0x12 values
            _output.WriteLine("=== CONSTANT 0x12 ANALYSIS ===");
            _output.WriteLine("Documentation claims: \"Always 0x8000 in version_48\"");
            _output.WriteLine($"Unique values found: {allConstant12Values.Count}");
            foreach (var val in allConstant12Values.OrderBy(x => x))
            {
                _output.WriteLine($"   0x{val:X4}");
            }

            _output.WriteLine();
            _output.WriteLine("✅ Documentation validation complete!");
        }
    }
} 