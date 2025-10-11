using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Xunit;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Tests;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class Pm4ChunkAnalysisTests
    {
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static string OutputRoot => OutputLocator.Central("ChunkAnalysis");

        [Fact]
        public void AnalyzeChunkRelationships_ForKeyFiles()
        {
            Console.WriteLine("=== PM4 Chunk Relationship Analysis ===");
            
            // Create output directory
            Directory.CreateDirectory(OutputRoot);
            
            var testFiles = new[]
            {
                Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4"),
                Path.Combine(TestDataRoot, "original_development", "development", "development_22_18.pm4")
            };

            var analyzer = new Pm4ChunkAnalyzer();
            var allResults = new List<Pm4ChunkAnalyzer.ChunkAnalysisResult>();

            foreach (var filePath in testFiles)
            {
                if (!File.Exists(filePath))
                {
                    Console.WriteLine($"⚠️  Skipping missing file: {Path.GetFileName(filePath)}");
                    continue;
                }

                Console.WriteLine($"\n📁 Analyzing: {Path.GetFileName(filePath)}");
                
                try
                {
                    var result = analyzer.AnalyzePm4File(filePath);
                    allResults.Add(result);

                    // Write individual report
                    var reportPath = Path.Combine(OutputRoot, $"{Path.GetFileNameWithoutExtension(filePath)}_analysis.md");
                    analyzer.WriteAnalysisReport(result, reportPath);
                    Console.WriteLine($"   📊 Report written: {Path.GetFileName(reportPath)}");

                    // Print key insights to console
                    Console.WriteLine("   🔍 Key Insights:");
                    foreach (var insight in result.Insights.Take(5))
                    {
                        Console.WriteLine($"      • {insight}");
                    }
                    if (result.Insights.Count > 5)
                    {
                        Console.WriteLine($"      ... and {result.Insights.Count - 5} more (see report)");
                    }

                    // Print concerning issues
                    var concerns = result.Insights.Where(i => i.Contains("❌") || i.Contains("⚠️")).ToList();
                    if (concerns.Any())
                    {
                        Console.WriteLine("   ⚠️  Concerns:");
                        foreach (var concern in concerns)
                        {
                            Console.WriteLine($"      • {concern}");
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   ❌ Error analyzing {Path.GetFileName(filePath)}: {ex.Message}");
                }
            }

            // Write comparative analysis
            if (allResults.Count > 1)
            {
                WriteComparativeAnalysis(allResults, Path.Combine(OutputRoot, "comparative_analysis.md"));
                Console.WriteLine($"\n📈 Comparative analysis written: comparative_analysis.md");
            }

            // Suggest next steps based on findings
            SuggestNextSteps(allResults);

            Console.WriteLine($"\n📂 All analysis files written to: {OutputRoot}");
            Assert.True(allResults.Count > 0, "Should have analyzed at least one file successfully");
        }

        private void WriteComparativeAnalysis(List<Pm4ChunkAnalyzer.ChunkAnalysisResult> results, string outputPath)
        {
            using var writer = new StreamWriter(outputPath);
            
            writer.WriteLine("# PM4 Comparative Chunk Analysis");
            writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine($"Files analyzed: {results.Count}");
            writer.WriteLine();

            // Chunk count comparison
            writer.WriteLine("## Chunk Count Comparison");
            writer.WriteLine("| File | MSVT | MSVI | MSPV | MSPI | MPRL | MPRR | MSLK | MSUR | MSCN |");
            writer.WriteLine("|------|------|------|------|------|------|------|------|------|------|");
            
            foreach (var result in results)
            {
                var c = result.Counts;
                writer.WriteLine($"| {result.FileName} | {c.MSVT_Vertices:N0} | {c.MSVI_Indices:N0} | {c.MSPV_Vertices:N0} | {c.MSPI_Indices:N0} | {c.MPRL_Entries:N0} | {c.MPRR_Sequences:N0} | {c.MSLK_Entries:N0} | {c.MSUR_Surfaces:N0} | {c.MSCN_Points:N0} |");
            }
            writer.WriteLine();

            // MPRR sequence analysis comparison
            writer.WriteLine("## MPRR Sequence Analysis Comparison");
            writer.WriteLine("| File | Total Sequences | Total Values | Max Value | Possible Index Target |");
            writer.WriteLine("|------|----------------|--------------|-----------|----------------------|");
            
            foreach (var result in results)
            {
                var mpr = result.MprAnalysis;
                writer.WriteLine($"| {result.FileName} | {mpr.TotalSequences:N0} | {mpr.TotalValues:N0} | {mpr.MaxValue} | {mpr.IndexTarget} |");
            }
            writer.WriteLine();

            // Connectivity analysis comparison
            writer.WriteLine("## Connectivity Analysis Comparison");
            writer.WriteLine("| File | MSUR Valid | MSUR Invalid | MSLK Valid | MSLK Invalid | MSCN/MSVT Overlap |");
            writer.WriteLine("|------|------------|--------------|------------|--------------|-------------------|");
            
            foreach (var result in results)
            {
                writer.WriteLine($"| {result.FileName} | {result.MsurAnalysis.ValidMsviRanges} | {result.MsurAnalysis.InvalidMsviRanges} | {result.MslkAnalysis.ValidMspiRanges} | {result.MslkAnalysis.InvalidMspiRanges} | {result.CoordAnalysis.MSVT_MSCN_Overlap:P1} |");
            }
            writer.WriteLine();

            // Pattern analysis
            writer.WriteLine("## Pattern Analysis");
            
            // MPRR/MPRL ratio analysis
            writer.WriteLine("### MPRR/MPRL Ratio Analysis");
            foreach (var result in results)
            {
                if (result.Counts.MPRL_Entries > 0)
                {
                    float ratio = (float)result.MprAnalysis.TotalValues / result.Counts.MPRL_Entries;
                    writer.WriteLine($"- **{result.FileName}**: MPRR/MPRL ratio = {ratio:F1}");
                    
                    if (ratio > 100)
                        writer.WriteLine($"  - ⚠️ Extremely high ratio suggests MPRR might contain mesh connectivity data");
                    else if (ratio > 10)
                        writer.WriteLine($"  - 🔍 High ratio suggests complex relationships worth investigating");
                    else
                        writer.WriteLine($"  - ✅ Normal ratio suggests simple point-to-point relationships");
                }
            }
            writer.WriteLine();

            // Common issues
            writer.WriteLine("### Common Issues Across Files");
            var allConcerns = results.SelectMany(r => r.Insights.Where(i => i.Contains("❌") || i.Contains("⚠️"))).ToList();
            var groupedConcerns = allConcerns.GroupBy(c => c.Split(':')[0]).ToList();
            
            foreach (var group in groupedConcerns.Where(g => g.Count() > 1))
            {
                writer.WriteLine($"- **{group.Key}**: Affects {group.Count()}/{results.Count} files");
            }
            writer.WriteLine();

            // Recommendations
            writer.WriteLine("## Recommendations");
            
            // Check for connectivity issues
            var hasConnectivityIssues = results.Any(r => !r.MsurAnalysis.HasValidConnectivity || !r.MslkAnalysis.HasValidGeometry);
            if (hasConnectivityIssues)
            {
                writer.WriteLine("1. **Investigate mesh connectivity issues**: Some files have invalid MSUR/MSLK ranges");
                writer.WriteLine("   - Review face generation logic in PM4FileTests.cs");
                writer.WriteLine("   - Validate MSVI/MSPI index bounds checking");
            }

            // Check for high MPRR ratios
            var hasHighMprrRatio = results.Any(r => r.Counts.MPRL_Entries > 0 && (float)r.MprAnalysis.TotalValues / r.Counts.MPRL_Entries > 50);
            if (hasHighMprrRatio)
            {
                writer.WriteLine("2. **Investigate MPRR sequences for mesh data**: High MPRR/MPRL ratios suggest additional connectivity information");
                writer.WriteLine("   - Analyze MPRR sequence patterns");
                writer.WriteLine("   - Check if MPRR contains face indices or triangle strips");
            }

            // Check coordinate alignment
            var hasAlignmentIssues = results.Any(r => r.CoordAnalysis.MSVT_MSCN_Overlap < 0.8f);
            if (hasAlignmentIssues)
            {
                writer.WriteLine("3. **Verify coordinate transformations**: Some files show poor MSCN/MSVT alignment");
                writer.WriteLine("   - Double-check Pm4CoordinateTransforms implementation");
                writer.WriteLine("   - Validate with additional test files");
            }
            else
            {
                writer.WriteLine("3. ✅ **Coordinate alignment looks good**: MSCN/MSVT overlap is excellent across files");
            }
        }

        private void SuggestNextSteps(List<Pm4ChunkAnalyzer.ChunkAnalysisResult> results)
        {
            Console.WriteLine("\n🎯 SUGGESTED NEXT STEPS:");

            // Check for mesh connectivity issues
            var connectivityIssues = results.Where(r => !r.MsurAnalysis.HasValidConnectivity).ToList();
            if (connectivityIssues.Any())
            {
                Console.WriteLine($"\n🔧 MESH CONNECTIVITY ISSUES ({connectivityIssues.Count} files):");
                Console.WriteLine("   → Review MSUR face generation logic in PM4FileTests.cs");
                Console.WriteLine("   → Check triangle fan vs triangle strip interpretation");
                Console.WriteLine("   → Validate MSVI index bounds checking");
            }

            // Check for MPRR investigation opportunities
            var highMprrRatio = results.Where(r => r.Counts.MPRL_Entries > 0 && 
                (float)r.MprAnalysis.TotalValues / r.Counts.MPRL_Entries > 50).ToList();
            if (highMprrRatio.Any())
            {
                Console.WriteLine($"\n📊 MPRR INVESTIGATION NEEDED ({highMprrRatio.Count} files):");
                Console.WriteLine("   → MPRR sequences might contain mesh connectivity data");
                Console.WriteLine("   → Analyze sequence patterns for face/triangle information");
                Console.WriteLine("   → Cross-reference MPRR values with MSVI indices");
            }

            // Check for MSLK geometry investigation
            var mslkIssues = results.Where(r => r.MslkAnalysis.InvalidMspiRanges > 0).ToList();
            if (mslkIssues.Any())
            {
                Console.WriteLine($"\n🏗️ MSLK GEOMETRY ISSUES ({mslkIssues.Count} files):");
                Console.WriteLine("   → Some MSLK entries have invalid MSPI ranges");
                Console.WriteLine("   → May indicate missing geometry data or different interpretation");
                Console.WriteLine("   → Consider alternative MSLK processing approaches");
            }

            // Coordinate alignment status
            var alignmentGood = results.All(r => r.CoordAnalysis.MSVT_MSCN_Overlap > 0.8f);
            if (alignmentGood)
            {
                Console.WriteLine("\n✅ COORDINATE ALIGNMENT: Excellent across all files");
            }
            else
            {
                Console.WriteLine("\n⚠️ COORDINATE ALIGNMENT: Some issues detected - review transforms");
            }

            Console.WriteLine("\n🔍 IMMEDIATE ACTIONS:");
            Console.WriteLine("   1. Focus on MSUR face generation - likely source of broken triangles");
            Console.WriteLine("   2. Investigate MPRR sequences for additional connectivity data");
            Console.WriteLine("   3. Create tool to visualize MPRR sequence patterns");
            Console.WriteLine("   4. Test alternative face generation algorithms");
        }

        [Fact]
        public void AnalyzeMprrSequencePatterns_ForMeshConnectivity()
        {
            Console.WriteLine("=== MPRR Sequence Pattern Analysis ===");
            
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"⚠️  Test file not found: {testFile}");
                return;
            }

            var pm4File = PM4File.FromFile(testFile);
            
            if (pm4File.MPRR?.Sequences == null || pm4File.MPRR.Sequences.Count == 0)
            {
                Console.WriteLine("❌ No MPRR sequences found");
                return;
            }

            Console.WriteLine($"📊 Analyzing {pm4File.MPRR.Sequences.Count} MPRR sequences...");

            // Analyze sequence patterns
            var allValues = pm4File.MPRR.Sequences.SelectMany(s => s).ToList();
            var uniqueValues = allValues.Distinct().OrderBy(v => v).ToList();
            var valueFreq = allValues.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());

            Console.WriteLine($"   • Total values: {allValues.Count:N0}");
            Console.WriteLine($"   • Unique values: {uniqueValues.Count:N0}");
            Console.WriteLine($"   • Value range: {uniqueValues.Min()} - {uniqueValues.Max()}");

            // Check if values could be indices into various arrays
            var msvtCount = pm4File.MSVT?.Vertices.Count ?? 0;
            var msviCount = pm4File.MSVI?.Indices.Count ?? 0;
            var mprlCount = pm4File.MPRL?.Entries.Count ?? 0;

            Console.WriteLine($"\n🎯 INDEX ANALYSIS:");
            Console.WriteLine($"   • MSVT vertices: {msvtCount:N0} (max MPRR: {uniqueValues.Max()})");
            Console.WriteLine($"   • MSVI indices: {msviCount:N0}");
            Console.WriteLine($"   • MPRL entries: {mprlCount:N0}");

            if (uniqueValues.Max() < msvtCount)
            {
                Console.WriteLine($"   ✅ MPRR values could index into MSVT vertices");
                AnalyzeMprrAsMeshIndices(pm4File, OutputRoot);
            }
            else if (uniqueValues.Max() < msviCount)
            {
                Console.WriteLine($"   ✅ MPRR values could index into MSVI indices");
            }
            else
            {
                Console.WriteLine($"   ❌ MPRR values too large for vertex/index arrays");
            }

            // Look for common sequence patterns
            Console.WriteLine($"\n🔍 SEQUENCE PATTERNS:");
            var sequenceLengths = pm4File.MPRR.Sequences.Select(s => s.Count).ToList();
            var lengthGroups = sequenceLengths.GroupBy(l => l).OrderBy(g => g.Key).ToList();
            
            foreach (var group in lengthGroups.Take(10))
            {
                Console.WriteLine($"   • Length {group.Key}: {group.Count()} sequences");
            }

            // Check for triangle-like patterns (sequences of 3)
            var triangleSequences = pm4File.MPRR.Sequences.Where(s => s.Count == 3).ToList();
            if (triangleSequences.Any())
            {
                Console.WriteLine($"\n🔺 POTENTIAL TRIANGLE SEQUENCES: {triangleSequences.Count}");
                Console.WriteLine($"   → These could represent triangle faces");
                
                // Export a sample as OBJ to test
                if (triangleSequences.Count > 0 && msvtCount > 0)
                {
                    var sampleObjPath = Path.Combine(OutputRoot, "mprr_triangles_sample.obj");
                    Directory.CreateDirectory(Path.GetDirectoryName(sampleObjPath) ?? OutputRoot);
                    ExportMprrTriangleSample(pm4File, triangleSequences, sampleObjPath);
                    Console.WriteLine($"   📄 Sample exported: {Path.GetFileName(sampleObjPath)}");
                }
            }
        }

        private void AnalyzeMprrAsMeshIndices(PM4File pm4File, string outputDir)
        {
            Console.WriteLine("\n🔬 ANALYZING MPRR AS MESH INDICES...");

            if (pm4File.MPRR?.Sequences == null || pm4File.MSVT?.Vertices == null)
                return;

            var outputPath = Path.Combine(outputDir, "mprr_mesh_analysis.obj");
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? outputDir);

            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# MPRR Mesh Analysis - {DateTime.Now}");
            writer.WriteLine("# Testing MPRR sequences as mesh connectivity");
            writer.WriteLine();

            // Write all MSVT vertices with PM4 coordinate transform
            writer.WriteLine("# MSVT Vertices");
            foreach (var vertex in pm4File.MSVT.Vertices)
            {
                var coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
            }
            writer.WriteLine();

            // Try to interpret MPRR sequences as faces
            writer.WriteLine("# MPRR Sequences as Faces");
            int facesWritten = 0;
            int sequenceIndex = 0;

            foreach (var sequence in pm4File.MPRR.Sequences.Take(100)) // Limit for testing
            {
                if (sequence.Count >= 3)
                {
                    // Check if all indices are valid
                    bool validIndices = sequence.All(idx => idx < pm4File.MSVT.Vertices.Count);
                    
                    if (validIndices)
                    {
                        if (sequence.Count == 3)
                        {
                            // Simple triangle
                            writer.WriteLine($"f {sequence[0] + 1} {sequence[1] + 1} {sequence[2] + 1} # MPRR_Seq_{sequenceIndex}");
                            facesWritten++;
                        }
                        else if (sequence.Count == 4)
                        {
                            // Quad - split into two triangles
                            writer.WriteLine($"f {sequence[0] + 1} {sequence[1] + 1} {sequence[2] + 1} # MPRR_Seq_{sequenceIndex}_tri1");
                            writer.WriteLine($"f {sequence[0] + 1} {sequence[2] + 1} {sequence[3] + 1} # MPRR_Seq_{sequenceIndex}_tri2");
                            facesWritten += 2;
                        }
                        else if (sequence.Count > 4)
                        {
                            // Polygon - fan triangulation
                            for (int i = 1; i < sequence.Count - 1; i++)
                            {
                                writer.WriteLine($"f {sequence[0] + 1} {sequence[i] + 1} {sequence[i + 1] + 1} # MPRR_Seq_{sequenceIndex}_fan_{i}");
                                facesWritten++;
                            }
                        }
                    }
                }
                sequenceIndex++;
            }

            Console.WriteLine($"   📊 Generated {facesWritten} faces from MPRR sequences");
            Console.WriteLine($"   📄 Test mesh: {Path.GetFileName(outputPath)}");
        }

        private void ExportMprrTriangleSample(PM4File pm4File, List<List<uint>> triangleSequences, string outputPath)
        {
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# MPRR Triangle Sample - {DateTime.Now}");
            writer.WriteLine($"# Sample of {Math.Min(triangleSequences.Count, 50)} triangle sequences");
            writer.WriteLine();

            // Write vertices
            if (pm4File.MSVT?.Vertices != null)
            {
                foreach (var vertex in pm4File.MSVT.Vertices)
                {
                    var coords = Pm4CoordinateTransforms.FromMsvtVertexSimple(vertex);
                    writer.WriteLine($"v {coords.X:F6} {coords.Y:F6} {coords.Z:F6}");
                }
            }

            // Write triangle faces from MPRR
            writer.WriteLine("\n# Triangle faces from MPRR sequences");
            int faceCount = 0;
            foreach (var triangle in triangleSequences.Take(50))
            {
                if (triangle.Count == 3 && pm4File.MSVT?.Vertices != null)
                {
                    // Validate indices
                    if (triangle.All(idx => idx < pm4File.MSVT.Vertices.Count))
                    {
                        writer.WriteLine($"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}");
                        faceCount++;
                    }
                }
            }

            Console.WriteLine($"      → Exported {faceCount} triangles to sample file");
        }
    }
} 