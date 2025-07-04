using System;
using System.IO;
using Xunit;
using WoWToolbox.Core.Navigation.PM4;
using WoWToolbox.Core.v2.Services.PM4;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;

namespace WoWToolbox.Tests.Navigation.PM4
{
    public class SimpleAnalysisTest
    {
        private readonly ICoordinateService _coordinateService;

        public SimpleAnalysisTest()
        {
            _coordinateService = new CoordinateService();
        }
        private static string TestDataRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "test_data"));
        private static string OutputRoot => Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "..", "output", DateTime.Now.ToString("yyyyMMdd_HHmmss"), "SimpleAnalysis"));

        [Fact]
        public void AnalyzePm4ChunkRelationships()
        {
            Console.WriteLine("=== PM4 Chunk Analysis for Mesh Connectivity Issues ===");
            
            // Create output directory
            Directory.CreateDirectory(OutputRoot);
            
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"⚠️  Test file not found: {testFile}");
                Console.WriteLine($"   Looking in: {Path.GetDirectoryName(testFile)}");
                if (Directory.Exists(Path.GetDirectoryName(testFile)))
                {
                    var files = Directory.GetFiles(Path.GetDirectoryName(testFile), "*.pm4");
                    Console.WriteLine($"   Available PM4 files: {files.Length}");
                    foreach (var file in files.Take(5))
                    {
                        Console.WriteLine($"     - {Path.GetFileName(file)}");
                    }
                }
                return;
            }

            Console.WriteLine($"📁 Analyzing: {Path.GetFileName(testFile)}");
            
            try
            {
                var analyzer = new Pm4ChunkAnalyzer();
                var result = analyzer.AnalyzePm4File(testFile);

                // Write analysis report
                var reportPath = Path.Combine(OutputRoot, "chunk_analysis_report.md");
                analyzer.WriteAnalysisReport(result, reportPath);
                Console.WriteLine($"📊 Analysis report written: {reportPath}");

                // Print key findings to console
                Console.WriteLine("\n🔍 KEY FINDINGS:");
                foreach (var insight in result.Insights)
                {
                    Console.WriteLine($"  • {insight}");
                }

                // Check for specific mesh connectivity issues
                Console.WriteLine("\n🔧 MESH CONNECTIVITY ANALYSIS:");
                Console.WriteLine($"  • MSUR surfaces: {result.MsurAnalysis.TotalSurfaces}");
                Console.WriteLine($"  • Valid MSVI ranges: {result.MsurAnalysis.ValidMsviRanges}");
                Console.WriteLine($"  • Invalid MSVI ranges: {result.MsurAnalysis.InvalidMsviRanges}");
                Console.WriteLine($"  • Triangle surfaces: {result.MsurAnalysis.TriangleSurfaces}");
                Console.WriteLine($"  • Quad surfaces: {result.MsurAnalysis.QuadSurfaces}");
                
                if (result.MsurAnalysis.InvalidMsviRanges > 0)
                {
                    Console.WriteLine($"  ⚠️  FOUND MESH CONNECTIVITY ISSUES: {result.MsurAnalysis.InvalidMsviRanges} invalid ranges");
                }

                // Check MPRR sequence patterns
                Console.WriteLine("\n📊 MPRR SEQUENCE ANALYSIS:");
                Console.WriteLine($"  • Total sequences: {result.MprAnalysis.TotalSequences}");
                Console.WriteLine($"  • Total values: {result.MprAnalysis.TotalValues}");
                Console.WriteLine($"  • Value range: {result.MprAnalysis.MinValue} - {result.MprAnalysis.MaxValue}");
                Console.WriteLine($"  • Possible index target: {result.MprAnalysis.IndexTarget}");
                
                if (result.Counts.MPRL_Entries > 0)
                {
                    float ratio = (float)result.MprAnalysis.TotalValues / result.Counts.MPRL_Entries;
                    Console.WriteLine($"  • MPRR/MPRL ratio: {ratio:F1}");
                    
                    if (ratio > 50)
                    {
                        Console.WriteLine($"  🔍 HIGH RATIO suggests MPRR might contain mesh connectivity data");
                    }
                }

                // Coordinate alignment check
                Console.WriteLine("\n🎯 COORDINATE ALIGNMENT:");
                Console.WriteLine($"  • MSCN/MSVT overlap: {result.CoordAnalysis.MSVT_MSCN_Overlap:P1}");
                
                if (result.CoordAnalysis.MSVT_MSCN_Overlap > 0.8f)
                {
                    Console.WriteLine($"  ✅ Excellent coordinate alignment");
                }
                else
                {
                    Console.WriteLine($"  ⚠️  Coordinate alignment needs investigation");
                }

                Console.WriteLine($"\n📂 Full analysis written to: {OutputRoot}");
                
                Assert.True(result.Insights.Count > 0, "Should have generated insights");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during analysis: {ex.Message}");
                throw;
            }
        }

        [Fact]
        public void QuickMprrPatternAnalysis()
        {
            Console.WriteLine("=== Quick MPRR Pattern Analysis ===");
            
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"⚠️  Test file not found: {testFile}");
                return;
            }

            try
            {
                var pm4File = PM4File.FromFile(testFile);
                
                Console.WriteLine($"📊 File: {Path.GetFileName(testFile)}");
                Console.WriteLine($"  • MSVT vertices: {pm4File.MSVT?.Vertices.Count ?? 0:N0}");
                Console.WriteLine($"  • MSVI indices: {pm4File.MSVI?.Indices.Count ?? 0:N0}");
                Console.WriteLine($"  • MPRL entries: {pm4File.MPRL?.Entries.Count ?? 0:N0}");
                Console.WriteLine($"  • MPRR sequences: {pm4File.MPRR?.Sequences?.Count ?? 0:N0}");

                if (pm4File.MPRR?.Sequences != null && pm4File.MPRR.Sequences.Count > 0)
                {
                    var allValues = pm4File.MPRR.Sequences.SelectMany(s => s).ToList();
                    var sequenceLengths = pm4File.MPRR.Sequences.Select(s => s.Count).ToList();
                    
                    Console.WriteLine($"\n🔍 MPRR Pattern Analysis:");
                    Console.WriteLine($"  • Total values: {allValues.Count:N0}");
                    Console.WriteLine($"  • Value range: {allValues.Min()} - {allValues.Max()}");
                    Console.WriteLine($"  • Sequence length range: {sequenceLengths.Min()} - {sequenceLengths.Max()}");
                    
                    var lengthGroups = sequenceLengths.GroupBy(l => l).OrderBy(g => g.Key).ToList();
                    Console.WriteLine($"  • Sequence length distribution:");
                    foreach (var group in lengthGroups.Take(10))
                    {
                        Console.WriteLine($"    - Length {group.Key}: {group.Count()} sequences");
                    }
                    
                    // Check if MPRR values could be indices
                    var msvtCount = pm4File.MSVT?.Vertices.Count ?? 0;
                    if (allValues.Max() < msvtCount)
                    {
                        Console.WriteLine($"  ✅ MPRR values could index into MSVT vertices");
                        
                        var triangleSequences = pm4File.MPRR.Sequences.Where(s => s.Count == 3).ToList();
                        if (triangleSequences.Any())
                        {
                            Console.WriteLine($"  🔺 Found {triangleSequences.Count} potential triangle sequences");
                            Console.WriteLine($"     → These could represent triangle faces for mesh connectivity");
                        }
                    }
                    else
                    {
                        Console.WriteLine($"  ❌ MPRR values too large for vertex indices");
                    }
                }
                else
                {
                    Console.WriteLine("  ❌ No MPRR sequences found");
                }
                
                Assert.True(pm4File != null, "Should successfully load PM4 file");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
                throw;
            }
        }

        [Fact]
        public void AnalyzeChunkSpatialDistribution()
        {
            Console.WriteLine("=== Analyzing Chunk Spatial Distribution ===");
            
            Directory.CreateDirectory(OutputRoot);
            
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"⚠️  Test file not found: {testFile}");
                return;
            }

            try
            {
                var pm4File = PM4File.FromFile(testFile);
                
                Console.WriteLine($"📊 Analyzing spatial distribution for: {Path.GetFileName(testFile)}");
                
                // Analyze each chunk type's coordinate ranges
                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var msvtCoords = pm4File.MSVT.Vertices.Select(v => _coordinateService.FromMsvtVertexSimple(v)).ToList();
                    var msvtBounds = CalculateBounds(msvtCoords);
                    Console.WriteLine($"\n🎯 MSVT Bounds: {msvtBounds}");
                }
                
                if (pm4File.MSCN?.ExteriorVertices != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                {
                    var mscnCoords = pm4File.MSCN.ExteriorVertices.Select(v => _coordinateService.FromMscnVertex(v)).ToList();
                    var mscnBounds = CalculateBounds(mscnCoords);
                    Console.WriteLine($"🛡️  MSCN Bounds: {mscnBounds}");
                }
                
                if (pm4File.MSPV?.Vertices != null && pm4File.MSPV.Vertices.Count > 0)
                {
                    var mspvCoords = pm4File.MSPV.Vertices.Select(v => _coordinateService.FromMspvVertex(v)).ToList();
                    var mspvBounds = CalculateBounds(mspvCoords);
                    Console.WriteLine($"📐 MSPV Bounds: {mspvBounds}");
                }
                
                if (pm4File.MPRL?.Entries != null && pm4File.MPRL.Entries.Count > 0)
                {
                    var mprlCoords = pm4File.MPRL.Entries.Select(e => _coordinateService.FromMprlEntry(e)).ToList();
                    var mprlBounds = CalculateBounds(mprlCoords);
                    Console.WriteLine($"📍 MPRL Bounds: {mprlBounds}");
                }

                // Export individual chunk OBJ files for detailed analysis
                var fileName = Path.GetFileNameWithoutExtension(testFile);
                
                if (pm4File.MSVT?.Vertices != null && pm4File.MSVT.Vertices.Count > 0)
                {
                    var msvtPath = Path.Combine(OutputRoot, $"{fileName}_msvt_only.obj");
                    ExportChunkToObj(pm4File.MSVT.Vertices.Select(v => _coordinateService.FromMsvtVertexSimple(v)), msvtPath, "MSVT");
                    Console.WriteLine($"📁 Exported MSVT → {msvtPath}");
                }
                
                if (pm4File.MSCN?.ExteriorVertices != null && pm4File.MSCN.ExteriorVertices.Count > 0)
                {
                    var mscnPath = Path.Combine(OutputRoot, $"{fileName}_mscn_only.obj");
                    ExportChunkToObj(pm4File.MSCN.ExteriorVertices.Select(v => _coordinateService.FromMscnVertex(v)), mscnPath, "MSCN");
                    Console.WriteLine($"📁 Exported MSCN → {mscnPath}");
                }
                
                if (pm4File.MSPV?.Vertices != null && pm4File.MSPV.Vertices.Count > 0)
                {
                    var mspvPath = Path.Combine(OutputRoot, $"{fileName}_mspv_only.obj");
                    ExportChunkToObj(pm4File.MSPV.Vertices.Select(v => _coordinateService.FromMspvVertex(v)), mspvPath, "MSPV");
                    Console.WriteLine($"📁 Exported MSPV → {mspvPath}");
                }
                
                if (pm4File.MPRL?.Entries != null && pm4File.MPRL.Entries.Count > 0)
                {
                    var mprlPath = Path.Combine(OutputRoot, $"{fileName}_mprl_only.obj");
                    ExportChunkToObj(pm4File.MPRL.Entries.Select(e => _coordinateService.FromMprlEntry(e)), mprlPath, "MPRL");
                    Console.WriteLine($"📁 Exported MPRL → {mprlPath}");
                }

                Console.WriteLine($"\n✅ Analysis complete. Check individual OBJ files in MeshLab to identify spatial issues.");
                
                Assert.True(pm4File != null, "Should successfully load PM4 file");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during spatial analysis: {ex.Message}");
                throw;
            }
        }

        [Fact]
        public void AnalyzeMsurMsviVertexMapping()
        {
            Console.WriteLine("=== MSUR→MSVI→Vertex Mapping Analysis ===");
            
            Directory.CreateDirectory(OutputRoot);
            
            var testFile = Path.Combine(TestDataRoot, "original_development", "development", "development_00_00.pm4");
            
            if (!File.Exists(testFile))
            {
                Console.WriteLine($"⚠️  Test file not found: {testFile}");
                return;
            }

            try
            {
                var pm4File = PM4File.FromFile(testFile);
                
                Console.WriteLine($"📊 Analyzing vertex mapping for: {Path.GetFileName(testFile)}");
                Console.WriteLine($"  • MSVT vertices: {pm4File.MSVT?.Vertices.Count ?? 0:N0}");
                Console.WriteLine($"  • MSPV vertices: {pm4File.MSPV?.Vertices.Count ?? 0:N0}");
                Console.WriteLine($"  • MSVI indices: {pm4File.MSVI?.Indices.Count ?? 0:N0}");
                Console.WriteLine($"  • MSUR surfaces: {pm4File.MSUR?.Entries.Count ?? 0:N0}");
                
                var analysisPath = Path.Combine(OutputRoot, "msur_msvi_mapping_analysis.log");
                using var writer = new StreamWriter(analysisPath);
                
                writer.WriteLine("=== MSUR→MSVI→Vertex Mapping Analysis ===");
                writer.WriteLine($"File: {Path.GetFileName(testFile)}");
                writer.WriteLine($"MSVT vertices: {pm4File.MSVT?.Vertices.Count ?? 0:N0}");
                writer.WriteLine($"MSPV vertices: {pm4File.MSPV?.Vertices.Count ?? 0:N0}");
                writer.WriteLine($"MSVI indices: {pm4File.MSVI?.Indices.Count ?? 0:N0}");
                writer.WriteLine($"MSUR surfaces: {pm4File.MSUR?.Entries.Count ?? 0:N0}");
                writer.WriteLine();

                if (pm4File.MSUR?.Entries != null && pm4File.MSVI?.Indices != null)
                {
                    writer.WriteLine("=== MSUR Surface Analysis ===");
                    
                    int validSurfaces = 0;
                    int invalidSurfaces = 0;
                    int totalTriangles = 0;
                    var indexStats = new List<uint>();
                    
                    for (int i = 0; i < pm4File.MSUR.Entries.Count && i < 20; i++) // Analyze first 20 surfaces
                    {
                        var msur = pm4File.MSUR.Entries[i];
                        writer.WriteLine($"\nSurface {i}:");
                        writer.WriteLine($"  MsviFirstIndex: {msur.MsviFirstIndex}");
                        writer.WriteLine($"  IndexCount: {msur.IndexCount}");
                        writer.WriteLine($"  Range: [{msur.MsviFirstIndex}, {msur.MsviFirstIndex + msur.IndexCount - 1}]");
                        
                        bool isValidRange = msur.MsviFirstIndex >= 0 && 
                                          msur.MsviFirstIndex + msur.IndexCount <= pm4File.MSVI.Indices.Count;
                        
                        if (isValidRange)
                        {
                            validSurfaces++;
                            
                            // Analyze the actual indices in this surface
                            var surfaceIndices = new List<uint>();
                            for (int j = 0; j < msur.IndexCount; j++)
                            {
                                var index = pm4File.MSVI.Indices[(int)msur.MsviFirstIndex + j];
                                surfaceIndices.Add(index);
                                indexStats.Add(index);
                            }
                            
                            writer.WriteLine($"  Indices: [{string.Join(", ", surfaceIndices.Take(10))}" + 
                                           (surfaceIndices.Count > 10 ? "..." : "") + "]");
                            writer.WriteLine($"  Index range in surface: {surfaceIndices.Min()} - {surfaceIndices.Max()}");
                            
                            // Check if indices are valid for MSVT
                            var msvtVertexCount = pm4File.MSVT?.Vertices.Count ?? 0;
                            var mspvVertexCount = pm4File.MSPV?.Vertices.Count ?? 0;
                            
                            bool validForMsvt = surfaceIndices.All(idx => idx < msvtVertexCount);
                            bool validForMspv = surfaceIndices.All(idx => idx < mspvVertexCount);
                            
                            writer.WriteLine($"  Valid for MSVT ({msvtVertexCount} vertices): {validForMsvt}");
                            writer.WriteLine($"  Valid for MSPV ({mspvVertexCount} vertices): {validForMspv}");
                            
                            if (!validForMsvt && !validForMspv)
                            {
                                writer.WriteLine($"  ⚠️  INDICES OUT OF BOUNDS FOR BOTH VERTEX BUFFERS!");
                                var maxIndex = surfaceIndices.Max();
                                writer.WriteLine($"      Max index: {maxIndex}, MSVT count: {msvtVertexCount}, MSPV count: {mspvVertexCount}");
                            }
                            
                            // Calculate potential triangles from this surface (triangle fan)
                            if (msur.IndexCount >= 3)
                            {
                                int trianglesFromSurface = (int)msur.IndexCount - 2;
                                totalTriangles += trianglesFromSurface;
                                writer.WriteLine($"  Potential triangles (fan): {trianglesFromSurface}");
                            }
                        }
                        else
                        {
                            invalidSurfaces++;
                            writer.WriteLine($"  ⚠️  INVALID RANGE: exceeds MSVI bounds");
                        }
                    }
                    
                    writer.WriteLine($"\n=== Summary ===");
                    writer.WriteLine($"Valid surfaces: {validSurfaces}");
                    writer.WriteLine($"Invalid surfaces: {invalidSurfaces}");
                    writer.WriteLine($"Total potential triangles: {totalTriangles}");
                    
                    if (indexStats.Count > 0)
                    {
                        writer.WriteLine($"Overall index statistics:");
                        writer.WriteLine($"  Min index: {indexStats.Min()}");
                        writer.WriteLine($"  Max index: {indexStats.Max()}");
                        writer.WriteLine($"  Unique indices: {indexStats.Distinct().Count()}");
                        
                        // Check index distribution
                        var indexGroups = indexStats.GroupBy(i => i / 100).OrderBy(g => g.Key).ToList();
                        writer.WriteLine($"  Index distribution (by hundreds):");
                        foreach (var group in indexGroups.Take(10))
                        {
                            writer.WriteLine($"    {group.Key * 100}-{(group.Key + 1) * 100 - 1}: {group.Count()} indices");
                        }
                    }
                }
                
                // Additional check: Compare coordinate ranges
                if (pm4File.MSVT?.Vertices != null && pm4File.MSPV?.Vertices != null)
                {
                    writer.WriteLine($"\n=== Coordinate Range Comparison ===");
                    
                    // MSVT coordinate analysis
                    var msvtCoords = pm4File.MSVT.Vertices.Select(v => _coordinateService.FromMsvtVertexSimple(v)).ToList();
                    if (msvtCoords.Any())
                    {
                        writer.WriteLine($"MSVT coordinate ranges:");
                        writer.WriteLine($"  X: {msvtCoords.Min(c => c.X):F2} to {msvtCoords.Max(c => c.X):F2}");
                        writer.WriteLine($"  Y: {msvtCoords.Min(c => c.Y):F2} to {msvtCoords.Max(c => c.Y):F2}");
                        writer.WriteLine($"  Z: {msvtCoords.Min(c => c.Z):F2} to {msvtCoords.Max(c => c.Z):F2}");
                    }
                    
                    // MSPV coordinate analysis
                    var mspvCoords = pm4File.MSPV.Vertices.Select(v => _coordinateService.FromMspvVertex(v)).ToList();
                    if (mspvCoords.Any())
                    {
                        writer.WriteLine($"MSPV coordinate ranges:");
                        writer.WriteLine($"  X: {mspvCoords.Min(c => c.X):F2} to {mspvCoords.Max(c => c.X):F2}");
                        writer.WriteLine($"  Y: {mspvCoords.Min(c => c.Y):F2} to {mspvCoords.Max(c => c.Y):F2}");
                        writer.WriteLine($"  Z: {mspvCoords.Min(c => c.Z):F2} to {mspvCoords.Max(c => c.Z):F2}");
                    }
                }

                Console.WriteLine($"📁 Detailed analysis written to: {analysisPath}");
                Console.WriteLine("\n🔍 KEY FINDINGS will be in the analysis file - check for:");
                Console.WriteLine("  • Index bounds violations");
                Console.WriteLine("  • MSVT vs MSPV vertex count mismatches");
                Console.WriteLine("  • Coordinate range differences");
                Console.WriteLine("  • Invalid surface definitions");
                
                Assert.True(pm4File != null, "Should successfully load PM4 file");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error during vertex mapping analysis: {ex.Message}");
                throw;
            }
        }

        private static string CalculateBounds(List<Vector3> coords)
        {
            if (!coords.Any()) return "No coordinates";
            
            var minX = coords.Min(c => c.X);
            var maxX = coords.Max(c => c.X);
            var minY = coords.Min(c => c.Y);
            var maxY = coords.Max(c => c.Y);
            var minZ = coords.Min(c => c.Z);
            var maxZ = coords.Max(c => c.Z);
            
            return $"X:[{minX:F2}, {maxX:F2}] Y:[{minY:F2}, {maxY:F2}] Z:[{minZ:F2}, {maxZ:F2}]";
        }

        private static void ExportChunkToObj(IEnumerable<Vector3> coords, string outputPath, string chunkType)
        {
            using var writer = new StreamWriter(outputPath);
            writer.WriteLine($"# {chunkType} Chunk Only - Spatial Analysis");
            writer.WriteLine($"# Generated: {DateTime.Now}");
            writer.WriteLine($"o {chunkType}_Points");
            
            foreach (var coord in coords)
            {
                writer.WriteLine($"v {coord.X:F6} {coord.Y:F6} {coord.Z:F6}");
            }
        }
    }
} 