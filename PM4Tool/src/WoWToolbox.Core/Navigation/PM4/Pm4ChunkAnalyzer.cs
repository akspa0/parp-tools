using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using WoWToolbox.Core.Navigation.PM4.Chunks;

namespace WoWToolbox.Core.Navigation.PM4
{
    /// <summary>
    /// Analysis tool for understanding PM4 chunk relationships and mesh connectivity
    /// </summary>
    public class Pm4ChunkAnalyzer
    {
        public class ChunkAnalysisResult
        {
            public string FileName { get; set; } = string.Empty;
            public ChunkCounts Counts { get; set; } = new();
            public MprSequenceAnalysis MprAnalysis { get; set; } = new();
            public MslkGeometryAnalysis MslkAnalysis { get; set; } = new();
            public MsurConnectivityAnalysis MsurAnalysis { get; set; } = new();
            public CoordinateRangeAnalysis CoordAnalysis { get; set; } = new();
            public List<string> Insights { get; set; } = new();
        }

        public class ChunkCounts
        {
            public int MSVT_Vertices { get; set; }
            public int MSVI_Indices { get; set; }
            public int MSPV_Vertices { get; set; }
            public int MSPI_Indices { get; set; }
            public int MPRL_Entries { get; set; }
            public int MPRR_Sequences { get; set; }
            public int MSLK_Entries { get; set; }
            public int MSUR_Surfaces { get; set; }
            public int MSCN_Points { get; set; }
            public int MDOS_Objects { get; set; }
            public int MDSF_Links { get; set; }
        }

        public class MprSequenceAnalysis
        {
            public int TotalSequences { get; set; }
            public int TotalValues { get; set; }
            public List<int> SequenceLengths { get; set; } = new();
            public List<uint> UniqueValues { get; set; } = new();
            public uint MaxValue { get; set; }
            public uint MinValue { get; set; }
            public Dictionary<uint, int> ValueFrequency { get; set; } = new();
            public bool PossibleIndices { get; set; }
            public string IndexTarget { get; set; } = "Unknown";
        }

        public class MslkGeometryAnalysis
        {
            public int GeometryEntries { get; set; }
            public int NodeEntries { get; set; }
            public List<int> GeometryIndexCounts { get; set; } = new();
            public Dictionary<uint, int> GroupFrequency { get; set; } = new();
            public bool HasValidGeometry { get; set; }
            public int ValidMspiRanges { get; set; }
            public int InvalidMspiRanges { get; set; }
        }

        public class MsurConnectivityAnalysis
        {
            public int TotalSurfaces { get; set; }
            public List<int> IndexCounts { get; set; } = new();
            public int TriangleSurfaces { get; set; }
            public int QuadSurfaces { get; set; }
            public int PolygonSurfaces { get; set; }
            public int ValidMsviRanges { get; set; }
            public int InvalidMsviRanges { get; set; }
            public bool HasValidConnectivity { get; set; }
        }

        public class CoordinateRangeAnalysis
        {
            public Vector3 MSVT_Min { get; set; }
            public Vector3 MSVT_Max { get; set; }
            public Vector3 MSCN_Min { get; set; }
            public Vector3 MSCN_Max { get; set; }
            public Vector3 MSPV_Min { get; set; }
            public Vector3 MSPV_Max { get; set; }
            public Vector3 MPRL_Min { get; set; }
            public Vector3 MPRL_Max { get; set; }
            public float MSVT_MSCN_Overlap { get; set; }
        }

        public ChunkAnalysisResult AnalyzePm4File(string filePath)
        {
            var result = new ChunkAnalysisResult
            {
                FileName = Path.GetFileName(filePath)
            };

            try
            {
                var pm4File = PM4File.FromFile(filePath);
                
                // Count all chunks
                AnalyzeChunkCounts(pm4File, result.Counts);
                
                // Analyze MPRR sequences
                if (pm4File.MPRR != null)
                    AnalyzeMprrSequences(pm4File.MPRR, result.MprAnalysis, result.Counts);
                
                // Analyze MSLK geometry
                if (pm4File.MSLK != null && pm4File.MSPI != null)
                    AnalyzeMslkGeometry(pm4File.MSLK, pm4File.MSPI, result.MslkAnalysis, result.Counts);
                
                // Analyze MSUR connectivity
                if (pm4File.MSUR != null && pm4File.MSVI != null)
                    AnalyzeMsurConnectivity(pm4File.MSUR, pm4File.MSVI, result.MsurAnalysis, result.Counts);
                
                // Analyze coordinate ranges
                AnalyzeCoordinateRanges(pm4File, result.CoordAnalysis);
                
                // Generate insights
                GenerateInsights(result);
            }
            catch (Exception ex)
            {
                result.Insights.Add($"ERROR: Failed to analyze file: {ex.Message}");
            }

            return result;
        }

        /// <summary>
        /// Performs comprehensive MSLK relationship analysis with Mermaid diagram output.
        /// </summary>
        /// <param name="filePath">Path to the PM4 file</param>
        /// <param name="outputMermaidToConsole">Whether to output Mermaid diagrams to console</param>
        public void AnalyzePm4FileWithMslkRelationships(string filePath, bool outputMermaidToConsole = true)
        {
            try
            {
                var pm4File = PM4File.FromFile(filePath);
                var fileName = Path.GetFileName(filePath);
                
                // Perform standard chunk analysis
                var standardAnalysis = AnalyzePm4File(filePath);
                
                // Output standard analysis insights
                Console.WriteLine($"🔍 STANDARD PM4 ANALYSIS: {fileName}");
                foreach (var insight in standardAnalysis.Insights)
                {
                    Console.WriteLine($"  📋 {insight}");
                }
                Console.WriteLine();
                
                // Perform enhanced MSLK relationship analysis
                Pm4MslkCliAnalyzer.AnalyzeAndOutputMslkRelationships(pm4File, fileName, outputMermaidToConsole);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error analyzing {Path.GetFileName(filePath)}: {ex.Message}");
            }
        }

        private void AnalyzeChunkCounts(PM4File pm4File, ChunkCounts counts)
        {
            counts.MSVT_Vertices = pm4File.MSVT?.Vertices.Count ?? 0;
            counts.MSVI_Indices = pm4File.MSVI?.Indices.Count ?? 0;
            counts.MSPV_Vertices = pm4File.MSPV?.Vertices.Count ?? 0;
            counts.MSPI_Indices = pm4File.MSPI?.Indices.Count ?? 0;
            counts.MPRL_Entries = pm4File.MPRL?.Entries.Count ?? 0;
            counts.MPRR_Sequences = pm4File.MPRR?.Sequences?.Count ?? 0;
            counts.MSLK_Entries = pm4File.MSLK?.Entries.Count ?? 0;
            counts.MSUR_Surfaces = pm4File.MSUR?.Entries.Count ?? 0;
            counts.MSCN_Points = pm4File.MSCN?.ExteriorVertices.Count ?? 0;
            counts.MDOS_Objects = pm4File.MDOS?.Entries.Count ?? 0;
            counts.MDSF_Links = pm4File.MDSF?.Entries?.Count ?? 0;
        }

        private void AnalyzeMprrSequences(MPRRChunk mprr, MprSequenceAnalysis analysis, ChunkCounts counts)
        {
            analysis.TotalSequences = mprr.Sequences?.Count ?? 0;
            
            if (mprr.Sequences == null || mprr.Sequences.Count == 0)
                return;

            var allValues = new List<uint>();
            
            foreach (var sequence in mprr.Sequences)
            {
                analysis.SequenceLengths.Add(sequence.Count);
                // Convert ushort to uint for analysis
                allValues.AddRange(sequence.Select(v => (uint)v));
            }

            analysis.TotalValues = allValues.Count;
            analysis.UniqueValues = allValues.Distinct().OrderBy(v => v).ToList();
            analysis.MaxValue = allValues.Any() ? allValues.Max() : 0;
            analysis.MinValue = allValues.Any() ? allValues.Min() : 0;

            // Calculate value frequency
            analysis.ValueFrequency = allValues.GroupBy(v => v)
                .ToDictionary(g => g.Key, g => g.Count());

            // Determine possible index targets
            if (analysis.MaxValue < counts.MPRL_Entries)
            {
                analysis.PossibleIndices = true;
                analysis.IndexTarget = "MPRL";
            }
            else if (analysis.MaxValue < counts.MSVT_Vertices)
            {
                analysis.PossibleIndices = true;
                analysis.IndexTarget = "MSVT";
            }
            else if (analysis.MaxValue < counts.MSPV_Vertices)
            {
                analysis.PossibleIndices = true;
                analysis.IndexTarget = "MSPV";
            }
            else
            {
                analysis.PossibleIndices = false;
                analysis.IndexTarget = "Unknown - values too large for any vertex array";
            }
        }

        private void AnalyzeMslkGeometry(MSLK mslk, MSPIChunk mspi, MslkGeometryAnalysis analysis, ChunkCounts counts)
        {
            foreach (var entry in mslk.Entries)
            {
                if (entry.MspiFirstIndex == -1)
                {
                    analysis.NodeEntries++;
                }
                else
                {
                    analysis.GeometryEntries++;
                    analysis.GeometryIndexCounts.Add(entry.MspiIndexCount);

                    // Check if MSPI range is valid
                    if (entry.MspiFirstIndex >= 0 && 
                        entry.MspiFirstIndex + entry.MspiIndexCount <= counts.MSPI_Indices)
                    {
                        analysis.ValidMspiRanges++;
                        analysis.HasValidGeometry = true;
                    }
                    else
                    {
                        analysis.InvalidMspiRanges++;
                    }
                }

                // Track group frequency using documented Index Reference field
                uint groupKey = entry.Unknown_0x04; // This is the documented "index somewhere" field
                if (!analysis.GroupFrequency.ContainsKey(groupKey))
                    analysis.GroupFrequency[groupKey] = 0;
                analysis.GroupFrequency[groupKey]++;
            }
        }

        private void AnalyzeMsurConnectivity(MSURChunk msur, MSVIChunk msvi, MsurConnectivityAnalysis analysis, ChunkCounts counts)
        {
            analysis.TotalSurfaces = msur.Entries.Count;

            foreach (var surface in msur.Entries)
            {
                analysis.IndexCounts.Add(surface.IndexCount);

                // Classify by index count
                if (surface.IndexCount == 3)
                    analysis.TriangleSurfaces++;
                else if (surface.IndexCount == 4)
                    analysis.QuadSurfaces++;
                else if (surface.IndexCount > 4)
                    analysis.PolygonSurfaces++;

                // Check if MSVI range is valid
                if (surface.MsviFirstIndex >= 0 && 
                    surface.MsviFirstIndex + surface.IndexCount <= counts.MSVI_Indices)
                {
                    analysis.ValidMsviRanges++;
                    analysis.HasValidConnectivity = true;
                }
                else
                {
                    analysis.InvalidMsviRanges++;
                }
            }
        }

        private void AnalyzeCoordinateRanges(PM4File pm4File, CoordinateRangeAnalysis analysis)
        {
            // MSVT coordinate ranges
            if (pm4File.MSVT?.Vertices.Count > 0)
            {
                var msvtCoords = pm4File.MSVT.Vertices.Select(v => Pm4CoordinateTransforms.FromMsvtVertexSimple(v)).ToList();
                analysis.MSVT_Min = new Vector3(
                    msvtCoords.Min(v => v.X),
                    msvtCoords.Min(v => v.Y),
                    msvtCoords.Min(v => v.Z)
                );
                analysis.MSVT_Max = new Vector3(
                    msvtCoords.Max(v => v.X),
                    msvtCoords.Max(v => v.Y),
                    msvtCoords.Max(v => v.Z)
                );
            }

            // MSCN coordinate ranges
            if (pm4File.MSCN?.ExteriorVertices.Count > 0)
            {
                var mscnCoords = pm4File.MSCN.ExteriorVertices.Select(v => Pm4CoordinateTransforms.FromMscnVertex(v)).ToList();
                analysis.MSCN_Min = new Vector3(
                    mscnCoords.Min(v => v.X),
                    mscnCoords.Min(v => v.Y),
                    mscnCoords.Min(v => v.Z)
                );
                analysis.MSCN_Max = new Vector3(
                    mscnCoords.Max(v => v.X),
                    mscnCoords.Max(v => v.Y),
                    mscnCoords.Max(v => v.Z)
                );

                // Calculate overlap with MSVT
                if (pm4File.MSVT?.Vertices.Count > 0)
                {
                    var overlapMin = Vector3.Max(analysis.MSVT_Min, analysis.MSCN_Min);
                    var overlapMax = Vector3.Min(analysis.MSVT_Max, analysis.MSCN_Max);
                    
                    if (overlapMin.X <= overlapMax.X && overlapMin.Y <= overlapMax.Y && overlapMin.Z <= overlapMax.Z)
                    {
                        var overlapVolume = (overlapMax.X - overlapMin.X) * (overlapMax.Y - overlapMin.Y) * (overlapMax.Z - overlapMin.Z);
                        var msvtVolume = (analysis.MSVT_Max.X - analysis.MSVT_Min.X) * (analysis.MSVT_Max.Y - analysis.MSVT_Min.Y) * (analysis.MSVT_Max.Z - analysis.MSVT_Min.Z);
                        analysis.MSVT_MSCN_Overlap = msvtVolume > 0 ? overlapVolume / msvtVolume : 0;
                    }
                }
            }

            // MSPV and MPRL ranges (similar logic)
            if (pm4File.MSPV?.Vertices.Count > 0)
            {
                var mspvCoords = pm4File.MSPV.Vertices.Select(v => Pm4CoordinateTransforms.FromMspvVertex(v)).ToList();
                analysis.MSPV_Min = new Vector3(mspvCoords.Min(v => v.X), mspvCoords.Min(v => v.Y), mspvCoords.Min(v => v.Z));
                analysis.MSPV_Max = new Vector3(mspvCoords.Max(v => v.X), mspvCoords.Max(v => v.Y), mspvCoords.Max(v => v.Z));
            }

            if (pm4File.MPRL?.Entries.Count > 0)
            {
                var mprlCoords = pm4File.MPRL.Entries.Select(e => Pm4CoordinateTransforms.FromMprlEntry(e)).ToList();
                analysis.MPRL_Min = new Vector3(mprlCoords.Min(v => v.X), mprlCoords.Min(v => v.Y), mprlCoords.Min(v => v.Z));
                analysis.MPRL_Max = new Vector3(mprlCoords.Max(v => v.X), mprlCoords.Max(v => v.Y), mprlCoords.Max(v => v.Z));
            }
        }

        private void GenerateInsights(ChunkAnalysisResult result)
        {
            var insights = result.Insights;
            var counts = result.Counts;
            var mpr = result.MprAnalysis;
            var mslk = result.MslkAnalysis;
            var msur = result.MsurAnalysis;

            // Mesh connectivity insights
            if (msur.HasValidConnectivity)
            {
                insights.Add($"✅ MSUR connectivity appears valid: {msur.ValidMsviRanges}/{msur.TotalSurfaces} surfaces have valid MSVI ranges");
            }
            else
            {
                insights.Add($"❌ MSUR connectivity issues: {msur.InvalidMsviRanges}/{msur.TotalSurfaces} surfaces have invalid MSVI ranges");
            }

            // MPRR sequence insights
            if (mpr.PossibleIndices)
            {
                insights.Add($"🔍 MPRR sequences might index into {mpr.IndexTarget} (max value: {mpr.MaxValue})");
                insights.Add($"📊 MPRR has {mpr.TotalSequences} sequences with {mpr.TotalValues} total values");
            }
            else
            {
                insights.Add($"❓ MPRR sequences don't appear to be indices (max value: {mpr.MaxValue} too large)");
            }

            // MSLK geometry insights
            if (mslk.HasValidGeometry)
            {
                insights.Add($"✅ MSLK geometry appears valid: {mslk.ValidMspiRanges} valid geometry entries");
            }
            else
            {
                insights.Add($"❌ MSLK geometry issues: {mslk.InvalidMspiRanges} invalid MSPI ranges");
            }

            // Coordinate alignment insights
            if (result.CoordAnalysis.MSVT_MSCN_Overlap > 0.8f)
            {
                insights.Add($"✅ Excellent MSVT/MSCN coordinate alignment: {result.CoordAnalysis.MSVT_MSCN_Overlap:P1} overlap");
            }
            else if (result.CoordAnalysis.MSVT_MSCN_Overlap > 0.5f)
            {
                insights.Add($"⚠️ Partial MSVT/MSCN coordinate alignment: {result.CoordAnalysis.MSVT_MSCN_Overlap:P1} overlap");
            }
            else
            {
                insights.Add($"❌ Poor MSVT/MSCN coordinate alignment: {result.CoordAnalysis.MSVT_MSCN_Overlap:P1} overlap");
            }

            // Data ratio insights
            float mprrToMprlRatio = counts.MPRL_Entries > 0 ? (float)mpr.TotalValues / counts.MPRL_Entries : 0;
            if (mprrToMprlRatio > 100)
            {
                insights.Add($"⚠️ Very high MPRR/MPRL ratio: {mprrToMprlRatio:F1} - possible connectivity data");
            }
            else if (mprrToMprlRatio > 10)
            {
                insights.Add($"🔍 High MPRR/MPRL ratio: {mprrToMprlRatio:F1} - investigate for patterns");
            }

            // Face generation insights
            if (msur.TriangleSurfaces > 0)
            {
                insights.Add($"📐 Found {msur.TriangleSurfaces} triangle surfaces, {msur.QuadSurfaces} quad surfaces, {msur.PolygonSurfaces} polygon surfaces");
            }

            if (counts.MSCN_Points > 0 && counts.MSVT_Vertices > 0)
            {
                float mscnToMsvtRatio = (float)counts.MSCN_Points / counts.MSVT_Vertices;
                if (Math.Abs(mscnToMsvtRatio - 1.0f) < 0.1f)
                {
                    insights.Add($"🎯 MSCN/MSVT count ratio near 1:1 ({mscnToMsvtRatio:F2}) - possible 1:1 correspondence");
                }
                else
                {
                    insights.Add($"📊 MSCN/MSVT count ratio: {mscnToMsvtRatio:F2}");
                }
            }
        }

        public void WriteAnalysisReport(ChunkAnalysisResult result, string outputPath)
        {
            using var writer = new StreamWriter(outputPath, false, Encoding.UTF8);
            
            writer.WriteLine($"# PM4 Chunk Analysis Report: {result.FileName}");
            writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            writer.WriteLine();

            // Chunk counts
            writer.WriteLine("## Chunk Counts");
            writer.WriteLine($"- MSVT Vertices: {result.Counts.MSVT_Vertices:N0}");
            writer.WriteLine($"- MSVI Indices: {result.Counts.MSVI_Indices:N0}");
            writer.WriteLine($"- MSPV Vertices: {result.Counts.MSPV_Vertices:N0}");
            writer.WriteLine($"- MSPI Indices: {result.Counts.MSPI_Indices:N0}");
            writer.WriteLine($"- MPRL Entries: {result.Counts.MPRL_Entries:N0}");
            writer.WriteLine($"- MPRR Sequences: {result.Counts.MPRR_Sequences:N0}");
            writer.WriteLine($"- MSLK Entries: {result.Counts.MSLK_Entries:N0}");
            writer.WriteLine($"- MSUR Surfaces: {result.Counts.MSUR_Surfaces:N0}");
            writer.WriteLine($"- MSCN Points: {result.Counts.MSCN_Points:N0}");
            writer.WriteLine($"- MDOS Objects: {result.Counts.MDOS_Objects:N0}");
            writer.WriteLine($"- MDSF Links: {result.Counts.MDSF_Links:N0}");
            writer.WriteLine();

            // MPRR analysis
            writer.WriteLine("## MPRR Sequence Analysis");
            writer.WriteLine($"- Total Sequences: {result.MprAnalysis.TotalSequences:N0}");
            writer.WriteLine($"- Total Values: {result.MprAnalysis.TotalValues:N0}");
            writer.WriteLine($"- Value Range: {result.MprAnalysis.MinValue} - {result.MprAnalysis.MaxValue}");
            writer.WriteLine($"- Possible Index Target: {result.MprAnalysis.IndexTarget}");
            if (result.MprAnalysis.SequenceLengths.Any())
            {
                writer.WriteLine($"- Sequence Length Range: {result.MprAnalysis.SequenceLengths.Min()} - {result.MprAnalysis.SequenceLengths.Max()}");
                writer.WriteLine($"- Average Sequence Length: {result.MprAnalysis.SequenceLengths.Average():F1}");
            }
            writer.WriteLine();

            // MSLK analysis
            writer.WriteLine("## MSLK Geometry Analysis");
            writer.WriteLine($"- Geometry Entries: {result.MslkAnalysis.GeometryEntries}");
            writer.WriteLine($"- Node Entries: {result.MslkAnalysis.NodeEntries}");
            writer.WriteLine($"- Valid MSPI Ranges: {result.MslkAnalysis.ValidMspiRanges}");
            writer.WriteLine($"- Invalid MSPI Ranges: {result.MslkAnalysis.InvalidMspiRanges}");
            writer.WriteLine();

            // MSUR analysis
            writer.WriteLine("## MSUR Connectivity Analysis");
            writer.WriteLine($"- Total Surfaces: {result.MsurAnalysis.TotalSurfaces}");
            writer.WriteLine($"- Triangle Surfaces: {result.MsurAnalysis.TriangleSurfaces}");
            writer.WriteLine($"- Quad Surfaces: {result.MsurAnalysis.QuadSurfaces}");
            writer.WriteLine($"- Polygon Surfaces: {result.MsurAnalysis.PolygonSurfaces}");
            writer.WriteLine($"- Valid MSVI Ranges: {result.MsurAnalysis.ValidMsviRanges}");
            writer.WriteLine($"- Invalid MSVI Ranges: {result.MsurAnalysis.InvalidMsviRanges}");
            writer.WriteLine();

            // Coordinate ranges
            writer.WriteLine("## Coordinate Range Analysis");
            writer.WriteLine($"- MSVT Range: ({result.CoordAnalysis.MSVT_Min.X:F1}, {result.CoordAnalysis.MSVT_Min.Y:F1}, {result.CoordAnalysis.MSVT_Min.Z:F1}) to ({result.CoordAnalysis.MSVT_Max.X:F1}, {result.CoordAnalysis.MSVT_Max.Y:F1}, {result.CoordAnalysis.MSVT_Max.Z:F1})");
            writer.WriteLine($"- MSCN Range: ({result.CoordAnalysis.MSCN_Min.X:F1}, {result.CoordAnalysis.MSCN_Min.Y:F1}, {result.CoordAnalysis.MSCN_Min.Z:F1}) to ({result.CoordAnalysis.MSCN_Max.X:F1}, {result.CoordAnalysis.MSCN_Max.Y:F1}, {result.CoordAnalysis.MSCN_Max.Z:F1})");
            writer.WriteLine($"- MSVT/MSCN Overlap: {result.CoordAnalysis.MSVT_MSCN_Overlap:P1}");
            writer.WriteLine();

            // Insights
            writer.WriteLine("## Key Insights");
            foreach (var insight in result.Insights)
            {
                writer.WriteLine($"- {insight}");
            }
        }
    }
} 