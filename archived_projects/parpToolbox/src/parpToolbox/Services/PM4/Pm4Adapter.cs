using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Formats.P4.Chunks.Common;
using System.Numerics;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4
{
    /// <inheritdoc/>
    public sealed partial class Pm4Adapter : IPm4Adapter
{
    /// <summary>Last captured raw MSVT chunk data for analysis purposes.</summary>
    public static byte[]? LastRawMsvtData { get; private set; }
    
    /// <summary>Raw chunk data captured during loading for analysis purposes.</summary>
    private readonly Dictionary<string, byte[]> _capturedRawData = new();
    
    /// <summary>Gets the captured raw chunk data from the last load operation.</summary>
    public IReadOnlyDictionary<string, byte[]> CapturedRawData => _capturedRawData;
    
    /// <inheritdoc/>
    public Pm4Scene Load(string path)
    {
        return Load(path, new Pm4LoadOptions());
    }
    
    /// <inheritdoc/>
    public Pm4Scene Load(string path, Pm4LoadOptions options)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("PM4 path must be provided", nameof(path));
            
        if (options.VerboseLogging)
            ConsoleLogger.WriteLine($"[Pm4Adapter] Loading PM4 file: {path}");
            
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);

        // Simple chunk scan variables
        MspvChunk? mspv = null;
        MsvtChunk? msvt = null;
        MspiChunk? mspi = null;
        MsviChunk? msvi = null;
        MsurChunk? msur = null;
        MslkChunk? mslk = null;
        MprrChunk? mprr = null;
        MprlChunk? mprl = null;
        MscnChunk? mscn = null;
        
        // Clear previous raw data capture
        _capturedRawData.Clear();
        
        // Capture raw MSVT data for analysis
        byte[]? rawMsvtData = null;
        
        var loadedChunks = new List<string>();

        while (br.BaseStream.Position + 8 <= br.BaseStream.Length)
        {
            string sig = FourCc.Read(br);
            uint size = br.ReadUInt32();
            long payloadStart = br.BaseStream.Position;
            byte[] data = br.ReadBytes((int)size);
            
            loadedChunks.Add(sig);
            
            if (options.VerboseLogging)
                ConsoleLogger.WriteLine($"[Pm4Adapter] Loading chunk {sig} ({size} bytes)");
            
            // Capture raw data if requested
            if (options.CaptureRawData)
                _capturedRawData[sig] = (byte[])data.Clone();

            switch (sig)
            {
                case MspvChunk.Signature:
                    mspv ??= new MspvChunk();
                    mspv.LoadBinaryData(data);
                    break;
                case MsvtChunk.Signature:
                    msvt ??= new MsvtChunk();
                    msvt.LoadBinaryData(data);
                    rawMsvtData = data; // Capture for analysis
                    LastRawMsvtData = data; // Store for external access
                    break;
                case MsviChunk.Signature:
                    msvi ??= new MsviChunk();
                    msvi.LoadBinaryData(data);
                    break;
                case MsurChunk.Signature:
                    msur ??= new MsurChunk();
                    msur.LoadBinaryData(data);
                    break;
                case MslkChunk.Signature:
                    mslk ??= new MslkChunk();
                    mslk.LoadBinaryData(data);
                    break;
                case MprrChunk.Signature:
                    mprr ??= new MprrChunk();
                    mprr.LoadBinaryData(data);
                    break;
                case MprlChunk.Signature:
                    mprl ??= new MprlChunk();
                    mprl.LoadBinaryData(data);
                    break;
                case MspiChunk.Signature:
                    mspi ??= new MspiChunk();
                    int vertCount = msvt?.Vertices.Count ?? mspv?.Vertices.Count ?? 0;
                    mspi.LoadBinaryData(data, vertCount);
                    break;
                case MscnChunk.Signature:
                    mscn ??= new MscnChunk();
                    mscn.LoadBinaryData(data);
                    break;
                default:
                    // skip unknown chunks for now
                    break;
            }

            // Ensure we really consumed expected bytes; seek to next aligned chunk.
            br.BaseStream.Position = payloadStart + size;
        }

        if ((mspi == null && msvi == null) || (msvt == null && mspv == null))
            throw new InvalidDataException("PM4 missing required chunks (MSPI + MSPV/MSVT)");

        IReadOnlyList<Vector3> verts = msvt?.Vertices.Count > 0 ? msvt.Vertices : mspv!.Vertices;
        var tris = msvi != null ? msvi.Triangulate().ToList() : mspi!.Triangles;

        // Build surface groups keyed by MSUR.SurfaceKey (Unknown_0x1C) – primary render-object key
        var groupFaces = new Dictionary<uint, List<(int A,int B,int C)>>();
        var groupAttr  = new Dictionary<uint, byte>();

        // Build groups by MSUR.SurfaceKey – confirmed unique render-object ID
        if (msur != null && msvi != null)
        {
            var indices = msvi.Indices;

            foreach (var surf in msur.Entries)
            {
                // Validate index range
                int first = (int)surf.MsviFirstIndex;
                int indexCount = surf.IndexCount;
                if (first < 0 || indexCount < 3 || first + indexCount > indices.Count)
                    continue;

                if (surf.IsM2Bucket) continue; // Ignore overlay-model bucket

                uint groupId = surf.SurfaceKey;

                if (!groupFaces.TryGetValue(groupId, out var list))
                {
                    list = new List<(int,int,int)>();
                    groupFaces[groupId] = list;
                    groupAttr[groupId] = surf.SurfaceAttributeMask;
                }

                int triCount = indexCount / 3;
                for (int i = 0; i < triCount; i++)
                {
                    int baseIdx = first + i * 3;
                    list.Add((indices[baseIdx], indices[baseIdx + 1], indices[baseIdx + 2]));
                }
            }
        }

        var groups = new List<SurfaceGroup>(groupFaces.Count);
        foreach (var kvp in groupFaces)
        {
            uint gid32 = kvp.Key;
            byte gidByte = (byte)(gid32 & 0xFF);
            string gName = $"G{gid32:X8}";
            groups.Add(new SurfaceGroup(gidByte, gName, kvp.Value, groupAttr[gid32]));
        }

        var scene = new Pm4Scene
        {
            Vertices = verts.ToList(),
            Triangles = tris.ToList(),
            Surfaces = (msur?.Entries ?? Array.Empty<MsurChunk.Entry>()).ToList(),
            Spis = mspi != null ? new List<MspiChunk> { mspi } : new List<MspiChunk>(),
            Indices = (msvi?.Indices ?? Array.Empty<int>()).ToList(),
            Groups = groups.ToList(),
            Links = (mslk?.Entries ?? Array.Empty<MslkEntry>()).ToList(),
            Properties = (mprr?.Entries ?? Array.Empty<MprrChunk.Entry>()).ToList(),
            Placements = (mprl?.Entries ?? Array.Empty<MprlChunk.Entry>()).ToList(),
            MscnVertices = mscn?.Vertices?.ToList() ?? new List<Vector3>(),
            ExtraChunks = mscn != null ? new List<IIffChunk>{ mscn } : new List<IIffChunk>()
        };
        
        if (options.VerboseLogging)
        {
            ConsoleLogger.WriteLine($"[Pm4Adapter] Loaded PM4 scene:");
            ConsoleLogger.WriteLine($"  - Chunks: {string.Join(", ", loadedChunks)}");
            ConsoleLogger.WriteLine($"  - Vertices: {scene.Vertices.Count:N0}");
            ConsoleLogger.WriteLine($"  - Triangles: {scene.Triangles.Count:N0}");
            ConsoleLogger.WriteLine($"  - Surface Groups: {scene.Groups.Count:N0}");
        }
        
        // Perform validation if requested
        if (options.ValidateData)
        {
            ValidateSceneData(scene, options.VerboseLogging);
        }
        
        // Perform index pattern analysis if requested
        if (options.AnalyzeIndexPatterns)
        {
            var indexAnalysis = AnalyzeIndexPatterns(scene);
            if (options.VerboseLogging && indexAnalysis.OutOfBoundsCount > 0)
            {
                ConsoleLogger.WriteLine($"[Pm4Adapter] WARNING: {indexAnalysis.OutOfBoundsCount:N0} out-of-bounds vertex references detected ({indexAnalysis.DataLossPercentage:F1}% data loss)");
            }
        }
        
        return scene;

        // local helper
        static List<Vector3> ExtractMscnVertices(object mscn)
        {
            var list = new List<Vector3>();
            var type = mscn.GetType();
            // Try public methods first
            var getCount = type.GetMethod("GetVectorCount");
            var getVector = type.GetMethod("GetVector");
            if (getCount != null && getVector != null)
            {
                int count = (int)getCount.Invoke(mscn, null)!;
                for (int i = 0; i < count; i++)
                {
                    var v4 = (System.Numerics.Vector4)getVector.Invoke(mscn, new object[]{i})!;
                    list.Add(new Vector3(v4.X, v4.Y, v4.Z));
                }
                return list;
            }
            // fallback: attempt to read private _data field (byte[])
            var dataField = type.GetField("_data", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (dataField?.GetValue(mscn) is byte[] raw && raw.Length >=16)
            {
                int count = raw.Length / 16;
                for (int i = 0; i < count; i++)
                {
                    int offset = i*16;
                    float x = BitConverter.ToSingle(raw, offset);
                    float y = BitConverter.ToSingle(raw, offset+4);
                    float z = BitConverter.ToSingle(raw, offset+8);
                    // skip w
                    list.Add(new Vector3(x,y,z));
                }
            }
            return list;
        }
    }
    
    /// <inheritdoc/>
    public Pm4AnalysisReport Analyze(Pm4Scene scene, Pm4AnalysisOptions options)
    {
        var report = new Pm4AnalysisReport();
        var issues = new List<AnalysisIssue>();
        
        ConsoleLogger.WriteLine("=== PM4 Comprehensive Analysis ===");
        
        // Data Structure Analysis
        if (options.AnalyzeDataStructure)
        {
            report.DataStructure = AnalyzeDataStructure(scene, issues);
            ConsoleLogger.WriteLine($"Data Structure: {scene.Vertices.Count:N0} vertices, {scene.Triangles.Count:N0} triangles, {scene.Groups.Count:N0} groups");
        }
        
        // Index Pattern Analysis
        if (options.AnalyzeIndexPatterns)
        {
            report.IndexPatterns = AnalyzeIndexPatterns(scene);
            if (report.IndexPatterns.OutOfBoundsCount > 0)
            {
                ConsoleLogger.WriteLine($"Index Patterns: {report.IndexPatterns.OutOfBoundsCount:N0} out-of-bounds references ({report.IndexPatterns.DataLossPercentage:F1}% data loss)");
                issues.Add(new AnalysisIssue
                {
                    Category = "Index Patterns",
                    Description = $"Detected {report.IndexPatterns.OutOfBoundsCount:N0} out-of-bounds vertex references indicating potential cross-tile dependencies",
                    Severity = "Critical",
                    Details = new Dictionary<string, object>
                    {
                        ["OutOfBoundsCount"] = report.IndexPatterns.OutOfBoundsCount,
                        ["DataLossPercentage"] = report.IndexPatterns.DataLossPercentage,
                        ["MaxVertexIndex"] = report.IndexPatterns.MaxVertexIndex,
                        ["AvailableVertices"] = report.IndexPatterns.AvailableVertices
                    }
                });
            }
        }
        
        // Unknown Field Analysis
        if (options.AnalyzeUnknownFields)
        {
            report.UnknownFields = AnalyzeUnknownFields(scene, issues);
            ConsoleLogger.WriteLine($"Unknown Fields: {report.UnknownFields.FieldDistributions.Count} field distributions analyzed");
        }
        
        // Chunk Relationship Analysis
        if (options.AnalyzeChunkRelationships)
        {
            report.ChunkRelationships = AnalyzeChunkRelationships(scene, issues);
            ConsoleLogger.WriteLine($"Chunk Relationships: {report.ChunkRelationships.MprlMslkMatches} MPRL↔MSLK matches, {report.ChunkRelationships.MprrSentinelCount} MPRR sentinels");
        }
        
        // Generate CSV reports if requested
        if (options.GenerateCsvReports && !string.IsNullOrEmpty(options.OutputDirectory))
        {
            GenerateCsvReports(scene, report, options.OutputDirectory);
        }
        
        report.Issues = issues;
        report.Summary = GenerateAnalysisSummary(report);
        
        ConsoleLogger.WriteLine($"Analysis complete. {issues.Count} issues found.");
        
        return report;
    }
    
    #region Analysis Helper Methods
    
    /// <summary>
    /// Validates PM4 scene data for common issues and anomalies.
    /// </summary>
    private void ValidateSceneData(Pm4Scene scene, bool verbose)
    {
        if (verbose)
            ConsoleLogger.WriteLine("[Pm4Adapter] Validating scene data...");
            
        // Check for basic data consistency
        if (scene.Vertices.Count == 0)
            throw new InvalidDataException("PM4 scene contains no vertices");
            
        if (scene.Triangles.Count == 0 && scene.Indices.Count == 0)
            throw new InvalidDataException("PM4 scene contains no triangles or indices");
            
        // Validate triangle indices are within vertex bounds
        int maxValidIndex = scene.Vertices.Count - 1;
        int invalidTriangles = 0;
        
        foreach (var (a, b, c) in scene.Triangles)
        {
            if (a < 0 || a > maxValidIndex || b < 0 || b > maxValidIndex || c < 0 || c > maxValidIndex)
                invalidTriangles++;
        }
        
        if (verbose && invalidTriangles > 0)
            ConsoleLogger.WriteLine($"[Pm4Adapter] WARNING: {invalidTriangles} triangles have invalid vertex indices");
    }
    
    /// <summary>
    /// Analyzes data structure and provides basic statistics.
    /// </summary>
    private DataStructureAnalysis AnalyzeDataStructure(Pm4Scene scene, List<AnalysisIssue> issues)
    {
        var analysis = new DataStructureAnalysis
        {
            VertexCount = scene.Vertices.Count,
            IndexCount = scene.Indices.Count,
            TriangleCount = scene.Triangles.Count,
            SurfaceCount = scene.Surfaces.Count,
            LinkCount = scene.Links.Count,
            PlacementCount = scene.Placements.Count,
            PropertyCount = scene.Properties.Count,
            ChunkCounts = new Dictionary<string, int>
            {
                ["MSPV/MSVT"] = scene.Vertices.Count > 0 ? 1 : 0,
                ["MSVI/MSPI"] = scene.Indices.Count > 0 || scene.Triangles.Count > 0 ? 1 : 0,
                ["MSUR"] = scene.Surfaces.Count > 0 ? 1 : 0,
                ["MSLK"] = scene.Links.Count > 0 ? 1 : 0,
                ["MPRL"] = scene.Placements.Count > 0 ? 1 : 0,
                ["MPRR"] = scene.Properties.Count > 0 ? 1 : 0,
                ["MSCN"] = scene.ExtraChunks.Count > 0 ? 1 : 0
            }
        };
        
        // Check for potential issues
        if (analysis.VertexCount > 100000)
        {
            issues.Add(new AnalysisIssue
            {
                Category = "Data Structure",
                Description = $"Large vertex count ({analysis.VertexCount:N0}) may indicate cross-tile references",
                Severity = "Warning"
            });
        }
        
        return analysis;
    }
    
    /// <summary>
    /// Analyzes index patterns to detect out-of-bounds references and potential encoding issues.
    /// Integrates functionality from Pm4IndexPatternAnalyzer.
    /// </summary>
    private IndexPatternAnalysis AnalyzeIndexPatterns(Pm4Scene scene)
    {
        var analysis = new IndexPatternAnalysis
        {
            AvailableVertices = scene.Vertices.Count
        };
        
        var allIndices = new List<int>();
        
        // Collect all indices from triangles and raw indices
        foreach (var (a, b, c) in scene.Triangles)
        {
            allIndices.AddRange(new[] { a, b, c });
        }
        
        allIndices.AddRange(scene.Indices);
        
        if (allIndices.Count == 0)
            return analysis;
            
        analysis.MaxVertexIndex = allIndices.Max();
        
        // Count out-of-bounds references
        int maxValidIndex = scene.Vertices.Count - 1;
        var outOfBoundsIndices = allIndices.Where(i => i < 0 || i > maxValidIndex).ToList();
        analysis.OutOfBoundsCount = outOfBoundsIndices.Count;
        
        if (allIndices.Count > 0)
            analysis.DataLossPercentage = (double)analysis.OutOfBoundsCount / allIndices.Count * 100.0;
            
        // Collect suspicious indices (just outside valid range)
        analysis.SuspiciousIndices = outOfBoundsIndices
            .Where(i => i >= 0 && i <= maxValidIndex + 100000) // Within reasonable range
            .Select(i => (uint)i)
            .Distinct()
            .OrderBy(i => i)
            .Take(100)
            .ToList();
            
        // Analyze high/low pair patterns in unknown fields
        analysis.HighLowPairs = AnalyzeHighLowPairs(scene);
        
        return analysis;
    }
    
    /// <summary>
    /// Analyzes high/low pair patterns in unknown fields that might encode 32-bit indices.
    /// </summary>
    private Dictionary<string, List<uint>> AnalyzeHighLowPairs(Pm4Scene scene)
    {
        var highLowPairs = new Dictionary<string, List<uint>>();
        
        // Analyze MPRL unknown fields for high/low patterns
        var mprlHighLowCandidates = new List<uint>();
        foreach (var placement in scene.Placements)
        {
            // Check if Unknown0/Unknown2 could be high/low pair (Unknown2 is signed, cast to ushort)
            uint combined = ((uint)(ushort)placement.Unknown2 << 16) | (uint)placement.Unknown0;
            if (combined > scene.Vertices.Count && combined < scene.Vertices.Count + 100000)
                mprlHighLowCandidates.Add(combined);
        }
        if (mprlHighLowCandidates.Count > 0)
            highLowPairs["MPRL.Unknown0+Unknown2"] = mprlHighLowCandidates.Take(50).ToList();
            
        // Analyze MSLK unknown fields for high/low patterns
        var mslkHighLowCandidates = new List<uint>();
        foreach (var link in scene.Links)
        {
            // Check possible high/low pair between ParentIndex (high) and MspiFirstIndex (low)
            if (link.ParentIndex > 0 && link.MspiFirstIndex > 0)
            {
                uint combined = ((uint)link.ParentIndex << 16) | (uint)(ushort)link.MspiFirstIndex;
                if (combined > scene.Vertices.Count && combined < scene.Vertices.Count + 100000)
                    mslkHighLowCandidates.Add(combined);
            }
        }
        if (mslkHighLowCandidates.Count > 0)
            highLowPairs["MSLK.ParentIndex+MspiFirstIndex"] = mslkHighLowCandidates.Take(50).ToList();
            
        return highLowPairs;
    }
    
    /// <summary>
    /// Analyzes unknown fields and their distributions.
    /// Integrates functionality from Pm4UnknownFieldAnalyzer.
    /// </summary>
    private UnknownFieldAnalysis AnalyzeUnknownFields(Pm4Scene scene, List<AnalysisIssue> issues)
    {
        var analysis = new UnknownFieldAnalysis();
        
        // Analyze MPRL field distributions
        if (scene.Placements.Count > 0)
        {
            analysis.FieldDistributions["MPRL.Unknown0"] = scene.Placements
                .GroupBy(p => (uint)p.Unknown0)
                .ToDictionary(g => g.Key, g => g.Count());
                
            analysis.FieldDistributions["MPRL.Unknown2"] = scene.Placements
                .GroupBy(p => (uint)(ushort)p.Unknown2)
                .ToDictionary(g => g.Key, g => g.Count());
                
            analysis.FieldDistributions["MPRL.Unknown4"] = scene.Placements
                .GroupBy(p => (uint)p.Unknown4)
                .ToDictionary(g => g.Key, g => g.Count());
        }
        
        // Analyze MSLK field distributions
        if (scene.Links.Count > 0)
        {
            analysis.FieldDistributions["MSLK.ParentIndex"] = scene.Links
                .GroupBy(l => (uint)l.ParentIndex)
                .ToDictionary(g => g.Key, g => g.Count());
                
            analysis.FieldDistributions["MSLK.MspiFirstIndex"] = scene.Links
                .GroupBy(l => (uint)(ushort)l.MspiFirstIndex)
                .ToDictionary(g => g.Key, g => g.Count());
        }
        
        // Find correlations between fields
        analysis.Correlations = FindFieldCorrelations(scene);
        
        return analysis;
    }
    
    /// <summary>
    /// Finds correlations between different unknown fields.
    /// </summary>
    private List<FieldCorrelation> FindFieldCorrelations(Pm4Scene scene)
    {
        var correlations = new List<FieldCorrelation>();
        
        // Check MPRL.Unknown4 vs MSLK.ParentIndex correlation (known from memory bank)
        if (scene.Placements.Count > 0 && scene.Links.Count > 0)
        {
            var mprlUnknown4Values = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
            var mslkParentIndexValues = scene.Links.Select(l => (uint)l.ParentIndex).ToHashSet();
            
            var matches = mprlUnknown4Values.Intersect(mslkParentIndexValues).Count();
            if (matches > 0)
            {
                correlations.Add(new FieldCorrelation
                {
                    Field1 = "MPRL.Unknown4",
                    Field2 = "MSLK.ParentIndex",
                    CorrelationStrength = (double)matches / Math.Max(mprlUnknown4Values.Count, mslkParentIndexValues.Count),
                    MatchCount = matches
                });
            }
        }
        
        return correlations;
    }
    
    /// <summary>
    /// Analyzes chunk relationships and patterns.
    /// Integrates functionality from Pm4DataAnalyzer.
    /// </summary>
    private ChunkRelationshipAnalysis AnalyzeChunkRelationships(Pm4Scene scene, List<AnalysisIssue> issues)
    {
        var analysis = new ChunkRelationshipAnalysis();
        
        // Analyze MPRL↔MSLK relationships (confirmed from memory bank)
        if (scene.Placements.Count > 0 && scene.Links.Count > 0)
        {
            var mprlUnknown4Values = scene.Placements.Select(p => (uint)p.Unknown4).ToHashSet();
            var mslkParentIndexValues = scene.Links.Select(l => (uint)l.ParentIndex).ToHashSet();
            analysis.MprlMslkMatches = mprlUnknown4Values.Intersect(mslkParentIndexValues).Count();
            
            if (analysis.MprlMslkMatches > 0)
            {
                analysis.RelationshipInsights.Add($"Found {analysis.MprlMslkMatches} confirmed MPRL.Unknown4 ↔ MSLK.ParentIndex matches");
            }
        }
        
        // Analyze MPRR sentinel patterns (Value1 = 65535)
        if (scene.Properties.Count > 0)
        {
            analysis.MprrSentinelCount = scene.Properties.Count(p => p.Value1 == 65535);
            if (analysis.MprrSentinelCount > 0)
            {
                analysis.RelationshipInsights.Add($"Found {analysis.MprrSentinelCount} MPRR sentinel markers (Value1=65535)");
            }
        }
        
        // Analyze MSUR surface group distribution
        if (scene.Surfaces.Count > 0)
        {
            analysis.SurfaceGroupDistribution = scene.Surfaces
                .GroupBy(s => s.SurfaceKey)
                .ToDictionary(g => g.Key, g => g.Count());
                
            var groupCount = analysis.SurfaceGroupDistribution.Count;
            analysis.RelationshipInsights.Add($"Found {groupCount} unique surface groups (MSUR.SurfaceKey)");
            
            // Check for M2 bucket pattern
            var m2BucketCount = scene.Surfaces.Count(s => s.IsM2Bucket);
            if (m2BucketCount > 0)
            {
                analysis.RelationshipInsights.Add($"Found {m2BucketCount} M2 bucket surfaces (overlay models)");
            }
        }
        
        return analysis;
    }
    
    /// <summary>
    /// Generates CSV reports for detailed analysis.
    /// </summary>
    private void GenerateCsvReports(Pm4Scene scene, Pm4AnalysisReport report, string outputDirectory)
    {
        Directory.CreateDirectory(outputDirectory);
        
        // Generate data structure report
        var dataStructurePath = Path.Combine(outputDirectory, "pm4_data_structure.csv");
        using (var writer = new StreamWriter(dataStructurePath))
        {
            writer.WriteLine("Category,Key,Value,Description");
            writer.WriteLine($"Vertices,Count,{scene.Vertices.Count},Total vertex count");
            writer.WriteLine($"Triangles,Count,{scene.Triangles.Count},Total triangle count");
            writer.WriteLine($"Surfaces,Count,{scene.Surfaces.Count},MSUR surface count");
            writer.WriteLine($"Links,Count,{scene.Links.Count},MSLK link count");
            writer.WriteLine($"Placements,Count,{scene.Placements.Count},MPRL placement count");
            writer.WriteLine($"Properties,Count,{scene.Properties.Count},MPRR property count");
        }
        
        // Generate index pattern report if available
        if (report.IndexPatterns != null)
        {
            var indexPatternPath = Path.Combine(outputDirectory, "pm4_index_patterns.csv");
            using (var writer = new StreamWriter(indexPatternPath))
            {
                writer.WriteLine("Category,Value,Description");
                writer.WriteLine($"MaxVertexIndex,{report.IndexPatterns.MaxVertexIndex},Highest vertex index found");
                writer.WriteLine($"AvailableVertices,{report.IndexPatterns.AvailableVertices},Available vertices in scene");
                writer.WriteLine($"OutOfBoundsCount,{report.IndexPatterns.OutOfBoundsCount},Out-of-bounds vertex references");
                writer.WriteLine($"DataLossPercentage,{report.IndexPatterns.DataLossPercentage:F2},Percentage of data loss");
            }
        }
        
        ConsoleLogger.WriteLine($"[Pm4Adapter] Generated CSV reports in: {outputDirectory}");
    }
    
    /// <summary>
    /// Generates a comprehensive analysis summary.
    /// </summary>
    private string GenerateAnalysisSummary(Pm4AnalysisReport report)
    {
        var summary = new List<string>();
        
        if (report.DataStructure != null)
        {
            summary.Add($"Data Structure: {report.DataStructure.VertexCount:N0} vertices, {report.DataStructure.TriangleCount:N0} triangles");
        }
        
        if (report.IndexPatterns != null && report.IndexPatterns.OutOfBoundsCount > 0)
        {
            summary.Add($"Index Issues: {report.IndexPatterns.OutOfBoundsCount:N0} out-of-bounds references ({report.IndexPatterns.DataLossPercentage:F1}% data loss)");
        }
        
        if (report.ChunkRelationships != null)
        {
            if (report.ChunkRelationships.MprlMslkMatches > 0)
                summary.Add($"Relationships: {report.ChunkRelationships.MprlMslkMatches} MPRL↔MSLK matches confirmed");
        }
        
        if (report.Issues.Any(i => i.Severity == "Critical"))
        {
            summary.Add($"Critical Issues: {report.Issues.Count(i => i.Severity == "Critical")} found");
        }
        
        return summary.Count > 0 ? string.Join("; ", summary) : "Analysis completed successfully with no major issues.";
    }
    
    #endregion
}
}
