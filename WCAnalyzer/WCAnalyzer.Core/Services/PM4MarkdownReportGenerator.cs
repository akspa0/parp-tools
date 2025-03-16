using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Models.PM4;
using WCAnalyzer.Core.Models.PM4.Chunks;

namespace WCAnalyzer.Core.Services
{
    /// <summary>
    /// Generates comprehensive Markdown reports for PM4 files with analysis of various data correlations.
    /// </summary>
    public class PM4MarkdownReportGenerator
    {
        private readonly ILogger? _logger;

        public PM4MarkdownReportGenerator(ILogger? logger = null)
        {
            _logger = logger;
        }

        /// <summary>
        /// Generates a comprehensive Markdown report for a PM4 file, with special focus on analyzing 
        /// the relationships between position data, special values, and other chunks.
        /// </summary>
        /// <param name="file">The PM4 file to analyze</param>
        /// <param name="outputFilePath">Path where the Markdown report should be saved</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task GenerateComprehensiveReportAsync(PM4File file, string outputFilePath)
        {
            if (file == null)
                throw new ArgumentNullException(nameof(file));
            if (string.IsNullOrEmpty(outputFilePath))
                throw new ArgumentException("Output file path cannot be null or empty", nameof(outputFilePath));

            try
            {
                _logger?.LogInformation("Generating comprehensive Markdown report for {FileName}", file.FileName ?? "unknown");

                var sb = new StringBuilder();
                
                // File header
                sb.AppendLine($"# PM4 File Analysis: {file.FileName ?? "Unknown"}");
                sb.AppendLine($"*Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}*");
                sb.AppendLine();
                
                // Table of contents
                sb.AppendLine("## Table of Contents");
                sb.AppendLine("1. [File Overview](#file-overview)");
                sb.AppendLine("2. [Chunk Summary](#chunk-summary)");
                sb.AppendLine("3. [Position Data Analysis](#position-data-analysis)");
                sb.AppendLine("   - [Special Value Analysis](#special-value-analysis)");
                sb.AppendLine("   - [Z Value Categorization](#z-value-categorization)");
                sb.AppendLine("4. [MSLK Links Analysis](#mslk-links-analysis)");
                sb.AppendLine("5. [Special Value Correlations](#special-value-correlations)");
                sb.AppendLine("6. [3D Model Data](#3d-model-data)");
                sb.AppendLine("7. [Integrated Analysis](#integrated-analysis)");
                sb.AppendLine();
                
                // File overview
                sb.AppendLine("## File Overview <a name=\"file-overview\"></a>");
                sb.AppendLine($"- **Filename**: {file.FileName ?? "Unknown"}");
                sb.AppendLine($"- **File Size**: {file.FileSize:N0} bytes");
                sb.AppendLine($"- **Version**: {file.Version?.Version ?? 0}");
                sb.AppendLine();
                
                // Chunk summary
                sb.AppendLine("## Chunk Summary <a name=\"chunk-summary\"></a>");
                sb.AppendLine("| Chunk | Description | Present | Size (bytes) |");
                sb.AppendLine("|-------|-------------|---------|--------------|");
                AppendChunkInfo(sb, "MSHD", "Shadow Data", file.ShadowDataChunk);
                AppendChunkInfo(sb, "MSPV", "Vertex Positions", file.VertexPositionsChunk);
                AppendChunkInfo(sb, "MSPI", "Vertex Indices", file.VertexIndicesChunk);
                AppendChunkInfo(sb, "MSCN", "Normal Coordinates", file.NormalCoordinatesChunk);
                AppendChunkInfo(sb, "MSLK", "Links", file.LinksChunk);
                AppendChunkInfo(sb, "MSVT", "Vertex Data", file.VertexDataChunk);
                AppendChunkInfo(sb, "MSVI", "Vertex Info", file.VertexInfoChunk);
                AppendChunkInfo(sb, "MSUR", "Surface Data", file.SurfaceDataChunk);
                AppendChunkInfo(sb, "MPRL", "Position Data", file.PositionDataChunk);
                sb.AppendLine();
                
                // Position data analysis
                if (file.PositionDataChunk != null && file.PositionDataChunk.Entries.Count > 0)
                {
                    await AppendPositionDataAnalysis(sb, file);
                }
                else
                {
                    sb.AppendLine("## Position Data Analysis <a name=\"position-data-analysis\"></a>");
                    sb.AppendLine("*No position data available in this file.*");
                    sb.AppendLine();
                }
                
                // MSLK links analysis
                if (file.LinksChunk != null && file.LinksChunk.Links.Count > 0)
                {
                    await AppendMslkLinksAnalysis(sb, file);
                }
                else
                {
                    sb.AppendLine("## MSLK Links Analysis <a name=\"mslk-links-analysis\"></a>");
                    sb.AppendLine("*No MSLK links data available in this file.*");
                    sb.AppendLine();
                }
                
                // Special value correlations
                if (file.PositionDataChunk != null && file.PositionDataChunk.Entries.Count > 0)
                {
                    await AppendSpecialValueCorrelations(sb, file);
                }
                else
                {
                    sb.AppendLine("## Special Value Correlations <a name=\"special-value-correlations\"></a>");
                    sb.AppendLine("*No position data available for correlation analysis.*");
                    sb.AppendLine();
                }
                
                // 3D model data
                if (file.VertexPositionsChunk != null && file.VertexPositionsChunk.Vertices.Count > 0)
                {
                    await Append3DModelData(sb, file);
                }
                else
                {
                    sb.AppendLine("## 3D Model Data <a name=\"3d-model-data\"></a>");
                    sb.AppendLine("*No 3D model data available in this file.*");
                    sb.AppendLine();
                }
                
                // Integrated analysis
                if (file.PositionDataChunk != null && file.VertexPositionsChunk != null)
                {
                    await AppendIntegratedAnalysis(sb, file);
                }
                else
                {
                    sb.AppendLine("## Integrated Analysis <a name=\"integrated-analysis\"></a>");
                    sb.AppendLine("*Not enough data available for integrated analysis.*");
                    sb.AppendLine();
                }
                
                // Write the report to file
                await File.WriteAllTextAsync(outputFilePath, sb.ToString());
                _logger?.LogInformation("Comprehensive Markdown report generated successfully: {FilePath}", outputFilePath);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error generating comprehensive Markdown report: {FilePath}", outputFilePath);
                throw;
            }
        }
        
        /// <summary>
        /// Appends chunk information to the report
        /// </summary>
        private void AppendChunkInfo(StringBuilder sb, string signature, string description, PM4Chunk? chunk)
        {
            bool present = chunk != null;
            int size = chunk?.Data?.Length ?? 0;
            sb.AppendLine($"| {signature} | {description} | {(present ? "Yes" : "No")} | {size:N0} |");
        }
        
        /// <summary>
        /// Analyzes position data and appends findings to the report
        /// </summary>
        private async Task AppendPositionDataAnalysis(StringBuilder sb, PM4File file)
        {
            var entries = file.PositionDataChunk?.Entries;
            if (entries == null || entries.Count == 0)
                return;
                
            sb.AppendLine("## Position Data Analysis <a name=\"position-data-analysis\"></a>");
            sb.AppendLine($"The file contains {entries.Count} position data entries.");
            
            // Count special vs regular entries
            int specialEntries = entries.Count(e => e.IsSpecialEntry);
            int regularEntries = entries.Count - specialEntries;
            
            sb.AppendLine($"- Special entries: {specialEntries}");
            sb.AppendLine($"- Regular position entries: {regularEntries}");
            sb.AppendLine();
            
            // Analyze special values
            sb.AppendLine("### Special Value Analysis <a name=\"special-value-analysis\"></a>");
            
            var specialValueEntries = entries.Where(e => e.IsSpecialEntry).ToList();
            if (specialValueEntries.Count > 0)
            {
                // Group special values to find patterns
                var specialValueGroups = specialValueEntries
                    .GroupBy(e => e.SpecialValue)
                    .OrderByDescending(g => g.Count())
                    .ToList();
                
                sb.AppendLine($"Found {specialValueGroups.Count} unique special values across {specialEntries} special entries.");
                sb.AppendLine();
                
                sb.AppendLine("#### Top Special Values by Frequency");
                sb.AppendLine("| Special Value (Hex) | Special Value (Dec) | As Float | Count | Percentage |");
                sb.AppendLine("|---------------------|---------------------|----------|-------|------------|");
                
                foreach (var group in specialValueGroups.Take(10))
                {
                    float asFloat = BitConverter.Int32BitsToSingle(group.Key);
                    double percentage = (double)group.Count() / specialEntries * 100;
                    
                    sb.AppendLine($"| 0x{group.Key:X8} | {group.Key} | {asFloat:F6} | {group.Count()} | {percentage:F2}% |");
                }
                
                if (specialValueGroups.Count > 10)
                {
                    sb.AppendLine("*Table limited to top 10 values. See full data in the CSV export.*");
                }
                
                sb.AppendLine();
                
                // Analyze special value distribution
                sb.AppendLine("#### Special Value Distribution");
                sb.AppendLine("The special values appear to represent *height/elevation data* when interpreted as floating-point values.");
                sb.AppendLine("Each special entry is typically followed by a position entry, forming a pair that defines a complete 3D point.");
                sb.AppendLine();
            }
            
            // Analyze Z values in position entries
            sb.AppendLine("### Z Value Categorization <a name=\"z-value-categorization\"></a>");
            
            var positionEntries = entries.Where(e => !e.IsSpecialEntry).ToList();
            if (positionEntries.Count > 0)
            {
                // Group Z values to identify categories
                var zValueGroups = positionEntries
                    .GroupBy(e => e.CoordinateZ)
                    .OrderByDescending(g => g.Count())
                    .ToList();
                
                sb.AppendLine($"Found {zValueGroups.Count} unique Z values across {positionEntries.Count} position entries.");
                sb.AppendLine();
                
                if (zValueGroups.Count <= 10)
                {
                    sb.AppendLine("#### Z Value Distribution");
                    sb.AppendLine("| Z Value | Count | Percentage | Likely Meaning |");
                    sb.AppendLine("|---------|-------|------------|----------------|");
                    
                    foreach (var group in zValueGroups)
                    {
                        double percentage = (double)group.Count() / positionEntries.Count * 100;
                        string meaning = group.Key switch
                        {
                            0.0f => "Standard waypoint",
                            2.0f => "Special waypoint (interaction point?)",
                            _ => "Unknown"
                        };
                        
                        sb.AppendLine($"| {group.Key:F6} | {group.Count()} | {percentage:F2}% | {meaning} |");
                    }
                    
                    sb.AppendLine();
                    sb.AppendLine("The Z values appear to function as category flags rather than actual vertical position data.");
                    sb.AppendLine("The actual height/elevation (Y coordinate) is derived from the special value in the corresponding special entry.");
                }
                else
                {
                    // If there are too many unique Z values, they might be actual coordinates rather than flags
                    sb.AppendLine("The Z values in this file appear to be actual coordinates rather than category flags.");
                    
                    // Show range of Z values
                    float minZ = positionEntries.Min(e => e.CoordinateZ);
                    float maxZ = positionEntries.Max(e => e.CoordinateZ);
                    sb.AppendLine($"Z value range: {minZ:F6} to {maxZ:F6}");
                    sb.AppendLine();
                }
            }
            
            // Sample data table
            sb.AppendLine("### Sample Position Data Entries");
            sb.AppendLine("| Entry # | Type | X | Y | Z | Special Value (Hex) | Special Value (Float) |");
            sb.AppendLine("|---------|------|---|---|---|---------------------|----------------------|");
            
            // Show the first 5 pairs of entries (special + position)
            for (int i = 0; i < Math.Min(entries.Count - 1, 10); i += 2)
            {
                if (i + 1 < entries.Count && entries[i].IsSpecialEntry && !entries[i + 1].IsSpecialEntry)
                {
                    var specialEntry = entries[i];
                    var posEntry = entries[i + 1];
                    float asFloat = BitConverter.Int32BitsToSingle(specialEntry.SpecialValue);
                    
                    sb.AppendLine($"| {specialEntry.Index} | Special | - | {specialEntry.CoordinateY:F6} | - | 0x{specialEntry.SpecialValue:X8} | {asFloat:F6} |");
                    sb.AppendLine($"| {posEntry.Index} | Position | {posEntry.CoordinateX:F6} | {posEntry.CoordinateY:F6} | {posEntry.CoordinateZ:F6} | - | - |");
                }
            }
            
            sb.AppendLine();
        }
        
        /// <summary>
        /// Analyzes MSLK links and appends findings to the report
        /// </summary>
        private async Task AppendMslkLinksAnalysis(StringBuilder sb, PM4File file)
        {
            var links = file.LinksChunk?.Links;
            if (links == null || links.Count == 0)
                return;
                
            sb.AppendLine("## MSLK Links Analysis <a name=\"mslk-links-analysis\"></a>");
            sb.AppendLine($"The file contains {links.Count} link entries connecting vertices.");
            sb.AppendLine();
            
            // Find vertex range
            uint minSourceIdx = links.Min(l => l.SourceIndex);
            uint maxSourceIdx = links.Max(l => l.SourceIndex);
            uint minTargetIdx = links.Min(l => l.TargetIndex);
            uint maxTargetIdx = links.Max(l => l.TargetIndex);
            
            sb.AppendLine($"- Source vertex index range: {minSourceIdx} to {maxSourceIdx}");
            sb.AppendLine($"- Target vertex index range: {minTargetIdx} to {maxTargetIdx}");
            sb.AppendLine();
            
            // Check if vertex counts match vertex position data
            int vertexCount = file.VertexPositionsChunk?.Vertices.Count ?? 0;
            sb.AppendLine($"- Total vertices in MSPV chunk: {vertexCount}");
            
            if (vertexCount > 0)
            {
                bool validIndices = maxSourceIdx < vertexCount && maxTargetIdx < vertexCount;
                sb.AppendLine($"- Link indices {(validIndices ? "are valid" : "exceed vertex count")} relative to MSPV chunk");
            }
            
            sb.AppendLine();
            
            // Analyze link structure
            var sourceGrouped = links.GroupBy(l => l.SourceIndex).OrderByDescending(g => g.Count()).ToList();
            var targetGrouped = links.GroupBy(l => l.TargetIndex).OrderByDescending(g => g.Count()).ToList();
            
            sb.AppendLine("### Link Structure Analysis");
            sb.AppendLine($"- Average outgoing links per vertex: {(double)links.Count / sourceGrouped.Count:F2}");
            sb.AppendLine($"- Average incoming links per vertex: {(double)links.Count / targetGrouped.Count:F2}");
            sb.AppendLine($"- Vertices with outgoing links: {sourceGrouped.Count}");
            sb.AppendLine($"- Vertices with incoming links: {targetGrouped.Count}");
            sb.AppendLine();
            
            // Sample link data
            sb.AppendLine("### Sample Link Entries");
            sb.AppendLine("| Link # | Source Vertex | Target Vertex |");
            sb.AppendLine("|--------|---------------|---------------|");
            
            for (int i = 0; i < Math.Min(links.Count, 10); i++)
            {
                sb.AppendLine($"| {i} | {links[i].SourceIndex} | {links[i].TargetIndex} |");
            }
            
            if (links.Count > 10)
            {
                sb.AppendLine("*Table limited to first 10 links. See full data in the export files.*");
            }
            
            sb.AppendLine();
        }
        
        /// <summary>
        /// Analyzes correlations between special values and other data, appending findings to the report
        /// </summary>
        private async Task AppendSpecialValueCorrelations(StringBuilder sb, PM4File file)
        {
            var entries = file.PositionDataChunk?.Entries;
            var links = file.LinksChunk?.Links;
            
            if (entries == null || entries.Count == 0)
                return;
                
            sb.AppendLine("## Special Value Correlations <a name=\"special-value-correlations\"></a>");
            
            // Extract pairs of special entries and their following position entries
            var pairs = new List<(ServerPositionData Special, ServerPositionData Position)>();
            
            for (int i = 0; i < entries.Count - 1; i++)
            {
                if (entries[i].IsSpecialEntry && !entries[i + 1].IsSpecialEntry)
                {
                    pairs.Add((entries[i], entries[i + 1]));
                }
            }
            
            sb.AppendLine($"Found {pairs.Count} special/position entry pairs for analysis.");
            sb.AppendLine();
            
            if (pairs.Count == 0)
                return;
                
            // Group by special value to see if there are clusters
            var clusters = pairs
                .GroupBy(p => p.Special.SpecialValue)
                .OrderByDescending(g => g.Count())
                .ToList();
                
            sb.AppendLine("### Special Value Clusters");
            sb.AppendLine($"Found {clusters.Count} distinct special values forming clusters of position points.");
            sb.AppendLine();
            
            sb.AppendLine("| Special Value (Hex) | As Float | Points Count | Average X | Average Z | Z Value |");
            sb.AppendLine("|---------------------|----------|--------------|-----------|-----------|---------|");
            
            foreach (var cluster in clusters.Take(15))
            {
                float asFloat = BitConverter.Int32BitsToSingle(cluster.Key);
                double avgX = cluster.Average(p => p.Position.CoordinateX);
                double avgZ = cluster.Average(p => p.Position.CoordinateZ);
                
                // Check if Z values are consistent within cluster
                var zValues = cluster.Select(p => p.Position.CoordinateZ).Distinct().ToList();
                string zValueStr = zValues.Count == 1 
                    ? $"{zValues[0]:F1}" 
                    : $"Mixed ({string.Join(", ", zValues.Take(3).Select(z => z.ToString("F1")))}{(zValues.Count > 3 ? "..." : "")})";
                
                sb.AppendLine($"| 0x{cluster.Key:X8} | {asFloat:F6} | {cluster.Count()} | {avgX:F2} | {avgZ:F2} | {zValueStr} |");
            }
            
            if (clusters.Count > 15)
            {
                sb.AppendLine("*Table limited to top 15 clusters. See full data in the export files.*");
            }
            
            sb.AppendLine();
            
            // Check correlations with MSLK links if available
            if (links != null && links.Count > 0 && file.VertexPositionsChunk != null)
            {
                sb.AppendLine("### MSLK Link Correlations");
                sb.AppendLine("This section analyzes whether special values might correlate with MSLK link structures.");
                sb.AppendLine();
                
                // Check if special values might correspond to vertex indices
                var vertices = file.VertexPositionsChunk.Vertices;
                var specialValues = pairs.Select(p => p.Special.SpecialValue).Distinct().ToList();
                
                // Count special values that could be valid vertex indices
                int possibleVertexIndices = specialValues.Count(sv => sv >= 0 && sv < vertices.Count);
                double percentage = (double)possibleVertexIndices / specialValues.Count * 100;
                
                sb.AppendLine($"- Special values that could be valid vertex indices: {possibleVertexIndices} ({percentage:F2}%)");
                
                // Check if there's correlation between special values and link indices
                bool foundCorrelation = false;
                
                // Look at the top few special values and see if they appear in link data
                foreach (var topCluster in clusters.Take(5))
                {
                    uint specialValue = (uint)topCluster.Key;
                    
                    int sourceMatches = links.Count(l => l.SourceIndex == specialValue);
                    int targetMatches = links.Count(l => l.TargetIndex == specialValue);
                    
                    if (sourceMatches > 0 || targetMatches > 0)
                    {
                        foundCorrelation = true;
                        sb.AppendLine($"- Special value 0x{specialValue:X8} appears as source in {sourceMatches} links and target in {targetMatches} links");
                    }
                }
                
                if (!foundCorrelation)
                {
                    sb.AppendLine("- No direct correlation found between special values and link indices");
                }
                
                sb.AppendLine();
            }
            
            // Analysis conclusion
            sb.AppendLine("### Analysis Findings");
            sb.AppendLine("Based on the data analysis, the Special Values appear to primarily represent:");
            sb.AppendLine();
            sb.AppendLine("1. **Height/Elevation Data**: When interpreted as floating-point values");
            sb.AppendLine("2. **Possibly Grouping Information**: Some clusters of special values might represent related waypoints");
            sb.AppendLine("3. **Potential Reference Values**: Some values may reference model vertices or other data structures");
            sb.AppendLine();
            
            // Check if special values appear to be elevations
            float minSpecialFloat = pairs.Min(p => BitConverter.Int32BitsToSingle(p.Special.SpecialValue));
            float maxSpecialFloat = pairs.Max(p => BitConverter.Int32BitsToSingle(p.Special.SpecialValue));
            
            sb.AppendLine($"Special values as floats range from {minSpecialFloat:F2} to {maxSpecialFloat:F2}, which is consistent with");
            sb.AppendLine("elevation/height values in a 3D world space, supporting the interpretation as Y-coordinate values.");
            sb.AppendLine();
        }
        
        /// <summary>
        /// Analyzes 3D model data and appends findings to the report
        /// </summary>
        private async Task Append3DModelData(StringBuilder sb, PM4File file)
        {
            var vertices = file.VertexPositionsChunk?.Vertices;
            var indices = file.VertexIndicesChunk?.Indices;
            var normals = file.NormalCoordinatesChunk?.Normals;
            
            if (vertices == null || vertices.Count == 0)
                return;
                
            sb.AppendLine("## 3D Model Data <a name=\"3d-model-data\"></a>");
            sb.AppendLine($"The file contains 3D model data with {vertices.Count:N0} vertices.");
            sb.AppendLine();
            
            // Vertex analysis
            float minX = vertices.Min(v => v.X);
            float minY = vertices.Min(v => v.Y);
            float minZ = vertices.Min(v => v.Z);
            float maxX = vertices.Max(v => v.X);
            float maxY = vertices.Max(v => v.Y);
            float maxZ = vertices.Max(v => v.Z);
            
            sb.AppendLine("### Vertex Data Analysis");
            sb.AppendLine($"- Vertex count: {vertices.Count:N0}");
            sb.AppendLine($"- Bounding box: Min({minX:F2}, {minY:F2}, {minZ:F2}) to Max({maxX:F2}, {maxY:F2}, {maxZ:F2})");
            sb.AppendLine($"- Size: Width={maxX-minX:F2}, Height={maxY-minY:F2}, Depth={maxZ-minZ:F2}");
            sb.AppendLine();
            
            // Triangle analysis
            if (indices != null && indices.Count > 0)
            {
                int triangleCount = indices.Count / 3;
                sb.AppendLine("### Triangle Data Analysis");
                sb.AppendLine($"- Index count: {indices.Count:N0}");
                sb.AppendLine($"- Triangle count: {triangleCount:N0}");
                sb.AppendLine($"- Vertex-to-triangle ratio: {(double)vertices.Count / triangleCount:F2}");
                sb.AppendLine();
            }
            
            // Normal vector analysis
            if (normals != null && normals.Count > 0)
            {
                sb.AppendLine("### Normal Vector Analysis");
                sb.AppendLine($"- Normal vector count: {normals.Count:N0}");
                
                if (vertices.Count == normals.Count)
                {
                    sb.AppendLine("- Normals match vertices (1:1 ratio)");
                }
                else
                {
                    sb.AppendLine($"- Vertex-to-normal ratio: {(double)vertices.Count / normals.Count:F2}");
                }
                
                sb.AppendLine();
            }
            
            // Sample vertex data
            sb.AppendLine("### Sample Vertex Data");
            sb.AppendLine("| Vertex # | X | Y | Z |");
            sb.AppendLine("|----------|---|---|---|");
            
            for (int i = 0; i < Math.Min(vertices.Count, 10); i++)
            {
                var vertex = vertices[i];
                sb.AppendLine($"| {i} | {vertex.X:F6} | {vertex.Y:F6} | {vertex.Z:F6} |");
            }
            
            if (vertices.Count > 10)
            {
                sb.AppendLine("*Table limited to first 10 vertices. See full data in the export files.*");
            }
            
            sb.AppendLine();
        }
        
        /// <summary>
        /// Performs integrated analysis of all data components and appends findings to the report
        /// </summary>
        private async Task AppendIntegratedAnalysis(StringBuilder sb, PM4File file)
        {
            var posEntries = file.PositionDataChunk?.Entries;
            var vertices = file.VertexPositionsChunk?.Vertices;
            var links = file.LinksChunk?.Links;
            
            if (posEntries == null || vertices == null)
                return;
                
            sb.AppendLine("## Integrated Analysis <a name=\"integrated-analysis\"></a>");
            sb.AppendLine("This section analyzes relationships between different data components in the PM4 file.");
            sb.AppendLine();
            
            // Compare coordinate spaces
            float posMinX = posEntries.Where(e => !e.IsSpecialEntry).Min(e => e.CoordinateX);
            float posMaxX = posEntries.Where(e => !e.IsSpecialEntry).Max(e => e.CoordinateX);
            float posMinZ = posEntries.Where(e => !e.IsSpecialEntry).Min(e => e.CoordinateZ);
            float posMaxZ = posEntries.Where(e => !e.IsSpecialEntry).Max(e => e.CoordinateZ);
            
            float vertexMinX = vertices.Min(v => v.X);
            float vertexMaxX = vertices.Max(v => v.X);
            float vertexMinZ = vertices.Min(v => v.Z);
            float vertexMaxZ = vertices.Max(v => v.Z);
            
            sb.AppendLine("### Coordinate Space Analysis");
            sb.AppendLine("| Data Type | X Range | Z Range |");
            sb.AppendLine("|-----------|---------|---------|");
            sb.AppendLine($"| Position Entries | {posMinX:F2} to {posMaxX:F2} | {posMinZ:F2} to {posMaxZ:F2} |");
            sb.AppendLine($"| Vertex Data | {vertexMinX:F2} to {vertexMaxX:F2} | {vertexMinZ:F2} to {vertexMaxZ:F2} |");
            sb.AppendLine();
            
            // Check for overlap
            bool xOverlap = (posMinX <= vertexMaxX && posMaxX >= vertexMinX);
            bool zOverlap = (posMinZ <= vertexMaxZ && posMaxZ >= vertexMinZ);
            
            sb.AppendLine($"The coordinate spaces of position entries and vertex data **{(xOverlap && zOverlap ? "do overlap" : "do not overlap")}**.");
            sb.AppendLine("This suggests that:");
            
            if (xOverlap && zOverlap)
            {
                sb.AppendLine("- Position entries and 3D model vertices share the same coordinate system");
                sb.AppendLine("- Special values likely represent the Y (height) component for position entries");
                sb.AppendLine("- Position entries may represent navigation waypoints within the 3D model space");
            }
            else
            {
                sb.AppendLine("- Position entries and 3D model vertices use different coordinate systems");
                sb.AppendLine("- They may represent different aspects of the game world");
                sb.AppendLine("- Additional transformation may be needed to align them");
            }
            
            sb.AppendLine();
            
            // Relationships between components
            sb.AppendLine("### Component Relationships");
            
            // Extract pairs of special entries and their following position entries
            var pairs = new List<(ServerPositionData Special, ServerPositionData Position)>();
            
            for (int i = 0; i < posEntries.Count - 1; i++)
            {
                if (posEntries[i].IsSpecialEntry && !posEntries[i + 1].IsSpecialEntry)
                {
                    pairs.Add((posEntries[i], posEntries[i + 1]));
                }
            }
            
            if (pairs.Count > 0 && links != null && links.Count > 0)
            {
                // Check if special values might correspond to link indices
                var specialValues = pairs.Select(p => (uint)p.Special.SpecialValue).ToList();
                var sourceIndices = links.Select(l => l.SourceIndex).ToList();
                var targetIndices = links.Select(l => l.TargetIndex).ToList();
                
                var intersectWithSource = specialValues.Intersect(sourceIndices).ToList();
                var intersectWithTarget = specialValues.Intersect(targetIndices).ToList();
                
                if (intersectWithSource.Count > 0 || intersectWithTarget.Count > 0)
                {
                    sb.AppendLine("Found potential correlations between special values and link indices:");
                    sb.AppendLine($"- {intersectWithSource.Count} special values match source indices in MSLK links");
                    sb.AppendLine($"- {intersectWithTarget.Count} special values match target indices in MSLK links");
                    sb.AppendLine();
                    
                    sb.AppendLine("This suggests that special values might be referencing specific vertices or structures");
                    sb.AppendLine("in the 3D model via the link system, possibly to associate waypoints with physical objects.");
                }
                else
                {
                    sb.AppendLine("No direct correlation found between special values and link indices.");
                    sb.AppendLine("Special values likely represent pure data values (e.g., elevations) rather than indices.");
                }
            }
            
            sb.AppendLine();
            
            // Final conclusions
            sb.AppendLine("### Overall Interpretation");
            sb.AppendLine("Based on the comprehensive analysis, the PM4 file appears to contain:");
            sb.AppendLine();
            sb.AppendLine("1. **3D Model Data**: Vertex positions, indices, and potentially normal vectors that define geometry");
            sb.AppendLine("2. **Navigation System**: Position entries that define waypoints for NPC movement");
            sb.AppendLine("3. **Link Structure**: Connections between vertices or other elements that may define pathing or relationships");
            sb.AppendLine();
            sb.AppendLine("The special values in position data entries most likely represent:");
            sb.AppendLine("- **Height/Y-coordinate** values when interpreted as floats");
            sb.AppendLine("- **Potentially group identifiers** that categorize waypoints or associate them with other data");
            sb.AppendLine();
            sb.AppendLine("The Z values in position entries appear to act as category flags rather than coordinates,");
            sb.AppendLine("with specific values (e.g., 0.0, 2.0) indicating different waypoint types.");
            sb.AppendLine();
        }
    }
} 