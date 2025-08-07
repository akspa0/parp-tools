using Microsoft.Data.Sqlite;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace PM4Rebuilder
{
    /// <summary>
    /// Analyzes higher-level building aggregation patterns in PM4 database to discover
    /// how multiple ParentIndex sub-components should group into complete buildings.
    /// </summary>
    internal static class BuildingAggregationAnalyzer
    {
        public static int Analyze(string dbPath, string outputDir)
        {
            if (!File.Exists(dbPath))
            {
                Console.WriteLine($"ERROR: Database file '{dbPath}' does not exist.");
                return 1;
            }

            Directory.CreateDirectory(outputDir);
            Console.WriteLine($"[BUILD-ANALYSIS] Starting building aggregation analysis from: {dbPath}");
            Console.WriteLine($"[BUILD-ANALYSIS] Output directory: {outputDir}");

            using var conn = new SqliteConnection($"Data Source={dbPath};Mode=ReadOnly");
            conn.Open();

            // Step 1: Analyze Type_0x01 patterns
            AnalyzeTypePatterns(conn, outputDir);

            // Step 2: Analyze spatial clustering potential
            AnalyzeSpatialClustering(conn, outputDir);

            // Step 3: Analyze RawFieldsJson for additional grouping candidates
            AnalyzeRawFieldPatterns(conn, outputDir);

            // Step 4: Investigate SortKey and other numeric field variations
            AnalyzeNumericFieldDistributions(conn, outputDir);

            // Step 5: Generate comprehensive building hypothesis report
            GenerateBuildingHypothesisReport(outputDir);

            Console.WriteLine($"[BUILD-ANALYSIS] Analysis complete. Reports written to: {outputDir}");
            return 0;
        }

        private static void AnalyzeTypePatterns(SqliteConnection conn, string outputDir)
        {
            Console.WriteLine("[BUILD-ANALYSIS] Analyzing Type_0x01 patterns...");
            
            const string sql = @"
                SELECT 
                    ParentIndex,
                    json_extract(RawFieldsJson, '$.Type_0x01') as TypeValue,
                    COUNT(*) as LinkCount,
                    SUM(CASE WHEN MspiFirstIndex = -1 THEN 0 ELSE 1 END) as GeometryLinks,
                    GROUP_CONCAT(DISTINCT json_extract(RawFieldsJson, '$.SurfaceRefIndex')) as SurfaceRefs
                FROM Links 
                GROUP BY ParentIndex, TypeValue
                ORDER BY ParentIndex, TypeValue
            ";

            var typePatterns = new List<(uint ParentIndex, int TypeValue, int LinkCount, int GeometryLinks, string SurfaceRefs)>();
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                var parentIndex = (uint)reader.GetInt32(0);
                var typeValue = reader.GetInt32(1);
                var linkCount = reader.GetInt32(2);
                var geometryLinks = reader.GetInt32(3);
                var surfaceRefs = reader.GetString(4);
                
                typePatterns.Add((parentIndex, typeValue, linkCount, geometryLinks, surfaceRefs));
            }

            // Write Type analysis
            var typeAnalysisPath = Path.Combine(outputDir, "type_0x01_analysis.csv");
            var sb = new StringBuilder();
            sb.AppendLine("ParentIndex,TypeValue,LinkCount,GeometryLinks,SurfaceRefs");
            
            foreach (var (parentIndex, typeValue, linkCount, geometryLinks, surfaceRefs) in typePatterns)
            {
                sb.AppendLine($"{parentIndex},{typeValue},{linkCount},{geometryLinks},\"{surfaceRefs}\"");
            }
            
            File.WriteAllText(typeAnalysisPath, sb.ToString());
            
            // Generate Type distribution summary
            var typeDistribution = typePatterns.GroupBy(x => x.TypeValue)
                                              .OrderBy(g => g.Key)
                                              .ToList();
            
            var typeSummaryPath = Path.Combine(outputDir, "type_distribution_summary.txt");
            var typeSummary = new StringBuilder();
            typeSummary.AppendLine("Type_0x01 Distribution Analysis");
            typeSummary.AppendLine("=================================");
            typeSummary.AppendLine();
            
            foreach (var group in typeDistribution)
            {
                var avgGeometry = group.Average(x => x.GeometryLinks);
                var avgLinks = group.Average(x => x.LinkCount);
                typeSummary.AppendLine($"Type {group.Key}:");
                typeSummary.AppendLine($"  Count: {group.Count()} ParentIndex values");
                typeSummary.AppendLine($"  Avg Links: {avgLinks:F1}");
                typeSummary.AppendLine($"  Avg Geometry Links: {avgGeometry:F1}");
                typeSummary.AppendLine();
            }
            
            File.WriteAllText(typeSummaryPath, typeSummary.ToString());
            Console.WriteLine($"[BUILD-ANALYSIS] Type patterns written: {typeAnalysisPath}");
        }

        private static void AnalyzeSpatialClustering(SqliteConnection conn, string outputDir)
        {
            Console.WriteLine("[BUILD-ANALYSIS] Analyzing spatial clustering potential...");
            
            // First check if Placements table has spatial columns
            const string checkColumnsSql = "PRAGMA table_info(Placements)";
            var columns = new List<string>();
            
            using (var checkCmd = new SqliteCommand(checkColumnsSql, conn))
            using (var checkReader = checkCmd.ExecuteReader())
            {
                while (checkReader.Read())
                {
                    columns.Add(checkReader.GetString(1)); // Column name is at index 1
                }
            }
            
            bool hasXYZ = columns.Contains("X") && columns.Contains("Y") && columns.Contains("Z");
            
            if (!hasXYZ)
            {
                Console.WriteLine("[BUILD-ANALYSIS] No spatial columns (X,Y,Z) found in Placements table.");
                WriteSpatialAnalysisSkipped(outputDir);
                return;
            }
            
            // Get placement positions and their ParentIndex relationships
            const string sql = @"
                SELECT DISTINCT
                    p.Unknown4 as PlacementId,
                    p.X, p.Y, p.Z,
                    COUNT(l.ParentIndex) as LinkCount
                FROM Placements p
                LEFT JOIN Links l ON l.ParentIndex = p.Unknown4
                WHERE p.X IS NOT NULL AND p.Y IS NOT NULL AND p.Z IS NOT NULL
                GROUP BY p.Unknown4, p.X, p.Y, p.Z
                HAVING p.X != 0 OR p.Y != 0 OR p.Z != 0
                ORDER BY p.Unknown4
            ";

            var positions = new List<(uint PlacementId, float X, float Y, float Z, int LinkCount)>();
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                var placementId = (uint)reader.GetInt32(0);
                var x = reader.GetFloat(1);
                var y = reader.GetFloat(2);
                var z = reader.GetFloat(3);
                var linkCount = reader.GetInt32(4);
                
                positions.Add((placementId, x, y, z, linkCount));
            }

            if (positions.Count == 0)
            {
                Console.WriteLine("[BUILD-ANALYSIS] No valid spatial positions found in Placements.");
                return;
            }

            // Write spatial data
            var spatialPath = Path.Combine(outputDir, "spatial_positions.csv");
            var spatialSb = new StringBuilder();
            spatialSb.AppendLine("PlacementId,X,Y,Z,LinkCount,Distance2D,Distance3D");
            
            // Calculate distances from origin and between points
            foreach (var (placementId, x, y, z, linkCount) in positions)
            {
                var distance2D = Math.Sqrt(x * x + y * y);
                var distance3D = Math.Sqrt(x * x + y * y + z * z);
                spatialSb.AppendLine($"{placementId},{x:F3},{y:F3},{z:F3},{linkCount},{distance2D:F3},{distance3D:F3}");
            }
            
            File.WriteAllText(spatialPath, spatialSb.ToString());
            
            // Generate spatial clustering suggestions
            var clusterSuggestions = AnalyzeSpatialClusters(positions);
            var clusterPath = Path.Combine(outputDir, "spatial_clustering_analysis.txt");
            File.WriteAllText(clusterPath, clusterSuggestions);
            
            Console.WriteLine($"[BUILD-ANALYSIS] Spatial analysis written: {spatialPath}");
        }

        private static void WriteSpatialAnalysisSkipped(string outputDir)
        {
            var spatialPath = Path.Combine(outputDir, "spatial_analysis_skipped.txt");
            var content = new StringBuilder();
            content.AppendLine("Spatial Analysis Skipped");
            content.AppendLine("========================");
            content.AppendLine();
            content.AppendLine("The Placements table does not contain spatial columns (X, Y, Z).");
            content.AppendLine("Building grouping analysis will focus on:");
            content.AppendLine("- Type_0x01 field patterns");
            content.AppendLine("- SurfaceRefIndex sharing patterns");
            content.AppendLine("- RawFieldsJson field analysis");
            content.AppendLine();
            content.AppendLine("Recommendation: Look for ParentIndex values that span multiple Type_0x01 values");
            content.AppendLine("as these likely represent complete buildings with different component types.");
            
            File.WriteAllText(spatialPath, content.ToString());
            Console.WriteLine($"[BUILD-ANALYSIS] Spatial analysis skipped: {spatialPath}");
        }

        private static string AnalyzeSpatialClusters(List<(uint PlacementId, float X, float Y, float Z, int LinkCount)> positions)
        {
            var sb = new StringBuilder();
            sb.AppendLine("Spatial Clustering Analysis");
            sb.AppendLine("==========================");
            sb.AppendLine();
            
            if (positions.Count < 2)
            {
                sb.AppendLine("Insufficient positions for clustering analysis.");
                return sb.ToString();
            }

            // Calculate distance matrix and find potential clusters
            var distances = new List<(uint Id1, uint Id2, double Distance)>();
            
            for (int i = 0; i < positions.Count; i++)
            {
                for (int j = i + 1; j < positions.Count; j++)
                {
                    var p1 = positions[i];
                    var p2 = positions[j];
                    var distance = Math.Sqrt(
                        Math.Pow(p1.X - p2.X, 2) + 
                        Math.Pow(p1.Y - p2.Y, 2) + 
                        Math.Pow(p1.Z - p2.Z, 2)
                    );
                    distances.Add((p1.PlacementId, p2.PlacementId, distance));
                }
            }

            // Find close pairs (potential building components)
            var closePairs = distances.Where(d => d.Distance < 50.0) // Arbitrary threshold
                                     .OrderBy(d => d.Distance)
                                     .Take(10)
                                     .ToList();

            sb.AppendLine($"Total positions analyzed: {positions.Count}");
            sb.AppendLine($"Total distance pairs calculated: {distances.Count}");
            sb.AppendLine();
            
            if (closePairs.Any())
            {
                sb.AppendLine("Closest ParentIndex pairs (potential building components):");
                sb.AppendLine("PlacementId1,PlacementId2,Distance");
                foreach (var (id1, id2, distance) in closePairs)
                {
                    sb.AppendLine($"{id1},{id2},{distance:F3}");
                }
            }
            else
            {
                sb.AppendLine("No close spatial pairs found (all distances > 50 units).");
                sb.AppendLine("This suggests either:");
                sb.AppendLine("- Sub-components are spatially distributed");
                sb.AppendLine("- Clustering threshold needs adjustment");
                sb.AppendLine("- Building grouping is not spatial-based");
            }
            
            return sb.ToString();
        }

        private static void AnalyzeRawFieldPatterns(SqliteConnection conn, string outputDir)
        {
            Console.WriteLine("[BUILD-ANALYSIS] Analyzing RawFieldsJson patterns...");
            
            const string sql = "SELECT ParentIndex, RawFieldsJson FROM Links ORDER BY ParentIndex LIMIT 100";
            
            var fieldAnalysis = new Dictionary<string, Dictionary<string, int>>();
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                var parentIndex = reader.GetInt32(0);
                var rawJson = reader.GetString(1);
                
                try
                {
                    var jsonDoc = JsonDocument.Parse(rawJson);
                    foreach (var property in jsonDoc.RootElement.EnumerateObject())
                    {
                        var fieldName = property.Name;
                        var fieldValue = property.Value.ToString();
                        
                        if (!fieldAnalysis.ContainsKey(fieldName))
                            fieldAnalysis[fieldName] = new Dictionary<string, int>();
                        
                        if (!fieldAnalysis[fieldName].ContainsKey(fieldValue))
                            fieldAnalysis[fieldName][fieldValue] = 0;
                        
                        fieldAnalysis[fieldName][fieldValue]++;
                    }
                }
                catch (JsonException ex)
                {
                    Console.WriteLine($"[BUILD-ANALYSIS] JSON parse error for ParentIndex {parentIndex}: {ex.Message}");
                }
            }

            // Write field analysis
            var fieldsPath = Path.Combine(outputDir, "rawfields_analysis.txt");
            var fieldsSb = new StringBuilder();
            fieldsSb.AppendLine("RawFieldsJson Field Analysis");
            fieldsSb.AppendLine("===========================");
            fieldsSb.AppendLine();
            
            foreach (var (fieldName, valueCounts) in fieldAnalysis.OrderBy(kv => kv.Key))
            {
                fieldsSb.AppendLine($"Field: {fieldName}");
                fieldsSb.AppendLine($"  Unique values: {valueCounts.Count}");
                
                if (valueCounts.Count <= 10) // Show all values if few
                {
                    foreach (var (value, count) in valueCounts.OrderByDescending(kv => kv.Value))
                    {
                        fieldsSb.AppendLine($"    {value}: {count} occurrences");
                    }
                }
                else // Show top values if many
                {
                    var topValues = valueCounts.OrderByDescending(kv => kv.Value).Take(5);
                    foreach (var (value, count) in topValues)
                    {
                        fieldsSb.AppendLine($"    {value}: {count} occurrences");
                    }
                    fieldsSb.AppendLine($"    ... and {valueCounts.Count - 5} more values");
                }
                fieldsSb.AppendLine();
            }
            
            File.WriteAllText(fieldsPath, fieldsSb.ToString());
            Console.WriteLine($"[BUILD-ANALYSIS] RawFields analysis written: {fieldsPath}");
        }

        private static void AnalyzeNumericFieldDistributions(SqliteConnection conn, string outputDir)
        {
            Console.WriteLine("[BUILD-ANALYSIS] Analyzing numeric field distributions...");
            
            const string sql = @"
                SELECT 
                    json_extract(RawFieldsJson, '$.SortKey_0x02') as SortKey,
                    json_extract(RawFieldsJson, '$.TileCoordsRaw') as TileCoords,
                    json_extract(RawFieldsJson, '$.Unknown_0x12') as Unknown12,
                    json_extract(RawFieldsJson, '$.LinkIdPadding') as LinkIdPadding,
                    COUNT(*) as Count
                FROM Links 
                GROUP BY SortKey, TileCoords, Unknown12, LinkIdPadding
                ORDER BY Count DESC
            ";

            var numericPath = Path.Combine(outputDir, "numeric_field_distributions.csv");
            var numericSb = new StringBuilder();
            numericSb.AppendLine("SortKey,TileCoords,Unknown12,LinkIdPadding,Count");
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                var sortKey = reader.IsDBNull(0) ? "NULL" : reader.GetValue(0).ToString();
                var tileCoords = reader.IsDBNull(1) ? "NULL" : reader.GetValue(1).ToString();
                var unknown12 = reader.IsDBNull(2) ? "NULL" : reader.GetValue(2).ToString();
                var linkIdPadding = reader.IsDBNull(3) ? "NULL" : reader.GetValue(3).ToString();
                var count = reader.GetInt32(4);
                
                numericSb.AppendLine($"{sortKey},{tileCoords},{unknown12},{linkIdPadding},{count}");
            }
            
            File.WriteAllText(numericPath, numericSb.ToString());
            Console.WriteLine($"[BUILD-ANALYSIS] Numeric distributions written: {numericPath}");
        }

        private static void GenerateBuildingHypothesisReport(string outputDir)
        {
            var reportPath = Path.Combine(outputDir, "building_aggregation_hypothesis.txt");
            var report = new StringBuilder();
            
            report.AppendLine("PM4 Building Aggregation Hypothesis Report");
            report.AppendLine("=========================================");
            report.AppendLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            report.AppendLine();
            
            report.AppendLine("CURRENT UNDERSTANDING:");
            report.AppendLine("- Each Placements.Unknown4 → ParentIndex represents a sub-component (single polygon)");
            report.AppendLine("- Sub-components contain 1-29 Link child groups (most are single polygons)");
            report.AppendLine("- Links with MspiFirstIndex=-1 are organizational containers (no geometry)");
            report.AppendLine();
            
            report.AppendLine("MISSING AGGREGATION LEVEL:");
            report.AppendLine("- Need to discover how multiple ParentIndex values group into complete buildings");
            report.AppendLine("- Potential grouping candidates:");
            report.AppendLine("  1. Type_0x01 variations (1,2,3,4) - building component types");
            report.AppendLine("  2. Spatial proximity clustering - nearby sub-components");
            report.AppendLine("  3. SurfaceRefIndex patterns - shared surface materials");
            report.AppendLine("  4. TileCoordinate/LinkId relationships - tile-based organization");
            report.AppendLine();
            
            report.AppendLine("ANALYSIS FILES GENERATED:");
            report.AppendLine("- type_0x01_analysis.csv: Type pattern analysis per ParentIndex");
            report.AppendLine("- type_distribution_summary.txt: Type usage statistics");
            report.AppendLine("- spatial_positions.csv: 3D positions of sub-components");
            report.AppendLine("- spatial_clustering_analysis.txt: Proximity-based grouping suggestions");
            report.AppendLine("- rawfields_analysis.txt: Complete field variation analysis");
            report.AppendLine("- numeric_field_distributions.csv: Numeric field pattern distributions");
            report.AppendLine();
            
            report.AppendLine("NEXT STEPS:");
            report.AppendLine("1. Review Type_0x01 patterns - do Types 1,2,3,4 represent building components?");
            report.AppendLine("2. Check spatial clustering results - are nearby sub-components related?");
            report.AppendLine("3. Investigate SurfaceRefIndex sharing - do buildings share material sets?");
            report.AppendLine("4. Analyze field variations - look for previously overlooked grouping fields");
            report.AppendLine("5. Test building assembly hypotheses on sample data");
            
            File.WriteAllText(reportPath, report.ToString());
            Console.WriteLine($"[BUILD-ANALYSIS] Hypothesis report written: {reportPath}");
        }
    }
}
