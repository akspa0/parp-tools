using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.Data.Sqlite;

namespace PM4Rebuilder
{
    /// <summary>
    /// Data structure representing a complete building definition with its component types.
    /// Updated for MPRR sentinel-based building boundaries.
    /// </summary>
    public class BuildingDefinition
    {
        // MPRR-based properties
        public int BuildingId { get; set; }
        public int StartPropertyIndex { get; set; }
        public int EndPropertyIndex { get; set; }
        public int EstimatedTriangleCount { get; set; }
        
        // Legacy properties for compatibility
        public uint ParentIndex { get; set; }
        public List<TypeGroup> TypeGroups { get; set; } = new List<TypeGroup>();
        public int TotalComponents { get; set; }
        public BuildingClassification Classification { get; set; }
        
        // Store the actual ParentIndex range this building represents (legacy + MPRR compatibility)
        public (uint StartIndex, uint EndIndex) ParentIndexRange { get; set; }
    }

    /// <summary>
    /// Represents a group of components with the same type value within a building.
    /// </summary>
    public class TypeGroup
    {
        public int TypeValue { get; set; }
        public int LinkCount { get; set; }
        public int GeometryLinkCount { get; set; }
        public List<string> SurfaceRefs { get; set; } = new List<string>();
    }

    /// <summary>
    /// Building classification based on triangle count and complexity (MPRR-based approach).
    /// </summary>
    public enum BuildingClassification
    {
        Fragment,           // < 1K triangles (rare, usually filtered out)
        Detail,             // 1K-10K triangles (building components)
        Object,             // 10K-38K triangles (small structures)
        Building,           // 38K-654K triangles (target building scale)
        Complex,            // > 654K triangles (massive structures)
        
        // Legacy classifications for compatibility
        Unknown,
        SimpleStructure,    // 1-2 types, low complexity
        StandardBuilding,   // 3-4 types, typical pattern
        ComplexBuilding,    // 5+ types, high complexity
        SpecialStructure    // Unusual type patterns
    }

    /// <summary>
    /// Result of a building export operation.
    /// </summary>
    public class BuildingExportResult
    {
        public uint ParentIndex { get; set; }
        public BuildingClassification Classification { get; set; }
        public int ComponentCount { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public bool ExportSuccessful { get; set; }
        public string? ErrorMessage { get; set; }
    }

    /// <summary>
    /// Exports complete PM4 buildings as unified OBJ files using ParentIndex + Type_0x01 grouping logic.
    /// </summary>
    public static class BuildingLevelExporter
    {
        /// <summary>
        /// Main export orchestrator - exports all buildings from the scene database.
        /// </summary>
        /// <param name="dbPath">Path to the scene.db SQLite database</param>
        /// <param name="outputDir">Output directory for OBJ files</param>
        /// <returns>Exit code (0 = success, 1 = error)</returns>
        public static int ExportAllBuildings(string dbPath, string outputDir)
        {
            try
            {
                Console.WriteLine($"[BUILDING EXPORTER] Starting export from: {dbPath}");
                Console.WriteLine($"[BUILDING EXPORTER] Output directory: {outputDir}");

                Directory.CreateDirectory(outputDir);

                var results = new List<BuildingExportResult>();

                using var connection = new SqliteConnection($"Data Source={dbPath}");
                connection.Open();

                // Get building definitions
                var buildings = GetBuildingDefinitions(connection);
                Console.WriteLine($"[BUILDING EXPORTER] Found {buildings.Count} building definitions");

                // Export each building
                int successCount = 0;
                int errorCount = 0;

                foreach (var building in buildings)
                {
                    try
                    {
                        var result = ExportBuilding(connection, building, outputDir);
                        results.Add(result);

                        if (result.ExportSuccessful)
                        {
                            successCount++;
                            Console.WriteLine($"[SUCCESS] Building {building.ParentIndex}: {result.VertexCount} vertices, {result.TriangleCount} triangles");
                        }
                        else
                        {
                            errorCount++;
                            Console.WriteLine($"[ERROR] Building {building.ParentIndex}: {result.ErrorMessage}");
                        }
                    }
                    catch (Exception ex)
                    {
                        errorCount++;
                        Console.WriteLine($"[ERROR] Building {building.ParentIndex}: {ex.Message}");
                        results.Add(new BuildingExportResult
                        {
                            ParentIndex = building.ParentIndex,
                            ExportSuccessful = false,
                            ErrorMessage = ex.Message
                        });
                    }
                }

                // Generate building index
                GenerateBuildingIndex(results, outputDir);

                Console.WriteLine($"[BUILDING EXPORTER] Export completed: {successCount} successful, {errorCount} errors");
                return errorCount > 0 ? 1 : 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BUILDING EXPORTER ERROR] {ex.Message}");
                return 1;
            }
        }

        /// <summary>
        /// Queries the database to build MPRR sentinel-based building definitions using Value1=65535 markers.
        /// MPRR sentinels define the true building boundaries and produce realistic building-scale objects.
        /// </summary>
        /// <param name="conn">SQLite database connection</param>
        /// <returns>List of building definitions based on MPRR sentinel boundaries</returns>
        public static List<BuildingDefinition> GetBuildingDefinitions(SqliteConnection conn)
        {
            Console.WriteLine("[BUILDING EXPORTER] Analyzing MPRR sentinel boundaries...");
            
            // Step 1: Check if Properties table exists (MPRR data)
            if (!DoesPropertiesTableExist(conn))
            {
                Console.WriteLine("[BUILDING EXPORTER ERROR] Properties table not found - database needs to be regenerated with MPRR data");
                Console.WriteLine("[BUILDING EXPORTER ERROR] Please re-run the database export to include MPRR chunk data");
                return new List<BuildingDefinition>();
            }
            
            // Step 2: Get all MPRR sentinels (Value1=65535) to define building boundaries
            var sentinelBoundaries = GetMprrSentinelBoundaries(conn);
            Console.WriteLine($"[BUILDING EXPORTER] Found {sentinelBoundaries.Count} MPRR sentinel boundaries");
            
            if (sentinelBoundaries.Count == 0)
            {
                Console.WriteLine("[BUILDING EXPORTER WARNING] No MPRR sentinels found - may need to regenerate database");
                return new List<BuildingDefinition>();
            }
            
            // Step 3: Build building ranges between consecutive MPRR sentinels
            var buildingRanges = BuildMprrBuildingRanges(sentinelBoundaries);
            Console.WriteLine($"[BUILDING EXPORTER] Identified {buildingRanges.Count} MPRR-defined building ranges");
            
            // Step 4: Convert ranges to BuildingDefinition objects with triangle count estimation
            var buildings = new List<BuildingDefinition>();
            
            foreach (var range in buildingRanges)
            {
                // Estimate triangle count for this building range
                int estimatedTriangles = EstimateTriangleCountInRange(conn, range.StartPropertyIndex, range.EndPropertyIndex);
                
                if (estimatedTriangles > 0) // Only create buildings that have geometry
                {
                    var building = new BuildingDefinition
                    {
                        BuildingId = range.BuildingId,
                        StartPropertyIndex = range.StartPropertyIndex,
                        EndPropertyIndex = range.EndPropertyIndex,
                        EstimatedTriangleCount = estimatedTriangles,
                        Classification = ClassifyBuildingByTriangleCount(estimatedTriangles),
                        ParentIndexRange = ((uint)range.StartPropertyIndex, (uint)range.EndPropertyIndex) // For compatibility
                    };
                    
                    buildings.Add(building);
                }
            }
            
            Console.WriteLine($"[BUILDING EXPORTER] Created {buildings.Count} building definitions with geometry");
            return buildings;
        }
        
        /// <summary>
        /// Gets all container boundary ParentIndex values (MspiFirstIndex = -1).
        /// </summary>
        private static List<uint> GetContainerBoundaries(SqliteConnection conn)
        {
            const string sql = @"
                SELECT DISTINCT ParentIndex 
                FROM Links 
                WHERE MspiFirstIndex = -1 
                ORDER BY ParentIndex";
                
            var boundaries = new List<uint>();
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                boundaries.Add((uint)reader.GetInt64("ParentIndex"));
            }
            
            return boundaries;
        }
        
        /// <summary>
        /// Builds building ranges between container boundaries.
        /// </summary>
        private static List<(uint StartIndex, uint EndIndex)> BuildBuildingRanges(List<uint> containerBoundaries)
        {
            var ranges = new List<(uint StartIndex, uint EndIndex)>();
            
            if (containerBoundaries.Count == 0)
                return ranges;
                
            // Create ranges between consecutive container boundaries
            for (int i = 0; i < containerBoundaries.Count - 1; i++)
            {
                uint start = containerBoundaries[i];
                uint end = containerBoundaries[i + 1] - 1;
                
                if (end > start) // Only add valid ranges
                {
                    ranges.Add((start, end));
                }
            }
            
            return ranges;
        }
        
        /// <summary>
        /// Counts geometry-bearing ParentIndex values in a given range.
        /// </summary>
        private static int GetGeometryCountInRange(SqliteConnection conn, uint startIndex, uint endIndex)
        {
            const string sql = @"
                SELECT COUNT(DISTINCT l.ParentIndex) as GeometryCount
                FROM Links l
                JOIN Surfaces s ON l.ReferenceIndex = s.GlobalIndex
                JOIN Triangles t ON t.GlobalIndex >= s.MsviFirstIndex 
                            AND t.GlobalIndex < (s.MsviFirstIndex + s.IndexCount)
                WHERE l.ParentIndex >= @start AND l.ParentIndex <= @end";
                
            using var cmd = new SqliteCommand(sql, conn);
            cmd.Parameters.AddWithValue("@start", startIndex);
            cmd.Parameters.AddWithValue("@end", endIndex);
            
            using var reader = cmd.ExecuteReader();
            return reader.Read() ? reader.GetInt32("GeometryCount") : 0;
        }

        /// <summary>
        /// Exports a single building (container range) as a unified OBJ file.
        /// </summary>
        /// <param name="conn">SQLite database connection</param>
        /// <param name="building">Building definition to export</param>
        /// <param name="outputDir">Output directory</param>
        /// <returns>Export result</returns>
        public static BuildingExportResult ExportBuilding(SqliteConnection conn, BuildingDefinition building, string outputDir)
        {
            var result = new BuildingExportResult
            {
                ParentIndex = building.ParentIndex,
                Classification = building.Classification,
                ComponentCount = building.TotalComponents
            };

            try
            {
                // Create output file path with range info
                string fileName = $"building_{building.ParentIndex:D6}_range_{building.ParentIndexRange.StartIndex}-{building.ParentIndexRange.EndIndex}_{building.Classification}.obj";
                string outputPath = Path.Combine(outputDir, fileName);
                result.OutputPath = outputPath;

                // Query to get all triangles for this building range via Links -> Surfaces relationship
                const string triangleQuery = @"
                    SELECT DISTINCT
                        t.VertexA, t.VertexB, t.VertexC
                    FROM Links l
                    JOIN Surfaces s ON l.ReferenceIndex = s.GlobalIndex
                    JOIN Triangles t ON t.GlobalIndex >= s.MsviFirstIndex 
                                    AND t.GlobalIndex < (s.MsviFirstIndex + s.IndexCount)
                    WHERE l.ParentIndex >= @startIndex AND l.ParentIndex <= @endIndex
                    ORDER BY t.GlobalIndex";

                using var cmd = new SqliteCommand(triangleQuery, conn);
                cmd.Parameters.AddWithValue("@startIndex", building.ParentIndexRange.StartIndex);
                cmd.Parameters.AddWithValue("@endIndex", building.ParentIndexRange.EndIndex);

                var triangles = new List<(int V1, int V2, int V3)>();
                var uniqueVertexIndices = new HashSet<int>();

                using var reader = cmd.ExecuteReader();
                while (reader.Read())
                {
                    var v1 = reader.GetInt32("VertexA");
                    var v2 = reader.GetInt32("VertexB");
                    var v3 = reader.GetInt32("VertexC");
                    
                    triangles.Add((v1, v2, v3));
                    uniqueVertexIndices.Add(v1);
                    uniqueVertexIndices.Add(v2);
                    uniqueVertexIndices.Add(v3);
                }

                // Get vertices for the unique vertex indices we found
                if (uniqueVertexIndices.Count == 0)
                {
                    result.ExportSuccessful = false;
                    result.ErrorMessage = "No triangles found for this building";
                    return result;
                }

                // Build parameterized query for vertex indices
                var parameterNames = uniqueVertexIndices.Select((_, i) => $"@v{i}").ToArray();
                string vertexQuery = $@"
                    SELECT GlobalIndex, X, Y, Z
                    FROM Vertices
                    WHERE GlobalIndex IN ({string.Join(",", parameterNames)})
                    ORDER BY GlobalIndex";

                var vertices = new List<(float X, float Y, float Z)>();
                var vertexIndexMap = new Dictionary<int, int>(); // Maps GlobalIndex to OBJ vertex index

                using var vertexCmd = new SqliteCommand(vertexQuery, conn);
                
                // Add parameters for vertex indices
                int paramIndex = 0;
                foreach (var vertexIndex in uniqueVertexIndices)
                {
                    vertexCmd.Parameters.AddWithValue($"@v{paramIndex}", vertexIndex);
                    paramIndex++;
                }
                
                using var vertexReader = vertexCmd.ExecuteReader();
                while (vertexReader.Read())
                {
                    var globalIndex = vertexReader.GetInt32("GlobalIndex");
                    // Apply X-axis flip to fix coordinate system issue
                    var vertex = (-vertexReader.GetFloat("X"), vertexReader.GetFloat("Y"), vertexReader.GetFloat("Z"));
                    
                    vertexIndexMap[globalIndex] = vertices.Count;
                    vertices.Add(vertex);
                }

                // Map triangles to OBJ vertex indices
                var mappedTriangles = new List<(int V1, int V2, int V3)>();
                foreach (var (v1, v2, v3) in triangles)
                {
                    if (vertexIndexMap.ContainsKey(v1) && vertexIndexMap.ContainsKey(v2) && vertexIndexMap.ContainsKey(v3))
                    {
                        mappedTriangles.Add((vertexIndexMap[v1], vertexIndexMap[v2], vertexIndexMap[v3]));
                    }
                }

                // Write OBJ file
                WriteObjFile(outputPath, vertices, mappedTriangles, building);

                result.VertexCount = vertices.Count;
                result.TriangleCount = mappedTriangles.Count;
                result.ExportSuccessful = true;

                return result;
            }
            catch (Exception ex)
            {
                result.ExportSuccessful = false;
                result.ErrorMessage = ex.Message;
                return result;
            }
        }

        /// <summary>
        /// Categorizes a building based on its component patterns.
        /// </summary>
        /// <param name="building">Building to classify</param>
        /// <returns>Building classification</returns>
        public static BuildingClassification ClassifyBuilding(BuildingDefinition building)
        {
            int typeCount = building.TypeGroups.Count;
            int totalGeometryLinks = building.TypeGroups.Sum(tg => tg.GeometryLinkCount);

            if (typeCount <= 2 && totalGeometryLinks < 10)
                return BuildingClassification.SimpleStructure;
            else if (typeCount <= 4 && totalGeometryLinks < 50)
                return BuildingClassification.StandardBuilding;
            else if (typeCount >= 5 || totalGeometryLinks >= 50)
                return BuildingClassification.ComplexBuilding;
            else
                return BuildingClassification.SpecialStructure;
        }

        /// <summary>
        /// Generates a CSV index of all exported buildings.
        /// </summary>
        /// <param name="results">Export results</param>
        /// <param name="outputDir">Output directory</param>
        public static void GenerateBuildingIndex(List<BuildingExportResult> results, string outputDir)
        {
            string indexPath = Path.Combine(outputDir, "building_index.csv");
            
            using var writer = new StreamWriter(indexPath, false, Encoding.UTF8);
            writer.WriteLine("ParentIndex,Classification,ComponentCount,VertexCount,TriangleCount,OutputFile,Success,ErrorMessage");

            foreach (var result in results.OrderBy(r => r.ParentIndex))
            {
                writer.WriteLine($"{result.ParentIndex}," +
                               $"{result.Classification}," +
                               $"{result.ComponentCount}," +
                               $"{result.VertexCount}," +
                               $"{result.TriangleCount}," +
                               $"{Path.GetFileName(result.OutputPath)}," +
                               $"{result.ExportSuccessful}," +
                               $"\"{result.ErrorMessage?.Replace("\"", "\"\"") ?? ""}\"");
            }

            Console.WriteLine($"[BUILDING EXPORTER] Generated index: {indexPath}");
        }

        /// <summary>
        /// Writes geometry data to an OBJ file.
        /// </summary>
        private static void WriteObjFile(string path, List<(float X, float Y, float Z)> vertices, 
                                       List<(int V1, int V2, int V3)> triangles, BuildingDefinition building)
        {
            using var writer = new StreamWriter(path, false, Encoding.UTF8);
            
            // Write header
            writer.WriteLine($"# Building {building.ParentIndex} - {building.Classification}");
            writer.WriteLine($"# Components: {building.TotalComponents}, Types: {string.Join(",", building.TypeGroups.Select(tg => tg.TypeValue))}");
            writer.WriteLine($"# Exported by PM4Rebuilder BuildingLevelExporter");
            writer.WriteLine();

            // Write vertices
            foreach (var vertex in vertices)
            {
                writer.WriteLine($"v {vertex.X.ToString("F6", CultureInfo.InvariantCulture)} " +
                               $"{vertex.Y.ToString("F6", CultureInfo.InvariantCulture)} " +
                               $"{vertex.Z.ToString("F6", CultureInfo.InvariantCulture)}");
            }

            writer.WriteLine();

            // Write faces (triangles)
            foreach (var triangle in triangles)
            {
                // OBJ indices are 1-based
                writer.WriteLine($"f {triangle.V1 + 1} {triangle.V2 + 1} {triangle.V3 + 1}");
            }
        }
        
        #region MPRR Sentinel-Based Building Grouping Methods
        
        /// <summary>
        /// Checks if the Properties table exists in the database (indicates MPRR data is available).
        /// </summary>
        private static bool DoesPropertiesTableExist(SqliteConnection conn)
        {
            const string query = @"
                SELECT COUNT(*) 
                FROM sqlite_master 
                WHERE type='table' AND name='Properties';";
            
            using var cmd = new SqliteCommand(query, conn);
            var result = Convert.ToInt32(cmd.ExecuteScalar());
            return result > 0;
        }
        
        /// <summary>
        /// Gets all MPRR sentinel boundaries (Value1=65535) from the Properties table.
        /// </summary>
        private static List<int> GetMprrSentinelBoundaries(SqliteConnection conn)
        {
            var sentinels = new List<int>();
            
            const string query = @"
                SELECT GlobalIndex
                FROM Properties 
                WHERE IsBoundarySentinel = 1
                ORDER BY GlobalIndex;";
            
            using var cmd = new SqliteCommand(query, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                sentinels.Add(reader.GetInt32("GlobalIndex"));
            }
            
            return sentinels;
        }
        
        /// <summary>
        /// Builds building ranges between consecutive MPRR sentinels.
        /// </summary>
        private static List<MprrBuildingRange> BuildMprrBuildingRanges(List<int> sentinelBoundaries)
        {
            var ranges = new List<MprrBuildingRange>();
            
            if (sentinelBoundaries.Count == 0)
                return ranges;
            
            int buildingId = 1;
            int currentStart = 0;
            
            foreach (var sentinelIndex in sentinelBoundaries)
            {
                if (sentinelIndex > currentStart)
                {
                    ranges.Add(new MprrBuildingRange
                    {
                        BuildingId = buildingId++,
                        StartPropertyIndex = currentStart,
                        EndPropertyIndex = sentinelIndex - 1
                    });
                }
                
                currentStart = sentinelIndex + 1;
            }
            
            // Handle final range after last sentinel
            // Note: This would need the total property count, for now we'll use a large number
            if (currentStart < int.MaxValue)
            {
                ranges.Add(new MprrBuildingRange
                {
                    BuildingId = buildingId,
                    StartPropertyIndex = currentStart,
                    EndPropertyIndex = int.MaxValue // Will be constrained by actual property count
                });
            }
            
            return ranges;
        }
        
        /// <summary>
        /// Estimates triangle count for a building range based on property indices.
        /// This is a simplified estimation - actual implementation would query geometry.
        /// </summary>
        private static int EstimateTriangleCountInRange(SqliteConnection conn, int startPropertyIndex, int endPropertyIndex)
        {
            // For now, return a reasonable estimate based on property range
            // In full implementation, this would query the actual triangles
            var propertyCount = endPropertyIndex - startPropertyIndex + 1;
            
            // Estimate: ~50-200 triangles per property (rough approximation)
            return Math.Min(propertyCount * 100, 654000); // Cap at max building size
        }
        
        /// <summary>
        /// Classifies building by triangle count using MPRR-based scaling.
        /// </summary>
        private static BuildingClassification ClassifyBuildingByTriangleCount(int triangleCount)
        {
            return triangleCount switch
            {
                < 1000 => BuildingClassification.Fragment,
                < 10000 => BuildingClassification.Detail,
                < 38000 => BuildingClassification.Object,
                <= 654000 => BuildingClassification.Building,  // Target scale
                _ => BuildingClassification.Complex
            };
        }
        
        #endregion
        
        #region MPRR Helper Data Structures
        
        /// <summary>
        /// Represents a building range defined by MPRR sentinel boundaries.
        /// </summary>
        private class MprrBuildingRange
        {
            public int BuildingId { get; set; }
            public int StartPropertyIndex { get; set; }
            public int EndPropertyIndex { get; set; }
        }
        
        #endregion
    }
}
