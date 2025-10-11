using Microsoft.Data.Sqlite;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace PM4Rebuilder
{
    /// <summary>
    /// Bulk exporter that automatically exports all PM4 sub-components from a database
    /// as individual OBJ files with comprehensive CSV documentation.
    /// </summary>
    internal static class BulkSubComponentExporter
    {
        public static int ExportAll(string dbPath, string outputDir)
        {
            if (!File.Exists(dbPath))
            {
                Console.WriteLine($"ERROR: Database file '{dbPath}' does not exist.");
                return 1;
            }

            Directory.CreateDirectory(outputDir);
            Console.WriteLine($"[BULK-EXPORT] Starting bulk sub-component export from: {dbPath}");
            Console.WriteLine($"[BULK-EXPORT] Output directory: {outputDir}");

            using var conn = new SqliteConnection($"Data Source={dbPath};Mode=ReadOnly");
            conn.Open();

            // Step 1: Get all distinct ObjectIds from Placements
            var objectIds = GetAllObjectIds(conn);
            Console.WriteLine($"[BULK-EXPORT] Found {objectIds.Count} distinct ObjectIds to export");

            if (objectIds.Count == 0)
            {
                Console.WriteLine("[BULK-EXPORT] No ObjectIds found in database.");
                return 0;
            }

            // Step 2: Export each sub-component as OBJ
            var exportSummary = new List<(uint ObjectId, int ChildGroups, int VertexCount, int FaceCount)>();
            int successCount = 0;

            foreach (var objectId in objectIds)
            {
                try
                {
                    var stats = ExportSubComponent(conn, objectId, outputDir);
                    exportSummary.Add((objectId, stats.ChildGroups, stats.VertexCount, stats.FaceCount));
                    successCount++;
                    
                    if (successCount % 10 == 0)
                    {
                        Console.WriteLine($"[BULK-EXPORT] Progress: {successCount}/{objectIds.Count} completed");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[BULK-EXPORT] ERROR exporting ObjectId {objectId}: {ex.Message}");
                }
            }

            // Step 3: Generate comprehensive documentation
            GenerateIndexCsv(exportSummary, outputDir);
            GenerateLinksCsv(conn, outputDir);
            GenerateSurfacesCsv(conn, outputDir);
            GenerateSummaryTxt(exportSummary, outputDir, dbPath);

            Console.WriteLine($"[BULK-EXPORT] Completed: {successCount}/{objectIds.Count} sub-components exported");
            Console.WriteLine($"[BULK-EXPORT] Documentation files written to: {outputDir}");
            return 0;
        }

        private static List<uint> GetAllObjectIds(SqliteConnection conn)
        {
            var objectIds = new List<uint>();
            const string sql = "SELECT DISTINCT Unknown4 FROM Placements ORDER BY Unknown4";
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            while (reader.Read())
            {
                objectIds.Add((uint)reader.GetInt32(0));
            }
            
            return objectIds;
        }

        private static (int ChildGroups, int VertexCount, int FaceCount) ExportSubComponent(SqliteConnection conn, uint objectId, string outputDir)
        {
            // Get child group count for this ObjectId
            int childGroups = GetChildGroupCount(conn, objectId);
            
            // Generate filename with format: id_{objectId}_subcomponent_{childGroups}.obj
            string fileName = $"id_{objectId}_subcomponent_{childGroups}.obj";
            string objPath = Path.Combine(outputDir, fileName);
            
            // Use existing SubComponentObjExporter to do the actual export
            int vertexCount = SubComponentObjExporter.Export(dbPath: conn.ConnectionString.Replace("Data Source=", "").Replace(";Mode=ReadOnly", ""), 
                                                           objectId: objectId, 
                                                           outPath: objPath);
            
            // Count faces by reading the OBJ file
            int faceCount = CountFacesInObjFile(objPath);
            
            return (childGroups, vertexCount, faceCount);
        }

        private static int GetChildGroupCount(SqliteConnection conn, uint objectId)
        {
            const string sql = "SELECT COUNT(*) FROM Links WHERE ParentIndex = @objectId";
            using var cmd = new SqliteCommand(sql, conn);
            cmd.Parameters.AddWithValue("@objectId", objectId);
            return Convert.ToInt32(cmd.ExecuteScalar());
        }

        private static int CountFacesInObjFile(string objPath)
        {
            if (!File.Exists(objPath))
                return 0;
                
            try
            {
                return File.ReadAllLines(objPath).Count(line => line.StartsWith("f "));
            }
            catch
            {
                return 0;
            }
        }

        private static void GenerateIndexCsv(List<(uint ObjectId, int ChildGroups, int VertexCount, int FaceCount)> exportSummary, string outputDir)
        {
            string csvPath = Path.Combine(outputDir, "index.csv");
            var sb = new StringBuilder();
            sb.AppendLine("ObjectId,ChildGroupCount,VertexCount,FaceCount,ObjFileName");
            
            foreach (var (objectId, childGroups, vertexCount, faceCount) in exportSummary)
            {
                string fileName = $"id_{objectId}_subcomponent_{childGroups}.obj";
                sb.AppendLine($"{objectId},{childGroups},{vertexCount},{faceCount},{fileName}");
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            Console.WriteLine($"[BULK-EXPORT] Written index.csv with {exportSummary.Count} entries");
        }

        private static void GenerateLinksCsv(SqliteConnection conn, string outputDir)
        {
            string csvPath = Path.Combine(outputDir, "links.csv");
            const string sql = "SELECT * FROM Links ORDER BY ParentIndex, rowid";
            
            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();
            
            var sb = new StringBuilder();
            // Write header
            for (int i = 0; i < reader.FieldCount; i++)
            {
                if (i > 0) sb.Append(",");
                sb.Append(reader.GetName(i));
            }
            sb.AppendLine();
            
            // Write data
            while (reader.Read())
            {
                for (int i = 0; i < reader.FieldCount; i++)
                {
                    if (i > 0) sb.Append(",");
                    sb.Append(reader.GetValue(i));
                }
                sb.AppendLine();
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            Console.WriteLine("[BULK-EXPORT] Written links.csv with all link data");
        }

        private static void GenerateSurfacesCsv(SqliteConnection conn, string outputDir)
        {
            string csvPath = Path.Combine(outputDir, "surfaces.csv");
            const string sql = "SELECT * FROM Surfaces ORDER BY SurfaceKey";
            
            try
            {
                using var cmd = new SqliteCommand(sql, conn);
                using var reader = cmd.ExecuteReader();
                
                var sb = new StringBuilder();
                // Write header
                for (int i = 0; i < reader.FieldCount; i++)
                {
                    if (i > 0) sb.Append(",");
                    sb.Append(reader.GetName(i));
                }
                sb.AppendLine();
                
                // Write data
                while (reader.Read())
                {
                    for (int i = 0; i < reader.FieldCount; i++)
                    {
                        if (i > 0) sb.Append(",");
                        sb.Append(reader.GetValue(i));
                    }
                    sb.AppendLine();
                }
                
                File.WriteAllText(csvPath, sb.ToString());
                Console.WriteLine("[BULK-EXPORT] Written surfaces.csv with all surface data");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[BULK-EXPORT] WARNING: Could not export surfaces.csv: {ex.Message}");
            }
        }

        private static void GenerateSummaryTxt(List<(uint ObjectId, int ChildGroups, int VertexCount, int FaceCount)> exportSummary, 
                                               string outputDir, string dbPath)
        {
            string summaryPath = Path.Combine(outputDir, "summary.txt");
            var sb = new StringBuilder();
            
            sb.AppendLine("PM4 Bulk Sub-Component Export Summary");
            sb.AppendLine("=====================================");
            sb.AppendLine($"Export Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
            sb.AppendLine($"Source Database: {Path.GetFileName(dbPath)}");
            sb.AppendLine($"Output Directory: {outputDir}");
            sb.AppendLine();
            
            sb.AppendLine("Export Statistics:");
            sb.AppendLine($"  Total Sub-Components Exported: {exportSummary.Count}");
            sb.AppendLine($"  Total Vertices: {exportSummary.Sum(x => x.VertexCount):N0}");
            sb.AppendLine($"  Total Faces: {exportSummary.Sum(x => x.FaceCount):N0}");
            sb.AppendLine($"  Total Child Groups: {exportSummary.Sum(x => x.ChildGroups):N0}");
            sb.AppendLine();
            
            sb.AppendLine("Child Group Distribution:");
            var groupCounts = exportSummary.GroupBy(x => x.ChildGroups)
                                         .OrderBy(g => g.Key)
                                         .ToList();
            foreach (var group in groupCounts)
            {
                sb.AppendLine($"  {group.Key} groups: {group.Count()} sub-components");
            }
            sb.AppendLine();
            
            sb.AppendLine("Vertex Count Distribution:");
            var vertexRanges = new[]
            {
                (0, 10, "1-10"),
                (10, 50, "11-50"),
                (50, 100, "51-100"),
                (100, 500, "101-500"),
                (500, 1000, "501-1000"),
                (1000, int.MaxValue, "1000+")
            };
            
            foreach (var (min, max, label) in vertexRanges)
            {
                int count = exportSummary.Count(x => x.VertexCount > min && x.VertexCount <= max);
                if (count > 0)
                {
                    sb.AppendLine($"  {label} vertices: {count} sub-components");
                }
            }
            
            File.WriteAllText(summaryPath, sb.ToString());
            Console.WriteLine("[BULK-EXPORT] Written summary.txt with export statistics");
        }
    }
}
