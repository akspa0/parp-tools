using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4
{
    public class Pm4ChunkFieldAnalyzer
    {
        public Pm4ChunkFieldAnalyzer()
        {
        }

        public void AnalyzeChunkFields(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Starting comprehensive PM4 chunk field analysis");
            
            Directory.CreateDirectory(outputDirectory);
            
            // Analyze each chunk type
            AnalyzeMprlFields(scene, outputDirectory);
            AnalyzeMslkFields(scene, outputDirectory);
            AnalyzeMsurFields(scene, outputDirectory);
            AnalyzeMprrFields(scene, outputDirectory);
            
            // Cross-chunk correlation analysis
            AnalyzeCrossChunkCorrelations(scene, outputDirectory);
            
            ConsoleLogger.WriteLine("PM4 chunk field analysis complete");
        }

        private void AnalyzeMprlFields(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Analyzing MPRL placement fields");
            
            var csvPath = Path.Combine(outputDirectory, "MPRL_Field_Analysis.csv");
            var sb = new StringBuilder();
            
            // Header
            sb.AppendLine("Index,Unknown0,Unknown2,Unknown4,Unknown6,PosX,PosY,PosZ,Unknown14,Unknown16");
            
            // Statistics tracking
            var stats = new Dictionary<string, FieldStatistics>
            {
                ["Unknown0"] = new FieldStatistics(),
                ["Unknown2"] = new FieldStatistics(),
                ["Unknown4"] = new FieldStatistics(),
                ["Unknown6"] = new FieldStatistics(),
                ["Unknown14"] = new FieldStatistics(),
                ["Unknown16"] = new FieldStatistics()
            };
            
            // Extract all placement data
            for (int i = 0; i < scene.Placements.Count; i++)
            {
                var placement = scene.Placements[i];
                
                sb.AppendLine($"{i},{placement.Unknown0},{placement.Unknown2},{placement.Unknown4}," +
                             $"{placement.Unknown6},{placement.Position.X},{placement.Position.Y},{placement.Position.Z}," +
                             $"{placement.Unknown14},{placement.Unknown16}");
                
                // Update statistics
                stats["Unknown0"].AddValue(placement.Unknown0);
                stats["Unknown2"].AddValue(placement.Unknown2);
                stats["Unknown4"].AddValue(placement.Unknown4);
                stats["Unknown6"].AddValue(placement.Unknown6);
                stats["Unknown14"].AddValue(placement.Unknown14);
                stats["Unknown16"].AddValue(placement.Unknown16);
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            
            // Write statistics report
            var statsPath = Path.Combine(outputDirectory, "MPRL_Statistics.txt");
            WriteFieldStatistics(statsPath, "MPRL", stats);
            
            ConsoleLogger.WriteLine($"MPRL analysis complete: {scene.Placements.Count} placements analyzed");
        }

        private void AnalyzeMslkFields(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Analyzing MSLK link fields");
            
            var csvPath = Path.Combine(outputDirectory, "MSLK_Field_Analysis.csv");
            var sb = new StringBuilder();
            
            // Header
            sb.AppendLine("Index,ParentIndex,MspiFirstIndex,MspiIndexCount,ReferenceIndex,IsContainer");
            
            // Statistics tracking
            var stats = new Dictionary<string, FieldStatistics>
            {
                ["ParentIndex"] = new FieldStatistics(),
                ["MspiFirstIndex"] = new FieldStatistics(),
                ["MspiIndexCount"] = new FieldStatistics(),
                ["ReferenceIndex"] = new FieldStatistics()
            };
            
            int containerCount = 0;
            int geometryCount = 0;
            
            // Extract all link data
            for (int i = 0; i < scene.Links.Count; i++)
            {
                var link = scene.Links[i];
                var isContainer = link.MspiFirstIndex == -1;
                
                if (isContainer) containerCount++;
                else geometryCount++;
                
                sb.AppendLine($"{i},{link.ParentIndex},{link.MspiFirstIndex},{link.MspiIndexCount}," +
                             $"{link.ReferenceIndex},{isContainer}");
                
                // Update statistics
                stats["ParentIndex"].AddValue(link.ParentIndex);
                if (!isContainer) stats["MspiFirstIndex"].AddValue(link.MspiFirstIndex);
                stats["MspiIndexCount"].AddValue(link.MspiIndexCount);
                stats["ReferenceIndex"].AddValue(link.ReferenceIndex);
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            
            // Write statistics report
            var statsPath = Path.Combine(outputDirectory, "MSLK_Statistics.txt");
            var statsContent = new StringBuilder();
            statsContent.AppendLine("=== MSLK Link Statistics ===");
            statsContent.AppendLine($"Total Links: {scene.Links.Count}");
            statsContent.AppendLine($"Container Nodes: {containerCount}");
            statsContent.AppendLine($"Geometry Nodes: {geometryCount}");
            statsContent.AppendLine();
            
            WriteFieldStatisticsToBuilder(statsContent, "MSLK", stats);
            File.WriteAllText(statsPath, statsContent.ToString());
            
            ConsoleLogger.WriteLine($"MSLK analysis complete: {scene.Links.Count} links analyzed ({containerCount} containers, {geometryCount} geometry)");
        }

        private void AnalyzeMsurFields(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Analyzing MSUR surface fields");
            
            var csvPath = Path.Combine(outputDirectory, "MSUR_Field_Analysis.csv");
            var sb = new StringBuilder();
            
            // Header - need to inspect actual MSUR structure
            sb.AppendLine("Index,MsviFirstIndex,IndexCount,SurfaceKey,GroupKey,SurfaceAttributeMask,IsM2Bucket");
            
            // Statistics tracking
            var stats = new Dictionary<string, FieldStatistics>
            {
                ["MsviFirstIndex"] = new FieldStatistics(),
                ["IndexCount"] = new FieldStatistics(),
                ["SurfaceKey"] = new FieldStatistics(),
                ["GroupKey"] = new FieldStatistics(),
                ["SurfaceAttributeMask"] = new FieldStatistics()
            };
            
            // Extract all surface data
            for (int i = 0; i < scene.Surfaces.Count; i++)
            {
                var surface = scene.Surfaces[i];
                
                sb.AppendLine($"{i},{surface.MsviFirstIndex},{surface.IndexCount}," +
                             $"{surface.SurfaceKey},{surface.GroupKey},{surface.SurfaceAttributeMask}," +
                             $"{surface.IsM2Bucket}");
                
                // Update statistics
                stats["MsviFirstIndex"].AddValue(surface.MsviFirstIndex);
                stats["IndexCount"].AddValue(surface.IndexCount);
                stats["SurfaceKey"].AddValue(surface.SurfaceKey);
                stats["GroupKey"].AddValue(surface.GroupKey);
                stats["SurfaceAttributeMask"].AddValue(surface.SurfaceAttributeMask);
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            
            // Write statistics report
            var statsPath = Path.Combine(outputDirectory, "MSUR_Statistics.txt");
            WriteFieldStatistics(statsPath, "MSUR", stats);
            
            ConsoleLogger.WriteLine($"MSUR analysis complete: {scene.Surfaces.Count} surfaces analyzed");
        }

        private void AnalyzeMprrFields(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Analyzing MPRR property fields");
            
            var csvPath = Path.Combine(outputDirectory, "MPRR_Field_Analysis.csv");
            var sb = new StringBuilder();
            
            // Header
            sb.AppendLine("Index,Value1,Value2,IsSeparator");
            
            // Statistics tracking
            var stats = new Dictionary<string, FieldStatistics>
            {
                ["Value1"] = new FieldStatistics(),
                ["Value2"] = new FieldStatistics()
            };
            
            int separatorCount = 0;
            int propertyCount = 0;
            
            // Extract all property data
            for (int i = 0; i < scene.Properties.Count; i++)
            {
                var property = scene.Properties[i];
                var isSeparator = property.Value1 == 65535;
                
                if (isSeparator) separatorCount++;
                else propertyCount++;
                
                sb.AppendLine($"{i},{property.Value1},{property.Value2},{isSeparator}");
                
                // Update statistics
                stats["Value1"].AddValue(property.Value1);
                stats["Value2"].AddValue(property.Value2);
            }
            
            File.WriteAllText(csvPath, sb.ToString());
            
            // Write statistics report
            var statsPath = Path.Combine(outputDirectory, "MPRR_Statistics.txt");
            var statsContent = new StringBuilder();
            statsContent.AppendLine("=== MPRR Property Statistics ===");
            statsContent.AppendLine($"Total Properties: {scene.Properties.Count}");
            statsContent.AppendLine($"Separators (65535): {separatorCount}");
            statsContent.AppendLine($"Property Values: {propertyCount}");
            statsContent.AppendLine();
            
            WriteFieldStatisticsToBuilder(statsContent, "MPRR", stats);
            File.WriteAllText(statsPath, statsContent.ToString());
            
            ConsoleLogger.WriteLine($"MPRR analysis complete: {scene.Properties.Count} properties analyzed ({separatorCount} separators)");
        }

        private void AnalyzeCrossChunkCorrelations(Pm4Scene scene, string outputDirectory)
        {
            ConsoleLogger.WriteLine("Analyzing cross-chunk correlations");
            
            var correlationPath = Path.Combine(outputDirectory, "Cross_Chunk_Correlations.txt");
            var sb = new StringBuilder();
            
            sb.AppendLine("=== Cross-Chunk Correlation Analysis ===");
            sb.AppendLine();
            
            // MPRL.Unknown4 → MSLK.ParentIndex correlation
            sb.AppendLine("MPRL.Unknown4 → MSLK.ParentIndex Correlation:");
            var mprlUnknown4Values = scene.Placements.Select(p => (uint)p.Unknown4).Distinct().ToList();
            var mslkParentIndexValues = scene.Links.Select(l => l.ParentIndex).Distinct().ToList();
            var matchingValues = mprlUnknown4Values.Intersect(mslkParentIndexValues).ToList();
            
            sb.AppendLine($"  MPRL.Unknown4 unique values: {mprlUnknown4Values.Count}");
            sb.AppendLine($"  MSLK.ParentIndex unique values: {mslkParentIndexValues.Count}");
            sb.AppendLine($"  Matching values: {matchingValues.Count}");
            sb.AppendLine($"  Match percentage: {(double)matchingValues.Count / Math.Max(mprlUnknown4Values.Count, mslkParentIndexValues.Count) * 100:F1}%");
            sb.AppendLine();
            
            // MSUR.IndexCount analysis for object grouping
            sb.AppendLine("MSUR.IndexCount Distribution for Object Grouping:");
            var indexCountGroups = scene.Surfaces.GroupBy(s => s.IndexCount).OrderBy(g => g.Key).ToList();
            foreach (var group in indexCountGroups.Take(20)) // Top 20 most common
            {
                sb.AppendLine($"  IndexCount {group.Key}: {group.Count()} surfaces");
            }
            sb.AppendLine();
            
            // Geometry availability analysis
            sb.AppendLine("Geometry Availability Analysis:");
            sb.AppendLine($"  Total Vertices: {scene.Vertices.Count}");
            sb.AppendLine($"  Total Triangles: {scene.Triangles.Count}");
            sb.AppendLine($"  Total Surfaces: {scene.Surfaces.Count}");
            sb.AppendLine($"  Total Links: {scene.Links.Count}");
            sb.AppendLine($"  Total Placements: {scene.Placements.Count}");
            
            File.WriteAllText(correlationPath, sb.ToString());
            
            ConsoleLogger.WriteLine("Cross-chunk correlation analysis complete");
        }

        private void WriteFieldStatistics(string filePath, string chunkName, Dictionary<string, FieldStatistics> stats)
        {
            var sb = new StringBuilder();
            WriteFieldStatisticsToBuilder(sb, chunkName, stats);
            File.WriteAllText(filePath, sb.ToString());
        }

        private void WriteFieldStatisticsToBuilder(StringBuilder sb, string chunkName, Dictionary<string, FieldStatistics> stats)
        {
            sb.AppendLine($"=== {chunkName} Field Statistics ===");
            sb.AppendLine();
            
            foreach (var kvp in stats)
            {
                var fieldName = kvp.Key;
                var stat = kvp.Value;
                
                sb.AppendLine($"{fieldName}:");
                sb.AppendLine($"  Count: {stat.Count}");
                sb.AppendLine($"  Min: {stat.Min}");
                sb.AppendLine($"  Max: {stat.Max}");
                sb.AppendLine($"  Unique Values: {stat.UniqueValues.Count}");
                
                if (stat.UniqueValues.Count <= 20)
                {
                    sb.AppendLine($"  Values: {string.Join(", ", stat.UniqueValues.OrderBy(v => v))}");
                }
                else
                {
                    var topValues = stat.ValueCounts.OrderByDescending(kvp => kvp.Value).Take(10);
                    sb.AppendLine($"  Top 10 Most Common:");
                    foreach (var valueCount in topValues)
                    {
                        sb.AppendLine($"    {valueCount.Key}: {valueCount.Value} times");
                    }
                }
                sb.AppendLine();
            }
        }
    }

    public class FieldStatistics
    {
        public int Count { get; private set; }
        public long Min { get; private set; } = long.MaxValue;
        public long Max { get; private set; } = long.MinValue;
        public HashSet<long> UniqueValues { get; } = new HashSet<long>();
        public Dictionary<long, int> ValueCounts { get; } = new Dictionary<long, int>();

        public void AddValue(long value)
        {
            Count++;
            Min = Math.Min(Min, value);
            Max = Math.Max(Max, value);
            UniqueValues.Add(value);
            
            if (ValueCounts.ContainsKey(value))
                ValueCounts[value]++;
            else
                ValueCounts[value] = 1;
        }
    }
}
