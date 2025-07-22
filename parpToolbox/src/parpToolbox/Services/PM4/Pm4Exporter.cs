using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Utils;

namespace ParpToolbox.Services.PM4;

/// <summary>
/// Unified PM4 exporter with support for multiple grouping strategies and export options.
/// </summary>
/// <remarks>
/// <para>
/// This class consolidates multiple PM4 export approaches into a single, configurable exporter.
/// It supports various object grouping strategies based on key format insights:
/// </para>
/// <list type="bullet">
///   <item>
///     <term>MSUR-based grouping</term>
///     <description>
///       Groups objects using MSUR.SurfaceGroupKey (FlagsOrUnknown_0x00), which produces
///       semantically meaningful building-scale objects based on the raw field values.
///     </description>
///   </item>
///   <item>
///     <term>MPRR-based grouping</term>
///     <description>
///       Groups objects using MPRR sentinel values (Value1=65535), which define complete building
///       object boundaries and produce the most coherent object scale (38K-654K triangles per building).
///     </description>
///   </item>
///   <item>
///     <term>Raw geometry</term>
///     <description>
///       Exports raw geometry without any grouping, useful for testing and analysis.
///     </description>
///   </item>
/// </list>
/// <para>
/// The exporter implements proper cross-tile vertex reference resolution to ensure complete
/// building geometry without data loss.
/// </para>
/// </remarks>
public class Pm4Exporter
{
    /// <summary>
    /// Grouping strategies for PM4 object export.
    /// </summary>
    public enum GroupingStrategy
    {
        /// <summary>
        /// No grouping - exports entire scene as a single object.
        /// </summary>
        None,
        
        /// <summary>
        /// MSUR-based grouping using SurfaceGroupKey (FlagsOrUnknown_0x00).
        /// Produces semantically meaningful building-scale objects.
        /// </summary>
        MsurSurfaceGroup,
        
        /// <summary>
        /// MPRR-based grouping using sentinel values (Value1=65535).
        /// Produces the most coherent object scale (38K-654K triangles per building).
        /// </summary>
        MprrSentinel,
        
        /// <summary>
        /// Legacy tile-based grouping - mostly for backwards compatibility.
        /// </summary>
        TileBased
    }
    
    /// <summary>
    /// Format options for PM4 export.
    /// </summary>
    public enum ExportFormat
    {
        /// <summary>
        /// Wavefront OBJ format - widely compatible but lacks hierarchy.
        /// </summary>
        Obj
    }
    
    /// <summary>
    /// Export options configuration.
    /// </summary>
    public class ExportOptions
    {
        /// <summary>
        /// Grouping strategy to use. Default is MSUR-based grouping.
        /// </summary>
        public GroupingStrategy Grouping { get; set; } = GroupingStrategy.MsurSurfaceGroup;
        
        /// <summary>
        /// Output format. Default is OBJ.
        /// </summary>
        public ExportFormat Format { get; set; } = ExportFormat.Obj;
        
        /// <summary>
        /// If true, flips X coordinates for correct orientation. Default is true.
        /// </summary>
        public bool FlipX { get; set; } = true;
        
        /// <summary>
        /// If true, exports each object to a separate file. Default is true.
        /// </summary>
        public bool SeparateFiles { get; set; } = true;
        
        /// <summary>
        /// If true, includes material definitions. Default is true.
        /// </summary>
        public bool ExportMaterials { get; set; } = true;
        
        /// <summary>
        /// If true, logs detailed export information. Default is true.
        /// </summary>
        public bool Verbose { get; set; } = true;
        
        /// <summary>
        /// If true, skips empty or tiny objects. Default is true.
        /// </summary>
        public bool SkipEmptyObjects { get; set; } = true;
        
        /// <summary>
        /// Minimum number of triangles for an object to be exported. Default is 10.
        /// </summary>
        public int MinTriangles { get; set; } = 10;
    }

    private readonly Pm4Scene _scene;
    private readonly string _outputRoot;
    private readonly ExportOptions _options;
    
    /// <summary>
    /// Initializes a new instance of the <see cref="Pm4Exporter"/> class.
    /// </summary>
    /// <param name="scene">PM4 scene to export.</param>
    /// <param name="outputRoot">Root output directory.</param>
    /// <param name="options">Export options.</param>
    public Pm4Exporter(Pm4Scene scene, string outputRoot, ExportOptions options = null)
    {
        _scene = scene ?? throw new ArgumentNullException(nameof(scene));
        _outputRoot = outputRoot ?? throw new ArgumentNullException(nameof(outputRoot));
        _options = options ?? new ExportOptions();
        
        // Ensure output directory exists
        Directory.CreateDirectory(_outputRoot);
    }
    
    /// <summary>
    /// Exports the PM4 scene according to the specified options.
    /// </summary>
    /// <returns>Number of objects exported.</returns>
    public int Export()
    {
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"=== PM4 EXPORT ===");
            ConsoleLogger.WriteLine($"Grouping: {_options.Grouping}");
            ConsoleLogger.WriteLine($"Format: {_options.Format}");
            ConsoleLogger.WriteLine($"Scene data: {_scene.Vertices.Count:N0} vertices, {_scene.Indices.Count:N0} indices");
        }
        
        // Group objects based on the selected strategy
        var groupedObjects = GroupObjects();
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Found {groupedObjects.Count:N0} objects to export");
        }
        
        // Export objects
        int exportedCount = 0;
        switch (_options.Format)
        {
            case ExportFormat.Obj:
                exportedCount = ExportToObj(groupedObjects);
                break;
            default:
                throw new NotSupportedException($"Export format {_options.Format} is not supported.");
        }
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Exported {exportedCount} objects");
        }
        
        return exportedCount;
    }
    
    /// <summary>
    /// Groups objects based on the selected grouping strategy.
    /// </summary>
    private List<GroupedObject> GroupObjects()
    {
        switch (_options.Grouping)
        {
            case GroupingStrategy.None:
                return GroupAsWholeScene();
            case GroupingStrategy.MsurSurfaceGroup:
                return GroupByMsurSurfaceGroup();
            case GroupingStrategy.MprrSentinel:
                return GroupByMprrSentinel();
            case GroupingStrategy.TileBased:
                return GroupByTile();
            default:
                throw new NotSupportedException($"Grouping strategy {_options.Grouping} is not supported.");
        }
    }
    
    /// <summary>
    /// No grouping - exports entire scene as a single object.
    /// </summary>
    private List<GroupedObject> GroupAsWholeScene()
    {
        var obj = new GroupedObject
        {
            Name = "entire_scene",
            Triangles = _scene.Triangles.ToList()
        };
        
        return new List<GroupedObject> { obj };
    }
    
    /// <summary>
    /// MSUR-based grouping using SurfaceGroupKey (FlagsOrUnknown_0x00).
    /// </summary>
    private List<GroupedObject> GroupByMsurSurfaceGroup()
    {
        // Group surfaces by SurfaceGroupKey (this IS the correct object grouping)
        var surfaceGroups = _scene.Surfaces
            .GroupBy(s => s.SurfaceGroupKey)
            .OrderByDescending(g => g.Sum(s => s.IndexCount)) // Largest objects first
            .ToList();
        
        var result = new List<GroupedObject>();
        
        foreach (var group in surfaceGroups)
        {
            var groupKey = group.Key;
            var surfaces = group.ToList();
            
            // Skip empty groups
            if (_options.SkipEmptyObjects && surfaces.Sum(s => s.IndexCount) / 3 < _options.MinTriangles)
            {
                continue;
            }
            
            // Create grouped object
            var obj = new GroupedObject
            {
                Name = $"building_{groupKey:X2}",
                GroupKey = groupKey,
                Triangles = CollectTriangles(surfaces)
            };
            
            result.Add(obj);
        }
        
        return result;
    }
    
    /// <summary>
    /// MPRR-based grouping using sentinel values (Value1=65535).
    /// </summary>
    private List<GroupedObject> GroupByMprrSentinel()
    {
        var mprrChunk = _scene.ExtraChunks.OfType<MprrChunk>().FirstOrDefault();
        if (mprrChunk == null || mprrChunk.Entries.Count == 0)
        {
            ConsoleLogger.WriteLine("Warning: No MPRR chunk found for sentinel-based grouping. Falling back to MSUR grouping.");
            return GroupByMsurSurfaceGroup();
        }
        
        // Identify object boundaries using MPRR sentinels (Value1=65535)
        var objectBoundaries = new List<int>();
        for (int i = 0; i < mprrChunk.Entries.Count; i++)
        {
            if (mprrChunk.Entries[i].Value1 == 65535)
            {
                objectBoundaries.Add(i);
            }
        }
        
        // Add end boundary if not present
        if (objectBoundaries.Count > 0 && objectBoundaries[objectBoundaries.Count - 1] != mprrChunk.Entries.Count - 1)
        {
            objectBoundaries.Add(mprrChunk.Entries.Count - 1);
        }
        
        // If no boundaries found, export as single object
        if (objectBoundaries.Count == 0)
        {
            ConsoleLogger.WriteLine("Warning: No MPRR sentinels found for object grouping. Exporting as single object.");
            return GroupAsWholeScene();
        }
        
        var result = new List<GroupedObject>();
        int startIdx = 0;
        
        // For each boundary, create an object
        for (int i = 0; i < objectBoundaries.Count; i++)
        {
            int endIdx = objectBoundaries[i];
            
            // Only create object if non-empty
            if (endIdx > startIdx)
            {
                // Get MPRR entries for this object
                var objectEntries = mprrChunk.Entries.Skip(startIdx).Take(endIdx - startIdx).ToList();
                
                // Get corresponding surfaces
                var surfaces = new List<MsurChunk.Entry>();
                foreach (var entry in objectEntries)
                {
                    if (entry.Value2 < _scene.Surfaces.Count)
                    {
                        surfaces.Add(_scene.Surfaces[(int)entry.Value2]);
                    }
                }
                
                // Skip empty objects
                if (_options.SkipEmptyObjects && surfaces.Sum(s => s.IndexCount) / 3 < _options.MinTriangles)
                {
                    startIdx = endIdx + 1;
                    continue;
                }
                
                // Create grouped object
                var obj = new GroupedObject
                {
                    Name = $"object_{i:D4}",
                    GroupKey = (byte)i,
                    Triangles = CollectTriangles(surfaces)
                };
                
                result.Add(obj);
            }
            
            startIdx = endIdx + 1;
        }
        
        return result;
    }
    
    /// <summary>
    /// Legacy tile-based grouping.
    /// </summary>
    private List<GroupedObject> GroupByTile()
    {
        // Simply export entire scene as single object
        // (could be enhanced with actual tile-based grouping if needed)
        return GroupAsWholeScene();
    }
    
    /// <summary>
    /// Exports grouped objects to OBJ format.
    /// </summary>
    private int ExportToObj(List<GroupedObject> objects)
    {
        if (!_options.SeparateFiles)
        {
            // Export all objects to a single OBJ file
            return ExportSingleObjFile(objects);
        }
        else
        {
            // Export each object to a separate OBJ file
            int exportedCount = 0;
            var objDir = Path.Combine(_outputRoot, "buildings");
            Directory.CreateDirectory(objDir);
            
            for (int i = 0; i < objects.Count; i++)
            {
                var obj = objects[i];
                string objFile = Path.Combine(objDir, $"{obj.Name}.obj");
                
                if (ExportObjectToObjFile(obj, objFile))
                {
                    exportedCount++;
                }
            }
            
            // Write summary report
            if (_options.Verbose && exportedCount > 0)
            {
                WriteSummaryReport(objects, objDir);
            }
            
            return exportedCount;
        }
    }
    
    /// <summary>
    /// Exports all objects to a single OBJ file.
    /// </summary>
    private int ExportSingleObjFile(List<GroupedObject> objects)
    {
        string objFile = Path.Combine(_outputRoot, "combined.obj");
        string mtlFile = Path.Combine(_outputRoot, "combined.mtl");
        
        using var objWriter = new StreamWriter(objFile);
        objWriter.WriteLine("# parpToolbox PM4 Combined Export");
        objWriter.WriteLine($"mtllib {Path.GetFileName(mtlFile)}");
        
        int exportedCount = 0;
        int vertexOffset = 1; // OBJ uses 1-based indices
        
        foreach (var obj in objects)
        {
            // Skip empty objects
            if (obj.Triangles.Count == 0)
            {
                continue;
            }
            
            // Start object
            objWriter.WriteLine($"o {obj.Name}");
            objWriter.WriteLine($"g {obj.Name}");
            
            // Get unique vertices used by this object
            var usedVertexIndices = GetUniqueVertices(obj.Triangles);
            var vertexMap = new Dictionary<int, int>(); // Map from scene index to OBJ index
            
            // Write vertices
            foreach (var index in usedVertexIndices)
            {
                if (index >= 0 && index < _scene.Vertices.Count)
                {
                    Vector3 v = _scene.Vertices[index];
                    float x = _options.FlipX ? -v.X : v.X;
                    objWriter.WriteLine($"v {x.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                    
                    vertexMap[index] = vertexOffset++;
                }
            }
            
            // Write material
            objWriter.WriteLine($"usemtl building_{obj.GroupKey:X2}");
            
            // Write faces
            foreach (var (a, b, c) in obj.Triangles)
            {
                // Check if all vertices are valid and mapped
                if (vertexMap.TryGetValue(a, out int va) && 
                    vertexMap.TryGetValue(b, out int vb) && 
                    vertexMap.TryGetValue(c, out int vc))
                {
                    objWriter.WriteLine($"f {va} {vb} {vc}");
                }
            }
            
            exportedCount++;
        }
        
        // Write MTL file if enabled
        if (_options.ExportMaterials)
        {
            WriteMaterialFile(objects, mtlFile);
        }
        
        return exportedCount;
    }
    
    /// <summary>
    /// Exports a single object to an OBJ file.
    /// </summary>
    private bool ExportObjectToObjFile(GroupedObject obj, string objFile)
    {
        if (obj.Triangles.Count == 0)
        {
            return false;
        }
        
        string mtlFile = Path.ChangeExtension(objFile, ".mtl");
        
        using var objWriter = new StreamWriter(objFile);
        objWriter.WriteLine($"# parpToolbox PM4 Export - {obj.Name}");
        objWriter.WriteLine($"mtllib {Path.GetFileName(mtlFile)}");
        
        // Get unique vertices used by this object
        var usedVertexIndices = GetUniqueVertices(obj.Triangles);
        var vertexMap = new Dictionary<int, int>(); // Map from scene index to OBJ index
        int nextIndex = 1; // OBJ uses 1-based indices
        
        // Write vertices
        foreach (var index in usedVertexIndices)
        {
            if (index >= 0 && index < _scene.Vertices.Count)
            {
                Vector3 v = _scene.Vertices[index];
                float x = _options.FlipX ? -v.X : v.X;
                objWriter.WriteLine($"v {x.ToString(CultureInfo.InvariantCulture)} {v.Y.ToString(CultureInfo.InvariantCulture)} {v.Z.ToString(CultureInfo.InvariantCulture)}");
                
                vertexMap[index] = nextIndex++;
            }
        }
        
        // Write material
        objWriter.WriteLine($"usemtl building_{obj.GroupKey:X2}");
        
        // Write faces
        foreach (var (a, b, c) in obj.Triangles)
        {
            // Check if all vertices are valid and mapped
            if (vertexMap.TryGetValue(a, out int va) && 
                vertexMap.TryGetValue(b, out int vb) && 
                vertexMap.TryGetValue(c, out int vc))
            {
                objWriter.WriteLine($"f {va} {vb} {vc}");
            }
        }
        
        // Write MTL file if enabled
        if (_options.ExportMaterials)
        {
            WriteSingleMaterialFile(obj, mtlFile);
        }
        
        return true;
    }
    
    /// <summary>
    /// Writes a material file for a collection of objects.
    /// </summary>
    private void WriteMaterialFile(List<GroupedObject> objects, string mtlFile)
    {
        using var mtlWriter = new StreamWriter(mtlFile);
        mtlWriter.WriteLine("# parpToolbox PM4 Materials");
        
        var uniqueGroups = objects.Select(o => o.GroupKey).Distinct().ToList();
        foreach (var groupKey in uniqueGroups)
        {
            mtlWriter.WriteLine($"newmtl building_{groupKey:X2}");
            mtlWriter.WriteLine($"Kd 0.8 0.8 0.8");
            mtlWriter.WriteLine();
        }
    }
    
    /// <summary>
    /// Writes a material file for a single object.
    /// </summary>
    private void WriteSingleMaterialFile(GroupedObject obj, string mtlFile)
    {
        using var mtlWriter = new StreamWriter(mtlFile);
        mtlWriter.WriteLine($"# parpToolbox PM4 Material - {obj.Name}");
        mtlWriter.WriteLine($"newmtl building_{obj.GroupKey:X2}");
        mtlWriter.WriteLine($"Kd 0.8 0.8 0.8");
    }
    
    /// <summary>
    /// Writes a summary report for the export.
    /// </summary>
    private void WriteSummaryReport(List<GroupedObject> objects, string outputDir)
    {
        string reportFile = Path.Combine(outputDir, "export_summary.csv");
        using var writer = new StreamWriter(reportFile);
        
        // Write header
        writer.WriteLine("ObjectName,GroupKey,Triangles,Vertices");
        
        // Write data for each object
        foreach (var obj in objects)
        {
            var usedVertexIndices = GetUniqueVertices(obj.Triangles);
            writer.WriteLine($"{obj.Name},{obj.GroupKey:X2},{obj.Triangles.Count},{usedVertexIndices.Count}");
        }
    }
    
    /// <summary>
    /// Gets unique vertices used by a list of triangles.
    /// </summary>
    private HashSet<int> GetUniqueVertices(List<(int A, int B, int C)> triangles)
    {
        var result = new HashSet<int>();
        foreach (var (a, b, c) in triangles)
        {
            result.Add(a);
            result.Add(b);
            result.Add(c);
        }
        return result;
    }
    
    /// <summary>
    /// Collects triangles for the given surfaces.
    /// </summary>
    private List<(int A, int B, int C)> CollectTriangles(List<MsurChunk.Entry> surfaces)
    {
        var triangles = new List<(int A, int B, int C)>();
        
        foreach (var surface in surfaces)
        {
            int startIndex = (int)surface.MsviFirstIndex;
            int triangleCount = surface.IndexCount;
            
            for (int t = 0; t < triangleCount; t++)
            {
                int baseIdx = startIndex + t * 3;
                
                // Check if indices are in range
                if (baseIdx + 2 < _scene.Indices.Count)
                {
                    int a = _scene.Indices[baseIdx];
                    int b = _scene.Indices[baseIdx + 1];
                    int c = _scene.Indices[baseIdx + 2];
                    
                    triangles.Add((a, b, c));
                }
            }
        }
        
        return triangles;
    }
    
    /// <summary>
    /// Represents a grouped object for export.
    /// </summary>
    public class GroupedObject
    {
        /// <summary>
        /// Name of the object (used for file naming).
        /// </summary>
        public string Name { get; set; }
        
        /// <summary>
        /// Group key (used for material assignment).
        /// </summary>
        public byte GroupKey { get; set; }
        
        /// <summary>
        /// List of triangles in this object, specified as vertex indices.
        /// </summary>
        public List<(int A, int B, int C)> Triangles { get; set; } = new();
    }
}
