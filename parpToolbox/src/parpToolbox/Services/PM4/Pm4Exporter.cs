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

/*
/// <summary>
/// Unified PM4 exporter with support for multiple grouping strategies and export options.
/// </summary>
public class Pm4Exporter
{
    /// <summary>
    /// Export format options.
    /// </summary>
    public enum ExportFormat
    {
        /// <summary>
        /// Wavefront OBJ format.
        /// </summary>
        Obj,
    }
    
    /// <summary>
    /// Grouping strategy for PM4 objects.
    /// </summary>
    public enum GroupingStrategy
    {
        /// <summary>
        /// No grouping - exports entire scene as a single object.
        /// </summary>
        None,
        
        /// <summary>
        /// Groups by MSLK ParentIndex_0x04 (4th field) - the correct approach for PM4 objects.
        /// </summary>
        ParentIndex,
        
        /// <summary>
        /// Groups by MSUR surface group key (FlagsOrUnknown_0x00).
        /// </summary>
        MsurSurfaceGroup,
        
        /// <summary>
        /// Groups by MPRR sentinel values (Value1=65535).
        /// </summary>
        MprrSentinel,
        
        /// <summary>
        /// Groups by tile index.
        /// </summary>
        TileBased,
    }
    
    /// <summary>
    /// Export options for <see cref="Pm4Exporter"/>.
    /// </summary>
    public class ExportOptions
    {
        /// <summary>
        /// Gets or sets the export format. Default is OBJ.
        /// </summary>
        public ExportFormat Format { get; set; } = ExportFormat.Obj;
        
        /// <summary>
        /// Gets or sets the grouping strategy. Default is ParentIndex.
        /// </summary>
        public GroupingStrategy Grouping { get; set; } = GroupingStrategy.ParentIndex;
        
        /// <summary>
        /// If true, exports each group to a separate file. Default is false.
        /// </summary>
        public bool SeparateFiles { get; set; } = false;
        
        /// <summary>
        /// If true, inverts the X coordinate to correct mirroring. Default is true.
        /// </summary>
        public bool FlipX { get; set; } = true;
        
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
    public Pm4Exporter(Pm4Scene scene, string outputRoot, ExportOptions? options = null)
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
            case GroupingStrategy.ParentIndex:
                return GroupByParentIndex();
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
    /// Groups objects by MSLK ParentIndex_0x04 (4th field) - the correct approach for PM4 objects.
    /// </summary>
    private List<GroupedObject> GroupByParentIndex()
    {
        var result = new List<GroupedObject>();
        
        if (_scene.MSLK == null || _scene.MSLK.Entries == null || _scene.MSLK.Entries.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MSLK entries found to group by ParentIndex");
            }
            return result;
        }
        
        // Group MSLK entries by ParentIndex_0x04 (4th field) - this is the actual object grouping key
        var objectsByParentIndex = _scene.MSLK.Entries
            .Where(entry => entry != null && GetParentIndex(entry) != 0xFFFFFFFF)
            .GroupBy(entry => GetParentIndex(entry))
            .ToList();
        
        foreach (var objectGroup in objectsByParentIndex)
        {
            var triangles = new List<(int A, int B, int C)>();
            
            // Collect triangles from all MSLK entries with this parent index
            foreach (var mslkEntry in objectGroup)
            {
                var entryTriangles = ExtractTrianglesForMslkEntry(mslkEntry);
                triangles.AddRange(entryTriangles);
            }
            
            if (triangles.Count == 0 && _options.SkipEmptyObjects)
                continue;
            
            if (triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            var obj = new GroupedObject
            {
                Name = $"building_{objectGroup.Key}",
                Triangles = triangles,
                GroupKey = objectGroup.Key
            };
            
            result.Add(obj);
        }
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Grouped {result.Count} objects by MSLK ParentIndex_0x04");
        }
        
        return result;
    }
    
    /// <summary>
    /// Gets the ParentIndex_0x04 value from an MSLK entry.
    /// </summary>
    private uint GetParentIndex(object mslkEntry)
    {
        return GetPropertyValue<uint>(mslkEntry, "ParentIndex_0x04");
    }
    
    /// <summary>
    /// Gets the TileIndex value from an MSLK entry.
    /// </summary>
    private uint GetTileIndex(object mslkEntry)
    {
        return GetPropertyValue<uint>(mslkEntry, "TileIndex");
    }
    
    /// <summary>
    /// Gets a property value from an object using reflection.
    /// </summary>
    private T GetPropertyValue<T>(object obj, string propertyName)
    {
        try
        {
            var type = obj.GetType();
            var property = type.GetProperty(propertyName);
            if (property != null)
            {
                return (T)property.GetValue(obj);
            }
            
            var field = type.GetField(propertyName);
            if (field != null)
            {
                return (T)field.GetValue(obj);
            }
            
            return default(T);
        }
        catch
        {
            return default(T);
        }
    }
    
    /// <summary>
    /// Extracts triangles for a single MSLK entry.
    /// </summary>
    private List<(int A, int B, int C)> ExtractTrianglesForMslkEntry(object mslkEntry)
    {
        var triangles = new List<(int A, int B, int C)>();
        
        if (mslkEntry == null)
            return triangles;
        
        try
        {
            // Get MSLK entry properties using reflection
            var mspiFirstIndex = GetPropertyValue<int>(mslkEntry, "MspiFirstIndex");
            var mspiIndexCount = GetPropertyValue<int>(mslkEntry, "MspiIndexCount");
            
            // Validate indices
            if (mspiFirstIndex < 0 || mspiIndexCount <= 0)
                return triangles;
            
            // Get triangles from MSPI using the indices
            if (_scene.MSPI != null && _scene.MSPI.Indices != null)
            {
                var startIndex = mspiFirstIndex;
                var endIndex = Math.Min(startIndex + mspiIndexCount, _scene.MSPI.Indices.Count);
                
                for (int i = startIndex; i + 2 < endIndex; i += 3)
                {
                    if (i + 2 < _scene.MSPI.Indices.Count)
                    {
                        triangles.Add((
                            _scene.MSPI.Indices[i],
                            _scene.MSPI.Indices[i + 1],
                            _scene.MSPI.Indices[i + 2]
                        ));
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine($"Error extracting triangles: {ex.Message}");
            }
        }
        
        return triangles;
    }
    
    /// <summary>
    /// Exports a single grouped object to an OBJ file.
    /// </summary>
    private void ExportObject(GroupedObject obj, string filePath)
    {
        using var writer = new StreamWriter(filePath, false, Encoding.UTF8);
        
        // Write header
        writer.WriteLine($"# PM4 Export - {obj.Name}");
        writer.WriteLine($"# Triangles: {obj.Triangles.Count}");
        writer.WriteLine();
        
        // Write vertices (we'll need to map indices to actual vertices)
        var vertices = new List<Vector3>();
        var vertexMap = new Dictionary<int, int>();
        
        if (_scene.MSVT != null && _scene.MSVT.Vertices != null)
        {
            foreach (var triangle in obj.Triangles)
            {
                if (!vertexMap.ContainsKey(triangle.A))
                {
                    vertexMap[triangle.A] = vertices.Count;
                    if (triangle.A < _scene.MSVT.Vertices.Count)
                    {
                        vertices.Add(_scene.MSVT.Vertices[triangle.A]);
                    }
                }
                
                if (!vertexMap.ContainsKey(triangle.B))
                {
                    vertexMap[triangle.B] = vertices.Count;
                    if (triangle.B < _scene.MSVT.Vertices.Count)
                    {
                        vertices.Add(_scene.MSVT.Vertices[triangle.B]);
                    }
                }
                
                if (!vertexMap.ContainsKey(triangle.C))
                {
                    vertexMap[triangle.C] = vertices.Count;
                    if (triangle.C < _scene.MSVT.Vertices.Count)
                    {
                        vertices.Add(_scene.MSVT.Vertices[triangle.C]);
                    }
                }
            }
            
            // Write vertices
            foreach (var vertex in vertices)
            {
                // Apply X-axis inversion if requested
                float x = _options.FlipX ? -vertex.X : vertex.X;
                writer.WriteLine($"v {x:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            writer.WriteLine();
            
            // Write faces
            foreach (var triangle in obj.Triangles)
            {
                if (vertexMap.ContainsKey(triangle.A) && 
                    vertexMap.ContainsKey(triangle.B) && 
                    vertexMap.ContainsKey(triangle.C))
                {
                    var a = vertexMap[triangle.A] + 1; // OBJ indices are 1-based
                    var b = vertexMap[triangle.B] + 1;
                    var c = vertexMap[triangle.C] + 1;
                    writer.WriteLine($"f {a} {b} {c}");
                }
            }
        }
    }
    
    /// <summary>
    /// Writes a material file for the exported objects.
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
    /// Exports grouped objects to OBJ format.
    /// </summary>
    private int ExportToObj(List<GroupedObject> groupedObjects)
    {
        if (_options.SeparateFiles)
        {
            return ExportToSeparateObjFiles(groupedObjects);
        }
        else
        {
            return ExportToCombinedObjFile(groupedObjects);
        }
    }
    
    /// <summary>
    /// Exports grouped objects to separate OBJ files.
    /// </summary>
    private int ExportToSeparateObjFiles(List<GroupedObject> groupedObjects)
    {
        int exportedCount = 0;
        
        foreach (var obj in groupedObjects)
        {
            if (obj.Triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            string fileName = $"{obj.Name}.obj";
            string filePath = Path.Combine(_outputRoot, fileName);
            
            try
            {
                ExportObject(obj, filePath);
                exportedCount++;
                
                if (_options.Verbose)
                {
                    ConsoleLogger.WriteLine($"Exported: {fileName} ({obj.Triangles.Count} triangles)");
                }
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Error exporting {fileName}: {ex.Message}");
            }
        }
        
        return exportedCount;
    }
    
    /// <summary>
    /// Exports all grouped objects to a single combined OBJ file.
    /// </summary>
    private int ExportToCombinedObjFile(List<GroupedObject> groupedObjects)
    {
        string objFile = Path.Combine(_outputRoot, "combined.obj");
        string mtlFile = Path.Combine(_outputRoot, "combined.mtl");
        
        using var writer = new StreamWriter(objFile, false, Encoding.UTF8);
        writer.WriteLine("# Combined PM4 Export");
        writer.WriteLine($"# Objects: {groupedObjects.Count}");
        writer.WriteLine($"mtllib combined.mtl");
        writer.WriteLine();
        
        int totalVertices = 0;
        int exportedCount = 0;
        
        foreach (var obj in groupedObjects)
        {
            if (obj.Triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            // Write vertices
            var vertices = new List<Vector3>();
            var vertexMap = new Dictionary<int, int>();
            
            if (_scene.MSVT != null && _scene.MSVT.Vertices != null)
            {
                foreach (var triangle in obj.Triangles)
                {
                    if (!vertexMap.ContainsKey(triangle.A))
                    {
                        vertexMap[triangle.A] = vertices.Count;
                        if (triangle.A < _scene.MSVT.Vertices.Count)
                        {
                            vertices.Add(_scene.MSVT.Vertices[triangle.A]);
                        }
                    }
                    
                    if (!vertexMap.ContainsKey(triangle.B))
                    {
                        vertexMap[triangle.B] = vertices.Count;
                        if (triangle.B < _scene.MSVT.Vertices.Count)
                        {
                            vertices.Add(_scene.MSVT.Vertices[triangle.B]);
                        }
                    }
                    
                    if (!vertexMap.ContainsKey(triangle.C))
                    {
                        vertexMap[triangle.C] = vertices.Count;
                        if (triangle.C < _scene.MSVT.Vertices.Count)
                        {
                            vertices.Add(_scene.MSVT.Vertices[triangle.C]);
                        }
                    }
                }
                
                // Write vertices with offset
                foreach (var vertex in vertices)
                {
                    // Apply X-axis inversion if requested
                    float x = _options.FlipX ? -vertex.X : vertex.X;
                    writer.WriteLine($"v {x:F6} {vertex.Y:F6} {vertex.Z:F6}");
                }
                
                writer.WriteLine();
                
                // Write object group
                writer.WriteLine($"g {obj.Name}");
                writer.WriteLine($"usemtl building_{obj.GroupKey:X2}");
                
                // Write faces with vertex index offset
                foreach (var triangle in obj.Triangles)
                {
                    if (vertexMap.ContainsKey(triangle.A) && 
                        vertexMap.ContainsKey(triangle.B) && 
                        vertexMap.ContainsKey(triangle.C))
                    {
                        var a = vertexMap[triangle.A] + 1 + totalVertices; // OBJ indices are 1-based
                        var b = vertexMap[triangle.B] + 1 + totalVertices;
                        var c = vertexMap[triangle.C] + 1 + totalVertices;
                        writer.WriteLine($"f {a} {b} {c}");
                    }
                }
                
                writer.WriteLine();
                totalVertices += vertices.Count;
                exportedCount++;
            }
        }
        
        // Write material file
        WriteMaterialFile(groupedObjects, mtlFile);
        
        return exportedCount;
    }
    
    /// <summary>
    /// Groups the entire scene as a single object.
    /// </summary>
    private List<GroupedObject> GroupAsWholeScene()
    {
        var result = new List<GroupedObject>();
        
        if (_scene.MSPI == null || _scene.MSPI.Indices == null || _scene.MSPI.Indices.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MSPI indices found to export");
            }
            return result;
        }
        
        // Collect all triangles from MSPI
        var triangles = new List<(int A, int B, int C)>();
        for (int i = 0; i + 2 < _scene.MSPI.Indices.Count; i += 3)
        {
            triangles.Add((
                _scene.MSPI.Indices[i],
                _scene.MSPI.Indices[i + 1],
                _scene.MSPI.Indices[i + 2]
            ));
        }
        
        if (triangles.Count == 0 && _options.SkipEmptyObjects)
            return result;
        
        if (triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
            return result;
        
        var obj = new GroupedObject
        {
            Name = "combined_scene",
            Triangles = triangles,
            GroupKey = 0
        };
        
        result.Add(obj);
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Grouped entire scene as single object with {triangles.Count} triangles");
        }
        
        return result;
    }
    
    /// <summary>
    /// Groups objects by MSUR surface group key (FlagsOrUnknown_0x00).
    /// </summary>
    private List<GroupedObject> GroupByMsurSurfaceGroup()
    {
        var result = new List<GroupedObject>();
        
        if (_scene.MSUR == null || _scene.MSUR.Entries == null || _scene.MSUR.Entries.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MSUR entries found to group by surface group key");
            }
            return result;
        }
        
        // Group MSUR entries by SurfaceGroupKey (FlagsOrUnknown_0x00)
        var objectsByGroupKey = _scene.MSUR.Entries
            .Where(entry => entry != null)
            .GroupBy(entry => entry.FlagsOrUnknown_0x00)
            .ToList();
        
        foreach (var objectGroup in objectsByGroupKey)
        {
            var triangles = new List<(int A, int B, int C)>();
            
            // Collect triangles from all MSUR entries with this group key
            foreach (var msurEntry in objectGroup)
            {
                var entryTriangles = CollectTriangles(new List<MsurChunk.Entry> { msurEntry });
                triangles.AddRange(entryTriangles);
            }
            
            if (triangles.Count == 0 && _options.SkipEmptyObjects)
                continue;
            
            if (triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            var obj = new GroupedObject
            {
                Name = $"surface_group_{objectGroup.Key}",
                Triangles = triangles,
                GroupKey = objectGroup.Key
            };
            
            result.Add(obj);
        }
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Grouped {result.Count} objects by MSUR surface group key");
        }
        
        return result;
    }
    
    /// <summary>
    /// Groups objects by MPRR sentinel values (Value1=65535).
    /// </summary>
    private List<GroupedObject> GroupByMprrSentinel()
    {
        var result = new List<GroupedObject>();
        
        if (_scene.MPRR == null || _scene.MPRR.Entries == null || _scene.MPRR.Entries.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MPRR entries found to group by sentinel values");
            }
            return result;
        }
        
        // Find sentinel positions (Value1=65535)
        var sentinelPositions = _scene.MPRR.Entries
            .Select((entry, index) => new { entry, index })
            .Where(x => x.entry.Value1 == 65535)
            .Select(x => x.index)
            .ToList();
        
        if (sentinelPositions.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MPRR sentinels found, falling back to single object");
            }
            
            // Fallback: treat entire MPRR as one object
            var triangles = CollectTriangles(_scene.MSUR?.Entries ?? new List<MsurChunk.Entry>());
            
            if (triangles.Count >= _options.MinTriangles || !_options.SkipEmptyObjects)
            {
                result.Add(new GroupedObject
                {
                    Name = "mprr_combined",
                    Triangles = triangles,
                    GroupKey = 0
                });
            }
            
            return result;
        }
        
        // Create object groups between sentinels
        for (int i = 0; i < sentinelPositions.Count; i++)
        {
            int startIdx = sentinelPositions[i] + 1;
            int endIdx = (i + 1 < sentinelPositions.Count) ? sentinelPositions[i + 1] : _scene.MPRR.Entries.Count;
            
            if (startIdx >= _scene.MPRR.Entries.Count)
                continue;
            
            // Get corresponding MSUR entries for this range
            var msurEntries = _scene.MSUR?.Entries
                .Where((entry, index) => index >= startIdx && index < endIdx)
                .ToList() ?? new List<MsurChunk.Entry>();
            
            var triangles = CollectTriangles(msurEntries);
            
            if (triangles.Count == 0 && _options.SkipEmptyObjects)
                continue;
            
            if (triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            var obj = new GroupedObject
            {
                Name = $"mprr_object_{i}",
                Triangles = triangles,
                GroupKey = (uint)i
            };
            
            result.Add(obj);
        }
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Grouped {result.Count} objects by MPRR sentinel values");
        }
        
        return result;
    }
    
    /// <summary>
    /// Groups objects by tile index.
    /// </summary>
    private List<GroupedObject> GroupByTile()
    {
        var result = new List<GroupedObject>();
        
        if (_scene.MSLK == null || _scene.MSLK.Entries == null || _scene.MSLK.Entries.Count == 0)
        {
            if (_options.Verbose)
            {
                ConsoleLogger.WriteLine("No MSLK entries found to group by tile index");
            }
            return result;
        }
        
        // Group MSLK entries by tile index
        var objectsByTile = _scene.MSLK.Entries
            .Where(entry => entry != null)
            .GroupBy(entry => GetTileIndex(entry))
            .ToList();
        
        foreach (var tileGroup in objectsByTile)
        {
            var triangles = new List<(int A, int B, int C)>();
            
            // Collect triangles from all MSLK entries in this tile
            foreach (var mslkEntry in tileGroup)
            {
                var entryTriangles = ExtractTrianglesForMslkEntry(mslkEntry);
                triangles.AddRange(entryTriangles);
            }
            
            if (triangles.Count == 0 && _options.SkipEmptyObjects)
                continue;
            
            if (triangles.Count < _options.MinTriangles && _options.SkipEmptyObjects)
                continue;
            
            var obj = new GroupedObject
            {
                Name = $"tile_{tileGroup.Key}",
                Triangles = triangles,
                GroupKey = tileGroup.Key
            };
            
            result.Add(obj);
        }
        
        if (_options.Verbose)
        {
            ConsoleLogger.WriteLine($"Grouped {result.Count} objects by tile index");
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
    internal class GroupedObject
    {
        /// <summary>
        /// Gets or sets the object name.
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the triangles that belong to this object.
        /// </summary>
        public List<(int A, int B, int C)> Triangles { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the group key used for material assignment.
        /// </summary>
        public uint GroupKey { get; set; }
    }
}
*/
