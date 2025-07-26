using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using ParpToolbox.Formats.P4.Chunks.Common;
using ParpToolbox.Formats.PM4;
using ParpToolbox.Services.WMO;
using ParpToolbox.Utils;
using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.WMO;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Unified PM4 exporter with correct object grouping based on MSLK.ParentIndex_0x04.
    /// </summary>
    public class NewPm4Exporter
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
            
            /// <summary>
            /// World of Warcraft WMO format.
            /// </summary>
            Wmo
        }
        
        /// <summary>
        /// Export options configuration.
        /// </summary>
        public class ExportOptions
        {
            /// <summary>
            /// Export format to use. Default is OBJ.
            /// </summary>
            public ExportFormat Format { get; set; } = ExportFormat.Obj;
            
            /// <summary>
            /// Minimum number of triangles for an object to be exported. Default is 10.
            /// </summary>
            public int MinTriangles { get; set; } = 10;
            
            /// <summary>
            /// Whether to apply X-axis inversion for proper orientation. Default is true.
            /// </summary>
            public bool ApplyXAxisInversion { get; set; } = true;
            
            /// <summary>
            /// Whether to include M2 objects (group key 0x00000000). Default is false.
            /// </summary>
            public bool IncludeM2Objects { get; set; } = false;
            
            /// <summary>
            /// Whether to apply MPRL placement transformations for correct world positioning. Default is false.
            /// </summary>
            public bool EnableMprlTransformations { get; set; } = false;
            
            /// <summary>
            /// Whether to enable cross-tile vertex resolution for complete geometry. Default is false.
            /// </summary>
            public bool EnableCrossTileResolution { get; set; } = false;
        }
        
        private readonly Pm4Scene _scene;
        private readonly ExportOptions _options;
        
        public NewPm4Exporter(Pm4Scene scene, ExportOptions options)
        {
            _scene = scene ?? throw new ArgumentNullException(nameof(scene));
            _options = options ?? throw new ArgumentNullException(nameof(options));
        }
        
        /// <summary>
        /// Creates a new exporter with cross-tile resolution enabled.
        /// </summary>
        /// <param name="directoryPath">Directory containing PM4 files to load as a unified scene.</param>
        /// <param name="options">Export options.</param>
        public NewPm4Exporter(string directoryPath, ExportOptions options)
        {
            if (string.IsNullOrWhiteSpace(directoryPath))
                throw new ArgumentException("Directory path must be provided", nameof(directoryPath));
                
            _options = options ?? throw new ArgumentNullException(nameof(options));
            
            // Load global scene if cross-tile resolution is enabled
            if (_options.EnableCrossTileResolution)
            {
                ConsoleLogger.WriteLine($"Loading global scene from directory: {directoryPath}");
                var globalScene = Pm4GlobalTileLoader.LoadRegion(directoryPath);
                _scene = Pm4GlobalTileLoader.ToStandardScene(globalScene);
                ConsoleLogger.WriteLine($"Loaded global scene with {globalScene.GlobalVertices.Count} vertices from {globalScene.TotalLoadedTiles} tiles");
            }
            else
            {
                // Load single file scene
                var adapter = new Pm4Adapter();
                _scene = adapter.Load(directoryPath);
                ConsoleLogger.WriteLine($"Loaded single scene with {_scene.Vertices.Count} vertices");
            }
        }
        
        /// <summary>
        /// Transforms an MPRL entry position to world coordinates.
        /// Based on legacy coordinate transformation: (-Z, Y, X) mapping.
        /// </summary>
        /// <param name="position">The raw MPRL position vector.</param>
        /// <returns>Transformed position in world coordinates.</returns>
        private static Vector3 TransformMprlPosition(Vector3 position)
        {
            // Legacy coordinate transformation: (-Z, Y, X) mapping
            return new Vector3(-position.Z, position.Y, position.X);
        }
        
        /// <summary>
        /// Builds MPRL to MSLK mapping based on cross-reference patterns.
        /// Maps MPRL.Unknown4 values to MSLK.ParentIndex values for transformation application.
        /// </summary>
        /// <returns>Dictionary mapping ParentIndex to list of MPRL position transforms.</returns>
        private Dictionary<uint, List<Vector3>> BuildMprlMslkMappings()
        {
            var mappings = new Dictionary<uint, List<Vector3>>();
            
            // Only build mappings if MPRL transformations are enabled
            if (!_options.EnableMprlTransformations || _scene.Placements == null || _scene.Links == null)
                return mappings;
            
            // Map MPRL.Unknown4 values to MSLK.ParentIndex values
            var mslkLookup = _scene.Links.ToDictionary(l => l.ParentIndex, l => l);
            
            foreach (var mprl in _scene.Placements)
            {
                var parentIndex = mprl.Unknown4;
                if (mslkLookup.TryGetValue(parentIndex, out var mslk))
                {
                    var position = TransformMprlPosition(mprl.Position);
                    if (!mappings.ContainsKey(mslk.ParentIndex))
                    {
                        mappings[mslk.ParentIndex] = new List<Vector3>();
                    }
                    mappings[mslk.ParentIndex].Add(position);
                }
            }
            
            return mappings;
        }
        
        /// <summary>
        /// Gets the ParentIndex_0x04 value from an MSLK entry.
        /// This is the correct object identifier for grouping.
        /// </summary>
        /// <param name="mslk">MSLK entry to extract ParentIndex from.</param>
        /// <returns>ParentIndex value.</returns>
        private uint GetMslkParentIndex(object mslk)
        {
            // Cast to the correct type and access the ParentIndex property
            if (mslk is ParpToolbox.Formats.P4.Chunks.Common.MslkEntry mslkEntry)
            {
                return mslkEntry.ParentIndex;
            }
            
            // Fallback to default value if casting fails
            return 0;
        }
        
        /// <summary>
        /// Exports the PM4 scene according to the specified options.
        /// </summary>
        /// <param name="outputDirectory">Directory to export files to.</param>
        /// <returns>Number of objects exported.</returns>
        public int Export(string outputDirectory)
        {
            if (string.IsNullOrWhiteSpace(outputDirectory))
                throw new ArgumentException("Output directory must be provided", nameof(outputDirectory));
                
            Directory.CreateDirectory(outputDirectory);
            
            switch (_options.Format)
            {
                case ExportFormat.Obj:
                    return ExportAsObj(outputDirectory);
                case ExportFormat.Wmo:
                    return ExportAsWmo(outputDirectory);
                default:
                    throw new NotSupportedException($"Export format {_options.Format} is not supported");
            }
        }
        
        /// <summary>
        /// Exports the PM4 scene as OBJ files.
        /// </summary>
        /// <param name="outputDirectory">Directory to export files to.</param>
        /// <returns>Number of objects exported.</returns>
        private int ExportAsObj(string outputDirectory)
        {
            Directory.CreateDirectory(outputDirectory);
            
            var objects = GroupObjectsByIndexCount();
            
            int exportedCount = 0;
            
            foreach (var obj in objects)
            {
                // Skip objects with too few triangles
                if (obj.Triangles.Count < _options.MinTriangles)
                    continue;
                    
                // Skip M2 objects if not included
                if (obj.ParentIndex == 0 && !_options.IncludeM2Objects)
                    continue;
                    
                var filename = Path.Combine(outputDirectory, $"Object_{obj.ParentIndex}.obj");
                ExportObjectAsObj(obj, filename);
                exportedCount++;
            }
            
            return exportedCount;
        }
        
        /// <summary>
        /// Exports the PM4 scene as WMO files.
        /// </summary>
        /// <param name="outputDirectory">Directory to export files to.</param>
        /// <returns>Number of objects exported.</returns>
        private int ExportAsWmo(string outputDirectory)
        {
            Directory.CreateDirectory(outputDirectory);
            
            var objects = GroupObjectsByIndexCount();
            var exportedCount = 0;
            
            foreach (var obj in objects)
            {
                // Skip objects with too few triangles
                if (obj.Triangles.Count < _options.MinTriangles)
                    continue;
                    
                // Skip M2 objects if not included
                if (obj.ParentIndex == 0 && !_options.IncludeM2Objects)
                    continue;
                
                var filename = Path.Combine(outputDirectory, $"Object_{obj.ParentIndex}.wmo");
                ExportObjectAsWmo(obj, filename);
                exportedCount++;
            }
            
            return exportedCount;
        }
        
        /// <summary>
        /// Groups PM4 objects by MSUR.IndexCount (0x01 field) which has been confirmed as the authoritative per-object identifier. SurfaceGroupKey (Unknown_0x00) is preserved in logs for diagnostic purposes.
        /// </summary>
        /// <returns>List of grouped objects.</returns>
        private List<Pm4Object> GroupObjectsByIndexCount()
        {
            var objects = new List<Pm4Object>();
            
            // Build MPRL to MSLK mappings if MPRL transformations are enabled
            var mprlMslkMap = BuildMprlMslkMappings();
            
            if (_scene.Surfaces == null || _scene.Surfaces.Count == 0)
            {
                ConsoleLogger.WriteLine("No MSUR surfaces found in scene");
                return objects;
            }
                
            // Group MSUR surfaces by IndexCount (0x01) – confirmed full-object grouping field
            var surfacesByGroup = _scene.Surfaces
                .Where(s => s.MsviFirstIndex >= 0 && s.IndexCount > 0)
                .GroupBy(s => (int)s.IndexCount)
                .ToDictionary(g => g.Key, g => g.ToList());
                
            ConsoleLogger.WriteLine($"Found {surfacesByGroup.Count} unique IndexCount groups");
            foreach (var kvp in surfacesByGroup)
            {
                ConsoleLogger.WriteLine($"IndexCount {kvp.Key}: {kvp.Value.Count} surfaces (SurfaceGroupKeys: {string.Join(",", kvp.Value.Select(v=>v.SurfaceGroupKey).Distinct())})");
            }
                
            foreach (var kvp in surfacesByGroup)
            {
                int groupKey = kvp.Key;
                var surfaces = kvp.Value;
                
                // Collect all triangles and vertices for this object
                var allTriangles = new List<(int A, int B, int C)>();
                var vertexMap = new Dictionary<uint, int>(); // Maps global vertex indices to local indices
                var remappedVertices = new List<Vector3>();
                
                int vertexCount = _scene.Vertices?.Count ?? 0;
                
                // Extract triangles from all surfaces in this group
                foreach (var surface in surfaces)
                {
                    uint firstIndex = surface.MsviFirstIndex;
                    uint lastIndex = firstIndex + surface.IndexCount;
                    
                    // Extract triangles (groups of 3 indices)
                    for (uint i = firstIndex; i + 2 < lastIndex; i += 3)
                    {
                        // Validate indices exist
                        if (i + 2 >= _scene.Indices.Count)
                            break;
                            
                        int a = _scene.Indices[(int)i];
                        int b = _scene.Indices[(int)i + 1];
                        int c = _scene.Indices[(int)i + 2];
                        
                        allTriangles.Add((a, b, c));
                    }
                }
                
                // Remap global vertex indices to local indices for this object
                var remappedTriangles = new List<(int A, int B, int C)>();
                
                // Process all triangles for this object
                int outOfBoundsCount = 0;
                foreach (var triangle in allTriangles)
                {
                    // Skip triangles with all zero vertices (invalid data)
                    if (triangle.Item1 == 0 && triangle.Item2 == 0 && triangle.Item3 == 0)
                    {
                        ConsoleLogger.WriteLine($"Skipping triangle with all zero vertices: indices {triangle.Item1}, {triangle.Item2}, {triangle.Item3}");
                        continue; // Skip this triangle
                    }
                    
                    // Resolve and remap vertices
                    var remappedTriangle = new int[3];
                    bool isValid = true;
                    
                    for (int i = 0; i < 3; i++)
                    {
                        uint rawIdx = i switch
                        {
                            0 => (uint)triangle.Item1,
                            1 => (uint)triangle.Item2,
                            2 => (uint)triangle.Item3,
                            _ => 0
                        };
                        
                        // Check if we've already mapped this vertex
                        if (vertexMap.TryGetValue(rawIdx, out int localIdx))
                        {
                            remappedTriangle[i] = localIdx;
                            continue;
                        }
                        
                        // Try to resolve the vertex
                        if (TryResolveVertex(rawIdx, vertexCount, _scene, out Vector3 vertex))
                        {
                            localIdx = remappedVertices.Count;
                            remappedVertices.Add(vertex);
                            vertexMap[rawIdx] = localIdx;
                            remappedTriangle[i] = localIdx;
                        }
                        else
                        {
                            isValid = false;
                            outOfBoundsCount++;
                            if (outOfBoundsCount <= 10) // Only log first 10 to avoid spam
                            {
                                ConsoleLogger.WriteLine($"Skipping triangle with invalid vertex index: {rawIdx} (max: {vertexCount - 1})");
                            }
                            break;
                        }
                    }
                    
                    if (isValid)
                    {
                        remappedTriangles.Add((remappedTriangle[0], remappedTriangle[1], remappedTriangle[2]));
                    }
                }
                
                // Apply MPRL transformations if enabled
                // Note: In the reference implementation, MPRL transformations are applied differently
                // For now, we'll keep the existing logic but it may need adjustment
                if (_options.EnableMprlTransformations && mprlMslkMap.TryGetValue((uint)groupKey, out var transforms))
                {
                    foreach (var transform in transforms)
                    {
                        for (int i = 0; i < remappedVertices.Count; i++)
                        {
                            remappedVertices[i] = TransformMprlPosition(remappedVertices[i] + transform);
                        }
                    }
                }
                
                // Apply X-axis inversion if enabled
                if (_options.ApplyXAxisInversion)
                {
                    for (int i = 0; i < remappedVertices.Count; i++)
                    {
                        var v = remappedVertices[i];
                        remappedVertices[i] = new Vector3(-v.X, v.Y, v.Z);
                    }
                }
                
                objects.Add(new Pm4Object
                {
                    ParentIndex = (uint)groupKey, // Using IndexCount as identifier
                    Triangles = remappedTriangles,
                    Vertices = remappedVertices
                });
            }
            
            return objects;
        }
        
        /// <summary>
        /// Attempts to resolve a vertex index to a Vector3 position.
        /// </summary>
        /// <param name="rawIdx">Raw vertex index from MSVI.</param>
        /// <param name="msvtCount">Count of MSVT vertices.</param>
        /// <param name="scene">Scene containing vertex data.</param>
        /// <param name="vertex">Output vertex position.</param>
        /// <returns>True if vertex was successfully resolved.</returns>
        private bool TryResolveVertex(uint rawIdx, int msvtCount, Pm4Scene scene, out Vector3 vertex)
        {
            vertex = Vector3.Zero;
            
            // Try MSVT vertices first (high-resolution)
            if (scene.Vertices != null && rawIdx < (uint)scene.Vertices.Count)
            {
                vertex = scene.Vertices[(int)rawIdx];
                return true;
            }

            // Reject clearly out-of-range indices early
            if (rawIdx >= (uint)scene.Vertices.Count && rawIdx < 0x80000000)
            {
                return false;
            }
            
            // Handle high/low pair encoding for 32-bit indices
            if (rawIdx >= 0x80000000 && scene.Vertices != null)
            {
                uint highPart = (rawIdx >> 16) & 0x7FFF;
                uint lowPart = rawIdx & 0xFFFF;
                uint combinedIdx = (highPart << 16) | lowPart;
                
                if (combinedIdx < (uint)scene.Vertices.Count)
                {
                    vertex = scene.Vertices[(int)combinedIdx];
                    return true;
                }
            }
            
            return false;
        }
        
        /// <summary>
        /// Extracts triangles for a single MSLK entry.
        /// </summary>
        /// <param name="mslkEntry">MSLK entry to extract triangles from.</param>
        /// <returns>List of triangles.</returns>
        private List<(int A, int B, int C)> ExtractTrianglesForMslkEntry(MslkEntry mslkEntry)
        {
            var triangles = new List<(int A, int B, int C)>();
            
            if (mslkEntry.MspiFirstIndex < 0 || mslkEntry.MspiIndexCount == 0)
                return triangles; // Skip container nodes
                
            int firstIndex = mslkEntry.MspiFirstIndex;
            int lastIndex = firstIndex + mslkEntry.MspiIndexCount;
            
            // Ensure we don't go out of bounds
            if (_scene.Indices == null || lastIndex > _scene.Indices.Count)
                lastIndex = _scene.Indices?.Count ?? 0;
                
            // Extract triangles (groups of 3 indices)
            for (int i = firstIndex; i + 2 < lastIndex; i += 3)
            {
                int a = (int)_scene.Indices[i];
                int b = (int)_scene.Indices[i + 1];
                int c = (int)_scene.Indices[i + 2];
                
                // Validate indices
                if (a >= 0 && a < _scene.Vertices.Count &&
                    b >= 0 && b < _scene.Vertices.Count &&
                    c >= 0 && c < _scene.Vertices.Count)
                {
                    triangles.Add((a, b, c));
                }
                else
                {
                    // Log out of bounds indices
                    ConsoleLogger.WriteLine($"Skipping triangle with out of bounds indices: {a}, {b}, {c} (vertex count: {_scene.Vertices.Count})");
                }
            }
            
            return triangles;
        }
        
        /// <summary>
        /// Exports a single PM4 object as an OBJ file.
        /// </summary>
        /// <param name="obj">Object to export.</param>
        /// <param name="filename">Output filename.</param>
        private void ExportObjectAsObj(Pm4Object obj, string filename)
        {
            using var writer = new StreamWriter(filename);
            writer.WriteLine($"# PM4 Object Export - ParentIndex: {obj.ParentIndex}");
            writer.WriteLine($"# Triangle Count: {obj.Triangles.Count}");
            writer.WriteLine($"# Vertex Count: {obj.Vertices.Count}");
            writer.WriteLine();
            
            // Write vertices
            foreach (var vertex in obj.Vertices)
            {
                writer.WriteLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }
            
            writer.WriteLine();
            
            // Write faces
            foreach (var (a, b, c) in obj.Triangles)
            {
                // OBJ indices are 1-based
                writer.WriteLine($"f {a + 1} {b + 1} {c + 1}");
            }
        }
        
        /// <summary>
        /// Exports a single PM4 object as a WMO file using wow.tools.local.
        /// </summary>
        /// <param name="obj">Object to export.</param>
        /// <param name="filename">Output filename.</param>
        private void ExportObjectAsWmo(Pm4Object obj, string filename)
        {
            try
            {
                // Create a basic WMO structure from the PM4 object
                var wmo = CreateWmoFromPm4Object(obj);
                
                // For now, we'll export as OBJ since direct WMO writing isn't implemented in wow.tools.local
                // In a full implementation, we would use WMO writing functionality
                ExportObjectAsObj(obj, filename.Replace(".wmo", ".obj"));
                
                ConsoleLogger.WriteLine($"Exported PM4 object {obj.ParentIndex} as WMO (currently as OBJ due to WMO writer limitations)");
            }
            catch (Exception ex)
            {
                ConsoleLogger.WriteLine($"Failed to export PM4 object {obj.ParentIndex} as WMO: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Creates a basic WMO structure from a PM4 object.
        /// </summary>
        /// <param name="obj">PM4 object to convert.</param>
        /// <returns>Basic WMO structure.</returns>
        private WoWFormatLib.Structs.WMO.WMO CreateWmoFromPm4Object(Pm4Object obj)
        {
            // This is a simplified implementation that creates a basic WMO structure
            // A full implementation would need to properly set up all WMO chunks
            
            var wmo = new WoWFormatLib.Structs.WMO.WMO
            {
                version = 17,
                header = new MOHD
                {
                    nMaterials = 1,
                    nGroups = 1,
                    nPortals = 0,
                    nLights = 0,
                    nDoodads = 0,
                    nSets = 0,
                    ambientColor = 0,
                    areaTableID = 0,
                    boundingBox1 = new Vector3(0, 0, 0),
                    boundingBox2 = new Vector3(0, 0, 0),
                    flags = 0,
                    nLod = 0
                },
                materials = new MOMT[1],
                group = new WMOGroupFile[1]
            };
            
            // Set up a basic material
            wmo.materials[0] = new MOMT
            {
                flags = 0,
                shader = 0,
                blendMode = 0,
                texture1 = 0,
                texture2 = 0,
                texture3 = 0,
                color1 = 0,
                color2 = 0,
                color3 = 0,
                groundType = 0,
                runtimeData0 = 0,
                runtimeData1 = 0,
                runtimeData2 = 0,
                runtimeData3 = 0
            };
            
            // Set up a basic group
            wmo.group[0] = new WMOGroupFile
            {
                mogp = new MOGP
                {
                    nameOffset = 0,
                    descriptiveNameOffset = 0,
                    flags = 0,
                    boundingBox1 = new Vector3(0, 0, 0),
                    boundingBox2 = new Vector3(0, 0, 0),
                    ofsPortals = 0,
                    numPortals = 0,
                    numBatchesA = 0,
                    numBatchesB = 0,
                    numBatchesC = 0,
                    numBatchesD = 0,
                    fogIndices_0 = 0,
                    fogIndices_1 = 0,
                    fogIndices_2 = 0,
                    fogIndices_3 = 0,
                    liquidType = 0,
                    groupID = 0,
                    flags2 = 0,
                    parentOrFirstChildSplitGroupindex = 0,
                    nextSplitChildGroupIndex = 0,
                    unused = 0,
                    materialInfo = new MOPY[0],
                    indices = obj.Triangles.SelectMany(t => new ushort[] { (ushort)t.A, (ushort)t.B, (ushort)t.C }).ToArray(),
                    vertices = obj.Vertices.Select(v => new MOVT { vector = v }).ToArray(),
                    normals = new MONR[0],
                    textureCoords = new MOTV[0][],
                    renderBatches = new MOBA[1],
                    shadowBatches = new MOBS[0],
                    bspNodes = new MOBN[0],
                    bspIndices = new ushort[0]
                }
            };
            
            // Set up a basic render batch
            wmo.group[0].mogp.renderBatches[0] = new MOBA
            {
                possibleBox1_1 = 0,
                possibleBox1_2 = 0,
                possibleBox1_3 = 0,
                possibleBox2_1 = 0,
                possibleBox2_2 = 0,
                possibleBox2_3 = 0,
                firstFace = 0,
                numFaces = (ushort)obj.Triangles.Count,
                firstVertex = 0,
                lastVertex = 0,
                flags = 0,
                materialID = 0
            };
            
            return wmo;
        }
    }
    
    /// <summary>
    /// Represents a PM4 object grouped by MSLK.ParentIndex_0x04.
    /// </summary>
    internal class Pm4Object
    {
        /// <summary>
        /// The ParentIndex that identifies this object.
        /// </summary>
        public uint ParentIndex { get; set; }
        
        /// <summary>
        /// Triangles that belong to this object.
        /// </summary>
        public List<(int A, int B, int C)> Triangles { get; set; } = new();
        
        /// <summary>
        /// Vertices that belong to this object.
        /// </summary>
        public List<Vector3> Vertices { get; set; } = new();
    }
}
