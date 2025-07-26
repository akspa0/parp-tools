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
            var objects = GroupObjectsByParentIndex();
            var exportedCount = 0;
            
            foreach (var obj in objects)
            {
                // Skip objects with too few triangles
                if (obj.Triangles.Count < _options.MinTriangles)
                    continue;
                    
                // Skip M2 objects if not requested
                if (obj.ParentIndex == 0 && !_options.IncludeM2Objects)
                    continue;
                
                var filename = Path.Combine(outputDirectory, $"Object_Group_{obj.ParentIndex}.obj");
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
            var objects = GroupObjectsByParentIndex();
            var exportedCount = 0;
            
            foreach (var obj in objects)
            {
                // Skip objects with too few triangles
                if (obj.Triangles.Count < _options.MinTriangles)
                    continue;
                    
                // Skip M2 objects if not requested
                if (obj.ParentIndex == 0 && !_options.IncludeM2Objects)
                    continue;
                
                var filename = Path.Combine(outputDirectory, $"Object_Group_{obj.ParentIndex}.wmo");
                ExportObjectAsWmo(obj, filename);
                exportedCount++;
            }
            
            return exportedCount;
        }
        
        /// <summary>
        /// Groups PM4 objects by MSLK.ParentIndex_0x04 as the correct object identifier.
        /// </summary>
        /// <returns>List of grouped objects.</returns>
        private List<Pm4Object> GroupObjectsByParentIndex()
        {
            var objects = new List<Pm4Object>();
            
            // Build MPRL to MSLK mappings if MPRL transformations are enabled
            var mprlMslkMap = BuildMprlMslkMappings();
            
            if (_scene.Links == null || _scene.Links.Count == 0)
                return objects;
                
            // Group MSLK entries by ParentIndex
            var entriesByParentIndex = _scene.Links
                .Where(link => link.MspiFirstIndex >= 0) // Skip container nodes
                .GroupBy(link => link.ParentIndex)
                .ToDictionary(g => g.Key, g => g.ToList());
                
            foreach (var kvp in entriesByParentIndex)
            {
                var parentIndex = kvp.Key;
                var entries = kvp.Value;
                
                var triangles = new List<(int A, int B, int C)>();
                var vertexIndices = new HashSet<int>();
                
                // Collect all triangles from entries with this ParentIndex
                foreach (var entry in entries)
                {
                    var entryTriangles = ExtractTrianglesForMslkEntry(entry);
                    triangles.AddRange(entryTriangles);
                    
                    // Collect unique vertex indices
                    foreach (var (a, b, c) in entryTriangles)
                    {
                        vertexIndices.Add(a);
                        vertexIndices.Add(b);
                        vertexIndices.Add(c);
                    }
                }
                
                // Create vertex mapping for this object
                var vertexMap = vertexIndices
                    .OrderBy(idx => idx)
                    .Select((originalIdx, newIndex) => new { originalIdx, newIndex })
                    .ToDictionary(x => x.originalIdx, x => x.newIndex);
                
                // Remap triangle indices
                var remappedTriangles = triangles
                    .Select(t => (vertexMap[t.A], vertexMap[t.B], vertexMap[t.C]))
                    .ToList();
                
                // Create remapped vertices
                var remappedVertices = vertexIndices
                    .OrderBy(idx => idx)
                    .Select(idx => _scene.Vertices[idx])
                    .ToList();
                
                // Apply MPRL transformations if available
                if (_scene.Placements != null && mprlMslkMap.ContainsKey(parentIndex))
                {
                    var transforms = mprlMslkMap[parentIndex];
                    foreach (var transform in transforms)
                    {
                        // Apply transformation to all vertices in this object
                        for (int i = 0; i < remappedVertices.Count; i++)
                        {
                            var vertex = remappedVertices[i];
                            remappedVertices[i] = new Vector3(
                                vertex.X + transform.X,
                                vertex.Y + transform.Y,
                                vertex.Z + transform.Z
                            );
                        }
                    }
                }
                
                // Apply coordinate transformation if needed
                if (_options.ApplyXAxisInversion)
                {
                    for (int i = 0; i < remappedVertices.Count; i++)
                    {
                        var vertex = remappedVertices[i];
                        remappedVertices[i] = new Vector3(-vertex.X, vertex.Y, vertex.Z);
                    }
                }
                
                objects.Add(new Pm4Object
                {
                    ParentIndex = parentIndex,
                    Triangles = remappedTriangles,
                    Vertices = remappedVertices
                });
            }
            
            return objects;
        }
        
        /// <summary>
        /// Extracts triangles for a single MSLK entry.
        /// </summary>
        /// <param name="mslkEntry">MSLK entry to extract triangles from.</param>
        /// <returns>List of triangles.</returns>
        private List<(int A, int B, int C)> ExtractTrianglesForMslkEntry(MslkEntry mslkEntry)
        {
            var triangles = new List<(int A, int B, int C)>();
            
            if (mslkEntry.MspiFirstIndex < 0 || mslkEntry.MspiIndexCount <= 0)
                return triangles;
                
            // Ensure we don't go out of bounds
            var maxIndex = mslkEntry.MspiFirstIndex + mslkEntry.MspiIndexCount;
            if (maxIndex > _scene.Triangles.Count)
                maxIndex = _scene.Triangles.Count;
                
            // Track out of bounds count for logging
            int outOfBoundsCount = 0;
                
            for (int i = mslkEntry.MspiFirstIndex; i < maxIndex; i++)
            {
                var triangle = _scene.Triangles[i];
                
                // Validate indices
                if (triangle.Item1 >= 0 && triangle.Item1 < _scene.Vertices.Count &&
                    triangle.Item2 >= 0 && triangle.Item2 < _scene.Vertices.Count &&
                    triangle.Item3 >= 0 && triangle.Item3 < _scene.Vertices.Count)
                {
                    // Additional validation to check for (0,0,0) vertices
                    var vertexA = _scene.Vertices[triangle.Item1];
                    var vertexB = _scene.Vertices[triangle.Item2];
                    var vertexC = _scene.Vertices[triangle.Item3];
                    
                    // Check if all vertices are at (0,0,0) which indicates invalid data
                    if (vertexA == Vector3.Zero && vertexB == Vector3.Zero && vertexC == Vector3.Zero)
                    {
                        outOfBoundsCount++;
                        if (outOfBoundsCount <= 5) // Only log first 5 to avoid spam
                        {
                            ConsoleLogger.WriteLine($"Skipping triangle with all zero vertices: indices {triangle.Item1}, {triangle.Item2}, {triangle.Item3}");
                        }
                        continue; // Skip this triangle
                    }
                    
                    triangles.Add(triangle);
                }
                else
                {
                    outOfBoundsCount++;
                    if (outOfBoundsCount <= 10) // Only log first 10 to avoid spam
                    {
                        ConsoleLogger.WriteLine($"Skipping triangle with invalid vertex indices: A={triangle.Item1}, B={triangle.Item2}, C={triangle.Item3} (max: {_scene.Vertices.Count - 1})");
                    }
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
