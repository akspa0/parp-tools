using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PM4.Chunks
{
    /// <summary>
    /// Represents a MSPV (Position Vertices) chunk in a PM4 file
    /// Contains position vertices for the model
    /// </summary>
    public class MspvChunk : PM4Chunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MSPV";
        
        /// <summary>
        /// Size of a single position vertex in bytes (X,Y,Z as 3 floats = 12 bytes)
        /// </summary>
        private const int VERTEX_SIZE = 12; // 3 * sizeof(float)
        
        /// <summary>
        /// Gets the list of vertices
        /// </summary>
        public List<Vector3> Vertices { get; private set; } = new List<Vector3>();
        
        /// <summary>
        /// Creates a new MSPV chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MspvChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is multiple of VERTEX_SIZE
            if (data.Length % VERTEX_SIZE != 0)
            {
                logger?.LogWarning($"MSPV chunk has irregular size: {data.Length} bytes. Not a multiple of {VERTEX_SIZE} bytes per vertex.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSPV chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MspvChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Parses the chunk data
        /// </summary>
        /// <returns>True if parsing succeeded, false otherwise</returns>
        public override bool Parse()
        {
            try
            {
                if (Data.Length < 4)
                {
                    LogWarning($"MSPV chunk data is too small: {Data.Length} bytes");
                    return false;
                }
                
                Vertices.Clear();
                
                using (MemoryStream ms = new MemoryStream(Data))
                using (BinaryReader reader = new BinaryReader(ms))
                {
                    // Determine number of vertices - each vertex is 12 bytes (3 floats)
                    int vertexCount = Data.Length / VERTEX_SIZE;
                    LogInformation($"Vertex count: {vertexCount}");
                    
                    for (int i = 0; i < vertexCount; i++)
                    {
                        float x = reader.ReadSingle();
                        float y = reader.ReadSingle();
                        float z = reader.ReadSingle();
                        
                        // According to documentation, these need to be transformed for actual game coordinates
                        // We store them as-is and provide helpers for transformation
                        Vertices.Add(new Vector3(x, y, z));
                    }
                    
                    LogInformation($"Parsed {Vertices.Count} vertices");
                    
                    IsParsed = true;
                    return true;
                }
            }
            catch (Exception ex)
            {
                LogError($"Error parsing MSPV chunk: {ex.Message}");
                return false;
            }
        }
        
        /// <summary>
        /// Writes the chunk data
        /// </summary>
        /// <returns>Byte array containing the chunk data</returns>
        public override byte[] Write()
        {
            using (MemoryStream ms = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(ms))
            {                
                foreach (Vector3 vertex in Vertices)
                {
                    writer.Write(vertex.X);
                    writer.Write(vertex.Y);
                    writer.Write(vertex.Z);
                }
                
                return ms.ToArray();
            }
        }
        
        /// <summary>
        /// Returns a string representation of this chunk
        /// </summary>
        public override string ToString()
        {
            return $"{SIGNATURE} (Vertices: {Vertices.Count})";
        }
        
        /// <summary>
        /// Gets the number of vertices in this chunk
        /// </summary>
        /// <returns>Number of vertices</returns>
        public int GetVertexCount()
        {
            return Data.Length / VERTEX_SIZE;
        }
        
        /// <summary>
        /// Gets a vertex at the specified index
        /// </summary>
        /// <param name="index">Index of the vertex to retrieve</param>
        /// <returns>Vector3 containing vertex position</returns>
        public Vector3 GetVertex(int index)
        {
            int vertexCount = GetVertexCount();
            
            if (index < 0 || index >= vertexCount)
            {
                LogWarning($"Vertex index {index} out of range [0-{vertexCount - 1}]. Returning Zero vector.");
                return Vector3.Zero;
            }
            
            int offset = index * VERTEX_SIZE;
            
            float x = BitConverter.ToSingle(Data, offset);
            float y = BitConverter.ToSingle(Data, offset + 4);
            float z = BitConverter.ToSingle(Data, offset + 8);
            
            return new Vector3(x, y, z);
        }
        
        /// <summary>
        /// Adds a vertex to the chunk
        /// </summary>
        /// <param name="vertex">Vertex to add</param>
        public void AddVertex(Vector3 vertex)
        {
            byte[] newData = new byte[Data.Length + VERTEX_SIZE];
            
            // Copy existing data
            Array.Copy(Data, 0, newData, 0, Data.Length);
            
            // Add new vertex data
            int offset = Data.Length;
            Array.Copy(BitConverter.GetBytes(vertex.X), 0, newData, offset, 4);
            Array.Copy(BitConverter.GetBytes(vertex.Y), 0, newData, offset + 4, 4);
            Array.Copy(BitConverter.GetBytes(vertex.Z), 0, newData, offset + 8, 4);
            
            // Update data
            Data = newData;
        }
        
        /// <summary>
        /// Gets world coordinates for a vertex based on the transformation rules in the documentation
        /// </summary>
        /// <param name="vertex">The vertex to transform</param>
        /// <returns>Transformed vertex in world coordinates</returns>
        public static Vector3 GetWorldCoordinates(Vector3 vertex)
        {
            // Based on documentation: 
            // worldPos.y = 17066.666 - position.y;
            // worldPos.x = 17066.666 - position.x;
            // worldPos.z = position.z / 36.0f;
            return new Vector3(
                17066.666f - vertex.X,
                17066.666f - vertex.Y,
                vertex.Z / 36.0f
            );
        }
        
        /// <summary>
        /// Gets the bounding box of all vertices
        /// </summary>
        /// <returns>Tuple containing min and max points of the bounding box</returns>
        public (Vector3 Min, Vector3 Max) GetBoundingBox()
        {
            if (Vertices.Count == 0)
                return (Vector3.Zero, Vector3.Zero);
                
            Vector3 min = new Vector3(float.MaxValue, float.MaxValue, float.MaxValue);
            Vector3 max = new Vector3(float.MinValue, float.MinValue, float.MinValue);
            
            foreach (Vector3 vertex in Vertices)
            {
                min = Vector3.Min(min, vertex);
                max = Vector3.Max(max, vertex);
            }
            
            return (min, max);
        }
    }
} 