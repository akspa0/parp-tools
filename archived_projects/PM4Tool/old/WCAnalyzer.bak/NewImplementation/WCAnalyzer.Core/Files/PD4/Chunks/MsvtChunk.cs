using System;
using System.Numerics;
using Microsoft.Extensions.Logging;

namespace WCAnalyzer.Core.Files.PD4.Chunks
{
    /// <summary>
    /// MSVT chunk - Contains vertex data for the PD4 format
    /// Each vertex has position (X,Y,Z) and potentially normal/texture coordinates
    /// </summary>
    public class MsvtChunk : PD4Chunk
    {
        /// <summary>
        /// Signature for this chunk ("MSVT")
        /// </summary>
        public const string SIGNATURE = "MSVT";
        
        /// <summary>
        /// Size of a basic vertex in bytes (X,Y,Z as 3 floats = 12 bytes)
        /// </summary>
        private const int BASE_VERTEX_SIZE = 12; // 3 * sizeof(float)
        
        /// <summary>
        /// Creates a new MSVT chunk from raw data
        /// </summary>
        /// <param name="data">Raw chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MsvtChunk(byte[] data, ILogger? logger = null)
            : base(SIGNATURE, data, logger)
        {
            // Check if data size is multiple of BASE_VERTEX_SIZE
            if (_data.Length % BASE_VERTEX_SIZE != 0)
            {
                Logger?.LogWarning($"MSVT chunk has irregular size: {_data.Length} bytes. Not a multiple of {BASE_VERTEX_SIZE} bytes per vertex.");
            }
        }
        
        /// <summary>
        /// Creates a new empty MSVT chunk
        /// </summary>
        /// <param name="logger">Optional logger</param>
        public MsvtChunk(ILogger? logger = null)
            : base(SIGNATURE, Array.Empty<byte>(), logger)
        {
        }
        
        /// <summary>
        /// Gets the number of vertices in this chunk
        /// </summary>
        /// <returns>Number of vertices</returns>
        public int GetVertexCount()
        {
            return _data.Length / BASE_VERTEX_SIZE;
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
                Logger?.LogWarning($"Vertex index {index} out of range [0-{vertexCount - 1}]. Returning Zero vector.");
                return Vector3.Zero;
            }
            
            int offset = index * BASE_VERTEX_SIZE;
            
            float x = BitConverter.ToSingle(_data, offset);
            float y = BitConverter.ToSingle(_data, offset + 4);
            float z = BitConverter.ToSingle(_data, offset + 8);
            
            return new Vector3(x, y, z);
        }
        
        /// <summary>
        /// Adds a vertex to the chunk
        /// </summary>
        /// <param name="vertex">Vertex to add</param>
        public void AddVertex(Vector3 vertex)
        {
            byte[] newData = new byte[_data.Length + BASE_VERTEX_SIZE];
            
            // Copy existing data
            Array.Copy(_data, 0, newData, 0, _data.Length);
            
            // Add new vertex data
            int offset = _data.Length;
            Array.Copy(BitConverter.GetBytes(vertex.X), 0, newData, offset, 4);
            Array.Copy(BitConverter.GetBytes(vertex.Y), 0, newData, offset + 4, 4);
            Array.Copy(BitConverter.GetBytes(vertex.Z), 0, newData, offset + 8, 4);
            
            // Update data
            _data = newData;
        }
        
        /// <summary>
        /// Gets a formatted string representation of this chunk
        /// </summary>
        /// <returns>String representation</returns>
        public override string ToString()
        {
            return $"{SIGNATURE} Chunk: {GetVertexCount()} vertices";
        }
        
        /// <summary>
        /// Writes this chunk to a byte array
        /// </summary>
        /// <returns>Byte array containing chunk data</returns>
        public override byte[] Write()
        {
            return _data;
        }
    }
} 