using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MBNV chunk - Contains normal vectors for terrain blending
    /// </summary>
    public class MbnvChunk : ADTChunk
    {
        /// <summary>
        /// The MBNV chunk signature
        /// </summary>
        public const string SIGNATURE = "MBNV";

        /// <summary>
        /// Gets the list of normal vectors
        /// </summary>
        public List<Vector3> NormalVectors { get; private set; } = new List<Vector3>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MbnvChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MbnvChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            Parse();
        }

        /// <summary>
        /// Parses the MBNV chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MBNV chunk has no data");
                    return;
                }
                
                using var ms = new MemoryStream(Data);
                using var reader = new BinaryReader(ms);
                
                // Each normal vector is 12 bytes (3 floats)
                const int VectorSize = 12;
                
                // Validate that the data size is a multiple of vector size
                if (Data.Length % VectorSize != 0)
                {
                    Logger?.LogWarning($"MBNV chunk data size {Data.Length} is not a multiple of {VectorSize}");
                }
                
                // Calculate how many vectors we should have
                int count = Data.Length / VectorSize;
                
                // Read normal vectors
                for (int i = 0; i < count; i++)
                {
                    try
                    {
                        var vector = new Vector3(
                            reader.ReadSingle(), // X
                            reader.ReadSingle(), // Y
                            reader.ReadSingle()  // Z
                        );
                        
                        NormalVectors.Add(vector);
                    }
                    catch (EndOfStreamException)
                    {
                        Logger?.LogWarning($"MBNV: Unexpected end of stream while reading normal vector {i}");
                        break;
                    }
                }
                
                Logger?.LogDebug($"MBNV: Read {NormalVectors.Count} normal vectors");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MBNV chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Writes the chunk data to a binary writer
        /// </summary>
        /// <param name="writer">The binary writer to write to</param>
        public override void Write(BinaryWriter writer)
        {
            if (writer == null)
            {
                Logger?.LogError("Cannot write MBNV chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header and data
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate the data size (12 bytes per vector)
                int dataSize = NormalVectors.Count * 12;
                writer.Write(dataSize);
                
                // Write each normal vector
                foreach (var vector in NormalVectors)
                {
                    writer.Write(vector.X);
                    writer.Write(vector.Y);
                    writer.Write(vector.Z);
                }
                
                Logger?.LogDebug($"MBNV: Wrote {NormalVectors.Count} normal vectors");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MBNV chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets a normal vector at the specified index
        /// </summary>
        /// <param name="index">The index of the vector</param>
        /// <returns>The normal vector, or Vector3.Zero if the index is invalid</returns>
        public Vector3 GetNormalVector(int index)
        {
            if (index < 0 || index >= NormalVectors.Count)
            {
                return Vector3.Zero;
            }
            
            return NormalVectors[index];
        }
    }
} 