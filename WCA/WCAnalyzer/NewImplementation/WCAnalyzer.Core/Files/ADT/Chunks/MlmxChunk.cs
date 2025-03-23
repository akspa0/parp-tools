using System;
using System.IO;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Common.Interfaces;

namespace NewImplementation.WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MLMX chunk - Legion Matrix Data
    /// This chunk appears to contain transformation matrices for Legion+ terrain features
    /// (MLMX could stand for Map Legion MatriX)
    /// </summary>
    public class MlmxChunk : ADTChunk
    {
        /// <summary>
        /// Signature for this chunk type
        /// </summary>
        public const string SIGNATURE = "MLMX";

        /// <summary>
        /// Gets the raw chunk data
        /// </summary>
        public byte[] RawData { get; private set; }
        
        /// <summary>
        /// Gets the size of the data
        /// </summary>
        public int DataSize => RawData.Length;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MlmxChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MlmxChunk(byte[] data, ILogger? logger = null) : base(SIGNATURE, data, logger)
        {
            RawData = Array.Empty<byte>();
            Parse();
        }

        /// <summary>
        /// Parses the chunk data
        /// </summary>
        protected virtual void Parse()
        {
            try
            {
                if (Data.Length == 0)
                {
                    Logger?.LogWarning($"{SIGNATURE} chunk has no data");
                    return;
                }
                
                // Store the raw data for later analysis
                RawData = new byte[Data.Length];
                Array.Copy(Data, RawData, Data.Length);
                
                Logger?.LogDebug($"{SIGNATURE}: Read {Data.Length} bytes of data");
                LogDataSummary();
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing {SIGNATURE} chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Logs a summary of the data for debugging purposes
        /// </summary>
        protected virtual void LogDataSummary()
        {
            if (RawData.Length == 0)
            {
                Logger?.LogWarning($"{SIGNATURE} chunk data is empty");
                return;
            }

            Logger?.LogDebug($"{SIGNATURE} chunk data size: {RawData.Length} bytes");

            // Check if the data can be interpreted as 4x4 transformation matrices
            // A 4x4 float matrix is 64 bytes (16 floats)
            if (RawData.Length % 64 == 0)
            {
                int matrixCount = RawData.Length / 64;
                Logger?.LogDebug($"Data could represent {matrixCount} 4x4 matrices");
                
                // Sample the first matrix if available
                if (matrixCount > 0)
                {
                    using (var ms = new MemoryStream(RawData))
                    using (var br = new BinaryReader(ms))
                    {
                        float[,] matrix = new float[4, 4];
                        StringBuilder matrixString = new StringBuilder();
                        matrixString.AppendLine("First 4x4 matrix:");
                        
                        for (int row = 0; row < 4; row++)
                        {
                            matrixString.Append("  ");
                            for (int col = 0; col < 4; col++)
                            {
                                matrix[row, col] = br.ReadSingle();
                                matrixString.Append($"{matrix[row, col]:F3} ");
                            }
                            matrixString.AppendLine();
                        }
                        
                        Logger?.LogDebug(matrixString.ToString());
                    }
                }
            }
            
            // Check if the data can be interpreted as 3x4 matrices (sometimes used for transforms)
            // A 3x4 float matrix is 48 bytes (12 floats)
            if (RawData.Length % 48 == 0)
            {
                int matrixCount = RawData.Length / 48;
                Logger?.LogDebug($"Data could represent {matrixCount} 3x4 matrices");
                
                // Sample the first matrix if available
                if (matrixCount > 0)
                {
                    using (var ms = new MemoryStream(RawData))
                    using (var br = new BinaryReader(ms))
                    {
                        float[,] matrix = new float[3, 4];
                        StringBuilder matrixString = new StringBuilder();
                        matrixString.AppendLine("First 3x4 matrix:");
                        
                        for (int row = 0; row < 3; row++)
                        {
                            matrixString.Append("  ");
                            for (int col = 0; col < 4; col++)
                            {
                                matrix[row, col] = br.ReadSingle();
                                matrixString.Append($"{matrix[row, col]:F3} ");
                            }
                            matrixString.AppendLine();
                        }
                        
                        Logger?.LogDebug(matrixString.ToString());
                    }
                }
            }
            
            // Check if the data can be interpreted as float arrays
            if (RawData.Length % 4 == 0)
            {
                int floatCount = RawData.Length / 4;
                Logger?.LogDebug($"Data could represent {floatCount} float values");
                
                // Sample some floats
                using (var ms = new MemoryStream(RawData))
                using (var br = new BinaryReader(ms))
                {
                    int samplesToRead = Math.Min(16, floatCount);
                    StringBuilder floatSamples = new StringBuilder();
                    floatSamples.Append("Float samples: ");
                    
                    for (int i = 0; i < samplesToRead; i++)
                    {
                        if (i > 0 && i % 4 == 0) floatSamples.AppendLine().Append("              ");
                        else if (i > 0) floatSamples.Append(" ");
                        
                        floatSamples.Append($"{br.ReadSingle():F4}");
                    }
                    
                    Logger?.LogDebug(floatSamples.ToString());
                }
            }
            
            // Log the first few bytes for debugging
            StringBuilder hexDump = new StringBuilder();
            for (int i = 0; i < Math.Min(RawData.Length, 64); i++)
            {
                hexDump.Append(RawData[i].ToString("X2"));
                if ((i + 1) % 16 == 0)
                    hexDump.Append("\n");
                else if ((i + 1) % 4 == 0)
                    hexDump.Append(" ");
            }
            
            Logger?.LogDebug($"First bytes as hex:\n{hexDump}");
        }
        
        /// <summary>
        /// Gets the data as an array of 4x4 matrices
        /// </summary>
        /// <returns>Array of Matrix4x4 values if the data size is divisible by 64, otherwise null</returns>
        public Matrix4x4[]? GetAsMatrix4x4Array()
        {
            if (RawData.Length == 0 || RawData.Length % 64 != 0)
                return null;
            
            int count = RawData.Length / 64;
            Matrix4x4[] matrices = new Matrix4x4[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    // Matrix4x4 is stored in row-major order
                    float m11 = br.ReadSingle();
                    float m12 = br.ReadSingle();
                    float m13 = br.ReadSingle();
                    float m14 = br.ReadSingle();
                    
                    float m21 = br.ReadSingle();
                    float m22 = br.ReadSingle();
                    float m23 = br.ReadSingle();
                    float m24 = br.ReadSingle();
                    
                    float m31 = br.ReadSingle();
                    float m32 = br.ReadSingle();
                    float m33 = br.ReadSingle();
                    float m34 = br.ReadSingle();
                    
                    float m41 = br.ReadSingle();
                    float m42 = br.ReadSingle();
                    float m43 = br.ReadSingle();
                    float m44 = br.ReadSingle();
                    
                    matrices[i] = new Matrix4x4(
                        m11, m12, m13, m14,
                        m21, m22, m23, m24,
                        m31, m32, m33, m34,
                        m41, m42, m43, m44
                    );
                }
            }
            
            return matrices;
        }
        
        /// <summary>
        /// Gets the data as an array of 32-bit floating point values
        /// </summary>
        /// <returns>Array of float values if the data size is divisible by 4, otherwise null</returns>
        public float[]? GetAsFloatArray()
        {
            if (RawData.Length == 0 || RawData.Length % 4 != 0)
                return null;
            
            int count = RawData.Length / 4;
            float[] values = new float[count];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    values[i] = br.ReadSingle();
                }
            }
            
            return values;
        }
        
        /// <summary>
        /// Gets the data as an array of 3x4 matrices (common for 3D transforms)
        /// </summary>
        /// <returns>Array of float[3,4] matrices if the data size is divisible by 48, otherwise null</returns>
        public float[,][]? GetAs3x4MatrixArray()
        {
            if (RawData.Length == 0 || RawData.Length % 48 != 0)
                return null;
            
            int count = RawData.Length / 48;
            float[,][] matrices = new float[count][,];
            
            using (var ms = new MemoryStream(RawData))
            using (var br = new BinaryReader(ms))
            {
                for (int i = 0; i < count; i++)
                {
                    float[,] matrix = new float[3, 4];
                    
                    for (int row = 0; row < 3; row++)
                    {
                        for (int col = 0; col < 4; col++)
                        {
                            matrix[row, col] = br.ReadSingle();
                        }
                    }
                    
                    matrices[i] = matrix;
                }
            }
            
            return matrices;
        }
        
        /// <summary>
        /// Gets the raw data for manual inspection
        /// </summary>
        /// <returns>The raw data</returns>
        public byte[] GetRawData()
        {
            return RawData;
        }
    }
} 