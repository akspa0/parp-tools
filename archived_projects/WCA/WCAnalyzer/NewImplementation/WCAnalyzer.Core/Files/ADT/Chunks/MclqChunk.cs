using System;
using System.IO;
using System.Numerics;
using System.Text;
using Microsoft.Extensions.Logging;
using WCAnalyzer.Core.Files.Common;

namespace WCAnalyzer.Core.Files.ADT.Chunks
{
    /// <summary>
    /// MCLQ chunk - Contains legacy liquid data for a map chunk
    /// Used in older versions (≤ BC) before being replaced by MH2O
    /// Still parsed for backward compatibility
    /// </summary>
    public class MclqChunk : BaseChunk
    {
        /// <summary>
        /// The signature of the MCLQ chunk
        /// </summary>
        public const string SIGNATURE = "MCLQ";
        
        /// <summary>
        /// The dimensions of the liquid grid (9x9)
        /// </summary>
        private const int LIQUID_GRID_SIZE = 9;
        
        /// <summary>
        /// The dimensions of the liquid tiles grid (8x8)
        /// </summary>
        private const int LIQUID_TILES_SIZE = 8;
        
        /// <summary>
        /// Magma scale constant for ADT
        /// </summary>
        private const float MAGMA_SCALE_ADT = 3.0f / 256.0f;
        
        /// <summary>
        /// Defines types of liquid that can appear in the tiles byte flags
        /// </summary>
        [Flags]
        public enum LiquidTileFlags : byte
        {
            /// <summary>
            /// Liquid type mask
            /// </summary>
            TypeMask = 0x0F,
            
            /// <summary>
            /// Don't render this tile
            /// </summary>
            DontRender = 0x0F,
            
            /// <summary>
            /// Ocean type
            /// </summary>
            Ocean = 0x01,
            
            /// <summary>
            /// Slime type
            /// </summary>
            Slime = 0x03,
            
            /// <summary>
            /// River type
            /// </summary>
            River = 0x04,
            
            /// <summary>
            /// Magma type
            /// </summary>
            Magma = 0x06,
            
            /// <summary>
            /// Unknown flag
            /// </summary>
            Unknown1 = 0x10,
            
            /// <summary>
            /// Unknown flag
            /// </summary>
            Unknown2 = 0x20,
            
            /// <summary>
            /// Not low depth (forces swimming)
            /// </summary>
            NotLowDepth = 0x40,
            
            /// <summary>
            /// Fatigue
            /// </summary>
            Fatigue = 0x80
        }
        
        /// <summary>
        /// Represents a water vertex in the MCLQ structure
        /// </summary>
        public struct WaterVertex
        {
            /// <summary>
            /// Gets the depth value
            /// </summary>
            public byte Depth { get; }
            
            /// <summary>
            /// Gets the flow0 percentage
            /// </summary>
            public byte Flow0Pct { get; }
            
            /// <summary>
            /// Gets the flow1 percentage
            /// </summary>
            public byte Flow1Pct { get; }
            
            /// <summary>
            /// Gets the height value
            /// </summary>
            public float Height { get; }
            
            /// <summary>
            /// Initializes a new instance of the <see cref="WaterVertex"/> struct
            /// </summary>
            public WaterVertex(byte depth, byte flow0Pct, byte flow1Pct, float height)
            {
                Depth = depth;
                Flow0Pct = flow0Pct;
                Flow1Pct = flow1Pct;
                Height = height;
            }
        }
        
        /// <summary>
        /// Represents an ocean vertex in the MCLQ structure
        /// </summary>
        public struct OceanVertex
        {
            /// <summary>
            /// Gets the depth value
            /// </summary>
            public byte Depth { get; }
            
            /// <summary>
            /// Gets the foam value
            /// </summary>
            public byte Foam { get; }
            
            /// <summary>
            /// Gets the wet value
            /// </summary>
            public byte Wet { get; }
            
            /// <summary>
            /// Gets the height value
            /// </summary>
            public float Height { get; }
            
            /// <summary>
            /// Initializes a new instance of the <see cref="OceanVertex"/> struct
            /// </summary>
            public OceanVertex(byte depth, byte foam, byte wet, float height)
            {
                Depth = depth;
                Foam = foam;
                Wet = wet;
                Height = height;
            }
        }
        
        /// <summary>
        /// Represents a magma vertex in the MCLQ structure
        /// </summary>
        public struct MagmaVertex
        {
            /// <summary>
            /// Gets the S texture coordinate
            /// </summary>
            public ushort S { get; }
            
            /// <summary>
            /// Gets the T texture coordinate
            /// </summary>
            public ushort T { get; }
            
            /// <summary>
            /// Gets the height value
            /// </summary>
            public float Height { get; }
            
            /// <summary>
            /// Initializes a new instance of the <see cref="MagmaVertex"/> struct
            /// </summary>
            public MagmaVertex(ushort s, ushort t, float height)
            {
                S = s;
                T = t;
                Height = height;
            }
            
            /// <summary>
            /// Gets the scaled S texture coordinate for rendering
            /// </summary>
            public float ScaledS => S * MAGMA_SCALE_ADT;
            
            /// <summary>
            /// Gets the scaled T texture coordinate for rendering
            /// </summary>
            public float ScaledT => T * MAGMA_SCALE_ADT;
        }
        
        /// <summary>
        /// Represents a flow vector in the MCLQ structure
        /// </summary>
        public struct FlowVector
        {
            /// <summary>
            /// Gets the sphere center
            /// </summary>
            public Vector3 SphereCenter { get; }
            
            /// <summary>
            /// Gets the sphere radius
            /// </summary>
            public float SphereRadius { get; }
            
            /// <summary>
            /// Gets the flow direction
            /// </summary>
            public Vector3 Direction { get; }
            
            /// <summary>
            /// Gets the flow velocity
            /// </summary>
            public float Velocity { get; }
            
            /// <summary>
            /// Gets the flow amplitude
            /// </summary>
            public float Amplitude { get; }
            
            /// <summary>
            /// Gets the flow frequency
            /// </summary>
            public float Frequency { get; }
            
            /// <summary>
            /// Initializes a new instance of the <see cref="FlowVector"/> struct
            /// </summary>
            public FlowVector(Vector3 sphereCenter, float sphereRadius, Vector3 direction, float velocity, float amplitude, float frequency)
            {
                SphereCenter = sphereCenter;
                SphereRadius = sphereRadius;
                Direction = direction;
                Velocity = velocity;
                Amplitude = amplitude;
                Frequency = frequency;
            }
        }
        
        /// <summary>
        /// Gets the liquid type based on MCNK flags
        /// </summary>
        public LiquidTileFlags LiquidType { get; private set; }
        
        /// <summary>
        /// Gets the water vertices (if liquid type is water)
        /// </summary>
        public WaterVertex[,] WaterVertices { get; private set; }
        
        /// <summary>
        /// Gets the ocean vertices (if liquid type is ocean)
        /// </summary>
        public OceanVertex[,] OceanVertices { get; private set; }
        
        /// <summary>
        /// Gets the magma vertices (if liquid type is magma/lava or slime)
        /// </summary>
        public MagmaVertex[,] MagmaVertices { get; private set; }
        
        /// <summary>
        /// Gets the liquid tiles information
        /// </summary>
        public byte[,] Tiles { get; private set; }
        
        /// <summary>
        /// Gets the number of flow vectors
        /// </summary>
        public uint FlowVectorsCount { get; private set; }
        
        /// <summary>
        /// Gets the flow vectors
        /// </summary>
        public FlowVector[] FlowVectors { get; private set; }
        
        /// <summary>
        /// Initializes a new instance of the <see cref="MclqChunk"/> class
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="mcnkFlags">The flags from the parent MCNK chunk, used to determine liquid type</param>
        /// <param name="logger">Optional logger</param>
        public MclqChunk(byte[] data, uint mcnkFlags, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            // Determine liquid type from MCNK flags
            // Bits 2-5 of MCNK.Flags store the liquid type
            LiquidType = (LiquidTileFlags)(((mcnkFlags >> 2) & 0x0F));
            
            Tiles = new byte[LIQUID_TILES_SIZE, LIQUID_TILES_SIZE];
            FlowVectors = new FlowVector[2]; // Always 2 in file, independent of FlowVectorsCount
            
            Parse();
        }

        /// <summary>
        /// Alternate constructor without MCNK flags (assumes default water type)
        /// </summary>
        /// <param name="data">The chunk data</param>
        /// <param name="logger">Optional logger</param>
        public MclqChunk(byte[] data, ILogger logger = null) : base(SIGNATURE, data, logger)
        {
            // Default to water type if no MCNK flags are provided
            LiquidType = LiquidTileFlags.River;
            
            Tiles = new byte[LIQUID_TILES_SIZE, LIQUID_TILES_SIZE];
            FlowVectors = new FlowVector[2]; // Always 2 in file, independent of FlowVectorsCount
            
            Parse();
        }

        /// <summary>
        /// Parses the MCLQ chunk data
        /// </summary>
        protected override void Parse()
        {
            try
            {
                if (Data == null || Data.Length == 0)
                {
                    Logger?.LogWarning("MCLQ chunk has no data");
                    return;
                }
                
                using (var ms = new MemoryStream(Data))
                using (var reader = new BinaryReader(ms))
                {
                    // Range of heights (unused in our implementation)
                    float minHeight = reader.ReadSingle();
                    float maxHeight = reader.ReadSingle();
                    
                    // Initialize the appropriate vertex arrays based on liquid type
                    if (LiquidType == LiquidTileFlags.Ocean)
                    {
                        OceanVertices = new OceanVertex[LIQUID_GRID_SIZE, LIQUID_GRID_SIZE];
                    }
                    else if (LiquidType == LiquidTileFlags.Magma || LiquidType == LiquidTileFlags.Slime)
                    {
                        MagmaVertices = new MagmaVertex[LIQUID_GRID_SIZE, LIQUID_GRID_SIZE];
                    }
                    else // Default to water
                    {
                        WaterVertices = new WaterVertex[LIQUID_GRID_SIZE, LIQUID_GRID_SIZE];
                    }
                    
                    // Read vertices (9x9 grid)
                    for (int y = 0; y < LIQUID_GRID_SIZE; y++)
                    {
                        for (int x = 0; x < LIQUID_GRID_SIZE; x++)
                        {
                            // Read differently based on liquid type
                            if (LiquidType == LiquidTileFlags.Ocean)
                            {
                                // Ocean vertex
                                byte depth = reader.ReadByte();
                                byte foam = reader.ReadByte();
                                byte wet = reader.ReadByte();
                                byte unused = reader.ReadByte(); // Padding
                                float height = reader.ReadSingle();
                                
                                OceanVertices[x, y] = new OceanVertex(depth, foam, wet, height);
                            }
                            else if (LiquidType == LiquidTileFlags.Magma || LiquidType == LiquidTileFlags.Slime)
                            {
                                // Magma/Slime vertex
                                ushort s = reader.ReadUInt16();
                                ushort t = reader.ReadUInt16();
                                float height = reader.ReadSingle();
                                
                                MagmaVertices[x, y] = new MagmaVertex(s, t, height);
                            }
                            else
                            {
                                // Water vertex
                                byte depth = reader.ReadByte();
                                byte flow0Pct = reader.ReadByte();
                                byte flow1Pct = reader.ReadByte();
                                byte unused = reader.ReadByte(); // Padding
                                float height = reader.ReadSingle();
                                
                                WaterVertices[x, y] = new WaterVertex(depth, flow0Pct, flow1Pct, height);
                            }
                        }
                    }
                    
                    // Read liquid tiles (8x8 grid)
                    for (int y = 0; y < LIQUID_TILES_SIZE; y++)
                    {
                        for (int x = 0; x < LIQUID_TILES_SIZE; x++)
                        {
                            Tiles[x, y] = reader.ReadByte();
                        }
                    }
                    
                    // Read flow vectors
                    FlowVectorsCount = reader.ReadUInt32();
                    
                    // Always read 2 flow vectors regardless of FlowVectorsCount
                    for (int i = 0; i < 2; i++)
                    {
                        // Read sphere
                        Vector3 sphereCenter = new Vector3(
                            reader.ReadSingle(),  // X
                            reader.ReadSingle(),  // Y
                            reader.ReadSingle()); // Z
                        float sphereRadius = reader.ReadSingle();
                        
                        // Read direction
                        Vector3 direction = new Vector3(
                            reader.ReadSingle(),  // X
                            reader.ReadSingle(),  // Y
                            reader.ReadSingle()); // Z
                        
                        // Read velocity, amplitude, frequency
                        float velocity = reader.ReadSingle();
                        float amplitude = reader.ReadSingle();
                        float frequency = reader.ReadSingle();
                        
                        FlowVectors[i] = new FlowVector(sphereCenter, sphereRadius, direction, velocity, amplitude, frequency);
                    }
                    
                    Logger?.LogDebug($"MCLQ: Parsed legacy liquid data with type {LiquidType}, flow vectors: {FlowVectorsCount}");
                }
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error parsing MCLQ chunk: {ex.Message}");
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
                Logger?.LogError("Cannot write MCLQ chunk: BinaryWriter is null");
                throw new ArgumentNullException(nameof(writer));
            }

            try
            {
                // Write chunk header
                writer.Write(SIGNATURE.ToCharArray());
                
                // Calculate data size
                int dataSize = 8; // Min/max height (2 floats)
                dataSize += LIQUID_GRID_SIZE * LIQUID_GRID_SIZE * 8; // Vertex data (8 bytes per vertex)
                dataSize += LIQUID_TILES_SIZE * LIQUID_TILES_SIZE; // Tile data (1 byte per tile)
                dataSize += 4; // Flow vectors count (1 uint)
                dataSize += 2 * 40; // Flow vectors (40 bytes per vector, always 2)
                
                // Write chunk size
                writer.Write(dataSize);
                
                // Calculate min/max heights
                float minHeight = float.MaxValue;
                float maxHeight = float.MinValue;
                
                if (LiquidType == LiquidTileFlags.Ocean && OceanVertices != null)
                {
                    for (int y = 0; y < LIQUID_GRID_SIZE; y++)
                    {
                        for (int x = 0; x < LIQUID_GRID_SIZE; x++)
                        {
                            minHeight = Math.Min(minHeight, OceanVertices[x, y].Height);
                            maxHeight = Math.Max(maxHeight, OceanVertices[x, y].Height);
                        }
                    }
                }
                else if ((LiquidType == LiquidTileFlags.Magma || LiquidType == LiquidTileFlags.Slime) && MagmaVertices != null)
                {
                    for (int y = 0; y < LIQUID_GRID_SIZE; y++)
                    {
                        for (int x = 0; x < LIQUID_GRID_SIZE; x++)
                        {
                            minHeight = Math.Min(minHeight, MagmaVertices[x, y].Height);
                            maxHeight = Math.Max(maxHeight, MagmaVertices[x, y].Height);
                        }
                    }
                }
                else if (WaterVertices != null)
                {
                    for (int y = 0; y < LIQUID_GRID_SIZE; y++)
                    {
                        for (int x = 0; x < LIQUID_GRID_SIZE; x++)
                        {
                            minHeight = Math.Min(minHeight, WaterVertices[x, y].Height);
                            maxHeight = Math.Max(maxHeight, WaterVertices[x, y].Height);
                        }
                    }
                }
                
                // Write min/max heights
                writer.Write(minHeight);
                writer.Write(maxHeight);
                
                // Write vertices
                for (int y = 0; y < LIQUID_GRID_SIZE; y++)
                {
                    for (int x = 0; x < LIQUID_GRID_SIZE; x++)
                    {
                        if (LiquidType == LiquidTileFlags.Ocean && OceanVertices != null)
                        {
                            // Ocean vertex
                            writer.Write(OceanVertices[x, y].Depth);
                            writer.Write(OceanVertices[x, y].Foam);
                            writer.Write(OceanVertices[x, y].Wet);
                            writer.Write((byte)0); // Padding
                            writer.Write(OceanVertices[x, y].Height);
                        }
                        else if ((LiquidType == LiquidTileFlags.Magma || LiquidType == LiquidTileFlags.Slime) && MagmaVertices != null)
                        {
                            // Magma/Slime vertex
                            writer.Write(MagmaVertices[x, y].S);
                            writer.Write(MagmaVertices[x, y].T);
                            writer.Write(MagmaVertices[x, y].Height);
                        }
                        else if (WaterVertices != null)
                        {
                            // Water vertex
                            writer.Write(WaterVertices[x, y].Depth);
                            writer.Write(WaterVertices[x, y].Flow0Pct);
                            writer.Write(WaterVertices[x, y].Flow1Pct);
                            writer.Write((byte)0); // Padding
                            writer.Write(WaterVertices[x, y].Height);
                        }
                        else
                        {
                            // Default water vertex with zeros
                            writer.Write((byte)0); // Depth
                            writer.Write((byte)0); // Flow0Pct
                            writer.Write((byte)0); // Flow1Pct
                            writer.Write((byte)0); // Padding
                            writer.Write(0.0f);    // Height
                        }
                    }
                }
                
                // Write tiles
                for (int y = 0; y < LIQUID_TILES_SIZE; y++)
                {
                    for (int x = 0; x < LIQUID_TILES_SIZE; x++)
                    {
                        writer.Write(Tiles[x, y]);
                    }
                }
                
                // Write flow vectors count
                writer.Write(FlowVectorsCount);
                
                // Write flow vectors (always 2)
                for (int i = 0; i < 2; i++)
                {
                    if (i < FlowVectors.Length)
                    {
                        // Write sphere
                        writer.Write(FlowVectors[i].SphereCenter.X);
                        writer.Write(FlowVectors[i].SphereCenter.Y);
                        writer.Write(FlowVectors[i].SphereCenter.Z);
                        writer.Write(FlowVectors[i].SphereRadius);
                        
                        // Write direction
                        writer.Write(FlowVectors[i].Direction.X);
                        writer.Write(FlowVectors[i].Direction.Y);
                        writer.Write(FlowVectors[i].Direction.Z);
                        
                        // Write velocity, amplitude, frequency
                        writer.Write(FlowVectors[i].Velocity);
                        writer.Write(FlowVectors[i].Amplitude);
                        writer.Write(FlowVectors[i].Frequency);
                    }
                    else
                    {
                        // Write default flow vector with zeros
                        writer.Write(0.0f); // Sphere X
                        writer.Write(0.0f); // Sphere Y
                        writer.Write(0.0f); // Sphere Z
                        writer.Write(0.0f); // Sphere Radius
                        
                        writer.Write(0.0f); // Direction X
                        writer.Write(0.0f); // Direction Y
                        writer.Write(0.0f); // Direction Z
                        
                        writer.Write(0.0f); // Velocity
                        writer.Write(0.0f); // Amplitude
                        writer.Write(0.0f); // Frequency
                    }
                }
                
                Logger?.LogDebug($"MCLQ: Wrote legacy liquid data with type {LiquidType}");
            }
            catch (Exception ex)
            {
                Logger?.LogError($"Error writing MCLQ chunk: {ex.Message}");
                throw;
            }
        }
        
        /// <summary>
        /// Gets the liquid height at the specified grid coordinates
        /// </summary>
        /// <param name="x">The X coordinate (0-8)</param>
        /// <param name="y">The Y coordinate (0-8)</param>
        /// <returns>The liquid height at the specified coordinates</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the coordinates are out of range</exception>
        public float GetHeight(int x, int y)
        {
            if (x < 0 || x >= LIQUID_GRID_SIZE)
            {
                throw new ArgumentOutOfRangeException(nameof(x), $"X coordinate must be between 0 and {LIQUID_GRID_SIZE - 1}");
            }

            if (y < 0 || y >= LIQUID_GRID_SIZE)
            {
                throw new ArgumentOutOfRangeException(nameof(y), $"Y coordinate must be between 0 and {LIQUID_GRID_SIZE - 1}");
            }

            if (LiquidType == LiquidTileFlags.Ocean && OceanVertices != null)
            {
                return OceanVertices[x, y].Height;
            }
            else if ((LiquidType == LiquidTileFlags.Magma || LiquidType == LiquidTileFlags.Slime) && MagmaVertices != null)
            {
                return MagmaVertices[x, y].Height;
            }
            else if (WaterVertices != null)
            {
                return WaterVertices[x, y].Height;
            }

            return 0.0f;
        }
        
        /// <summary>
        /// Gets the liquid tile flags at the specified grid coordinates
        /// </summary>
        /// <param name="x">The X coordinate (0-7)</param>
        /// <param name="y">The Y coordinate (0-7)</param>
        /// <returns>The liquid tile flags at the specified coordinates</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if the coordinates are out of range</exception>
        public LiquidTileFlags GetTileFlags(int x, int y)
        {
            if (x < 0 || x >= LIQUID_TILES_SIZE)
            {
                throw new ArgumentOutOfRangeException(nameof(x), $"X coordinate must be between 0 and {LIQUID_TILES_SIZE - 1}");
            }

            if (y < 0 || y >= LIQUID_TILES_SIZE)
            {
                throw new ArgumentOutOfRangeException(nameof(y), $"Y coordinate must be between 0 and {LIQUID_TILES_SIZE - 1}");
            }

            return (LiquidTileFlags)Tiles[x, y];
        }
        
        /// <summary>
        /// Determines if a tile should be rendered
        /// </summary>
        /// <param name="x">The X coordinate (0-7)</param>
        /// <param name="y">The Y coordinate (0-7)</param>
        /// <returns>True if the tile should be rendered, false otherwise</returns>
        public bool ShouldRenderTile(int x, int y)
        {
            LiquidTileFlags flags = GetTileFlags(x, y);
            return ((flags & LiquidTileFlags.TypeMask) != LiquidTileFlags.DontRender);
        }
        
        /// <summary>
        /// Gets the active flow vectors
        /// </summary>
        /// <returns>An array of flow vectors that are active</returns>
        public FlowVector[] GetActiveFlowVectors()
        {
            if (FlowVectorsCount == 0 || FlowVectors == null)
            {
                return Array.Empty<FlowVector>();
            }

            int count = (int)Math.Min(FlowVectorsCount, (uint)FlowVectors.Length);
            FlowVector[] result = new FlowVector[count];
            Array.Copy(FlowVectors, result, count);
            return result;
        }
    }
} 