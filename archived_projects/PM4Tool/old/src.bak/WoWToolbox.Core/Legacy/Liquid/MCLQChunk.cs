using System;
using System.IO;
using Warcraft.NET;
using Warcraft.NET.Files.Interfaces;
using DBCD;

namespace WoWToolbox.Core.Legacy.Liquid
{
    /// <summary>
    /// Legacy liquid data chunk (MCLQ) used in pre-Cataclysm versions of World of Warcraft.
    /// This chunk was used to define liquid surfaces like water, lava, and slime within MCNK chunks.
    /// </summary>
    public class MCLQChunk : IIFFChunk, IBinarySerializable
    {
        public const string Signature = "MCLQ";
        public const int HEIGHT_MAP_SIZE = 9;
        public const int ALPHA_MAP_SIZE = 8;

        private static IDBCDStorage? liquidTypeStorage;

        #region Properties

        /// <summary>
        /// First water level height
        /// </summary>
        public float HeightLevel1 { get; set; }

        /// <summary>
        /// Second water level height
        /// </summary>
        public float HeightLevel2 { get; set; }

        /// <summary>
        /// Liquid flags
        /// </summary>
        public MCLQFlags Flags { get; set; }

        /// <summary>
        /// Unknown data value
        /// </summary>
        public byte RawData { get; set; }

        /// <summary>
        /// X coordinate of liquid area
        /// </summary>
        public float X { get; set; }

        /// <summary>
        /// Y coordinate of liquid area
        /// </summary>
        public float Y { get; set; }

        /// <summary>
        /// X offset in the liquid texture
        /// </summary>
        public byte XOffset { get; set; }

        /// <summary>
        /// Y offset in the liquid texture
        /// </summary>
        public byte YOffset { get; set; }

        /// <summary>
        /// Width of the liquid area
        /// </summary>
        public byte Width { get; set; }

        /// <summary>
        /// Height of the liquid area
        /// </summary>
        public byte Height { get; set; }

        /// <summary>
        /// Liquid type ID from LiquidType.dbc
        /// </summary>
        public ushort LiquidEntry { get; set; }

        /// <summary>
        /// Format of vertex data
        /// </summary>
        public byte LiquidVertexFormat { get; set; }

        /// <summary>
        /// Additional liquid flags
        /// </summary>
        public byte LiquidFlags { get; set; }

        /// <summary>
        /// Liquid type
        /// </summary>
        public LiquidType LiquidType { get; set; }

        /// <summary>
        /// Height map for liquid surface (9x9 grid)
        /// </summary>
        public float[,] HeightMap { get; set; } = new float[HEIGHT_MAP_SIZE, HEIGHT_MAP_SIZE];

        /// <summary>
        /// Alpha map for transparency (8x8 grid), only present if MCLQ_HAS_ALPHA flag is set
        /// </summary>
        public byte[,] AlphaMap { get; set; } = new byte[ALPHA_MAP_SIZE, ALPHA_MAP_SIZE];

        #endregion

        #region Computed Properties

        /// <summary>
        /// Gets whether this chunk has an alpha map
        /// </summary>
        public bool HasAlphaMap => (Flags & MCLQFlags.HasAlpha) != 0;

        /// <summary>
        /// Gets whether the liquid entry is valid according to LiquidType.dbc
        /// Returns true if validation is disabled
        /// </summary>
        public bool IsValidLiquidEntry => liquidTypeStorage?.ContainsKey(LiquidEntry) ?? true;

        #endregion

        #region IIFFChunk Implementation

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] inData)
        {
            using var ms = new MemoryStream(inData);
            using var br = new BinaryReader(ms);
            Deserialize(br);
        }

        /// <inheritdoc/>
        public string GetSignature()
        {
            return Signature;
        }

        /// <inheritdoc/>
        public byte[] Data { get; set; } = Array.Empty<byte>();

        #endregion

        #region IBinarySerializable Implementation

        /// <inheritdoc/>
        public uint GetSize()
        {
            return (uint)(24 + (HEIGHT_MAP_SIZE * HEIGHT_MAP_SIZE * 4) + (HasAlphaMap ? ALPHA_MAP_SIZE * ALPHA_MAP_SIZE : 0));
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            Serialize(bw);
            return ms.ToArray();
        }

        #endregion

        #region Static Methods

        /// <summary>
        /// Configures the LiquidType.dbc validator
        /// </summary>
        /// <param name="storage">The IDBCDStorage to use</param>
        public static void ConfigureValidator(IDBCDStorage storage)
        {
            liquidTypeStorage = storage;
        }

        /// <summary>
        /// Disables LiquidType.dbc validation
        /// </summary>
        public static void DisableValidation()
        {
            liquidTypeStorage = null;
        }

        #endregion

        #region Constructors

        /// <summary>
        /// Creates a new instance of the MCLQChunk class
        /// </summary>
        public MCLQChunk()
        {
        }

        /// <summary>
        /// Creates a new instance of the MCLQChunk class with the specified data
        /// </summary>
        public MCLQChunk(byte[] data)
        {
            Data = data;
            using var ms = new MemoryStream(data);
            using var br = new BinaryReader(ms);
            Deserialize(br);
        }

        #endregion

        #region IBinarySerializable Implementation

        public void Deserialize(BinaryReader reader)
        {
            HeightLevel1 = reader.ReadSingle();
            HeightLevel2 = reader.ReadSingle();
            Flags = (MCLQFlags)reader.ReadByte();
            RawData = reader.ReadByte();
            X = reader.ReadSingle();
            Y = reader.ReadSingle();
            XOffset = reader.ReadByte();
            YOffset = reader.ReadByte();
            Width = reader.ReadByte();
            Height = reader.ReadByte();
            LiquidEntry = reader.ReadUInt16();
            LiquidVertexFormat = reader.ReadByte();
            LiquidFlags = reader.ReadByte();
            LiquidType = (LiquidType)reader.ReadUInt16();

            // Read the height map - 9x9 grid
            for (int y = 0; y < HEIGHT_MAP_SIZE; y++)
            {
                for (int x = 0; x < HEIGHT_MAP_SIZE; x++)
                {
                    HeightMap[x, y] = reader.ReadSingle();
                }
            }

            // Read the alpha map if present - 8x8 grid
            if (HasAlphaMap)
            {
                for (int y = 0; y < ALPHA_MAP_SIZE; y++)
                {
                    for (int x = 0; x < ALPHA_MAP_SIZE; x++)
                    {
                        AlphaMap[x, y] = reader.ReadByte();
                    }
                }
            }
        }

        public void Serialize(BinaryWriter writer)
        {
            writer.Write(HeightLevel1);
            writer.Write(HeightLevel2);
            writer.Write((byte)Flags);
            writer.Write(RawData);
            writer.Write(X);
            writer.Write(Y);
            writer.Write(XOffset);
            writer.Write(YOffset);
            writer.Write(Width);
            writer.Write(Height);
            writer.Write(LiquidEntry);
            writer.Write(LiquidVertexFormat);
            writer.Write(LiquidFlags);
            writer.Write((ushort)LiquidType);

            // Write the height map
            for (int y = 0; y < HEIGHT_MAP_SIZE; y++)
            {
                for (int x = 0; x < HEIGHT_MAP_SIZE; x++)
                {
                    writer.Write(HeightMap[x, y]);
                }
            }

            // Write the alpha map if present
            if (HasAlphaMap)
            {
                for (int y = 0; y < ALPHA_MAP_SIZE; y++)
                {
                    for (int x = 0; x < ALPHA_MAP_SIZE; x++)
                    {
                        writer.Write(AlphaMap[x, y]);
                    }
                }
            }
        }

        #endregion
    }
} 