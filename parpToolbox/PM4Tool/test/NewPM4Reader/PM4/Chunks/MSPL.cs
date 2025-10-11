using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using NewPM4Reader.Interfaces;

namespace NewPM4Reader.PM4.Chunks
{
    /// <summary>
    /// Represents the MSPL (Map Splitter) chunk in a PM4 file.
    /// </summary>
    public class MSPL : IPM4Chunk
    {
        /// <summary>
        /// Gets the signature of the chunk.
        /// </summary>
        public string Signature => "MSPL";

        /// <summary>
        /// Gets or sets the version.
        /// </summary>
        public uint Version { get; set; }

        /// <summary>
        /// Gets or sets the number of splits.
        /// </summary>
        public uint NumSplits { get; set; }

        /// <summary>
        /// Gets or sets the list of map splits.
        /// </summary>
        public List<MapSplit> Splits { get; private set; } = new List<MapSplit>();

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPL"/> class.
        /// </summary>
        public MSPL()
        {
            Version = 1;
            NumSplits = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MSPL"/> class from binary data.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public MSPL(BinaryReader reader)
        {
            ReadBinary(reader);
        }

        /// <summary>
        /// Reads the chunk data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            Version = reader.ReadUInt32();
            NumSplits = reader.ReadUInt32();
            
            Splits = new List<MapSplit>((int)NumSplits);
            
            for (int i = 0; i < NumSplits; i++)
            {
                var split = new MapSplit();
                split.ReadBinary(reader);
                Splits.Add(split);
            }
        }

        /// <summary>
        /// Writes the chunk data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Version);
            writer.Write(NumSplits);
            
            foreach (var split in Splits)
            {
                split.WriteBinary(writer);
            }
        }
    }

    /// <summary>
    /// Represents a single map split.
    /// </summary>
    public class MapSplit
    {
        /// <summary>
        /// Gets or sets the center point of the split.
        /// </summary>
        public Vector3 Center { get; set; }

        /// <summary>
        /// Gets or sets the normal vector.
        /// </summary>
        public Vector3 Normal { get; set; }

        /// <summary>
        /// Gets or sets the height of the split.
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MapSplit"/> class.
        /// </summary>
        public MapSplit()
        {
        }

        /// <summary>
        /// Reads the map split data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader.</param>
        public void ReadBinary(BinaryReader reader)
        {
            Center = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle());

            Normal = new Vector3(
                reader.ReadSingle(),
                reader.ReadSingle(),
                reader.ReadSingle());

            Height = reader.ReadSingle();
        }

        /// <summary>
        /// Writes the map split data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer.</param>
        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Center.X);
            writer.Write(Center.Y);
            writer.Write(Center.Z);

            writer.Write(Normal.X);
            writer.Write(Normal.Y);
            writer.Write(Normal.Z);

            writer.Write(Height);
        }
    }
} 