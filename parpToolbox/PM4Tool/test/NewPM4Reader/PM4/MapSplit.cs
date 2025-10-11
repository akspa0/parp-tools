using System;
using System.IO;
using System.Numerics;

namespace NewPM4Reader.PM4
{
    /// <summary>
    /// Represents a map split in the PM4 file format.
    /// </summary>
    public class MapSplit
    {
        /// <summary>
        /// Gets or sets the center point of the map split.
        /// </summary>
        public Vector3 Center { get; set; }

        /// <summary>
        /// Gets or sets the normal vector of the map split.
        /// </summary>
        public Vector3 Normal { get; set; }

        /// <summary>
        /// Gets or sets the height of the map split.
        /// </summary>
        public float Height { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MapSplit"/> class with default values.
        /// </summary>
        public MapSplit()
        {
            Center = Vector3.Zero;
            Normal = new Vector3(0, 1, 0); // Default to Y-up
            Height = 0;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="MapSplit"/> class with specified values.
        /// </summary>
        /// <param name="center">The center point of the map split.</param>
        /// <param name="normal">The normal vector of the map split.</param>
        /// <param name="height">The height of the map split.</param>
        public MapSplit(Vector3 center, Vector3 normal, float height)
        {
            Center = center;
            Normal = normal;
            Height = height;
        }

        /// <summary>
        /// Reads the map split data from a binary reader.
        /// </summary>
        /// <param name="reader">The binary reader to read from.</param>
        public void ReadBinary(BinaryReader reader)
        {
            float centerX = reader.ReadSingle();
            float centerY = reader.ReadSingle();
            float centerZ = reader.ReadSingle();
            Center = new Vector3(centerX, centerY, centerZ);

            float normalX = reader.ReadSingle();
            float normalY = reader.ReadSingle();
            float normalZ = reader.ReadSingle();
            Normal = new Vector3(normalX, normalY, normalZ);

            Height = reader.ReadSingle();
        }

        /// <summary>
        /// Writes the map split data to a binary writer.
        /// </summary>
        /// <param name="writer">The binary writer to write to.</param>
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