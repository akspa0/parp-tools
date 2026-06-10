using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics; // Assuming Vector3 will be needed later for coordinate transformations
using Warcraft.NET.Files.Interfaces;
// using System.Runtime.InteropServices; // No longer needed unless BitConverter is reintroduced

namespace WoWToolbox.Core.Navigation.PM4.Chunks
{
    /// <summary>
    /// Represents a vertex entry in the MSVT chunk.
    /// Note the YXZ ordering from the file.
    /// Reading as 12-byte C3Vector (float) based on test log analysis, overriding documentation.
    /// </summary>
    public struct MsvtVertex
    {
        public float Y { get; set; } // Read first as float
        public float X { get; set; } // Read second as float
        public float Z { get; set; } // Read third as float

        // Size in file is 12 bytes (float Y, float X, float Z)
        public const int StructSize = 12;

        public override string ToString() => $"(X:{X:F3}, Y:{Y:F3}, Z:{Z:F3})"; // Format as float

        // Constants for coordinate transformation (from documentation PD4.md / M005_MSVT.md)
        private const float CoordinateOffset = 17066.666f;
        private const float HeightScaleFactor = 36.0f; // Factor to divide Z by

        /// <summary>
        /// Converts the internal file coordinates (YXZ floats) to world coordinates (XYZ floats)
        /// according to documentation.
        /// </summary>
        /// <returns>Vector3 representing world coordinates.</returns>
        public Vector3 ToWorldCoordinates()
        {
            // Apply documented transformation (Offset - X, Offset - Y, Z / Scale)
            // The file reads Y value first, then X value.
            // worldX uses the logical X coordinate (which is the second value read, stored in the X property).
            // worldY uses the logical Y coordinate (which is the first value read, stored in the Y property).
            // --- Reverting swap based on log feedback: World X should use File X, World Y should use File Y ---
            // float worldX = CoordinateOffset - X; // Use the X property (read second)
            // float worldY = CoordinateOffset - Y; // Use the Y property (read first)
            // float worldZ = Z / HeightScaleFactor; // Use the Z property (read third)

            // --- Applying simplified transform based on user analysis (X, Y, Z/Scale) --- Attempt 10 ---
            // float worldX = X; // Use X directly
            // float worldY = Y; // Use Y directly
            // float worldZ = Z / HeightScaleFactor; // Scale Z only
            
            // --- Applying revised transform based on user analysis (Y, X, Z*Scale) --- Attempt 11 ---
            // float worldX = Y; // Use Y property (first float read) for world X
            // float worldY = X; // Use X property (second float read) for world Y
            // float worldZ = Z * HeightScaleFactor; // Multiply Z by scale
            
            // --- Standardizing on (Y, Z, -X) transform --- Attempt 12 ---
            // float worldX = Y;  // Use Y property (first float read) for OBJ X
            // float worldY = Z;  // Use Z property (third float read) for OBJ Y (Up)
            // float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)

            // --- Applying (Y, X, Z) transform based on log analysis --- Attempt 13 ---
            // float worldX = Y;  // Use Y property (first float read) for OBJ X
            // float worldY = X;  // Use X property (second float read) for OBJ Y (Up?)
            // float worldZ = Z;  // Use Z property (third float read) for OBJ Z (Depth?)

            // --- Applying standard Z-up -> Y-up transform (X, Z, -Y) --- Attempt 31 ---
            // float worldX = CoordinateOffset - X;
            // float worldY = CoordinateOffset - Y;
            // float worldZ = Z / HeightScaleFactor;

            // --- Reverting MSVT to standard (X, Z, -Y) again --- Attempt 30/34 ---
            // float worldX = X;  // Use X property (second float read) for OBJ X
            // float worldY = Z;  // Use Z property (third float read) for OBJ Y (Up)
            // float worldZ = -Y; // Use Y property (first float read), negated, for OBJ Z (Depth)

            // --- Applying (-Y, Z, -X) for 90 CCW rotation from standard --- Attempt 43 ---
            // float worldX = -Y; // Use Y property (first float read), negated, for OBJ X
            // float worldY = Z;  // Use Z property (third float read) for OBJ Y (Up)
            // float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)

            // --- Applying (-Y, -Z, -X) to invert Y --- Attempt 44 ---
            // float worldX = -Y; // Use Y property (first float read), negated, for OBJ X
            // float worldY = -Z; // Use Z property (third float read), NEGATED, for OBJ Y (Up)
            // float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)

            // --- Applying (-Z, -Y, -X) based on "-Y is top" --- Attempt 45 ---
            // float worldX = -Z; // Use Z property (third float read), negated, for OBJ X
            // float worldY = -Y; // Use Y property (first float read), negated, for OBJ Y (Up)
            // float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)

            // --- Revisiting (X, -Z, Y) --- Attempt 25/46 ---
            // float worldX = X;  // Use X property (second float read) for OBJ X
            // float worldY = -Z; // Use Z property (third float read), NEGATED, for OBJ Y (Up)
            // float worldZ = Y; // Use Y property (first float read) for OBJ Z (Depth)

            // --- Applying (Y, Z, -X) based on upside-down/rotation feedback --- Attempt 47 ---
            float worldX = Y;  // Use Y property (first float read) for OBJ X
            float worldY = Z;  // Use Z property (third float read) for OBJ Y (Up)
            float worldZ = -X; // Use X property (second float read), negated, for OBJ Z (Depth)

            return new Vector3(worldX, worldY, worldZ);
        }

        /// <summary>
        /// Creates an MsvtVertex from standard world coordinates (XYZ floats).
        /// </summary>
        /// <param name="worldPos">Vector3 representing world coordinates.</param>
        /// <returns>MsvtVertex with internal file coordinates (YXZ floats).</returns>
        public static MsvtVertex FromWorldCoordinates(Vector3 worldPos)
        {
            // Note: Inverse calculation - ensure results are float
            return new MsvtVertex
            {
                 X = CoordinateOffset - worldPos.X, // Assign float result
                 Y = CoordinateOffset - worldPos.Y, // Assign float result
                 // Reverse the Z calculation: WorldZ = FileZ / Scale => FileZ = WorldZ * Scale
                 Z = worldPos.Z * HeightScaleFactor // Assign float result
            };
        }
    }

    /// <summary>
    /// Represents the MSVT chunk containing geometry vertices.
    /// </summary>
    public class MSVTChunk : IIFFChunk, IBinarySerializable
    {
        public const string ExpectedSignature = "MSVT";
        public string GetSignature() => ExpectedSignature;

        public List<MsvtVertex> Vertices { get; private set; } = new List<MsvtVertex>();

        /// <inheritdoc/>
        public uint GetSize()
        {
            // Size of MsvtVertex in file is assumed 12 bytes for serialization
            return (uint)(Vertices.Count * MsvtVertex.StructSize);
        }

        /// <inheritdoc/>
        public void LoadBinaryData(byte[] chunkData)
        {
            if (chunkData == null) throw new ArgumentNullException(nameof(chunkData));

            using var ms = new MemoryStream(chunkData);
            using var br = new BinaryReader(ms);
            Load(br);
        }

        /// <inheritdoc/>
        public void Load(BinaryReader br)
        {
            long startPosition = br.BaseStream.Position;
            long size = br.BaseStream.Length - startPosition; 

            if (size < 0) throw new InvalidOperationException("Stream position is beyond its length.");

            // Check against the correct 12-byte structure size
            if (size % MsvtVertex.StructSize != 0)
            {
                Vertices.Clear();
                Console.WriteLine($"Warning: MSVT chunk size {size} is not a multiple of {MsvtVertex.StructSize} bytes (expected 12-byte structure). Vertex data might be corrupt.");
                return; 
            }

            int vertexCount = (int)(size / MsvtVertex.StructSize); // Calculate based on 12 bytes
            Vertices = new List<MsvtVertex>(vertexCount);
            // Console.WriteLine($"Debug: MSVT calculated vertexCount = {vertexCount} from size {size} (using 12-byte struct)"); // DEBUG Line

            for (int i = 0; i < vertexCount; i++)
            {
                if (br.BaseStream.Position + MsvtVertex.StructSize > br.BaseStream.Length)
                {
                    Console.WriteLine($"Warning: MSVT chunk unexpected end of stream at vertex {i}. Read {Vertices.Count} vertices out of expected {vertexCount}.");
                    break; // Stop reading
                }
                // Read in YXZ order as float (12 bytes total)
                var vertex = new MsvtVertex
                {
                    Y = br.ReadSingle(),
                    X = br.ReadSingle(),
                    Z = br.ReadSingle() // Read Z as float
                };
                Vertices.Add(vertex);
            }

            long bytesRead = br.BaseStream.Position - startPosition;
            if (bytesRead != size)
            {
                 Console.WriteLine($"Warning: MSVT chunk read {bytesRead} bytes, expected size {size}. Potential partial read.");
            }
        }

        /// <inheritdoc/>
        public byte[] Serialize(long offset = 0)
        {
            using var ms = new MemoryStream((int)GetSize());
            using var bw = new BinaryWriter(ms);

            foreach (var vertex in Vertices)
            {
                // Write in YXZ order as float
                bw.Write(vertex.Y);
                bw.Write(vertex.X);
                bw.Write(vertex.Z); // Write Z as float
            }

            return ms.ToArray();
        }

        public override string ToString()
        {
            return $"MSVT Chunk [{Vertices.Count} Vertices] (12-byte float struct assumption)";
        }
    }
} 