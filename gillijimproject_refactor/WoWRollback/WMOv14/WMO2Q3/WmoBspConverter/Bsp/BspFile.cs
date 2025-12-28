using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;

namespace WmoBspConverter.Bsp
{
    /// <summary>
    /// Represents a Quake 3 BSP file with its various data lumps.
    /// </summary>
    public class BspFile
    {
        public BspHeader Header { get; set; } = new BspHeader();
        public List<BspVertex> Vertices { get; set; } = new List<BspVertex>();
        public List<BspFace> Faces { get; set; } = new List<BspFace>();
        public List<BspTexture> Textures { get; set; } = new List<BspTexture>();
        public List<byte[]> Lightmaps { get; set; } = new List<byte[]>();
        public List<BspNode> Nodes { get; set; } = new List<BspNode>();
        public List<int> LeafFaces { get; set; } = new List<int>();
        public List<BspLeaf> Leaves { get; set; } = new List<BspLeaf>();
        public List<int> LeafBrushes { get; set; } = new List<int>();
        public List<BspModel> Models { get; set; } = new List<BspModel>();
        public List<BspBrush> Brushes { get; set; } = new List<BspBrush>();
        public List<BspBrushSide> BrushSides { get; set; } = new List<BspBrushSide>();
        public List<BspPlane> Planes { get; set; } = new List<BspPlane>();
        public int NumClusters { get; set; }
        public int BytesPerCluster { get; set; }
        public byte[] VisData { get; set; } = Array.Empty<byte>();
        public List<string> Entities { get; set; } = new List<string>();
        public List<int> MeshVertices { get; set; } = new List<int>();

        public void Save(string filePath)
        {
            using var stream = File.Create(filePath);
            Save(stream);
        }

        public void Save(Stream stream)
        {
            using var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, leaveOpen: true);

            // Prepare lump data buffers
            var lumpData = new Dictionary<BspLumpType, byte[]>();
            var entitiesText = string.Join("\n", Entities);
            if (!entitiesText.EndsWith("\0")) entitiesText += "\0"; // Q3 expects null-terminated entities
            lumpData[BspLumpType.Entities] = System.Text.Encoding.UTF8.GetBytes(entitiesText);
            lumpData[BspLumpType.Textures] = Textures.SelectMany(t => t.ToByteArray()).ToArray();
            lumpData[BspLumpType.Planes] = Planes.SelectMany(p => p.ToByteArray()).ToArray();
            lumpData[BspLumpType.Nodes] = Nodes.SelectMany(n => n.ToByteArray()).ToArray();
            lumpData[BspLumpType.Leaves] = Leaves.SelectMany(l => l.ToByteArray()).ToArray();
            lumpData[BspLumpType.LeafFaces] = LeafFaces.Select(BitConverter.GetBytes).SelectMany(x => x).ToArray();
            lumpData[BspLumpType.LeafBrushes] = LeafBrushes.Select(BitConverter.GetBytes).SelectMany(x => x).ToArray();
            lumpData[BspLumpType.Models] = Models.SelectMany(m => m.ToByteArray()).ToArray();
            lumpData[BspLumpType.Brushes] = Brushes.SelectMany(b => b.ToByteArray()).ToArray();
            lumpData[BspLumpType.BrushSides] = BrushSides.SelectMany(b => b.ToByteArray()).ToArray();
            lumpData[BspLumpType.Vertices] = Vertices.SelectMany(v => v.ToByteArray()).ToArray();
            lumpData[BspLumpType.MeshVertices] = MeshVertices.Select(BitConverter.GetBytes).SelectMany(x => x).ToArray();
            lumpData[BspLumpType.Effects] = Array.Empty<byte>();
            lumpData[BspLumpType.FaceIndex] = Faces.SelectMany(f => f.ToByteArray()).ToArray();
            lumpData[BspLumpType.Lightmaps] = Lightmaps.SelectMany(l => l).ToArray();
            lumpData[BspLumpType.LightGrid] = Array.Empty<byte>();
            // VisData needs proper header: numClusters + bytesPerCluster + bitset data
            if (NumClusters > 0 && BytesPerCluster > 0)
            {
                var visBytes = new List<byte>();
                visBytes.AddRange(BitConverter.GetBytes(NumClusters));
                visBytes.AddRange(BitConverter.GetBytes(BytesPerCluster));
                visBytes.AddRange(VisData);
                lumpData[BspLumpType.VisData] = visBytes.ToArray();
            }
            else
            {
                lumpData[BspLumpType.VisData] = Array.Empty<byte>();
            }

            // Reserve header space first
            long offset = BspHeader.Size;
            writer.BaseStream.Position = offset;

            // Directory to write back later
            var directory = new BspLumpInfo[17];
            for (int i = 0; i < directory.Length; i++) directory[i] = new BspLumpInfo();

            // Write all lump data in index order with 4-byte alignment
            for (int i = 0; i < 17; i++)
            {
                var type = (BspLumpType)i;
                if (!lumpData.TryGetValue(type, out var data))
                {
                    directory[i] = new BspLumpInfo { Offset = 0, Length = 0 };
                    continue;
                }

                directory[i] = new BspLumpInfo { Offset = (int)offset, Length = data.Length };
                writer.Write(data);
                offset += data.Length;

                // 4-byte align
                long pad = (4 - (offset % 4)) % 4;
                if (pad > 0)
                {
                    writer.Write(new byte[(int)pad]);
                    offset += pad;
                }
            }

            // Now write header at the beginning
            writer.BaseStream.Position = 0;
            writer.Write(BspHeader.Magic);
            writer.Write(BspHeader.Version);
            for (int i = 0; i < 17; i++)
            {
                writer.Write(directory[i].Offset);
                writer.Write(directory[i].Length);
            }

            // Store directory in header object
            Header.Lumps = directory;
        }

        private void WriteLump(BinaryWriter writer, byte[] data, BspLumpType type, ref long offset, List<BspLumpInfo> lumps)
        {
            if (data.Length == 0)
            {
                lumps.Add(new BspLumpInfo { Offset = 0, Length = 0 });
                return;
            }

            lumps.Add(new BspLumpInfo { Offset = (int)offset, Length = data.Length });
            writer.Write(data);
            offset += data.Length;
            
            // Align to 4-byte boundary
            long padding = (4 - (offset % 4)) % 4;
            if (padding > 0)
            {
                writer.Write(new byte[(int)padding]);
                offset += padding;
            }
        }
    }

    public class BspHeader
    {
        // Quake 3 BSP format constants
        public const int Magic = 0x50534249; // 'IBSP' as little-endian int
        public const int Version = 46; // Quake 3 BSP version
        public const int Size = 4 + 4 + (17 * 8); // Magic + Version + 17 LumpInfo structs
        
        public BspLumpInfo[] Lumps { get; set; } = new BspLumpInfo[17];

        public BspHeader()
        {
            // Initialize lump array
            for (int i = 0; i < 17; i++)
            {
                Lumps[i] = new BspLumpInfo();
            }
        }
    }

    public class BspLumpInfo
    {
        public int Offset { get; set; }
        public int Length { get; set; }
    }

    public enum BspLumpType
    {
        Entities = 0,
        Textures = 1,
        Planes = 2,
        Nodes = 3,
        Leaves = 4,
        LeafFaces = 5,
        LeafBrushes = 6,
        Models = 7,
        Brushes = 8,
        BrushSides = 9,
        Vertices = 10,
        MeshVertices = 11,
        Effects = 12,
        FaceIndex = 13,
        Lightmaps = 14,
        LightGrid = 15,
        VisData = 16
    }

    public class BspVertex
    {
        public Vector3 Position { get; set; }
        public Vector2 TextureCoordinate { get; set; }
        public Vector2 LightmapCoordinate { get; set; }
        public Vector3 Normal { get; set; }
        public byte[] Color { get; set; } = new byte[4];

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Position.X));
            result.AddRange(BitConverter.GetBytes(Position.Y));
            result.AddRange(BitConverter.GetBytes(Position.Z));
            result.AddRange(BitConverter.GetBytes(TextureCoordinate.X));
            result.AddRange(BitConverter.GetBytes(TextureCoordinate.Y));
            result.AddRange(BitConverter.GetBytes(LightmapCoordinate.X));
            result.AddRange(BitConverter.GetBytes(LightmapCoordinate.Y));
            result.AddRange(BitConverter.GetBytes(Normal.X));
            result.AddRange(BitConverter.GetBytes(Normal.Y));
            result.AddRange(BitConverter.GetBytes(Normal.Z));
            result.AddRange(Color);
            return result.ToArray();
        }

        public static int Size => 44; // 3*4 + 2*2*4 + 3*4 + 4
    }

    public class BspFace
    {
        public int Texture { get; set; }
        public int Effect { get; set; }
        public int Type { get; set; } // 1=polygon, 2=patch, 3=mesh, 4=billboard
        public int FirstVertex { get; set; }
        public int NumVertices { get; set; }
        public int FirstMeshVertex { get; set; }
        public int NumMeshVertices { get; set; }
        public int Lightmap { get; set; }
        public int LightmapStartS { get; set; }
        public int LightmapStartT { get; set; }
        public int LightmapSizeW { get; set; }
        public int LightmapSizeH { get; set; }
        public Vector3 LightmapOrigin { get; set; }
        public Vector3 LightmapVecsS { get; set; }
        public Vector3 LightmapVecsT { get; set; }
        public Vector3 Normal { get; set; }
        public int PatchSizeW { get; set; }
        public int PatchSizeH { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Texture));
            result.AddRange(BitConverter.GetBytes(Effect));
            result.AddRange(BitConverter.GetBytes(Type));
            result.AddRange(BitConverter.GetBytes(FirstVertex));
            result.AddRange(BitConverter.GetBytes(NumVertices));
            result.AddRange(BitConverter.GetBytes(FirstMeshVertex));
            result.AddRange(BitConverter.GetBytes(NumMeshVertices));
            result.AddRange(BitConverter.GetBytes(Lightmap));
            result.AddRange(BitConverter.GetBytes(LightmapStartS));
            result.AddRange(BitConverter.GetBytes(LightmapStartT));
            result.AddRange(BitConverter.GetBytes(LightmapSizeW));
            result.AddRange(BitConverter.GetBytes(LightmapSizeH));
            result.AddRange(BitConverter.GetBytes(LightmapOrigin.X));
            result.AddRange(BitConverter.GetBytes(LightmapOrigin.Y));
            result.AddRange(BitConverter.GetBytes(LightmapOrigin.Z));
            result.AddRange(BitConverter.GetBytes(LightmapVecsS.X));
            result.AddRange(BitConverter.GetBytes(LightmapVecsS.Y));
            result.AddRange(BitConverter.GetBytes(LightmapVecsS.Z));
            result.AddRange(BitConverter.GetBytes(LightmapVecsT.X));
            result.AddRange(BitConverter.GetBytes(LightmapVecsT.Y));
            result.AddRange(BitConverter.GetBytes(LightmapVecsT.Z));
            result.AddRange(BitConverter.GetBytes(Normal.X));
            result.AddRange(BitConverter.GetBytes(Normal.Y));
            result.AddRange(BitConverter.GetBytes(Normal.Z));
            result.AddRange(BitConverter.GetBytes(PatchSizeW));
            result.AddRange(BitConverter.GetBytes(PatchSizeH));
            return result.ToArray();
        }

        public static int Size => 104;
    }

    public class BspTexture
    {
        public string Name { get; set; } = "";
        public int Flags { get; set; }
        public int Contents { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            var nameBytes = new byte[64];
            var nameBytesEncoded = System.Text.Encoding.UTF8.GetBytes(Name);
            Array.Copy(nameBytesEncoded, nameBytes, Math.Min(nameBytesEncoded.Length, 64));
            result.AddRange(nameBytes);
            result.AddRange(BitConverter.GetBytes(Flags));
            result.AddRange(BitConverter.GetBytes(Contents));
            return result.ToArray();
        }

        public static int Size => 72; // 64 + 4 + 4
    }

    public class BspModel
    {
        public Vector3 Min { get; set; }
        public Vector3 Max { get; set; }
        public int FirstFace { get; set; }
        public int NumFaces { get; set; }
        public int FirstBrush { get; set; }
        public int NumBrushes { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Min.X));
            result.AddRange(BitConverter.GetBytes(Min.Y));
            result.AddRange(BitConverter.GetBytes(Min.Z));
            result.AddRange(BitConverter.GetBytes(Max.X));
            result.AddRange(BitConverter.GetBytes(Max.Y));
            result.AddRange(BitConverter.GetBytes(Max.Z));
            result.AddRange(BitConverter.GetBytes(FirstFace));
            result.AddRange(BitConverter.GetBytes(NumFaces));
            result.AddRange(BitConverter.GetBytes(FirstBrush));
            result.AddRange(BitConverter.GetBytes(NumBrushes));
            return result.ToArray();
        }

        public static int Size => 40;
    }

    public class BspPlane
    {
        public Vector3 Normal { get; set; }
        public float Distance { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Normal.X));
            result.AddRange(BitConverter.GetBytes(Normal.Y));
            result.AddRange(BitConverter.GetBytes(Normal.Z));
            result.AddRange(BitConverter.GetBytes(Distance));
            return result.ToArray();
        }

        public static int Size => 16;
    }

    public class BspNode
    {
        public int Plane { get; set; }
        public int[] Children { get; set; } = new int[2];
        public Vector3 Min { get; set; }
        public Vector3 Max { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Plane));
            result.AddRange(BitConverter.GetBytes(Children[0]));
            result.AddRange(BitConverter.GetBytes(Children[1]));
            // Quake 3 stores node AABB as int32s
            result.AddRange(BitConverter.GetBytes((int)Min.X));
            result.AddRange(BitConverter.GetBytes((int)Min.Y));
            result.AddRange(BitConverter.GetBytes((int)Min.Z));
            result.AddRange(BitConverter.GetBytes((int)Max.X));
            result.AddRange(BitConverter.GetBytes((int)Max.Y));
            result.AddRange(BitConverter.GetBytes((int)Max.Z));
            return result.ToArray();
        }

        public static int Size => 36;
    }

    public class BspLeaf
    {
        public int Cluster { get; set; }
        public int Area { get; set; }
        public Vector3 Min { get; set; }
        public Vector3 Max { get; set; }
        public int FirstFace { get; set; }
        public int NumFaces { get; set; }
        public int FirstBrush { get; set; }
        public int NumBrushes { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Cluster));
            result.AddRange(BitConverter.GetBytes(Area));
            // Quake 3 stores leaf AABB as int32s
            result.AddRange(BitConverter.GetBytes((int)Min.X));
            result.AddRange(BitConverter.GetBytes((int)Min.Y));
            result.AddRange(BitConverter.GetBytes((int)Min.Z));
            result.AddRange(BitConverter.GetBytes((int)Max.X));
            result.AddRange(BitConverter.GetBytes((int)Max.Y));
            result.AddRange(BitConverter.GetBytes((int)Max.Z));
            result.AddRange(BitConverter.GetBytes(FirstFace));
            result.AddRange(BitConverter.GetBytes(NumFaces));
            result.AddRange(BitConverter.GetBytes(FirstBrush));
            result.AddRange(BitConverter.GetBytes(NumBrushes));
            return result.ToArray();
        }

        public static int Size => 48;
    }

    public class BspBrush
    {
        public int FirstSide { get; set; }
        public int NumSides { get; set; }
        public int Texture { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(FirstSide));
            result.AddRange(BitConverter.GetBytes(NumSides));
            result.AddRange(BitConverter.GetBytes(Texture));
            return result.ToArray();
        }

        public static int Size => 12;
    }

    public class BspBrushSide
    {
        public int Plane { get; set; }
        public int Texture { get; set; }

        public byte[] ToByteArray()
        {
            var result = new List<byte>();
            result.AddRange(BitConverter.GetBytes(Plane));
            result.AddRange(BitConverter.GetBytes(Texture));
            return result.ToArray();
        }

        public static int Size => 8;
    }
}