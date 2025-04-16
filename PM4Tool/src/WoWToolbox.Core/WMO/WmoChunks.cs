using System;
using System.IO;
using System.Numerics;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace WoWToolbox.Core.WMO
{
    // MOGP chunk: WMO Group Header (adapted from Warcraft.NET, as struct for binary parsing)
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MOGP
    {
        public uint GroupNameOffset;
        public uint DescriptiveGroupNameOffset;
        public uint Flags;
        public Vector3 BoundingBoxMin;
        public Vector3 BoundingBoxMax;
        public ushort FirstPortalReferenceIndex;
        public ushort PortalReferenceCount;
        public ushort RenderBatchCountA;
        public ushort RenderBatchCountInterior;
        public ushort RenderBatchCountExterior;
        public ushort Unknown;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public byte[] FogIndices;
        public uint LiquidType;
        public uint GroupID;
        public uint TerrainFlags;
        public uint Unused;

        public static MOGP FromBinaryReader(BinaryReader br)
        {
            var mogp = new MOGP
            {
                GroupNameOffset = br.ReadUInt32(),
                DescriptiveGroupNameOffset = br.ReadUInt32(),
                Flags = br.ReadUInt32(),
                BoundingBoxMin = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                BoundingBoxMax = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                FirstPortalReferenceIndex = br.ReadUInt16(),
                PortalReferenceCount = br.ReadUInt16(),
                RenderBatchCountA = br.ReadUInt16(),
                RenderBatchCountInterior = br.ReadUInt16(),
                RenderBatchCountExterior = br.ReadUInt16(),
                Unknown = br.ReadUInt16(),
                FogIndices = br.ReadBytes(4),
                LiquidType = br.ReadUInt32(),
                GroupID = br.ReadUInt32(),
                TerrainFlags = br.ReadUInt32(),
                Unused = br.ReadUInt32()
            };
            return mogp;
        }
    }

    // MOVT chunk: Vertex positions (array of Vector3)
    public struct MOVT
    {
        public Vector3 Position;
        public static List<Vector3> ReadArray(BinaryReader br, int count)
        {
            var list = new List<Vector3>(count);
            for (int i = 0; i < count; i++)
            {
                float x = br.ReadSingle();
                float z = br.ReadSingle();
                float y = -br.ReadSingle(); // WoW Y/Z swap
                list.Add(new Vector3(x, y, z));
            }
            return list;
        }
    }

    // MONR chunk: Normals (array of Vector3)
    public struct MONR
    {
        public Vector3 Normal;
        public static List<Vector3> ReadArray(BinaryReader br, int count)
        {
            var list = new List<Vector3>(count);
            for (int i = 0; i < count; i++)
            {
                float x = br.ReadSingle();
                float z = br.ReadSingle();
                float y = -br.ReadSingle();
                list.Add(new Vector3(x, y, z));
            }
            return list;
        }
    }

    // MOTV chunk: UVs (array of Vector2)
    public struct MOTV
    {
        public Vector2 UV;
        public static List<Vector2> ReadArray(BinaryReader br, int count)
        {
            var list = new List<Vector2>(count);
            for (int i = 0; i < count; i++)
            {
                float u = br.ReadSingle();
                float v = br.ReadSingle();
                list.Add(new Vector2(u, v));
            }
            return list;
        }
    }

    // MOVI chunk: Indices (array of ushort)
    public struct MOVI
    {
        public ushort Index;
        public static List<ushort> ReadArray(BinaryReader br, int count)
        {
            var list = new List<ushort>(count);
            for (int i = 0; i < count; i++)
                list.Add(br.ReadUInt16());
            return list;
        }
    }

    // MVER chunk: Version (uint32)
    public struct MVER
    {
        public uint Version;
        public static MVER FromBinaryReader(BinaryReader br) => new MVER { Version = br.ReadUInt32() };
    }

    // MOHD chunk: WMO Root Header
    public struct MOHD
    {
        public uint MaterialCount;
        public uint GroupCount;
        public uint PortalCount;
        public uint LightCount;
        public uint ModelCount;
        public uint DoodadCount;
        public uint SetCount;
        public uint AmbientColor;
        public uint AreaTableID;
        public Vector3 BoundingBoxMin;
        public Vector3 BoundingBoxMax;
        public ushort Flags;
        public ushort LodCount;
        public static MOHD FromBinaryReader(BinaryReader br)
        {
            return new MOHD
            {
                MaterialCount = br.ReadUInt32(),
                GroupCount = br.ReadUInt32(),
                PortalCount = br.ReadUInt32(),
                LightCount = br.ReadUInt32(),
                ModelCount = br.ReadUInt32(),
                DoodadCount = br.ReadUInt32(),
                SetCount = br.ReadUInt32(),
                AmbientColor = br.ReadUInt32(),
                AreaTableID = br.ReadUInt32(),
                BoundingBoxMin = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                BoundingBoxMax = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                Flags = br.ReadUInt16(),
                LodCount = br.ReadUInt16()
            };
        }
    }

    // MOTX chunk: Texture names (null-terminated string block)
    public struct MOTX
    {
        // Helper to read all null-terminated strings from a byte array
        public static List<string> ReadStrings(byte[] data)
        {
            var result = new List<string>();
            int start = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0)
                {
                    if (i > start)
                        result.Add(System.Text.Encoding.UTF8.GetString(data, start, i - start));
                    start = i + 1;
                }
            }
            // Handle trailing string without null terminator
            if (start < data.Length)
                result.Add(System.Text.Encoding.UTF8.GetString(data, start, data.Length - start));
            return result;
        }
    }

    // MFOG chunk: Fog (array of 48-byte structs)
    public struct MFOG
    {
        public uint Flags;
        public Vector3 Position;
        public float RadiusSmall;
        public float RadiusLarge;
        public float FogEnd;
        public float FogStartScalar;
        public uint FogColor;
        public float UnderwaterFogEnd;
        public float UnderwaterFogStartScalar;
        public uint UnderwaterFogColor;
        public static MFOG FromBinaryReader(BinaryReader br)
        {
            return new MFOG
            {
                Flags = br.ReadUInt32(),
                Position = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                RadiusSmall = br.ReadSingle(),
                RadiusLarge = br.ReadSingle(),
                FogEnd = br.ReadSingle(),
                FogStartScalar = br.ReadSingle(),
                FogColor = br.ReadUInt32(),
                UnderwaterFogEnd = br.ReadSingle(),
                UnderwaterFogStartScalar = br.ReadSingle(),
                UnderwaterFogColor = br.ReadUInt32()
            };
        }
    }

    // MOMT chunk: Material (array of 64-byte structs)
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MOMT
    {
        public uint Flags;
        public uint Shader;
        public uint BlendMode;
        public uint Texture1;
        public uint Color1;
        public uint Color1b;
        public uint Texture2;
        public uint Color2;
        public uint GroupType;
        public uint Texture3;
        public uint Color3;
        public uint Flags3;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public uint[] RuntimeData;
        public static MOMT FromBinaryReader(BinaryReader br)
        {
            var momt = new MOMT
            {
                Flags = br.ReadUInt32(),
                Shader = br.ReadUInt32(),
                BlendMode = br.ReadUInt32(),
                Texture1 = br.ReadUInt32(),
                Color1 = br.ReadUInt32(),
                Color1b = br.ReadUInt32(),
                Texture2 = br.ReadUInt32(),
                Color2 = br.ReadUInt32(),
                GroupType = br.ReadUInt32(),
                Texture3 = br.ReadUInt32(),
                Color3 = br.ReadUInt32(),
                Flags3 = br.ReadUInt32(),
                RuntimeData = new uint[4]
            };
            for (int i = 0; i < 4; i++) momt.RuntimeData[i] = br.ReadUInt32();
            return momt;
        }
    }

    // MOGI chunk: Group Info (array of 32-byte structs)
    public struct MOGI
    {
        public uint Flags;
        public Vector3 BoundingBoxMin;
        public Vector3 BoundingBoxMax;
        public int NameIndex;
        public static MOGI FromBinaryReader(BinaryReader br)
        {
            return new MOGI
            {
                Flags = br.ReadUInt32(),
                BoundingBoxMin = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                BoundingBoxMax = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                NameIndex = br.ReadInt32()
            };
        }
    }

    // MODS chunk: Doodad Set (array of 32-byte structs)
    public struct MODS
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 20)]
        public string Name;
        public uint FirstInstanceIndex;
        public uint DoodadCount;
        public uint Unused;
        public static MODS FromBinaryReader(BinaryReader br)
        {
            var nameBytes = br.ReadBytes(20);
            return new MODS
            {
                Name = System.Text.Encoding.UTF8.GetString(nameBytes).TrimEnd('\0'),
                FirstInstanceIndex = br.ReadUInt32(),
                DoodadCount = br.ReadUInt32(),
                Unused = br.ReadUInt32()
            };
        }
    }

    // MODI chunk: Doodad IDs (array of uint32)
    public struct MODI
    {
        public static List<uint> ReadArray(BinaryReader br, int count)
        {
            var list = new List<uint>(count);
            for (int i = 0; i < count; i++)
                list.Add(br.ReadUInt32());
            return list;
        }
    }

    // MODN chunk: Doodad Names (null-terminated string block)
    public struct MODN
    {
        // Helper to read all null-terminated strings from a byte array
        public static List<string> ReadStrings(byte[] data)
        {
            var result = new List<string>();
            int start = 0;
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] == 0)
                {
                    if (i > start)
                        result.Add(System.Text.Encoding.UTF8.GetString(data, start, i - start));
                    start = i + 1;
                }
            }
            if (start < data.Length)
                result.Add(System.Text.Encoding.UTF8.GetString(data, start, data.Length - start));
            return result;
        }
    }

    // MODD chunk: Doodad Definitions (array of 40-byte structs)
    public struct MODD
    {
        public uint Offset;
        public byte Flags;
        public Vector3 Position;
        public Quaternion Rotation;
        public float Scale;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public byte[] Color;
        public static MODD FromBinaryReader(BinaryReader br)
        {
            return new MODD
            {
                Offset = br.ReadUInt32(),
                Flags = br.ReadByte(),
                Position = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                Rotation = new Quaternion(br.ReadSingle(), br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                Scale = br.ReadSingle(),
                Color = br.ReadBytes(4)
            };
        }
    }

    // GFID chunk: Group file Data IDs (array of uint32)
    public struct GFID
    {
        public static List<uint> ReadArray(BinaryReader br, int count)
        {
            var list = new List<uint>(count);
            for (int i = 0; i < count; i++)
                list.Add(br.ReadUInt32());
            return list;
        }
    }

    // MOPY chunk: Material info per triangle (array of 2-byte structs)
    public struct MOPY
    {
        public byte Flags;
        public byte MaterialId;
        public static List<MOPY> ReadArray(BinaryReader br, int count)
        {
            var list = new List<MOPY>(count);
            for (int i = 0; i < count; i++)
                list.Add(new MOPY { Flags = br.ReadByte(), MaterialId = br.ReadByte() });
            return list;
        }
    }

    // MOCV chunk: Vertex colors (array of uint32)
    public struct MOCV
    {
        public uint Color;
        public static List<uint> ReadArray(BinaryReader br, int count)
        {
            var list = new List<uint>(count);
            for (int i = 0; i < count; i++)
                list.Add(br.ReadUInt32());
            return list;
        }
    }

    // MLIQ chunk: Liquid geometry (header + variable data)
    public struct MLIQ
    {
        public uint LiquidVertsX;
        public uint LiquidVertsY;
        public uint LiquidTilesX;
        public uint LiquidTilesY;
        public Vector3 Corner;
        public ushort MaterialID;
        public byte[] RawData; // Remaining data
        public static MLIQ FromBinaryReader(BinaryReader br, int chunkSize)
        {
            var start = br.BaseStream.Position;
            var mliq = new MLIQ
            {
                LiquidVertsX = br.ReadUInt32(),
                LiquidVertsY = br.ReadUInt32(),
                LiquidTilesX = br.ReadUInt32(),
                LiquidTilesY = br.ReadUInt32(),
                Corner = new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()),
                MaterialID = br.ReadUInt16()
            };
            int bytesRead = (int)(br.BaseStream.Position - start);
            mliq.RawData = br.ReadBytes(chunkSize - bytesRead);
            return mliq;
        }
    }

    // MDAL chunk: Ambient Color (uint32)
    public struct MDAL
    {
        public uint AmbientColor;
        public static MDAL FromBinaryReader(BinaryReader br) => new MDAL { AmbientColor = br.ReadUInt32() };
    }

    // MOBA chunk: Bounding box/area info (commonly 3 uint32 fields)
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct MOBA
    {
        public uint MinIndex;
        public uint MaxIndex;
        public uint Flags;
        public static MOBA FromBinaryReader(BinaryReader br)
        {
            return new MOBA
            {
                MinIndex = br.ReadUInt32(),
                MaxIndex = br.ReadUInt32(),
                Flags = br.ReadUInt32()
            };
        }

        public static List<MOBA> ReadArray(BinaryReader br, int count)
        {
            var list = new List<MOBA>(count);
            for (int i = 0; i < count; i++)
                list.Add(FromBinaryReader(br));
            return list;
        }
    }
} 