using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using WoWMapConverter.Core.Formats.Shared;

namespace WoWMapConverter.Core.Formats.LichKing
{
    public class Mcnk
    {
        public const string Signature = "MCNK";
        public McnkHeader Header;
        public float[] Heightmap; // MCVT
        public Mcal AlphaMaps; // MCAL
        public List<MclyEntry> TextureLayers; // MCLY
        public byte[] GeneratedNormals; // To store normals if needed

        // Raw subchunks if needed
        public byte[] McvtData;
        public byte[] MccvData;
        public byte[] McnrData;
        public byte[] McshData;

        public Mcnk(byte[] data)
        {
            Parse(data);
        }

        private void Parse(byte[] data)
        {
            if (data.Length < 128)
                throw new InvalidDataException("MCNK data too short for header");

            using (var ms = new MemoryStream(data))
            using (var br = new BinaryReader(ms))
            {
                // Read Header (128 bytes)
                Header = ReadHeader(br);

                
                // Read chunks by Offset
                // This is robust against Header size variations (e.g. 128 vs 128+Padding)
                // Note: Offsets are relative to the start of the MCNK chunk (Position 0 of 'ms')
                
                // MCVT (Heights)
                if (Header.HeightmapOffset > 0 && Header.HeightmapOffset < data.Length)
                {
                    ms.Seek(Header.HeightmapOffset, SeekOrigin.Begin);
                    uint magic = br.ReadUInt32();
                    uint size = br.ReadUInt32();
                    if (magic == 0x4D435654) // MCVT
                    {
                        Heightmap = ReadMcvtData(br, size);
                    }
                }

                // MCLY (Layers)
                if (Header.TextureLayersOffset > 0 && Header.TextureLayersOffset < data.Length)
                {
                    ms.Seek(Header.TextureLayersOffset, SeekOrigin.Begin);
                    uint magic = br.ReadUInt32();
                    uint size = br.ReadUInt32();
                    if (magic == 0x4D434C59) // MCLY
                    {
                        TextureLayers = ReadMclyData(br, size);
                    }
                }

                // MCAL (Alpha Maps)
                if (Header.AlphaMapsOffset > 0 && Header.AlphaMapsOffset < data.Length)
                {
                    ms.Seek(Header.AlphaMapsOffset, SeekOrigin.Begin);
                    uint magic = br.ReadUInt32();
                    uint size = br.ReadUInt32();
                    if (magic == 0x4D43414C) // MCAL
                    {
                        AlphaMaps = new Mcal(br.ReadBytes((int)size));
                    }
                }

                // MCNR (Normals)
                if (Header.VertexNormalOffset > 0 && Header.VertexNormalOffset < data.Length)
                {
                    ms.Seek(Header.VertexNormalOffset, SeekOrigin.Begin);
                    uint magic = br.ReadUInt32();
                    uint size = br.ReadUInt32();
                    if (magic == 0x4D434E52) // MCNR
                    {
                        McnrData = br.ReadBytes((int)size);
                    }
                }

                // MCSH (Shadows)
                if (Header.BakedShadowsOffset > 0 && Header.BakedShadowsOffset < data.Length)
                {
                    ms.Seek(Header.BakedShadowsOffset, SeekOrigin.Begin);
                    uint magic = br.ReadUInt32();
                    uint size = br.ReadUInt32();
                    if (magic == 0x4D435348) // MCSH
                    {
                        McshData = br.ReadBytes((int)size);
                    }
                }
            }
        }

        private McnkHeader ReadHeader(BinaryReader br)
        {
            var h = new McnkHeader();
            h.Flags = (McnkFlags)br.ReadUInt32();
            h.IndexX = br.ReadUInt32();
            h.IndexY = br.ReadUInt32();
            h.Layers = br.ReadUInt32();
            h.DoodadRefs = br.ReadUInt32();
            
            // Read offsets (still useful for some data, or debugging)
            h.HeightmapOffset = br.ReadUInt32();        // MCVT
            h.VertexNormalOffset = br.ReadUInt32();     // MCNR
            h.TextureLayersOffset = br.ReadUInt32();    // MCLY
            h.ModelReferencesOffset = br.ReadUInt32();  // MCRF
            h.AlphaMapsOffset = br.ReadUInt32();        // MCAL
            h.BakedShadowsOffset = br.ReadUInt32();     // MCSH Offset
            h.BakedShadowsSize = br.ReadUInt32();       // MCSH Size (0x2C)
            h.AreaId = br.ReadUInt32();                 // 0x30
            h.MapObjRefs = br.ReadUInt32();             // 0x34
            h.Holes = br.ReadUInt32();                  // 0x38
            
            // Skip to 0x70 for Position (Z, X, Y)
            // Current pos: 0x3C (60 bytes read). 0x70 is 112.
            // 112 - 60 = 52 bytes skip.
            // Confirmed via Ghidra (FUN_00699ac0 reads offset 0x70 for Z).
            br.ReadBytes(52); 
            
            h.Position = new float[] {
                br.ReadSingle(), // Z (Height)
                br.ReadSingle(), // X
                br.ReadSingle()  // Y
            };

            // Header is 128 bytes. Current pos: 104 + 12 = 116.
            // 128 - 116 = 12 bytes remaining.
            br.ReadBytes(12);

            return h;
        }

        private float[] ReadMcvtData(BinaryReader br, uint size)
        {
            int count = (int)(size / 4);
            var heights = new float[count];
            for (int i = 0; i < count; i++)
            {
                heights[i] = br.ReadSingle();
            }
            return heights;
        }

        private List<MclyEntry> ReadMclyData(BinaryReader br, uint size)
        {
            var list = new List<MclyEntry>();
            long end = br.BaseStream.Position + size;
            
            while (br.BaseStream.Position < end)
            {
                var e = new MclyEntry();
                e.TextureId = br.ReadUInt32();
                e.Flags = (MclyFlags)br.ReadUInt32();
                e.AlphaMapOffset = br.ReadUInt32();
                e.EffectId = br.ReadUInt32();
                list.Add(e);
            }
            return list;
        }
    }

    public struct McnkHeader
    {
        public McnkFlags Flags;
        public uint IndexX;
        public uint IndexY;
        public uint Layers;
        public uint DoodadRefs;
        public uint HeightmapOffset;
        public uint VertexNormalOffset;
        public uint TextureLayersOffset;
        public uint ModelReferencesOffset;
        public uint AlphaMapsOffset;
        public uint BakedShadowsSize;
        public uint BakedShadowsOffset;
        public uint AreaId;
        public uint MapObjRefs;
        public uint Holes;
        public float[] Position;
    }

    [Flags]
    public enum McnkFlags : uint
    {
        HasShadows = 0x1,
        Impassable = 0x2,
        River = 0x4,
        Ocean = 0x8,
        HasMagma = 0x10,
        HasSlime = 0x20,
        HasMccv = 0x40,
        // ...
        HasBakedShadows = 0x20000 // Just referencing likely flags
    }
}
