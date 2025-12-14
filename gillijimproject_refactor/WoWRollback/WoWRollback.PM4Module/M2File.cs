using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WoWRollback.PM4Module
{
    /// <summary>
    /// Minimal M2 parser for extracting geometry (vertices) for PM4 matching.
    /// Based on WoWFormatLib.Structs.M2.
    /// </summary>
    public class M2File
    {
        public M2Header Header { get; private set; }
        public List<Vector3> Vertices { get; } = new();
        public Vector3 BoundsMin { get; private set; }
        public Vector3 BoundsMax { get; private set; }

        public M2File(string path) : this(File.ReadAllBytes(path)) { }

        public M2File(byte[] data)
        {
            using var ms = new MemoryStream(data);
            using var br = new BinaryReader(ms);

            // Parse Chunked Header (MD21/MD20)
            uint magic = br.ReadUInt32();
            uint size = br.ReadUInt32();

            if (magic == 0x3132444D) // "MD21"
            {
                // MD21 wraps the MD20 chunk
                long md20Start = br.BaseStream.Position;
                uint md20Magic = br.ReadUInt32();
                if (md20Magic != 0x3032444D) // "MD20"
                    throw new Exception("Invalid M2: MD21 chunk does not contain MD20.");
                
                // Read MD20 Header
                ParseMD20(br, md20Start);
            }
            else if (magic == 0x3032444D) // "MD20"
            {
                // Raw MD20 header
                ParseMD20(br, 0); // Offset 0 is technically start of MD20 if no chunk header? 
                // Wait, if we read magic+size, we are 8 bytes in.
                // Standard M2 usually starts with MD20. 
                // Line 173 of M2Reader: ReadUInt32() -> Check MD20.
                
                // My logic above consumed 8 bytes.
                // If it was MD20, we already read Magic (4) and Size (4)? 
                // Actually M2Reader Line 55 reads ChunkName then Size.
                
                // Let's reset and parse properly assuming standard chunk structure check
                // But old M2s might just be MD20 header directly?
                // M2Reader handles `chunkName == MD21`.
                // ParseYeOldeM2Struct reads header and checks MD20.
                
                // Let's restart stream read inside ParseMD20 context if needed.
                // The provided logic assumes standard chunk wrapping if MD21.
                // If MD20, we need to handle the fact we read 8 bytes.
                
                // Correct approach:
                // M2 files start with MD20 directly usually in older clients, but current clients use chunks?
                // M2Reader line 55: reads magic.
                
                // Re-implementing logic to be safe:
                ms.Position = 0;
                magic = br.ReadUInt32();
                
                if (magic == 0x3132444D) // MD21
                {
                    uint chunkSize = br.ReadUInt32();
                    long chunkStart = br.BaseStream.Position;
                    // Inside MD21 is the M2 data starting with MD20
                    uint subMagic = br.ReadUInt32();
                    if (subMagic != 0x3032444D) throw new Exception("MD21 without MD20");
                    // Read MD20 header (subMagic was the first 4 bytes)
                    // Pass offset to start of MD20
                    ParseMD20(br, chunkStart);
                }
                else if (magic == 0x3032444D) // MD20
                {
                    // Direct MD20
                    ParseMD20(br, 0); 
                    // Note: ParseMD20 will seek to relative offsets. 
                    // offsets in M2 are relative to the start of MD20 data.
                }
                else
                {
                    throw new Exception($"Unknown M2 magic: {magic:X}");
                }
            }
        }

        private void ParseMD20(BinaryReader br, long baseOffset)
        {
            br.BaseStream.Position = baseOffset;
            
            // M2 Header (MD20)
            uint magic = br.ReadUInt32(); // MD20
            uint version = br.ReadUInt32();
            
            uint lName = br.ReadUInt32();
            uint ofsName = br.ReadUInt32();
            
            uint flags = br.ReadUInt32();
            
            uint nGlobalSequences = br.ReadUInt32();
            uint ofsGlobalSequences = br.ReadUInt32();
            
            uint nAnimations = br.ReadUInt32();
            uint ofsAnimations = br.ReadUInt32();
            
            uint nAnimationLookup = br.ReadUInt32();
            uint ofsAnimationLookup = br.ReadUInt32();
            
            uint nBones = br.ReadUInt32();
            uint ofsBones = br.ReadUInt32();
            
            uint nKeyBoneLookup = br.ReadUInt32();
            uint ofsKeyBoneLookup = br.ReadUInt32();
            
            uint nVertices = br.ReadUInt32();
            uint ofsVertices = br.ReadUInt32();
            
            uint nViews = br.ReadUInt32(); // nSkins normally
            
            // Skip to bounding box (need to skip Colors, Textures, Transparency, UVAnim, TexReplace, RenderFlags, BoneLookup, TexLookup, Unk1, TransLookup, UVAnimLookup)
            // Struct Layout:
            // ... (Skipping fields)
            
            // Let's just calculate offset to BoundingBox if possible?
            // Offsets are dynamic.
            // BoundingBox is in the fixed header part.
            // Let's conform to M2Reader.cs ParseYeOldeM2Struct order.
            
            // We are at nViews (offset 0x34 usually?)
            // struct M2Header {
            //   0x00 MD20
            //   0x04 Version
            //   0x08 lName
            //   0x0C ofsName
            //   0x10 Flags
            //   0x14 nGlobalSeq
            //   0x18 ofsGlobalSeq
            //   0x1C nAnim
            //   0x20 ofsAnim
            //   0x24 nAnimLookup
            //   0x28 ofsAnimLookup
            //   0x2C nBones
            //   0x30 ofsBones
            //   0x34 nKeyBone
            //   0x38 ofsKeyBone
            //   0x3C nVertices
            //   0x40 ofsVertices
            //   0x44 nViews
            //   ... 
            // }
            
            // Wait, standard header:
            // 00: MD20
            // 04: Version
            // 08: lName
            // 0C: ofsName
            // 10: Flags
            // 14: nLoops
            // 18: ofsLoops
            // 1C: nSeq
            // 20: ofsSeq
            // 24: nSeqLook
            // 28: ofsSeqLook
            // 2C: nBones
            // 30: ofsBones
            // 34: nBoneLook
            // 38: ofsBoneLook
            // 3C: nVertices
            // 40: ofsVertices
            // 44: nSkins (Views)
            
            // 48: nColors
            // 4C: ofsColors
            // 50: nTextures
            // 54: ofsTextures
            // 58: nTrans
            // 5C: ofsTrans
            // 60: nUvAnim
            // 64: ofsUvAnim
            // 68: nTexReplace
            // 6C: ofsTexReplace
            // 70: nRenderFlags
            // 74: ofsRenderFlags
            // 78: nBoneLookTable
            // 7C: ofsBoneLookTable
            // 80: nTexLook
            // 84: ofsTexLook
            // 88: nUnk1
            // 8C: ofsUnk1
            // 90: nTransLook
            // 94: ofsTransLook
            // 98: nUvAnimLook
            // 9C: ofsUvAnimLook
            // A0: VertexBox[0] (Min)
            // AC: VertexBox[1] (Max)
            // B8: VertexRadius
            // BC: BoundingBox[0] (Min)
            // C8: BoundingBox[1] (Max)
            // D4: BoundingRadius

            // Let's seek to Bounding Box
            // BoundingBox starts at 0xBC (188 decimal)
            // But checking M2Reader.cs again to be sure:
            // nUVAnimLookup (0x98), ofs (0x9C)
            // vertexbox (0xA0)
            
            br.BaseStream.Position = baseOffset + 0xA0;
            Vector3 vMin = FromRaw(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            Vector3 vMax = FromRaw(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
            float vRad = br.ReadSingle();
            
            BoundsMin = FromRaw(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()); // BoundingBox Min
            BoundsMax = FromRaw(br.ReadSingle(), br.ReadSingle(), br.ReadSingle()); // BoundingBox Max
            
            // Read Vertices
            br.BaseStream.Position = baseOffset + ofsVertices;
            for (int i = 0; i < nVertices; i++)
            {
                // Read Position (12 bytes)
                float x = br.ReadSingle();
                float y = br.ReadSingle();
                float z = br.ReadSingle();
                
                // Convert (X, Z, -Y) or just Raw? 
                // M2 is usually Z-up in WoW world space? No, M2 is Y-up locally (Z points out). 
                // Wait. M2 Vertices: X (Right), Y (Up), Z (Back)? 
                // Or X (Right), Y (Back), Z (Up)?
                // WoW World: Z is Up. X is North? Y is West? 
                // Actually:
                // ADT: X (North), Y (West), Z (Up).
                // M2: X (Right), Y (Back), Z (Up).
                // PM4: Z-up.
                // Pm4WmoGeometryMatcher expects Z-up.
                // Standard M2 reader reads raw floats.
                // Let's store raw for now and rely on Matcher to fix rotation (it computes pca axes).
                // BUT if I mess up the hand (Mirroring), PCA can't fix it if determinant is negative?
                // Matcher handles rotation. Does it handle mirroring? 
                // "WMOs cannot be scaled" -> "M2s can be scaled".
                
                Vertices.Add(new Vector3(x, y, z));
                
                // Skip rest of vertex stride (48 - 12 = 36 bytes)
                br.BaseStream.Seek(36, SeekOrigin.Current);
            }
        }
        
        private Vector3 FromRaw(float x, float y, float z) => new Vector3(x, y, z);
        
        public struct M2Header
        {
            // Fields if needed
        }
    }
}
