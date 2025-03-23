using System;
using System.Collections.Generic;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// Materials used in this map object. 64 bytes per texture (BLP file).
    /// </summary>
    public class MOMT : IChunk
    {
        public const int MATERIAL_SIZE = 64;
        public const int V14_MATERIAL_SIZE = 40;

        [Flags]
        public enum MaterialFlags : uint
        {
            F_UNLIT = 0x1,         // Disable lighting logic in shader (but can still use vertex colors)
            F_UNFOGGED = 0x2,      // Disable fog shading (rarely used)
            F_UNCULLED = 0x4,      // Two-sided
            F_EXTLIGHT = 0x8,      // Darkened, the intern face of windows are flagged 0x08
            F_SIDN = 0x10,         // Bright at night, unshaded (used on windows and lamps in Stormwind)
            F_WINDOW = 0x20,       // Lighting related (flag checked in CMapObj::UpdateSceneMaterials)
            F_CLAMP_S = 0x40,      // Tex clamp S (force this material's textures to use clamp s addressing)
            F_CLAMP_T = 0x80,      // Tex clamp T (force this material's textures to use clamp t addressing)
            F_UNKNOWN = 0x100      // Unknown flag
        }

        public class Material
        {
            public MaterialFlags Flags { get; set; }
            public uint Shader { get; set; }            // Not present in v14
            public uint BlendMode { get; set; }
            public uint Texture1 { get; set; }          // Offset into MOTX or FileDataId in 8.1+
            public CImVector SidnColor { get; set; }    // Emissive color
            public CImVector FrameSidnColor { get; set; } // SIDN emissive color set at runtime
            public uint Texture2 { get; set; }
            public CImVector DiffColor { get; set; }
            public uint GroundType { get; set; }
            public uint Texture3 { get; set; }          // Not present in v14
            public uint Color2 { get; set; }            // Not present in v14
            public uint Flags2 { get; set; }            // Not present in v14
            public uint[] RuntimeData { get; set; }     // Not present in v14, size 4

            public Material()
            {
                SidnColor = new CImVector();
                FrameSidnColor = new CImVector();
                DiffColor = new CImVector();
                RuntimeData = new uint[4];
            }
        }

        public List<Material> Materials { get; } = new();
        public bool IsV14 { get; private set; }

        public void Read(BinaryReader reader)
        {
            var startPos = reader.BaseStream.Position;
            var size = reader.BaseStream.Length - startPos;

            // Determine version based on chunk size and material count
            var materialCount = (int)size / MATERIAL_SIZE;
            var v14MaterialCount = (int)size / V14_MATERIAL_SIZE;

            // If the size perfectly divides by V14_MATERIAL_SIZE but not MATERIAL_SIZE, it's v14
            IsV14 = size % MATERIAL_SIZE != 0 && size % V14_MATERIAL_SIZE == 0;

            Materials.Clear();

            for (int i = 0; i < (IsV14 ? v14MaterialCount : materialCount); i++)
            {
                var material = new Material();

                if (IsV14)
                {
                    // Version 14 has a version field instead of shader
                    uint version = reader.ReadUInt32();
                    material.Flags = (MaterialFlags)reader.ReadUInt32();
                    material.BlendMode = reader.ReadUInt32();
                    material.Texture1 = reader.ReadUInt32();
                    material.SidnColor = reader.ReadCImVector();
                    material.FrameSidnColor = reader.ReadCImVector();
                    material.Texture2 = reader.ReadUInt32();
                    material.DiffColor = reader.ReadCImVector();
                    material.GroundType = reader.ReadUInt32();
                    reader.ReadBytes(8); // In-memory padding
                }
                else
                {
                    material.Flags = (MaterialFlags)reader.ReadUInt32();
                    material.Shader = reader.ReadUInt32();
                    material.BlendMode = reader.ReadUInt32();
                    material.Texture1 = reader.ReadUInt32();
                    material.SidnColor = reader.ReadCImVector();
                    material.FrameSidnColor = reader.ReadCImVector();
                    material.Texture2 = reader.ReadUInt32();
                    material.DiffColor = reader.ReadCImVector();
                    material.GroundType = reader.ReadUInt32();
                    material.Texture3 = reader.ReadUInt32();
                    material.Color2 = reader.ReadUInt32();
                    material.Flags2 = reader.ReadUInt32();
                    for (int j = 0; j < 4; j++)
                    {
                        material.RuntimeData[j] = reader.ReadUInt32();
                    }
                }

                Materials.Add(material);
            }
        }

        public void Write(BinaryWriter writer)
        {
            foreach (var material in Materials)
            {
                if (IsV14)
                {
                    writer.Write((uint)14); // Version
                    writer.Write((uint)material.Flags);
                    writer.Write(material.BlendMode);
                    writer.Write(material.Texture1);
                    writer.Write(material.SidnColor);
                    writer.Write(material.FrameSidnColor);
                    writer.Write(material.Texture2);
                    writer.Write(material.DiffColor);
                    writer.Write(material.GroundType);
                    writer.Write(new byte[8]); // In-memory padding
                }
                else
                {
                    writer.Write((uint)material.Flags);
                    writer.Write(material.Shader);
                    writer.Write(material.BlendMode);
                    writer.Write(material.Texture1);
                    writer.Write(material.SidnColor);
                    writer.Write(material.FrameSidnColor);
                    writer.Write(material.Texture2);
                    writer.Write(material.DiffColor);
                    writer.Write(material.GroundType);
                    writer.Write(material.Texture3);
                    writer.Write(material.Color2);
                    writer.Write(material.Flags2);
                    for (int i = 0; i < 4; i++)
                    {
                        writer.Write(material.RuntimeData[i]);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the number of textures required for a given shader type.
        /// </summary>
        public static int GetShaderTextureCount(uint shader)
        {
            return shader switch
            {
                0 => 1,  // Diffuse
                1 => 1,  // Specular
                2 => 1,  // Metal
                3 => 2,  // Env
                4 => 1,  // Opaque
                5 => 2,  // EnvMetal
                6 => 2,  // TwoLayerDiffuse
                7 => 3,  // TwoLayerEnvMetal
                8 => 2,  // TwoLayerTerrain
                9 => 2,  // DiffuseEmissive
                10 => 1, // WaterWindow
                11 => 3, // MaskedEnvMetal
                12 => 3, // EnvMetalEmissive
                13 => 2, // TwoLayerDiffuseOpaque
                14 => 1, // SubmarineWindow
                15 => 2, // Unknown
                16 => 1, // Diffuse Terrain
                _ => 1
            };
        }

        /// <summary>
        /// Gets the number of texture coordinates required for a given shader type.
        /// </summary>
        public static int GetShaderTexCoordCount(uint shader)
        {
            return shader switch
            {
                6 => 2,  // TwoLayerDiffuse
                7 => 2,  // TwoLayerEnvMetal
                9 => 2,  // DiffuseEmissive
                11 => 2, // MaskedEnvMetal
                12 => 2, // EnvMetalEmissive
                13 => 2, // TwoLayerDiffuseOpaque
                15 => 2, // Unknown
                _ => 1
            };
        }

        /// <summary>
        /// Gets the number of colors required for a given shader type.
        /// </summary>
        public static int GetShaderColorCount(uint shader)
        {
            return shader switch
            {
                6 => 2,  // TwoLayerDiffuse
                7 => 2,  // TwoLayerEnvMetal
                8 => 2,  // TwoLayerTerrain
                9 => 2,  // DiffuseEmissive
                11 => 2, // MaskedEnvMetal
                12 => 2, // EnvMetalEmissive
                13 => 2, // TwoLayerDiffuseOpaque
                15 => 2, // Unknown
                _ => 1
            };
        }

        /// <summary>
        /// Validates that a material's textures exist in the MOTX chunk.
        /// </summary>
        public bool ValidateMaterialTextures(Material material, MOTX motx)
        {
            if (motx == null) return true; // If no MOTX chunk, assume FileDataIds are used (8.1+)

            var textureCount = GetShaderTextureCount(material.Shader);
            
            // Check primary texture
            if (material.Texture1 > 0 && motx.GetFilenameByOffset((int)material.Texture1) == null)
                return false;

            // Check secondary texture if needed
            if (textureCount > 1 && material.Texture2 > 0 && motx.GetFilenameByOffset((int)material.Texture2) == null)
                return false;

            // Check tertiary texture if needed
            if (textureCount > 2 && material.Texture3 > 0 && motx.GetFilenameByOffset((int)material.Texture3) == null)
                return false;

            return true;
        }
    }
} 