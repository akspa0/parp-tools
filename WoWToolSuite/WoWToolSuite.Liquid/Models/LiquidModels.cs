using System;
using System.Collections.Generic;
using System.Numerics;
using System.IO;

namespace WowToolSuite.Liquid.Models
{
    public static class LiquidConstants
    {
        public const float COORDINATE_THRESHOLD = 17066.66f;

        public static readonly Dictionary<uint, string> LiquidTypeMapping = new Dictionary<uint, string>
        {
            { 0, "Water" },
            { 1, "Ocean" },
            { 2, "Magma" },
            { 3, "Slime" },
            { 4, "Slow Water" },
            { 5, "Slow Ocean" },
            { 6, "Slow Magma" },
            { 7, "Slow Slime" },
            { 8, "Fast Water" },
            { 9, "Fast Ocean" },
            { 10, "Fast Magma" },
            { 11, "Fast Slime" },
            { 12, "WMO Water" },
            { 13, "WMO Ocean" },
            { 14, "Green Lava" },
            { 15, "WMO Magma" },
            { 16, "WMO Slime" },
            { 17, "Naxxramas Slime" },
            { 18, "Coilfang Raid Water" },
            { 19, "Hyjal Past Water" },
            { 20, "WMO Green Lava" },
            { 21, "Sunwell Raid Water" }
        };

        public static readonly Dictionary<string, string> TextureMapping = new()
        {
            { "ocean", "textures/Blue_1.png" },
            { "still", "textures/WaterBlue_1.png" },
            { "river", "textures/WaterBlue_1.png" },
            { "fast flowing", "textures/WaterBlue_1.png" },
            { "magma", "textures/Red_1.png" },
            { "?", "textures/Grey_1.png" },
            { "slime", "textures/Green_1.png" }
        };

        public static string GetTextureFilename(string liquidType, string outputDirectory)
        {
            // Get the base texture name
            var baseTexture = TextureMapping.GetValueOrDefault(liquidType, "textures/Grey_1.png");
            
            // Check if texture exists in output directory's texture folder
            var outputTexturePath = Path.Combine(outputDirectory, baseTexture);
            if (File.Exists(outputTexturePath))
            {
                return baseTexture;
            }

            // Check if texture exists in executable's directory texture folder
            var exeDirectory = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? "";
            var exeTexturePath = Path.Combine(exeDirectory, baseTexture);
            if (File.Exists(exeTexturePath))
            {
                // Copy texture to output directory
                var outputTextureDir = Path.Combine(outputDirectory, "textures");
                Directory.CreateDirectory(outputTextureDir);
                File.Copy(exeTexturePath, outputTexturePath, true);
                return baseTexture;
            }

            // Return default texture if none found
            return "textures/Grey_1.png";
        }
    }

    public class LiquidHeader
    {
        public string Magic { get; set; } = string.Empty;
        public ushort Version { get; set; }
        public ushort Unk06 { get; set; }
        public ushort LiquidType { get; set; }
        public ushort Padding { get; set; }
        public uint BlockCount { get; set; }

        public string LiquidTypeString => LiquidConstants.LiquidTypeMapping.GetValueOrDefault(LiquidType, "unknown");
    }

    public class LiquidFile
    {
        public LiquidHeader Header { get; set; } = new LiquidHeader();
        public string FilePath { get; set; } = string.Empty;
        public bool IsWlm { get; set; }
        public List<LiquidBlock> Blocks { get; set; } = new List<LiquidBlock>();
    }
} 