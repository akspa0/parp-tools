using System.Collections.Generic;
using System.Numerics;

namespace WowToolSuite.Liquid.Models
{
    public static class LiquidConstants
    {
        public static readonly Dictionary<uint, string> LiquidTypeMapping = new()
        {
            { 0, "still" },
            { 1, "ocean" },
            { 2, "?" },
            { 3, "slime" },
            { 4, "river" },
            { 6, "magma" },
            { 8, "fast flowing" }
        };

        public static readonly Dictionary<string, string> TextureMapping = new()
        {
            { "ocean", "Blue_1.png" },
            { "still", "WaterBlue_1.png" },
            { "river", "WaterBlue_1.png" },
            { "fast flowing", "WaterBlue_1.png" },
            { "magma", "Red_1.png" },
            { "?", "Grey_1.png" },
            { "slime", "Green_1.png" }
        };

        public const int COORDINATE_THRESHOLD = 32767;
    }

    public class LiquidHeader
    {
        public string Magic { get; set; } = string.Empty;
        public ushort Version { get; set; }
        public ushort Unk06 { get; set; }
        public ushort LiquidType { get; set; }
        public ushort Padding { get; set; }
        public uint BlockCount { get; set; }

        public string LiquidTypeString => LiquidConstants.LiquidTypeMapping.TryGetValue(LiquidType, out var type) ? type : "unknown";
        public string TextureFilename => LiquidConstants.TextureMapping.TryGetValue(LiquidTypeString, out var texture) ? texture : "Grey_1.png";
    }

    public class LiquidBlock
    {
        public List<Vector3> Vertices { get; set; } = new(16); // 16 vertices in a 4x4 grid
        public Vector2 Coord { get; set; }
        public ushort[] Data { get; set; } = new ushort[80];
    }

    public class LiquidFile
    {
        public LiquidHeader Header { get; set; } = new();
        public List<LiquidBlock> Blocks { get; set; } = new();
        public string FilePath { get; set; } = string.Empty;
        public bool IsWlm { get; set; }
    }
} 