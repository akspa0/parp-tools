using System;
using System.Text;

namespace WCAnalyzer.Core.Models
{
    /// <summary>
    /// Represents a texture layer in a terrain chunk.
    /// </summary>
    public class TextureLayer
    {
        /// <summary>
        /// Gets or sets the texture ID for the layer.
        /// </summary>
        public int TextureId { get; set; }

        /// <summary>
        /// Gets or sets the flags for the layer.
        /// </summary>
        public uint Flags { get; set; }

        /// <summary>
        /// Gets or sets the effect ID for the layer.
        /// </summary>
        public int EffectId { get; set; }

        /// <summary>
        /// Gets or sets the alpha map offset for the layer.
        /// </summary>
        public int AlphaMapOffset { get; set; }

        /// <summary>
        /// Gets or sets the alpha map size for the layer.
        /// </summary>
        public int AlphaMapSize { get; set; }

        /// <summary>
        /// Gets or sets the alpha map data for the layer.
        /// </summary>
        public byte[] AlphaMap { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// Gets or sets the texture name for the layer.
        /// </summary>
        public string TextureName { get; set; } = string.Empty;

        /// <summary>
        /// Gets a value indicating whether the texture layer uses alpha blending.
        /// </summary>
        public bool HasAlphaMap => AlphaMap != null && AlphaMap.Length > 0;

        /// <summary>
        /// Gets the blend mode for the layer.
        /// </summary>
        public int BlendMode => (int)((Flags >> 24) & 0x7);

        /// <summary>
        /// Gets a value indicating whether the texture is compressed.
        /// </summary>
        public bool IsCompressed => (Flags & 0x200) != 0;

        /// <summary>
        /// Gets a human-readable representation of the texture layer's alpha map.
        /// </summary>
        /// <returns>A string containing the alpha map data.</returns>
        public string GetAlphaMapString()
        {
            if (AlphaMap == null || AlphaMap.Length == 0)
                return "No alpha map data available.";

            var sb = new StringBuilder();
            sb.AppendLine($"Alpha map for texture {TextureId} ({TextureName}):");

            // Alpha maps are typically 64x64 for a chunk
            int size = (int)Math.Sqrt(AlphaMap.Length);
            if (size * size != AlphaMap.Length)
            {
                sb.AppendLine($"Alpha map size is not square: {AlphaMap.Length} bytes");
                return sb.ToString();
            }

            // Output a simplified representation of the alpha map
            for (int y = 0; y < size; y += 4)
            {
                for (int x = 0; x < size; x += 4)
                {
                    // Calculate average alpha value for this 4x4 block
                    int sum = 0;
                    int count = 0;
                    for (int dy = 0; dy < 4 && y + dy < size; dy++)
                    {
                        for (int dx = 0; dx < 4 && x + dx < size; dx++)
                        {
                            sum += AlphaMap[(y + dy) * size + (x + dx)];
                            count++;
                        }
                    }
                    int avg = count > 0 ? sum / count : 0;

                    // Output a character representing the alpha value
                    char c = ' ';
                    if (avg > 224)
                        c = '#';
                    else if (avg > 192)
                        c = '=';
                    else if (avg > 160)
                        c = '+';
                    else if (avg > 128)
                        c = '-';
                    else if (avg > 96)
                        c = '.';
                    else if (avg > 64)
                        c = ',';
                    else if (avg > 32)
                        c = '`';

                    sb.Append(c);
                }
                sb.AppendLine();
            }

            return sb.ToString();
        }
    }
} 