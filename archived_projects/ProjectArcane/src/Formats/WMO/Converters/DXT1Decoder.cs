using System;

namespace ArcaneFileParser.Core.Formats.WMO.Converters
{
    /// <summary>
    /// Handles decoding of DXT1 compressed textures used in v14 WMO lightmaps
    /// </summary>
    public class DXT1Decoder
    {
        private const int BLOCK_SIZE = 8; // 8 bytes per 4x4 pixel block
        private const int BLOCK_DIM = 4;  // 4x4 pixels per block

        public static Color[] DecodeBlock(byte[] data, int offset)
        {
            Color[] colors = new Color[16]; // 4x4 block
            
            // Extract color endpoints (RGB565 format)
            ushort color0 = (ushort)(data[offset] | (data[offset + 1] << 8));
            ushort color1 = (ushort)(data[offset + 2] | (data[offset + 3] << 8));

            // Convert RGB565 to RGB888
            Color c0 = RGB565ToColor(color0);
            Color c1 = RGB565ToColor(color1);

            // Calculate interpolated colors
            Color[] palette = new Color[4];
            palette[0] = c0;
            palette[1] = c1;

            if (color0 > color1)
            {
                // Four-color block
                palette[2] = InterpolateColors(c0, c1, 2, 3);
                palette[3] = InterpolateColors(c0, c1, 1, 3);
            }
            else
            {
                // Three-color block
                palette[2] = InterpolateColors(c0, c1, 1, 2);
                palette[3] = new Color(0, 0, 0, 0); // Transparent
            }

            // Extract color indices (2 bits per pixel)
            uint indices = BitConverter.ToUInt32(data, offset + 4);
            
            // Map indices to colors
            for (int i = 0; i < 16; i++)
            {
                int index = (int)((indices >> (i * 2)) & 0x3);
                colors[i] = palette[index];
            }

            return colors;
        }

        private static Color RGB565ToColor(ushort rgb565)
        {
            int r = (rgb565 >> 11) & 0x1F;
            int g = (rgb565 >> 5) & 0x3F;
            int b = rgb565 & 0x1F;

            // Scale up to 8-bit values
            byte red = (byte)((r * 255 + 15) / 31);
            byte green = (byte)((g * 255 + 31) / 63);
            byte blue = (byte)((b * 255 + 15) / 31);

            return new Color(red, green, blue, 255);
        }

        private static Color InterpolateColors(Color c0, Color c1, int num, int den)
        {
            return new Color(
                (byte)((c0.R * (den - num) + c1.R * num) / den),
                (byte)((c0.G * (den - num) + c1.G * num) / den),
                (byte)((c0.B * (den - num) + c1.B * num) / den),
                255
            );
        }

        public static Color SampleTexture(byte[] textureData, int width, int height, float u, float v)
        {
            // Convert UV coordinates to pixel coordinates
            int x = Math.Min((int)(u * width), width - 1);
            int y = Math.Min((int)(v * height), height - 1);

            // Calculate block coordinates
            int blockX = x / BLOCK_DIM;
            int blockY = y / BLOCK_DIM;
            int blocksPerRow = (width + BLOCK_DIM - 1) / BLOCK_DIM;
            int blockOffset = (blockY * blocksPerRow + blockX) * BLOCK_SIZE;

            // Get pixel within block
            int pixelX = x % BLOCK_DIM;
            int pixelY = y % BLOCK_DIM;
            int pixelIndex = pixelY * BLOCK_DIM + pixelX;

            // Decode block and return the correct pixel
            Color[] blockColors = DecodeBlock(textureData, blockOffset);
            return blockColors[pixelIndex];
        }
    }
} 