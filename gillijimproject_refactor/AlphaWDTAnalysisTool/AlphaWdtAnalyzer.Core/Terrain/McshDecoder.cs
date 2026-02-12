using System;

namespace AlphaWdtAnalyzer.Core.Terrain;

/// <summary>
/// Decodes MCSH (MapChunk Shadow) compressed shadow maps.
/// Reference: noggit-red/src/noggit/MapChunk.cpp lines 235-264
/// </summary>
public static class McshDecoder
{
    /// <summary>
    /// Decodes 512-byte compressed MCSH to 4096-byte shadow map (64×64).
    /// Each bit unpacks to a shadow value: 0 = shadowed (dark), 85 = lit (bright).
    /// </summary>
    /// <param name="compressed">512-byte compressed shadow data</param>
    /// <returns>4096-byte uncompressed shadow map</returns>
    public static byte[] Decode(byte[] compressed)
    {
        if (compressed.Length != 512)
            throw new ArgumentException("MCSH data must be 512 bytes", nameof(compressed));
        
        var uncompressed = new byte[4096];
        int outputIndex = 0;
        
        // Each byte unpacks to 8 shadow values
        // Reference: noggit MapChunk.cpp
        foreach (byte compressedByte in compressed)
        {
            // Process each bit (LSB to MSB)
            for (int bit = 0; bit < 8; bit++)
            {
                int mask = 1 << bit;
                // Bit set = 85 (lit), bit clear = 0 (shadowed)
                uncompressed[outputIndex++] = (compressedByte & mask) != 0 ? (byte)85 : (byte)0;
            }
        }
        
        return uncompressed;
    }
    
    /// <summary>
    /// Decodes and applies edge fixing (duplicates row/col 62 to 63).
    /// This prevents rendering artifacts at chunk edges.
    /// </summary>
    public static byte[] DecodeWithEdgeFix(byte[] compressed)
    {
        var uncompressed = Decode(compressed);
        
        // Fix last row and column by copying second-to-last
        // Reference: noggit MapChunk.cpp edge fixing
        for (int i = 0; i < 64; i++)
        {
            uncompressed[i * 64 + 63] = uncompressed[i * 64 + 62];  // Right edge
            uncompressed[63 * 64 + i] = uncompressed[62 * 64 + i];  // Bottom edge
        }
        uncompressed[63 * 64 + 63] = uncompressed[62 * 64 + 62];  // Corner
        
        return uncompressed;
    }
    
    /// <summary>
    /// Encodes shadow map as intensity digits (0-5) for compact CSV storage.
    /// Each character represents shadow intensity: '0' = dark, '5' = light.
    /// </summary>
    public static string EncodeAsDigits(byte[] shadowMap)
    {
        if (shadowMap.Length != 4096)
            throw new ArgumentException("Shadow map must be 4096 bytes (64×64)", nameof(shadowMap));
        
        // Map: 0 → '0' (dark), 85 → '5' (light)
        var chars = new char[shadowMap.Length];
        for (int i = 0; i < shadowMap.Length; i++)
        {
            chars[i] = shadowMap[i] == 0 ? '0' : '5';
        }
        return new string(chars);
    }
    
    /// <summary>
    /// Decodes intensity digits back to shadow map for rendering.
    /// </summary>
    public static byte[] DecodeFromDigits(string digits)
    {
        if (digits.Length != 4096)
            throw new ArgumentException("Shadow digit string must be 4096 chars", nameof(digits));
        
        var shadowMap = new byte[4096];
        for (int i = 0; i < 4096; i++)
        {
            shadowMap[i] = digits[i] == '0' ? (byte)0 : (byte)85;
        }
        return shadowMap;
    }
}
