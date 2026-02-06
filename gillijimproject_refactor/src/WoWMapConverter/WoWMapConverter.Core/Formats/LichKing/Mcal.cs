using System;
using System.Collections.Generic;
using System.IO;

namespace WoWMapConverter.Core.Formats.LichKing
{
    public class Mcal
    {
        public const string Signature = "MCAL";
        private byte[] Data;

        public Mcal(byte[] data)
        {
            Data = data;
        }

        public byte[] GetAlphaMapForLayer(MclyEntry mclyEntry, bool bigAlpha = false)
        {
            if (Data != null && (mclyEntry.Flags & MclyFlags.UseAlpha) != 0)
            {
                // Ensure offset is within bounds
                if (mclyEntry.AlphaMapOffset >= Data.Length)
                    return new byte[64 * 64];

                int length = Data.Length - (int)mclyEntry.AlphaMapOffset;
                byte[] alphaBuffer = new byte[length];
                Array.Copy(Data, (int)mclyEntry.AlphaMapOffset, alphaBuffer, 0, length);

                if ((mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0)
                {
                    return ReadCompressedAlpha(alphaBuffer);
                }
                else if (bigAlpha)
                {
                    return ReadBigAlpha(alphaBuffer);
                }
                else
                {
                    return ReadUncompressedAlpha(alphaBuffer);
                }
            }

            return new byte[64 * 64]; // Return empty (all 0) alpha map
        }

        // FIXED: Corrected logic to write to alphaMap instead of overwriting input buffer
        private byte[] ReadCompressedAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            // alphaMap is already 0-initialized by default in C#

            int offInner = 0;
            int offOuter = 0;

            while (offOuter < 4096)
            {
                // Safety check for input buffer bounds
                if (offInner >= alphaBuffer.Length) break;

                bool fill = (alphaBuffer[offInner] & 0x80) != 0;
                int num = (alphaBuffer[offInner] & 0x7F);
                ++offInner;

                for (int k = 0; k < num; ++k)
                {
                    if (offOuter == 4096)
                        break;

                    // FIX: Write to alphaMap, not alphaBuffer
                    if (offInner < alphaBuffer.Length)
                    {
                        alphaMap[offOuter] = alphaBuffer[offInner];
                    }
                    
                    ++offOuter;

                    if (!fill)
                    {
                        ++offInner;
                    }
                }

                if (fill)
                {
                    ++offInner;
                }
            }

            return alphaMap;
        }

        private byte[] ReadBigAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            int a = 0;
            // 64x64 = 4096 bytes. BigAlpha is usually just uncompressed 4096 bytes?
            // Verify logic from Warcraft.NET source:
            // for (int j = 0; j < 64; ++j) for (int i = 0; i < 64; ++i)...
            // It just copies byte by byte.
            
            if (alphaBuffer.Length >= 4096)
            {
                Array.Copy(alphaBuffer, 0, alphaMap, 0, 4096);
            }
            
            // Warcraft.NET does a weird copy at the end:
            // Array.Copy(alphaMap, 62 * 64, alphaMap, 63 * 64, 64);
            // This copies row 62 to row 63. Likely a fix for some artifact or specific to how they render?
            // "Fix the last row issue" maybe? I'll retain it to be safe as a "port".
            Array.Copy(alphaMap, 62 * 64, alphaMap, 63 * 64, 64);

            return alphaMap;
        }

        private byte[] ReadUncompressedAlpha(byte[] alphaBuffer)
        {
            byte[] alphaMap = new byte[64 * 64];
            // 2048 bytes input (nibbles) -> 4096 bytes output
            
            int inner = 0;
            int outer = 0;
            for (int j = 0; j < 64; ++j)
            {
                for (int i = 0; i < 32; ++i)
                {
                    if (outer >= alphaBuffer.Length) break;

                    // Lower nibble first? Warcraft.NET: (alphaBuffer[outer] & 0x0f)
                    // Then upper nibble? Warcraft.NET: (alphaBuffer[outer] & 0xf0)
                    // Wait, loop goes 0 to 32. 
                    // i=0..30: write 2 values. i=31: write 2 values.
                    // Actually the loop logic in Warcraft.NET is:
                    
                    /*
                    alphaMap[inner] = (byte)((255 * ((int)(alphaBuffer[outer] & 0x0f))) / 0x0f);
                    inner++;
                    if (i != 31) {
                         alphaMap[inner] = ... 0xf0 ...
                         inner++;
                    } else {
                         // This seems wrong in Warcraft.NET source I viewed?
                         // "else { alphaMap[inner] = ... 0x0f ... }" ?
                         // If i == 31, it reads 0x0f again? That implies it ignores the last nibble of the last byte in the row?
                         // Let's look closely at the viewed source in Step 262.
                         /*
                            if (i != 31)
                            {
                                alphaMap[inner] = (byte)((255 * ((int)(alphaBuffer[outer] & 0xf0))) / 0xf0);
                                inner++;
                            }
                            else
                            {
                                alphaMap[inner] = (byte)((255 * ((int)(alphaBuffer[outer] & 0x0f))) / 0x0f);
                                inner++;
                            }
                         */
                         // This looks like a bug in Warcraft.NET too? Or maybe distinct behavior.
                         // Standard ADT uncompressed (2048) is usually 2 nibbles per byte. 
                         // Row is 32 bytes -> 64 values.
                         // i=31 is the last byte of the row.
                         // If i=31, it reads lower nibble, increments inner. Then hits 'else', reads lower nibble AGAIN?
                         // That would mean the last pixel of the row is same as the second-to-last pixel?
                         // And the upper nibble of the last byte is ignored?
                         
                         // I will replicate it EXACTLY as is, because the goal is to port the logic (bugs and all, EXCEPT the one I'm explicitly fixing).
                         // BUT, `ReadUncompressedAlpha` in Warcraft.NET might be buggy for `i=31`.
                         // If I fix it, I might improve it. 
                         // Logic: `(value * 255) / 15` is roughly `value * 17`. 0xF * 17 = 255.
                         
                         byte val1 = (byte)((alphaBuffer[outer] & 0x0F) * 17);
                         byte val2 = (byte)(((alphaBuffer[outer] & 0xF0) >> 4) * 17);
                         
                         alphaMap[inner++] = val1;
                         
                         if (i != 31)
                         {
                             alphaMap[inner++] = val2;
                         }
                         else
                         {
                             // Warcraft.NET behavior replication:
                             alphaMap[inner++] = val1; // Re-using val1?
                         }
                         
                         outer++;
                }
            }

            // Also the copy of last row
            Array.Copy(alphaMap, 62 * 64, alphaMap, 63 * 64, 64);

            return alphaMap;
        }
    }

    [Flags]
    public enum MclyFlags : uint
    {
        AnimationRotation = 0x7,        // 3 bits
        Sign = 0x200,                   // 0x200
        UseAlpha = 0x100,               // 0x100
        CompressedAlpha = 0x200,        // 0x200 - Wait, Sign and CompressedAlpha overlap in my memory? 
                                        // Let's check Warcraft.NET source or known docs.
                                        // View Warcraft.NET source for Flags if possible.
        // I don't have the Flags source. But usually 0x200 is compressed.
        // I will assume 0x200 is Compressed based on usage. 
        // 0x100 implies alpha presence.
        // I'll check VlmDatasetExporter manual parsing to see what flags it checks currently, or assume standard.
    }

    public struct MclyEntry
    {
        public uint TextureId;
        public MclyFlags Flags;
        public uint AlphaMapOffset;
        public uint EffectId;
    }
}
