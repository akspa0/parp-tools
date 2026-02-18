using System;
using System.Collections.Generic;
using System.IO;

namespace WoWMapConverter.Core.Formats.LichKing
{
    public class Mcal
    {
        public const string Signature = "MCAL";
        private readonly byte[] _data;

        public Mcal(byte[] data)
        {
            _data = data;
        }

        /// <summary>
        /// Decodes the alpha map for a single MCLY layer.
        /// Returns null when the layer does not use an alpha map.
        ///
        /// Notes (3.3.5 client behavior, Ghidra / wowdev.wiki):
        /// - BigAlpha (MPHD 0x4|0x80): 4096 bytes, 8-bit per pixel.  Already 64×64 — NO edge fix.
        /// - Compressed (MCLY 0x200): RLE stream that expands to 4096 bytes. Already 64×64 — NO edge fix.
        /// - 4-bit (default): 2048 nibble bytes, expanded to 64×64.
        ///     If MCNK FLAG_DO_NOT_FIX_ALPHA_MAP (0x8000) is NOT set, the data is the 63×63 unfixed
        ///     variant and the last row/column must be duplicated (edge fix).
        ///     If the flag IS set, the data is already 64×64 — NO edge fix.
        /// </summary>
        /// <param name="doNotFixAlphaMap">
        /// Pass true when MCNK header flag 0x8000 (DoNotFixAlphaMap) is set. Suppresses the
        /// edge-duplication fix for 4-bit alpha maps that are already stored in 64×64 form.
        /// </param>
        public byte[]? GetAlphaMapForLayer(MclyEntry mclyEntry, bool bigAlpha = false, bool doNotFixAlphaMap = false)
        {
            if ((_data == null || _data.Length == 0) || (mclyEntry.Flags & MclyFlags.UseAlpha) == 0)
                return null;

            int offset = checked((int)mclyEntry.AlphaMapOffset);
            if ((uint)offset >= (uint)_data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
            {
                // RLE decompresses to exactly 4096 bytes (64×64) — edge fix never needed.
                return ReadCompressedAlpha(_data, offset);
            }

            if (bigAlpha)
            {
                // 4096 bytes (64×64) — edge fix never needed.
                return ReadBigAlpha(_data, offset);
            }

            // 2048 bytes (4-bit nibbles) — edge fix only when 63×63 unfixed format.
            var result = ReadUncompressedAlpha4Bit(_data, offset);
            return doNotFixAlphaMap ? result : ApplyEdgeFix(result);
        }

        /// <summary>
        /// Relaxed decoder for cases where MCLY flags are missing/mis-set but MCAL + AlphaMapOffset
        /// are still valid. Uses next-layer offset (when provided) to infer whether the alpha is
        /// likely 2048-byte 4-bit vs 4096-byte 8-bit.
        /// </summary>
        /// <param name="doNotFixAlphaMap">
        /// Pass true when MCNK header flag 0x8000 (DoNotFixAlphaMap) is set. Suppresses the
        /// edge-duplication fix for 4-bit alpha maps that are already stored in 64×64 form.
        /// </param>
        public byte[]? GetAlphaMapForLayerRelaxed(MclyEntry mclyEntry, int? nextLayerAlphaOffset, bool bigAlphaDefault = false, bool doNotFixAlphaMap = false)
        {
            if (_data == null || _data.Length == 0)
                return null;

            int offset = unchecked((int)mclyEntry.AlphaMapOffset);
            if (offset < 0 || offset >= _data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
            {
                // RLE decompresses to 4096 bytes (64×64) — edge fix never needed.
                return ReadCompressedAlpha(_data, offset);
            }

            int available = _data.Length - offset;
            int inferredSpan = available;
            if (nextLayerAlphaOffset.HasValue)
            {
                int next = nextLayerAlphaOffset.Value;
                if (next > offset)
                    inferredSpan = Math.Min(inferredSpan, next - offset);
            }

            // If the WDT bigAlpha flag is set, prefer 8-bit (4096 bytes, already 64×64).
            if (bigAlphaDefault)
                return ReadBigAlpha(_data, offset);

            // Infer 8-bit when the layer span clearly supports it.
            if (inferredSpan >= 4096)
                return ReadBigAlpha(_data, offset);

            // Default to 4-bit for 3.3.5 when not bigAlpha.
            if (inferredSpan >= 2048)
            {
                var result = ReadUncompressedAlpha4Bit(_data, offset);
                return doNotFixAlphaMap ? result : ApplyEdgeFix(result);
            }

            // Last resort: try compressed decode (some chunks have odd spans/padding).
            // Compressed always decompresses to 64×64 — no edge fix.
            return ReadCompressedAlpha(_data, offset);
        }

        public static byte[] ReadCompressedAlpha(byte[] src, int offset)
            => ReadCompressedAlphaWithSize(src, offset).Data;

        /// <summary>
        /// RLE-decompress a WotLK MCAL compressed-alpha stream starting at <paramref name="offset"/>.
        /// Returns the decompressed 64×64 data and the number of source bytes consumed.
        /// </summary>
        public static (byte[] Data, int BytesConsumed) ReadCompressedAlphaWithSize(byte[] src, int offset)
        {
            var dst = new byte[64 * 64];
            int srcPos = offset;
            int dstPos = 0;

            while (dstPos < dst.Length && srcPos < src.Length)
            {
                byte ctrl = src[srcPos++];
                bool fill = (ctrl & 0x80) != 0;
                int count = ctrl & 0x7F;
                if (count == 0) continue;

                if (fill)
                {
                    if (srcPos >= src.Length) break;
                    byte value = src[srcPos++];
                    int n = Math.Min(count, dst.Length - dstPos);
                    if (n > 0)
                    {
                        Array.Fill(dst, value, dstPos, n);
                        dstPos += n;
                    }
                }
                else
                {
                    int n = Math.Min(count, dst.Length - dstPos);
                    int available = Math.Min(n, src.Length - srcPos);
                    if (available > 0)
                    {
                        Buffer.BlockCopy(src, srcPos, dst, dstPos, available);
                        srcPos += available;
                        dstPos += available;
                    }

                    // If the stream ends early, leave the rest as 0.
                    if (available < n) break;
                }
            }

            return (dst, srcPos - offset);
        }

        private static byte[] ReadBigAlpha(byte[] src, int offset)
        {
            var dst = new byte[64 * 64];
            int available = Math.Min(dst.Length, src.Length - offset);
            if (available > 0)
                Buffer.BlockCopy(src, offset, dst, 0, available);
            return dst;
        }

        private static byte[] ReadUncompressedAlpha4Bit(byte[] src, int offset)
        {
            var dst = new byte[64 * 64];

            // 2048 bytes input (nibbles) -> 4096 bytes output.
            // Warcraft/Noggit behavior: for the last packed byte in each row (col=31),
            // duplicate low nibble into both output texels instead of using high nibble.
            int readPos = offset;
            int dataEnd = src.Length;
            int writePos = 0;

            for (int row = 0; row < 64 && readPos < dataEnd; row++)
            {
                for (int col = 0; col < 32 && readPos < dataEnd; col++)
                {
                    byte packed = src[readPos++];
                    byte lowVal = (byte)((packed & 0x0F) * 17);
                    byte highVal = (byte)(((packed >> 4) & 0x0F) * 17);

                    dst[writePos++] = lowVal;
                    dst[writePos++] = (col == 31) ? lowVal : highVal;
                }
            }

            return dst;
        }

        private static byte[] ApplyEdgeFix(byte[] alpha)
        {
            if (alpha.Length != 64 * 64)
                return alpha;

            // Duplicate last column from col 62
            for (int y = 0; y < 64; y++)
                alpha[y * 64 + 63] = alpha[y * 64 + 62];

            // Duplicate last row from row 62
            Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);

            return alpha;
        }
    }

    [Flags]
    public enum MclyFlags : uint
    {
        AnimationRotationMask = 0x7, // low 3 bits
        UseAlpha = 0x100,
        CompressedAlpha = 0x200,
    }

    public struct MclyEntry
    {
        public uint TextureId;
        public MclyFlags Flags;
        public uint AlphaMapOffset;
        public uint EffectId;
    }
}
