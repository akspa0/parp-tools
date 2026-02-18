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
        /// Notes (3.3.5 client behavior, Ghidra):
        /// - Uncompressed (4-bit) alpha is stored as 2048 bytes (nibbles) and expanded to 64x64.
        /// - Big alpha uses 4096 bytes (8-bit) and is used when MPHD bigAlpha flag is set.
        /// - Compressed alpha uses an RLE stream (high bit = fill) that expands to 4096 bytes.
        /// - The client duplicates the last row/column to avoid seams; we apply the same edge fix.
        /// </summary>
        public byte[]? GetAlphaMapForLayer(MclyEntry mclyEntry, bool bigAlpha = false)
        {
            if ((_data == null || _data.Length == 0) || (mclyEntry.Flags & MclyFlags.UseAlpha) == 0)
                return null;

            int offset = checked((int)mclyEntry.AlphaMapOffset);
            if ((uint)offset >= (uint)_data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
            {
                // RLE stream length is not explicitly known; decode until 4096 output bytes are produced.
                return ApplyEdgeFix(ReadCompressedAlpha(_data, offset));
            }

            if (bigAlpha)
            {
                // 4096 bytes (64x64)
                return ApplyEdgeFix(ReadBigAlpha(_data, offset));
            }

            // 2048 bytes (4-bit, nibbles)
            return ApplyEdgeFix(ReadUncompressedAlpha4Bit(_data, offset));
        }

        /// <summary>
        /// Relaxed decoder for cases where MCLY flags are missing/mis-set but MCAL + AlphaMapOffset
        /// are still valid. Uses next-layer offset (when provided) to infer whether the alpha is
        /// likely 2048-byte 4-bit vs 4096-byte 8-bit.
        /// </summary>
        public byte[]? GetAlphaMapForLayerRelaxed(MclyEntry mclyEntry, int? nextLayerAlphaOffset, bool bigAlphaDefault = false)
        {
            if (_data == null || _data.Length == 0)
                return null;

            int offset = unchecked((int)mclyEntry.AlphaMapOffset);
            if (offset < 0 || offset >= _data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
                return ApplyEdgeFix(ReadCompressedAlpha(_data, offset));

            int available = _data.Length - offset;
            int inferredSpan = available;
            if (nextLayerAlphaOffset.HasValue)
            {
                int next = nextLayerAlphaOffset.Value;
                if (next > offset)
                    inferredSpan = Math.Min(inferredSpan, next - offset);
            }

            // If the WDT bigAlpha flag is set, prefer 8-bit.
            if (bigAlphaDefault)
                return ApplyEdgeFix(ReadBigAlpha(_data, offset));

            // Infer 8-bit when the layer span clearly supports it.
            if (inferredSpan >= 4096)
                return ApplyEdgeFix(ReadBigAlpha(_data, offset));

            // Default to 4-bit for 3.3.5 when not bigAlpha.
            if (inferredSpan >= 2048)
                return ApplyEdgeFix(ReadUncompressedAlpha4Bit(_data, offset));

            // Last resort: try compressed decode (some chunks have odd spans/padding).
            return ApplyEdgeFix(ReadCompressedAlpha(_data, offset));
        }

        private static byte[] ReadCompressedAlpha(byte[] src, int offset)
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

            return dst;
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

            // 2048 bytes input (nibbles) -> 4096 bytes output
            int maxBytes = Math.Min(2048, src.Length - offset);
            for (int i = 0; i < maxBytes; i++)
            {
                byte packed = src[offset + i];
                int outIndex = i * 2;
                dst[outIndex] = (byte)((packed & 0x0F) * 17);
                dst[outIndex + 1] = (byte)(((packed >> 4) & 0x0F) * 17);
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
