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

        public byte[]? GetAlphaMapForLayer(MclyEntry mclyEntry, bool bigAlpha = false, bool doNotFixAlphaMap = false)
        {
            if ((_data == null || _data.Length == 0) || (mclyEntry.Flags & MclyFlags.UseAlpha) == 0)
                return null;

            int offset = checked((int)mclyEntry.AlphaMapOffset);
            if ((uint)offset >= (uint)_data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
                return ReadCompressedAlpha(_data, offset);

            if (bigAlpha)
                return ReadBigAlpha(_data, offset);

            var result = ReadUncompressedAlpha4Bit(_data, offset);
            return doNotFixAlphaMap ? result : ApplyEdgeFix(result);
        }

        public byte[]? GetAlphaMapForLayerRelaxed(MclyEntry mclyEntry, int? nextLayerAlphaOffset, bool bigAlphaDefault = false, bool doNotFixAlphaMap = false)
        {
            if (_data == null || _data.Length == 0)
                return null;

            int offset = unchecked((int)mclyEntry.AlphaMapOffset);
            if (offset < 0 || offset >= _data.Length)
                return null;

            bool compressed = (mclyEntry.Flags & MclyFlags.CompressedAlpha) != 0;
            if (compressed)
                return ReadCompressedAlpha(_data, offset);

            int available = _data.Length - offset;
            int inferredSpan = available;
            if (nextLayerAlphaOffset.HasValue)
            {
                int next = nextLayerAlphaOffset.Value;
                if (next > offset)
                    inferredSpan = Math.Min(inferredSpan, next - offset);
            }

            if (bigAlphaDefault)
                return ReadBigAlpha(_data, offset);

            if (inferredSpan >= 4096)
                return ReadBigAlpha(_data, offset);

            if (inferredSpan >= 2048)
            {
                var result = ReadUncompressedAlpha4Bit(_data, offset);
                return doNotFixAlphaMap ? result : ApplyEdgeFix(result);
            }

            return ReadCompressedAlpha(_data, offset);
        }

        public static byte[] ReadCompressedAlpha(byte[] src, int offset)
            => ReadCompressedAlphaWithSize(src, offset).Data;

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
                if (count == 0)
                    continue;

                if (fill)
                {
                    if (srcPos >= src.Length)
                        break;

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
                    int availableCopy = Math.Min(n, src.Length - srcPos);
                    if (availableCopy > 0)
                    {
                        Buffer.BlockCopy(src, srcPos, dst, dstPos, availableCopy);
                        srcPos += availableCopy;
                        dstPos += availableCopy;
                    }

                    if (availableCopy < n)
                        break;
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
                    dst[writePos++] = col == 31 ? lowVal : highVal;
                }
            }

            return dst;
        }

        private static byte[] ApplyEdgeFix(byte[] alpha)
        {
            if (alpha.Length != 64 * 64)
                return alpha;

            for (int y = 0; y < 64; y++)
                alpha[y * 64 + 63] = alpha[y * 64 + 62];

            Buffer.BlockCopy(alpha, 62 * 64, alpha, 63 * 64, 64);
            return alpha;
        }
    }

    [Flags]
    public enum MclyFlags : uint
    {
        AnimationRotationMask = 0x7,
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
