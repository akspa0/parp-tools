// docs/AlphaWDTReader/snippets/chunk_fourcc.cs
// Purpose: Central FourCC constants and helpers used across snippets.
// Endianness note: ADT files store FourCC as 4 ASCII bytes in little-endian order.
// BinaryReader.ReadUInt32() returns a little-endian uint matching the constants below.
// Some hex viewers display the uint value as a big-endian char sequence (appearing reversed),
// so 'MCLQ' bytes [4D 43 4C 51] might be shown as 'QLCM' even though the file bytes are 'M','C','L','Q'.

using System;
using System.IO;
using System.Text;

namespace Snippets
{
    public static class FourCC
    {
        // Top-level
        public const uint MVER = 0x5245564Du; // 'M','V','E','R' (bytes: 4D 56 45 52)
        public const uint MHDR = 0x5244484Du; // 'M','H','D','R'
        public const uint MCIN = 0x4E49434Du; // 'M','C','I','N'
        public const uint MTEX = 0x5845544Du; // 'M','T','E','X'
        public const uint MMDX = 0x58444D4Du; // 'M','M','D','X'
        public const uint MMID = 0x44494D4Du; // 'M','M','I','D'
        public const uint MWMO = 0x4F4D574Du; // 'M','W','M','O'
        public const uint MWID = 0x4449574Du; // 'M','W','I','D'
        public const uint MDDF = 0x4644444Du; // 'M','D','D','F'
        public const uint MODF = 0x46444F4Du; // 'M','O','D','F'
        public const uint MH2O = 0x4F32484Du; // 'M','H','2','O'
        public const uint MTXF = 0x4658544Du; // 'M','T','X','F' (optional)
        public const uint MFBO = 0x4F42464Du; // 'M','F','B','O' (conditional)

        // Per-chunk
        public const uint MCNK = 0x4B4E434Du; // 'M','C','N','K'
        public const uint MCVT = 0x5456434Du; // 'M','C','V','T'
        public const uint MCNR = 0x524E434Du; // 'M','C','N','R'
        public const uint MCLY = 0x594C434Du; // 'M','C','L','Y'
        public const uint MCRF = 0x4652434Du; // 'M','C','R','F'
        public const uint MCAL = 0x4C41434Du; // 'M','C','A','L'
        public const uint MCSH = 0x4853434Du; // 'M','C','S','H'
        public const uint MCCV = 0x5643434Du; // 'M','C','C','V'
        public const uint MCLQ = 0x514C434Du; // 'M','C','L','Q' (forbidden in outputs)

        // --- Helpers ---

        // Convert 4-char tag to a little-endian uint constant (same scheme as above constants)
        public static uint ToFourCC(string tag)
        {
            if (tag == null || tag.Length != 4) throw new ArgumentException("tag must be 4 chars");
            var b = Encoding.ASCII.GetBytes(tag);
            return (uint)(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24));
        }

        // Read a FourCC (little-endian) from the stream
        public static uint ReadFourCC(BinaryReader br)
        {
            return br.ReadUInt32();
        }

        // Write a FourCC (little-endian) to the stream
        public static void WriteFourCC(BinaryWriter bw, uint fourcc)
        {
            bw.Write(fourcc);
        }

        // Expect the next FourCC to match 'expected'; throws with readable tags on mismatch
        public static void Expect(BinaryReader br, uint expected, string context = "")
        {
            long pos = br.BaseStream.Position;
            uint got = ReadFourCC(br);
            if (got != expected)
            {
                string msg = $"Expected FourCC {ToTag(expected)} but got {ToTag(got)} at 0x{pos:X8}.";
                if (!string.IsNullOrEmpty(context)) msg += $" Context: {context}";
                throw new InvalidDataException(msg);
            }
        }

        // Human-readable tag from the uint (canonical order, e.g., 0x514C434D -> "MCLQ")
        public static string ToTag(uint fourcc)
        {
            var b = new byte[]
            {
                (byte)(fourcc & 0xFF),
                (byte)((fourcc >> 8) & 0xFF),
                (byte)((fourcc >> 16) & 0xFF),
                (byte)((fourcc >> 24) & 0xFF)
            };
            return Encoding.ASCII.GetString(b);
        }

        // Alternate view: bytes shown reversed (how some tools visualize the uint), e.g., "QLCM"
        public static string ToTagReversed(uint fourcc)
        {
            var b = new byte[]
            {
                (byte)((fourcc >> 24) & 0xFF),
                (byte)((fourcc >> 16) & 0xFF),
                (byte)((fourcc >> 8) & 0xFF),
                (byte)(fourcc & 0xFF)
            };
            return Encoding.ASCII.GetString(b);
        }

        // Reverse the byte order of the uint (endianness swap)
        public static uint Reverse(uint v)
        {
            return ((v & 0x000000FFU) << 24) |
                   ((v & 0x0000FF00U) << 8)  |
                   ((v & 0x00FF0000U) >> 8)  |
                   ((v & 0xFF000000U) >> 24);
        }
    }
}
