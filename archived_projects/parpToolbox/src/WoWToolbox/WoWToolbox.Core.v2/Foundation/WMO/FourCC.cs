using System;
using System.Buffers.Binary;
using System.Linq;
using System.Text;

namespace WoWToolbox.Core.v2.Foundation.WMO
{
    /// <summary>
    /// Utility helpers around Blizzard four-character codes that are stored in little-endian byte order.
    /// </summary>
    public static class FourCC
    {
        /// <summary>
        /// Reads four bytes and returns the canonical four-character code string, automatically handling both normal and reversed order.
        /// </summary>
        /// <remarks>
        /// Some legacy versions (WMO v14) keep the bytes but developers historically read them as UInt32 and then treat the value as big-endian,
        /// so reading the bytes in reverse often yields the correct printable code.  We attempt to detect a printable ASCII sequence and fall
        /// back to the reversed string in ambiguous cases.
        /// </remarks>
        public static string ToString(byte[] fourBytes)
        {
            if (fourBytes.Length != 4) throw new ArgumentException("FourCC needs exactly 4 bytes", nameof(fourBytes));
            string normal = Encoding.ASCII.GetString(fourBytes);
            string reversed = new string(fourBytes.Reverse().Select(b => (char)b).ToArray());

            // Heuristic: if normal already looks like A-Z characters then keep it, else use reversed.
            bool normalLooksAscii = normal.All(ch => ch >= 32 && ch <= 126);
            return normalLooksAscii ? normal : reversed;
        }

        /// <summary>
        /// Reads from a Span&lt;byte&gt; and advances the offset by 4.
        /// </summary>
        public static string ReadString(ReadOnlySpan<byte> span, ref int offset)
        {
            string result = ToString(span.Slice(offset, 4).ToArray());
            offset += 4;
            return result;
        }
    }
}
