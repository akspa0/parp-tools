using System;
using System.Linq;
using System.Text;

namespace WoWToolbox.Core.WMO
{
    /// <summary>
    /// Simple FourCC helper for the legacy Core project (duplicate of Core.v2 helper).
    /// </summary>
    internal static class FourCC
    {
        public static string ToString(byte[] fourBytes)
        {
            if (fourBytes == null || fourBytes.Length != 4) throw new ArgumentException("FourCC must be 4 bytes", nameof(fourBytes));
            string normal = Encoding.ASCII.GetString(fourBytes);
            string reversed = new string(fourBytes.Reverse().Select(b => (char)b).ToArray());
            bool normalAscii = normal.All(ch => ch >= 32 && ch <= 126);
            return normalAscii ? normal : reversed;
        }
    }
}
