using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWToolbox.Core.v2.Infrastructure
{
    /// <summary>
    /// Generic helper to parse null-terminated string blocks used in many WoW binary formats (e.g. MOTX, MOGN, MODN).
    /// Returns a list where the index corresponds to the byte offset of the string inside the original block.
    /// Missing offsets are filled with empty strings to preserve index alignment.
    /// </summary>
    public static class StringBlockParser
    {
        public static IReadOnlyList<string> Parse(byte[] data)
        {
            var list = new List<string>();
            int pos = 0;
            while (pos < data.Length)
            {
                int end = Array.IndexOf<byte>(data, 0, pos);
                if (end < 0) end = data.Length;
                string s = Encoding.ASCII.GetString(data, pos, end - pos);
                list.Add(s);
                pos = end + 1;
            }
            return list;
        }
    }
}
