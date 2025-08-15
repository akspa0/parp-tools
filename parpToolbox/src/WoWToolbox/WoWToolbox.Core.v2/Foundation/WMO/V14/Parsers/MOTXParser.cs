using System;
using System.Collections.Generic;
using System.Text;

namespace WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers
{
    internal static class MOTXParser
    {
        /// <summary>
        /// Parses the payload of a MOTX chunk, which is a null-terminated sequence of C-style strings.
        /// </summary>
        public static List<string> Parse(ReadOnlySpan<byte> payload)
        {
            var list = new List<string>();
            int start = 0;
            for (int i = 0; i < payload.Length; i++)
            {
                if (payload[i] == 0)
                {
                    if (i > start)
                    {
                        list.Add(Encoding.ASCII.GetString(payload.Slice(start, i - start)));
                    }
                    start = i + 1;
                }
            }
            // Handle final segment if file is malformed and lacks null term
            if (start < payload.Length)
                list.Add(Encoding.ASCII.GetString(payload[start..]));
            return list;
        }
    }
}
