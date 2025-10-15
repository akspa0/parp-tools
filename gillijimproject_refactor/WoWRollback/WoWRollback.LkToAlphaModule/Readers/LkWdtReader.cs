using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.LkToAlphaModule.Readers;

public sealed class LkWdtReader
{
    public byte[] ReadMainTileFlags(string wdtPath)
    {
        if (string.IsNullOrWhiteSpace(wdtPath)) throw new ArgumentException("WDT path required", nameof(wdtPath));
        if (!File.Exists(wdtPath)) throw new FileNotFoundException("WDT not found", wdtPath);

        var bytes = File.ReadAllBytes(wdtPath);
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            // On disk FourCC is reversed. MAIN -> 'NIAM'
            string fourCC = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;

            if (fourCC == "NIAM")
            {
                var main = new byte[size];
                Buffer.BlockCopy(bytes, dataStart, main, 0, size);
                return main;
            }

            i = next;
        }

        return Array.Empty<byte>();
    }

    public List<string> ReadWmoNames(string wdtPath)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(wdtPath)) return result;
        if (!File.Exists(wdtPath)) return result;

        var bytes = File.ReadAllBytes(wdtPath);
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            // On disk FourCC is reversed. MWMO -> 'OMWM'
            string fourCC = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;

            if (fourCC == "OMWM")
            {
                // MWMO contains null-terminated strings
                // Note: Alpha client counts by splitting on nulls, which includes empty strings
                // So "name\0" splits to ["name", ""] = 2 parts
                int pos = dataStart;
                int end = dataStart + size;
                while (pos < end)
                {
                    int nullPos = Array.IndexOf(bytes, (byte)0, pos, end - pos);
                    if (nullPos == -1) nullPos = end;
                    
                    int len = nullPos - pos;
                    if (len > 0)
                    {
                        string name = Encoding.UTF8.GetString(bytes, pos, len);
                        if (!string.IsNullOrWhiteSpace(name))
                        {
                            result.Add(name);
                        }
                    }
                    // Note: We only add non-empty names, but the count will be adjusted in the writer
                    
                    pos = nullPos + 1;
                }
                break;
            }

            i = next;
        }

        return result;
    }
}
