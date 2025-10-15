using System;
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
}
