using System;
using System.IO;
using System.Text;

class InspectLkAdt
{
    static void Main(string[] args)
    {
        if (args.Length == 0)
        {
            Console.WriteLine("Usage: InspectLkAdt <path-to-lk-adt>");
            return;
        }

        var path = args[0];
        if (!File.Exists(path))
        {
            Console.WriteLine($"File not found: {path}");
            return;
        }

        var bytes = File.ReadAllBytes(path);
        Console.WriteLine($"File: {path}");
        Console.WriteLine($"Size: {bytes.Length:N0} bytes");
        Console.WriteLine();

        // Find MHDR
        int mhdrOffset = FindChunk(bytes, "RDHM");
        if (mhdrOffset < 0)
        {
            Console.WriteLine("ERROR: MHDR chunk not found!");
            return;
        }

        Console.WriteLine($"MHDR found at offset: 0x{mhdrOffset:X}");
        int mhdrDataStart = mhdrOffset + 8;
        
        // Read MCIN offset from MHDR
        int mcinRelativeOffset = BitConverter.ToInt32(bytes, mhdrDataStart + 4);
        Console.WriteLine($"MCIN relative offset (from MHDR+8): 0x{mcinRelativeOffset:X} ({mcinRelativeOffset})");
        
        if (mcinRelativeOffset == 0)
        {
            Console.WriteLine("ERROR: MCIN offset is 0!");
            return;
        }

        int mcinAbsoluteOffset = mhdrDataStart + mcinRelativeOffset;
        Console.WriteLine($"MCIN absolute offset: 0x{mcinAbsoluteOffset:X}");

        // Check MCIN chunk
        if (mcinAbsoluteOffset + 8 > bytes.Length)
        {
            Console.WriteLine($"ERROR: MCIN offset {mcinAbsoluteOffset} is beyond file size {bytes.Length}!");
            return;
        }

        string mcinFcc = Encoding.ASCII.GetString(bytes, mcinAbsoluteOffset, 4);
        int mcinSize = BitConverter.ToInt32(bytes, mcinAbsoluteOffset + 4);
        Console.WriteLine($"MCIN FourCC: '{mcinFcc}'");
        Console.WriteLine($"MCIN size: {mcinSize} bytes");

        if (mcinFcc != "NICM")
        {
            Console.WriteLine($"ERROR: Expected 'NICM', got '{mcinFcc}'");
            return;
        }

        // Read first few MCNK offsets
        int mcinDataStart = mcinAbsoluteOffset + 8;
        Console.WriteLine("\nFirst 10 MCNK entries in MCIN:");
        for (int i = 0; i < 10 && i * 16 + 4 <= mcinSize; i++)
        {
            int mcnkRelativeOffset = BitConverter.ToInt32(bytes, mcinDataStart + i * 16);
            int mcnkSize = BitConverter.ToInt32(bytes, mcinDataStart + i * 16 + 4);
            
            if (mcnkRelativeOffset > 0)
            {
                int mcnkAbsoluteOffset = mhdrDataStart + mcnkRelativeOffset;
                string mcnkFcc = mcnkAbsoluteOffset + 4 <= bytes.Length 
                    ? Encoding.ASCII.GetString(bytes, mcnkAbsoluteOffset, 4) 
                    : "????";
                Console.WriteLine($"  [{i}] Relative: 0x{mcnkRelativeOffset:X}, Absolute: 0x{mcnkAbsoluteOffset:X}, Size: {mcnkSize}, FourCC: '{mcnkFcc}'");
            }
            else
            {
                Console.WriteLine($"  [{i}] (empty)");
            }
        }

        // Count total valid MCNK entries
        int validCount = 0;
        for (int i = 0; i < 256 && i * 16 + 4 <= mcinSize; i++)
        {
            int mcnkRelativeOffset = BitConverter.ToInt32(bytes, mcinDataStart + i * 16);
            if (mcnkRelativeOffset > 0)
            {
                int mcnkAbsoluteOffset = mhdrDataStart + mcnkRelativeOffset;
                if (mcnkAbsoluteOffset < bytes.Length)
                    validCount++;
            }
        }

        Console.WriteLine($"\nTotal valid MCNK entries: {validCount}/256");
    }

    static int FindChunk(byte[] data, string fourCC)
    {
        for (int i = 0; i + 8 <= data.Length;)
        {
            string fcc = Encoding.ASCII.GetString(data, i, 4);
            if (fcc == fourCC)
                return i;
            
            int size = BitConverter.ToInt32(data, i + 4);
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i || next > data.Length) break;
            i = next;
        }
        return -1;
    }
}
