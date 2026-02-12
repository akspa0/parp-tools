using System.Runtime.InteropServices;
using System.Text;

namespace WoWMapConverter.Core.Utilities;

/// <summary>
/// File and data utilities for WoW format parsing.
/// </summary>
public static class FileUtils
{
    public static byte[] GetWholeFile(string filePath) => File.ReadAllBytes(filePath);

    public static string GetLettersFromFile(FileStream fs, int position)
    {
        Span<byte> buf = stackalloc byte[4];
        fs.Seek(position, SeekOrigin.Begin);
        if (fs.Read(buf) != 4) throw new EndOfStreamException();
        return Encoding.ASCII.GetString(buf);
    }

    public static int GetIntFromFile(FileStream fs, int position)
    {
        Span<byte> buf = stackalloc byte[4];
        fs.Seek(position, SeekOrigin.Begin);
        if (fs.Read(buf) != 4) throw new EndOfStreamException();
        return BitConverter.ToInt32(buf);
    }

    public static byte[] GetBytesFromFile(FileStream fs, int position, int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        var data = new byte[length];
        fs.Seek(position, SeekOrigin.Begin);
        if (fs.Read(data, 0, length) != length) throw new EndOfStreamException();
        return data;
    }

    /// <summary>
    /// Parse NUL-separated string table from byte data.
    /// </summary>
    public static List<string> GetFileNames(ReadOnlySpan<byte> data)
    {
        var list = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                var s = Encoding.ASCII.GetString(data.Slice(start, i - start));
                if (!string.IsNullOrEmpty(s))
                    list.Add(s);
                start = i + 1;
            }
        }
        return list;
    }

    /// <summary>
    /// Get WDT version from file header.
    /// </summary>
    public static int GetWdtVersion(string wdtPath)
    {
        using var fs = File.OpenRead(wdtPath);
        var value = GetIntFromFile(fs, 0x98);
        return value == 65536 ? 0 : 1;
    }

    /// <summary>
    /// Get ADT version from file header.
    /// </summary>
    public static int GetAdtVersion(string adtPath)
    {
        using var fs = File.OpenRead(adtPath);
        var value = GetIntFromFile(fs, 0x18);
        return value == 0 ? 4 : 3;
    }

    public static T ByteArrayToStruct<T>(byte[] bytes) where T : struct
    {
        GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        try
        {
            return Marshal.PtrToStructure<T>(handle.AddrOfPinnedObject());
        }
        finally
        {
            handle.Free();
        }
    }

    public static byte[] StructToByteArray<T>(T structure) where T : struct
    {
        int size = Marshal.SizeOf(structure);
        byte[] arr = new byte[size];
        IntPtr ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(structure, ptr, false);
            Marshal.Copy(ptr, arr, 0, size);
            return arr;
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }
}
