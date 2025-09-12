using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace GillijimProject.Utilities;

/// <summary>
/// [PORT] C# port of utilities found in lib/gillijimproject/utilities/Utilities.{h,cpp}
/// Parity-first; refined later as we port more callers.
/// </summary>
public static class Utilities
{
    // [PORT] getWholeFile → returns byte[] instead of mutating List<char>
    public static byte[] GetWholeFile(string filePath)
    {
        // [PORT] Direct byte[] read; original allocated then assigned into vector<char>.
        return File.ReadAllBytes(filePath);
    }

    // [PORT] getStringFromCharVector
    public static string GetStringFromCharVector(ReadOnlySpan<byte> data, int start, int length)
    {
        if ((uint)(start + length) > (uint)data.Length) throw new ArgumentOutOfRangeException();
        return Encoding.ASCII.GetString(data.Slice(start, length));
    }

    // [PORT] getLettersFromFile (reads 4 bytes at position as ASCII)
    public static string GetLettersFromFile(FileStream fs, int position)
    {
        Span<byte> buf = stackalloc byte[4];
        fs.Seek(position, SeekOrigin.Begin);
        var read = fs.Read(buf);
        if (read != 4) throw new EndOfStreamException();
        return Encoding.ASCII.GetString(buf);
    }

    // [PORT] getIntFromFile (reads little-endian 32-bit)
    public static int GetIntFromFile(FileStream fs, int position)
    {
        Span<byte> buf = stackalloc byte[4];
        fs.Seek(position, SeekOrigin.Begin);
        var read = fs.Read(buf);
        if (read != 4) throw new EndOfStreamException();
        return BitConverter.ToInt32(buf);
    }

    // [PORT] getCharVectorFromFile → byte[]
    public static byte[] GetCharVectorFromFile(FileStream fs, int position, int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        var data = new byte[length];
        fs.Seek(position, SeekOrigin.Begin);
        var read = fs.Read(data, 0, length);
        if (read != length) throw new EndOfStreamException();
        return data;
    }

    // [PORT] getCharVectorFromInt → little-endian 4 bytes
    public static byte[] GetCharVectorFromInt(int value)
    {
        return BitConverter.GetBytes(value);
    }

    // [PORT] getCharVectorFromFloat → raw IEEE754 bytes
    public static byte[] GetCharVectorFromFloat(float value)
    {
        return BitConverter.GetBytes(value);
    }

    // [PORT] flagsExist
    public static bool FlagsExist(int bitmask, int whichFlags) => (bitmask & whichFlags) == whichFlags;

    // [PORT] getAdtVersion: reads int at 0x18; if 0 then 4 else 3
    public static int GetAdtVersion(string adtName)
    {
        using var fs = File.OpenRead(adtName);
        var value = GetIntFromFile(fs, 0x18);
        return value == 0 ? 4 : 3;
    }

    // [PORT] getWdtVersion: reads int at 0x98; if 65536 -> 0 else 1
    public static int GetWdtVersion(string wdtName)
    {
        using var fs = File.OpenRead(wdtName);
        var value = GetIntFromFile(fs, 0x98);
        return value == 65536 ? 0 : 1;
    }

    // [PORT] getFileNames: split by NUL from a byte buffer
    public static List<string> GetFileNames(ReadOnlySpan<byte> data)
    {
        var list = new List<string>();
        int start = 0;
        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] == 0)
            {
                var s = Encoding.ASCII.GetString(data.Slice(start, i - start));
                list.Add(s);
                start = i + 1;
            }
        }
        return list;
    }

    // [PORT] Utilities::Point
    public readonly struct Point
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Z;
        public Point(float x, float y, float z) { X = x; Y = y; Z = z; }
    }

    /// <summary>
    /// [PORT] Convert a byte array to a struct
    /// </summary>
    /// <typeparam name="T">Struct type</typeparam>
    /// <param name="bytes">Byte array containing struct data</param>
    /// <returns>Populated struct</returns>
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

    /// <summary>
    /// [PORT] Convert a struct to a byte array
    /// </summary>
    /// <typeparam name="T">Struct type</typeparam>
    /// <param name="structure">Struct to convert</param>
    /// <returns>Byte array representation of struct</returns>
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

    /// <summary>
    /// [PORT] Get a byte array from a file at the specified position and length
    /// </summary>
    /// <param name="filePath">Path to the file</param>
    /// <param name="position">Position in the file to start reading</param>
    /// <param name="length">Number of bytes to read</param>
    /// <returns>Byte array containing the read data</returns>
    public static byte[] GetByteArrayFromFile(string filePath, int position, int length)
    {
        using (FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            byte[] buffer = new byte[length];
            fs.Seek(position, SeekOrigin.Begin);
            int bytesRead = fs.Read(buffer, 0, length);
            
            if (bytesRead < length)
            {
                // Resize buffer if we read less than requested
                byte[] resizedBuffer = new byte[bytesRead];
                Array.Copy(buffer, resizedBuffer, bytesRead);
                return resizedBuffer;
            }
            
            return buffer;
        }
    }

    // [PORT] Overload for reading from an already-open FileStream
    public static byte[] GetByteArrayFromFile(FileStream fs, int position, int length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        var buffer = new byte[length];
        fs.Seek(position, SeekOrigin.Begin);
        int read = fs.Read(buffer, 0, length);
        if (read != length) throw new EndOfStreamException();
        return buffer;
    }
}
