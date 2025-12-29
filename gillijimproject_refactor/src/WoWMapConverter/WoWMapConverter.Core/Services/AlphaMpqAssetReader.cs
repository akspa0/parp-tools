using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Reads assets from Alpha 0.5.3 per-asset MPQ archives.
/// Alpha stores WMO, WDT, and WDL files in individual MPQ archives without listfiles.
/// Each archive contains 2 files: the data at index 0 and an MD5 checksum at index 1.
/// </summary>
public static class AlphaMpqAssetReader
{
    #region StormLib P/Invoke
    
    private const string STORMLIB = "StormLib.dll";
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true, CharSet = CharSet.Auto)]
    private static extern bool SFileOpenArchive(
        [MarshalAs(UnmanagedType.LPTStr)] string szMpqName,
        uint dwPriority,
        uint dwFlags,
        out IntPtr phMpq);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseArchive(IntPtr hMpq);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern IntPtr SFileFindFirstFile(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szMask,
        out SFILE_FIND_DATA lpFindFileData,
        [MarshalAs(UnmanagedType.LPStr)] string szListFile);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileFindNextFile(IntPtr hFind, ref SFILE_FIND_DATA lpFindFileData);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileFindClose(IntPtr hFind);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileOpenFileEx(
        IntPtr hMpq,
        [MarshalAs(UnmanagedType.LPStr)] string szFileName,
        uint dwSearchScope,
        out IntPtr phFile);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern uint SFileGetFileSize(IntPtr hFile, out uint fileSizeHigh);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileReadFile(
        IntPtr hFile,
        IntPtr lpBuffer,
        uint dwToRead,
        out uint pdwRead,
        IntPtr lpOverlapped);
    
    [DllImport(STORMLIB, CallingConvention = CallingConvention.Winapi, SetLastError = true)]
    private static extern bool SFileCloseFile(IntPtr hFile);
    
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    private struct SFILE_FIND_DATA
    {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 260)]
        public string cFileName;
        public IntPtr szPlainName;
        public uint dwHashIndex;
        public uint dwBlockIndex;
        public uint dwFileSize;
        public uint dwFileFlags;
        public uint dwCompSize;
        public uint dwFileTimeLo;
        public uint dwFileTimeHi;
        public uint lcLocale;
    }
    
    // Archive open flags
    private const uint SFILE_OPEN_HARD_DISK_FILE = 2;
    private const uint MPQ_OPEN_READ_ONLY = 0x00000100;
    
    #endregion
    
    /// <summary>
    /// Read asset data from a per-asset MPQ archive (e.g., castle01.wmo.MPQ).
    /// Returns the largest file in the archive (the actual data, not the MD5).
    /// </summary>
    public static byte[]? ReadFromMpq(string mpqPath)
    {
        if (!File.Exists(mpqPath))
            return null;
        
        if (!SFileOpenArchive(mpqPath, SFILE_OPEN_HARD_DISK_FILE, MPQ_OPEN_READ_ONLY, out var hMpq))
        {
            var errorCode = Marshal.GetLastWin32Error();
            Console.WriteLine($"[WARN] Failed to open MPQ: {mpqPath} (error: {errorCode})");
            return null;
        }
        
        try
        {
            // Enumerate files and find the largest one (the data, not the MD5)
            var files = EnumerateFiles(hMpq);
            
            string? dataFile = null;
            uint maxSize = 0;
            
            foreach (var (name, size) in files)
            {
                if (size > maxSize)
                {
                    maxSize = size;
                    dataFile = name;
                }
            }
            
            if (dataFile == null)
            {
                Console.WriteLine($"[WARN] No files found in MPQ: {mpqPath}");
                return null;
            }
            
            return ReadFile(hMpq, dataFile);
        }
        finally
        {
            SFileCloseArchive(hMpq);
        }
    }
    
    /// <summary>
    /// Try to read a file, falling back to MPQ version if not found.
    /// Works for WDT, WMO, and WDL files.
    /// If the path is already an MPQ archive, extracts from it.
    /// </summary>
    public static byte[]? ReadWithMpqFallback(string filePath)
    {
        // If path is already an MPQ archive, extract from it
        if (filePath.EndsWith(".mpq", StringComparison.OrdinalIgnoreCase))
            return ReadFromMpq(filePath);
        
        // Try direct file first (non-MPQ)
        if (File.Exists(filePath))
            return File.ReadAllBytes(filePath);
        
        // Try MPQ version (e.g., castle01.wmo -> castle01.wmo.MPQ)
        var mpqPath = filePath + ".MPQ";
        if (File.Exists(mpqPath))
            return ReadFromMpq(mpqPath);
        
        // Try lowercase mpq extension
        mpqPath = filePath + ".mpq";
        if (File.Exists(mpqPath))
            return ReadFromMpq(mpqPath);
        
        return null;
    }
    
    /// <summary>
    /// Check if a file exists (either directly or as an MPQ).
    /// </summary>
    public static bool Exists(string filePath)
    {
        return File.Exists(filePath) || 
               File.Exists(filePath + ".MPQ") || 
               File.Exists(filePath + ".mpq");
    }
    
    /// <summary>
    /// Get the actual path (direct file or MPQ version).
    /// </summary>
    public static string? GetActualPath(string filePath)
    {
        if (File.Exists(filePath))
            return filePath;
        if (File.Exists(filePath + ".MPQ"))
            return filePath + ".MPQ";
        if (File.Exists(filePath + ".mpq"))
            return filePath + ".mpq";
        return null;
    }
    
    private static List<(string name, uint size)> EnumerateFiles(IntPtr hMpq)
    {
        var files = new List<(string, uint)>();
        
        var hFind = SFileFindFirstFile(hMpq, "*", out var findData, null);
        if (hFind == IntPtr.Zero || hFind == new IntPtr(-1))
            return files;
        
        try
        {
            do
            {
                if (!string.IsNullOrEmpty(findData.cFileName))
                {
                    files.Add((findData.cFileName, findData.dwFileSize));
                }
            }
            while (SFileFindNextFile(hFind, ref findData));
        }
        finally
        {
            SFileFindClose(hFind);
        }
        
        return files;
    }
    
    private static byte[]? ReadFile(IntPtr hMpq, string fileName)
    {
        if (!SFileOpenFileEx(hMpq, fileName, 0, out var hFile))
            return null;
        
        try
        {
            uint sizeHigh = 0;
            var size = SFileGetFileSize(hFile, out sizeHigh);
            if (size == 0 || size == uint.MaxValue)
                return null;
            
            var buffer = new byte[size];
            var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
            
            try
            {
                if (!SFileReadFile(hFile, handle.AddrOfPinnedObject(), size, out var bytesRead, IntPtr.Zero))
                    return null;
                
                if (bytesRead != size)
                    Array.Resize(ref buffer, (int)bytesRead);
                
                return buffer;
            }
            finally
            {
                handle.Free();
            }
        }
        finally
        {
            SFileCloseFile(hFile);
        }
    }
}
