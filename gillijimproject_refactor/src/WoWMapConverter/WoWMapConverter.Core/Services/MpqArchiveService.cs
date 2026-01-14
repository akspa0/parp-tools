using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace WoWMapConverter.Core.Services;

/// <summary>
/// Service to manage global MPQ archives (Textures.mpq, Terrain.mpq, etc.)
/// and provide unified file access.
/// </summary>
public class MpqArchiveService : IDisposable
{
    private readonly List<IntPtr> _archives = new();
    private bool _disposed;

    #region StormLib P/Invoke (Copied from AlphaMpqAssetReader)
    
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
    
    // Archive open flags
    private const uint SFILE_OPEN_HARD_DISK_FILE = 2;
    private const uint MPQ_OPEN_READ_ONLY = 0x00000100;
    
    #endregion

    public void LoadArchives(IEnumerable<string> searchPaths)
    {
        foreach (var path in searchPaths)
        {
            if (Directory.Exists(path))
            {
                var mpqFiles = Directory.GetFiles(path, "*.mpq");
                foreach (var mpq in mpqFiles)
                {
                    OpenArchive(mpq);
                }
                
                // Also check uppercase
                var mpqFilesUpper = Directory.GetFiles(path, "*.MPQ");
                foreach (var mpq in mpqFilesUpper)
                {
                    OpenArchive(mpq);
                }
            }
        }
        Console.WriteLine($"Initialized MpqArchiveService with {_archives.Count} archives.");
    }

    private void OpenArchive(string path)
    {
        // Avoid duplicate loading
        // (Primitive check, stormlib handles multiples fine but let's be safe)
        
        if (SFileOpenArchive(path, SFILE_OPEN_HARD_DISK_FILE, MPQ_OPEN_READ_ONLY, out var hMpq))
        {
            _archives.Add(hMpq);
            Console.WriteLine($"Opened archive: {Path.GetFileName(path)}");
        }
        else
        {
            // var err = Marshal.GetLastWin32Error();
            // Console.WriteLine($"Failed to open archive {path}: {err}");
        }
    }

    public bool FileExists(string virtualPath)
    {
        // SFileHasFile is not exposed in our P/Invoke set, 
        // so we try to open it.
        
        foreach (var hMpq in _archives)
        {
            if (SFileOpenFileEx(hMpq, virtualPath, 0, out var hFile))
            {
                SFileCloseFile(hFile);
                return true;
            }
        }
        return false;
    }

    public byte[]? ReadFile(string virtualPath)
    {
        foreach (var hMpq in _archives)
        {
            if (SFileOpenFileEx(hMpq, virtualPath, 0, out var hFile))
            {
                try
                {
                    uint sizeHigh = 0;
                    var size = SFileGetFileSize(hFile, out sizeHigh);
                    if (size == 0 || size == uint.MaxValue) continue;

                    var buffer = new byte[size];
                    var handle = GCHandle.Alloc(buffer, GCHandleType.Pinned);
                    try
                    {
                        if (SFileReadFile(hFile, handle.AddrOfPinnedObject(), size, out var bytesRead, IntPtr.Zero))
                        {
                            if (bytesRead != size) Array.Resize(ref buffer, (int)bytesRead);
                            return buffer;
                        }
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
        return null;
    }

    public void Dispose()
    {
        if (_disposed) return;
        foreach (var hMpq in _archives)
        {
            SFileCloseArchive(hMpq);
        }
        _archives.Clear();
        _disposed = true;
    }
}
