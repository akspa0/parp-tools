using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace WoWRollback.Core.Services.Archive
{
    public sealed class MpqArchiveSource : IArchiveSource
    {
        private sealed class MpqHandle : IDisposable
        {
            public IntPtr Handle { get; private set; }
            public string Path { get; }

            public MpqHandle(string path)
            {
                Path = path;
                if (!StormLib.SFileOpenArchive(path, 0, StormLib.SFILE_OPEN_READ_ONLY, out var h))
                {
                    var err = Marshal.GetLastWin32Error();
                    Console.WriteLine($"[warn] Failed to open MPQ: {path} (Win32={err})");
                    Handle = IntPtr.Zero;
                }
                else
                {
                    Handle = h;
                }
            }

            public void Dispose()
            {
                if (Handle != IntPtr.Zero)
                {
                    try { StormLib.SFileCloseArchive(Handle); }
                    catch { /* ignore */ }
                    Handle = IntPtr.Zero;
                }
            }
        }

        private readonly List<MpqHandle> _archives; // priority: earlier = lower, later = higher

        public MpqArchiveSource(IEnumerable<string> mpqPaths)
        {
            _archives = mpqPaths
                .Where(File.Exists)
                .Select(p => new MpqHandle(p))
                .ToList();

            var opened = _archives.Count(a => a.Handle != IntPtr.Zero);
            var failed = _archives.Count - opened;
            Console.WriteLine($"[probe] MPQ open summary: opened={opened}, failed={failed}");
        }

        private static string Normalize(string virtualPath) => PathUtils.NormalizeVirtual(virtualPath).ToLowerInvariant();

        public bool FileExists(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                var h = _archives[i].Handle;
                if (h == IntPtr.Zero) continue;
                if (StormLib.SFileHasFile(h, norm)) return true;
            }
            return false;
        }

        public Stream OpenFile(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                var h = _archives[i].Handle;
                if (h == IntPtr.Zero) continue;

                if (!StormLib.SFileHasFile(h, norm))
                    continue;

                if (!StormLib.SFileOpenFileEx(h, norm, StormLib.SFILE_OPEN_PATCHED_FILE, out var file))
                    continue;

                try
                {
                    const int Chunk = 64 * 1024;
                    var ms = new MemoryStream();
                    var unmanaged = Marshal.AllocHGlobal(Chunk);
                    try
                    {
                        while (true)
                        {
                            if (!StormLib.SFileReadFile(file, unmanaged, (uint)Chunk, out var read, IntPtr.Zero))
                            {
                                var err = Marshal.GetLastWin32Error();
                                // ERROR_HANDLE_EOF = 38 is normal end-of-file for SFileReadFile
                                if (err == 38 || err == 0)
                                    break;
                                throw new IOException($"Failed to read file from MPQ: {norm} (Win32={err})");
                            }
                            if (read == 0)
                                break;
                            var managed = new byte[read];
                            Marshal.Copy(unmanaged, managed, 0, (int)read);
                            ms.Write(managed, 0, (int)read);
                        }
                        ms.Position = 0;
                        return ms;
                    }
                    finally
                    {
                        Marshal.FreeHGlobal(unmanaged);
                        StormLib.SFileCloseFile(file);
                    }
                }
                catch
                {
                    try { StormLib.SFileCloseFile(file); } catch { }
                    throw;
                }
            }

            throw new FileNotFoundException($"File not found in MPQs: {virtualPath}");
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            var results = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var mask = string.IsNullOrWhiteSpace(pattern) ? "*" : PathUtils.NormalizeVirtual(pattern).ToLowerInvariant();

            foreach (var arc in _archives)
            {
                if (arc.Handle == IntPtr.Zero) continue;
                var find = StormLib.SFileFindFirstFile(arc.Handle, mask, out var data, null);
                if (find == IntPtr.Zero) continue;

                try
                {
                    do
                    {
                        var name = PathUtils.NormalizeVirtual(data.cFileName).ToLowerInvariant();
                        results.Add(name);
                    } while (StormLib.SFileFindNextFile(find, out data));
                }
                finally
                {
                    StormLib.SFileFindClose(find);
                }
            }

            return results;
        }

        public void Dispose()
        {
            foreach (var arc in _archives)
            {
                arc.Dispose();
            }
            _archives.Clear();
        }
    }
}
