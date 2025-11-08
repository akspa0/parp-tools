using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MPQToTACT.MPQ;
using MPQToTACT.MPQ.Native;

namespace WoWRollback.Core.Services.Archive
{
    public sealed class MpqArchiveSource : IArchiveSource
    {
        private readonly List<(MpqArchive Arc, string Path, bool IsLocale, bool IsPatch)> _archives;

        public MpqArchiveSource(IEnumerable<string> mpqPaths)
        {
            _archives = new List<(MpqArchive, string, bool, bool)>();
            int opened = 0, failed = 0;
            
            foreach (var path in mpqPaths.Where(File.Exists))
            {
                try
                {
                    var mpq = new MpqArchive(path, FileAccess.Read);
                    var isLocale = IsLocalePath(path);
                    var file = System.IO.Path.GetFileName(path);
                    var isPatch = file.StartsWith("patch", StringComparison.OrdinalIgnoreCase);
                    _archives.Add((mpq, path, isLocale, isPatch));
                    opened++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[warn] Failed to open MPQ: {path} ({ex.Message})");
                    failed++;
                }
            }

            Console.WriteLine($"[probe] MPQ open summary: opened={opened}, failed={failed}");
        }

        private static string Normalize(string virtualPath) => PathUtils.NormalizeVirtual(virtualPath);
        private static bool IsLocalePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path)) return false;
            var p = path.Replace('\\', '/');
            var idx = p.IndexOf("/Data/", StringComparison.OrdinalIgnoreCase);
            if (idx < 0) return false;
            var rest = p.Substring(idx + 6);
            var slash = rest.IndexOf('/') >= 0 ? rest.IndexOf('/') : rest.Length;
            if (slash <= 0) return false;
            var seg = rest.Substring(0, slash);
            if (seg.Length != 4) return false;
            bool letters = char.IsLetter(seg[0]) && char.IsLetter(seg[1]) && char.IsLetter(seg[2]) && char.IsLetter(seg[3]);
            return letters;
        }

        public bool FileExists(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            // Try backslash variant (MPQ standard)
            var normBackslash = norm.Replace('/', '\\');
            
            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                var arc = _archives[i].Arc;
                if (arc.HasFile(normBackslash) || arc.HasFile(norm))
                    return true;
            }
            return false;
        }

        public Stream OpenFile(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            var normBackslash = norm.Replace('/', '\\');
            
            var isDbc = norm.StartsWith("DBFilesClient/", StringComparison.OrdinalIgnoreCase);
            if (isDbc)
            {
                // Prefer locale patches first (letter > numeric due to reverse scan), then root patches
                // Pass 1: locale patches
                for (int i = _archives.Count - 1; i >= 0; i--)
                {
                    var meta = _archives[i];
                    if (!(meta.IsPatch && meta.IsLocale)) continue;
                    var mpq = meta.Arc;
                    MpqFileStream? fileStream = null;
                    try { fileStream = mpq.OpenFile(normBackslash); } catch { }
                    if (fileStream == null) { try { fileStream = mpq.OpenFile(norm); } catch { } }
                    if (fileStream != null)
                    {
                        try
                        {
                            var ms = new MemoryStream();
                            fileStream.CopyTo(ms);
                            fileStream.Dispose();
                            ms.Position = 0;
                            return ms;
                        }
                        catch { fileStream?.Dispose(); }
                    }
                }
                // Pass 2: root patches
                for (int i = _archives.Count - 1; i >= 0; i--)
                {
                    var meta = _archives[i];
                    if (!(meta.IsPatch && !meta.IsLocale)) continue;
                    var mpq = meta.Arc;
                    MpqFileStream? fileStream = null;
                    try { fileStream = mpq.OpenFile(normBackslash); } catch { }
                    if (fileStream == null) { try { fileStream = mpq.OpenFile(norm); } catch { } }
                    if (fileStream != null)
                    {
                        try
                        {
                            var ms = new MemoryStream();
                            fileStream.CopyTo(ms);
                            fileStream.Dispose();
                            ms.Position = 0;
                            return ms;
                        }
                        catch { fileStream?.Dispose(); }
                    }
                }
            }

            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                var mpq = _archives[i].Arc;

                MpqFileStream? fileStream = null;
                try
                {
                    fileStream = mpq.OpenFile(normBackslash);
                }
                catch
                {
                }
                
                if (fileStream == null)
                {
                    try
                    {
                        fileStream = mpq.OpenFile(norm);
                    }
                    catch
                    {
                    }
                }
                
                if (fileStream != null)
                {
                    try
                    {
                        var ms = new MemoryStream();
                        fileStream.CopyTo(ms);
                        fileStream.Dispose();
                        ms.Position = 0;
                        return ms;
                    }
                    catch
                    {
                        fileStream?.Dispose();
                    }
                }
            }

            throw new FileNotFoundException($"File not found in MPQs: {virtualPath}");
        }

        public IEnumerable<string> EnumerateFiles(string pattern = "*")
        {
            // Note: Enumeration not fully implemented with working wrapper
            // For now, return empty - not needed for minimap resolution which uses md5translate
            return Enumerable.Empty<string>();
        }

        public void Dispose()
        {
            foreach (var meta in _archives)
            {
                meta.Arc.Dispose();
            }
            _archives.Clear();
        }
    }
}
