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
        private readonly List<MpqArchive> _archives; // priority: earlier = lower, later = higher

        public MpqArchiveSource(IEnumerable<string> mpqPaths)
        {
            _archives = new List<MpqArchive>();
            int opened = 0, failed = 0;
            
            foreach (var path in mpqPaths.Where(File.Exists))
            {
                try
                {
                    var mpq = new MpqArchive(path, FileAccess.Read);
                    _archives.Add(mpq);
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

        public bool FileExists(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            // Try backslash variant (MPQ standard)
            var normBackslash = norm.Replace('/', '\\');
            
            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                if (_archives[i].HasFile(normBackslash) || _archives[i].HasFile(norm))
                    return true;
            }
            return false;
        }

        public Stream OpenFile(string virtualPath)
        {
            var norm = Normalize(virtualPath);
            var normBackslash = norm.Replace('/', '\\');
            
            for (int i = _archives.Count - 1; i >= 0; i--)
            {
                var mpq = _archives[i];
                
                // Try backslash first (MPQ standard), then forward slash
                MpqFileStream? fileStream = null;
                try
                {
                    fileStream = mpq.OpenFile(normBackslash);
                }
                catch
                {
                    // Try forward slash if backslash fails
                }
                
                if (fileStream == null)
                {
                    try
                    {
                        fileStream = mpq.OpenFile(norm);
                    }
                    catch
                    {
                        // Continue to next archive
                    }
                }
                
                if (fileStream != null)
                {
                    try
                    {
                        // Copy to MemoryStream since MpqFileStream might not be fully compatible
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
            foreach (var arc in _archives)
            {
                arc.Dispose();
            }
            _archives.Clear();
        }
    }
}
