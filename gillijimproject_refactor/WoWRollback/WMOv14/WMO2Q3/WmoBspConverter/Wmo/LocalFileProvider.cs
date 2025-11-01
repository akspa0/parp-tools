using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using WoWFormatLib;

namespace WmoBspConverter.Wmo
{
    public class LocalFileProvider : IFileProvider
    {
        private readonly string _baseDirectory;
        private readonly Dictionary<uint, string> _fileDataIdMap = new();

        public LocalFileProvider(string baseDirectory)
        {
            _baseDirectory = baseDirectory;
            
            if (!Directory.Exists(_baseDirectory))
            {
                throw new DirectoryNotFoundException($"Directory not found: {_baseDirectory}");
            }
        }

        public void SetBuild(string build)
        {
            // Local files don't need build versioning
            // This is a no-op for local file provider
        }

        public bool FileExists(string filename)
        {
            var fullPath = Path.Combine(_baseDirectory, filename);
            return File.Exists(fullPath);
        }

        public Stream OpenFile(string filename)
        {
            var fullPath = Path.Combine(_baseDirectory, filename);
            if (!File.Exists(fullPath))
            {
                throw new FileNotFoundException($"File not found: {fullPath}");
            }
            return File.OpenRead(fullPath);
        }

        public bool FileExists(uint filedataid)
        {
            return _fileDataIdMap.ContainsKey(filedataid);
        }

        public Stream OpenFile(uint filedataid)
        {
            if (!_fileDataIdMap.TryGetValue(filedataid, out var filename))
            {
                throw new FileNotFoundException($"File not found by FiledataID: {filedataid}");
            }
            return OpenFile(filename);
        }

        public bool FileExists(byte[] cKey)
        {
            // For local file provider, we don't use CKey lookup
            // This would require file metadata indexing
            return false;
        }

        public Stream OpenFile(byte[] cKey)
        {
            // For local file provider, we don't use CKey lookup
            throw new NotImplementedException("CKey lookup not supported in LocalFileProvider");
        }

        public uint GetFileDataIdByName(string filename)
        {
            // Generate a simple hash-based FiledataID for local files
            using (var md5 = MD5.Create())
            {
                var hash = md5.ComputeHash(System.Text.Encoding.UTF8.GetBytes(filename));
                return BitConverter.ToUInt32(hash, 0);
            }
        }
    }
}