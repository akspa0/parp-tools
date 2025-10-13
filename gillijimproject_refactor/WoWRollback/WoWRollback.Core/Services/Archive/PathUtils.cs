using System;
using System.IO;

namespace WoWRollback.Core.Services.Archive
{
    internal static class PathUtils
    {
        public static string NormalizeVirtual(string virtualPath)
        {
            if (string.IsNullOrWhiteSpace(virtualPath)) return string.Empty;
            var p = virtualPath.Replace('\\', '/').Trim();
            if (p.StartsWith('/')) p = p[1..];
            return p;
        }

        public static string CombineVirtual(params string[] parts)
        {
            var joined = string.Join('/', parts).Replace('\\', '/');
            return NormalizeVirtual(joined);
        }

        public static string ToOsPath(string root, string virtualPath)
        {
            var norm = NormalizeVirtual(virtualPath);
            var combined = Path.Combine(root, norm.Replace('/', Path.DirectorySeparatorChar));
            return combined;
        }
    }
}
