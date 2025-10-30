using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.Core.Services.Assets
{
    public sealed class AssetGate
    {
        private readonly ListfileIndex _target;
        public AssetGate(ListfileIndex target) { _target = target ?? throw new ArgumentNullException(nameof(target)); }

        public IReadOnlyList<string> FilterNames(IEnumerable<string> names, out IReadOnlyList<string> dropped)
        {
            var keep = new List<string>();
            var drop = new List<string>();
            foreach (var n in names)
            {
                if (string.IsNullOrWhiteSpace(n)) continue;
                var fwd = n.Replace('\\', '/');
                if (_target.ContainsPath(fwd)) keep.Add(n);
                else drop.Add(n);
            }
            dropped = drop;
            return keep;
        }

        public static void WriteDropReport(string outCsvPath, IReadOnlyList<string> droppedM2, IReadOnlyList<string> droppedWmo)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(outCsvPath) ?? ".");
            using var sw = new StreamWriter(outCsvPath);
            sw.WriteLine("type,path");
            foreach (var p in droppedM2) sw.WriteLine($"m2,{Escape(p)}");
            foreach (var p in droppedWmo) sw.WriteLine($"wmo,{Escape(p)}");
        }

        private static string Escape(string s)
        {
            if (s.Contains(',')) return '"' + s.Replace("\"", "\"\"") + '"';
            return s;
        }
    }
}
