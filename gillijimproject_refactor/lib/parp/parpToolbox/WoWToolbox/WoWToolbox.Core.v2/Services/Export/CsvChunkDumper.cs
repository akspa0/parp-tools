using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using WoWToolbox.Core.v2.Services.Export;

namespace WoWToolbox.Core.v2.Services.Export
{
    /// <summary>
    /// Dumps every PM4/PD4 chunk object to a CSV file using reflection.
    ///  • If the chunk exposes an IEnumerable Entries property, each entry is exported as a row.
    ///  • Otherwise the chunk itself becomes a single-row CSV of its primitive properties.
    /// Complex nested objects are skipped; only public primitive/enum/string fields & props are emitted.
    /// </summary>
    public static class CsvChunkDumper
    {
        private static readonly Type[] _primitiveLike =
        {
            typeof(byte),typeof(sbyte),typeof(short),typeof(ushort),typeof(int),typeof(uint),typeof(long),typeof(ulong),
            typeof(float),typeof(double),typeof(bool),typeof(char),typeof(string)
        };

        public static void DumpChunk(object chunkObj, string outDir, string filePrefix, ICsvExporter exporter)
        {
            if (chunkObj == null) return;
            var type = chunkObj.GetType();
            string chunkName = type.Name.Replace("Chunk","");
            string path = Path.Combine(outDir, $"{filePrefix}_{chunkName}.csv");

            // Try to find Entries (IEnumerable) member
            var entriesMember = type.GetMembers(BindingFlags.Public|BindingFlags.Instance)
                .FirstOrDefault(m => m.Name.Equals("Entries", StringComparison.OrdinalIgnoreCase));
            IEnumerable<object>? rows = null;
            if (entriesMember != null)
            {
                object? val = GetValue(chunkObj, entriesMember);
                if (val is IEnumerable enumerable)
                {
                    rows = enumerable.Cast<object>().ToList();
                }
            }
            if (rows == null)
            {
                rows = new[]{ chunkObj };
            }

            // Filter to rows that have at least one primitive-like property
            var list = rows.Where(r => r != null).ToList();
            if (list.Count == 0) return;
            var rowType = list.First().GetType();
            var cols = rowType.GetProperties(BindingFlags.Public|BindingFlags.Instance)
                .Where(p => IsPrimitiveLike(p.PropertyType))
                .ToArray();
            if (cols.Length == 0) return;

            var sb = new System.Text.StringBuilder();
            sb.AppendLine(string.Join(",", cols.Select(c => c.Name)));
            foreach (var row in list)
            {
                sb.AppendLine(string.Join(",", cols.Select(c => Sanitize(c.GetValue(row)))));
            }
            Directory.CreateDirectory(outDir);
            File.WriteAllText(path, sb.ToString());
        }

        private static string Sanitize(object? obj)
        {
            if (obj == null) return "";
            string s = obj.ToString() ?? "";
            return s.Contains(',') ? $"\"{s}\"" : s;
        }

        private static bool IsPrimitiveLike(Type t)
        {
            if (_primitiveLike.Contains(t)) return true;
            if (t.IsEnum) return true;
            return false;
        }

        private static object? GetValue(object obj, MemberInfo member)
        {
            return member switch
            {
                PropertyInfo pi => pi.GetValue(obj),
                FieldInfo fi => fi.GetValue(obj),
                _ => null
            };
        }
    }
}
