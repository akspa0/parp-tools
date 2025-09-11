using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using DBCD;

namespace DBCTool
{
    internal static class CsvExporter
    {
        public static void WriteCsv(IDBCDStorage storage, string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            var columns = storage.AvailableColumns;
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            using var sw = new StreamWriter(fs, new UTF8Encoding(encoderShouldEmitUTF8Identifier: true));

            // Header
            sw.WriteLine(string.Join(',', columns.Select(EscapeCsv)));

            // Stable iteration by ascending ID
            foreach (var id in storage.Keys.OrderBy(k => k))
            {
                var row = storage[id];
                var values = new string[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    var col = columns[i];
                    object? val = null;
                    try
                    {
                        val = row.Field<object>(col);
                    }
                    catch
                    {
                        val = null;
                    }

                    values[i] = EscapeCsv(FormatValue(val));
                }
                sw.WriteLine(string.Join(',', values));
            }
        }

        private static string FormatValue(object? value)
        {
            if (value == null)
                return string.Empty;

            switch (value)
            {
                case string s:
                    return s;
                case Array arr:
                    {
                        var parts = new List<string>(arr.Length);
                        foreach (var item in arr)
                        {
                            parts.Add(ConvertToString(item));
                        }
                        // Join arrays with '|'
                        return string.Join('|', parts);
                    }
                default:
                    return ConvertToString(value);
            }
        }

        private static string ConvertToString(object? o)
        {
            if (o == null) return string.Empty;
            return o switch
            {
                IFormattable f => f.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty,
                _ => o.ToString() ?? string.Empty
            };
        }

        private static string EscapeCsv(string field)
        {
            if (field.IndexOfAny(new[] { '"', ',', '\n', '\r' }) >= 0)
            {
                return '"' + field.Replace("\"", "\"\"") + '"';
            }
            return field;
        }
    }
}
