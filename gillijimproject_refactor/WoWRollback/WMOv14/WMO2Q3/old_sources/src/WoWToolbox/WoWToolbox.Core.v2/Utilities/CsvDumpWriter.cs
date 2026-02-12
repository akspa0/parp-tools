using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;


namespace WoWToolbox.Core.v2.Utilities
{
    /// <summary>
    /// Simple helper that writes arbitrary records to a CSV with CsvHelper default configuration.
    /// It detects the output directory and ensures it exists.
    /// </summary>
    public static class CsvDumpWriter
    {
        public static void WriteDump<T>(IEnumerable<T> records, string outputPath)
        {
            var dir = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            using var writer = new StreamWriter(outputPath);

            bool headerWritten = false;
            string[] headers = System.Array.Empty<string>();

            foreach (var rec in records)
            {
                if (rec == null) continue;
                var type = rec.GetType();
                var props = type.GetProperties(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);

                // build header once using first record's property order
                if (!headerWritten)
                {
                    headers = props.Select(p => p.Name).ToArray();
                    writer.WriteLine(string.Join(',', headers));
                    headerWritten = true;
                }

                string CsvEscape(string? val)
                {
                    val ??= string.Empty;
                    if (val.Contains('"')) val = val.Replace("\"", "\"\"");
                    if (val.Contains(',') || val.Contains('\n') || val.Contains('\r')) val = $"\"{val}\"";
                    return val;
                }

                var values = headers.Select(h =>
                {
                    var p = props.FirstOrDefault(pp => pp.Name == h);
                    object? v = p?.GetValue(rec, null);
                    return CsvEscape(v?.ToString());
                });
                writer.WriteLine(string.Join(',', values));
            }
        }
    }
}
