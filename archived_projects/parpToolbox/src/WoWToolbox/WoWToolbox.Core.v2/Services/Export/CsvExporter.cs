using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace WoWToolbox.Core.v2.Services.Export
{
    public class CsvExporter : ICsvExporter
    {
        public void Export<T>(IEnumerable<T> data, string filePath)
        {
            if (data == null || !data.Any())
            {
                return;
            }

            var sb = new StringBuilder();
            var properties = typeof(T).GetProperties();

            // Header
            sb.AppendLine(string.Join(",", properties.Select(p => p.Name)));

            // Rows
            foreach (var item in data)
            {
                var values = properties.Select(p =>
                {
                    var value = p.GetValue(item, null)?.ToString() ?? "";
                    // Handle commas in values by enclosing in quotes
                    if (value.Contains(","))
                    {
                        value = $"\"{value}\"";
                    }
                    return value;
                });
                sb.AppendLine(string.Join(",", values));
            }

            File.WriteAllText(filePath, sb.ToString());
        }
    }
}
