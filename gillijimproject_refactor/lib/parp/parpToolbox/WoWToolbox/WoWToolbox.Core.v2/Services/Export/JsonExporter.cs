using System.IO;
using System.Text.Json;

namespace WoWToolbox.Core.v2.Services.Export
{
    public class JsonExporter : IJsonExporter
    {
        public void Export(object data, string filePath)
        {
            if (data == null)
            {
                return;
            }

            var options = new JsonSerializerOptions
            {
                WriteIndented = true
            };

            var jsonString = JsonSerializer.Serialize(data, options);
            File.WriteAllText(filePath, jsonString);
        }
    }
}
