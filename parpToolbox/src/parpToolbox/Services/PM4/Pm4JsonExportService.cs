using ParpToolbox.Services.PM4.Core;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4
{
    /// <summary>
    /// Service for exporting PM4 data to JSON for validation purposes.
    /// </summary>
    public class Pm4JsonExportService
    {
        private readonly IPm4ChunkAccessService _chunkAccessService;
        private readonly IPm4FieldMappingService _fieldMappingService;

        public Pm4JsonExportService(IPm4ChunkAccessService chunkAccessService, IPm4FieldMappingService fieldMappingService)
        {
            _chunkAccessService = chunkAccessService;
            _fieldMappingService = fieldMappingService;
        }

        /// <summary>
        /// Exports the surface and vertex data from a PM4 scene to a JSON file.
        /// </summary>
        /// <param name="scene">The loaded PM4 scene.</param>
        /// <param name="outputJsonPath">The path to write the output JSON file.</param>
        public async Task ExportToJsonAsync(Pm4Scene scene, string outputJsonPath)
        {
            var objectGroups = _fieldMappingService.BuildObjectGroups(scene);
            var vertices = _chunkAccessService.GetMsvtVertices(scene);

            var exportData = new
            {
                ObjectGroups = objectGroups,
                Vertices = vertices.Select(v => new { v.X, v.Y, v.Z })
            };

            var options = new JsonSerializerOptions { WriteIndented = true };
            var jsonString = JsonSerializer.Serialize(exportData, options);

            await File.WriteAllTextAsync(outputJsonPath, jsonString);
        }
    }
}
