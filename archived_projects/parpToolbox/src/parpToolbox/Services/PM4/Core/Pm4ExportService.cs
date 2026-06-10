using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using parpToolbox.Models.PM4.Export;
using ParpToolbox.Formats.PM4;

namespace ParpToolbox.Services.PM4.Core
{
    /// <summary>
    /// A unified service for exporting PM4 data.
    /// </summary>
    public class Pm4ExportService : IPm4ExportService
    {
        private readonly IPm4ChunkAccessService _chunkAccessService;
        private readonly IPm4FieldMappingService _fieldMappingService;

        public Pm4ExportService(IPm4ChunkAccessService chunkAccessService, IPm4FieldMappingService fieldMappingService)
        {
            _chunkAccessService = chunkAccessService;
            _fieldMappingService = fieldMappingService;
        }

        public async Task ExportAsync(Pm4Scene scene, Pm4ExportOptions options)
        {
            switch (options.Strategy)
            {
                case ExportStrategy.RawSurfaces:
                    await ExportRawSurfacesAsync(scene, options.OutputPath);
                    break;
                case ExportStrategy.PerSurfaceGroup:
                    await ExportPerSurfaceGroupAsync(scene, options.OutputPath);
                    break;
                // Other strategies will be implemented here.
                default:
                    throw new System.NotImplementedException($"Export strategy {options.Strategy} is not yet implemented.");
            }
        }

        private async Task ExportRawSurfacesAsync(Pm4Scene scene, string outputPath)
        {
            var vertices = _chunkAccessService.GetMsvtVertices(scene).ToList();
            var surfaces = _chunkAccessService.GetMsurChunks(scene).ToList();

            var sb = new StringBuilder();

            // Write all vertices
            foreach (var vertex in vertices)
            {
                sb.AppendLine($"v {vertex.X:F6} {vertex.Y:F6} {vertex.Z:F6}");
            }

            // Write all faces from all surfaces
            sb.AppendLine("o RawSurfaceExport");
            foreach (var surface in surfaces)
            {
                for (int i = 0; i < surface.IndexCount; i += 3)
                {
                    // OBJ indices are 1-based
                    var i1 = surface.MsviFirstIndex + i + 1;
                    var i2 = surface.MsviFirstIndex + i + 2;
                    var i3 = surface.MsviFirstIndex + i + 3;
                    sb.AppendLine($"f {i1} {i2} {i3}");
                }
            }

            await File.WriteAllTextAsync(outputPath, sb.ToString());
        }

        private async Task ExportPerSurfaceGroupAsync(Pm4Scene scene, string outputPath)
        {
            // For per-surface group export, we'll use the existing Pm4SurfaceGroupExporter
            // but adapt it to work with our current scene and output path
            var outputDir = Path.GetDirectoryName(outputPath) ?? ".";
            Pm4SurfaceGroupExporter.ExportSurfaceGroupsFromScene(scene, outputDir);
            
            // Since ExportSurfaceGroupsFromScene doesn't return a task, we don't need to await
            await Task.CompletedTask;
        }
    }
}
