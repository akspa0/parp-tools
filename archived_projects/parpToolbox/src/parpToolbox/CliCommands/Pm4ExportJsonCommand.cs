using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Threading.Tasks;
using ParpToolbox.Services.Coordinate;
using ParpToolbox.Services.PM4;
using ParpToolbox.Services.PM4.Core;

using ParpToolbox.Formats.PM4;

namespace ParpToolbox.CliCommands
{
    public class Pm4ExportJsonCommand : Command
    {
        public Pm4ExportJsonCommand() : base("pm4-export-json", "Export PM4 surface and vertex data to a JSON file for validation.")
        {
            var inputOption = new Option<FileInfo>(
                new[] { "-i", "--input" },
                "Input PM4 file path.")
            {
                IsRequired = true
            };

            var outputOption = new Option<FileInfo>(
                new[] { "-o", "--output" },
                "Output JSON file path.")
            {
                IsRequired = true
            };

            AddOption(inputOption);
            AddOption(outputOption);

            this.SetHandler(async (context) =>
            {
                var input = context.ParseResult.GetValueForOption(inputOption);
                var output = context.ParseResult.GetValueForOption(outputOption);
                if (input == null || output == null)
                {
                    System.Console.WriteLine("Error: Input and output files are required.");
                    return;
                }
                await HandleCommand(input, output);
            });
        }

        private async Task HandleCommand(FileInfo inputFile, FileInfo outputFile)
        {
            var chunkAccessService = new Pm4ChunkAccessService();
            var fieldMappingService = new Pm4FieldMappingService(chunkAccessService);
            var jsonExportService = new Pm4JsonExportService(chunkAccessService, fieldMappingService);

            var adapter = new Pm4Adapter();
            var loadOptions = new Pm4LoadOptions { CaptureRawData = true };
            Pm4Scene scene;
            if (inputFile.FullName.Contains("_00_00") || inputFile.FullName.Contains("_000"))
            {
                scene = adapter.LoadRegion(inputFile.FullName, loadOptions);
            }
            else
            {
                scene = adapter.Load(inputFile.FullName, loadOptions);
            }

            await jsonExportService.ExportToJsonAsync(scene, outputFile.FullName);

            System.Console.WriteLine($"Successfully exported PM4 data to {outputFile.FullName}");
        }
    }
}
