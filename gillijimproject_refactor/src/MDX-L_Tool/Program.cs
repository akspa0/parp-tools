using System.CommandLine;
using MdxLTool.Formats.Mdx;

namespace MdxLTool;

/// <summary>
/// MDX-L_Tool: Digital Archaeology Model Converter
/// Converts between WC3 MDL/MDX and WoW M2 formats.
/// </summary>
class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("MDX-L_Tool - Digital Archaeology Model Converter");

        // convert command
        var inputArg = new Argument<FileInfo>("input", "Input model file (.mdx, .mdl, .m2)");
        var outputArg = new Argument<FileInfo>("output", "Output model file");
        var targetOption = new Option<string>("--target", () => "auto", "Target format: auto, mdl, mdx, wotlk");

        var convertCommand = new Command("convert", "Convert between model formats")
        {
            inputArg,
            outputArg,
            targetOption
        };
        convertCommand.SetHandler(ConvertHandler, inputArg, outputArg, targetOption);

        // info command
        var infoInputArg = new Argument<FileInfo>("input", "Model file to inspect");
        var infoCommand = new Command("info", "Display information about a model file")
        {
            infoInputArg
        };
        infoCommand.SetHandler(InfoHandler, infoInputArg);

        rootCommand.AddCommand(convertCommand);
        rootCommand.AddCommand(infoCommand);

        return await rootCommand.InvokeAsync(args);
    }

    static void ConvertHandler(FileInfo input, FileInfo output, string target)
    {
        if (!input.Exists)
        {
            Console.Error.WriteLine($"Error: Input file not found: {input.FullName}");
            return;
        }

        Console.WriteLine($"Converting: {input.Name} -> {output.Name}");
        Console.WriteLine($"Target format: {target}");

        var inputExt = input.Extension.ToLowerInvariant();
        var outputExt = output.Extension.ToLowerInvariant();

        try
        {
            if (inputExt == ".mdx")
            {
                var mdx = MdxFile.Load(input.FullName);
                Console.WriteLine($"Loaded MDX: Version {mdx.Version}, {mdx.Geosets.Count} geosets");

                if (outputExt == ".mdl")
                {
                    mdx.SaveMdl(output.FullName);
                    Console.WriteLine($"Saved MDL: {output.FullName}");
                }
                else if (outputExt == ".m2")
                {
                    Console.WriteLine("M2 conversion not yet implemented (Phase 2)");
                }
            }
            else if (inputExt == ".mdl")
            {
                Console.WriteLine("MDL reading not yet implemented");
            }
            else if (inputExt == ".m2")
            {
                Console.WriteLine("M2 reading not yet implemented (Phase 2)");
            }
            else
            {
                Console.Error.WriteLine($"Unsupported input format: {inputExt}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }

    static void InfoHandler(FileInfo input)
    {
        if (!input.Exists)
        {
            Console.Error.WriteLine($"Error: File not found: {input.FullName}");
            return;
        }

        Console.WriteLine($"File: {input.Name}");
        Console.WriteLine($"Size: {input.Length:N0} bytes");

        var ext = input.Extension.ToLowerInvariant();
        try
        {
            if (ext == ".mdx")
            {
                var mdx = MdxFile.Load(input.FullName);
                Console.WriteLine($"Format: MDX (Warcraft III Binary)");
                Console.WriteLine($"Version: {mdx.Version}");
                Console.WriteLine($"Model: {mdx.ModelName}");
                Console.WriteLine($"Geosets: {mdx.Geosets.Count}");
                Console.WriteLine($"Bones: {mdx.Bones.Count}");
                Console.WriteLine($"Sequences: {mdx.Sequences.Count}");
                Console.WriteLine($"Materials: {mdx.Materials.Count}");
                Console.WriteLine($"Textures: {mdx.Textures.Count}");
            }
            else
            {
                Console.WriteLine($"Format inspection for {ext} not yet implemented");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error parsing file: {ex.Message}");
        }
    }
}
