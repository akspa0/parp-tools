using System.CommandLine;
using MdxLTool.Formats.Mdx;
using MdxLTool.Services;

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
        var gamePathOption = new Option<DirectoryInfo>("--game-path", "Path to WoW game directory for texture extraction");

        var convertCommand = new Command("convert", "Convert between model formats")
        {
            inputArg,
            outputArg,
            targetOption,
            gamePathOption
        };
        convertCommand.SetHandler(ConvertHandler, inputArg, outputArg, targetOption, gamePathOption);

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

    static void ConvertHandler(FileInfo input, FileInfo output, string target, DirectoryInfo? gamePath)
    {
        if (!input.Exists)
        {
            Console.Error.WriteLine($"Error: Input file not found: {input.FullName}");
            return;
        }

        Console.WriteLine($"Converting: {input.Name} -> {output.Name}");
        if (gamePath != null) Console.WriteLine($"Game path: {gamePath.FullName}");

        var inputExt = input.Extension.ToLowerInvariant();
        var outputExt = output.Extension.ToLowerInvariant();

        try
        {
            if (inputExt == ".mdx")
            {
                var mdx = MdxFile.Load(input.FullName);
                Console.WriteLine($"Loaded MDX: Version {mdx.Version}, {mdx.Geosets.Count} geosets");

                // Texture Export Phase
                ExportTextures(mdx, input.DirectoryName ?? ".", output.DirectoryName ?? ".", gamePath);

                if (outputExt == ".mdl")
                {
                    // Ensure output directory exists
                    if (output.Directory != null && !output.Directory.Exists)
                    {
                        output.Directory.Create();
                    }
                    
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
            Console.Error.WriteLine(ex.StackTrace);
        }
    }

    static void ExportTextures(MdxFile mdx, string modelDir, string outputDir, DirectoryInfo? gamePath)
    {
        using var mpqService = new NativeMpqService();
        if (gamePath != null && gamePath.Exists)
        {
            Console.WriteLine("Loading game archives...");
            mpqService.LoadArchives(new[] { gamePath.FullName });
        }

        var textureService = new TextureService(mpqService);
        int exported = 0;

        foreach (var tex in mdx.Textures)
        {
            if (string.IsNullOrEmpty(tex.Path) && tex.ReplaceableId == 0) continue;

            // For replaceable IDs, we might want to map them to default names later,
            // but for now let's focus on explicit paths.
            if (!string.IsNullOrEmpty(tex.Path))
            {
                var relativePng = textureService.ExportTexture(tex.Path, modelDir, outputDir);
                if (relativePng != null)
                {
                    Console.WriteLine($"  Exported texture: {tex.Path} -> {relativePng}");
                    tex.Path = relativePng; // Update path for MDL reference
                    exported++;
                }
                else
                {
                    Console.WriteLine($"  [WARN] Could not resolve texture: {tex.Path}");
                }
            }
        }

        if (exported > 0) Console.WriteLine($"Total textures exported: {exported}");
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
