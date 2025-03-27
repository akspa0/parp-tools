using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using WowToolSuite.Liquid.Parsers;
using WowToolSuite.Liquid.Converters;

namespace WowToolSuite.Liquid
{
    class Program
    {
        static int Main(string[] args)
        {
            var rootCommand = new RootCommand("World of Warcraft Liquid File Converter");

            var inputOption = new Option<string>(
                "--input",
                "Input directory containing WLW/WLM files"
            )
            { IsRequired = true };

            var outputOption = new Option<string>(
                "--output",
                "Output directory for OBJ files"
            )
            { IsRequired = true };

            var verboseOption = new Option<bool>(
                "--verbose",
                () => false,
                "Enable verbose output"
            );

            rootCommand.AddOption(inputOption);
            rootCommand.AddOption(outputOption);
            rootCommand.AddOption(verboseOption);

            rootCommand.SetHandler((string input, string output, bool verbose) =>
            {
                try
                {
                    ConvertFiles(input, output, verbose);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"Error: {ex.Message}");
                    if (verbose)
                    {
                        Console.Error.WriteLine(ex.StackTrace);
                    }
                    Environment.Exit(1);
                }
            }, inputOption, outputOption, verboseOption);

            return rootCommand.Invoke(args);
        }

        private static void ConvertFiles(string inputDirectory, string outputDirectory, bool verbose)
        {
            if (!Directory.Exists(inputDirectory))
            {
                throw new DirectoryNotFoundException($"Input directory not found: {inputDirectory}");
            }

            Directory.CreateDirectory(outputDirectory);

            var wlwFiles = Directory.GetFiles(inputDirectory, "*.wlw", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(inputDirectory, "*.wlm", SearchOption.AllDirectories))
                .ToList();

            if (!wlwFiles.Any())
            {
                Console.WriteLine("No WLW/WLM files found in the input directory.");
                return;
            }

            foreach (var file in wlwFiles)
            {
                if (verbose)
                {
                    Console.WriteLine($"Processing: {file}");
                }

                var isWlm = Path.GetExtension(file).ToLower() == ".wlm";
                var liquidFile = LiquidParser.ParseWlwOrWlmFile(file, isWlm, verbose);
                if (liquidFile == null)
                {
                    Console.WriteLine($"Failed to parse file: {file}");
                    continue;
                }

                // Try to find corresponding WLQ file
                var wlqPath = Path.ChangeExtension(file, ".wlq");
                var wlqFile = File.Exists(wlqPath) ? LiquidParser.ParseWlqFile(wlqPath, verbose) : null;

                var relativePath = Path.GetRelativePath(inputDirectory, file);
                var outputPath = Path.Combine(outputDirectory, relativePath);
                var outputDir = Path.GetDirectoryName(outputPath);

                if (!string.IsNullOrEmpty(outputDir))
                {
                    Directory.CreateDirectory(outputDir);
                }

                var converter = new LiquidToObjConverter(liquidFile, wlqFile);
                converter.ConvertToObj(Path.ChangeExtension(outputPath, ".obj"), verbose);

                if (verbose)
                {
                    Console.WriteLine($"Converted: {file} -> {Path.ChangeExtension(outputPath, ".obj")}");
                }
            }

            Console.WriteLine($"Conversion complete. Processed {wlwFiles.Count} files.");
        }
    }
} 