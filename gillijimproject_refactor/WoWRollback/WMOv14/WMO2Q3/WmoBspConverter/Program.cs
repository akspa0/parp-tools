using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using WmoBspConverter.Wmo;

namespace WmoBspConverter
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Simple command-line parsing since System.CommandLine version has compatibility issues
            if (args.Length == 0)
            {
                ShowUsage();
                return 1;
            }

            string inputFile = args[0];
            string? outputFile = null;
            bool extractTextures = true;
            string? outputDir = null;
            bool verbose = false;

            // Parse simple command line arguments
            for (int i = 1; i < args.Length; i++)
            {
                switch (args[i].ToLowerInvariant())
                {
                    case "--output":
                    case "-o":
                        if (i + 1 < args.Length)
                        {
                            outputFile = args[++i];
                        }
                        break;
                    case "--extract-textures":
                    case "-t":
                        if (i + 1 < args.Length && bool.TryParse(args[i + 1], out bool value))
                        {
                            extractTextures = value;
                            i++;
                        }
                        else
                        {
                            extractTextures = true;
                        }
                        break;
                    case "--output-dir":
                    case "-d":
                        if (i + 1 < args.Length)
                        {
                            outputDir = args[++i];
                        }
                        break;
                    case "--verbose":
                    case "-v":
                        verbose = true;
                        break;
                    case "--help":
                    case "-h":
                        ShowUsage();
                        return 0;
                }
            }

            try
            {
                await ConvertAsync(inputFile, outputFile, extractTextures, outputDir, verbose);
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine($"Stack trace: {ex.StackTrace}");
                }
                return 1;
            }
        }

        static void ShowUsage()
        {
            Console.WriteLine("WMO v14 → Quake 3 BSP Converter");
            Console.WriteLine("Usage: WmoBspConverter <input.wmo> [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --output, -o <file>       Output BSP file path");
            Console.WriteLine("  --extract-textures, -t    Extract and convert BLP textures to PNG (default: true)");
            Console.WriteLine("  --output-dir, -d <dir>    Output directory for textures and shaders");
            Console.WriteLine("  --verbose, -v             Enable verbose logging");
            Console.WriteLine("  --help, -h                Show this help message");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  WmoBspConverter building.wmo");
            Console.WriteLine("  WmoBspConverter dungeon.wmo --output maps/dungeon.bsp --extract-textures");
            Console.WriteLine("  WmoBspConverter tower.wmo -o tower.bsp -d ./output -v");
        }

        static async Task ConvertAsync(string inputFile, string? outputFile, bool extractTextures, string? outputDir, bool verbose)
        {
            // Validate input file
            if (!File.Exists(inputFile))
            {
                throw new FileNotFoundException($"Input file not found: {inputFile}");
            }

            // Establish output directory first
            var outputDirectory = outputDir ?? (string.IsNullOrEmpty(outputFile) ? Path.GetDirectoryName(inputFile) : Path.GetDirectoryName(outputFile)) ?? ".";

            // Set output file path: if not specified, place BSP under outputDirectory
            if (string.IsNullOrEmpty(outputFile))
            {
                var name = Path.GetFileNameWithoutExtension(inputFile) + ".bsp";
                outputFile = Path.Combine(outputDirectory, name);
            }

            // Ensure output directory exists
            Directory.CreateDirectory(outputDirectory);
            var textureOutputDir = Path.Combine(outputDirectory, "textures");
            
            Console.WriteLine("WMO v14 → Quake 3 BSP Converter (.NET 9)");
            Console.WriteLine("==========================================");
            Console.WriteLine($"Input: {Path.GetFullPath(inputFile)}");
            Console.WriteLine($"Output: {Path.GetFullPath(outputFile)}");
            Console.WriteLine($"Texture extraction: {(extractTextures ? "Enabled" : "Disabled")}");
            Console.WriteLine($"Output directory: {Path.GetFullPath(outputDirectory)}");
            Console.WriteLine();

            if (verbose)
            {
                Console.WriteLine("[VERBOSE] Verbose mode enabled");
            }

            try
            {
                // Create converter with enhanced features
                var converter = new WmoV14ToBspConverter(outputDirectory, extractTextures);

                // Run enhanced conversion (handles WMOReader geometry and saving internally)
                var result = await converter.ConvertAsync(inputFile, outputDirectory, System.Threading.CancellationToken.None);

                // Display conversion statistics
                Console.WriteLine("✓ Conversion completed successfully!");
                Console.WriteLine($"  ⏱️  Time: {result.ConversionTime.TotalMilliseconds:N0}ms");
                Console.WriteLine($"  📐 Vertices: {result.TotalVertices:N0}");
                Console.WriteLine($"  🔺 Faces: {result.TotalFaces:N0}");
                Console.WriteLine($"  🎨 Textures: {result.TextureCount:N0}");
                Console.WriteLine($"  📦 Models: 1");
                Console.WriteLine($"  🏠 Leaves: 1");
                Console.WriteLine($"  ⚙️  Entities: 1");
                var outBsp = System.IO.Path.Combine(outputDirectory, System.IO.Path.GetFileNameWithoutExtension(inputFile) + ".bsp");
                if (File.Exists(outBsp))
                {
                    Console.WriteLine($"  💾 File size: {new FileInfo(outBsp).Length:N0} bytes");
                }

                // Show texture extraction info
                if (extractTextures && Directory.Exists(textureOutputDir))
                {
                    var textureFiles = Directory.GetFiles(textureOutputDir, "*.png", SearchOption.AllDirectories);
                    Console.WriteLine($"  🖼️  Extracted textures: {textureFiles.Length}");
                    
                    var shaderFiles = Directory.GetFiles(textureOutputDir, "*.shader", SearchOption.AllDirectories);
                    if (shaderFiles.Length > 0)
                    {
                        Console.WriteLine($"  📝 Shader scripts: {shaderFiles.Length}");
                    }
                }

                // Success message for Quake 3 usage
                Console.WriteLine();
                Console.WriteLine("💡 The BSP file is ready for use in Quake 3 mapping tools:");
                Console.WriteLine($"   • GtkRadiant: Load {Path.GetFileName(outputFile)}");
                Console.WriteLine($"   • ioquake3: Copy to baseq3/maps/ directory");
                
                if (extractTextures)
                {
                    Console.WriteLine($"   • Textures: Copy {Path.GetFileName(textureOutputDir)}/ to baseq3/textures/");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Conversion failed: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine();
                    Console.WriteLine("Stack trace:");
                    Console.WriteLine(ex.StackTrace);
                }
                throw;
            }
        }
    }
}