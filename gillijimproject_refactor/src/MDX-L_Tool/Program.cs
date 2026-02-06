using System.CommandLine;
using System.Text;
using MdxLTool.Formats.Mdx;
using MdxLTool.Services;

namespace MdxLTool;

class Program
{
    static async Task<int> Main(string[] args)
    {
        var rootCommand = new RootCommand("MDX-L_Tool - Alpha 0.5.3 Model Converter");

        var convertCommand = new Command("convert", "Convert model files");
        var inputArg = new Argument<FileInfo>("input", "Input model file");
        var outputArg = new Argument<FileInfo>("output", "Output path");
        var targetOption = new Option<string>("--target", () => "mdl", "Target format (mdl, obj, m2)");
        var gamePathOption = new Option<DirectoryInfo?>("--game-path", "Path to game folder for asset resolution");
        var singleOption = new Option<bool>("--single", "Export as single OBJ file instead of split by geoset");

        convertCommand.AddArgument(inputArg);
        convertCommand.AddArgument(outputArg);
        convertCommand.AddOption(targetOption);
        convertCommand.AddOption(gamePathOption);
        convertCommand.AddOption(singleOption);

        convertCommand.SetHandler(ConvertHandler, inputArg, outputArg, targetOption, gamePathOption, singleOption);
        rootCommand.AddCommand(convertCommand);

        var batchCommand = new Command("batch", "Batch convert multiple models");
        var inputPatternArg = new Argument<string>("pattern", "Input file pattern (e.g. \"*.mdx\")");
        var outputDirArg = new Argument<DirectoryInfo>("output", "Output directory");
        
        batchCommand.AddArgument(inputPatternArg);
        batchCommand.AddArgument(outputDirArg);
        batchCommand.AddOption(targetOption);
        batchCommand.AddOption(gamePathOption);
        batchCommand.AddOption(singleOption);
        
        batchCommand.SetHandler(BatchHandler, inputPatternArg, outputDirArg, targetOption, gamePathOption, singleOption);
        rootCommand.AddCommand(batchCommand);

        var infoCommand = new Command("info", "Display information about a model file");
        infoCommand.AddArgument(inputArg);
        infoCommand.SetHandler(InfoHandler, inputArg);
        rootCommand.AddCommand(infoCommand);

        return await rootCommand.InvokeAsync(args);
    }

    static void ConvertHandler(FileInfo input, FileInfo output, string target, DirectoryInfo? gamePath, bool single)
    {
        string inputExt = input.Extension.ToLower();
        string outputExt = output.Extension.ToLower();
        string outputDir = output.DirectoryName ?? ".";
        string modelDir = input.DirectoryName ?? ".";

        Console.WriteLine($"Converting: {input.Name} -> {output.Name}");
        if (gamePath != null) Console.WriteLine($"Game path: {gamePath.FullName}");

        try
        {
            MdxFile mdx;
            if (inputExt == ".mdx" && input.Exists)
            {
                mdx = MdxFile.Load(input.FullName);
            }
            else if (gamePath != null && gamePath.Exists)
            {
                // Attempt to load from game path (MPQ)
                Console.WriteLine($"Local file not found. Attempting to load from game path: {input.Name}");
                using var mpq = new NativeMpqService();
                mpq.LoadArchives(new[] { gamePath.FullName });

                // Standardize the virtual path.
                string virtualPath = input.ToString(); 
                if (virtualPath.Contains(gamePath.FullName))
                {
                    virtualPath = virtualPath.Replace(gamePath.FullName, "").TrimStart('\\');
                }
                virtualPath = virtualPath.Replace('/', '\\').TrimStart('\\');
                
                // For MPQ textures, we need the virtual directory
                modelDir = Path.GetDirectoryName(virtualPath) ?? ".";

                var mdxData = mpq.ReadFile(virtualPath);
                if (mdxData == null)
                {
                    Console.WriteLine($"[ERROR] Could not find virtual path \"{virtualPath}\" in archives.");
                    return;
                }

                using var ms = new MemoryStream(mdxData);
                mdx = MdxFile.Load(ms);
                mdx.ModelName = Path.GetFileNameWithoutExtension(input.Name);
                mdx.RawData = mdxData; // Helper property
                Console.WriteLine($"Loaded MDX from MPQ: {virtualPath}");
            }
            else
            {
                Console.WriteLine($"[ERROR] File not found: {input.FullName}");
                return;
            }

            Console.WriteLine($"Loaded MDX: Version {mdx.Version}, {mdx.Geosets.Count} geosets");

            var sb = new StringBuilder();
            sb.AppendLine($"Texture Count: {mdx.Textures.Count}");
            for (int i = 0; i < mdx.Textures.Count; i++)
                 sb.AppendLine($"Texture[{i}]: {mdx.Textures[i].Path} (ID: {mdx.Textures[i].ReplaceableId})");
            File.WriteAllText("debug_textures.txt", sb.ToString());

            // Dump for inspection
            File.WriteAllBytes("debug_dump.mdx", mdx.RawData ?? new byte[0]); // Need to expose RawData or re-read? 
            // MdxFile might not store RawData. We can save it using SaveMdl? No.
            // MPQ extraction provides `mdxData`.
            
            // Texture Export Phase
            ExportTextures(mdx, input.FullName, modelDir, outputDir, gamePath);

            if (outputExt == ".mdl")
            {
                if (output.Directory != null && !output.Directory.Exists)
                    output.Directory.Create();
                
                mdx.SaveMdl(output.FullName);
                Console.WriteLine($"Saved MDL: {output.FullName}");
            }
            else if (outputExt == ".obj")
            {
                if (output.Directory != null && !output.Directory.Exists)
                    output.Directory.Create();

                mdx.SaveObj(output.FullName, split: !single);
                if (single)
                    Console.WriteLine($"Saved OBJ (Single): {output.FullName}");
                else
                    Console.WriteLine($"Saved OBJ (Split): {output.FullName} (and associated geoset files)");
            }
            else if (outputExt == ".m2")
            {
                Console.WriteLine("M2 writing not yet implemented (Phase 2)");
            }
            else
            {
                Console.Error.WriteLine($"Unsupported output format: {outputExt}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
        }
    }

    static void BatchHandler(string pattern, DirectoryInfo outputDir, string target, DirectoryInfo? gamePath, bool single)
    {
        string searchDir = Directory.GetCurrentDirectory();
        string searchPattern = pattern;

        // Reset output directory logic
        if (!outputDir.Exists) outputDir.Create();

        // Handle full paths in pattern
        if (Path.IsPathRooted(pattern))
        {
            searchDir = Path.GetDirectoryName(pattern) ?? ".";
            searchPattern = Path.GetFileName(pattern);
        }

        Console.WriteLine($"Batch processing: \"{searchPattern}\" in \"{searchDir}\"");
        
        var filesToProcess = new List<string>();

        // 1. Local Files
        if (Directory.Exists(searchDir))
        {
            var localFiles = Directory.GetFiles(searchDir, searchPattern);
            filesToProcess.AddRange(localFiles);
        }

        // 2. MPQ Files (if game path provided)
        if (gamePath != null && gamePath.Exists)
        {
            Console.WriteLine("Scanning MPQs for matching files...");
            using var mpq = new NativeMpqService();
            mpq.LoadArchives(new[] { gamePath.FullName });
            var mpqFiles = mpq.ListFiles(searchPattern); // Use the filename part (e.g. *.mdx)
            Console.WriteLine($"Found {mpqFiles.Count} files in MPQs.");
            filesToProcess.AddRange(mpqFiles);
            // Note: mpqFiles are virtual paths like "Creature\Foo\Bar.mdx"
        }

        Console.WriteLine($"Total files to process: {filesToProcess.Count}");
        
        // Deduplicate?
        // Maybe. Local files might override MPQ files.
        // But for now let's just process them. Local files are full paths, MPQ files are relative.

        int success = 0;
        int fail = 0;

        foreach (var file in filesToProcess)
        {
            try
            {
                // Determine input/output names
                string inputName = Path.GetFileName(file);
                // For MPQ files, file string is the virtual path.
                
                // We need a FileInfo object for ConvertHandler.
                // If it's a full path (local), easy.
                // If it's a virtual path, we can create a FileInfo but Exists will be false.
                
                FileInfo fileInfo;
                if (Path.IsPathRooted(file))
                    fileInfo = new FileInfo(file);
                else
                    fileInfo = new FileInfo(file); // Relative/Virtual path

                // Output filename
                // If virtual path has directories, we might want to mirror them?
                // E.g. Creature\Foo.mdx -> output\Creature\Foo.obj
                // The current ConvertHandler flattens or uses outputDir.
                
                string relPath = file;
                if (Path.IsPathRooted(file)) relPath = Path.GetFileName(file); // Flatten local files for now? Or keep structure?
                
                // Let's preserve directory structure if it's an MPQ file
                string newExt = "." + target;
                string newRelPath = Path.ChangeExtension(relPath, newExt);
                string finalOutputPath = Path.Combine(outputDir.FullName, newRelPath);
                
                var outputFile = new FileInfo(finalOutputPath);

                Console.WriteLine($"[{success+fail+1}/{filesToProcess.Count}] Converting {relPath}...");
                ConvertHandler(fileInfo, outputFile, target, gamePath, single);
                success++;
            }
            catch (Exception ex)
            {
                // Console.Error.WriteLine($"[ERROR] Failed to convert {file}: {ex.Message}");
                // ConvertHandler logs errors.
                fail++;
            }
        }

        Console.WriteLine($"Batch Complete. Success: {success}, Failed: {fail}");
    }

    static void ExportTextures(MdxFile mdx, string modelPath, string modelDir, string outputDir, DirectoryInfo? gamePath)
    {
        using var mpqService = new NativeMpqService();
        DbcService? dbcService = null;

        if (gamePath != null && gamePath.Exists)
        {
            Console.WriteLine("Loading game archives for texture export...");
            mpqService.LoadArchives(new[] { gamePath.FullName });

            // Initialize DBC Service if we have a game path
            // We expect DBC CSVs to be in a specific folder relative to the tool or provided path
            // For now, let's assume they are in DBCTool/out/0.5.3 relative to the workspace root
            // Or we can look for them in the gamePath itself?
            // User provided: j:\wowDev\parp-tools\gillijimproject_refactor\DBCTool\out\0.5.3
            string dbcPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "DBCTool", "out", "0.5.3");
            if (!Directory.Exists(dbcPath)) dbcPath = @".\DBCTool\out\0.5.3"; // Fallback
            
            Console.WriteLine($"Initializing DBC Service from: {dbcPath}");
            dbcService = new DbcService(dbcPath);
            try { dbcService.Initialize(); } catch { Console.WriteLine("[WARN] Failed to initialize DBC service."); }
        }

        var textureService = new TextureService(mpqService, dbcService ?? new DbcService(""));
        int exported = 0;

        foreach (var tex in mdx.Textures)
        {
            if (string.IsNullOrEmpty(tex.Path) && tex.ReplaceableId == 0) 
                continue;

            Console.WriteLine($"  Processing texture: Path=\"{tex.Path}\", ReplaceableId={tex.ReplaceableId}");
            var pngPath = textureService.ExportTexture(tex, mdx.ModelName, modelPath, modelDir, outputDir);
            if (pngPath != null)
            {
                Console.WriteLine($"    Exported: {pngPath}");
                exported++;
            }
            else
            {
                Console.WriteLine($"    [WARN] Failed to export texture.");
            }
        }
        Console.WriteLine($"Total textures exported: {exported}");
    }

    static void InfoHandler(FileInfo input)
    {
        if (!input.Exists)
        {
            Console.Error.WriteLine($"Error: File not found: {input.FullName}");
            return;
        }

        try
        {
            string ext = input.Extension.ToLower();
            if (ext == ".mdx")
            {
                var mdx = MdxFile.Load(input.FullName);
                Console.WriteLine($"Model: {mdx.ModelName}");
                Console.WriteLine($"Version: {mdx.Version}");
                Console.WriteLine($"Geosets: {mdx.Geosets.Count}");
                Console.WriteLine($"Textures: {mdx.Textures.Count}");
                for (int i = 0; i < mdx.Textures.Count; i++)
                {
                    var tex = mdx.Textures[i];
                    Console.WriteLine($"  [{i}] Path=\"{tex.Path}\", ReplaceableId={tex.ReplaceableId}");
                }
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Error: {ex.Message}");
        }
    }
}
