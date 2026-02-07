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
        var singleOption = new Option<bool>("--single", () => true, "Export as single OBJ file instead of split by geoset");

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

        try
        {
            MdxFile mdx;
            if (inputExt == ".mdx" && input.Exists)
            {
                mdx = MdxFile.Load(input.FullName);
            }
            else if (gamePath != null && gamePath.Exists)
            {
                using var mpq = new NativeMpqService();
                mpq.LoadArchives(new[] { gamePath.FullName });

                string virtualPath = input.ToString(); 
                if (virtualPath.Contains(gamePath.FullName))
                    virtualPath = virtualPath.Replace(gamePath.FullName, "").TrimStart('\\');
                virtualPath = virtualPath.Replace('/', '\\').TrimStart('\\');
                modelDir = Path.GetDirectoryName(virtualPath) ?? ".";

                var mdxData = mpq.ReadFile(virtualPath);
                if (mdxData == null)
                {
                    Console.Error.WriteLine($"[ERROR] Could not find \"{virtualPath}\" in archives.");
                    return;
                }

                using var ms = new MemoryStream(mdxData);
                mdx = MdxFile.Load(ms);
                mdx.ModelName = Path.GetFileNameWithoutExtension(input.Name);
                mdx.RawData = mdxData;
            }
            else
            {
                Console.Error.WriteLine($"[ERROR] File not found: {input.FullName}");
                return;
            }

            // Validation gate: check for usable geometry
            int validGeosets = mdx.Geosets.Count(g => g.Vertices.Count > 0 && g.Indices.Count > 0);
            Console.WriteLine($"Loaded: v{mdx.Version}, {mdx.Geosets.Count} geosets ({validGeosets} valid), {mdx.Textures.Count} textures");

            if (validGeosets == 0 && outputExt == ".obj")
            {
                Console.Error.WriteLine($"[SKIP] No valid geometry in {input.Name} — skipping OBJ write.");
                return;
            }

            // Ensure output directory exists
            if (output.Directory != null && !output.Directory.Exists)
                output.Directory.Create();

            // Texture Export Phase — returns map of textureIdx -> exported PNG filename
            var exportedTextures = ExportTextures(mdx, input.FullName, modelDir, outputDir, gamePath);

            if (outputExt == ".mdl")
            {
                mdx.SaveMdl(output.FullName);
                Console.WriteLine($"Saved MDL: {output.FullName}");
            }
            else if (outputExt == ".obj")
            {
                mdx.SaveObj(output.FullName, split: !single, exportedTextures: exportedTextures);
            }
            else if (outputExt == ".m2")
            {
                Console.WriteLine("M2 writing not yet implemented.");
            }
            else
            {
                Console.Error.WriteLine($"Unsupported output format: {outputExt}");
            }
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ERROR] {input.Name}: {ex.Message}");
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

    /// <summary>
    /// Exports textures and returns a map of texture index -> exported PNG filename.
    /// </summary>
    static Dictionary<int, string> ExportTextures(MdxFile mdx, string modelPath, string modelDir, string outputDir, DirectoryInfo? gamePath)
    {
        var result = new Dictionary<int, string>();
        using var mpqService = new NativeMpqService();
        DbcService? dbcService = null;

        if (gamePath != null && gamePath.Exists)
        {
            mpqService.LoadArchives(new[] { gamePath.FullName });

            string dbcPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "..", "DBCTool", "out", "0.5.3");
            if (!Directory.Exists(dbcPath)) dbcPath = @".\DBCTool\out\0.5.3";
            
            dbcService = new DbcService(dbcPath);
            try { dbcService.Initialize(); } catch { /* DBC optional */ }
        }

        var textureService = new TextureService(mpqService, dbcService ?? new DbcService(""));

        for (int i = 0; i < mdx.Textures.Count; i++)
        {
            var tex = mdx.Textures[i];
            if (string.IsNullOrEmpty(tex.Path) && tex.ReplaceableId == 0) 
                continue;

            var pngName = textureService.ExportTexture(tex, mdx.ModelName, modelPath, modelDir, outputDir);
            if (pngName != null)
            {
                result[i] = pngName;
                Console.WriteLine($"  Texture[{i}]: {pngName}");
            }
            else
            {
                Console.WriteLine($"  Texture[{i}]: [WARN] export failed (Path=\"{tex.Path}\", ReplaceableId={tex.ReplaceableId})");
            }
        }

        return result;
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
