using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using WowToolSuite.Liquid.Coordinates;
using WowToolSuite.Liquid.Models;
using WowToolSuite.Liquid.Converters;
using System.Diagnostics;
using Newtonsoft.Json;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.Entries;
using Warcraft.NET.Files.ADT.Chunks;
using CommandLine;

namespace WowToolSuite.Liquid
{
    public static class LiquidParser
    {
        public static LiquidFile? ParseWlwOrWlmFile(string filePath, bool isWlm = false, bool verbose = false)
        {
            if (verbose)
            {
                Console.WriteLine($"Parsing {(isWlm ? "WLM" : "WLW")} file: {filePath}");
            }

            try
            {
                using var stream = File.OpenRead(filePath);
                using var reader = new BinaryReader(stream);

                if (stream.Length < 16)
                {
                    if (verbose)
                    {
                        Console.WriteLine($"File is too short to contain required header: {filePath}");
                    }
                    return null;
                }

                var header = new LiquidHeader
                {
                    Magic = Encoding.UTF8.GetString(reader.ReadBytes(4)),
                    Version = reader.ReadUInt16(),
                    Unk06 = reader.ReadUInt16(),
                    LiquidType = isWlm ? (ushort)6 : reader.ReadUInt16(), // WLM files always have liquidType as 6 (Magma)
                    Padding = reader.ReadUInt16(),
                    BlockCount = reader.ReadUInt32()
                };

                var result = new LiquidFile
                {
                    Header = header,
                    FilePath = filePath,
                    IsWlm = isWlm
                };

                // Calculate expected file size
                long expectedSize = 16 + header.BlockCount * (48 * 4 + 2 * 4 + 80 * 2);
                if (stream.Length < expectedSize)
                {
                    if (verbose)
                    {
                        Console.WriteLine($"File is too short to contain expected block data: {filePath}");
                    }
                    return null;
                }

                for (int i = 0; i < header.BlockCount; i++)
                {
                    var block = new LiquidBlock();

                    // Read 16 vertices (48 floats total, 3 floats per vertex)
                    for (int j = 0; j < 16; j++)
                    {
                        float x = reader.ReadSingle();
                        float y = reader.ReadSingle();
                        float z = reader.ReadSingle();
                        block.Vertices.Add(new Vector3(x, y, z));
                    }

                    // Read coordinate data (2 floats)
                    block.Coord = new Vector2(reader.ReadSingle(), reader.ReadSingle());

                    // Read block data (80 ushorts)
                    for (int j = 0; j < 80; j++)
                    {
                        block.Data[j] = reader.ReadUInt16();
                    }

                    result.Blocks.Add(block);
                }

                if (verbose)
                {
                    Console.WriteLine($"Finished parsing {(isWlm ? "WLM" : "WLW")} file: {filePath}");
                }

                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing {(isWlm ? "WLM" : "WLW")} file {filePath}: {ex.Message}");
                return null;
            }
        }
    }

    class Program
    {
        private static async Task<int> Main(string[] args)
        {
            return await Parser.Default.ParseArguments<CommandLineOptions>(args)
                .MapResult(async (CommandLineOptions opts) =>
                {
                    try
                    {
                        var stopwatch = new Stopwatch();
                        stopwatch.Start();

                        Directory.CreateDirectory(opts.OutputDirectory);

                        // Test coordinate conversion if verbose mode is enabled
                        if (opts.Verbose)
                        {
                            Console.WriteLine("Testing coordinate conversions...");
                            CoordinateConverter.DebugCoordinates();
                        }

                        var liquid_mappings = new Dictionary<string, List<LiquidBlock>>();
                        var per_file_block_counts = new Dictionary<string, int>();
                        var processor = new LiquidProcessor();

                        double world_center_offset = CoordinateConverter.WORLD_CENTER_OFFSET;

                        // Process all .wlw files in the directory
                        foreach (var file in Directory.GetFiles(opts.WlwDirectory, "*.wlw", SearchOption.AllDirectories))
                        {
                            if (opts.Verbose)
                            {
                                Console.WriteLine($"Processing file: {file}");
                            }

                            try
                            {
                                Console.WriteLine($"Parsing file: {Path.GetFileName(file)}");
                                var blocks = processor.ProcessLiquidFile(file);
                                Console.WriteLine($"Finished parsing file: {Path.GetFileName(file)}");

                                if (blocks.Count > 0)
                                {
                                    per_file_block_counts[file] = blocks.Count;

                                    // Group blocks by ADT
                                    foreach (var block in blocks)
                                    {
                                        if (opts.YamlOutput)
                                        {
                                            OutputBlockToYaml(block, file, opts.OutputDirectory);
                                        }
                                        
                                        // Calculate the center point of the block
                                        float minX = float.MaxValue, maxX = float.MinValue;
                                        float minY = float.MaxValue, maxY = float.MinValue;
                                        
                                        foreach (var vertex in block.Vertices)
                                        {
                                            minX = Math.Min(minX, vertex.X);
                                            maxX = Math.Max(maxX, vertex.X);
                                            minY = Math.Min(minY, vertex.Y);
                                            maxY = Math.Max(maxY, vertex.Y);
                                        }
                                        
                                        float centerX = (minX + maxX) / 2;
                                        float centerY = (minY + maxY) / 2;
                                        
                                        // Convert to ADT coordinates (32,32 is center of map)
                                        var adtCoords = CoordinateConverter.WorldToAdtCoordinates(centerX, centerY);
                                        
                                        // Create ADT filename
                                        string adtFilename = $"development_{adtCoords.X}_{adtCoords.Y}.adt";
                                        
                                        if (!liquid_mappings.ContainsKey(adtFilename))
                                        {
                                            liquid_mappings[adtFilename] = new List<LiquidBlock>();
                                        }
                                        
                                        liquid_mappings[adtFilename].Add(block);
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error processing file {file}: {ex.Message}");
                            }
                        }

                        // Create bin files from liquid mappings
                        var binDirectory = Path.Combine(opts.OutputDirectory, "bin_chunks");
                        Directory.CreateDirectory(binDirectory);

                        if (opts.Verbose)
                        {
                            Console.WriteLine($"Creating bin files in directory: {binDirectory}");
                            Console.WriteLine($"Total ADT mappings to process: {liquid_mappings.Count}");
                        }

                        foreach (var mapping in liquid_mappings)
                        {
                            try
                            {
                                string binFile = Path.Combine(binDirectory, mapping.Key.Replace(".adt", "_mh2o.bin"));
                                
                                if (opts.Verbose)
                                {
                                    Console.WriteLine($"Processing MH2O chunk for {mapping.Key} with {mapping.Value.Count} blocks");
                                }
                                
                                // Create our MH2O chunk
                                var waterData = new MH2OOld();
                                waterData.MH2OHeaders = new MH2OHeader[256];
                                for (int i = 0; i < 256; i++)
                                {
                                    waterData.MH2OHeaders[i] = new MH2OHeader(new byte[MH2OHeader.GetSize()]);
                                }
                                
                                foreach (var block in mapping.Value)
                                {
                                    // Calculate chunk coordinates
                                    var adtInjector = new AdtWaterInjector();
                                    var chunkCoordinates = adtInjector.CalculateChunkCoordinates(block);
                                    var chunkIndex = chunkCoordinates.X + chunkCoordinates.Y * 16;
                                    
                                    if (chunkIndex < 0 || chunkIndex >= 256)
                                    {
                                        if (opts.Verbose)
                                        {
                                            Console.WriteLine($"Warning: Invalid chunk index {chunkIndex} for block in {mapping.Key}");
                                        }
                                        continue;
                                    }
                                    
                                    var header = waterData.MH2OHeaders[chunkIndex];
                                    if (header.LayerCount == 0)
                                    {
                                        header.Instances = new MH2OInstance[1];
                                        header.LayerCount = 1;
                                        header.Instances[0] = new MH2OInstance(new byte[MH2OInstance.GetSize()]);
                                    }
                                    
                                    var instance = header.Instances[0];
                                    instance.LiquidTypeId = (ushort)block.LiquidType;
                                    instance.MinHeightLevel = block.MinHeight;
                                    instance.MaxHeightLevel = block.MaxHeight;
                                    instance.Width = 8;
                                    instance.Height = 8;
                                    
                                    // Create vertex data using the parameterless constructor
                                    instance.VertexData = new MH2OInstanceVertexData();
                                    // Now assign the correctly sized height map
                                    instance.VertexData.HeightMap = new float[9, 9];
                                    instance.RenderBitmapBytes = new byte[8];
                                    
                                    // Set all vertices to enabled
                                    for (int i = 0; i < 8; i++)
                                    {
                                        instance.RenderBitmapBytes[i] = 0xFF;
                                    }
                                    
                                    // Set height values - creating a gentle slope from min to max height
                                    for (int y = 0; y <= 8; y++)
                                    {
                                        for (int x = 0; x <= 8; x++)
                                        {
                                            instance.VertexData.HeightMap[y, x] = block.MinHeight + 
                                                (block.MaxHeight - block.MinHeight) * (y / 8.0f);
                                        }
                                    }
                                }
                                
                                if (opts.YamlOutput)
                                {
                                    OutputMH2OToYaml(waterData, mapping.Key, opts.OutputDirectory);
                                }
                                
                                // Ensure the directory exists again (just to be safe)
                                if (!Directory.Exists(binDirectory))
                                {
                                    if (opts.Verbose)
                                    {
                                        Console.WriteLine($"Re-creating bin directory: {binDirectory}");
                                    }
                                    Directory.CreateDirectory(binDirectory);
                                }
                                
                                // Write the chunk data to a binary file
                                try
                                {
                                    // Get the raw chunk data (without the MH2O header)
                                    byte[] rawMh2oData;
                                    try
                                    {
                                        // Make sure vertex data is properly initialized before serialization
                                        foreach (var header in waterData.MH2OHeaders)
                                        {
                                            if (header.LayerCount > 0)
                                            {
                                                foreach (var instance in header.Instances)
                                                {
                                                    if (instance != null)
                                                    {
                                                        // Ensure render bitmap is initialized
                                                        if (instance.RenderBitmapBytes == null)
                                                        {
                                                            instance.RenderBitmapBytes = new byte[8];
                                                            for (int i = 0; i < 8; i++) 
                                                            {
                                                                instance.RenderBitmapBytes[i] = 0xFF;
                                                            }
                                                        }
                                                        
                                                        // Ensure vertex data is initialized
                                                        if (instance.VertexData == null)
                                                        {
                                                            instance.VertexData = new MH2OInstanceVertexData();
                                                            instance.VertexData.HeightMap = new float[instance.Height + 1, instance.Width + 1];
                                                            
                                                            for (int y = 0; y <= instance.Height; y++)
                                                            {
                                                                for (int x = 0; x <= instance.Width; x++)
                                                                {
                                                                    instance.VertexData.HeightMap[y, x] = instance.MinHeightLevel;
                                                                }
                                                            }
                                                        }
                                                        // Make sure HeightMap dimensions match instance Width and Height
                                                        else if (instance.VertexData.HeightMap.GetLength(0) != instance.Height + 1 || 
                                                                 instance.VertexData.HeightMap.GetLength(1) != instance.Width + 1)
                                                        {
                                                            // Recreate HeightMap with correct dimensions
                                                            float[,] newHeightMap = new float[instance.Height + 1, instance.Width + 1];
                                                            
                                                            for (int y = 0; y <= instance.Height; y++)
                                                            {
                                                                for (int x = 0; x <= instance.Width; x++)
                                                                {
                                                                    newHeightMap[y, x] = instance.MinHeightLevel;
                                                                }
                                                            }
                                                            
                                                            instance.VertexData.HeightMap = newHeightMap;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        
                                        if (opts.Verbose)
                                        {
                                            Console.WriteLine($"Initialized vertex data for {waterData.MH2OHeaders.Count(h => h.LayerCount > 0)} active chunks in {mapping.Key}");
                                        }
                                        
                                        // Now serialize using our custom serializer to avoid Warcraft.NET's serialization issues
                                        var serializedData = MH2OSerializer.Serialize(waterData);
                                        rawMh2oData = serializedData.Skip(8).ToArray();
                                        
                                        if (opts.Verbose)
                                        {
                                            Console.WriteLine($"Successfully serialized MH2O data: {serializedData.Length} bytes (header: 8, data: {rawMh2oData.Length})");
                                        }
                                    }
                                    catch (Exception ex)
                                    {
                                        Console.WriteLine($"Error serializing MH2O data for {mapping.Key}: {ex.Message}");
                                        Console.WriteLine(ex.StackTrace);
                                        continue; // Skip this ADT and move to the next
                                    }
                                    
                                    if (opts.Verbose)
                                    {
                                        Console.WriteLine($"Writing binary data to {binFile}");
                                    }
                                    
                                    using (FileStream fs = new FileStream(binFile, FileMode.Create))
                                    using (BinaryWriter bw = new BinaryWriter(fs))
                                    {
                                        // Write reversed header for WoTLK
                                        bw.Write(Encoding.ASCII.GetBytes("O2HM"));
                                        
                                        // Write the size of the raw data
                                        bw.Write((uint)rawMh2oData.Length);
                                        
                                        // Write the actual data
                                        bw.Write(rawMh2oData);
                                        
                                        // Flush to ensure data is written
                                        fs.Flush();
                                    }
                                    
                                    if (File.Exists(binFile))
                                    {
                                        Console.WriteLine($"Created MH2O chunk file: {binFile} ({new FileInfo(binFile).Length} bytes)");
                                    }
                                    else
                                    {
                                        Console.WriteLine($"Error: Failed to create file {binFile} for unknown reason");
                                    }
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"Error writing bin file {binFile}: {ex.Message}");
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Error creating bin file for {mapping.Key}: {ex.Message}");
                                Console.WriteLine(ex.StackTrace);
                            }
                        }

                        // Write summary file
                        using (StreamWriter writer = new StreamWriter(Path.Combine(opts.OutputDirectory, "liquid_mapping_summary.txt")))
                        {
                            writer.WriteLine("Liquid Mapping Summary");
                            writer.WriteLine("=====================");
                            writer.WriteLine();
                            
                            int totalBlocks = liquid_mappings.Sum(m => m.Value.Count);
                            writer.WriteLine($"Total liquid blocks: {totalBlocks}");
                            writer.WriteLine();
                            
                            writer.WriteLine("ADT Mapping");
                            writer.WriteLine("==========");
                            writer.WriteLine();
                            
                            foreach (var mapping in liquid_mappings.OrderBy(m => m.Key))
                            {
                                writer.WriteLine($"ADT: {mapping.Key}");
                                writer.WriteLine($"  Total blocks: {mapping.Value.Count}");
                                writer.WriteLine();
                            }
                            
                            writer.WriteLine("Per-File Summary");
                            writer.WriteLine("===============");
                            writer.WriteLine();
                            
                            foreach (var file in per_file_block_counts.OrderBy(f => f.Key))
                            {
                                writer.WriteLine($"File: {file.Key}");
                                writer.WriteLine($"  Blocks: {file.Value}");
                                writer.WriteLine();
                            }
                        }

                        // Write JSON mapping file
                        var jsonMapping = new Dictionary<string, object>();
                        jsonMapping["total_blocks"] = liquid_mappings.Sum(m => m.Value.Count);
                        jsonMapping["adt_mappings"] = liquid_mappings.ToDictionary(
                            k => k.Key, 
                            v => new { block_count = v.Value.Count }
                        );
                        jsonMapping["per_file_summary"] = per_file_block_counts;
                        
                        string jsonOutput = JsonConvert.SerializeObject(jsonMapping, Formatting.Indented);
                        File.WriteAllText(Path.Combine(opts.OutputDirectory, "liquid_mapping.json"), jsonOutput);

                        // Generate MH2O structure reports
                        if (opts.YamlOutput)
                        {
                            var blocksPerFile = new Dictionary<string, List<LiquidBlock>>();
                            foreach (var mapping in liquid_mappings)
                            {
                                foreach (var block in mapping.Value)
                                {
                                    if (!blocksPerFile.ContainsKey(block.SourceFile))
                                    {
                                        blocksPerFile[block.SourceFile] = new List<LiquidBlock>();
                                    }
                                    blocksPerFile[block.SourceFile].Add(block);
                                }
                            }
                            GenerateMh2oReports(opts.OutputDirectory, blocksPerFile);
                        }

                        // Patch ADT files if adt-dir is provided
                        if (!string.IsNullOrEmpty(opts.AdtDirectory) && !string.IsNullOrEmpty(opts.PatchedAdtDirectory))
                        {
                            Directory.CreateDirectory(opts.PatchedAdtDirectory);
                            var injector = new AdtWaterInjector();
                            int totalAdtFiles = 0;
                            int successfullyPatched = 0;

                            foreach (var mapping in liquid_mappings)
                            {
                                string adtFile = Path.Combine(opts.AdtDirectory, mapping.Key);
                                string outputAdtFile = Path.Combine(opts.PatchedAdtDirectory, mapping.Key);

                                if (File.Exists(adtFile))
                                {
                                    totalAdtFiles++;
                                    Console.WriteLine($"Injecting MH2O chunk into ADT file: {mapping.Key}");
                                    if (injector.InjectWaterIntoAdt(adtFile, outputAdtFile, mapping.Value.ToArray()))
                                    {
                                        successfullyPatched++;
                                    }
                                }
                                else
                                {
                                    if (opts.Verbose)
                                    {
                                        Console.WriteLine($"ADT file not found: {adtFile}");
                                    }
                                }
                            }

                            // For other ADT files without liquid, just copy them
                            if (opts.PatchAllAdts)
                            {
                                foreach (var file in Directory.GetFiles(opts.AdtDirectory, "*.adt", SearchOption.TopDirectoryOnly))
                                {
                                    string fileName = Path.GetFileName(file);
                                    if (!liquid_mappings.ContainsKey(fileName))
                                    {
                                        string outputFile = Path.Combine(opts.PatchedAdtDirectory, fileName);
                                        if (!File.Exists(outputFile))
                                        {
                                            totalAdtFiles++;
                                            Console.WriteLine($"Copying ADT file without changes: {fileName}");
                                            File.Copy(file, outputFile, true);
                                        }
                                    }
                                }
                            }

                            Console.WriteLine($"ADT processing complete. Processed {totalAdtFiles} ADT files, successfully patched {successfullyPatched} with water data.");
                            Console.WriteLine($"Patched ADT files saved to: {opts.PatchedAdtDirectory}");
                        }

                        stopwatch.Stop();
                        Console.WriteLine($"Elapsed time: {stopwatch.Elapsed}");

                        return 0;
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Error: {ex.Message}");
                        Console.WriteLine(ex.StackTrace);
                        return 1;
                    }
                }, err => Task.FromResult(1));
        }

        private static void OutputBlockToYaml(LiquidBlock block, string sourceFile, string outputDir)
        {
            var yamlDir = Path.Combine(outputDir, "yaml_output", "blocks");
            Directory.CreateDirectory(yamlDir);
            
            // Generate a unique identifier for the filename only - not for the content
            float avgX = block.Vertices.Average(v => v.X);
            float avgY = block.Vertices.Average(v => v.Y);
            string blockFileId = $"{Path.GetFileNameWithoutExtension(sourceFile).Replace(" ", "_")}_{avgX:F0}_{avgY:F0}_{block.LiquidType}";
            var filename = Path.Combine(yamlDir, $"block_{blockFileId}.yaml");
            
            var yaml = new StringBuilder();
            yaml.AppendLine("# Liquid Block Data");
            yaml.AppendLine($"source_file: {Path.GetFileName(sourceFile)}");
            yaml.AppendLine($"liquid_type: {block.LiquidType}");
            
            // Add human-readable liquid type name
            string liquidTypeName = GetLiquidTypeName(block.LiquidType);
            yaml.AppendLine($"liquid_type_name: \"{liquidTypeName}\"");
            
            yaml.AppendLine($"min_height: {block.MinHeight}");
            yaml.AppendLine($"max_height: {block.MaxHeight}");
            
            // Calculate center point and derived ADT coordinates
            float minX = float.MaxValue, maxX = float.MinValue;
            float minY = float.MaxValue, maxY = float.MinValue;
            
            foreach (var vertex in block.Vertices)
            {
                minX = Math.Min(minX, vertex.X);
                maxX = Math.Max(maxX, vertex.X);
                minY = Math.Min(minY, vertex.Y);
                maxY = Math.Max(maxY, vertex.Y);
            }
            
            float centerX = (minX + maxX) / 2;
            float centerY = (minY + maxY) / 2;
            
            yaml.AppendLine($"center_point:");
            yaml.AppendLine($"  x: {centerX}");
            yaml.AppendLine($"  y: {centerY}");
            
            var adtCoords = CoordinateConverter.WorldToAdtCoordinates(centerX, centerY);
            yaml.AppendLine($"adt_coordinates:");
            yaml.AppendLine($"  adt_x: {adtCoords.X}");
            yaml.AppendLine($"  adt_y: {adtCoords.Y}");
            yaml.AppendLine($"  adt_file: development_{adtCoords.X}_{adtCoords.Y}.adt");
            
            // Calculate chunk coordinates within ADT
            var adtInjector = new AdtWaterInjector();
            var chunkCoordinates = adtInjector.CalculateChunkCoordinates(block);
            yaml.AppendLine($"chunk_coordinates:");
            yaml.AppendLine($"  chunk_x: {chunkCoordinates.X}");
            yaml.AppendLine($"  chunk_y: {chunkCoordinates.Y}");
            yaml.AppendLine($"  chunk_index: {chunkCoordinates.X + chunkCoordinates.Y * 16}");
            
            yaml.AppendLine("vertices:");
            for (int i = 0; i < block.Vertices.Count; i++)
            {
                var v = block.Vertices[i];
                yaml.AppendLine($"  - index: {i}");
                yaml.AppendLine($"    x: {v.X}");
                yaml.AppendLine($"    y: {v.Y}");
                yaml.AppendLine($"    z: {v.Z}");
            }
            
            File.WriteAllText(filename, yaml.ToString());
        }
        
        /// <summary>
        /// Maps liquid type IDs to their human-readable names
        /// </summary>
        private static string GetLiquidTypeName(ushort liquidTypeId)
        {
            switch (liquidTypeId)
            {
                case 0: return "Water";
                case 1: return "Ocean";
                case 2: return "Magma/Lava";
                case 3: return "Slime";
                case 4: return "River";
                case 6: return "Magma";
                case 8: return "Fast Flowing";
                case 11: return "Naxxramas Slime";
                default: return $"Unknown ({liquidTypeId})";
            }
        }
        
        private static void OutputMH2OToYaml(MH2OOld waterData, string adtFile, string outputDir)
        {
            var yamlDir = Path.Combine(outputDir, "yaml_output", "mh2o");
            Directory.CreateDirectory(yamlDir);
            
            var filename = Path.Combine(yamlDir, adtFile.Replace(".adt", "_mh2o.yaml"));
            
            var yaml = new StringBuilder();
            yaml.AppendLine("# MH2O Chunk Data");
            yaml.AppendLine($"adt_file: {adtFile}");
            
            int totalLayers = 0;
            for (int i = 0; i < waterData.MH2OHeaders.Length; i++)
            {
                var header = waterData.MH2OHeaders[i];
                if (header.LayerCount > 0)
                {
                    int chunkX = i % 16;
                    int chunkY = i / 16;
                    
                    yaml.AppendLine($"chunk_{chunkX}_{chunkY}:");
                    yaml.AppendLine($"  chunk_index: {i}");
                    yaml.AppendLine($"  layer_count: {header.LayerCount}");
                    
                    for (int j = 0; j < header.LayerCount; j++)
                    {
                        var instance = header.Instances[j];
                        totalLayers++;
                        
                        yaml.AppendLine($"  layer_{j}:");
                        yaml.AppendLine($"    liquid_type_id: {instance.LiquidTypeId}");
                        yaml.AppendLine($"    min_height_level: {instance.MinHeightLevel}");
                        yaml.AppendLine($"    max_height_level: {instance.MaxHeightLevel}");
                        yaml.AppendLine($"    width: {instance.Width}");
                        yaml.AppendLine($"    height: {instance.Height}");
                        
                        yaml.AppendLine($"    render_bitmap: {BitConverter.ToString(instance.RenderBitmapBytes).Replace("-", "")}");
                        
                        // Sample height map values (first few points)
                        yaml.AppendLine($"    height_map_samples:");
                        for (int y = 0; y < Math.Min(3, 9); y++)
                        {
                            for (int x = 0; x < Math.Min(3, 9); x++)
                            {
                                if (instance.VertexData != null && instance.VertexData.HeightMap != null)
                                {
                                    yaml.AppendLine($"      - position: [{x},{y}]");
                                    yaml.AppendLine($"        height: {instance.VertexData.HeightMap[y, x]}");
                                }
                            }
                        }
                    }
                }
            }
            
            yaml.AppendLine($"summary:");
            yaml.AppendLine($"  total_layers: {totalLayers}");
            yaml.AppendLine($"  active_chunks: {waterData.MH2OHeaders.Count(h => h.LayerCount > 0)}");
            
            File.WriteAllText(filename, yaml.ToString());
        }

        private static void GenerateMh2oReports(string outputDir, Dictionary<string, List<LiquidBlock>> liquidBlocksPerFile)
        {
            string reportDir = Path.Combine(outputDir, "mh2o_reports");
            Directory.CreateDirectory(reportDir);
            
            // Create a summary report
            using (var summaryWriter = new StreamWriter(Path.Combine(reportDir, "mh2o_summary.txt")))
            {
                summaryWriter.WriteLine("MH2O Structure Summary Report");
                summaryWriter.WriteLine("============================");
                summaryWriter.WriteLine();
                
                foreach (var entry in liquidBlocksPerFile)
                {
                    string sourceFile = entry.Key;
                    var blocks = entry.Value;
                    
                    summaryWriter.WriteLine($"File: {sourceFile}");
                    summaryWriter.WriteLine($"  Total blocks: {blocks.Count}");
                    summaryWriter.WriteLine();
                    
                    // Create detailed report for this file
                    string detailReportPath = Path.Combine(reportDir, $"mh2o_detail_{Path.GetFileNameWithoutExtension(sourceFile)}.txt");
                    using (var detailWriter = new StreamWriter(detailReportPath))
                    {
                        detailWriter.WriteLine($"MH2O Detailed Report for {sourceFile}");
                        detailWriter.WriteLine("=".PadRight(50, '='));
                        detailWriter.WriteLine();
                        
                        foreach (var block in blocks)
                        {
                            // Calculate the center point of the block (same as in main processing)
                            float minX = float.MaxValue, maxX = float.MinValue;
                            float minY = float.MaxValue, maxY = float.MinValue;
                            
                            foreach (var vertex in block.Vertices)
                            {
                                minX = Math.Min(minX, vertex.X);
                                maxX = Math.Max(maxX, vertex.X);
                                minY = Math.Min(minY, vertex.Y);
                                maxY = Math.Max(maxY, vertex.Y);
                            }
                            
                            float centerX = (minX + maxX) / 2;
                            float centerY = (minY + maxY) / 2;
                            
                            // Convert using center coordinates instead of block.GlobalX/Y
                            var adtCoords = CoordinateConverter.WorldToAdtCoordinates(centerX, centerY);
                            var adtInjector = new AdtWaterInjector();
                            
                            // Need to use updated center coordinates with the block
                            var blockCopy = new LiquidBlock();
                            blockCopy.Vertices.AddRange(block.Vertices);
                            blockCopy.LiquidType = block.LiquidType;
                            blockCopy.SourceFile = block.SourceFile;
                            blockCopy.Coord = new Vector2(centerX, centerY);
                            
                            var chunkCoordinates = adtInjector.CalculateChunkCoordinates(blockCopy);
                            var chunkIndex = chunkCoordinates.X + chunkCoordinates.Y * 16;
                            
                            detailWriter.WriteLine($"Block at ({centerX:F2}, {centerY:F2})");
                            detailWriter.WriteLine($"  ADT: development_{adtCoords.X}_{adtCoords.Y}.adt");
                            detailWriter.WriteLine($"  Chunk: {chunkIndex} (X:{chunkCoordinates.X}, Y:{chunkCoordinates.Y})");
                            detailWriter.WriteLine();
                            detailWriter.WriteLine("  MH2O Structure:");
                            detailWriter.WriteLine("  --------------");
                            detailWriter.WriteLine($"  LiquidTypeId: {block.LiquidType} ({GetLiquidTypeName(block.LiquidType)})");
                            detailWriter.WriteLine($"  MinHeight: {block.MinHeight:F6}");
                            detailWriter.WriteLine($"  MaxHeight: {block.MaxHeight:F6}");
                            detailWriter.WriteLine($"  Width/Height: 8/8");
                            detailWriter.WriteLine($"  VertexFormat: 0 (standard)");
                            detailWriter.WriteLine($"  RenderBitmap: All cells visible (8 bytes of 0xFF)");
                            detailWriter.WriteLine();
                            detailWriter.WriteLine("  Vertex Height Map Sample (9x9 grid, values in meters):");
                            detailWriter.WriteLine("  -----------------------------------------------");
                            
                            // Show a sample of the height map (first/last rows & middle)
                            detailWriter.WriteLine("  Top row:    " + string.Join(" ", Enumerable.Range(0, 9).Select(x => 
                                (block.MinHeight + (block.MaxHeight - block.MinHeight) * (0 / 8.0f)).ToString("F4"))));
                            detailWriter.WriteLine("  Middle row: " + string.Join(" ", Enumerable.Range(0, 9).Select(x => 
                                (block.MinHeight + (block.MaxHeight - block.MinHeight) * (4 / 8.0f)).ToString("F4"))));
                            detailWriter.WriteLine("  Bottom row: " + string.Join(" ", Enumerable.Range(0, 9).Select(x => 
                                (block.MinHeight + (block.MaxHeight - block.MinHeight) * (8 / 8.0f)).ToString("F4"))));
                            
                            detailWriter.WriteLine();
                            detailWriter.WriteLine("  Binary Structure Size:");
                            detailWriter.WriteLine("  --------------------");
                            detailWriter.WriteLine("  MH2OHeader: 12 bytes");
                            detailWriter.WriteLine("  MH2OInstance: 20 bytes");
                            detailWriter.WriteLine("  RenderBitmap: 8 bytes");
                            detailWriter.WriteLine("  VertexHeightMap: 324 bytes (9×9×4)");
                            detailWriter.WriteLine("  Total: ~364 bytes per instance");
                            detailWriter.WriteLine();
                            detailWriter.WriteLine(new string('-', 50));
                            detailWriter.WriteLine();
                        }
                    }
                    
                    summaryWriter.WriteLine($"  Detailed report: {Path.GetFileName(detailReportPath)}");
                    summaryWriter.WriteLine();
                }
            }
            
            Console.WriteLine($"Generated MH2O structure reports in {reportDir}");
        }
    }
} 