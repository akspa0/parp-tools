// Program.cs in WoWToolbox.FileDumper
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using WoWToolbox.Core.Navigation.PM4;
using Warcraft.NET.Files.ADT.TerrainObject.Zero;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using WoWToolbox.FileDumper.DTOs;
using System.Numerics;
using WoWToolbox.Core.Navigation.PM4.Chunks;
using Warcraft.NET.Files.ADT.Chunks;

namespace WoWToolbox.FileDumper
{
    public class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("WoWToolbox File Dumper");
            Console.WriteLine("======================");

            string? inputDirectory = null;
            string? outputDirectory = null;

            // --- Argument Parsing ---
            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "-d":
                    case "--directory":
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("-"))
                        {
                            inputDirectory = args[++i];
                        }
                        else
                        {
                            WriteError("Missing value for input directory argument (-d/--directory).");
                            PrintHelp();
                            return;
                        }
                        break;
                    case "-o":
                    case "--output":
                        if (i + 1 < args.Length && !args[i + 1].StartsWith("-"))
                        {
                            outputDirectory = args[++i];
                        }
                        else
                        {
                            WriteError("Missing value for output directory argument (-o/--output).");
                            PrintHelp();
                            return;
                        }
                        break;
                    case "-h":
                    case "--help":
                        PrintHelp();
                        return;
                    default:
                        WriteError($"Unknown argument: {args[i]}");
                        PrintHelp();
                        return;
                }
            }

            // --- Validation ---
            if (string.IsNullOrEmpty(inputDirectory))
            {
                WriteError("Input directory (-d/--directory) is required.");
                PrintHelp();
                return;
            }
             if (string.IsNullOrEmpty(outputDirectory))
            {
                WriteError("Output directory (-o/--output) is required.");
                PrintHelp();
                return;
            }

            if (!Directory.Exists(inputDirectory))
            {
                WriteError($"Input directory not found: {inputDirectory}");
                return;
            }

            // --- Create Output Directory ---
            try
            {
                Directory.CreateDirectory(outputDirectory);
                Console.WriteLine($"Output will be written to: {Path.GetFullPath(outputDirectory)}");
            }
            catch (Exception ex)
            {
                WriteError($"Failed to create output directory '{outputDirectory}': {ex.Message}");
                return;
            }

            // --- Process Files ---
            ProcessDirectory(inputDirectory, outputDirectory);

            Console.WriteLine("\nFile dumping complete. Press Enter to exit.");
            Console.ReadLine();
        }

        private static void ProcessDirectory(string inputDirectory, string outputDirectory)
        {
            Console.WriteLine($"\nScanning input directory '{inputDirectory}' for PM4 files...");

            var pm4Files = Directory.EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly).ToList();
            Console.WriteLine($"Found {pm4Files.Count} PM4 files.");

            if (pm4Files.Count == 0)
            {
                Console.WriteLine("No PM4 files found to process.");
                return;
            }

            // Initialize YamlDotNet serializer
            var serializer = new SerializerBuilder()
                .WithNamingConvention(PascalCaseNamingConvention.Instance)
                .ConfigureDefaultValuesHandling(DefaultValuesHandling.OmitNull | DefaultValuesHandling.OmitEmptyCollections | DefaultValuesHandling.OmitDefaults) // Make YAML cleaner
                .DisableAliases() // Avoid YAML anchors/aliases for simplicity
                .Build();

            Console.WriteLine("\nProcessing files...");
            foreach (var pm4FilePath in pm4Files)
            {
                string pm4FileName = Path.GetFileName(pm4FilePath);
                string pm4BaseName = Path.GetFileNameWithoutExtension(pm4FilePath);
                Console.WriteLine($"  Processing {pm4FileName}...");

                // --- PM4 Processing ---
                PM4File? pm4File = null; // Use nullable
                try
                {
                    // TODO: Consider PM4File.FromFile() if it handles streams better
                    byte[] pm4Bytes = File.ReadAllBytes(pm4FilePath);
                    if (pm4Bytes.Length == 0)
                    {
                        Console.WriteLine("      Skipping empty PM4 file.");
                        continue;
                    }
                    pm4File = new PM4File(pm4Bytes); // Assuming constructor takes byte[]
                }
                catch (Exception pm4LoadEx)
                {
                    WriteError($"  Error loading PM4 file '{pm4FileName}': {pm4LoadEx.Message}");
                    continue; // Skip to next file if loading fails
                }

                if (pm4File != null)
                {
                    try
                    {
                        // --- Map PM4 to DTO ---
                        Console.WriteLine("      Mapping PM4 data to DTO...");
                        Pm4FileDto pm4Dto = MapPm4ToDto(pm4File);

                        // --- Serialize PM4 DTO --- 
                        Console.WriteLine("      Serializing PM4 DTO...");
                        string pm4YamlOutput = serializer.Serialize(pm4Dto);
                        string pm4YamlPath = Path.Combine(outputDirectory, $"{pm4BaseName}.pm4.yaml");
                        File.WriteAllText(pm4YamlPath, $"# Dump of {pm4FileName}\n---\n" + pm4YamlOutput);
                        Console.WriteLine($"      -> Saved PM4 YAML to {Path.GetFileName(pm4YamlPath)}");

                        // --- ADT (_obj0) Processing ---
                        string obj0AdtPath = Path.Combine(inputDirectory, pm4BaseName + "_obj0.adt");
                        if (File.Exists(obj0AdtPath))
                        {
                            Console.WriteLine($"    Found corresponding {Path.GetFileName(obj0AdtPath)}. Processing...");
                            TerrainObjectZero? adtObj0Data = null; // Use nullable
                            try
                            {
                                byte[] obj0Bytes = File.ReadAllBytes(obj0AdtPath);
                                if (obj0Bytes.Length > 0)
                                {
                                    adtObj0Data = new TerrainObjectZero(obj0Bytes);
                                }
                                else
                                {
                                    Console.WriteLine($"        Skipping empty file: {Path.GetFileName(obj0AdtPath)}");
                                }
                            }
                            catch (Exception adtLoadEx)
                            {
                                WriteError($"    Error loading ADT file '{Path.GetFileName(obj0AdtPath)}': {adtLoadEx.Message}");
                            }

                            if (adtObj0Data != null)
                            {
                                try
                                {
                                    // --- Map ADT to DTO ---
                                    Console.WriteLine("        Mapping ADT _obj0 data to DTO...");
                                    AdtObj0Dto adtDto = MapAdtToDto(adtObj0Data);

                                    // --- Serialize ADT DTO ---
                                    Console.WriteLine("        Serializing ADT _obj0 DTO...");
                                    string adtYamlOutput = serializer.Serialize(adtDto);
                                    string adtYamlPath = Path.Combine(outputDirectory, $"{pm4BaseName}_obj0.adt.yaml");
                                    File.WriteAllText(adtYamlPath, $"# Dump of {Path.GetFileName(obj0AdtPath)}\n---\n" + adtYamlOutput);
                                    Console.WriteLine($"        -> Saved ADT YAML to {Path.GetFileName(adtYamlPath)}");
                                }
                                catch (Exception adtMapSerializeEx)
                                {
                                    WriteError($"    Error mapping/serializing ADT file '{Path.GetFileName(obj0AdtPath)}': {adtMapSerializeEx.Message}");
                                }
                            }
                        }
                        else
                        {
                            Console.WriteLine($"    Corresponding _obj0.adt not found.");
                        }
                    }
                    catch (Exception pm4MapSerializeEx)
                    {
                        WriteError($"  Error mapping/serializing PM4 file '{pm4FileName}': {pm4MapSerializeEx.Message}");
                    }
                }
            }
        }

        // --- Mappers --- 
        private static Pm4FileDto MapPm4ToDto(PM4File source)
        {
            var dto = new Pm4FileDto();

            // Map MVER
            if (source.MVER != null)
            {
                dto.MVER = new MverChunkDto { Version = source.MVER.Version };
            }

            // Map MSHD (Simplified - Assuming PM4 MSHD doesn't have all ADT fields)
            if (source.MSHD != null)
            {
                dto.MSHD = new MshdChunkDto();
                // TODO: Verify which fields actually exist in Core.MSHDChunk and map them.
                // For now, leaving it empty or mapping only known simple fields if any.
                // Example if Flags existed:
                // dto.MSHD.Flags = source.MSHD.Flags;
            }

            // Map MPRL
            if (source.MPRL != null && source.MPRL.Entries != null)
            {
                dto.MPRL = new MprlChunkDto();
                foreach (var entry in source.MPRL.Entries)
                {
                    // Map C3Vector to System.Numerics.Vector3
                    var positionDto = new Vector3(entry.Position.X, entry.Position.Y, entry.Position.Z);
                    // Assuming Flags doesn't exist or mapping removed based on build error
                    dto.MPRL.Entries.Add(new MprlEntryDto { Position = positionDto /*, Flags = entry.Flags */ });
                }
            }

             // Map MSLK
            if (source.MSLK != null && source.MSLK.Entries != null)
            {
                dto.MSLK = new MslkChunkDto();
                foreach (var entry in source.MSLK.Entries)
                {
                    dto.MSLK.Entries.Add(new MslkEntryDto {
                        Unknown_0x00 = entry.Unknown_0x00, Unknown_0x01 = entry.Unknown_0x01, Unknown_0x02 = entry.Unknown_0x02,
                        Unknown_0x04 = entry.Unknown_0x04, MspiFirstIndex = entry.MspiFirstIndex, MspiIndexCount = entry.MspiIndexCount,
                        Unknown_0x0C = entry.Unknown_0x0C, Unknown_0x10 = entry.Unknown_0x10, Unknown_0x12 = entry.Unknown_0x12
                     });
                }
            }

            // Map MSVT
            if (source.MSVT != null && source.MSVT.Vertices != null)
            {
                dto.MSVT = new MsvtChunkDto();
                foreach (var vertex in source.MSVT.Vertices)
                {
                    dto.MSVT.Vertices.Add(new MsvtVertexDto { X = vertex.X, Y = vertex.Y, Z = vertex.Z });
                }
            }

             // Map MSPV
            if (source.MSPV != null && source.MSPV.Vertices != null)
            {
                dto.MSPV = new MspvChunkDto();
                foreach (var vertex in source.MSPV.Vertices)
                {
                    dto.MSPV.Vertices.Add(new MspvVertexDto { X = vertex.X, Y = vertex.Y, Z = vertex.Z });
                }
            }

             // Map MSPI
            if (source.MSPI != null && source.MSPI.Indices != null)
            {
                dto.MSPI = new MspiChunkDto { Indices = new List<uint>(source.MSPI.Indices) };
            }

            // Map MSVI
            if (source.MSVI != null && source.MSVI.Indices != null)
            {
                dto.MSVI = new MsviChunkDto { Indices = new List<uint>(source.MSVI.Indices) };
            }

            // Map MSUR
            if (source.MSUR != null && source.MSUR.Entries != null)
            {
                dto.MSUR = new MsurChunkDto();
                foreach (var entry in source.MSUR.Entries)
                {
                    dto.MSUR.Entries.Add(new MsurEntryDto {
                        FlagsOrUnknown_0x00 = entry.FlagsOrUnknown_0x00,
                        IndexCount = entry.IndexCount,
                        Unknown_0x02 = entry.Unknown_0x02,
                        MsviFirstIndex = entry.MsviFirstIndex,
                        MdosIndex = entry.MdosIndex
                    });
                }
            }

             // Map MDSF
            if (source.MDSF != null && source.MDSF.Entries != null)
            {
                dto.MDSF = new MdsfChunkDto();
                foreach (var entry in source.MDSF.Entries)
                {
                    dto.MDSF.Entries.Add(new MdsfEntryDto { msur_index = entry.msur_index, mdos_index = entry.mdos_index });
                }
            }

            // Map MDOS
            if (source.MDOS != null && source.MDOS.Entries != null)
            {
                dto.MDOS = new MdosChunkDto();
                foreach (var entry in source.MDOS.Entries)
                {
                    dto.MDOS.Entries.Add(new MdosEntryDto {
                        m_destructible_building_index = entry.m_destructible_building_index,
                        destruction_state = entry.destruction_state
                        // Map other MDOS fields if added to DTO
                    });
                }
            }

            // Map MSCN (Corrected: Just a list of Vector3)
            if (source.MSCN != null && source.MSCN.Vectors != null)
            {
                //dto.MSCN = new MscnChunkDto(); // Removed specific DTO
                // foreach (var vectorData in source.MSCN.Vectors) // No longer iterating complex struct
                // {
                //     dto.MSCN.Vectors.Add(new MscnVectorDataDto {
                //         X = vectorData.X, Y = vectorData.Y, Z = vectorData.Z,
                //         NX = vectorData.NX, NY = vectorData.NY, NZ = vectorData.NZ // These don't exist here
                //     });
                // }
                // If we want to store the raw Vector3 list:
                 dto.MSCN = new MscnChunkDto { Vectors = new List<Vector3>(source.MSCN.Vectors) }; 
            }

             // Map MDBH
            if (source.MDBH != null && source.MDBH.Entries != null)
            {
                dto.MDBH = new MdbhChunkDto();
                foreach (var entry in source.MDBH.Entries)
                {
                    dto.MDBH.Entries.Add(new MdbhEntryDto { Index = entry.Index, Filename = entry.Filename ?? string.Empty });
                }
            }

             // Map MCRR (If exists and needed)
            // if (source.MCRR != null && source.MCRR.Entries != null)
            // {
            //     dto.MCRR = new McrrChunkDto();
            //     // Map entries
            // }

             // Map MPRR
            if (source.MPRR != null && source.MPRR.Entries != null)
            {
                dto.MPRR = new MprrChunkDto();
                foreach (var entry in source.MPRR.Entries)
                {
                    dto.MPRR.Entries.Add(new MprrEntryDto { Unknown_0x00 = entry.Unknown_0x00, Unknown_0x02 = entry.Unknown_0x02 });
                }
            }

            return dto;
        }

        private static AdtObj0Dto MapAdtToDto(TerrainObjectZero source)
        {
            var dto = new AdtObj0Dto();

            // Map MVER
            if (source.Version != null) // Assuming MVER is directly accessible or via a property
            {
                // Assuming MVER chunk structure is the same as PM4's MVER
                 dto.MVER = new MverChunkDto { Version = source.Version.Version }; // Need to confirm property name in Warcraft.NET
            }

            // Map MDDF (Model Placements)
            if (source.ModelPlacementInfo != null && source.ModelPlacementInfo.MDDFEntries != null)
            {
                dto.ModelPlacementInfo = new MddfChunkDto();
                foreach (var entry in source.ModelPlacementInfo.MDDFEntries)
                {
                    // Map Rotator (Pitch, Yaw, Roll) to System.Numerics.Vector3 (X, Y, Z)
                    var rotationDto = new Vector3(entry.Rotation.Pitch, entry.Rotation.Yaw, entry.Rotation.Roll);
                    dto.ModelPlacementInfo.MDDFEntries.Add(new MddfEntryDto {
                        NameId = entry.NameId,
                        UniqueID = entry.UniqueID,
                        Position = entry.Position, // Assuming this is already System.Numerics.Vector3
                        Rotation = rotationDto,
                        Scale = entry.ScalingFactor, // Corrected property name
                        Flags = (ushort)entry.Flags // Explicit cast from MDDFFlags enum
                    });
                }
            }

            // Map MODF (WMO Placements)
            if (source.WorldModelObjectPlacementInfo != null && source.WorldModelObjectPlacementInfo.MODFEntries != null)
            {
                dto.WorldModelObjectPlacementInfo = new ModfChunkDto();
                foreach (var entry in source.WorldModelObjectPlacementInfo.MODFEntries)
                {
                    // Map Rotator (Pitch, Yaw, Roll) to System.Numerics.Vector3 (X, Y, Z)
                    var rotationDto = new Vector3(entry.Rotation.Pitch, entry.Rotation.Yaw, entry.Rotation.Roll);
                    // Map Extents (from BoundingBox.Minimum/Maximum)
                    var extentsMinDto = new Vector3(entry.BoundingBox.Minimum.X, entry.BoundingBox.Minimum.Y, entry.BoundingBox.Minimum.Z);
                    var extentsMaxDto = new Vector3(entry.BoundingBox.Maximum.X, entry.BoundingBox.Maximum.Y, entry.BoundingBox.Maximum.Z);

                    dto.WorldModelObjectPlacementInfo.MODFEntries.Add(new ModfEntryDto {
                        NameId = entry.NameId,
                        UniqueId = (uint)entry.UniqueId, // Cast from int?
                        Position = entry.Position, // Assuming this is already System.Numerics.Vector3
                        Rotation = rotationDto,
                        ExtentsMin = extentsMinDto, // Corrected mapping
                        ExtentsMax = extentsMaxDto, // Corrected mapping
                        Flags = (ushort)entry.Flags, // Explicit cast from MODFFlags enum
                        DoodadSet = entry.DoodadSet,
                        NameSet = entry.NameSet,
                        Scale = entry.Scale // Corrected property name
                    });
                }
            }

            return dto;
        }

        private static void PrintHelp()
        {
             Console.WriteLine("\nUsage: WoWToolbox.FileDumper -d <input_directory> -o <output_directory>");
             Console.WriteLine("\nOptions:");
             Console.WriteLine("  -d, --directory <path>    Required. Directory containing PM4/ADT files to dump.");
             Console.WriteLine("  -o, --output    <path>    Required. Directory where YAML dump files will be saved.");
             Console.WriteLine("  -h, --help                Show this help message.");
        }

        private static void WriteError(string message)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"\nError: {message}");
            Console.ResetColor();
        }
    }
}
