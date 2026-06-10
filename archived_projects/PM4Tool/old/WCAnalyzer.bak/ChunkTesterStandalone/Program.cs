using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using System.Linq;

namespace ChunkTesterStandalone
{
    class Program
    {
        private static readonly Dictionary<string, string> ChunkDescriptions = new Dictionary<string, string>
        {
            // Common chunks
            { "REVM", "Version information" },
            
            // PM4 specific chunks
            { "MCRC", "CRC data (PD4)" },
            { "DHSM", "Header (MSHD)" },
            { "VPSM", "Vertices (MSPV)" },
            { "IPSM", "Indices (MSPI)" },
            { "NCSM", "Normals (MSCN)" },
            { "KLSM", "Mesh layer (MSLK)" },
            { "TVSM", "Vertices (MSVT)" },
            { "IVSM", "Indices (MSVI)" },
            { "RUSM", "Surface (MSUR)" },
            { "LRPM", "Property Record List (MPRL)" },
            { "RRPM", "Property Record (MPRR)" },
            { "HBDM", "Destructible Building Header (MDBH)" },
            { "FBDM", "Destructible Building Filename (MDBF)" },
            { "IBDM", "Destructible Building Index (MDBI)" },
            { "SODM", "Object Storage (MDOS)" },
            { "FSDM", "String Format (MDSF)" },
            
            // ADT specific chunks
            { "RDHM", "ADT Header (MHDR)" },
            { "NICM", "ADT Map Chunk Info (MCIN)" },
            { "XETM", "ADT Textures (MTEX)" },
            { "XDMM", "ADT Model Names (MMDX)" },
            { "DIMM", "ADT Model Indices (MMID)" },
            { "OMWM", "ADT WMO Names (MWMO)" },
            { "DIWM", "ADT WMO Indices (MWID)" },
            { "FDMM", "ADT Doodad Placement (MDDF)" },
            { "FDOM", "ADT WMO Placement (MODF)" },
            { "KNCM", "ADT Map Chunk (MCNK)" },
            { "O2HM", "ADT Water/Liquid (MH2O)" },
            { "OBFM", "ADT Flight Bounds (MFBO)" },
            { "FXTM", "ADT Texture Flags (MTXF)" }
        };

        static int Main(string[] args)
        {
            Console.WriteLine("PM4/PD4/ADT Chunk Tester");
            Console.WriteLine("=========================");
            
            if (args.Length == 0)
            {
                Console.WriteLine("Please provide a file path to a PM4, PD4, or ADT file.");
                return 1;
            }

            string filePath = args[0];
            bool yamlOutput = args.Length > 1 && args[1] == "--yaml";
            string? outputPath = null;
            
            if (args.Length > 2 && args[1] == "--output")
            {
                outputPath = args[2];
            }
            else if (args.Length > 2 && args[2] == "--output")
            {
                outputPath = args[3];
            }
            
            Console.WriteLine($"Looking for file: {filePath}");
            Console.WriteLine($"Working directory: {Directory.GetCurrentDirectory()}");
            
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"File not found: {filePath}");
                return 1;
            }

            try
            {
                // Process the file
                using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
                using (BinaryReader reader = new BinaryReader(fileStream))
                {
                    return ProcessFile(reader, filePath, yamlOutput, outputPath);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing file: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                return 1;
            }
        }

        static int ProcessFile(BinaryReader reader, string filePath, bool outputYaml = false, string outputFile = null)
        {
            Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");
            Console.WriteLine("---------------------------------");

            // Determine file type based on extension
            string fileType = "Unknown";
            string adtSubType = "";

            string fileExtension = Path.GetExtension(filePath).ToLower();
            string fileName = Path.GetFileName(filePath).ToLower();

            if (fileExtension == ".pd4" || fileExtension == ".pm4")
            {
                fileType = fileExtension.Substring(1).ToUpper();
            }
            else if (fileExtension == ".adt")
            {
                fileType = "ADT";
                
                // Determine ADT subtype based on filename pattern
                if (fileName.Contains("_obj"))
                {
                    adtSubType = "Object Data";
                }
                else if (fileName.Contains("_tex"))
                {
                    adtSubType = "Texture Data";
                }
                else
                {
                    adtSubType = "Base Terrain";
                }
                
                Console.WriteLine($"File type: {fileType} ({adtSubType})");
            }
            else
            {
                Console.WriteLine($"Unknown file extension: {fileExtension}");
                return 1;
            }

            var chunks = new List<ChunkInfo>();
            long fileLength = reader.BaseStream.Length;
            long position = 0;
            
            Console.WriteLine($"File size: {fileLength} bytes");

            // Dictionary to hold chunk data for YAML output
            var yamlData = new Dictionary<string, object>();
            yamlData["FileName"] = Path.GetFileName(filePath);
            yamlData["FileType"] = fileType;
            if (adtSubType != "")
            {
                yamlData["FileSubType"] = adtSubType;
            }
            yamlData["FileSize"] = fileLength;
            var chunksData = new List<Dictionary<string, object>>();
            yamlData["Chunks"] = chunksData;

            while (position < fileLength)
            {
                reader.BaseStream.Position = position;
                
                // Make sure we have at least 8 bytes to read (ChunkID + Size)
                if (fileLength - position < 8)
                {
                    Console.WriteLine($"Not enough bytes left to read chunk header at position {position}");
                    break;
                }

                // Read chunk ID (4 bytes)
                byte[] chunkIdBytes = reader.ReadBytes(4);
                string chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                
                // Read chunk size (4 bytes)
                uint chunkSize = reader.ReadUInt32();
                
                Console.WriteLine($"Found chunk {chunkId} at position {position} with size {chunkSize}");
                
                // Check if we can read this chunk's data
                if (position + 8 + chunkSize > fileLength)
                {
                    Console.WriteLine($"Warning: Chunk {chunkId} extends beyond file boundary. Expected size: {chunkSize}, Available bytes: {fileLength - position - 8}");
                    chunkSize = (uint)(fileLength - position - 8);
                }
                
                // Read chunk data
                byte[] chunkData;
                try
                {
                    chunkData = reader.ReadBytes((int)chunkSize);
                    if (chunkData.Length < chunkSize)
                    {
                        Console.WriteLine($"Warning: Could only read {chunkData.Length} bytes of {chunkSize} for chunk {chunkId}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error reading chunk data: {ex.Message}");
                    break;
                }
                
                // Store chunk info
                var chunkInfo = new ChunkInfo
                {
                    Id = chunkId,
                    Size = chunkSize,
                    Offset = position,
                    Data = chunkData
                };
                chunks.Add(chunkInfo);
                
                // Store chunk data for YAML output
                var chunkDataDict = new Dictionary<string, object>
                {
                    ["Id"] = chunkId,
                    ["Size"] = chunkSize,
                    ["Offset"] = position
                };
                
                // Decode chunk data based on ID
                try
                {
                    chunkDataDict["DecodedData"] = DecodeChunkData(chunkInfo, fileType, adtSubType);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error decoding chunk {chunkId}: {ex.Message}");
                    chunkDataDict["DecodedData"] = new { Error = ex.Message };
                }
                chunksData.Add(chunkDataDict);
                
                // Move to the next chunk
                position += 8 + chunkSize;
            }

            // Display chunk information
            Console.WriteLine($"Found {chunks.Count} chunks:");
            Console.WriteLine();
            
            if (outputYaml)
            {
                try
                {
                    // Create YAML serializer
                    var serializer = new SerializerBuilder()
                        .WithNamingConvention(CamelCaseNamingConvention.Instance)
                        .Build();
                    
                    // Serialize to YAML
                    string yaml = serializer.Serialize(yamlData);
                    
                    // Output YAML
                    if (outputFile != null)
                    {
                        File.WriteAllText(outputFile, yaml);
                        Console.WriteLine($"YAML output written to: {outputFile}");
                    }
                    else
                    {
                        Console.WriteLine(yaml);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error serializing to YAML: {ex.Message}");
                }
            }
            else
            {
                // Display standard output
                Console.WriteLine("ID   | Size      | Offset     ");
                Console.WriteLine("-----|-----------|------------");
                
                foreach (var chunk in chunks)
                {
                    Console.WriteLine($"{chunk.Id} | {chunk.Size,10} | {chunk.Offset,10}");
                }
            }
            
            Console.WriteLine();
            Console.WriteLine("Done!");
            return 0;
        }

        private static object DecodeChunkData(ChunkInfo chunk, string fileType = "", string adtSubType = "")
        {
            using (MemoryStream ms = new MemoryStream(chunk.Data))
            using (BinaryReader reader = new BinaryReader(ms))
            {
                switch (chunk.Id)
                {
                    // Common chunks
                    case "REVM": // MVER
                        if (chunk.Size >= 4)
                            return new { Version = reader.ReadUInt32() };
                        break;
                        
                    // ADT-specific chunks with special handling for subtypes
                    case "KNCM": // MCNK - Map Chunk
                        {
                            // For small MCNK chunks (usually in texture files), just return a basic reference note
                            if (chunk.Size <= 32)
                            {
                                return $"MCNK texture reference (size: {chunk.Size})";
                            }

                            // Process the MCNK chunk with its sub-chunks
                            var mcnkData = new Dictionary<string, object>();
                            using (MemoryStream ms = new MemoryStream(chunk.Data))
                            using (BinaryReader reader = new BinaryReader(ms))
                            {
                                try
                                {
                                    // MCNK Header (if we have enough data)
                                    if (ms.Length >= 128)
                                    {
                                        mcnkData["flags"] = reader.ReadUInt32();
                                        mcnkData["indexX"] = reader.ReadUInt32();
                                        mcnkData["indexY"] = reader.ReadUInt32();
                                        mcnkData["layers"] = reader.ReadUInt32(); // Number of texture layers (MCLY)
                                        mcnkData["doodadRefs"] = reader.ReadUInt32(); // Number of doodad references
                                        
                                        // Position data
                                        float x = reader.ReadSingle();
                                        float y = reader.ReadSingle();
                                        float z = reader.ReadSingle();
                                        mcnkData["position"] = new float[] { x, y, z };
                                        
                                        // Convert to world coordinates (based on ADT documentation)
                                        float worldX = 17066.666f - x;
                                        float worldY = 17066.666f - y;
                                        mcnkData["worldPosition"] = new float[] { worldX, worldY, z };
                                        
                                        // Skip the rest of the header
                                        reader.BaseStream.Position = 128;
                                        
                                        // Process sub-chunks
                                        var subChunks = new Dictionary<string, object>();
                                        while (ms.Position < ms.Length - 8)
                                        {
                                            string subChunkId = Encoding.ASCII.GetString(reader.ReadBytes(4));
                                            uint subChunkSize = reader.ReadUInt32();
                                            
                                            if (ms.Position + subChunkSize > ms.Length)
                                            {
                                                subChunks[subChunkId] = $"Error: Sub-chunk extends beyond data boundary ({subChunkSize} bytes)";
                                                break;
                                            }
                                            
                                            byte[] subChunkData = reader.ReadBytes((int)subChunkSize);
                                            
                                            // Process specific sub-chunks
                                            switch (subChunkId)
                                            {
                                                case "TVCM": // MCVT - Height map
                                                    var heightMap = new List<float>();
                                                    using (MemoryStream subMs = new MemoryStream(subChunkData))
                                                    using (BinaryReader subReader = new BinaryReader(subMs))
                                                    {
                                                        // MCVT contains 145 height values (9x9 + 8x8) for the terrain
                                                        int heightCount = 145;
                                                        for (int i = 0; i < heightCount && subMs.Position < subMs.Length; i++)
                                                        {
                                                            heightMap.Add(subReader.ReadSingle());
                                                        }
                                                    }
                                                    subChunks[subChunkId] = new { 
                                                        Description = "Terrain heightmap",
                                                        Values = heightMap.Count > 20 ? 
                                                            new List<float>(heightMap.Take(10).Concat(new[] { 0f }).Concat(heightMap.Skip(heightMap.Count - 10))) :
                                                            heightMap,
                                                        Count = heightMap.Count,
                                                        Min = heightMap.Count > 0 ? heightMap.Min() : 0,
                                                        Max = heightMap.Count > 0 ? heightMap.Max() : 0,
                                                        Average = heightMap.Count > 0 ? heightMap.Average() : 0
                                                    };
                                                    break;
                                                    
                                                case "RNCM": // MCNR - Normal vectors
                                                    var normals = new List<string>();
                                                    using (MemoryStream subMs = new MemoryStream(subChunkData))
                                                    using (BinaryReader subReader = new BinaryReader(subMs))
                                                    {
                                                        // Each normal is 3 bytes (compressed into -127 to +127 range)
                                                        while (subMs.Position < subMs.Length - 2)
                                                        {
                                                            sbyte nx = (sbyte)subReader.ReadByte();
                                                            sbyte ny = (sbyte)subReader.ReadByte();
                                                            sbyte nz = (sbyte)subReader.ReadByte();
                                                            
                                                            // Convert to normalized float vector
                                                            float fnx = nx / 127.0f;
                                                            float fny = ny / 127.0f;
                                                            float fnz = nz / 127.0f;
                                                            
                                                            normals.Add($"({fnx:F2}, {fny:F2}, {fnz:F2})");
                                                            
                                                            // Only store a few normals for YAML output
                                                            if (normals.Count >= 10)
                                                            {
                                                                // Skip to the end and count the remaining normals
                                                                int remaining = (int)((subMs.Length - subMs.Position) / 3);
                                                                subMs.Position = subMs.Length;
                                                                normals.Add($"... {remaining} more normals ...");
                                                                break;
                                                            }
                                                        }
                                                    }
                                                    subChunks[subChunkId] = new { 
                                                        Description = "Terrain normals", 
                                                        Values = normals,
                                                        Count = normals.Count
                                                    };
                                                    break;
                                                    
                                                case "YLCM": // MCLY - Texture layer info
                                                    var layers = new List<Dictionary<string, object>>();
                                                    using (MemoryStream subMs = new MemoryStream(subChunkData))
                                                    using (BinaryReader subReader = new BinaryReader(subMs))
                                                    {
                                                        // Each layer is 16 bytes
                                                        int layerCount = (int)(subChunkSize / 16);
                                                        for (int i = 0; i < layerCount && subMs.Position < subMs.Length - 15; i++)
                                                        {
                                                            var layer = new Dictionary<string, object>
                                                            {
                                                                ["textureId"] = subReader.ReadUInt32(),
                                                                ["flags"] = subReader.ReadUInt32(),
                                                                ["offsetMCAL"] = subReader.ReadUInt32(),
                                                                ["effectId"] = subReader.ReadInt32()
                                                            };
                                                            layers.Add(layer);
                                                        }
                                                    }
                                                    subChunks[subChunkId] = new { 
                                                        Description = "Terrain texture layers", 
                                                        Layers = layers,
                                                        Count = layers.Count
                                                    };
                                                    break;
                                                    
                                                case "LACM": // MCAL - Alpha map
                                                    subChunks[subChunkId] = new { 
                                                        Description = "Alpha map for texture blending", 
                                                        Size = subChunkData.Length
                                                    };
                                                    break;
                                                    
                                                default:
                                                    // Store basic information for other sub-chunks
                                                    subChunks[subChunkId] = new { 
                                                        Size = subChunkData.Length, 
                                                        Description = $"Sub-chunk {subChunkId}"
                                                    };
                                                    break;
                                            }
                                        }
                                        
                                        mcnkData["subChunks"] = subChunks;
                                    }
                                    else
                                    {
                                        return new { Error = $"MCNK data too small ({chunk.Size} bytes) to read header" };
                                    }
                                }
                                catch (Exception ex)
                                {
                                    return new { Error = $"Error processing MCNK: {ex.Message}" };
                                }
                            }
                            
                            return mcnkData;
                        }
                    
                    case "CRCM": // MCRC (PD4)
                        if (chunk.Size >= 4)
                        {
                            return new Dictionary<string, object>
                            {
                                ["_0x00"] = reader.ReadUInt32() // Always 0 in version_48
                            };
                        }
                        break;

                    // PD4 chunks - many are similar to PM4 but with slightly different internal structure
                    // We've already defined these earlier for PM4, so only add the ones that are specific to PD4 or have different structures

                    case "O2HM": // MH2O (ADT - Water/Liquid)
                        {
                            // This is a complex chunk with multiple arrays
                            var data = new Dictionary<string, object>();
                            
                            // Header (256 entries)
                            var headers = new List<Dictionary<string, object>>();
                            for (int i = 0; i < 256 && ms.Position + 16 <= ms.Length; i++)
                            {
                                var header = new Dictionary<string, object>
                                {
                                    ["offset_instances"] = reader.ReadUInt32(),
                                    ["layer_count"] = reader.ReadUInt32(),
                                    ["offset_attributes"] = reader.ReadUInt32()
                                };
                                
                                // The last value is sometimes used differently depending on version
                                header["offset_instance_vertex_data"] = reader.ReadUInt32();
                                
                                headers.Add(header);
                            }
                            
                            data["headers"] = headers;
                            
                            // The rest of the data would need to be parsed by following the offsets
                            // For now, we just return the headers
                            return data;
                        }
                        
                    case "OBFM": // MFBO (ADT - Flight Bounds)
                        {
                            if (chunk.Size >= 24)
                            {
                                uint maxX = reader.ReadUInt32(); // max X coordinate
                                uint maxY = reader.ReadUInt32(); // max Y coordinate
                                uint maxZ = reader.ReadUInt32(); // max Z coordinate
                                uint minX = reader.ReadUInt32(); // min X coordinate
                                uint minY = reader.ReadUInt32(); // min Y coordinate
                                uint minZ = reader.ReadUInt32(); // min Z coordinate
                                
                                return new Dictionary<string, object>
                                {
                                    ["max"] = new Dictionary<string, uint>
                                    {
                                        ["X"] = maxX,
                                        ["Y"] = maxY,
                                        ["Z"] = maxZ
                                    },
                                    ["min"] = new Dictionary<string, uint>
                                    {
                                        ["X"] = minX,
                                        ["Y"] = minY,
                                        ["Z"] = minZ
                                    }
                                };
                            }
                            break;
                        }
                        
                    case "FXTM": // MTXF (ADT - Texture Flags)
                        {
                            var flags = new List<uint>();
                            int flagCount = (int)chunk.Size / 4;
                            
                            for (int i = 0; i < flagCount && ms.Position + 4 <= ms.Length; i++)
                            {
                                flags.Add(reader.ReadUInt32());
                            }
                            
                            return flags;
                        }
                    
                    case "XETM": // MTEX - Texture names
                        {
                            Console.WriteLine("Processing MTEX chunk...");
                            // Debug: Dump the raw MTEX chunk to a file
                            File.WriteAllBytes("debug_mtex_chunk.bin", chunk.Data);
                            List<string> filenames = new List<string>();
                            using (MemoryStream msTexture = new MemoryStream(chunk.Data))
                            using (BinaryReader readerTexture = new BinaryReader(msTexture))
                            {
                                try
                                {
                                    while (msTexture.Position < msTexture.Length)
                                    {
                                        string filename = ReadCString(readerTexture);
                                        if (!string.IsNullOrEmpty(filename))
                                        {
                                            filenames.Add(filename);
                                            Console.WriteLine($"Found texture: {filename}");
                                        }
                                    }
                                    
                                    if (filenames.Count > 0)
                                    {
                                        return new { Textures = filenames };
                                    }
                                    else
                                    {
                                        return new { Error = "No texture filenames found in MTEX chunk" };
                                    }
                                }
                                catch (Exception ex)
                                {
                                    if (filenames.Count > 0)
                                    {
                                        return new { 
                                            Textures = filenames,
                                            Error = $"Error reading some texture names: {ex.Message}"
                                        };
                                    }
                                    else
                                    {
                                        return new { Error = $"Failed to read texture names: {ex.Message}" };
                                    }
                                }
                            }
                        }
                    
                    case "XDMM": // MMDX (ADT)
                        {
                            // Model filenames
                            List<string> filenames = new List<string>();
                            
                            try
                            {
                                // Read all the null-terminated strings
                                while (ms.Position < ms.Length)
                                {
                                    string filename = ReadCString(reader);
                                    if (!string.IsNullOrEmpty(filename))
                                    {
                                        filenames.Add(filename);
                                    }
                                }
                                
                                return filenames;
                            }
                            catch (Exception ex)
                            {
                                // If we encounter an issue, at least return what we've parsed so far
                                if (filenames.Count > 0)
                                {
                                    var result = new Dictionary<string, object>
                                    {
                                        ["models"] = filenames,
                                        ["error"] = ex.Message
                                    };
                                    return result;
                                }
                                
                                // Otherwise, return a simple error message
                                return $"Error decoding MMDX: {ex.Message}";
                            }
                        }
                        
                    case "OMWM": // MWMO (ADT)
                        {
                            // WMO filenames
                            List<string> filenames = new List<string>();
                            
                            try
                            {
                                // Read all the null-terminated strings
                                while (ms.Position < ms.Length)
                                {
                                    string filename = ReadCString(reader);
                                    if (!string.IsNullOrEmpty(filename))
                                    {
                                        filenames.Add(filename);
                                    }
                                }
                                
                                return filenames;
                            }
                            catch (Exception ex)
                            {
                                // If we encounter an issue, at least return what we've parsed so far
                                if (filenames.Count > 0)
                                {
                                    var result = new Dictionary<string, object>
                                    {
                                        ["wmos"] = filenames,
                                        ["error"] = ex.Message
                                    };
                                    return result;
                                }
                                
                                // Otherwise, return a simple error message
                                return $"Error decoding MWMO: {ex.Message}";
                            }
                        }
                    
                    case "DIMM": // MMID (ADT)
                        {
                            var offsets = new List<uint>();
                            int offsetCount = (int)chunk.Size / 4;
                            
                            for (int i = 0; i < offsetCount && ms.Position + 4 <= ms.Length; i++)
                            {
                                offsets.Add(reader.ReadUInt32());
                            }
                            
                            return offsets;
                        }
                        
                    case "DIWM": // MWID (ADT)
                        {
                            var offsets = new List<uint>();
                            int offsetCount = (int)chunk.Size / 4;
                            
                            for (int i = 0; i < offsetCount && ms.Position + 4 <= ms.Length; i++)
                            {
                                offsets.Add(reader.ReadUInt32());
                            }
                            
                            return offsets;
                        }
                    
                    case "NICM": // MCIN - Map Chunk Index
                        {
                            var mcin = new List<Dictionary<string, object>>();
                            using (MemoryStream ms = new MemoryStream(chunk.Data))
                            using (BinaryReader reader = new BinaryReader(ms))
                            {
                                try
                                {
                                    // Each MCIN entry is 16 bytes (4 uint32 values)
                                    int entryCount = chunk.Size / 16;
                                    for (int i = 0; i < entryCount && ms.Position < ms.Length - 15; i++)
                                    {
                                        var entry = new Dictionary<string, object>
                                        {
                                            ["offset"] = reader.ReadUInt32(), // Absolute offset to MCNK chunk
                                            ["size"] = reader.ReadUInt32(),   // Size of the MCNK chunk
                                            ["flags"] = reader.ReadUInt32(),  // Always 0 in file
                                            ["asyncId"] = reader.ReadUInt32() // Used by client only
                                        };
                                        mcin.Add(entry);
                                    }
                                    
                                    // Calculate grid coordinates for each chunk based on its index
                                    for (int i = 0; i < mcin.Count; i++)
                                    {
                                        // Convert chunk index to grid coordinates (16x16 grid)
                                        int gridX = i % 16;
                                        int gridY = i / 16;
                                        mcin[i]["indexX"] = gridX;
                                        mcin[i]["indexY"] = gridY;
                                        
                                        // Calculate world coordinates for this chunk
                                        // Each chunk is 33.33333 yards (100 feet) in size
                                        // Top-left corner of the map is (17066, 17066)
                                        float worldX = 17066.666f - (gridX * 33.33333f);
                                        float worldY = 17066.666f - (gridY * 33.33333f);
                                        mcin[i]["worldCoordsTopLeft"] = new float[] { worldX, worldY };
                                    }
                                }
                                catch (Exception ex)
                                {
                                    return new { Error = $"Error processing MCIN: {ex.Message}" };
                                }
                            }
                            return new { Description = "Map Chunk Index", ChunkCount = mcin.Count, Entries = mcin };
                        }
                        
                    case "FDMM": // MDDF (ADT)
                        {
                            var entries = new List<Dictionary<string, object>>();
                            int entryCount = (int)chunk.Size / 36; // Each MDDF entry is 36 bytes
                            
                            for (int i = 0; i < entryCount && ms.Position + 36 <= ms.Length; i++)
                            {
                                var entry = new Dictionary<string, object>
                                {
                                    ["nameId"] = reader.ReadUInt32(),
                                    ["uniqueId"] = reader.ReadUInt32(),
                                    ["position"] = new Dictionary<string, int>
                                    {
                                        ["X"] = reader.ReadInt32(),
                                        ["Y"] = reader.ReadInt32(),
                                        ["Z"] = reader.ReadInt32()
                                    },
                                    ["rotation"] = new Dictionary<string, int>
                                    {
                                        ["X"] = reader.ReadInt32(),
                                        ["Y"] = reader.ReadInt32(),
                                        ["Z"] = reader.ReadInt32()
                                    },
                                    ["scale"] = reader.ReadUInt16(),
                                    ["flags"] = reader.ReadUInt16()
                                };
                                
                                entries.Add(entry);
                            }
                            
                            return entries;
                        }
                        
                    case "FDOM": // MODF (ADT)
                        {
                            var entries = new List<Dictionary<string, object>>();
                            int entryCount = (int)chunk.Size / 64; // Each MODF entry is 64 bytes
                            
                            for (int i = 0; i < entryCount && ms.Position + 64 <= ms.Length; i++)
                            {
                                var entry = new Dictionary<string, object>
                                {
                                    ["nameId"] = reader.ReadUInt32(),
                                    ["uniqueId"] = reader.ReadUInt32(),
                                    ["position"] = new Dictionary<string, float>
                                    {
                                        ["X"] = reader.ReadSingle(),
                                        ["Y"] = reader.ReadSingle(),
                                        ["Z"] = reader.ReadSingle()
                                    },
                                    ["rotation"] = new Dictionary<string, float>
                                    {
                                        ["X"] = reader.ReadSingle(),
                                        ["Y"] = reader.ReadSingle(),
                                        ["Z"] = reader.ReadSingle()
                                    },
                                    ["extents"] = new Dictionary<string, object>
                                    {
                                        ["min"] = new Dictionary<string, float>
                                        {
                                            ["X"] = reader.ReadSingle(),
                                            ["Y"] = reader.ReadSingle(),
                                            ["Z"] = reader.ReadSingle()
                                        },
                                        ["max"] = new Dictionary<string, float>
                                        {
                                            ["X"] = reader.ReadSingle(),
                                            ["Y"] = reader.ReadSingle(),
                                            ["Z"] = reader.ReadSingle()
                                        }
                                    },
                                    ["flags"] = reader.ReadUInt16(),
                                    ["doodadSet"] = reader.ReadUInt16(),
                                    ["nameSet"] = reader.ReadUInt16(),
                                    ["scale"] = reader.ReadUInt16()
                                };
                                
                                entries.Add(entry);
                            }
                            
                            return entries;
                        }
                    
                    case "RDHM": // MHDR - Map Header
                        {
                            var mhdr = new Dictionary<string, object>();
                            using (MemoryStream ms = new MemoryStream(chunk.Data))
                            using (BinaryReader reader = new BinaryReader(ms))
                            {
                                try
                                {
                                    // MHDR contains offsets to other chunks
                                    mhdr["flags"] = reader.ReadUInt32();
                                    
                                    // Store offsets to various chunks - these are relative to the start of the MHDR data
                                    Dictionary<string, uint> offsets = new Dictionary<string, uint>();
                                    offsets["mcin"] = reader.ReadUInt32(); // MCIN (Map Chunk Index)
                                    offsets["mtex"] = reader.ReadUInt32(); // MTEX (Texture list)
                                    offsets["mmdx"] = reader.ReadUInt32(); // MMDX (Model filenames)
                                    offsets["mmid"] = reader.ReadUInt32(); // MMID (Model filename offsets)
                                    offsets["mwmo"] = reader.ReadUInt32(); // MWMO (WMO filenames)
                                    offsets["mwid"] = reader.ReadUInt32(); // MWID (WMO filename offsets)
                                    offsets["mddf"] = reader.ReadUInt32(); // MDDF (Doodad placement)
                                    offsets["modf"] = reader.ReadUInt32(); // MODF (WMO placement)
                                    
                                    // In newer versions, these might be present
                                    if (ms.Position + 4 <= ms.Length)
                                    {
                                        offsets["mfbo"] = reader.ReadUInt32(); // MFBO (Fog)
                                    }
                                    
                                    if (ms.Position + 4 <= ms.Length)
                                    {
                                        offsets["mh2o"] = reader.ReadUInt32(); // MH2O (Water)
                                    }
                                    
                                    if (ms.Position + 4 <= ms.Length)
                                    {
                                        offsets["mtxf"] = reader.ReadUInt32(); // MTXF (Texture flags)
                                    }
                                    
                                    mhdr["offsets"] = offsets;
                                    
                                    // Calculate actual file offsets for each referenced chunk
                                    // by adding the offset to the start of the MHDR data
                                    Dictionary<string, uint> fileOffsets = new Dictionary<string, uint>();
                                    foreach (var kvp in offsets)
                                    {
                                        if (kvp.Value > 0)
                                        {
                                            // If MHDR is at offset X, and chunk.offset points to offset Y relative to MHDR data,
                                            // then the actual file offset is X + 8 + Y
                                            fileOffsets[kvp.Key] = (uint)(args.FileOffset + 8 + kvp.Value);
                                        }
                                    }
                                    mhdr["fileOffsets"] = fileOffsets;
                                    
                                    // Parse flags
                                    List<string> flagNames = new List<string>();
                                    if ((mhdr["flags"] is uint flags) && (flags & 0x1) != 0)
                                    {
                                        flagNames.Add("MFBO");
                                    }
                                    if ((mhdr["flags"] is uint flags2) && (flags2 & 0x2) != 0)
                                    {
                                        flagNames.Add("Northrend");
                                    }
                                    mhdr["flagNames"] = flagNames;
                                }
                                catch (Exception ex)
                                {
                                    return new { Error = $"Error processing MHDR: {ex.Message}" };
                                }
                            }
                            return mhdr;
                        }
                    
                    case "FDDM": // MDDF - Model Placement
                        {
                            var mddf = new List<Dictionary<string, object>>();
                            using (MemoryStream ms = new MemoryStream(chunk.Data))
                            using (BinaryReader reader = new BinaryReader(ms))
                            {
                                try
                                {
                                    // Each MDDF entry is 36 bytes
                                    int entryCount = chunk.Size / 36;
                                    for (int i = 0; i < entryCount && ms.Position < ms.Length - 35; i++)
                                    {
                                        var entry = new Dictionary<string, object>
                                        {
                                            ["nameId"] = reader.ReadUInt32(),        // References an entry in the MMID chunk
                                            ["uniqueId"] = reader.ReadUInt32(),      // Unique ID for this instance
                                            ["position"] = new float[]               // Position vector
                                            {
                                                reader.ReadSingle(),                 // X
                                                reader.ReadSingle(),                 // Y
                                                reader.ReadSingle()                  // Z
                                            },
                                            ["rotation"] = new float[]               // Rotation vector (degrees)
                                            {
                                                reader.ReadSingle(),                 // X
                                                reader.ReadSingle(),                 // Y
                                                reader.ReadSingle()                  // Z
                                            },
                                            ["scale"] = reader.ReadUInt16() / 1024f, // 1024 = 1.0 scale
                                            ["flags"] = reader.ReadUInt16()          // Flags from enum MDDFFlags
                                        };
                                        
                                        // Calculate world coordinates
                                        if (entry["position"] is float[] pos)
                                        {
                                            entry["worldPosition"] = new float[]
                                            {
                                                32 * 533.33333f - pos[0],  // X coordinate conversion
                                                pos[1],                     // Y coordinate stays the same
                                                32 * 533.33333f - pos[2]    // Z coordinate conversion
                                            };
                                        }
                                        
                                        // Parse flags
                                        if (entry["flags"] is ushort flags)
                                        {
                                            var flagNames = new List<string>();
                                            if ((flags & 0x1) != 0) flagNames.Add("Biodome");
                                            if ((flags & 0x2) != 0) flagNames.Add("Shrubbery");
                                            if ((flags & 0x4) != 0) flagNames.Add("Unknown_4");
                                            if ((flags & 0x8) != 0) flagNames.Add("Unknown_8");
                                            if ((flags & 0x10) != 0) flagNames.Add("Unknown_10");
                                            if ((flags & 0x20) != 0) flagNames.Add("LiquidKnown");
                                            if ((flags & 0x40) != 0) flagNames.Add("EntryIsFileDataId");
                                            if ((flags & 0x100) != 0) flagNames.Add("Unknown_100");
                                            if ((flags & 0x1000) != 0) flagNames.Add("AcceptProjTextures");
                                            entry["flagNames"] = flagNames;
                                        }
                                        
                                        mddf.Add(entry);
                                    }
                                }
                                catch (Exception ex)
                                {
                                    return new { Error = $"Error processing MDDF: {ex.Message}" };
                                }
                            }
                            return new { Description = "Doodad (M2) Placement", EntryCount = mddf.Count, Entries = mddf };
                        }
                        
                    case "FDOM": // MODF - WMO Placement
                        {
                            var modf = new List<Dictionary<string, object>>();
                            using (MemoryStream ms = new MemoryStream(chunk.Data))
                            using (BinaryReader reader = new BinaryReader(ms))
                            {
                                try
                                {
                                    // Each MODF entry is 64 bytes
                                    int entryCount = chunk.Size / 64;
                                    for (int i = 0; i < entryCount && ms.Position < ms.Length - 63; i++)
                                    {
                                        var entry = new Dictionary<string, object>
                                        {
                                            ["nameId"] = reader.ReadUInt32(),     // References an entry in the MWID chunk
                                            ["uniqueId"] = reader.ReadUInt32(),   // Unique ID for this instance
                                            ["position"] = new float[]            // Position vector
                                            {
                                                reader.ReadSingle(),              // X
                                                reader.ReadSingle(),              // Y
                                                reader.ReadSingle()               // Z
                                            },
                                            ["rotation"] = new float[]            // Rotation vector (degrees)
                                            {
                                                reader.ReadSingle(),              // X
                                                reader.ReadSingle(),              // Y
                                                reader.ReadSingle()               // Z
                                            },
                                            ["lowerBounds"] = new float[]         // Bounding box - lower corner
                                            {
                                                reader.ReadSingle(),              // X
                                                reader.ReadSingle(),              // Y
                                                reader.ReadSingle()               // Z
                                            },
                                            ["upperBounds"] = new float[]         // Bounding box - upper corner
                                            {
                                                reader.ReadSingle(),              // X
                                                reader.ReadSingle(),              // Y
                                                reader.ReadSingle()               // Z
                                            },
                                            ["flags"] = reader.ReadUInt16(),      // Flags
                                            ["doodadSet"] = reader.ReadUInt16(),  // WMO doodad set index
                                            ["nameSet"] = reader.ReadUInt16(),    // WMO name set index
                                            ["padding"] = reader.ReadUInt16()     // Always 0
                                        };
                                        
                                        // Calculate world coordinates
                                        if (entry["position"] is float[] pos)
                                        {
                                            entry["worldPosition"] = new float[]
                                            {
                                                32 * 533.33333f - pos[0],  // X coordinate conversion
                                                pos[1],                     // Y coordinate stays the same
                                                32 * 533.33333f - pos[2]    // Z coordinate conversion
                                            };
                                        }
                                        
                                        // Parse flags
                                        if (entry["flags"] is ushort flags)
                                        {
                                            var flagNames = new List<string>();
                                            if ((flags & 0x1) != 0) flagNames.Add("DestroyOnOverwrite");
                                            if ((flags & 0x2) != 0) flagNames.Add("UseLod");
                                            if ((flags & 0x8) != 0) flagNames.Add("EntryIsFileDataId");
                                            entry["flagNames"] = flagNames;
                                        }
                                        
                                        modf.Add(entry);
                                    }
                                }
                                catch (Exception ex)
                                {
                                    return new { Error = $"Error processing MODF: {ex.Message}" };
                                }
                            }
                            return new { Description = "WMO Placement", EntryCount = modf.Count, Entries = modf };
                        }
                    
                    default:
                        // For unhandled chunk types, return simplified binary representation
                        if (chunk.Size <= 100)
                        {
                            // For small chunks, return hex representation
                            return BitConverter.ToString(chunk.Data);
                        }
                        else
                        {
                            // For larger chunks, just return size info
                            return new { DataSize = chunk.Size, Note = "Large binary data" };
                        }
                }
            }
            
            return new { Note = "Unable to decode data" };
        }

        private class ChunkInfo
        {
            public string Id { get; set; } = string.Empty;
            public uint Size { get; set; }
            public long Offset { get; set; }
            public byte[] Data { get; set; } = Array.Empty<byte>();
        }

        private static string ReadCString(BinaryReader reader)
        {
            List<byte> stringBytes = new List<byte>();
            
            try
            {
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    byte b = reader.ReadByte();
                    if (b == 0) break;
                    stringBytes.Add(b);
                }
                
                if (stringBytes.Count > 0)
                {
                    return Encoding.ASCII.GetString(stringBytes.ToArray());
                }
            }
            catch (IOException)
            {
                // End of stream reached
            }
            
            return string.Empty;
        }
    }
} 