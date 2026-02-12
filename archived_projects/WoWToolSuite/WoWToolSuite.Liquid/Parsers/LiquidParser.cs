using System;
using System.IO;
using System.Text;
using System.Numerics;
using System.Collections.Generic;
using System.Linq;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid.Parsers
{
    public class LiquidFile
    {
        public List<LiquidBlock> Blocks { get; set; } = new List<LiquidBlock>();
        public required LiquidHeader Header { get; set; }
        public required string FilePath { get; set; }
        public bool IsWlm { get; set; }
    }

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
                    block.LiquidType = header.LiquidType;

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

                    // Fill height map from vertices (arranged in a 4x4 grid)
                    for (int h = 0; h < 4; h++)
                    {
                        for (int w = 0; w < 4; w++)
                        {
                            block.Heights[h][w] = block.Vertices[h * 4 + w].Z;
                        }
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

        public static LiquidFile? ParseWlqFile(string filePath, bool verbose = false)
        {
            if (verbose)
            {
                Console.WriteLine($"Parsing WLQ file: {filePath}");
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
                    LiquidType = reader.ReadUInt16(),
                    Padding = reader.ReadUInt16(),
                    BlockCount = reader.ReadUInt32()
                };

                var result = new LiquidFile
                {
                    Header = header,
                    FilePath = filePath,
                    IsWlm = false
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
                    block.LiquidType = header.LiquidType;

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

                    // Skip blocks with coordinates beyond threshold
                    if (block.Coord.X > LiquidConstants.COORDINATE_THRESHOLD || block.Coord.Y > LiquidConstants.COORDINATE_THRESHOLD)
                    {
                        continue;
                    }

                    // Read block data (80 ushorts)
                    for (int j = 0; j < 80; j++)
                    {
                        block.Data[j] = reader.ReadUInt16();
                    }

                    // Fill height map from vertices (arranged in a 4x4 grid)
                    for (int h = 0; h < 4; h++)
                    {
                        for (int w = 0; w < 4; w++)
                        {
                            block.Heights[h][w] = block.Vertices[h * 4 + w].Z;
                        }
                    }

                    result.Blocks.Add(block);
                }

                if (verbose)
                {
                    Console.WriteLine($"Finished parsing WLQ file: {filePath}");
                }

                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing WLQ file {filePath}: {ex.Message}");
                return null;
            }
        }
    }
} 