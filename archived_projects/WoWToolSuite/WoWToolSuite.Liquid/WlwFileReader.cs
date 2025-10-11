using System;
using System.IO;
using System.Collections.Generic;
using System.Numerics;
using System.Text;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid
{
    public class WlwFileReader
    {
        public List<LiquidBlock> ReadWlwFile(string filePath)
        {
            var blocks = new List<LiquidBlock>();
            
            using (var br = new BinaryReader(File.OpenRead(filePath)))
            {
                // Read header - don't validate magic number, just store it
                var magic = Encoding.UTF8.GetString(br.ReadBytes(4));
                Console.WriteLine($"Magic string: {magic}");
                
                // Read the rest of the header
                var version = br.ReadUInt16();
                var unk06 = br.ReadUInt16(); // Always 1
                var liquidType = br.ReadUInt16();
                var padding = br.ReadUInt16();
                var blockCount = br.ReadUInt32();

                Console.WriteLine($"Version: {version}, Unk06: {unk06}, LiquidType: {liquidType}, BlockCount: {blockCount}");

                if (version > 2)
                {
                    throw new InvalidDataException($"Unsupported WLW version: {version}");
                }

                // Read blocks
                for (int i = 0; i < blockCount; i++)
                {
                    var block = new LiquidBlock();
                    block.LiquidType = liquidType;

                    // Read 16 vertices (C3Vector - 3 floats each)
                    for (int v = 0; v < 16; v++)
                    {
                        float x = br.ReadSingle();
                        float y = br.ReadSingle();
                        float z = br.ReadSingle();
                        block.Vertices.Add(new Vector3(x, y, z));
                    }

                    // Read internal coordinates (C2Vector - 2 floats)
                    var coordX = br.ReadSingle();
                    var coordY = br.ReadSingle();
                    block.Coord = new Vector2(coordX, coordY);

                    // Read additional data
                    for (int d = 0; d < 0x50; d++)
                    {
                        block.Data[d] = br.ReadUInt16();
                    }

                    // Fill height map from vertices (arranged in a 4x4 grid, starting from lower right)
                    for (int h = 0; h < 4; h++)
                    {
                        for (int w = 0; w < 4; w++)
                        {
                            // Convert from lower-right to upper-left ordering
                            int vertexIndex = (3 - h) * 4 + (3 - w);
                            block.Heights[h][w] = block.Vertices[vertexIndex].Z;
                        }
                    }

                    blocks.Add(block);
                }
            }

            return blocks;
        }
    }
} 