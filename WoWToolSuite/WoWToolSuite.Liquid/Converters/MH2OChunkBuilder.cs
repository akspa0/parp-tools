using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid.Converters
{
    public class MH2OChunkBuilder
    {
        private const int CHUNK_SIZE = 16;
        private const int CHUNKS_PER_MAP_SIDE = 16;
        private const int TOTAL_CHUNKS = 256;
        private const int WLW_GRID_SIZE = 4;
        private const int MH2O_VERTEX_GRID_SIZE = 9;
        private const int MH2O_RENDER_GRID_SIZE = 8;

        public Warcraft.NET.Files.ADT.Chunks.MH2O BuildChunk(LiquidBlock[] liquidBlocks, bool verbose = false)
        {
            if (verbose)
            {
                Console.WriteLine($"Building MH2O chunk for {liquidBlocks.Length} liquid blocks");
            }

            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);

            // Write chunk header
            bw.Write(Encoding.ASCII.GetBytes("MH2O")); // Magic number
            var sizePos = ms.Position;
            bw.Write(0); // Placeholder for size

            // Write header array (256 entries)
            var headerStartPos = ms.Position;
            for (int i = 0; i < TOTAL_CHUNKS; i++)
            {
                bw.Write(0U); // OffsetInstances
                bw.Write(0U); // LayerCount
                bw.Write(0U); // OffsetAttributes
            }

            // Group blocks by chunk coordinates
            var blocksByChunk = liquidBlocks
                .GroupBy(b => ((int)(b.GlobalX / CHUNK_SIZE) % CHUNKS_PER_MAP_SIDE) + ((int)(b.GlobalY / CHUNK_SIZE) % CHUNKS_PER_MAP_SIDE) * CHUNKS_PER_MAP_SIDE)
                .ToDictionary(g => g.Key, g => g.ToList());

            // Process each chunk that has water
            foreach (var kvp in blocksByChunk)
            {
                var chunkIndex = kvp.Key;
                var blocks = kvp.Value;

                // Seek to header position for this chunk
                ms.Position = headerStartPos + (chunkIndex * 12); // 12 bytes per header

                // Write instance offset
                var instanceOffset = ms.Length;
                bw.Write((uint)instanceOffset);
                bw.Write((uint)blocks.Count); // LayerCount
                bw.Write(0U); // No attributes for now

                // Save position
                var returnPos = ms.Position;

                // Write instances
                ms.Position = instanceOffset;
                
                foreach (var block in blocks)
                {
                    // Save instance start for offset calculations
                    var instanceStart = ms.Position;
                    
                    // Instance header
                    bw.Write((ushort)block.LiquidType); // LiquidTypeId
                    
                    // Determine vertex format based on liquid type
                    ushort vertexFormat = DetermineVertexFormat(block.LiquidType);
                    bw.Write(vertexFormat);
                    
                    bw.Write(block.MinHeight); // MinHeightLevel
                    bw.Write(block.MaxHeight); // MaxHeightLevel
                    bw.Write((byte)0); // OffsetX
                    bw.Write((byte)0); // OffsetY
                    bw.Write((byte)MH2O_VERTEX_GRID_SIZE); // Width (9)
                    bw.Write((byte)MH2O_VERTEX_GRID_SIZE); // Height (9)
                    
                    // Placeholder for offsets - we'll come back and update these
                    var offsetBitmapPos = ms.Position;
                    bw.Write(0U); // OffsetBitmap placeholder
                    var offsetVertexDataPos = ms.Position;
                    bw.Write(0U); // OffsetVertexData placeholder
                    
                    // Write render bitmap (all enabled)
                    var bitmapPos = ms.Position;
                    for (int i = 0; i < MH2O_RENDER_GRID_SIZE; i++)
                    {
                        bw.Write((byte)0xFF);
                    }
                    
                    // Write vertex data
                    var vertexDataPos = ms.Position;
                    
                    // Write vertex data based on format
                    switch (vertexFormat)
                    {
                        case 0: // Height + Depth
                            WriteHeightMap(bw, block);
                            WriteDepthMap(bw, block);
                            break;
                        case 1: // Height + UV
                            WriteHeightMap(bw, block);
                            WriteUVMap(bw, block);
                            break;
                        case 2: // Depth only
                            WriteDepthMap(bw, block);
                            break;
                        case 3: // Height + UV + Depth
                            WriteHeightMap(bw, block);
                            WriteUVMap(bw, block);
                            WriteDepthMap(bw, block);
                            break;
                    }
                    
                    // Now go back and update the offsets with actual values
                    var currentPos = ms.Position;
                    
                    // Update bitmap offset (relative to start of instance)
                    ms.Position = offsetBitmapPos;
                    bw.Write((uint)(bitmapPos - instanceStart));
                    
                    // Update vertex data offset (relative to start of instance)
                    ms.Position = offsetVertexDataPos;
                    bw.Write((uint)(vertexDataPos - instanceStart));
                    
                    // Return to end position to continue writing
                    ms.Position = currentPos;
                }

                // Restore position for next header
                ms.Position = returnPos;
            }

            // Write final size
            var endPos = ms.Position;
            ms.Position = sizePos;
            bw.Write((uint)(endPos - sizePos - 4));

            // Return the MH2O chunk
            return new Warcraft.NET.Files.ADT.Chunks.MH2O(ms.ToArray());
        }

        private ushort DetermineVertexFormat(int liquidType)
        {
            // Default to format 0 (height + depth) for most liquid types
            switch (liquidType)
            {
                case 2: // Magma/Lava
                    return 2; // Depth map only
                case 3: // Slime
                case 4: // River
                    return 1; // Height + UV
                default:
                    return 0; // Height + depth
            }
        }

        private void WriteHeightMap(BinaryWriter bw, LiquidBlock block)
        {
            // Interpolate from 4x4 to 9x9 grid
            for (int y = 0; y < MH2O_VERTEX_GRID_SIZE; y++)
            {
                for (int x = 0; x < MH2O_VERTEX_GRID_SIZE; x++)
                {
                    // Calculate fractional position in the original 4x4 grid
                    float xFrac = x * (WLW_GRID_SIZE - 1) / (float)(MH2O_VERTEX_GRID_SIZE - 1);
                    float yFrac = y * (WLW_GRID_SIZE - 1) / (float)(MH2O_VERTEX_GRID_SIZE - 1);
                    
                    // Integer and fractional parts
                    int xLow = (int)xFrac;
                    int yLow = (int)yFrac;
                    float xWeight = xFrac - xLow;
                    float yWeight = yFrac - yLow;
                    
                    // Ensure we don't go out of bounds
                    xLow = Math.Min(xLow, WLW_GRID_SIZE - 2);
                    yLow = Math.Min(yLow, WLW_GRID_SIZE - 2);
                    
                    // Bilinear interpolation
                    float h00 = block.Heights[yLow][xLow];
                    float h10 = block.Heights[yLow][xLow + 1];
                    float h01 = block.Heights[yLow + 1][xLow];
                    float h11 = block.Heights[yLow + 1][xLow + 1];
                    
                    float h0 = h00 * (1 - xWeight) + h10 * xWeight;
                    float h1 = h01 * (1 - xWeight) + h11 * xWeight;
                    
                    float interpolatedHeight = h0 * (1 - yWeight) + h1 * yWeight;
                    
                    // Write interpolated height
                    bw.Write(interpolatedHeight);
                }
            }
        }

        private void WriteDepthMap(BinaryWriter bw, LiquidBlock block)
        {
            // Write a full 9x9 grid of depth values
            for (int y = 0; y < MH2O_VERTEX_GRID_SIZE; y++)
            {
                for (int x = 0; x < MH2O_VERTEX_GRID_SIZE; x++)
                {
                    bw.Write((byte)255); // Full depth
                }
            }
        }

        private void WriteUVMap(BinaryWriter bw, LiquidBlock block)
        {
            // Write a full 9x9 grid of UV coordinates
            for (int y = 0; y < MH2O_VERTEX_GRID_SIZE; y++)
            {
                for (int x = 0; x < MH2O_VERTEX_GRID_SIZE; x++)
                {
                    // Write U coordinate (0-1 range)
                    bw.Write((float)x / (MH2O_VERTEX_GRID_SIZE - 1));
                    // Write V coordinate (0-1 range)
                    bw.Write((float)y / (MH2O_VERTEX_GRID_SIZE - 1));
                }
            }
        }

        public void BuildAndSaveChunk(string outputPath, LiquidBlock[] liquidBlocks, bool verbose = false)
        {
            if (verbose)
            {
                Console.WriteLine($"Building and saving MH2O chunk to {outputPath}");
            }

            var mh2oChunk = BuildChunk(liquidBlocks, verbose);
            
            // Save to file
            var directory = Path.GetDirectoryName(outputPath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
            File.WriteAllBytes(outputPath, mh2oChunk.Serialize());

            if (verbose)
            {
                Console.WriteLine($"Saved MH2O chunk to {outputPath}");
            }
        }
    }
} 