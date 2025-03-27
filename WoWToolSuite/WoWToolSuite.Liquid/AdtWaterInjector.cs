using System;
using System.IO;
using System.Linq;
using System.Text;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Files.ADT.Entries;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using WowToolSuite.Liquid.Coordinates;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid
{
    public class AdtWaterInjector
    {
        private const int CHUNK_SIZE = 16;
        private const int CHUNKS_PER_MAP_SIDE = 16;
        private const int MH2O_VERTEX_GRID_SIZE = 9;

        public bool InjectWaterIntoAdt(string adtPath, string outputPath, LiquidBlock[] liquidBlocks)
        {
            if (adtPath.Contains("_obj0") || adtPath.Contains("_tex0"))
            {
                Console.WriteLine($"Skipping {adtPath} as it's an _obj0 or _tex0 file");
                return false;
            }

            if (!File.Exists(adtPath))
            {
                Console.WriteLine($"ADT file {adtPath} does not exist, skipping");
                return false;
            }

            try
            {
                // Create output directory if it doesn't exist
                string? directoryName = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrEmpty(directoryName))
                {
                    Directory.CreateDirectory(directoryName);
                }

                // Build our MH2O chunk data (only the raw data, no header)
                var waterData = new MH2OOld();
                waterData.MH2OHeaders = new MH2OHeader[256];
                for (int i = 0; i < 256; i++)
                {
                    waterData.MH2OHeaders[i] = new MH2OHeader(new byte[MH2OHeader.GetSize()]);
                }

                // Process each water block
                foreach (var liquidBlock in liquidBlocks)
                {
                    // Calculate ADT-local coordinates
                    var chunkCoordinates = CalculateChunkCoordinates(liquidBlock);
                    var chunkIndex = chunkCoordinates.X + chunkCoordinates.Y * CHUNKS_PER_MAP_SIDE;

                    var header = waterData.MH2OHeaders[chunkIndex];
                    if (header.LayerCount == 0)
                    {
                        header.Instances = new MH2OInstance[1];
                        header.LayerCount = 1;
                        header.Instances[0] = new MH2OInstance(new byte[MH2OInstance.GetSize()]);
                    }

                    var instance = header.Instances[0];
                    instance.LiquidTypeId = (ushort)liquidBlock.LiquidType;
                    instance.MinHeightLevel = liquidBlock.MinHeight;
                    instance.MaxHeightLevel = liquidBlock.MaxHeight;
                    instance.Width = MH2O_VERTEX_GRID_SIZE;
                    instance.Height = MH2O_VERTEX_GRID_SIZE;

                    // Create vertex data
                    instance.VertexData = new MH2OInstanceVertexData();
                    instance.VertexData.HeightMap = new float[MH2O_VERTEX_GRID_SIZE, MH2O_VERTEX_GRID_SIZE];
                    instance.RenderBitmapBytes = new byte[8];

                    // Set all vertices to enabled
                    for (int i = 0; i < 8; i++)
                    {
                        instance.RenderBitmapBytes[i] = 0xFF;
                    }

                    // Set height values
                    for (int y = 0; y <= 8; y++)
                    {
                        for (int x = 0; x <= 8; x++)
                        {
                            instance.VertexData.HeightMap[y, x] = liquidBlock.MinHeight + 
                                (liquidBlock.MaxHeight - liquidBlock.MinHeight) * (y / 8.0f);
                        }
                    }
                }

                // Read the ADT file
                byte[] adtData = File.ReadAllBytes(adtPath);
                
                // Serialize the MH2O data with the O2HM header
                using (var ms = new MemoryStream())
                using (var bw = new BinaryWriter(ms))
                {
                    // Write the O2HM header (WoTLK format)
                    bw.Write(Encoding.ASCII.GetBytes("O2HM"));
                    
                    // Get the raw chunk data (without the MH2O header)
                    byte[] rawMh2oData = waterData.Serialize().Skip(8).ToArray();
                    
                    // Write the size of the raw data
                    bw.Write((uint)rawMh2oData.Length);
                    
                    // Write the actual data
                    bw.Write(rawMh2oData);
                    
                    var mh2oChunkWithHeader = ms.ToArray();
                    
                    // Process the ADT file and inject the MH2O chunk
                    using (var outputStream = new MemoryStream())
                    using (var writer = new BinaryWriter(outputStream))
                    {
                        // Track if we've injected MH2O
                        bool mh2oInjected = false;
                        bool mhdrFound = false;
                        long mhdrPos = 0;
                        long mhdrSizePos = 0;
                        byte[] mhdrData = new byte[0];
                        
                        using (var reader = new BinaryReader(new MemoryStream(adtData)))
                        {
                            // First, ensure we have a valid MVER chunk
                            if (!IsValidHeader(reader, "REVM"))
                            {
                                Console.WriteLine($"Error: {adtPath} does not have a valid MVER chunk");
                                File.Copy(adtPath, outputPath, true);
                                return false;
                            }
                            
                            // Copy MVER chunk
                            uint mverSize = reader.ReadUInt32();
                            byte[] mverData = reader.ReadBytes((int)mverSize);
                            writer.Write(Encoding.ASCII.GetBytes("REVM"));
                            writer.Write(mverSize);
                            writer.Write(mverData);
                            
                            // Process the remaining chunks
                            while (reader.BaseStream.Position < reader.BaseStream.Length)
                            {
                                // Read chunk header
                                if (!TryReadHeader(reader, out string chunkId, out uint chunkSize))
                                {
                                    break; // End of file or bad data
                                }
                                
                                if (chunkId == "RDHM" && !mhdrFound)
                                {
                                    // Save MHDR position for later update
                                    mhdrFound = true;
                                    mhdrPos = writer.BaseStream.Position - 8; // Start of MHDR
                                    mhdrSizePos = writer.BaseStream.Position - 4; // Size field
                                    
                                    // Read original MHDR data
                                    mhdrData = reader.ReadBytes((int)chunkSize);
                                    
                                    // Write the MHDR header
                                    writer.Write(Encoding.ASCII.GetBytes("RDHM"));
                                    writer.Write(chunkSize);
                                    writer.Write(mhdrData);
                                }
                                else if ((chunkId == "O2HM" || chunkId == "MH2O") && !mh2oInjected)
                                {
                                    // Skip the existing water chunk
                                    reader.BaseStream.Seek(chunkSize, SeekOrigin.Current);
                                    
                                    // Write our new MH2O chunk
                                    writer.Write(mh2oChunkWithHeader);
                                    mh2oInjected = true;
                                    
                                    Console.WriteLine($"Replaced existing water chunk in {Path.GetFileName(adtPath)}");
                                }
                                else if (chunkId == "KNCM" && !mh2oInjected)
                                {
                                    // Before first MCNK and if MH2O not added yet, add it here
                                    // Write our MH2O chunk first
                                    writer.Write(mh2oChunkWithHeader);
                                    mh2oInjected = true;
                                    
                                    // Then continue with the MCNK chunk
                                    writer.Write(Encoding.ASCII.GetBytes(chunkId));
                                    writer.Write(chunkSize);
                                    writer.Write(reader.ReadBytes((int)chunkSize));
                                    
                                    Console.WriteLine($"Added water chunk before MCNK in {Path.GetFileName(adtPath)}");
                                }
                                else
                                {
                                    // Copy chunk as is
                                    writer.Write(Encoding.ASCII.GetBytes(chunkId));
                                    writer.Write(chunkSize);
                                    writer.Write(reader.ReadBytes((int)chunkSize));
                                }
                            }
                            
                            // If we haven't added MH2O yet, add it at the end
                            if (!mh2oInjected)
                            {
                                writer.Write(mh2oChunkWithHeader);
                                mh2oInjected = true;
                                Console.WriteLine($"Added water chunk at the end of {Path.GetFileName(adtPath)}");
                            }
                        }
                        
                        // Write the updated ADT file
                        File.WriteAllBytes(outputPath, outputStream.ToArray());
                    }
                }
                
                Console.WriteLine($"Successfully patched ADT file: {Path.GetFileName(adtPath)}");
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing ADT {adtPath}: {ex.Message}");
                
                try
                {
                    // Just copy the file without modifying it
                    string? directoryName = Path.GetDirectoryName(outputPath);
                    if (!string.IsNullOrEmpty(directoryName))
                    {
                        Directory.CreateDirectory(directoryName);
                    }
                    
                    File.Copy(adtPath, outputPath, true);
                    Console.WriteLine($"Copied original ADT without modifications: {Path.GetFileName(adtPath)}");
                    return false;
                }
                catch (Exception copyEx)
                {
                    Console.WriteLine($"Error copying ADT {adtPath}: {copyEx.Message}");
                    return false;
                }
            }
        }
        
        private bool IsValidHeader(BinaryReader reader, string expectedHeader)
        {
            if (reader.BaseStream.Position + 4 > reader.BaseStream.Length)
            {
                return false;
            }
            
            byte[] headerBytes = reader.ReadBytes(4);
            string header = Encoding.ASCII.GetString(headerBytes);
            return header == expectedHeader;
        }
        
        private bool TryReadHeader(BinaryReader reader, out string chunkId, out uint chunkSize)
        {
            chunkId = string.Empty;
            chunkSize = 0;
            
            if (reader.BaseStream.Position + 8 > reader.BaseStream.Length)
            {
                return false;
            }
            
            try
            {
                byte[] headerBytes = reader.ReadBytes(4);
                chunkId = Encoding.ASCII.GetString(headerBytes);
                chunkSize = reader.ReadUInt32();
                return true;
            }
            catch
            {
                return false;
            }
        }
        
        public (int X, int Y) CalculateChunkCoordinates(LiquidBlock block)
        {
            // Calculate the center of the block from its vertices
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
            
            // Get the ADT coordinates for this block
            var adtCoords = CoordinateConverter.WorldToAdtCoordinates(centerX, centerY);
            
            // Get the top-left world coordinates of this ADT
            var adtTopLeft = CoordinateConverter.AdtToWorldCoordinates(adtCoords.X, adtCoords.Y);
            
            // Calculate the relative position within the ADT (0-16 for chunks)
            // FIXED: Account for our swapped coordinate system
            float relativeX = adtTopLeft.X - centerX;  // Distance from left edge
            float relativeY = adtTopLeft.Y - centerY;  // Distance from top edge
            
            // Convert to chunk indices (0-15)
            int chunkX = (int)Math.Floor(relativeX / CHUNK_SIZE);
            int chunkY = (int)Math.Floor(relativeY / CHUNK_SIZE);
            
            // Ensure coordinates are within valid range
            chunkX = Math.Clamp(chunkX, 0, CHUNKS_PER_MAP_SIDE - 1);
            chunkY = Math.Clamp(chunkY, 0, CHUNKS_PER_MAP_SIDE - 1);
            
            return (chunkX, chunkY);
        }
        
        private bool IsChunkMagic(byte[] magic, string expectedMagic)
        {
            return Encoding.ASCII.GetString(magic) == expectedMagic;
        }
    }
} 