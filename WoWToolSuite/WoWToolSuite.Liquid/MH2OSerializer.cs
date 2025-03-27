using System;
using System.IO;
using System.Linq;
using System.Text;
using Warcraft.NET.Files.ADT.Chunks;
using Warcraft.NET.Files.ADT.Entries;

namespace WowToolSuite.Liquid
{
    /// <summary>
    /// Custom serializer for MH2O chunks to avoid problems with the Warcraft.NET library's serialization
    /// </summary>
    public class MH2OSerializer
    {
        /// <summary>
        /// Serializes an MH2O chunk to a byte array without relying on Warcraft.NET's serialization
        /// </summary>
        /// <param name="waterData">The MH2O chunk data to serialize</param>
        /// <returns>Serialized binary data for the MH2O chunk</returns>
        public static byte[] Serialize(MH2OOld waterData)
        {
            using (var ms = new MemoryStream())
            using (var bw = new BinaryWriter(ms))
            {
                // Write magic header
                bw.Write(Encoding.ASCII.GetBytes("MH2O"));
                
                // Reserve space for the MH2O headers (will fill in later)
                long headersStart = bw.BaseStream.Position;
                bw.BaseStream.Position += 256 * GetMH2OHeaderSize();

                // Track instance data positions
                uint[] instanceOffsets = new uint[256];
                
                // Write MH2O instances (reserve space for now)
                for (int i = 0; i < 256; i++)
                {
                    var header = waterData.MH2OHeaders[i];
                    if (header.LayerCount > 0)
                    {
                        instanceOffsets[i] = (uint)bw.BaseStream.Position;
                        bw.BaseStream.Position += header.Instances.Length * GetMH2OInstanceSize();
                    }
                }

                // Write all the render bitmaps and vertex data
                for (int i = 0; i < 256; i++)
                {
                    var header = waterData.MH2OHeaders[i];
                    if (header.LayerCount > 0)
                    {
                        // Write attributes if any
                        if (header.Attributes != null)
                        {
                            header.OffsetAttributes = (uint)bw.BaseStream.Position;
                            WriteMH2OAttribute(bw, header.Attributes);
                        }
                        else
                        {
                            header.OffsetAttributes = 0;
                        }

                        foreach (var instance in header.Instances)
                        {
                            if (instance != null)
                            {
                                // Write render bitmap
                                if (instance.RenderBitmapBytes != null && instance.RenderBitmapBytes.Length > 0)
                                {
                                    instance.OffsetExistsBitmap = (uint)bw.BaseStream.Position;
                                    bw.Write(instance.RenderBitmapBytes);
                                }
                                else
                                {
                                    instance.OffsetExistsBitmap = 0;
                                }

                                // Write vertex data
                                if (instance.VertexData != null && instance.VertexData.HeightMap != null)
                                {
                                    instance.OffsetVertexData = (uint)bw.BaseStream.Position;
                                    WriteVertexData(bw, instance);
                                }
                                else
                                {
                                    instance.OffsetVertexData = 0;
                                }
                            }
                        }
                    }
                }

                // Now go back and fill in all the instances with their offsets set
                for (int i = 0; i < 256; i++)
                {
                    var header = waterData.MH2OHeaders[i];
                    if (header.LayerCount > 0 && instanceOffsets[i] > 0)
                    {
                        bw.BaseStream.Position = instanceOffsets[i];
                        foreach (var instance in header.Instances)
                        {
                            WriteMH2OInstance(bw, instance);
                        }
                    }
                }

                // Finally, go back and fill in the headers
                bw.BaseStream.Position = headersStart;
                for (int i = 0; i < 256; i++)
                {
                    WriteMH2OHeader(bw, waterData.MH2OHeaders[i], instanceOffsets[i]);
                }

                return ms.ToArray();
            }
        }

        private static void WriteMH2OHeader(BinaryWriter bw, MH2OHeader header, uint offsetInstances)
        {
            bw.Write(header.OffsetAttributes);
            bw.Write(header.LayerCount);
            bw.Write(offsetInstances);
        }

        private static void WriteMH2OInstance(BinaryWriter bw, MH2OInstance instance)
        {
            bw.Write(instance.LiquidTypeId);
            bw.Write(instance.LiquidObjectOrVertexFormat);
            bw.Write(instance.MinHeightLevel);
            bw.Write(instance.MaxHeightLevel);
            bw.Write(instance.OffsetX);
            bw.Write(instance.OffsetY);
            bw.Write(instance.Width);
            bw.Write(instance.Height);
            bw.Write(instance.OffsetExistsBitmap);
            bw.Write(instance.OffsetVertexData);
        }

        private static void WriteMH2OAttribute(BinaryWriter bw, MH2OAttribute attribute)
        {
            // Write Fishable and Deep byte arrays
            bw.Write(attribute.Fishable);
            bw.Write(attribute.Deep);
        }

        private static void WriteVertexData(BinaryWriter bw, MH2OInstance instance)
        {
            // MH2O vertex data grid size is Width+1 x Height+1
            int width = instance.Width + 1;
            int height = instance.Height + 1;

            // Write the height map - required for all formats
            if (instance.LiquidObjectOrVertexFormat != 2) // Skip for format 2
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        if (instance.VertexData.HeightMap != null &&
                            y < instance.VertexData.HeightMap.GetLength(0) &&
                            x < instance.VertexData.HeightMap.GetLength(1))
                        {
                            bw.Write(instance.VertexData.HeightMap[y, x]);
                        }
                        else
                        {
                            bw.Write(instance.MinHeightLevel);
                        }
                    }
                }
            }

            // Write UV data if format is 1 or 3
            if (instance.LiquidObjectOrVertexFormat == 1 || instance.LiquidObjectOrVertexFormat == 3)
            {
                if (instance.VertexData.UVMap != null)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            if (y < instance.VertexData.UVMap.GetLength(0) && 
                                x < instance.VertexData.UVMap.GetLength(1))
                            {
                                bw.Write(instance.VertexData.UVMap[y, x, 0]); // U coordinate
                                bw.Write(instance.VertexData.UVMap[y, x, 1]); // V coordinate
                            }
                            else
                            {
                                bw.Write(0.0f); // Default U
                                bw.Write(0.0f); // Default V
                            }
                        }
                    }
                }
                else
                {
                    // Write default UV values
                    for (int i = 0; i < width * height; i++)
                    {
                        bw.Write(0.0f); // Default U
                        bw.Write(0.0f); // Default V
                    }
                }
            }

            // Write depth data if format is 0, 2, or 3
            if (instance.LiquidObjectOrVertexFormat == 0 || 
                instance.LiquidObjectOrVertexFormat == 2 || 
                instance.LiquidObjectOrVertexFormat == 3)
            {
                if (instance.VertexData.DepthMap != null)
                {
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            if (y < instance.VertexData.DepthMap.GetLength(0) && 
                                x < instance.VertexData.DepthMap.GetLength(1))
                            {
                                bw.Write(instance.VertexData.DepthMap[y, x]);
                            }
                            else
                            {
                                bw.Write((byte)255); // Default depth
                            }
                        }
                    }
                }
                else
                {
                    // Write default depth values
                    for (int i = 0; i < width * height; i++)
                    {
                        bw.Write((byte)255); // Default depth
                    }
                }
            }
        }

        // Sizes of the various structures
        private static int GetMH2OHeaderSize()
        {
            return 12; // 3 uint32 values
        }

        private static int GetMH2OInstanceSize()
        {
            return 20; // 2 ushort + 2 float + 4 byte + 2 uint values
        }
    }
} 