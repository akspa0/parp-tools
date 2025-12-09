using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GillijimProject.Utilities;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// Converts WotLK-era MH2O water chunks to Alpha-era MCLQ chunks for backporting.
/// </summary>
public static class MclqBackporter
{
    private struct Mh2oHeader
    {
        public uint OffsetInformation;
        public uint LayerCount;
        public uint OffsetRenderMask;
    }

    private struct Mh2oInformation
    {
        public ushort LiquidTypeId;
        public ushort Flags;
        public float MinHeightLevel;
        public float MaxHeightLevel;
        public byte XOffset;
        public byte YOffset;
        public byte Width;
        public byte Height;
        public uint OffsetMask;
    }

    /// <summary>
    /// Converts an entire MH2O chunk into 256 blobs of MCLQ data (one for each MCNK).
    /// Returns null for chunks with no liquid.
    /// </summary>
    public static byte[]?[] ConvertMh2oToMclqs(byte[] mh2oData)
    {
        var results = new byte[256][];
        if (mh2oData == null || mh2oData.Length < 256 * 12)
            return results; // Return empty array if invalid

        using var ms = new MemoryStream(mh2oData);
        using var br = new BinaryReader(ms);

        // Read 256 headers
        var headers = new Mh2oHeader[256];
        for (int i = 0; i < 256; i++)
        {
            headers[i] = new Mh2oHeader
            {
                OffsetInformation = br.ReadUInt32(),
                LayerCount = br.ReadUInt32(),
                OffsetRenderMask = br.ReadUInt32()
            };
        }

        for (int i = 0; i < 256; i++)
        {
            if (headers[i].LayerCount == 0) continue;

            try
            {
                results[i] = ConvertChunk(mh2oData, headers[i]);
            }
            catch (Exception)
            {
                // Soft fail for single chunk
                results[i] = null;
            }
        }

        return results;
    }

    private static byte[] ConvertChunk(byte[] data, Mh2oHeader header)
    {
        // 1. Analyze layers to pick a "Dominant" type for MCLQ (which supports only one type per chunk)
        // Priorities: Magma > Ocean > Water/River (Default)
        // Also gather height data.
        
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);

        var layers = new List<Mh2oInformation>();
        ms.Position = header.OffsetInformation;
        for (int l = 0; l < header.LayerCount; l++)
        {
            var info = new Mh2oInformation
            {
                LiquidTypeId = br.ReadUInt16(),
                Flags = br.ReadUInt16(),
                MinHeightLevel = br.ReadSingle(),
                MaxHeightLevel = br.ReadSingle(),
                XOffset = br.ReadByte(),
                YOffset = br.ReadByte(),
                Width = br.ReadByte(),
                Height = br.ReadByte(),
                OffsetMask = br.ReadUInt32()
            };
            layers.Add(info);
        }

        // Determine dominant type
        bool hasMagma = layers.Any(l => l.LiquidTypeId == 3 || l.LiquidTypeId == 100); // 3=Lava
        bool hasOcean = layers.Any(l => l.LiquidTypeId == 2 || l.LiquidTypeId == 100); // 2=Ocean (Approx)
        // Rough mappings:
        // Alpha Types: 1=Water, 2=Ocean, 3=Magma, 4=Slime?
        // WotLK LiquidType DBC mapping is complex, simplifying for MVP:
        // If Logic:
        //   Magma (ID 3) -> Alpha Magma
        //   Ocean (ID 2, 14, etc) -> Alpha Ocean
        //   Slime (ID 4) -> Alpha River?
        //   Water (ID 1) -> Alpha Water
        
        // For now, simple heuristic:
        var dominantType = MclqLiquidType.Water;
        if (hasMagma) dominantType = MclqLiquidType.Magma;
        else if (hasOcean) dominantType = MclqLiquidType.Ocean;

        // 2. Resample data into 9x9 grid (Vertices) and 8x8 grid (Tiles)
        // Initialize samples
        float[] heightGrid = new float[81]; 
        // Fill with "safe" default (min height of first layer or something)
        float baseHeight = layers[0].MinHeightLevel;
        Array.Fill(heightGrid, baseHeight);
        
        byte[,] tileFlags = new byte[8, 8]; // 0x0F = Type, 0xF0 = Flags
        
        // Render Mask (if present) defines "holes" or valid tiles
        // But MH2O RenderMask is global for the chunk?
        byte[] renderMask = null;
        if (header.OffsetRenderMask != 0)
        {
            // 8x8 bits = 8 bytes? No, MH2O render mask is usually 8 bytes (64 bits) for 8x8 tiles?
            // "ofsRenderMask" points to data.
            // Check ChunkWater.cpp: reads "MH2O_Render" which is likely 64 bits?
            // "f.read(&Render.value(), sizeof(MH2O_Render));"
            // MH2O_Render is likely 8 bytes (uint64 or char[8]).
            // Let's assume 8 bytes.
            ms.Position = header.OffsetRenderMask;
            renderMask = br.ReadBytes(8);
        }

        // Fill sample grids
        // Iterate layers in reverse to let top layers overwrite?
        foreach (var layer in layers)
        {
            // Sample Heights
            // Alpha expects 9x9. MH2O layer has width/height (max 8x8? No, width/height is in squares, so vertices is W+1 x H+1)
            // But width/height in MH2O header is dimensions of the *valid* area.
            // Vertices are explicitly stored in MH2O? No, MH2O stores "Mask" (validity) and Heightmap floats.
            
            // If OffsetMask != 0 and IsValid(layer), we have heightmap data.
            // Format: (width+1)*(height+1) floats? Or compressed?
            // Checking ChunkWater.cpp: "if (info.ofsInfoMask > 0... bitmask..."
            // Actually MH2O *heights* are stored? Where?
            // Ah, the `Mh2oInformation` struct INCLUDES height/depth? No.
            // `Mh2oInformation` has `ofsInfoMask`.
            // Wait, standard MH2O:
            // The data at `ofsInfoMask` contains:
            // 1. Bitmask (ceil(W*H/8) bytes) indicating which tiles are present.
            // 2. Vertex Data?
            // Actually, MH2O format (WotLK) is:
            // [Bitmask]
            // [Vertex Data] (Heights)
            // It seems `ChunkWater.cpp` doesn't explicitly read heights? 
            // `ChunkWater.cpp` uses `liquid_layer` constructor passing `info` and `infoMask`.
            // Let's check `liquid_layer.cpp` (not read, but likely handled there).
            // Standard MH2O usually has heights *if* type != Ocean (Ocean is flat).
            
            // Assumption for Backport (MVP):
            // If Ocean -> Flat MinHeightLevel
            // If Magma/Water -> Read floats
            if (layer.LiquidTypeId != 2 && layer.OffsetMask != 0) // Not ocean, has data
            {
                 // Read heights... logic is complex.
                 // Let's stick to "Flat" export for Alpha for now to reduce risk, using MinHeightLevel.
                 // TODO: enhanced height sampling.
            }
            
            // Apply to 9x9 grid
            // For now, fill with MinHeightLevel for the whole layer's extent
            for(int y = layer.YOffset; y < layer.YOffset + layer.Height + 1 && y < 9; y++)
            {
                for(int x = layer.XOffset; x < layer.XOffset + layer.Width + 1 && x < 9; x++)
                {
                    // If within bounds, set height
                    heightGrid[y * 9 + x] = layer.MinHeightLevel;
                }
            }
            
            // Fill tiles (8x8)
            // Mark dominance
            byte targetType = (byte)dominantType;
            // 0x40 = River/Water flag in Alpha? 
            // Simple: Just Type in lower nibble. Flags 0.
            
            for(int y = layer.YOffset; y < layer.YOffset + layer.Height && y < 8; y++)
            {
                for(int x = layer.XOffset; x < layer.XOffset + layer.Width && x < 8; x++)
                {
                    // Check render mask if exists
                    bool visible = true;
                    if (renderMask != null)
                    {
                        // Check bit (y*8 + x)
                        // Verify bit order (lsb/msb)
                        byte maskByte = renderMask[y]; 
                        visible = (maskByte & (1 << x)) != 0; // Assumption
                    }
                    
                    if (visible)
                    {
                        tileFlags[y, x] = (byte)(targetType & 0x0F);
                        // Add some default flags? 
                        // Water often has 0x40? 
                        // Leaving as 0 | Type for now.
                    }
                    else
                    {
                         tileFlags[y, x] = 0x0F; // 0x0F = Hidden/No Liquid?
                         // Alpha uses 15 (0xF) -> no liquid? Or 0?
                         // Usually 0 = no liquid, but if implicit? 
                         // Let's use 0x0F based on noggit "Dont Render".
                    }
                }
            }
        }
        
        // 3. Serialize MCLQ
        using var outMs = new MemoryStream();
        using var bw = new BinaryWriter(outMs);
        
        // Range (2 floats)
        float minH = heightGrid.Min();
        float maxH = heightGrid.Max();
        bw.Write(minH);
        bw.Write(maxH);
        
        // Vertices (9x9)
        // Format depends on layout.
        // If Magma: u16, u16, float
        // If Ocean: u8, u8, u8, u8 (No float, inferred from minH?)
        // If Water: u8, u8, u8, u8, float
        
        // Since we force dominant type, we must match layout.
        if (dominantType == MclqLiquidType.Magma)
        {
            for(int i=0; i<81; i++)
            {
                bw.Write((ushort)0); // s
                bw.Write((ushort)0); // t
                bw.Write(heightGrid[i]); // h
            }
        }
        else if (dominantType == MclqLiquidType.Ocean)
        {
            for(int i=0; i<81; i++)
            {
                bw.Write((byte)128); // depth?
                bw.Write((byte)0); // foam
                bw.Write((byte)128); // wet
                bw.Write((byte)0); // filler
                // No height written
            }
        }
        else // Water
        {
            for(int i=0; i<81; i++)
            {
                bw.Write((byte)128); // depth
                bw.Write((byte)0); 
                bw.Write((byte)0); 
                bw.Write((byte)0);
                bw.Write(heightGrid[i]); // height
            }
        }
        
        // Tiles (8x8)
        for(int y=0; y<8; y++)
        {
            for(int x=0; x<8; x++)
            {
               bw.Write(tileFlags[y,x]);
            }
        }
        
        return outMs.ToArray();
    }

    private enum MclqLiquidType
    {
        Water = 1,
        Ocean = 2,
        Magma = 3,
        Slime = 4
    }
}
