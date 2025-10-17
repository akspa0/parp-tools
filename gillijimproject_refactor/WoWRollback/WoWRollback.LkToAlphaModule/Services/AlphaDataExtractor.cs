using System;
using System.IO;
using GillijimProject.WowFiles.Alpha;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Services;

/// <summary>
/// Extracts raw chunk data from Alpha ADT files to populate LkMcnkSource for round-trip testing.
/// </summary>
public static class AlphaDataExtractor
{
    private const int ChunkHeaderSize = 8; // FourCC (4) + Size (4)
    
    /// <summary>
    /// Reads an Alpha MCNK chunk and extracts raw data to populate LkMcnkSource.
    /// This enables round-trip testing: Alpha → LK → Alpha.
    /// </summary>
    public static LkMcnkSource ExtractFromAlphaMcnk(string alphaAdtPath, int mcnkOffset, int mcnkIndex)
    {
        if (string.IsNullOrWhiteSpace(alphaAdtPath))
            throw new ArgumentException("Alpha ADT path required", nameof(alphaAdtPath));
        if (!File.Exists(alphaAdtPath))
            throw new FileNotFoundException("Alpha ADT not found", alphaAdtPath);

        using var fs = File.OpenRead(alphaAdtPath);
        
        // Read MCNK header (128 bytes after FourCC+size)
        fs.Seek(mcnkOffset + ChunkHeaderSize, SeekOrigin.Begin);
        var headerBytes = new byte[128];
        fs.Read(headerBytes, 0, 128);
        
        // Parse header fields we need
        int indexX = BitConverter.ToInt32(headerBytes, 0x04);
        int indexY = BitConverter.ToInt32(headerBytes, 0x08);
        int areaId = BitConverter.ToInt32(headerBytes, 0x38); // Unknown3 used as AreaId
        
        // Read offsets from header (relative to start of MCNK chunk)
        int mcvtOffset = BitConverter.ToInt32(headerBytes, 0x18);
        int mcnrOffset = BitConverter.ToInt32(headerBytes, 0x1C);
        int mclyOffset = BitConverter.ToInt32(headerBytes, 0x20);
        int mcrfOffset = BitConverter.ToInt32(headerBytes, 0x24);
        int mcalOffset = BitConverter.ToInt32(headerBytes, 0x28);
        int mcalSize = BitConverter.ToInt32(headerBytes, 0x2C);
        int mcshOffset = BitConverter.ToInt32(headerBytes, 0x30);
        int mcshSize = BitConverter.ToInt32(headerBytes, 0x34);
        
        var source = new LkMcnkSource
        {
            IndexX = indexX,
            IndexY = indexY,
            AreaId = (uint)areaId,
            Flags = 0,
            HolesLowRes = 0,
            Radius = 100.0f,
            DoodadRefCount = 0,
            MapObjectRefs = 0,
            NoEffectDoodad = 0,
            OffsLiquid = 0,
            OffsSndEmitters = 0,
            SndEmitterCount = 0
        };
        
        // Extract MCVT raw data (580 bytes, no chunk header in Alpha)
        if (mcvtOffset > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mcvtOffset, SeekOrigin.Begin);
            source.McvtRaw = new byte[580];
            fs.Read(source.McvtRaw, 0, 580);
        }
        
        // Extract MCNR raw data (448 bytes, no chunk header in Alpha)
        if (mcnrOffset > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mcnrOffset, SeekOrigin.Begin);
            source.McnrRaw = new byte[448];
            fs.Read(source.McnrRaw, 0, 448);
        }
        
        // Extract MCLY chunk (with header)
        if (mclyOffset > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mclyOffset, SeekOrigin.Begin);
            var mclyHeader = new byte[ChunkHeaderSize];
            fs.Read(mclyHeader, 0, ChunkHeaderSize);
            int mclyDataSize = BitConverter.ToInt32(mclyHeader, 4);
            
            // Read just the data portion (no header)
            source.MclyRaw = new byte[mclyDataSize];
            fs.Read(source.MclyRaw, 0, mclyDataSize);
        }
        
        // Extract MCRF chunk (with header)
        if (mcrfOffset > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mcrfOffset, SeekOrigin.Begin);
            var mcrfHeader = new byte[ChunkHeaderSize];
            fs.Read(mcrfHeader, 0, ChunkHeaderSize);
            int mcrfDataSize = BitConverter.ToInt32(mcrfHeader, 4);
            
            source.McrfRaw = new byte[mcrfDataSize];
            fs.Read(source.McrfRaw, 0, mcrfDataSize);
        }
        
        // Extract MCSH data (no chunk header, just raw data)
        if (mcshOffset > 0 && mcshSize > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mcshOffset, SeekOrigin.Begin);
            source.McshRaw = new byte[mcshSize];
            fs.Read(source.McshRaw, 0, mcshSize);
        }
        
        // Extract MCAL chunk (with header)
        if (mcalOffset > 0 && mcalSize > 0)
        {
            fs.Seek(mcnkOffset + ChunkHeaderSize + 128 + mcalOffset, SeekOrigin.Begin);
            var mcalHeader = new byte[ChunkHeaderSize];
            fs.Read(mcalHeader, 0, ChunkHeaderSize);
            int mcalDataSize = BitConverter.ToInt32(mcalHeader, 4);
            
            // Store MCAL as alpha layers
            var mcalData = new byte[mcalDataSize];
            fs.Read(mcalData, 0, mcalDataSize);
            
            // Parse MCLY to determine layer count
            int layerCount = source.MclyRaw.Length / 16;
            
            // Convert MCAL data to alpha layers (4096 bytes each)
            // For now, just store raw - the encoder will handle it
            // TODO: Parse MCAL properly based on MCLY flags
            source.AlphaLayers.Clear();
            for (int i = 0; i < layerCount - 1; i++) // First layer has no alpha
            {
                var layer = new byte[4096];
                int offset = i * 4096;
                if (offset + 4096 <= mcalData.Length)
                {
                    Buffer.BlockCopy(mcalData, offset, layer, 0, 4096);
                }
                source.AlphaLayers.Add(new LkMcnkAlphaLayer
                {
                    LayerIndex = i + 1, // Layer 0 is base, alpha layers start at 1
                    ColumnMajorAlpha = layer
                });
            }
        }
        
        // MCSE not commonly used in Alpha, leave empty
        source.McseRaw = Array.Empty<byte>();
        
        return source;
    }
    
    /// <summary>
    /// Extracts all 256 MCNK chunks from an Alpha ADT file.
    /// </summary>
    public static LkAdtSource ExtractFromAlphaAdt(string alphaAdtPath)
    {
        if (string.IsNullOrWhiteSpace(alphaAdtPath))
            throw new ArgumentException("Alpha ADT path required", nameof(alphaAdtPath));
        if (!File.Exists(alphaAdtPath))
            throw new FileNotFoundException("Alpha ADT not found", alphaAdtPath);

        var source = new LkAdtSource
        {
            MapName = Path.GetFileNameWithoutExtension(alphaAdtPath),
            TileX = 0, // TODO: Parse from filename
            TileY = 0
        };
        
        using var fs = File.OpenRead(alphaAdtPath);
        var allBytes = new byte[fs.Length];
        fs.Read(allBytes, 0, (int)fs.Length);
        
        // Alpha ADTs are just 256 MCNK chunks concatenated (no MVER/MHDR)
        int offset = 0;
        for (int i = 0; i < 256; i++)
        {
            if (offset + 8 > allBytes.Length) break;
            
            // Check for MCNK signature
            string fourCC = System.Text.Encoding.ASCII.GetString(allBytes, offset, 4);
            if (fourCC == "KNCM") // "MCNK" reversed
            {
                var mcnkSource = ExtractFromAlphaMcnk(alphaAdtPath, offset, i);
                source.Mcnks.Add(mcnkSource);
                
                // Move to next MCNK
                int mcnkSize = BitConverter.ToInt32(allBytes, offset + 4);
                offset += 8 + mcnkSize;
            }
            else
            {
                // Add empty placeholder
                source.Mcnks.Add(new LkMcnkSource
                {
                    IndexX = i % 16,
                    IndexY = i / 16
                });
                break; // No more MCNKs
            }
        }
        
        // Fill remaining slots with empty MCNKs
        while (source.Mcnks.Count < 256)
        {
            int i = source.Mcnks.Count;
            source.Mcnks.Add(new LkMcnkSource
            {
                IndexX = i % 16,
                IndexY = i / 16
            });
        }
        
        return source;
    }
}
