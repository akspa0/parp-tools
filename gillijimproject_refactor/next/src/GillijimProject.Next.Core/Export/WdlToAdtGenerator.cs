using System;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Export;

/// <summary>
/// Generates 3.3.5 (WotLK) ADT terrain files from WDL low-resolution height data.
/// WDL has 17x17 outer + 16x16 inner heights per tile (coarse grid).
/// ADT MCNK has 145 vertices (9x9 + 8x8 interleaved) per chunk, 16x16 chunks per tile.
/// This interpolates WDL heights to ADT resolution.
/// </summary>
public static class WdlToAdtGenerator
{
    private const float TileSize = 533.3333f;
    private const float ChunkSize = TileSize / 16f;
    private const float UnitSize = ChunkSize / 8f;

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights.
    /// </summary>
    public static byte[] GenerateAdt(WdlTile wdlTile, int tileX, int tileY)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MVER - version 18 for 3.3.5
        WriteChunk(bw, "MVER", BitConverter.GetBytes(18u));

        // MHDR - header (64 bytes, will be updated later)
        long mhdrPos = ms.Position;
        WriteChunk(bw, "MHDR", new byte[64]);

        // MCIN - 256 entries, 16 bytes each (will be updated later)
        long mcinPos = ms.Position;
        WriteChunk(bw, "MCIN", new byte[256 * 16]);

        // MTEX - empty texture list
        WriteChunk(bw, "MTEX", Array.Empty<byte>());

        // MMDX - empty M2 list
        WriteChunk(bw, "MMDX", Array.Empty<byte>());

        // MMID - empty M2 offsets
        WriteChunk(bw, "MMID", Array.Empty<byte>());

        // MWMO - empty WMO list
        WriteChunk(bw, "MWMO", Array.Empty<byte>());

        // MWID - empty WMO offsets
        WriteChunk(bw, "MWID", Array.Empty<byte>());

        // MDDF - empty M2 placements
        WriteChunk(bw, "MDDF", Array.Empty<byte>());

        // MODF - empty WMO placements
        WriteChunk(bw, "MODF", Array.Empty<byte>());

        // Generate 256 MCNK chunks (16x16 grid)
        var mcnkOffsets = new uint[256];
        var mcnkSizes = new uint[256];

        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                int idx = cy * 16 + cx;
                mcnkOffsets[idx] = (uint)ms.Position;
                
                var mcnkData = GenerateMcnk(wdlTile, tileX, tileY, cx, cy);
                WriteChunk(bw, "MCNK", mcnkData);
                
                mcnkSizes[idx] = (uint)mcnkData.Length;
            }
        }

        // MFBO - flight bounds (optional, 36 bytes)
        var mfboData = new byte[36];
        // Set reasonable defaults for flight bounds
        short maxHeight = 500;
        short minHeight = -500;
        for (int i = 0; i < 9; i++)
        {
            BitConverter.GetBytes(maxHeight).CopyTo(mfboData, i * 2);
            BitConverter.GetBytes(minHeight).CopyTo(mfboData, 18 + i * 2);
        }
        WriteChunk(bw, "MFBO", mfboData);

        // Update MCIN with chunk offsets
        var result = ms.ToArray();
        for (int i = 0; i < 256; i++)
        {
            int mcinEntryPos = (int)mcinPos + 8 + i * 16;
            BitConverter.GetBytes(mcnkOffsets[i]).CopyTo(result, mcinEntryPos);
            BitConverter.GetBytes(mcnkSizes[i] + 8).CopyTo(result, mcinEntryPos + 4); // +8 for chunk header
            // flags and asyncId remain 0
        }

        // Update MHDR offsets (relative to MHDR data start)
        int mhdrDataStart = (int)mhdrPos + 8;
        // ofsMCIN at offset 0x00 in MHDR data
        BitConverter.GetBytes((uint)(mcinPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x00);
        // Other offsets remain 0 (no MTEX, MMDX, etc. with data)

        return result;
    }

    /// <summary>
    /// Generate a single MCNK chunk with terrain interpolated from WDL.
    /// </summary>
    private static byte[] GenerateMcnk(WdlTile wdlTile, int tileX, int tileY, int chunkX, int chunkY)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // Calculate base position for this chunk
        float baseX = (32 - tileX) * TileSize - chunkX * ChunkSize;
        float baseY = (32 - tileY) * TileSize - chunkY * ChunkSize;
        
        // Get interpolated heights for this chunk (145 vertices)
        float[] heights = InterpolateChunkHeights(wdlTile, chunkX, chunkY);
        float baseZ = heights[0]; // Reference height for relative values

        // MCNK header (128 bytes)
        uint flags = 0x40; // has_mccv flag
        bw.Write(flags);            // 0x00: flags
        bw.Write(chunkX);           // 0x04: indexX
        bw.Write(chunkY);           // 0x08: indexY
        bw.Write(0u);               // 0x0C: nLayers
        bw.Write(0u);               // 0x10: nDoodadRefs
        
        // Offsets will be filled in after we know subchunk positions
        long ofsHeightPos = ms.Position;
        bw.Write(0u);               // 0x14: ofsHeight (MCVT)
        bw.Write(0u);               // 0x18: ofsNormal (MCNR)
        bw.Write(0u);               // 0x1C: ofsLayer (MCLY)
        bw.Write(0u);               // 0x20: ofsRefs (MCRF)
        bw.Write(0u);               // 0x24: ofsAlpha (MCAL)
        bw.Write(0u);               // 0x28: sizeAlpha
        bw.Write(0u);               // 0x2C: ofsShadow (MCSH)
        bw.Write(0u);               // 0x30: sizeShadow
        bw.Write(0u);               // 0x34: areaid
        bw.Write(0u);               // 0x38: nMapObjRefs
        bw.Write((ushort)0);        // 0x3C: holes_low_res
        bw.Write((ushort)0);        // 0x3E: unknown
        
        // Low quality texture map (16 bytes)
        for (int i = 0; i < 16; i++) bw.Write((byte)0);
        
        bw.Write(0u);               // 0x50: predTex
        bw.Write(0u);               // 0x54: noEffectDoodad
        bw.Write(0u);               // 0x58: ofsSndEmitters (MCSE)
        bw.Write(0u);               // 0x5C: nSndEmitters
        bw.Write(0u);               // 0x60: ofsLiquid (MCLQ)
        bw.Write(0u);               // 0x64: sizeLiquid
        
        // Position
        bw.Write(baseX);            // 0x68: position.x
        bw.Write(baseY);            // 0x6C: position.y
        bw.Write(baseZ);            // 0x70: position.z
        
        long ofsMccvPos = ms.Position;
        bw.Write(0u);               // 0x74: ofsMCCV
        bw.Write(0u);               // 0x78: ofsMCLV
        bw.Write(0u);               // 0x7C: unused

        // Pad to 128 bytes
        while (ms.Position < 128)
            bw.Write((byte)0);

        // MCVT - height map (145 floats = 580 bytes)
        uint mcvtOffset = (uint)ms.Position + 8; // +8 for MCNK chunk header
        bw.Write(Encoding.ASCII.GetBytes("TVCM")); // Reversed
        bw.Write(145 * 4);
        for (int i = 0; i < 145; i++)
            bw.Write(heights[i] - baseZ); // Heights relative to baseZ

        // MCCV - vertex colors (145 * 4 bytes = 580 bytes)
        uint mccvOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("VCCM")); // Reversed
        bw.Write(145 * 4);
        for (int i = 0; i < 145; i++)
        {
            bw.Write((byte)0x7F); // Blue
            bw.Write((byte)0x7F); // Green
            bw.Write((byte)0x7F); // Red
            bw.Write((byte)0x00); // Alpha
        }

        // MCNR - normals (145 * 3 bytes + 13 padding = 448 bytes)
        uint mcnrOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("RNCM")); // Reversed
        bw.Write(448);
        for (int i = 0; i < 145; i++)
        {
            // Default to pointing up (0, 127, 0) in X, Z, Y order
            bw.Write((sbyte)0);    // X normal
            bw.Write((sbyte)127);  // Z normal (up)
            bw.Write((sbyte)0);    // Y normal
        }
        // 13 bytes padding
        for (int i = 0; i < 13; i++)
            bw.Write((byte)0);

        // MCLY - texture layers (empty)
        uint mclyOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("YLCM"));
        bw.Write(0);

        // MCRF - doodad/WMO refs (empty)
        uint mcrfOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("FRCM"));
        bw.Write(0);

        // MCAL - alpha maps (empty)
        uint mcalOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("LACM"));
        bw.Write(0);

        // MCSH - shadows (empty)
        uint mcshOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("HSCM"));
        bw.Write(0);

        // MCSE - sound emitters (empty)
        uint mcseOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("ESCM"));
        bw.Write(0);

        var result = ms.ToArray();

        // Update header offsets
        BitConverter.GetBytes(mcvtOffset).CopyTo(result, 0x14);  // ofsHeight
        BitConverter.GetBytes(mcnrOffset).CopyTo(result, 0x18);  // ofsNormal
        BitConverter.GetBytes(mclyOffset).CopyTo(result, 0x1C);  // ofsLayer
        BitConverter.GetBytes(mcrfOffset).CopyTo(result, 0x20);  // ofsRefs
        BitConverter.GetBytes(mcalOffset).CopyTo(result, 0x24);  // ofsAlpha
        BitConverter.GetBytes(mcshOffset).CopyTo(result, 0x2C);  // ofsShadow
        BitConverter.GetBytes(mcseOffset).CopyTo(result, 0x58);  // ofsSndEmitters
        BitConverter.GetBytes(mccvOffset).CopyTo(result, 0x74);  // ofsMCCV

        return result;
    }

    /// <summary>
    /// Interpolate WDL heights to ADT MCNK resolution (145 vertices per chunk).
    /// WDL has 17x17 outer grid covering the entire tile (16 chunks).
    /// Each chunk covers ~1/16 of the tile, so we interpolate between WDL grid points.
    /// </summary>
    private static float[] InterpolateChunkHeights(WdlTile wdlTile, int chunkX, int chunkY)
    {
        var heights = new float[145];
        
        // WDL 17x17 grid covers the entire tile
        // Each chunk is 1/16 of the tile, so WDL grid spacing is 1 cell per chunk
        // We need to interpolate within that cell
        
        // Get the 4 corner heights from WDL for this chunk's region
        // WDL indices: chunkX, chunkY to chunkX+1, chunkY+1
        float h00 = wdlTile.Height17[chunkY, chunkX];
        float h10 = wdlTile.Height17[chunkY, Math.Min(chunkX + 1, 16)];
        float h01 = wdlTile.Height17[Math.Min(chunkY + 1, 16), chunkX];
        float h11 = wdlTile.Height17[Math.Min(chunkY + 1, 16), Math.Min(chunkX + 1, 16)];

        // Also use inner 16x16 grid for center interpolation if available
        float hCenter = (chunkX < 16 && chunkY < 16) 
            ? wdlTile.Height16[chunkY, chunkX] 
            : (h00 + h10 + h01 + h11) / 4f;

        // Generate 145 vertices: 9x9 outer + 8x8 inner (interleaved rows)
        int idx = 0;
        for (int row = 0; row < 17; row++)
        {
            bool isInnerRow = (row % 2) == 1;
            int colCount = isInnerRow ? 8 : 9;

            for (int col = 0; col < colCount; col++)
            {
                // Calculate normalized position within chunk (0-1)
                float u, v;
                if (isInnerRow)
                {
                    // Inner row: offset by half unit
                    u = (col + 0.5f) / 8f;
                    v = row / 16f;
                }
                else
                {
                    // Outer row
                    u = col / 8f;
                    v = row / 16f;
                }

                // Bilinear interpolation from WDL corner heights
                float height = BilinearInterpolate(h00, h10, h01, h11, u, v);
                
                // Blend with center height for smoother terrain
                float centerWeight = 1f - 2f * Math.Max(Math.Abs(u - 0.5f), Math.Abs(v - 0.5f));
                centerWeight = Math.Max(0f, centerWeight);
                height = height * (1f - centerWeight * 0.3f) + hCenter * (centerWeight * 0.3f);

                heights[idx++] = height;
            }
        }

        return heights;
    }

    private static float BilinearInterpolate(float h00, float h10, float h01, float h11, float u, float v)
    {
        float top = h00 * (1f - u) + h10 * u;
        float bottom = h01 * (1f - u) + h11 * u;
        return top * (1f - v) + bottom * v;
    }

    private static void WriteChunk(BinaryWriter bw, string sig, byte[] data)
    {
        // Write reversed signature (as stored on disk)
        var sigBytes = Encoding.ASCII.GetBytes(sig);
        Array.Reverse(sigBytes);
        bw.Write(sigBytes);
        bw.Write(data.Length);
        bw.Write(data);
    }
}
