using System;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module.Services;

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

    /// <summary>
    /// Simple WDL tile data structure.
    /// </summary>
    public class WdlTileData
    {
        public short[,] Height17 { get; } = new short[17, 17];
        public short[,] Height16 { get; } = new short[16, 16];
        public ushort[] HoleMask16 { get; } = new ushort[16];
    }

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights.
    /// </summary>
    public static byte[] GenerateAdt(WdlTileData wdlTile, int tileX, int tileY)
        => GenerateAdt(wdlTile, tileX, tileY, null);

    /// <summary>
    /// Generate a complete 3.3.5 monolithic ADT from WDL tile heights with optional minimap MCCV.
    /// </summary>
    /// <param name="mccvData">Optional array of 256 MCCV byte arrays (one per MCNK), or null for neutral gray.</param>
    public static byte[] GenerateAdt(WdlTileData wdlTile, int tileX, int tileY, byte[][]? mccvData)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MVER - version 18 for 3.3.5
        WriteChunk(bw, "MVER", BitConverter.GetBytes(18u));

        // MHDR - header (64 bytes, will be updated later)
        long mhdrPos = ms.Position;
        WriteChunk(bw, "MHDR", new byte[64]);

        // Track chunk positions for MHDR offsets
        long mcinPos = ms.Position;
        WriteChunk(bw, "MCIN", new byte[256 * 16]);

        long mtexPos = ms.Position;
        WriteChunk(bw, "MTEX", Array.Empty<byte>());

        long mmdxPos = ms.Position;
        WriteChunk(bw, "MMDX", Array.Empty<byte>());

        long mmidPos = ms.Position;
        WriteChunk(bw, "MMID", Array.Empty<byte>());

        long mwmoPos = ms.Position;
        WriteChunk(bw, "MWMO", Array.Empty<byte>());

        long mwidPos = ms.Position;
        WriteChunk(bw, "MWID", Array.Empty<byte>());

        long mddfPos = ms.Position;
        WriteChunk(bw, "MDDF", Array.Empty<byte>());

        long modfPos = ms.Position;
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
                
                var mcnkMccv = mccvData?[idx];
                var mcnkData = GenerateMcnk(wdlTile, tileX, tileY, cx, cy, mcnkMccv);
                WriteChunk(bw, "MCNK", mcnkData);
                
                mcnkSizes[idx] = (uint)mcnkData.Length;
            }
        }

        // MFBO - flight bounds (optional, 36 bytes)
        long mfboPos = ms.Position;
        var mfboData = new byte[36];
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
            BitConverter.GetBytes(mcnkSizes[i] + 8).CopyTo(result, mcinEntryPos + 4);
        }

        // Update MHDR offsets (relative to MHDR data start)
        int mhdrDataStart = (int)mhdrPos + 8;
        
        uint mhdrFlags = 0x01; // has MFBO
        BitConverter.GetBytes(mhdrFlags).CopyTo(result, mhdrDataStart + 0x00);
        BitConverter.GetBytes((uint)(mcinPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x04);
        BitConverter.GetBytes((uint)(mtexPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x08);
        BitConverter.GetBytes((uint)(mmdxPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x0C);
        BitConverter.GetBytes((uint)(mmidPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x10);
        BitConverter.GetBytes((uint)(mwmoPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x14);
        BitConverter.GetBytes((uint)(mwidPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x18);
        BitConverter.GetBytes((uint)(mddfPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x1C);
        BitConverter.GetBytes((uint)(modfPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x20);
        BitConverter.GetBytes((uint)(mfboPos - mhdrDataStart)).CopyTo(result, mhdrDataStart + 0x24);

        return result;
    }

    private static byte[] GenerateMcnk(WdlTileData wdlTile, int tileX, int tileY, int chunkX, int chunkY, byte[]? mccvData = null)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        float baseX = (32 - tileX) * TileSize - chunkX * ChunkSize;
        float baseY = (32 - tileY) * TileSize - chunkY * ChunkSize;
        
        float[] heights = InterpolateChunkHeights(wdlTile, chunkX, chunkY);
        float baseZ = heights[0];

        // MCNK header (128 bytes)
        bw.Write(0x40u); // has_mccv flag
        bw.Write((uint)chunkX);
        bw.Write((uint)chunkY);
        bw.Write(0u); // nLayers
        bw.Write(0u); // nDoodadRefs
        bw.Write(0u); // ofsHeight
        bw.Write(0u); // ofsNormal
        bw.Write(0u); // ofsLayer
        bw.Write(0u); // ofsRefs
        bw.Write(0u); // ofsAlpha
        bw.Write(0u); // sizeAlpha
        bw.Write(0u); // ofsShadow
        bw.Write(0u); // sizeShadow
        bw.Write(0u); // areaid
        bw.Write(0u); // nMapObjRefs
        bw.Write(0u); // holes
        for (int i = 0; i < 8; i++) bw.Write((ushort)0); // doodadMapping
        for (int i = 0; i < 8; i++) bw.Write((byte)0);   // doodadStencil
        bw.Write(0u); // ofsSndEmitters
        bw.Write(0u); // nSndEmitters
        bw.Write(0u); // ofsLiquid
        bw.Write(0u); // sizeLiquid
        bw.Write(baseX); // zpos (WoW X)
        bw.Write(baseY); // xpos (WoW Z)
        bw.Write(baseZ); // ypos (WoW Y)
        bw.Write(0u); // ofsMCCV
        bw.Write(0u); // unused1
        bw.Write(0u); // unused2

        // MCVT - height map
        uint mcvtOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("TVCM"));
        bw.Write(145 * 4);
        for (int i = 0; i < 145; i++)
            bw.Write(heights[i] - baseZ);

        // MCCV - vertex colors
        uint mccvOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("VCCM"));
        bw.Write(145 * 4);
        if (mccvData != null && mccvData.Length == 145 * 4)
        {
            bw.Write(mccvData);
        }
        else
        {
            for (int i = 0; i < 145; i++)
            {
                bw.Write((byte)0x7F);
                bw.Write((byte)0x7F);
                bw.Write((byte)0x7F);
                bw.Write((byte)0x00);
            }
        }

        // MCNR - normals
        uint mcnrOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("RNCM"));
        bw.Write(448);
        for (int i = 0; i < 145; i++)
        {
            bw.Write((sbyte)0);
            bw.Write((sbyte)127);
            bw.Write((sbyte)0);
        }
        for (int i = 0; i < 13; i++) bw.Write((byte)0);

        // Empty subchunks
        uint mclyOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("YLCM")); bw.Write(0);
        
        uint mcrfOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("FRCM")); bw.Write(0);
        
        uint mcalOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("LACM")); bw.Write(0);
        
        uint mcshOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("HSCM")); bw.Write(0);
        
        uint mcseOffset = (uint)ms.Position + 8;
        bw.Write(Encoding.ASCII.GetBytes("ESCM")); bw.Write(0);

        var result = ms.ToArray();

        // Update header offsets in MCNK
        BitConverter.GetBytes(mcvtOffset).CopyTo(result, 0x14);
        BitConverter.GetBytes(mcnrOffset).CopyTo(result, 0x18);
        BitConverter.GetBytes(mclyOffset).CopyTo(result, 0x1C);
        BitConverter.GetBytes(mcrfOffset).CopyTo(result, 0x20);
        BitConverter.GetBytes(mcalOffset).CopyTo(result, 0x24);
        BitConverter.GetBytes(mcshOffset).CopyTo(result, 0x2C);
        BitConverter.GetBytes(mcseOffset).CopyTo(result, 0x58);
        BitConverter.GetBytes(mccvOffset).CopyTo(result, 0x74);

        return result;
    }

    private static float[] InterpolateChunkHeights(WdlTileData wdlTile, int chunkX, int chunkY)
    {
        var heights = new float[145];
        
        float h00 = wdlTile.Height17[chunkY, chunkX];
        float h10 = wdlTile.Height17[chunkY, Math.Min(chunkX + 1, 16)];
        float h01 = wdlTile.Height17[Math.Min(chunkY + 1, 16), chunkX];
        float h11 = wdlTile.Height17[Math.Min(chunkY + 1, 16), Math.Min(chunkX + 1, 16)];

        float hCenter = (chunkX < 16 && chunkY < 16) 
            ? wdlTile.Height16[chunkY, chunkX] 
            : (h00 + h10 + h01 + h11) / 4f;

        int idx = 0;
        for (int row = 0; row < 17; row++)
        {
            bool isInnerRow = (row % 2) == 1;
            int colCount = isInnerRow ? 8 : 9;

            for (int col = 0; col < colCount; col++)
            {
                float u, v;
                if (isInnerRow)
                {
                    u = (col + 0.5f) / 8f;
                    v = row / 16f;
                }
                else
                {
                    u = col / 8f;
                    v = row / 16f;
                }

                float height = BilinearInterpolate(h00, h10, h01, h11, u, v);
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
        var sigBytes = Encoding.ASCII.GetBytes(sig);
        Array.Reverse(sigBytes);
        bw.Write(sigBytes);
        bw.Write(data.Length);
        bw.Write(data);
    }
}
