using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace WoWRollback.PM4Module.Services;

/// <summary>
/// Extracts texture data (MTEX, MCLY, MCAL, MCCV) from museum ADT files
/// for injection into WDL-generated ADTs.
/// </summary>
public class MuseumTextureExtractor
{
    /// <summary>
    /// Extracted texture data from a museum ADT.
    /// </summary>
    public record MuseumTextureData(
        byte[] MtexData,           // Raw MTEX chunk data (texture paths)
        byte[][] MclyPerChunk,     // 256 MCLY data arrays (one per MCNK)
        byte[][] McalPerChunk,     // 256 MCAL data arrays (one per MCNK)
        byte[][] MccvPerChunk,     // 256 MCCV data arrays (one per MCNK)
        byte[][] McvtPerChunk,     // 256 MCVT data arrays (one per MCNK) - heights
        int[] NLayersPerChunk      // Number of layers per MCNK (for MCLY sizing)
    );

    /// <summary>
    /// Extract texture data from a museum ADT file.
    /// </summary>
    public static MuseumTextureData? Extract(string adtPath)
    {
        if (!File.Exists(adtPath)) return null;

        try
        {
            var bytes = File.ReadAllBytes(adtPath);
            return ExtractFromBytes(bytes);
        }
        catch
        {
            return null;
        }
    }

    private static MuseumTextureData? ExtractFromBytes(byte[] data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);

        byte[] mtexData = Array.Empty<byte>();
        var mclyPerChunk = new byte[256][];
        var mcalPerChunk = new byte[256][];
        var mccvPerChunk = new byte[256][];
        var mcvtPerChunk = new byte[256][];
        var nLayersPerChunk = new int[256];

        // Initialize arrays
        for (int i = 0; i < 256; i++)
        {
            mclyPerChunk[i] = Array.Empty<byte>();
            mcalPerChunk[i] = Array.Empty<byte>();
            mccvPerChunk[i] = Array.Empty<byte>();
            mcvtPerChunk[i] = Array.Empty<byte>();
        }

        // Parse root chunks
        while (ms.Position + 8 <= ms.Length)
        {
            long chunkStart = ms.Position;
            var sigBytes = br.ReadBytes(4);
            Array.Reverse(sigBytes);
            string sig = Encoding.ASCII.GetString(sigBytes);
            uint size = br.ReadUInt32();
            long dataStart = ms.Position;

            switch (sig)
            {
                case "MTEX":
                    mtexData = br.ReadBytes((int)size);
                    break;

                case "MCNK":
                    ParseMcnk(br, (int)size, chunkStart, mclyPerChunk, mcalPerChunk, mccvPerChunk, mcvtPerChunk, nLayersPerChunk);
                    break;
            }

            ms.Position = dataStart + size;
        }

        return new MuseumTextureData(mtexData, mclyPerChunk, mcalPerChunk, mccvPerChunk, mcvtPerChunk, nLayersPerChunk);
    }

    private static void ParseMcnk(BinaryReader br, int size, long mcnkStart,
        byte[][] mclyPerChunk, byte[][] mcalPerChunk, byte[][] mccvPerChunk, byte[][] mcvtPerChunk, int[] nLayersPerChunk)
    {
        long dataStart = br.BaseStream.Position;

        // Read MCNK header (128 bytes)
        uint flags = br.ReadUInt32();
        uint ix = br.ReadUInt32();
        uint iy = br.ReadUInt32();
        uint nLayers = br.ReadUInt32();
        br.ReadUInt32(); // nDoodadRefs
        uint ofsHeight = br.ReadUInt32();
        uint ofsNormal = br.ReadUInt32();
        uint ofsLayer = br.ReadUInt32();
        br.ReadUInt32(); // ofsRefs
        uint ofsAlpha = br.ReadUInt32();
        uint sizeAlpha = br.ReadUInt32();
        br.ReadUInt32(); // ofsShadow
        br.ReadUInt32(); // sizeShadow
        br.ReadUInt32(); // areaid
        br.ReadUInt32(); // nMapObjRefs
        br.ReadUInt32(); // holes
        br.ReadBytes(16); // doodadMapping
        br.ReadBytes(8);  // doodadStencil
        br.ReadUInt32(); // ofsSndEmitters
        br.ReadUInt32(); // nSndEmitters
        br.ReadUInt32(); // ofsLiquid
        br.ReadUInt32(); // sizeLiquid
        br.ReadSingle(); br.ReadSingle(); br.ReadSingle(); // position
        uint ofsMCCV = br.ReadUInt32();

        // FIX: chunk index is row * 16 + column = iy * 16 + ix
        int chunkIdx = (int)(iy * 16 + ix);
        if (chunkIdx < 0 || chunkIdx >= 256) return;

        nLayersPerChunk[chunkIdx] = (int)nLayers;

        // Offsets are relative to MCNK chunk start (including 8-byte chunk header)

        // Read MCVT (heights) if present
        if (ofsHeight > 0)
        {
            br.BaseStream.Position = mcnkStart + ofsHeight;
            var sig = br.ReadBytes(4);
            uint mcvtSize = br.ReadUInt32();
            if (Encoding.ASCII.GetString(sig.Reverse().ToArray()) == "MCVT")
            {
                mcvtPerChunk[chunkIdx] = br.ReadBytes((int)mcvtSize);
            }
        }

        // Read MCLY if present
        if (ofsLayer > 0 && nLayers > 0)
        {
            br.BaseStream.Position = mcnkStart + ofsLayer;
            var sig = br.ReadBytes(4);
            uint mclySize = br.ReadUInt32();
            if (Encoding.ASCII.GetString(sig.Reverse().ToArray()) == "MCLY")
            {
                mclyPerChunk[chunkIdx] = br.ReadBytes((int)mclySize);
            }
        }

        // Read MCAL if present
        if (ofsAlpha > 0 && sizeAlpha > 0)
        {
            br.BaseStream.Position = mcnkStart + ofsAlpha;
            var sig = br.ReadBytes(4);
            uint mcalSize = br.ReadUInt32();
            if (Encoding.ASCII.GetString(sig.Reverse().ToArray()) == "MCAL")
            {
                mcalPerChunk[chunkIdx] = br.ReadBytes((int)mcalSize);
            }
        }

        // Read MCCV if present
        bool hasMccv = (flags & 0x40) != 0; // has_mccv flag
        if (hasMccv && ofsMCCV > 0)
        {
            br.BaseStream.Position = mcnkStart + ofsMCCV;
            var sig = br.ReadBytes(4);
            uint mccvSize = br.ReadUInt32();
            if (Encoding.ASCII.GetString(sig.Reverse().ToArray()) == "MCCV")
            {
                mccvPerChunk[chunkIdx] = br.ReadBytes((int)mccvSize);
            }
        }
    }
}

