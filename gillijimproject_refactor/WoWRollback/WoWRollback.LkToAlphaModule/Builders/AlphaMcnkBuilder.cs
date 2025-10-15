using System;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using Util = GillijimProject.Utilities.Utilities;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMcnkBuilder
{
    private const int McnkHeaderSize = 0x80;
    private const int ChunkLettersAndSize = 8;

    public static byte[] BuildFromLk(byte[] lkAdtBytes, int mcNkOffset)
    {
        int headerStart = mcNkOffset;
        // Read LK MCNK header to get IndexX/IndexY
        var lkHeader = ReadLkMcnkHeader(lkAdtBytes, mcNkOffset);

        // Find MCVT chunk inside this LK MCNK
        int subStart = mcNkOffset + ChunkLettersAndSize + McnkHeaderSize;
        int subEnd = mcNkOffset + 8 + BitConverter.ToInt32(lkAdtBytes, mcNkOffset + 4);
        if (subEnd > lkAdtBytes.Length) subEnd = lkAdtBytes.Length;

        byte[]? mcvtLkWhole = null;
        for (int p = subStart; p + 8 <= subEnd;)
        {
            string fcc = Encoding.ASCII.GetString(lkAdtBytes, p, 4);
            int size = BitConverter.ToInt32(lkAdtBytes, p + 4);
            int dataStart = p + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > subEnd) break;

            if (fcc == "TVCM") // 'MCVT' reversed on disk
            {
                mcvtLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                Buffer.BlockCopy(lkAdtBytes, p, mcvtLkWhole, 0, mcvtLkWhole.Length);
                break;
            }
            p = next;
        }

        // Build alpha MCVT chunk
        Chunk alphaMcvt;
        if (mcvtLkWhole != null)
        {
            // Convert LK-order MCVT to Alpha-order (inverse of McvtAlpha.ToMcvt)
            var lkData = new byte[BitConverter.ToInt32(mcvtLkWhole, 4)];
            Buffer.BlockCopy(mcvtLkWhole, 8, lkData, 0, lkData.Length);
            var alphaData = ConvertMcvtLkToAlpha(lkData);
            alphaMcvt = new Chunk("MCVT", alphaData.Length, alphaData);
        }
        else
        {
            alphaMcvt = new Chunk("MCVT", 0, Array.Empty<byte>());
        }

        // Compose Alpha MCNK header
        var hdr = new McnkAlphaHeader
        {
            Flags = 0,
            IndexX = lkHeader.IndexX,
            IndexY = lkHeader.IndexY,
            Unknown1 = 0,
            NLayers = 0,
            M2Number = 0,
            McvtOffset = 0, // first subchunk immediately after header
            McnrOffset = 0,
            MclyOffset = 0,
            McrfOffset = 0,
            McalOffset = 0,
            McalSize = 0,
            McshOffset = 0,
            McshSize = 0,
            Unknown3 = 0,
            WmoNumber = 0,
            Holes = 0,
            GroundEffectsMap1 = 0,
            GroundEffectsMap2 = 0,
            GroundEffectsMap3 = 0,
            GroundEffectsMap4 = 0,
            Unknown6 = 0,
            Unknown7 = 0,
            McnkChunksSize = alphaMcvt.GetWholeChunk().Length, // total size of subchunks region
            Unknown8 = 0,
            MclqOffset = 0,
            Unused1 = 0,
            Unused2 = 0,
            Unused3 = 0,
            Unused4 = 0,
            Unused5 = 0,
            Unused6 = 0
        };

        int givenSize = McnkHeaderSize + alphaMcvt.GetRealSize();

        using var ms = new MemoryStream();
        // Write MCNK letters reversed ('KNCM')
        var reversedLetters = Encoding.ASCII.GetBytes("KNCM");
        ms.Write(reversedLetters, 0, 4);
        ms.Write(BitConverter.GetBytes(givenSize), 0, 4);

        // Header
        var hdrBytes = Util.StructToByteArray(hdr);
        if (hdrBytes.Length != McnkHeaderSize)
        {
            Array.Resize(ref hdrBytes, McnkHeaderSize);
        }
        ms.Write(hdrBytes, 0, McnkHeaderSize);

        // Subchunks (only MCVT for now)
        var mcvtWhole = alphaMcvt.GetWholeChunk();
        ms.Write(mcvtWhole, 0, mcvtWhole.Length);

        return ms.ToArray();
    }

    public static byte[] BuildEmpty(int indexX, int indexY)
    {
        var hdr = new McnkAlphaHeader
        {
            Flags = 0,
            IndexX = indexX,
            IndexY = indexY,
            Unknown1 = 0,
            NLayers = 0,
            M2Number = 0,
            McvtOffset = 0,
            McnrOffset = 0,
            MclyOffset = 0,
            McrfOffset = 0,
            McalOffset = 0,
            McalSize = 0,
            McshOffset = 0,
            McshSize = 0,
            Unknown3 = 0,
            WmoNumber = 0,
            Holes = 0,
            GroundEffectsMap1 = 0,
            GroundEffectsMap2 = 0,
            GroundEffectsMap3 = 0,
            GroundEffectsMap4 = 0,
            Unknown6 = 0,
            Unknown7 = 0,
            McnkChunksSize = 0,
            Unknown8 = 0,
            MclqOffset = 0,
            Unused1 = 0,
            Unused2 = 0,
            Unused3 = 0,
            Unused4 = 0,
            Unused5 = 0,
            Unused6 = 0
        };

        int givenSize = McnkHeaderSize; // header only
        using var ms = new MemoryStream();
        ms.Write(Encoding.ASCII.GetBytes("KNCM"), 0, 4);
        ms.Write(BitConverter.GetBytes(givenSize), 0, 4);
        var hdrBytes = Util.StructToByteArray(hdr);
        if (hdrBytes.Length != McnkHeaderSize) Array.Resize(ref hdrBytes, McnkHeaderSize);
        ms.Write(hdrBytes, 0, McnkHeaderSize);
        return ms.ToArray();
    }

    private static McnkHeader ReadLkMcnkHeader(byte[] bytes, int mcNkOffset)
    {
        int headerOffset = mcNkOffset + ChunkLettersAndSize;
        var headerContent = new byte[McnkHeaderSize];
        Buffer.BlockCopy(bytes, headerOffset, headerContent, 0, McnkHeaderSize);
        return Util.ByteArrayToStruct<McnkHeader>(headerContent);
    }

    private static byte[] ConvertMcvtLkToAlpha(byte[] mcvtLk)
    {
        // LK order is interleaved [outer row 0 (9 floats), inner row 0 (8 floats), ..., outer row 8 (9 floats)].
        // Alpha order requires all outer 9x9 first, then concatenated 8 inner rows of 8 floats each.
        const int floatSize = 4;
        const int outerRowFloats = 9;
        const int innerRowFloats = 8;
        const int outerRowBytes = outerRowFloats * floatSize; // 36
        const int innerRowBytes = innerRowFloats * floatSize; // 32
        const int outerBlockBytes = outerRowBytes * 9; // 324

        var alphaData = new byte[mcvtLk.Length];
        int src = 0;
        // Fill outer block
        for (int i = 0; i < 9; i++)
        {
            Buffer.BlockCopy(mcvtLk, src, alphaData, i * outerRowBytes, outerRowBytes);
            src += outerRowBytes;
            if (i == 8) break;
            // inner row i goes after outer block at innerDataStart
            int innerDest = outerBlockBytes + (i * innerRowBytes);
            Buffer.BlockCopy(mcvtLk, src, alphaData, innerDest, innerRowBytes);
            src += innerRowBytes;
        }
        return alphaData;
    }
}
