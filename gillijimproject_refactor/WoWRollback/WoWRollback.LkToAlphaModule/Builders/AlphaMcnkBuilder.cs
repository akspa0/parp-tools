using System;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using Util = GillijimProject.Utilities.Utilities;
using WoWRollback.LkToAlphaModule;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class AlphaMcnkBuilder
{
    private const int McnkHeaderSize = 0x80;
    private const int ChunkLettersAndSize = 8;

    public static byte[] BuildFromLk(byte[] lkAdtBytes, int mcNkOffset, LkToAlphaOptions? opts = null)
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

        // Build alpha raw MCVT data (no named subchunk in Alpha)
        byte[] alphaMcvtRaw = Array.Empty<byte>();
        if (mcvtLkWhole != null)
        {
            // Convert LK-order MCVT to Alpha-order with absolute heights (add base Z from LK header)
            var lkData = new byte[BitConverter.ToInt32(mcvtLkWhole, 4)];
            Buffer.BlockCopy(mcvtLkWhole, 8, lkData, 0, lkData.Length);
            alphaMcvtRaw = ConvertMcvtLkToAlpha(lkData, lkHeader.PosZ);
        }
        // Apply debug-flat override if requested
        if (opts?.DebugFlatMcvt is float flatH)
        {
            alphaMcvtRaw = new byte[145 * 4];
            var fb = BitConverter.GetBytes(flatH);
            for (int i = 0; i < 145; i++)
            {
                Buffer.BlockCopy(fb, 0, alphaMcvtRaw, i * 4, 4);
            }
        }

        // Compose Alpha MCNK header
        var mcnrRaw = new byte[448]; // 145*3 + 13 pad
        // One base layer referencing texture 0. Layout (16 bytes):
        // int textureId; uint flags; uint ofsMCAL; int effectId
        var mclyData = new byte[16];
        // textureId = 0 by default; flags=0; ofsMCAL=0; effectId=0
        var mclyChunk = new Chunk("MCLY", mclyData.Length, mclyData);
        var mclyWhole = mclyChunk.GetWholeChunk();
        var mcrfEmpty = new Chunk("MCRF", 0, Array.Empty<byte>());
        var mcrfWhole = mcrfEmpty.GetWholeChunk();
        var mcshEmpty = new Chunk("MCSH", 0, Array.Empty<byte>());
        var mcshWhole = mcshEmpty.GetWholeChunk();
        var mcalEmpty = new Chunk("MCAL", 0, Array.Empty<byte>());
        var mcalWhole = mcalEmpty.GetWholeChunk();
        var hdr = new McnkAlphaHeader
        {
            Flags = 0,
            IndexX = lkHeader.IndexX,
            IndexY = lkHeader.IndexY,
            Unknown1 = 0,
            NLayers = 1,
            M2Number = 0,
            McvtOffset = 0, // first subchunk immediately after header
            McnrOffset = alphaMcvtRaw.Length,
            MclyOffset = alphaMcvtRaw.Length + mcnrRaw.Length,
            McrfOffset = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length,
            McalOffset = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length,
            McalSize = 0,
            McshOffset = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length,
            McshSize = 0,
            Unknown3 = (lkHeader.AreaId == 0 && opts?.ForceAreaId is int forced && forced > 0) ? forced : lkHeader.AreaId,
            WmoNumber = 0,
            Holes = 0,
            GroundEffectsMap1 = 0,
            GroundEffectsMap2 = 0,
            GroundEffectsMap3 = 0,
            GroundEffectsMap4 = 0,
            Unknown6 = 0,
            Unknown7 = 0,
            McnkChunksSize = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length, // total size of sub-blocks region
            Unknown8 = 0,
            MclqOffset = 0,
            Unused1 = 0,
            Unused2 = 0,
            Unused3 = 0,
            Unused4 = 0,
            Unused5 = 0,
            Unused6 = 0
        };

        int givenSize = McnkHeaderSize + alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length;

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

        // Sub-blocks in Alpha order: raw MCVT, raw MCNR, then named MCLY (empty)
        if (alphaMcvtRaw.Length > 0) ms.Write(alphaMcvtRaw, 0, alphaMcvtRaw.Length);
        if (mcnrRaw.Length > 0) ms.Write(mcnrRaw, 0, mcnrRaw.Length);
        ms.Write(mclyWhole, 0, mclyWhole.Length);
        ms.Write(mcrfWhole, 0, mcrfWhole.Length);
        ms.Write(mcshWhole, 0, mcshWhole.Length);
        ms.Write(mcalWhole, 0, mcalWhole.Length);

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

    private static byte[] ConvertMcvtLkToAlpha(byte[] mcvtLk, float baseZ)
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

        for (int i = 0; i < 9; i++)
        {
            // Outer row i: 9 floats
            for (int j = 0; j < outerRowFloats; j++)
            {
                float v = BitConverter.ToSingle(mcvtLk, src + j * floatSize) + baseZ;
                byte[] vb = BitConverter.GetBytes(v);
                Buffer.BlockCopy(vb, 0, alphaData, (i * outerRowFloats + j) * floatSize, floatSize);
            }
            src += outerRowBytes;

            // Inner row i: 8 floats (rows 0..7 only)
            if (i < 8)
            {
                int innerDestBase = outerBlockBytes + (i * innerRowBytes);
                for (int j = 0; j < innerRowFloats; j++)
                {
                    float v = BitConverter.ToSingle(mcvtLk, src + j * floatSize) + baseZ;
                    byte[] vb = BitConverter.GetBytes(v);
                    Buffer.BlockCopy(vb, 0, alphaData, innerDestBase + j * floatSize, floatSize);
                }
                src += innerRowBytes;
            }
        }
        return alphaData;
    }
}
