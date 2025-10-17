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

    public static byte[] BuildFromLk(byte[] lkAdtBytes, int mcNkOffset, LkToAlphaOptions? opts = null, byte[]? lkTexAdtBytes = null, int texMcNkOffset = -1)
    {
        int headerStart = mcNkOffset;
        // Read LK MCNK header to get IndexX/IndexY
        var lkHeader = ReadLkMcnkHeader(lkAdtBytes, mcNkOffset);

        // Find MCVT/MCNR chunks inside this LK terrain MCNK
        int mcnkSize = BitConverter.ToInt32(lkAdtBytes, mcNkOffset + 4);
        int subStart = mcNkOffset + ChunkLettersAndSize + McnkHeaderSize;
        int subEnd = mcNkOffset + 8 + mcnkSize;
        if (subEnd > lkAdtBytes.Length) subEnd = lkAdtBytes.Length;
        
        // Debug output removed - extraction confirmed working

        byte[]? mcvtLkWhole = null;
        byte[]? mcnrLkWhole = null;
        byte[]? mclyLkWhole = null;
        byte[]? mcalLkWhole = null;
        byte[]? mcshLkWhole = null;
        byte[]? mcseLkWhole = null;
        
        // Extract MCLY, MCAL, MCSH using header offsets (LK stores them as proper chunks)
        if (lkHeader.MclyOffset > 0)
        {
            int mclyPos = mcNkOffset + lkHeader.MclyOffset;
            if (mclyPos + 8 <= lkAdtBytes.Length)
            {
                int mclySize = BitConverter.ToInt32(lkAdtBytes, mclyPos + 4);
                if (mclySize > 0 && mclyPos + 8 + mclySize <= lkAdtBytes.Length)
                {
                    mclyLkWhole = new byte[8 + mclySize + ((mclySize & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkAdtBytes, mclyPos, mclyLkWhole, 0, mclyLkWhole.Length);
                }
            }
        }
        
        if (lkHeader.McalOffset > 0 && lkHeader.McalSize > 0)
        {
            int mcalPos = mcNkOffset + lkHeader.McalOffset;
            if (mcalPos + 8 <= lkAdtBytes.Length)
            {
                int mcalSize = BitConverter.ToInt32(lkAdtBytes, mcalPos + 4);
                if (mcalSize > 0 && mcalPos + 8 + mcalSize <= lkAdtBytes.Length)
                {
                    mcalLkWhole = new byte[8 + mcalSize + ((mcalSize & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkAdtBytes, mcalPos, mcalLkWhole, 0, mcalLkWhole.Length);
                }
            }
        }
        
        if (lkHeader.McshOffset > 0 && lkHeader.McshOffset != lkHeader.McalOffset)
        {
            int mcshPos = mcNkOffset + lkHeader.McshOffset;
            if (mcshPos + 8 <= lkAdtBytes.Length)
            {
                int mcshSize = BitConverter.ToInt32(lkAdtBytes, mcshPos + 4);
                if (mcshSize > 0 && mcshPos + 8 + mcshSize <= lkAdtBytes.Length)
                {
                    mcshLkWhole = new byte[8 + mcshSize + ((mcshSize & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkAdtBytes, mcshPos, mcshLkWhole, 0, mcshLkWhole.Length);
                }
            }
        }
        
        // Scan for MCVT and MCNR (these are in the sub-chunk area)
        for (int p = subStart; p + 8 <= subEnd;)
        {
            // Validate bounds before accessing
            if (p < 0 || p + 4 > lkAdtBytes.Length) break;
            
            string fcc = Encoding.ASCII.GetString(lkAdtBytes, p, 4);
            int size = BitConverter.ToInt32(lkAdtBytes, p + 4);
            
            // Validate size to prevent issues
            if (size < 0 || size > lkAdtBytes.Length) break;
            
            int dataStart = p + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > subEnd) break;
            
            // Ensure forward progress
            if (next <= p) break;

            if (fcc == "TVCM") // 'MCVT' reversed on disk, read as ASCII gives us "TVCM"
            {
                mcvtLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                Buffer.BlockCopy(lkAdtBytes, p, mcvtLkWhole, 0, mcvtLkWhole.Length);
            }
            else if (fcc == "RNCM") // 'MCNR' reversed on disk, read as ASCII gives us "RNCM"
            {
                mcnrLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                Buffer.BlockCopy(lkAdtBytes, p, mcnrLkWhole, 0, mcnrLkWhole.Length);
            }
            else if (fcc == "ESCM") // 'MCSE' reversed on disk
            {
                mcseLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                Buffer.BlockCopy(lkAdtBytes, p, mcseLkWhole, 0, mcseLkWhole.Length);
            }
            // Note: MCLY, MCAL, MCSH are extracted via header offsets above, not scanned here
            
            p = next;
        }
        
        // If texture ADT provided, scan it for MCLY/MCAL/MCSH/MCSE
        if (lkTexAdtBytes != null && texMcNkOffset >= 0)
        {
            int texSubStart = texMcNkOffset + ChunkLettersAndSize + McnkHeaderSize;
            int texSubEnd = texMcNkOffset + 8 + BitConverter.ToInt32(lkTexAdtBytes, texMcNkOffset + 4);
            if (texSubEnd > lkTexAdtBytes.Length) texSubEnd = lkTexAdtBytes.Length;
            
            for (int p = texSubStart; p + 8 <= texSubEnd;)
            {
                if (p < 0 || p + 4 > lkTexAdtBytes.Length) break;
                
                string fcc = Encoding.ASCII.GetString(lkTexAdtBytes, p, 4);
                int size = BitConverter.ToInt32(lkTexAdtBytes, p + 4);
                
                if (size < 0 || size > lkTexAdtBytes.Length) break;
                
                int dataStart = p + 8;
                int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
                if (dataStart + size > texSubEnd) break;
                if (next <= p) break;
                
                if (fcc == "YLCM") // 'MCLY' reversed on disk
                {
                    mclyLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p, mclyLkWhole, 0, mclyLkWhole.Length);
                }
                else if (fcc == "LACM") // 'MCAL' reversed on disk
                {
                    mcalLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p, mcalLkWhole, 0, mcalLkWhole.Length);
                }
                else if (fcc == "HSCM") // 'MCSH' reversed on disk
                {
                    mcshLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p, mcshLkWhole, 0, mcshLkWhole.Length);
                }
                else if (fcc == "ESCM") // 'MCSE' reversed on disk
                {
                    mcseLkWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p, mcseLkWhole, 0, mcseLkWhole.Length);
                }
                
                p = next;
            }
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

        // Build alpha raw MCNR data (normals, no named subchunk in Alpha)
        byte[] mcnrRaw;
        if (mcnrLkWhole != null)
        {
            // Extract LK MCNR data
            int mcnrSize = BitConverter.ToInt32(mcnrLkWhole, 4);
            var lkData = new byte[mcnrSize];
            Buffer.BlockCopy(mcnrLkWhole, 8, lkData, 0, mcnrSize);
            
            // Convert LK-order MCNR to Alpha-order
            mcnrRaw = ConvertMcnrLkToAlpha(lkData);
        }
        else
        {
            // Fallback: empty normals if MCNR missing from LK
            mcnrRaw = new byte[448];
        }

        // Build MCLY raw - use extracted LK data or create minimal fallback
        byte[] mclyRaw;
        if (mclyLkWhole != null && mclyLkWhole.Length > 8)
        {
            // Strip LK chunk header (8 bytes) -> raw table
            int sz = BitConverter.ToInt32(mclyLkWhole, 4);
            mclyRaw = new byte[sz];
            Buffer.BlockCopy(mclyLkWhole, 8, mclyRaw, 0, sz);
        }
        else
        {
            // Fallback: One base layer referencing texture 0. Layout (16 bytes):
            // uint32 textureId; uint32 props; uint32 offsAlpha; uint16 effectId; uint8 pad[2]
            mclyRaw = new byte[16];
            // all zeros is acceptable minimal layer
        }
        
        // Build MCAL raw - use extracted LK data or create empty fallback
        byte[] mcalRaw;
        if (mcalLkWhole != null && mcalLkWhole.Length > 8)
        {
            int sz = BitConverter.ToInt32(mcalLkWhole, 4);
            mcalRaw = new byte[sz];
            Buffer.BlockCopy(mcalLkWhole, 8, mcalRaw, 0, sz);
            DumpMcalData("lk", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);
        }
        else
        {
            mcalRaw = Array.Empty<byte>();
        }
        
        // Build MCSH raw - use extracted LK data or create empty fallback
        byte[] mcshRaw;
        if (mcshLkWhole != null && mcshLkWhole.Length > 8)
        {
            int sz = BitConverter.ToInt32(mcshLkWhole, 4);
            mcshRaw = new byte[sz];
            Buffer.BlockCopy(mcshLkWhole, 8, mcshRaw, 0, sz);
        }
        else
        {
            mcshRaw = Array.Empty<byte>();
        }
        
        // Build MCSE raw - use extracted LK data or create empty
        byte[] mcseRaw;
        if (mcseLkWhole != null && mcseLkWhole.Length > 8)
        {
            int sz = BitConverter.ToInt32(mcseLkWhole, 4);
            mcseRaw = new byte[sz];
            Buffer.BlockCopy(mcseLkWhole, 8, mcseRaw, 0, sz);
        }
        else
        {
            mcseRaw = Array.Empty<byte>();
        }

        // MCRF raw is empty in our current flow
        var mcrfRaw = Array.Empty<byte>();

        // Build named chunk wrappers (Alpha expects FourCC headers for these)
        var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
        var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);
        var mcshChunk = new Chunk("MCSH", mcshRaw.Length, mcshRaw);
        DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);

        var mcalChunk = new Chunk("MCAL", mcalRaw.Length, mcalRaw);
        var mcseChunk = new Chunk("MCSE", mcseRaw.Length, mcseRaw);

        byte[] mclyWhole = mclyChunk.GetWholeChunk();
        byte[] mcrfWhole = mcrfChunk.GetWholeChunk();
        byte[] mcshWhole = mcshChunk.GetWholeChunk();
        byte[] mcalWhole = mcalChunk.GetWholeChunk();
        byte[] mcseWhole = mcseChunk.GetWholeChunk();

        int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length + mcseWhole.Length;
        
        // Calculate bounding sphere radius from MCVT heights
        float radius = CalculateRadius(alphaMcvtRaw);
        
        // Calculate number of texture layers from MCLY raw table (each entry is 16 bytes)
        int nLayers = mclyRaw.Length / 16;
        
        // Compute Alpha SMChunk header fields (offsets relative to BEGINNING of MCNK chunk)
        const int headerTotal = 8 + McnkHeaderSize; // FourCC+size + 128-byte header
        int offsHeight = headerTotal; // MCVT raw starts immediately after header
        int offsNormal = offsHeight + alphaMcvtRaw.Length;
        int offsLayer  = offsNormal + mcnrRaw.Length;
        int offsRefs   = offsLayer  + mclyWhole.Length;
        int offsShadow = offsRefs   + mcrfWhole.Length;
        int offsAlpha  = offsShadow + mcshWhole.Length;
        int offsSnd    = offsAlpha  + mcalWhole.Length;

        int sizeShadow = mcshRaw.Length;
        int sizeAlpha  = mcalRaw.Length;
        int sizeSnd    = mcseRaw.Length;

        // Best-effort nSndEmitters detection: prefer 76-byte entries (0.5.3), else 52-byte (1.12.1), else 0
        int nSnd = 0;
        if (sizeSnd > 0)
        {
            if ((sizeSnd % 76) == 0) nSnd = sizeSnd / 76;
            else if ((sizeSnd % 52) == 0) nSnd = sizeSnd / 52;
        }
        int areaIdVal  = (lkHeader.AreaId == 0 && opts?.ForceAreaId is int forced && forced > 0) ? forced : lkHeader.AreaId;

        int givenSize = McnkHeaderSize + alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length + mcseWhole.Length;

        using var ms = new MemoryStream();
        // Write MCNK letters reversed ('KNCM')
        var reversedLetters = Encoding.ASCII.GetBytes("KNCM");
        ms.Write(reversedLetters, 0, 4);
        ms.Write(BitConverter.GetBytes(givenSize), 0, 4);

        // Build Alpha SMChunk header (128 bytes)
        Span<byte> smh = stackalloc byte[McnkHeaderSize];
        smh.Clear();
        // Offsets within header (see Alpha.md):
        // 0x00 flags
        // 0x04 indexX
        // 0x08 indexY
        // 0x0C radius (float)
        // 0x10 nLayers
        // 0x14 nDoodadRefs
        // 0x18 offsHeight (MCVT)
        // 0x1C offsNormal (MCNR)
        // 0x20 offsLayer (MCLY)
        // 0x24 offsRefs (MCRF)
        // 0x28 offsAlpha (MCAL)
        // 0x2C sizeAlpha
        // 0x30 offsShadow (MCSH)
        // 0x34 sizeShadow
        // 0x38 areaid
        // 0x3C nMapObjRefs
        // 0x40 holes (uint16)
        // 0x42 pad0 (uint16)
        // 0x44 predTex[8] (uint16 x8)
        // 0x54 noEffectDoodad[8] (uint8 x8)
        // 0x5C offsSndEmitters
        // 0x60 nSndEmitters
        // 0x64 offsLiquid
        // 0x68 pad1[24]

        BitConverter.GetBytes(0).CopyTo(smh[0x00..]); // flags
        BitConverter.GetBytes(lkHeader.IndexX).CopyTo(smh[0x04..]);
        BitConverter.GetBytes(lkHeader.IndexY).CopyTo(smh[0x08..]);
        BitConverter.GetBytes(radius).CopyTo(smh[0x0C..]);
        BitConverter.GetBytes(nLayers).CopyTo(smh[0x10..]);
        BitConverter.GetBytes(0).CopyTo(smh[0x14..]); // nDoodadRefs (we write empty MCRF)
        BitConverter.GetBytes(offsHeight).CopyTo(smh[0x18..]);
        BitConverter.GetBytes(offsNormal).CopyTo(smh[0x1C..]);
        BitConverter.GetBytes(offsLayer).CopyTo(smh[0x20..]);
        BitConverter.GetBytes(offsRefs).CopyTo(smh[0x24..]);
        BitConverter.GetBytes(offsAlpha).CopyTo(smh[0x28..]);
        BitConverter.GetBytes(sizeAlpha).CopyTo(smh[0x2C..]);
        BitConverter.GetBytes(offsShadow).CopyTo(smh[0x30..]);
        BitConverter.GetBytes(sizeShadow).CopyTo(smh[0x34..]);
        BitConverter.GetBytes(areaIdVal).CopyTo(smh[0x38..]);
        BitConverter.GetBytes(0).CopyTo(smh[0x3C..]); // nMapObjRefs
        // holes (uint16) at 0x40, pad0 at 0x42 (leave zeros)
        // predTex[8] at 0x44, noEffectDoodad[8] at 0x54 (already zeros)
        BitConverter.GetBytes(offsSnd).CopyTo(smh[0x5C..]);
        BitConverter.GetBytes(nSnd).CopyTo(smh[0x60..]);
        // offsLiquid zero by default

        // Write header
        ms.Write(smh);

        // Sub-blocks in Alpha order (raw, no named headers): MCVT, MCNR, MCLY, MCRF, MCSH, MCAL, MCSE
        if (alphaMcvtRaw.Length > 0) ms.Write(alphaMcvtRaw, 0, alphaMcvtRaw.Length);
        if (mcnrRaw.Length > 0) ms.Write(mcnrRaw, 0, mcnrRaw.Length);
        ms.Write(mclyWhole, 0, mclyWhole.Length);
        ms.Write(mcrfWhole, 0, mcrfWhole.Length);
        ms.Write(mcshWhole, 0, mcshWhole.Length);
        ms.Write(mcalWhole, 0, mcalWhole.Length);
        ms.Write(mcseWhole, 0, mcseWhole.Length);

        return ms.ToArray();
    }

    private static void DumpMcalData(string stage, int indexX, int indexY, byte[] data, LkToAlphaOptions? opts)
    {
        if (opts?.VerboseLogging != true) return;
        if (data is null || data.Length == 0) return;

        try
        {
            string root = Path.Combine("debug_mcal", $"{indexY:D2}_{indexX:D2}");
            Directory.CreateDirectory(root);
            string path = Path.Combine(root, $"{stage}_mcal.bin");
            File.WriteAllBytes(path, data);
            Console.WriteLine($"[dump] MCAL {stage} -> {path} ({data.Length} bytes)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[dump] Failed to write MCAL {stage} for tile {indexY:D2}_{indexX:D2}: {ex.Message}");
        }
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
    
    private static byte[] ConvertMcnrLkToAlpha(byte[] mcnrLk)
    {
        // LK order is interleaved [outer row 0 (9 normals), inner row 0 (8 normals), ..., outer row 8 (9 normals)].
        // Alpha order requires all outer 9x9 first, then concatenated 8 inner rows of 8 normals each, then 13 pad.
        // Each normal is 3 bytes (X, Y, Z as signed bytes).
        const int normalSize = 3;
        const int outerRowNormals = 9;
        const int innerRowNormals = 8;
        const int outerRowBytes = outerRowNormals * normalSize; // 27
        const int innerRowBytes = innerRowNormals * normalSize; // 24
        const int outerBlockBytes = outerRowBytes * 9; // 243 (81 normals × 3)
        const int innerBlockBytes = innerRowBytes * 8; // 192 (64 normals × 3)
        const int paddingBytes = 13;
        
        var alphaData = new byte[outerBlockBytes + innerBlockBytes + paddingBytes]; // 448 bytes
        int src = 0;

        // Reorder from interleaved to outer-first
        for (int i = 0; i < 9; i++)
        {
            // Outer row i: 9 normals (27 bytes)
            int outerDest = i * outerRowBytes;
            if (src + outerRowBytes <= mcnrLk.Length)
            {
                Buffer.BlockCopy(mcnrLk, src, alphaData, outerDest, outerRowBytes);
            }
            src += outerRowBytes;

            // Inner row i: 8 normals (24 bytes) - rows 0..7 only
            if (i < 8)
            {
                int innerDest = outerBlockBytes + (i * innerRowBytes);
                if (src + innerRowBytes <= mcnrLk.Length)
                {
                    Buffer.BlockCopy(mcnrLk, src, alphaData, innerDest, innerRowBytes);
                }
                src += innerRowBytes;
            }
        }
        
        // Padding bytes are already zero-initialized
        return alphaData;
    }
    
    private static float CalculateRadius(byte[] mcvtRaw)
    {
        // MCVT contains 145 floats (9x9 outer + 8x8 inner)
        if (mcvtRaw.Length < 145 * 4) return 0f;
        
        float minH = float.MaxValue;
        float maxH = float.MinValue;
        
        for (int i = 0; i < 145; i++)
        {
            float h = BitConverter.ToSingle(mcvtRaw, i * 4);
            if (h < minH) minH = h;
            if (h > maxH) maxH = h;
        }
        
        // Simple bounding sphere: use height range as radius approximation
        // Real calculation would be sqrt(dx^2 + dy^2 + dz^2) but this is close enough
        // Alpha chunks are ~33.33 units wide, so diagonal is ~47 units
        // Add height range for vertical component
        float heightRange = maxH - minH;
        float horizontalRadius = 23.57f; // ~sqrt(33.33^2 + 33.33^2) / 2
        
        // Combine horizontal and vertical components
        return (float)Math.Sqrt(horizontalRadius * horizontalRadius + (heightRange / 2) * (heightRange / 2));
    }
}
