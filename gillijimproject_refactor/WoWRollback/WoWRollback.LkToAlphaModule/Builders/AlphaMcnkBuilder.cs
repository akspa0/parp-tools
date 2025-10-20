using System;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using Util = GillijimProject.Utilities.Utilities;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Mcal;
using WoWRollback.LkToAlphaModule.Liquids;

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
        byte[]? mcrfLkWhole = null;
        byte[]? mh2oLkWhole = null;

        // Extract MCLY, MCAL, MCSH using header offsets (LK stores them as proper chunks)
        if (lkHeader.MclyOffset > 0)
        {
            int mclyPos = mcNkOffset + lkHeader.MclyOffset;
            if (mclyPos + 8 <= lkAdtBytes.Length)
            {
                int mclySize = BitConverter.ToInt32(lkAdtBytes, mclyPos + 4);
                if (mclySize > 0 && mclyPos + 8 + mclySize <= lkAdtBytes.Length)
                {
                    mclyLkWhole = new byte[mclySize];
                    Buffer.BlockCopy(lkAdtBytes, mclyPos + 8, mclyLkWhole, 0, mclySize);
                }
            }
        }
        else if (opts?.VerboseLogging == true)
        {
            Console.WriteLine($"[MCLY] MCNK {lkHeader.IndexX},{lkHeader.IndexY}: MclyOffset=0, nLayers={lkHeader.NLayers}");
        }
        if (lkHeader.McalOffset > 0 && lkHeader.McalSize > 0)
        {
            int mcalPos = mcNkOffset + lkHeader.McalOffset;
            if (mcalPos + 8 <= lkAdtBytes.Length)
            {
                int mcalSize = BitConverter.ToInt32(lkAdtBytes, mcalPos + 4);
                if (mcalSize > 0 && mcalPos + 8 + mcalSize <= lkAdtBytes.Length)
                {
                    mcalLkWhole = new byte[mcalSize];
                    Buffer.BlockCopy(lkAdtBytes, mcalPos + 8, mcalLkWhole, 0, mcalSize);
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
                    mcshLkWhole = new byte[mcshSize];
                    Buffer.BlockCopy(lkAdtBytes, mcshPos + 8, mcshLkWhole, 0, mcshSize);
                }
            }
        }
        
        if (lkHeader.McrfOffset > 0)
        {
            int mcrfPos = mcNkOffset + lkHeader.McrfOffset;
            if (mcrfPos + 8 <= lkAdtBytes.Length)
            {
                int mcrfSize = BitConverter.ToInt32(lkAdtBytes, mcrfPos + 4);
                if (mcrfSize > 0 && mcrfPos + 8 + mcrfSize <= lkAdtBytes.Length)
                {
                    mcrfLkWhole = new byte[mcrfSize];
                    Buffer.BlockCopy(lkAdtBytes, mcrfPos + 8, mcrfLkWhole, 0, mcrfSize);
                }
            }
        }
        
        // Extract MH2O (liquids) if present and not skipped
        if (lkHeader.MclqOffset > 0 && opts?.SkipLiquids != true)
        {
            int mh2oPos = mcNkOffset + lkHeader.MclqOffset;
            if (mh2oPos + 8 <= lkAdtBytes.Length)
            {
                int mh2oSize = BitConverter.ToInt32(lkAdtBytes, mh2oPos + 4);
                if (mh2oSize > 0 && mh2oPos + 8 + mh2oSize <= lkAdtBytes.Length)
                {
                    mh2oLkWhole = new byte[8 + mh2oSize]; // Include chunk header
                    Buffer.BlockCopy(lkAdtBytes, mh2oPos, mh2oLkWhole, 0, 8 + mh2oSize);
                }
            }
        }
        
        // Extract MCSE (sound emitters) if present
        if (lkHeader.McseOffset > 0)
        {
            int mcsePos = mcNkOffset + lkHeader.McseOffset;
            if (mcsePos + 8 <= lkAdtBytes.Length)
            {
                int mcseSize = BitConverter.ToInt32(lkAdtBytes, mcsePos + 4);
                if (mcseSize > 0 && mcsePos + 8 + mcseSize <= lkAdtBytes.Length)
                {
                    mcseLkWhole = new byte[mcseSize];
                    Buffer.BlockCopy(lkAdtBytes, mcsePos + 8, mcseLkWhole, 0, mcseSize);
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
            else if (fcc == "YLCM" && mclyLkWhole == null) // 'MCLY' reversed - fallback if header offset was 0
            {
                mclyLkWhole = new byte[size];
                Buffer.BlockCopy(lkAdtBytes, p + 8, mclyLkWhole, 0, size);
            }
            else if (fcc == "LACM" && mcalLkWhole == null) // 'MCAL' reversed - fallback if header offset was 0
            {
                mcalLkWhole = new byte[size];
                Buffer.BlockCopy(lkAdtBytes, p + 8, mcalLkWhole, 0, size);
            }
            else if (fcc == "HSCM" && mcshLkWhole == null) // 'MCSH' reversed - fallback if header offset was 0
            {
                mcshLkWhole = new byte[size];
                Buffer.BlockCopy(lkAdtBytes, p + 8, mcshLkWhole, 0, size);
            }
            // Note: MCLY, MCAL, MCSH are primarily extracted via header offsets above, but we scan as fallback
            
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
                    mclyLkWhole = new byte[size];
                    Buffer.BlockCopy(lkTexAdtBytes, p + 8, mclyLkWhole, 0, size);
                }
                else if (fcc == "LACM") // 'MCAL' reversed on disk
                {
                    mcalLkWhole = new byte[size];
                    Buffer.BlockCopy(lkTexAdtBytes, p + 8, mcalLkWhole, 0, size);
                }
                else if (fcc == "HSCM") // 'MCSH' reversed on disk
                {
                    mcshLkWhole = new byte[size];
                    Buffer.BlockCopy(lkTexAdtBytes, p + 8, mcshLkWhole, 0, size);
                }
                else if (fcc == "ESCM") // 'MCSE' reversed on disk
                {
                    mcseLkWhole = new byte[size];
                    Buffer.BlockCopy(lkTexAdtBytes, p + 8, mcseLkWhole, 0, size);
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

        // Build MCLY raw - prefer passthrough of LK payload when present
        byte[] mclyRaw;
        if (mclyLkWhole != null && mclyLkWhole.Length > 0)
        {
            mclyRaw = new byte[mclyLkWhole.Length];
            Buffer.BlockCopy(mclyLkWhole, 0, mclyRaw, 0, mclyRaw.Length);
        }
        else
        {
            mclyRaw = new byte[16];
        }

        DumpMclyTable(lkHeader.IndexX, lkHeader.IndexY, mclyRaw, opts);

        // Build MCAL raw
        byte[] mcalRaw;
        int layerCount = mclyRaw.Length / 16;
        bool hasAlphaFlags = mclyRaw.Length >= 16 && LayerUsesAlpha(mclyRaw);
        if (opts?.UseManagedBuilders == false)
        {
            // Passthrough: use LK MCAL payload directly and do not modify MCLY offsets
            mcalRaw = (mcalLkWhole != null) ? new ReadOnlySpan<byte>(mcalLkWhole).ToArray() : Array.Empty<byte>();
        }
        else if (mcalLkWhole != null && mcalLkWhole.Length > 0 && hasAlphaFlags)
        {
            int totalMcalSize = mcalLkWhole.Length;
            var mcalSource = new ReadOnlySpan<byte>(mcalLkWhole, 0, totalMcalSize);
            int maxAlphaLayers = Math.Max(0, layerCount - 1);
            var assembled = new byte[4096 * maxAlphaLayers];
            int writeOffset = 0;
            if (opts?.VerboseLogging == true)
            {
                Console.WriteLine($"[MCAL] Tile ({lkHeader.IndexX},{lkHeader.IndexY}) layers={layerCount} totalSize={totalMcalSize}");
            }
            for (int layer = 0; layer < layerCount; layer++)
            {
                int entryOffset = layer * 16;
                uint flags = BitConverter.ToUInt32(mclyRaw, entryOffset + 4);
                bool usesAlpha = (flags & 0x80) != 0;
                if (!usesAlpha && (flags & 0x100) != 0 && totalMcalSize > 0)
                {
                    flags |= 0x80;
                    BitConverter.GetBytes(flags).CopyTo(mclyRaw, entryOffset + 4);
                    usesAlpha = true;
                    if (opts?.VerboseLogging == true)
                    {
                        Console.WriteLine($"    -> patched missing alpha bit (flags now 0x{flags:X})");
                    }
                }
                if (opts?.VerboseLogging == true)
                {
                    uint textureId = BitConverter.ToUInt32(mclyRaw, entryOffset + 0);
                    uint rawOffset = BitConverter.ToUInt32(mclyRaw, entryOffset + 8);
                    Console.WriteLine($"  [Layer {layer}] tex={textureId} flags=0x{flags:X} rawOffset={rawOffset} usesAlpha={usesAlpha}");
                }
                if (!usesAlpha)
                {
                    BitConverter.GetBytes(0u).CopyTo(mclyRaw, entryOffset + 8);
                    continue;
                }
                int originalOffset = (int)BitConverter.ToUInt32(mclyRaw, entryOffset + 8);
                int start = Math.Min(originalOffset, totalMcalSize);
                int end = totalMcalSize;
                for (int next = layer + 1; next < layerCount; next++)
                {
                    uint nextFlags = BitConverter.ToUInt32(mclyRaw, next * 16 + 4);
                    if ((nextFlags & 0x80) != 0)
                    {
                        end = Math.Min(end, (int)BitConverter.ToUInt32(mclyRaw, next * 16 + 8));
                        break;
                    }
                }
                if (start < 0) start = 0;
                if (end < start) end = start;
                int length = Math.Max(0, Math.Min(totalMcalSize - start, end - start));
                var layerSpan = mcalSource.Slice(start, length);
                var converted = McalAlphaDecoder.DecodeToColumnMajor(layerSpan, flags, opts?.DisableAlphaEdgeFix != true);
                if (assembled.Length < writeOffset + converted.Length)
                {
                    Array.Resize(ref assembled, writeOffset + converted.Length);
                }
                Buffer.BlockCopy(converted, 0, assembled, writeOffset, converted.Length);
                BitConverter.GetBytes((uint)writeOffset).CopyTo(mclyRaw, entryOffset + 8);
                if (opts?.VerboseLogging == true)
                {
                    uint textureId = BitConverter.ToUInt32(mclyRaw, entryOffset + 0);
                    Console.WriteLine($"    -> usesAlpha start={start} end={end} len={length} newOffset={writeOffset} outLen={converted.Length}");
                }
                writeOffset += converted.Length;
            }
            Array.Resize(ref assembled, writeOffset);
            DumpMcalData("lk", lkHeader.IndexX, lkHeader.IndexY, new ReadOnlySpan<byte>(assembled).ToArray(), opts);
            mcalRaw = assembled;
        }
        else
        {
            mcalRaw = Array.Empty<byte>();
            for (int layer = 0; layer < layerCount; layer++)
            {
                BitConverter.GetBytes(0u).CopyTo(mclyRaw, layer * 16 + 8);
            }
        }
        
        // Build MCSH raw - use extracted LK data or create empty fallback
        byte[] mcshRaw;
        if (mcshLkWhole != null && mcshLkWhole.Length > 0)
        {
            mcshRaw = new byte[mcshLkWhole.Length];
            Buffer.BlockCopy(mcshLkWhole, 0, mcshRaw, 0, mcshRaw.Length);
        }
        else
        {
            mcshRaw = Array.Empty<byte>();
        }
        
        // Build MCSE raw - convert LK format (52 bytes) to Alpha format (76 bytes)
        byte[] mcseRaw;
        if (mcseLkWhole != null && mcseLkWhole.Length > 0)
        {
            try
            {
                mcseRaw = ConvertMcseLkToAlpha(mcseLkWhole, opts);
                if (opts?.VerboseLogging == true && mcseRaw.Length > 0)
                {
                    int lkCount = mcseLkWhole.Length / 52;
                    int alphaCount = mcseRaw.Length / 76;
                    Console.WriteLine($"[MCSE] Tile ({lkHeader.IndexX},{lkHeader.IndexY}) converted {lkCount} LK entries -> {alphaCount} Alpha entries");
                }
            }
            catch (Exception ex)
            {
                if (opts?.VerboseLogging == true)
                {
                    Console.WriteLine($"[MCSE] Tile ({lkHeader.IndexX},{lkHeader.IndexY}) conversion failed: {ex.Message}");
                }
                mcseRaw = Array.Empty<byte>();
            }
        }
        else
        {
            mcseRaw = Array.Empty<byte>();
        }
        
        // Build MCLQ raw - convert MH2O to Alpha MCLQ if liquids enabled
        byte[] mclqRaw = Array.Empty<byte>();
        if (mh2oLkWhole != null && mh2oLkWhole.Length > 0 && opts?.SkipLiquids != true)
        {
            try
            {
                mclqRaw = ConvertMh2oToMclq(mh2oLkWhole, opts);
                if (opts?.VerboseLogging == true && mclqRaw.Length > 0)
                {
                    Console.WriteLine($"[MCLQ] Tile ({lkHeader.IndexX},{lkHeader.IndexY}) converted MH2O -> MCLQ ({mclqRaw.Length} bytes)");
                }
            }
            catch (Exception ex)
            {
                if (opts?.VerboseLogging == true)
                {
                    Console.WriteLine($"[MCLQ] Tile ({lkHeader.IndexX},{lkHeader.IndexY}) conversion failed: {ex.Message}");
                }
                mclqRaw = Array.Empty<byte>();
            }
        }

        // MCRF raw is empty in our current flow
        var mcrfRaw = BuildAlphaMcrfTable(mcrfLkWhole, lkHeader.M2Number, lkHeader.WmoNumber, out int alphaDoodadRefs, out int alphaWmoRefs);

        DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);

        byte[] mclyWhole = mclyRaw;
        byte[] mcrfWhole = mcrfRaw;
        byte[] mcshWhole = mcshRaw;
        byte[] mcalWhole = mcalRaw;
        byte[] mcseWhole = mcseRaw;
        byte[] mclqWhole = mclqRaw;

        int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length + mcseWhole.Length + mclqWhole.Length;
        
        // Calculate bounding sphere radius from MCVT heights
        float radius = CalculateRadius(alphaMcvtRaw);
        
        // Calculate number of texture layers from MCLY raw table (each entry is 16 bytes)
        
        // Compute Alpha SMChunk header fields (offsets relative to BEGINNING of MCNK chunk)
        const int headerTotal = 8 + McnkHeaderSize; // FourCC+size + 128-byte header
        const int mclyChunkHeaderSize = 8; // MCLY has chunk header in Alpha: "YLCM" + size
        int offsHeight = 0; // Offsets are relative to the subchunk data region (immediately after header)
        int offsNormal = offsHeight + alphaMcvtRaw.Length;
        int offsLayer  = offsNormal + mcnrRaw.Length;
        int offsRefs   = offsLayer  + mclyChunkHeaderSize + mclyWhole.Length; // MCLY includes 8-byte chunk header
        int offsShadow = offsRefs   + mcrfWhole.Length;
        int offsAlpha  = offsShadow + mcshWhole.Length;
        int offsSnd    = mcseWhole.Length > 0 ? offsAlpha + mcalWhole.Length : 0;
        int offsLiquid = mclqWhole.Length > 0 ? (offsSnd > 0 ? offsSnd + mcseWhole.Length : offsAlpha + mcalWhole.Length) : 0;

        int sizeShadow = mcshRaw.Length;
        int sizeAlpha  = mcalRaw.Length;
        int sizeSnd    = mcseRaw.Length;
        int sizeLiquid = mclqRaw.Length;

        // Best-effort nSndEmitters detection: prefer 76-byte entries (0.5.3), else 52-byte (1.12.1), else 0
        int nSnd = 0;
        if (sizeSnd > 0)
        {
            if ((sizeSnd % 76) == 0) nSnd = sizeSnd / 76;
            else if ((sizeSnd % 52) == 0) nSnd = sizeSnd / 52;
        }
        int areaIdVal  = (lkHeader.AreaId == 0 && opts?.ForceAreaId is int forced && forced > 0) ? forced : lkHeader.AreaId;

        int givenSize = McnkHeaderSize + alphaMcvtRaw.Length + mcnrRaw.Length + mclyChunkHeaderSize + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length + mcseWhole.Length + mclqWhole.Length;

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

        BitConverter.GetBytes(lkHeader.Flags).CopyTo(smh[0x00..]);
        BitConverter.GetBytes(lkHeader.IndexX).CopyTo(smh[0x04..]);
        BitConverter.GetBytes(lkHeader.IndexY).CopyTo(smh[0x08..]);
        BitConverter.GetBytes(radius).CopyTo(smh[0x0C..]);
        BitConverter.GetBytes(layerCount).CopyTo(smh[0x10..]);
        BitConverter.GetBytes(alphaDoodadRefs).CopyTo(smh[0x14..]);
        BitConverter.GetBytes(offsHeight).CopyTo(smh[0x18..]);
        BitConverter.GetBytes(offsNormal).CopyTo(smh[0x1C..]);
        BitConverter.GetBytes(offsLayer).CopyTo(smh[0x20..]);
        BitConverter.GetBytes(offsRefs).CopyTo(smh[0x24..]);
        BitConverter.GetBytes(offsAlpha).CopyTo(smh[0x28..]);
        BitConverter.GetBytes(sizeAlpha).CopyTo(smh[0x2C..]);
        BitConverter.GetBytes(offsShadow).CopyTo(smh[0x30..]);
        BitConverter.GetBytes(sizeShadow).CopyTo(smh[0x34..]);
        BitConverter.GetBytes(areaIdVal).CopyTo(smh[0x38..]);
        BitConverter.GetBytes(alphaWmoRefs).CopyTo(smh[0x3C..]);
        BitConverter.GetBytes((ushort)lkHeader.Holes).CopyTo(smh[0x40..]);

        // predictor textures stored as int in LK header; pack low/high words into first two slots
        ushort predLow = (ushort)(lkHeader.PredTex & 0xFFFF);
        ushort predHigh = (ushort)((lkHeader.PredTex >> 16) & 0xFFFF);
        BitConverter.GetBytes(predLow).CopyTo(smh[0x44..]);
        BitConverter.GetBytes(predHigh).CopyTo(smh[0x46..]);

        uint noEffect = (uint)lkHeader.NEffectDoodad;
        smh[0x54] = (byte)(noEffect & 0xFF);
        smh[0x55] = (byte)((noEffect >> 8) & 0xFF);
        smh[0x56] = (byte)((noEffect >> 16) & 0xFF);
        smh[0x57] = (byte)((noEffect >> 24) & 0xFF);

        BitConverter.GetBytes(offsSnd).CopyTo(smh[0x5C..]);
        BitConverter.GetBytes(nSnd).CopyTo(smh[0x60..]);
        BitConverter.GetBytes(offsLiquid).CopyTo(smh[0x64..]);
        // Note: sizeLiquid not stored in Alpha header (0x68 onwards is padding)
        
        // Write header
        ms.Write(smh);

        // Sub-blocks in Alpha order: MCVT, MCNR (raw, no headers), MCLY (HAS header), MCRF, MCSH, MCAL, MCSE, MCLQ (raw)
        if (alphaMcvtRaw.Length > 0) ms.Write(alphaMcvtRaw, 0, alphaMcvtRaw.Length);
        if (mcnrRaw.Length > 0) ms.Write(mcnrRaw, 0, mcnrRaw.Length);
        // MCLY in Alpha format requires chunk header: "YLCM" + size + data
        ms.Write(Encoding.ASCII.GetBytes("YLCM"), 0, 4);
        ms.Write(BitConverter.GetBytes(mclyWhole.Length), 0, 4);
        ms.Write(mclyWhole, 0, mclyWhole.Length);
        ms.Write(mcrfWhole, 0, mcrfWhole.Length);
        ms.Write(mcshWhole, 0, mcshWhole.Length);
        ms.Write(mcalWhole, 0, mcalWhole.Length);
        if (mcseWhole.Length > 0) ms.Write(mcseWhole, 0, mcseWhole.Length);
        if (mclqWhole.Length > 0) ms.Write(mclqWhole, 0, mclqWhole.Length);

        return ms.ToArray();
    }

    private static byte[] BuildAlphaMcrfTable(byte[]? lkMcrfRaw, int doodadCount, int wmoCount, out int alphaDoodadRefs, out int alphaWmoRefs)
    {
        int safeDoodadCount = Math.Max(doodadCount, 0);
        int safeWmoCount = Math.Max(wmoCount, 0);
        int totalRefs = safeDoodadCount + safeWmoCount;
        alphaDoodadRefs = 0;
        alphaWmoRefs = 0;
        if (totalRefs == 0 || lkMcrfRaw is null || lkMcrfRaw.Length == 0)
        {
            return Array.Empty<byte>();
        }

        int availableRefs = lkMcrfRaw.Length / sizeof(int);
        if (availableRefs == 0)
        {
            return Array.Empty<byte>();
        }

        if (availableRefs < totalRefs)
        {
            totalRefs = availableRefs;
            safeDoodadCount = Math.Min(safeDoodadCount, totalRefs);
            safeWmoCount = Math.Min(safeWmoCount, totalRefs - safeDoodadCount);
        }

        int doodadBytes = safeDoodadCount * sizeof(int);
        int wmoBytes = safeWmoCount * sizeof(int);
        var output = new byte[doodadBytes + wmoBytes];

        alphaDoodadRefs = safeDoodadCount;
        alphaWmoRefs = safeWmoCount;

        if (doodadBytes > 0)
        {
            Buffer.BlockCopy(lkMcrfRaw, 0, output, 0, doodadBytes);
        }

        if (wmoBytes > 0)
        {
            int wmoSourceOffset = Math.Max(0, (availableRefs - safeWmoCount) * sizeof(int));
            Buffer.BlockCopy(lkMcrfRaw, wmoSourceOffset, output, doodadBytes, Math.Min(wmoBytes, lkMcrfRaw.Length - wmoSourceOffset));
        }

        return output;
    }

    private static bool LayerUsesAlpha(ReadOnlySpan<byte> mclyRaw)
    {
        for (int offset = 0; offset + 16 <= mclyRaw.Length; offset += 16)
        {
            uint flags = BitConverter.ToUInt32(mclyRaw.Slice(offset + 4, 4));
            if ((flags & 0x80) != 0) return true;
        }

        return false;
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

    private static void DumpMclyTable(int indexX, int indexY, byte[] table, LkToAlphaOptions? opts)
    {
        if (table is null || table.Length == 0) return;
        if (opts?.VerboseLogging == true)
        {
            Console.WriteLine($"[MCLY] Tile ({indexX},{indexY}) entries={table.Length / 16}");
            int layers = table.Length / 16;
            for (int layer = 0; layer < layers; layer++)
            {
                int offset = layer * 16;
                uint textureId = BitConverter.ToUInt32(table, offset + 0);
                uint flags = BitConverter.ToUInt32(table, offset + 4);
                uint rawOffset = BitConverter.ToUInt32(table, offset + 8);
                ushort effectId = BitConverter.ToUInt16(table, offset + 12);
                var hex = BitConverter.ToString(table, offset, Math.Min(16, table.Length - offset)).Replace("-", string.Empty);
                Console.WriteLine($"  [Layer {layer}] tex={textureId} flags=0x{flags:X} rawOffset={rawOffset} effect={effectId} raw={hex}");
            }

            try
            {
                string root = Path.Combine("debug_mcal", $"{indexY:D2}_{indexX:D2}");
                Directory.CreateDirectory(root);
                string path = Path.Combine(root, "mcly_raw.bin");
                File.WriteAllBytes(path, table.ToArray());
                Console.WriteLine($"[dump] MCLY raw -> {path} ({table.Length} bytes)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[dump] Failed to write MCLY raw for tile {indexY:D2}_{indexX:D2}: {ex.Message}");
            }
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

    private static byte[] ConvertMh2oToMclq(byte[] mh2oChunkBytes, LkToAlphaOptions? opts)
    {
        // Parse MH2O chunk - note: this is a single-chunk MH2O for one MCNK
        // The chunk format is: [4 bytes FourCC "MH2O"] [4 bytes size] [payload]
        if (mh2oChunkBytes.Length < 8)
            return Array.Empty<byte>();
            
        // Skip FourCC and size to get payload
        byte[] mh2oPayload = new byte[mh2oChunkBytes.Length - 8];
        Buffer.BlockCopy(mh2oChunkBytes, 8, mh2oPayload, 0, mh2oPayload.Length);
        
        // Parse MH2O using existing infrastructure
        // Note: MH2O in LK ADT is per-chunk, not per-ADT array
        var mh2oChunk = ParseSingleMh2oChunk(mh2oPayload);
        
        if (mh2oChunk == null || mh2oChunk.IsEmpty)
            return Array.Empty<byte>();
        
        // Convert MH2O to MCLQ using LiquidsConverter
        var liquidsOpts = new LiquidsOptions
        {
            EnableLiquids = true,
            Precedence = new[] { MclqLiquidType.Magma, MclqLiquidType.Slime, MclqLiquidType.River, MclqLiquidType.Ocean },
            GreenLava = false,
            Mapping = LiquidTypeMapping.CreateDefault()
        };
        
        var mclqData = LiquidsConverter.Mh2oToMclq(mh2oChunk, liquidsOpts);
        
        // Serialize MCLQ to Alpha format
        return SerializeMclq(mclqData);
    }
    
    private static Mh2oChunk? ParseSingleMh2oChunk(byte[] payload)
    {
        // Single-chunk MH2O format (per MCNK):
        // [12 bytes header: offsetInstances(4), layerCount(4), offsetAttributes(4)]
        // [instance data...]
        if (payload.Length < 12)
            return null;
            
        using var ms = new MemoryStream(payload);
        using var reader = new BinaryReader(ms);
        
        uint offsetInstances = reader.ReadUInt32();
        uint layerCount = reader.ReadUInt32();
        uint offsetAttributes = reader.ReadUInt32();
        
        if (layerCount == 0)
            return null;
            
        var chunk = new Mh2oChunk();
        
        // Read instances
        if (offsetInstances > 0 && offsetInstances < payload.Length)
        {
            ms.Position = offsetInstances;
            for (int i = 0; i < layerCount; i++)
            {
                if (ms.Position + 24 > payload.Length)
                    break;
                    
                var instance = new Mh2oInstance
                {
                    LiquidTypeId = reader.ReadUInt16(),
                    Lvf = (LiquidVertexFormat)reader.ReadUInt16(),
                    MinHeightLevel = reader.ReadSingle(),
                    MaxHeightLevel = reader.ReadSingle(),
                    XOffset = reader.ReadByte(),
                    YOffset = reader.ReadByte(),
                    Width = reader.ReadByte(),
                    Height = reader.ReadByte(),
                    ExistsBitmap = null,
                    HeightMap = null,
                    DepthMap = null
                };
                
                uint offsetExistsBitmap = reader.ReadUInt32();
                uint offsetVertexData = reader.ReadUInt32();
                
                // Read exists bitmap if present
                byte[]? existsBitmap = null;
                if (offsetExistsBitmap > 0 && offsetExistsBitmap < payload.Length)
                {
                    long savedPos = ms.Position;
                    ms.Position = offsetExistsBitmap;
                    int bitmapSize = (instance.Width * instance.Height + 7) / 8;
                    existsBitmap = reader.ReadBytes(bitmapSize);
                    ms.Position = savedPos;
                }
                
                // Read vertex data if present
                float[]? heightMap = null;
                byte[]? depthMap = null;
                if (offsetVertexData > 0 && offsetVertexData < payload.Length)
                {
                    long savedPos = ms.Position;
                    ms.Position = offsetVertexData;
                    
                    int vertexCount = (instance.Width + 1) * (instance.Height + 1);
                    switch (instance.Lvf)
                    {
                        case LiquidVertexFormat.HeightDepth:
                        {
                            heightMap = new float[vertexCount];
                            for (int v = 0; v < vertexCount; v++)
                                heightMap[v] = reader.ReadSingle();
                            depthMap = reader.ReadBytes(vertexCount);
                            break;
                        }
                        case LiquidVertexFormat.DepthOnly:
                        {
                            depthMap = reader.ReadBytes(vertexCount);
                            break;
                        }
                    }
                    
                    ms.Position = savedPos;
                }
                
                // Create final instance with all data
                instance = new Mh2oInstance
                {
                    LiquidTypeId = instance.LiquidTypeId,
                    Lvf = instance.Lvf,
                    MinHeightLevel = instance.MinHeightLevel,
                    MaxHeightLevel = instance.MaxHeightLevel,
                    XOffset = instance.XOffset,
                    YOffset = instance.YOffset,
                    Width = instance.Width,
                    Height = instance.Height,
                    ExistsBitmap = existsBitmap,
                    HeightMap = heightMap,
                    DepthMap = depthMap
                };
                
                chunk.Add(instance);
            }
        }
        
        // Read attributes if present
        if (offsetAttributes > 0 && offsetAttributes < payload.Length)
        {
            ms.Position = offsetAttributes;
            if (ms.Position + 16 <= payload.Length)
            {
                ulong fishable = 0;
                ulong deep = 0;
                for (int y = 0; y < 8; y++)
                {
                    byte fishRow = reader.ReadByte();
                    for (int x = 0; x < 8; x++)
                    {
                        if ((fishRow & (1 << x)) != 0)
                            fishable |= 1UL << (y * 8 + x);
                    }
                }
                for (int y = 0; y < 8; y++)
                {
                    byte deepRow = reader.ReadByte();
                    for (int x = 0; x < 8; x++)
                    {
                        if ((deepRow & (1 << x)) != 0)
                            deep |= 1UL << (y * 8 + x);
                    }
                }
                chunk = new Mh2oChunk { Attributes = new Mh2oAttributes(fishable, deep) };
                foreach (var inst in chunk.Instances.ToArray())
                    chunk.Add(inst);
            }
        }
        
        return chunk;
    }
    
    private static byte[] SerializeMclq(MclqData mclqData)
    {
        // Alpha MCLQ format:
        // [9x9 float heights = 324 bytes]
        // [9x9 byte depths = 81 bytes]
        // [8x8 byte tile data (type in low nibble, flags in high nibble) = 64 bytes]
        // Total: 469 bytes
        
        using var ms = new MemoryStream(469);
        using var writer = new BinaryWriter(ms);
        
        // Write heights (9x9 = 81 floats)
        for (int i = 0; i < 81; i++)
            writer.Write(mclqData.Heights[i]);
        
        // Write depths (9x9 = 81 bytes)
        writer.Write(mclqData.Depth, 0, 81);
        
        // Write tile data (8x8 = 64 bytes, type | flags)
        for (int y = 0; y < 8; y++)
        {
            for (int x = 0; x < 8; x++)
            {
                byte tileType = (byte)mclqData.Types[x, y];
                byte tileFlags = (byte)mclqData.Flags[x, y];
                byte combined = (byte)(tileType | tileFlags);
                writer.Write(combined);
            }
        }
        
        return ms.ToArray();
    }
    
    private static byte[] ConvertMcseLkToAlpha(byte[] lkMcseData, LkToAlphaOptions? opts)
    {
        // LK MCSE format: 52 bytes per entry (1.12.1 / 3.3.5)
        // Alpha MCSE format: 76 bytes per entry (0.5.3)
        const int LkEntrySize = 52;
        const int AlphaEntrySize = 76;
        
        if (lkMcseData.Length % LkEntrySize != 0)
        {
            if (opts?.VerboseLogging == true)
            {
                Console.WriteLine($"[MCSE] Warning: LK MCSE data size {lkMcseData.Length} is not a multiple of {LkEntrySize}");
            }
            // Try to process what we can
        }
        
        int entryCount = lkMcseData.Length / LkEntrySize;
        if (entryCount == 0)
            return Array.Empty<byte>();
        
        using var ms = new MemoryStream(entryCount * AlphaEntrySize);
        using var reader = new BinaryReader(new MemoryStream(lkMcseData));
        using var writer = new BinaryWriter(ms);
        
        for (int i = 0; i < entryCount; i++)
        {
            // Read LK format (52 bytes)
            uint soundPointID = reader.ReadUInt32();      // 0x00
            uint soundNameID = reader.ReadUInt32();       // 0x04
            float posX = reader.ReadSingle();             // 0x08
            float posY = reader.ReadSingle();             // 0x0C
            float posZ = reader.ReadSingle();             // 0x10
            float minDistance = reader.ReadSingle();      // 0x14
            float maxDistance = reader.ReadSingle();      // 0x18
            float cutoffDistance = reader.ReadSingle();   // 0x1C
            ushort startTime = reader.ReadUInt16();       // 0x20
            ushort endTime = reader.ReadUInt16();         // 0x22
            ushort mode = reader.ReadUInt16();            // 0x24
            byte loopCountMin = reader.ReadByte();        // 0x26
            byte loopCountMax = reader.ReadByte();        // 0x27
            ushort groupSilenceMin = reader.ReadUInt16(); // 0x28
            ushort groupSilenceMax = reader.ReadUInt16(); // 0x2A
            ushort playInstancesMin = reader.ReadUInt16();// 0x2C
            ushort playInstancesMax = reader.ReadUInt16();// 0x2E
            ushort interSoundGapMin = reader.ReadUInt16();// 0x30
            ushort interSoundGapMax = reader.ReadUInt16();// 0x32
            
            // Write Alpha format (76 bytes) - expanding smaller types to UINT32
            writer.Write(soundPointID);                   // 0x00
            writer.Write(soundNameID);                    // 0x04
            writer.Write(posX);                           // 0x08
            writer.Write(posY);                           // 0x0C
            writer.Write(posZ);                           // 0x10
            writer.Write(minDistance);                    // 0x14
            writer.Write(maxDistance);                    // 0x18
            writer.Write(cutoffDistance);                 // 0x1C
            writer.Write((uint)startTime);                // 0x20 - expand uint16 to uint32
            writer.Write((uint)endTime);                  // 0x24 - expand uint16 to uint32
            writer.Write((uint)mode);                     // 0x28 - expand uint16 to uint32
            writer.Write((uint)groupSilenceMin);          // 0x2C - expand uint16 to uint32
            writer.Write((uint)groupSilenceMax);          // 0x30 - expand uint16 to uint32
            writer.Write((uint)playInstancesMin);         // 0x34 - expand uint16 to uint32
            writer.Write((uint)playInstancesMax);         // 0x38 - expand uint16 to uint32
            writer.Write((uint)loopCountMin);             // 0x3C - expand uint8 to uint32
            writer.Write((uint)loopCountMax);             // 0x40 - expand uint8 to uint32
            writer.Write((uint)interSoundGapMin);         // 0x44 - expand uint16 to uint32
            writer.Write((uint)interSoundGapMax);         // 0x48 - expand uint16 to uint32
        }
        
        return ms.ToArray();
    }
}
