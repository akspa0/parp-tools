using System;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using Util = GillijimProject.Utilities.Utilities;
using WoWRollback.LkToAlphaModule;
using WoWRollback.LkToAlphaModule.Mcal;

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
                    mclyLkWhole = new byte[mclySize];
                    Buffer.BlockCopy(lkAdtBytes, mclyPos + 8, mclyLkWhole, 0, mclySize);
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

        // Build MCLY raw - use extracted LK data or create minimal fallback
        byte[] mclyRaw;
        if (mclyLkWhole != null && mclyLkWhole.Length > 0)
        {
            mclyRaw = new byte[mclyLkWhole.Length];
            Buffer.BlockCopy(mclyLkWhole, 0, mclyRaw, 0, mclyRaw.Length);
        }
        else
        {
            mclyRaw = new byte[16];
            // all zeros is acceptable minimal layer
        }

        DumpMclyTable(lkHeader.IndexX, lkHeader.IndexY, mclyRaw, opts);

        // Build MCAL raw - use extracted LK data or create empty fallback
        byte[] mcalRaw;
        int layerCount = mclyRaw.Length / 16;
        bool hasAlphaFlags = mclyRaw.Length >= 16 && LayerUsesAlpha(mclyRaw);
        if (mcalLkWhole != null && mcalLkWhole.Length > 0 && hasAlphaFlags)
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
        
        // Build MCSE raw - use extracted LK data or create empty
        byte[] mcseRaw;
        if (mcseLkWhole != null && mcseLkWhole.Length > 0)
        {
            mcseRaw = new byte[mcseLkWhole.Length];
            Buffer.BlockCopy(mcseLkWhole, 0, mcseRaw, 0, mcseRaw.Length);
        }
        else
        {
            mcseRaw = Array.Empty<byte>();
        }

        // MCRF raw is empty in our current flow
        var mcrfRaw = Array.Empty<byte>();

        DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);

        byte[] mclyWhole = mclyRaw;
        byte[] mcrfWhole = mcrfRaw;
        byte[] mcshWhole = mcshRaw;
        byte[] mcalWhole = mcalRaw;
        byte[] mcseWhole = mcseRaw;

        int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshWhole.Length + mcalWhole.Length + mcseWhole.Length;
        
        // Calculate bounding sphere radius from MCVT heights
        float radius = CalculateRadius(alphaMcvtRaw);
        
        // Calculate number of texture layers from MCLY raw table (each entry is 16 bytes)
        
        // Compute Alpha SMChunk header fields (offsets relative to BEGINNING of MCNK chunk)
        const int headerTotal = 8 + McnkHeaderSize; // FourCC+size + 128-byte header
        int offsHeight = 0; // Offsets are relative to the subchunk data region (immediately after header)
        int offsNormal = offsHeight + alphaMcvtRaw.Length;
        int offsLayer  = offsNormal + mcnrRaw.Length;
        int offsRefs   = offsLayer  + mclyWhole.Length;
        int offsShadow = offsRefs   + mcrfWhole.Length;
        int offsAlpha  = offsShadow + mcshWhole.Length;
        int offsSnd    = mcseWhole.Length > 0 ? offsAlpha + mcalWhole.Length : 0;

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
        BitConverter.GetBytes(layerCount).CopyTo(smh[0x10..]);
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
}
