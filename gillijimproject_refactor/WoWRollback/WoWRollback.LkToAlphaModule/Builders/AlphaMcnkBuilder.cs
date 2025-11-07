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

    public static byte[] BuildFromLk(byte[] lkAdtBytes, int mcNkOffset, LkToAlphaOptions? opts = null, byte[]? lkTexAdtBytes = null, int texMcNkOffset = -1, System.Collections.Generic.IReadOnlyList<int>? doodadRefs = null, System.Collections.Generic.IReadOnlyList<int>? mapObjRefs = null)
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
        byte[]? mclqWhole = null;
        
        // Extract MCLY using header offsets (LK offsets point to subchunk letters/FourCC)
        if (lkHeader.MclyOffset > 0)
        {
            int mclyPos = mcNkOffset + lkHeader.MclyOffset;
            bool ok = mclyPos >= 0 && mclyPos + 8 <= lkAdtBytes.Length && Encoding.ASCII.GetString(lkAdtBytes, mclyPos, 4) == "YLCM";
            if (!ok)
            {
                // Fallback: scan subregion for YLCM
                for (int p = subStart; p + 8 <= subEnd;)
                {
                    string fcc = Encoding.ASCII.GetString(lkAdtBytes, p, 4);
                    int size = BitConverter.ToInt32(lkAdtBytes, p + 4);
                    int dataStart = p + 8;
                    int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
                    if (fcc == "YLCM") { mclyPos = p; ok = true; break; }
                    if (dataStart + size > subEnd || next <= p) break; p = next;
                }
            }
            if (ok)
            {
                int mclySize = BitConverter.ToInt32(lkAdtBytes, mclyPos + 4);
                if (mclySize > 0 && mclyPos + 8 + mclySize <= lkAdtBytes.Length)
                {
                    mclyLkWhole = new byte[8 + mclySize + ((mclySize & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkAdtBytes, mclyPos, mclyLkWhole, 0, mclyLkWhole.Length);
                }
            }
        }
        // After scanning subchunks: if no MCLQ present, try to synthesize from MH2O (LK/TBC/WotLK)
        if (mclqWhole == null)
        {
            var cand = TryBuildMclqFromMh2o(lkAdtBytes, lkHeader.IndexX, lkHeader.IndexY, opts);
            if (cand != null && cand.Length > 0)
            {
                mclqWhole = cand;
                if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) MCLQ built from MH2O");
                DumpMclqData("built", lkHeader.IndexX, lkHeader.IndexY, mclqWhole, opts);
            }
            else
            {
                if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) no MCLQ and MH2O build failed");
            }
        }
        
        if (lkHeader.McalOffset > 0)
        {
            int mcalPos = mcNkOffset + lkHeader.McalOffset;
            bool ok = mcalPos >= 0 && mcalPos + 8 <= lkAdtBytes.Length && Encoding.ASCII.GetString(lkAdtBytes, mcalPos, 4) == "LACM";
            if (!ok)
            {
                // Fallback: scan subregion for LACM
                for (int p = subStart; p + 8 <= subEnd;)
                {
                    string fcc = Encoding.ASCII.GetString(lkAdtBytes, p, 4);
                    int size = BitConverter.ToInt32(lkAdtBytes, p + 4);
                    int dataStart = p + 8;
                    int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
                    if (fcc == "LACM") { mcalPos = p; ok = true; break; }
                    if (dataStart + size > subEnd || next <= p) break; p = next;
                }
            }
            if (ok)
            {
                // Prefer LK header-reported size (includes 8-byte header); fallback to chunk's own size
                int payloadFromHeader = (int)lkHeader.McalSize - ChunkLettersAndSize;
                int payloadSize = payloadFromHeader > 0 ? payloadFromHeader : BitConverter.ToInt32(lkAdtBytes, mcalPos + 4);
                if (payloadSize > 0)
                {
                    int pad = (payloadSize & 1) == 1 ? 1 : 0;
                    int totalLen = ChunkLettersAndSize + payloadSize + pad;
                    int maxAvail = Math.Min(subEnd, lkAdtBytes.Length) - mcalPos;
                    int copyLen = Math.Min(totalLen, Math.Max(0, maxAvail));
                    if (copyLen >= ChunkLettersAndSize)
                    {
                        mcalLkWhole = new byte[copyLen];
                        Buffer.BlockCopy(lkAdtBytes, mcalPos, mcalLkWhole, 0, copyLen);
                    }
                }
            }
        }
        
        if (lkHeader.McshOffset > 0 && lkHeader.McshOffset != lkHeader.McalOffset)
        {
            int mcshPos = mcNkOffset + lkHeader.McshOffset;
            bool ok = mcshPos >= 0 && mcshPos + 8 <= lkAdtBytes.Length && Encoding.ASCII.GetString(lkAdtBytes, mcshPos, 4) == "HSCM";
            if (!ok)
            {
                // Fallback: scan subregion for HSCM
                for (int p = subStart; p + 8 <= subEnd;)
                {
                    string fcc = Encoding.ASCII.GetString(lkAdtBytes, p, 4);
                    int size = BitConverter.ToInt32(lkAdtBytes, p + 4);
                    int dataStart = p + 8;
                    int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
                    if (fcc == "HSCM") { mcshPos = p; ok = true; break; }
                    if (dataStart + size > subEnd || next <= p) break; p = next;
                }
            }
            if (ok)
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
            else if (fcc == "QLCM") // 'MCLQ' reversed on disk
            {
                mclqWhole = new byte[8 + size + ((size & 1) == 1 ? 1 : 0)];
                Buffer.BlockCopy(lkAdtBytes, p, mclqWhole, 0, mclqWhole.Length);
                if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) MCLQ passthrough size={mclqWhole.Length}");
                DumpMclqData("passthrough", lkHeader.IndexX, lkHeader.IndexY, mclqWhole, opts);
            }
            // Note: MCLY, MCAL, MCSH are extracted via header offsets above, not scanned here
            
            p = next;
        }
        
        // If texture ADT provided, scan its MCNK subrange for YLCM/LACM/HSCM and only use as fallback when root is missing
        if (lkTexAdtBytes != null && texMcNkOffset >= 0)
        {
            int texSize = BitConverter.ToInt32(lkTexAdtBytes, texMcNkOffset + 4);
            int texSubStart = texMcNkOffset + ChunkLettersAndSize + McnkHeaderSize;
            int texSubEnd = texMcNkOffset + ChunkLettersAndSize + Math.Max(0, texSize);
            if (texSubEnd > lkTexAdtBytes.Length) texSubEnd = lkTexAdtBytes.Length;

            bool preferTex = opts?.PreferTexLayers == true;
            for (int p2 = texSubStart; p2 + 8 <= texSubEnd;)
            {
                string fcc2 = Encoding.ASCII.GetString(lkTexAdtBytes, p2, 4);
                int size2 = BitConverter.ToInt32(lkTexAdtBytes, p2 + 4);
                int dataStart2 = p2 + 8;
                int next2 = dataStart2 + size2 + ((size2 & 1) == 1 ? 1 : 0);
                if (dataStart2 + size2 > texSubEnd || next2 <= p2) break;

                if (fcc2 == "YLCM")
                {
                    var cand = new byte[8 + size2 + ((size2 & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p2, cand, 0, cand.Length);
                    if (preferTex || mclyLkWhole == null) mclyLkWhole = cand;
                    if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] {(preferTex ? "prefer" : "fallback")} _tex MCLY for chunk ({lkHeader.IndexX},{lkHeader.IndexY})");
                }
                else if (fcc2 == "LACM")
                {
                    var cand = new byte[8 + size2 + ((size2 & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p2, cand, 0, cand.Length);
                    if (preferTex || mcalLkWhole == null) mcalLkWhole = cand;
                    if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] {(preferTex ? "prefer" : "fallback")} _tex MCAL for chunk ({lkHeader.IndexX},{lkHeader.IndexY})");
                }
                else if (fcc2 == "HSCM")
                {
                    var cand = new byte[8 + size2 + ((size2 & 1) == 1 ? 1 : 0)];
                    Buffer.BlockCopy(lkTexAdtBytes, p2, cand, 0, cand.Length);
                    if (preferTex || mcshLkWhole == null) mcshLkWhole = cand;
                    if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] {(preferTex ? "prefer" : "fallback")} _tex MCSH for chunk ({lkHeader.IndexX},{lkHeader.IndexY})");
                }
                p2 = next2;
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
        
        // Build MCAL raw - convert LK 8-bit (4096 bytes per layer) to Alpha 4-bit packed (2048 bytes per layer)
        byte[] mcalRaw = Array.Empty<byte>();
        {
            // Extract LK MCAL payload (strip chunk header)
            byte[] mcalLkRaw = Array.Empty<byte>();
            if (mcalLkWhole != null && mcalLkWhole.Length > 8)
            {
                int payloadLen = Math.Max(0, mcalLkWhole.Length - 8);
                mcalLkRaw = new byte[payloadLen];
                Buffer.BlockCopy(mcalLkWhole, 8, mcalLkRaw, 0, payloadLen);
                DumpMcalData("lk", lkHeader.IndexX, lkHeader.IndexY, mcalLkRaw, opts);
            }

            // Prepare updated MCLY table and new Alpha MCAL stream
            int numLayers = mclyRaw.Length / 16;
            if (opts?.VerboseLogging == true)
            {
                Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) nLayers={numLayers}");
            }
            var mclyOut = new byte[mclyRaw.Length];
            Buffer.BlockCopy(mclyRaw, 0, mclyOut, 0, mclyRaw.Length);
            const uint FLAG_USE_ALPHA = 0x100;
            const uint FLAG_ALPHA_COMP = 0x200;

            // Ensure base layer (0) has no alpha
            if (numLayers > 0)
            {
                uint flags0 = BitConverter.ToUInt32(mclyOut, 4);
                flags0 &= ~FLAG_USE_ALPHA;
                flags0 &= ~FLAG_ALPHA_COMP;
                Buffer.BlockCopy(BitConverter.GetBytes(flags0), 0, mclyOut, 4, 4);
                Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, mclyOut, 8, 4);
            }

            // If explicitly requested: force raw pass-through from LK MCAL
            bool forcedRaw = false;
            if (opts?.RawCopyLkLayers == true && mcalLkRaw.Length > 0 && numLayers >= 1)
            {
                if (numLayers > 0)
                {
                    uint f0 = BitConverter.ToUInt32(mclyOut, 4);
                    f0 &= ~FLAG_USE_ALPHA; f0 &= ~FLAG_ALPHA_COMP;
                    Buffer.BlockCopy(BitConverter.GetBytes(f0), 0, mclyOut, 4, 4);
                    Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, mclyOut, 8, 4);
                }
                for (int i = 1; i < numLayers; i++)
                {
                    int baseOff = i * 16;
                    uint fi = BitConverter.ToUInt32(mclyOut, baseOff + 4);
                    fi |= FLAG_USE_ALPHA; fi &= ~FLAG_ALPHA_COMP;
                    Buffer.BlockCopy(BitConverter.GetBytes(fi), 0, mclyOut, baseOff + 4, 4);
                    uint desiredOff = (uint)((i - 1) * 2048);
                    Buffer.BlockCopy(BitConverter.GetBytes(desiredOff), 0, mclyOut, baseOff + 8, 4);
                }

                int needed = 2048 * Math.Max(0, numLayers - 1);
                var passthrough = new byte[needed];
                int toCopy = Math.Min(needed, mcalLkRaw.Length);
                if (toCopy > 0) Buffer.BlockCopy(mcalLkRaw, 0, passthrough, 0, toCopy);
                mcalRaw = passthrough;
                mclyRaw = mclyOut;
                if (opts?.VerboseLogging == true)
                {
                    Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) FORCED RAW-PASSTHROUGH size={mcalRaw.Length}");
                }
                DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);
                forcedRaw = true;
            }
            if (!forcedRaw)
            {
                // Collect LK offsets (relative to MCAL data start) for layers > 0
                var entries = new System.Collections.Generic.List<(int idx, int offs)>();
                for (int i = 1; i < numLayers; i++)
                {
                    int baseOff = i * 16;
                    int off = unchecked((int)BitConverter.ToUInt32(mclyRaw, baseOff + 8));
                    if (off < 0) off = 0;
                    if (mcalLkRaw.Length > 0 && off > mcalLkRaw.Length) off = mcalLkRaw.Length; // clamp
                    entries.Add((i, off));
                }
                entries.Sort((a, b) => a.offs.CompareTo(b.offs));

                // Fast-path: if LK payload looks like old 4bpp (every extra layer span == 2048), just pass MCAL through verbatim
                bool looks4bpp = entries.Count > 0 && mcalLkRaw.Length > 0;
                if (looks4bpp)
                {
                    for (int k = 0; k < entries.Count; k++)
                    {
                        int start = entries[k].offs;
                        int next = (k + 1 < entries.Count) ? entries[k + 1].offs : mcalLkRaw.Length;
                        int span = Math.Max(0, next - start);
                        if (span != 2048) { looks4bpp = false; break; }
                    }
                }
                // Also accept pass-through if total payload size matches 2048 per extra layer
                if (!looks4bpp && entries.Count > 0)
                {
                    int expected = 2048 * entries.Count;
                    if (mcalLkRaw.Length == expected) looks4bpp = true;
                }
                if (looks4bpp)
                {
                    // Fix flags: base layer no alpha, others use alpha and not compressed
                    if (numLayers > 0)
                    {
                        uint f0 = BitConverter.ToUInt32(mclyOut, 4);
                        f0 &= ~FLAG_USE_ALPHA; f0 &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(f0), 0, mclyOut, 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, mclyOut, 8, 4);
                    }
                    for (int i = 1; i < numLayers; i++)
                    {
                        int baseOff = i * 16;
                        uint fi = BitConverter.ToUInt32(mclyOut, baseOff + 4);
                        fi |= FLAG_USE_ALPHA; fi &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(fi), 0, mclyOut, baseOff + 4, 4);
                        // If offsets are not strictly sequential 2048*n, rewrite them
                        uint desiredOff = (uint)((i - 1) * 2048);
                        Buffer.BlockCopy(BitConverter.GetBytes(desiredOff), 0, mclyOut, baseOff + 8, 4);
                    }

                    mcalRaw = mcalLkRaw; // pass-through
                    mclyRaw = mclyOut;
                    if (opts?.VerboseLogging == true)
                    {
                        Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) RAW-PASSTHROUGH MCAL (4bpp spans) size={mcalRaw.Length}");
                    }
                    DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);
                }
                else
                {
                // Normal path: decode LK layers then pack to 4bpp
                using var msAlpha = new MemoryStream();
                int slicesWritten = 0;
                for (int k = 0; k < entries.Count; k++)
                {
                    int idx = entries[k].idx;
                    int off = entries[k].offs;
                    // Find next DISTINCT offset to compute span; identical offsets should not zero-length earlier slices
                    int nextDistinct = mcalLkRaw.Length;
                    for (int j = k + 1; j < entries.Count; j++)
                    {
                        if (entries[j].offs > off) { nextDistinct = entries[j].offs; break; }
                    }
                    int available = Math.Max(0, nextDistinct - off);

                    int outOffset = (int)msAlpha.Length;
                    int layerBase = idx * 16;

                    // Determine LK flags for this layer to detect compression
                    uint lkFlags = BitConverter.ToUInt32(mclyRaw, layerBase + 4);
                    bool lkCompressed = (lkFlags & 0x200) != 0;

                    // Decode LK MCAL to 8bpp 64x64 (handles compressed/uncompressed)
                    // If span is zero or negative, fallback to reading until end of payload; decoder will stop at 4096
                    var src8 = DecodeLkMcalTo8bpp(mcalLkRaw, off, available, lkCompressed);

                    if (src8.Length == 4096)
                    {
                        // Pack 8bpp 64x64 -> 4bpp 2048 without 63x63 duplication, LSB-first nibble
                        if (opts?.VerboseLogging == true)
                        {
                            Console.WriteLine($"[alpha][mcal-pack] start ({lkHeader.IndexX},{lkHeader.IndexY}) layer {idx}");
                        }
                        var packed = Pack8To4_64x64(src8, 0);
                        msAlpha.Write(packed, 0, packed.Length);
                        slicesWritten++;
                        if (opts?.VerboseLogging == true)
                        {
                            Console.WriteLine($"[alpha][mcal-pack] done ({lkHeader.IndexX},{lkHeader.IndexY}) layer {idx} out=2048");
                        }

                        uint flags = BitConverter.ToUInt32(mclyOut, layerBase + 4);
                        flags |= FLAG_USE_ALPHA;
                        // IMPORTANT: 4bpp (2048) is UNCOMPRESSED; 0x200 indicates compressed (RLE, 8-bit only)
                        flags &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(flags), 0, mclyOut, layerBase + 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes((uint)outOffset), 0, mclyOut, layerBase + 8, 4);

                        if (opts?.VerboseLogging == true)
                        {
                            Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) layer {idx}: lkOff={off}, avail={available}, outOff={outOffset}, slice=2048");
                        }
                    }
                    else
                    {
                        // No valid alpha payload -> clear flags and offset
                        uint flags = BitConverter.ToUInt32(mclyOut, layerBase + 4);
                        flags &= ~FLAG_USE_ALPHA;
                        flags &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(flags), 0, mclyOut, layerBase + 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, mclyOut, layerBase + 8, 4);
                        if (opts?.VerboseLogging == true)
                        {
                            Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) layer {idx}: MISSING alpha (lkOff={off}, avail={available})");
                        }
                    }
                }

                mcalRaw = msAlpha.ToArray();
                if (opts?.VerboseLogging == true)
                {
                    Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) wroteSlices={slicesWritten} mcalBytes={mcalRaw.Length}");
                }
                mclyRaw = mclyOut;
                DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);

                // If nothing was written but we do have a LK payload, fallback to raw pass-through (treat as 4bpp 2048 per extra layer)
                if (mcalRaw.Length == 0 && entries.Count > 0 && mcalLkRaw.Length > 0)
                {
                    if (numLayers > 0)
                    {
                        uint f0 = BitConverter.ToUInt32(mclyOut, 4);
                        f0 &= ~FLAG_USE_ALPHA; f0 &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(f0), 0, mclyOut, 4, 4);
                        Buffer.BlockCopy(BitConverter.GetBytes(0u), 0, mclyOut, 8, 4);
                    }
                    for (int i = 1; i < numLayers; i++)
                    {
                        int baseOff = i * 16;
                        uint fi = BitConverter.ToUInt32(mclyOut, baseOff + 4);
                        fi |= FLAG_USE_ALPHA; fi &= ~FLAG_ALPHA_COMP;
                        Buffer.BlockCopy(BitConverter.GetBytes(fi), 0, mclyOut, baseOff + 4, 4);
                        uint desiredOff = (uint)((i - 1) * 2048);
                        Buffer.BlockCopy(BitConverter.GetBytes(desiredOff), 0, mclyOut, baseOff + 8, 4);
                    }
                    mcalRaw = mcalLkRaw;
                    mclyRaw = mclyOut;
                    if (opts?.VerboseLogging == true)
                    {
                        Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) FALLBACK RAW-PASSTHROUGH size={mcalRaw.Length}");
                    }
                    DumpMcalData("alpha", lkHeader.IndexX, lkHeader.IndexY, mcalRaw, opts);
                }
                }
            }

            // end build MCAL
        }
        
        // Build MCSH raw - use extracted LK data or create empty fallback
        byte[] mcshRaw;
        if (mcshLkWhole != null && mcshLkWhole.Length > 8)
        {
            int sz = BitConverter.ToInt32(mcshLkWhole, 4);
            mcshRaw = new byte[sz];
            Buffer.BlockCopy(mcshLkWhole, 8, mcshRaw, 0, sz);
            // Invert bits so shadowed areas appear dark (not white) in Alpha
            for (int i = 0; i < mcshRaw.Length; i++) mcshRaw[i] = (byte)~mcshRaw[i];
            if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) MCSH inverted size={mcshRaw.Length}");
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

        // Build MCRF raw from provided references (Alpha uses MDNM indices followed by MONM indices)
        byte[] mcrfRaw;
        int nDoodadRefs = doodadRefs?.Count ?? 0;
        int nMapObjRefs = mapObjRefs?.Count ?? 0;
        if (nDoodadRefs > 0 || nMapObjRefs > 0)
        {
            using var msRefs = new MemoryStream();
            if (nDoodadRefs > 0)
            {
                for (int i = 0; i < nDoodadRefs; i++) msRefs.Write(BitConverter.GetBytes(doodadRefs![i]));
            }
            if (nMapObjRefs > 0)
            {
                for (int i = 0; i < nMapObjRefs; i++) msRefs.Write(BitConverter.GetBytes(mapObjRefs![i]));
            }
            mcrfRaw = msRefs.ToArray();
        }
        else
        {
            mcrfRaw = Array.Empty<byte>();
        }

        // Build named chunk wrappers ONLY for MCLY and MCRF (Alpha v18 format)
        // CRITICAL: MCSH, MCAL, MCSE do NOT have headers in Alpha - raw data only!
        var mclyChunk = new Chunk("MCLY", mclyRaw.Length, mclyRaw);
        var mcrfChunk = new Chunk("MCRF", mcrfRaw.Length, mcrfRaw);

        byte[] mclyWhole = mclyChunk.GetWholeChunk();  // Has header (FourCC+size)
        byte[] mcrfWhole = mcrfChunk.GetWholeChunk();  // Has header (FourCC+size)
        // mcshRaw, mcalRaw, mcseRaw used directly - NO headers!

        // Validate MCLY/MCAL integrity to catch offset corruption bugs
        ValidateMclyMcalIntegrity(mclyRaw, mcalRaw, lkHeader.IndexX, lkHeader.IndexY, opts);

        int totalSubChunkSize = alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshRaw.Length + mcalRaw.Length + mcseRaw.Length + (mclqWhole?.Length ?? 0);
        
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
        int offsAlpha  = offsShadow + mcshRaw.Length;   // MCSH is raw, no header
        int offsSnd    = offsAlpha  + mcalRaw.Length;   // MCAL is raw, no header

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

        int givenSize = McnkHeaderSize + alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshRaw.Length + mcalRaw.Length + mcseRaw.Length + (mclqWhole?.Length ?? 0);

        using var ms = new MemoryStream();
        // Write MCNK letters reversed on disk ('KNCM') to match Alpha v18 expectations
        var letters = Encoding.ASCII.GetBytes("KNCM");
        ms.Write(letters, 0, 4);
        ms.Write(BitConverter.GetBytes(givenSize), 0, 4);

        // Build Alpha SMChunk header (128 bytes)
        // Use heap-backed Span to reduce stack usage across many chunks
        Span<byte> smh = new Span<byte>(new byte[McnkHeaderSize]);
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

        // Flags: set liquid flags based on MCLQ content and set has_mcsh when present
        uint flagsVal = 0;
        if (mcshRaw.Length > 0) flagsVal |= 0x1; // has_mcsh
        if (mclqWhole != null && mclqWhole.Length >= 8 + 8 + 81 * 8 + 64)
        {
            int tileStart = 8 + 8 + 81 * 8; // chunk header + min/max + vertices
            for (int i = 0; i < 64; i++)
            {
                byte tb = mclqWhole[tileStart + i];
                bool dontRender = (tb & 0x08) != 0;
                if (dontRender) continue;
                int t = tb & 0x7;
                if (t == 4) flagsVal |= (1u << 2); // river
                else if (t == 1) flagsVal |= (1u << 3); // ocean
                else if (t == 6) flagsVal |= (1u << 4); // magma
                else if (t == 3) flagsVal |= (1u << 5); // slime
            }
        }
        BitConverter.GetBytes(flagsVal).CopyTo(smh[0x00..]); // flags
        BitConverter.GetBytes(lkHeader.IndexX).CopyTo(smh[0x04..]);
        BitConverter.GetBytes(lkHeader.IndexY).CopyTo(smh[0x08..]);
        BitConverter.GetBytes(radius).CopyTo(smh[0x0C..]);
        BitConverter.GetBytes(nLayers).CopyTo(smh[0x10..]);
        BitConverter.GetBytes(nDoodadRefs).CopyTo(smh[0x14..]); // nDoodadRefs
        BitConverter.GetBytes(offsHeight).CopyTo(smh[0x18..]);
        BitConverter.GetBytes(offsNormal).CopyTo(smh[0x1C..]);
        BitConverter.GetBytes(offsLayer).CopyTo(smh[0x20..]);
        BitConverter.GetBytes(offsRefs).CopyTo(smh[0x24..]);
        BitConverter.GetBytes(offsAlpha).CopyTo(smh[0x28..]);
        BitConverter.GetBytes(sizeAlpha).CopyTo(smh[0x2C..]);
        BitConverter.GetBytes(offsShadow).CopyTo(smh[0x30..]);
        BitConverter.GetBytes(sizeShadow).CopyTo(smh[0x34..]);
        BitConverter.GetBytes(areaIdVal).CopyTo(smh[0x38..]);
        BitConverter.GetBytes(nMapObjRefs).CopyTo(smh[0x3C..]); // nMapObjRefs
        // holes (uint16) at 0x40, pad0 at 0x42 (leave zeros)
        // predTex[8] at 0x44, noEffectDoodad[8] at 0x54 (already zeros)
        BitConverter.GetBytes(offsSnd).CopyTo(smh[0x5C..]);
        BitConverter.GetBytes(nSnd).CopyTo(smh[0x60..]);
        // offsLiquid zero by default

        // Write header
        ms.Write(smh);

        // Sub-blocks in Alpha order:
        // MCVT, MCNR: raw data (no headers)
        // MCLY, MCRF: with headers (FourCC+size+data)
        // MCSH, MCAL, MCSE: raw data (NO headers in Alpha v18!)
        // MCLQ: with header (FourCC+size+data)
        if (alphaMcvtRaw.Length > 0) ms.Write(alphaMcvtRaw, 0, alphaMcvtRaw.Length);
        if (mcnrRaw.Length > 0) ms.Write(mcnrRaw, 0, mcnrRaw.Length);
        ms.Write(mclyWhole, 0, mclyWhole.Length);  // Has header
        ms.Write(mcrfWhole, 0, mcrfWhole.Length);  // Has header
        if (mcshRaw.Length > 0) ms.Write(mcshRaw, 0, mcshRaw.Length);  // Raw data only!
        if (mcalRaw.Length > 0) ms.Write(mcalRaw, 0, mcalRaw.Length);  // Raw data only!
        if (mcseRaw.Length > 0) ms.Write(mcseRaw, 0, mcseRaw.Length);  // Raw data only!
        int offsLiquid = 0;
        if (mclqWhole != null && mclqWhole.Length > 0)
        {
            offsLiquid = headerTotal + alphaMcvtRaw.Length + mcnrRaw.Length + mclyWhole.Length + mcrfWhole.Length + mcshRaw.Length + mcalRaw.Length + mcseRaw.Length;
            if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({lkHeader.IndexX},{lkHeader.IndexY}) offsLiquid={offsLiquid} mclqLen={mclqWhole.Length}");
            ms.Write(mclqWhole, 0, mclqWhole.Length);
        }

        // Even-byte pad: if MCNK data size is odd, write a single 0x00 pad byte (not counted in size)
        if ((givenSize & 1) == 1)
        {
            ms.WriteByte(0);
        }
        var outBuf = ms.ToArray();
        if (offsLiquid > 0)
        {
            // Patch offsLiquid into header (relative to beginning of MCNK chunk)
            // MCNK header starts at byte 8 in this buffer
            int headerPos = 8;
            // 0x64 relative within header
            BitConverter.GetBytes(offsLiquid).CopyTo(outBuf, headerPos + 0x64);
        }
        return outBuf;
    }

    private static byte[]? TryBuildMclqFromMh2o(byte[] bytes, int chunkX, int chunkY, LkToAlphaOptions? opts)
    {
        int mh2o = FindFourCCReversed(bytes, "MH2O");
        if (mh2o < 0 || mh2o + 8 > bytes.Length)
        {
            if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] MH2O not found for mcnk ({chunkX},{chunkY})");
            return null;
        }
        int mh2oSize = BitConverter.ToInt32(bytes, mh2o + 4);
        int basePos = mh2o + 8;
        if (basePos + 256 * 12 > bytes.Length) return null; // 256 headers x 12 bytes

        int idx = chunkY * 16 + chunkX;
        int hdrPos = basePos + idx * 12;
        int ofsInformation = BitConverter.ToInt32(bytes, hdrPos + 0);
        int nLayers = BitConverter.ToInt32(bytes, hdrPos + 4);
        int ofsAttributes = BitConverter.ToInt32(bytes, hdrPos + 8);
        if (nLayers <= 0 || ofsInformation <= 0) return null;

        ulong fishable = 0xFFFFFFFFFFFFFFFFUL;
        ulong fatigue = 0UL;
        if (ofsAttributes > 0 && basePos + ofsAttributes + 16 <= bytes.Length)
        {
            fishable = BitConverter.ToUInt64(bytes, basePos + ofsAttributes + 0);
            fatigue = BitConverter.ToUInt64(bytes, basePos + ofsAttributes + 8);
        }

        float[] heights = new float[9 * 9];
        bool[] hasH = new bool[9 * 9];
        byte[] depths = new byte[9 * 9];
        for (int i = 0; i < depths.Length; i++) depths[i] = 0xFF;
        byte[] tilesType = new byte[8 * 8];
        bool[] tilesUsed = new bool[8 * 8];

        int infoBase = basePos + ofsInformation;
        for (int k = 0; k < nLayers; k++)
        {
            int ip = infoBase + k * 24; // MH2O_Information size
            if (ip + 24 > bytes.Length) break;
            ushort liquidId = BitConverter.ToUInt16(bytes, ip + 0);
            ushort vfmt = BitConverter.ToUInt16(bytes, ip + 2);
            float minH = BitConverter.ToSingle(bytes, ip + 4);
            float maxH = BitConverter.ToSingle(bytes, ip + 8);
            int xOff = bytes[ip + 12];
            int yOff = bytes[ip + 13];
            int width = bytes[ip + 14];
            int height = bytes[ip + 15];
            int ofsInfoMask = BitConverter.ToInt32(bytes, ip + 16);
            int ofsHeightMap = BitConverter.ToInt32(bytes, ip + 20);

            ulong mask = 0xFFFFFFFFFFFFFFFFUL;
            if (ofsInfoMask > 0 && height > 0 && width > 0)
            {
                int bits = width * height;
                int bsz = (bits + 7) / 8;
                if (basePos + ofsInfoMask + bsz <= bytes.Length)
                {
                    mask = 0UL;
                    for (int b = 0; b < bsz; b++)
                    {
                        mask |= ((ulong)bytes[basePos + ofsInfoMask + b]) << (b * 8);
                    }
                }
            }

            int p = basePos + ofsHeightMap;
            if (ofsHeightMap > 0 && p >= 0 && p <= bytes.Length)
            {
                bool hasHeights = (vfmt == 0 /*HEIGHT_DEPTH*/ || vfmt == 1 /*HEIGHT_UV*/);
                if (hasHeights)
                {
                    for (int zy = yOff; zy <= yOff + height; zy++)
                    {
                        for (int xx = xOff; xx <= xOff + width; xx++)
                        {
                            if (p + 4 > bytes.Length) { xx = xOff + width; zy = yOff + height; break; }
                            float h = BitConverter.ToSingle(bytes, p); p += 4;
                            int vi = zy * 9 + xx;
                            if (vi >= 0 && vi < 81)
                            {
                                heights[vi] = h;
                                hasH[vi] = true;
                            }
                        }
                    }
                }
                if (vfmt == 1)
                {
                    int count = (width + 1) * (height + 1);
                    int need = count * 4;
                    p += need; // skip UVs
                }
                if (vfmt == 0 || vfmt == 2)
                {
                    for (int zy = yOff; zy <= yOff + height; zy++)
                    {
                        for (int xx = xOff; xx <= xOff + width; xx++)
                        {
                            if (p + 1 > bytes.Length) { xx = xOff + width; zy = yOff + height; break; }
                            byte d = bytes[p++];
                            int vi = zy * 9 + xx;
                            if (vi >= 0 && vi < 81) depths[vi] = d;
                        }
                    }
                }
            }

            for (int zy = 0; zy < height; zy++)
            {
                for (int xx = 0; xx < width; xx++)
                {
                    int bit = zy * width + xx;
                    bool on = ((mask >> bit) & 1UL) != 0UL;
                    int tx = xOff + xx;
                    int ty = yOff + zy;
                    if (tx >= 0 && tx < 8 && ty >= 0 && ty < 8)
                    {
                        int ti = ty * 8 + tx;
                        if (on)
                        {
                            tilesUsed[ti] = true;
                            byte t = MapLiquidType(liquidId);
                            if (Priority(t) >= Priority(tilesType[ti])) tilesType[ti] = t;
                        }
                    }
                }
            }
        }

        float minAll = float.MaxValue, maxAll = float.MinValue;
        for (int i = 0; i < 81; i++) if (hasH[i]) { if (heights[i] < minAll) minAll = heights[i]; if (heights[i] > maxAll) maxAll = heights[i]; }
        if (minAll == float.MaxValue) { minAll = 0f; maxAll = 0f; }

        using var ms = new MemoryStream();
        // Wrap as proper chunk with reversed FourCC via Chunk helper
        var payload = BuildMclqPayload(minAll, maxAll, heights, hasH, depths, tilesUsed, tilesType, fishable, fatigue);
        var ch = new Chunk("MCLQ", payload.Length, payload);
        var whole = ch.GetWholeChunk();
        if (opts?.VerboseLogging == true) Console.WriteLine($"[alpha] mcnk ({chunkX},{chunkY}) built MCLQ payloadLen={payload.Length} wholeLen={whole.Length}");
        DumpMclqData("built_payload", chunkX, chunkY, whole, opts);
        return whole;
    }

    private static int FindFourCCReversed(byte[] buf, string forwardFourCC)
    {
        if (buf == null || buf.Length < 8) return -1;
        if (string.IsNullOrEmpty(forwardFourCC) || forwardFourCC.Length != 4) return -1;
        string reversed = new string(new[] { forwardFourCC[3], forwardFourCC[2], forwardFourCC[1], forwardFourCC[0] });
        for (int i = 0; i + 8 <= buf.Length;)
        {
            if (i < 0 || i + 4 > buf.Length) break;
            string fcc = Encoding.ASCII.GetString(buf, i, 4);
            int size = BitConverter.ToInt32(buf, i + 4);
            if (size < 0 || size > buf.Length) break;
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == reversed) return i;
            if (dataStart + size > buf.Length) break;
            if (next <= i) break;
            i = next;
        }
        return -1;
    }

    private static byte[] BuildMclqPayload(float minH, float maxH, float[] heights, bool[] hasH, byte[] depths, bool[] tilesUsed, byte[] tilesType, ulong fishable, ulong fatigue)
    {
        using var ms = new MemoryStream();
        ms.Write(BitConverter.GetBytes(minH));
        ms.Write(BitConverter.GetBytes(maxH));
        for (int i = 0; i < 81; i++)
        {
            // union water/magma: write water-style by default
            ms.WriteByte(depths[i]);
            ms.WriteByte(0); // flow0
            ms.WriteByte(0); // flow1
            ms.WriteByte(0); // filler
            float h = hasH[i] ? heights[i] : minH;
            ms.Write(BitConverter.GetBytes(h));
        }
        for (int i = 0; i < 64; i++)
        {
            byte b = 0;
            if (tilesUsed[i])
            {
                byte t = (byte)(tilesType[i] & 0x7);
                b = (byte)(t | (0 << 3));
                if (((fishable >> i) & 1UL) != 0) b |= 0x40;
                if (((fatigue  >> i) & 1UL) != 0) b |= 0x80;
            }
            else
            {
                b = 0x0F; // dont_render (top 4 bits set in Noggit but we use 0x0F-> type=7,dont_render=1)
            }
            ms.WriteByte(b);
        }
        ms.Write(BitConverter.GetBytes(0)); // n_flowvs
        // always 2 flowvs reserved (write zeros)
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 3; j++) ms.Write(BitConverter.GetBytes(0f)); // pos
            ms.Write(BitConverter.GetBytes(0f)); // radius
            for (int j = 0; j < 3; j++) ms.Write(BitConverter.GetBytes(0f)); // dir
            ms.Write(BitConverter.GetBytes(0f)); // velocity
            ms.Write(BitConverter.GetBytes(0f)); // amplitude
            ms.Write(BitConverter.GetBytes(0f)); // frequency
        }
        return ms.ToArray();
    }

    private static int FindFourCCForward(byte[] buf, string forwardFourCC)
    {
        if (buf == null || buf.Length < 8) return -1;
        if (string.IsNullOrEmpty(forwardFourCC) || forwardFourCC.Length != 4) return -1;
        var pat = Encoding.ASCII.GetBytes(forwardFourCC);
        for (int i = 0; i + 4 <= buf.Length; i++)
        {
            if (buf[i] == pat[0] && buf[i + 1] == pat[1] && buf[i + 2] == pat[2] && buf[i + 3] == pat[3]) return i;
        }
        return -1;
    }

    private static byte MapLiquidType(ushort liquidId)
    {
        // 1=water -> river(4), 2=ocean->1, 3=magma->6, 4=slime->3
        return liquidId switch
        {
            2 => (byte)1,
            3 => (byte)6,
            4 => (byte)3,
            _ => (byte)4,
        };
    }
    private static int Priority(byte mclqType)
    {
        return mclqType switch { 6 => 3, 3 => 2, 1 => 1, 4 => 0, _ => -1 };
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

    private static void DumpMclqData(string stage, int indexX, int indexY, byte[] data, LkToAlphaOptions? opts)
    {
        if (opts?.VerboseLogging != true) return;
        if (data is null || data.Length == 0) return;
        try
        {
            string root = Path.Combine("debug_mclq", $"{indexY:D2}_{indexX:D2}");
            Directory.CreateDirectory(root);
            string path = Path.Combine(root, $"{stage}_mclq.bin");
            File.WriteAllBytes(path, data);
            Console.WriteLine($"[dump] MCLQ {stage} -> {path} ({data.Length} bytes)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[dump] Failed to write MCLQ {stage} for tile {indexY:D2}_{indexX:D2}: {ex.Message}");
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
    
    private static byte[] Pack8To4_63x63(byte[] src, int off)
    {
        // Convert 64x64 8bpp -> 2048 bytes (4bpp) with 63x63 duplication rule; LSB-first nibbles
        var dst = new byte[2048];
        int di = 0;
        for (int y = 0; y < 64; y++)
        {
            int yy = (y == 63) ? 62 : y;
            int rowBase = off + yy * 64;
            for (int i = 0; i < 32; i++)
            {
                int x0 = i * 2;
                int x1 = i * 2 + 1;
                if (x0 >= 63) x0 = 62;
                if (x1 >= 63) x1 = 62;
                byte lo = (byte)((src[rowBase + x0] + 8) >> 4);
                byte hi = (byte)((src[rowBase + x1] + 8) >> 4);
                dst[di++] = (byte)((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
        }
        return dst;
    }
    private static byte[] Pack8To4_64x64(byte[] src, int off)
    {
        // Convert 64x64 8bpp -> 2048 bytes (4bpp) without duplication; LSB-first nibbles
        var dst = new byte[2048];
        int di = 0;
        for (int y = 0; y < 64; y++)
        {
            int rowBase = off + y * 64;
            for (int i = 0; i < 32; i++)
            {
                int x0 = i * 2;
                int x1 = x0 + 1;
                byte lo = (byte)((src[rowBase + x0] + 8) >> 4);
                byte hi = (byte)((src[rowBase + x1] + 8) >> 4);
                dst[di++] = (byte)((lo & 0x0F) | ((hi & 0x0F) << 4));
            }
        }
        return dst;
    }
    
    private static byte[] DecodeLkMcalTo8bpp(byte[] src, int off, int available, bool compressed)
    {
        // Returns exactly 4096 bytes (64x64) or Array.Empty<byte>() on failure
        if (src == null || off < 0 || off >= src.Length) return Array.Empty<byte>();
        // If available<=0, treat as unlimited (read until end or until 4096 produced)
        int limit;
        if (available <= 0)
            limit = src.Length;
        else
            limit = Math.Min(src.Length, off + available);
        if (!compressed)
        {
            // Decide between 8bpp (4096) and 4bpp (2048) based on available span
            int span = Math.Max(0, limit - off);
            if (span >= 4096)
            {
                var dst = new byte[4096];
                Buffer.BlockCopy(src, off, dst, 0, 4096);
                return dst;
            }
            else if (span >= 2048)
            {
                return Expand4bppTo8bpp_64x64(src, off);
            }
            else
            {
                // If span is too small, but buffer still has enough data beyond current limit, fallback to reading to end
                int remaining = src.Length - off;
                if (remaining >= 4096)
                {
                    var dst = new byte[4096];
                    Buffer.BlockCopy(src, off, dst, 0, 4096);
                    return dst;
                }
                return Array.Empty<byte>();
            }
        }

        // Compressed RLE per wiki (rows cannot span); stop at 4096 bytes
        var outBuf = new byte[4096];
        int produced = 0;
        int p = off;
        for (int row = 0; row < 64 && produced < 4096; row++)
        {
            int rowPos = 0;
            while (rowPos < 64 && produced < 4096)
            {
                if (p >= limit) return Array.Empty<byte>();
                byte control = src[p++];
                bool fill = (control & 0x80) != 0;
                int count = control & 0x7F; // 1..127 typical; treat 0 as 64 for safety
                if (count == 0) count = 64;
                int room = 64 - rowPos;
                int take = Math.Min(count, room);
                if (fill)
                {
                    if (p >= limit) return Array.Empty<byte>();
                    byte v = src[p++];
                    for (int i = 0; i < take; i++) outBuf[produced + i] = v;
                    // discard excess if run exceeds row
                }
                else
                {
                    if (p + take > limit) return Array.Empty<byte>();
                    Buffer.BlockCopy(src, p, outBuf, produced, take);
                    p += take;
                    // skip any excess beyond row
                    int excess = count - take;
                    if (excess > 0)
                    {
                        int skip = Math.Min(excess, Math.Max(0, limit - p));
                        p += skip;
                    }
                }
                produced += take;
                rowPos += take;
            }
        }
        return (produced == 4096) ? outBuf : Array.Empty<byte>();
    }

    private static byte[] Expand4bppTo8bpp_63x63(byte[] src, int off)
    {
        // Expand 2048 bytes (64 rows × 32 bytes) to 4096 bytes (64×64),
        // mapping low/high nibbles to 8-bit and duplicating last column and last row.
        var dst = new byte[4096];
        int di = 0;
        for (int y = 0; y < 64; y++)
        {
            int rowBase = off + y * 32; // 32 bytes per row in 4bpp
            for (int i = 0; i < 32; i++)
            {
                byte b = src[rowBase + i];
                byte lo = (byte)(b & 0x0F);
                byte hi = (byte)((b >> 4) & 0x0F);
                // First pixel: low nibble scaled to 8-bit
                dst[di++] = (byte)(lo * 17);
                if (i != 31)
                {
                    // Normal case: second pixel from high nibble
                    dst[di++] = (byte)(hi * 17);
                }
                else
                {
                    // Duplicate last column: use low nibble again for pixel 63
                    dst[di++] = (byte)(lo * 17);
                }
            }
        }
        // Duplicate last row from row 62 into row 63
        Buffer.BlockCopy(dst, 62 * 64, dst, 63 * 64, 64);
        return dst;
    }
    private static byte[] Expand4bppTo8bpp_64x64(byte[] src, int off)
    {
        // Expand 2048 bytes (64×32) to 4096 bytes (64×64) without duplication
        var dst = new byte[4096];
        int di = 0;
        for (int y = 0; y < 64; y++)
        {
            int rowBase = off + y * 32;
            for (int i = 0; i < 32; i++)
            {
                byte b = src[rowBase + i];
                byte lo = (byte)(b & 0x0F);
                byte hi = (byte)((b >> 4) & 0x0F);
                dst[di++] = (byte)(lo * 17);
                dst[di++] = (byte)(hi * 17);
            }
        }
        return dst;
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
    
    private static void ValidateMclyMcalIntegrity(byte[] mclyRaw, byte[] mcalRaw, int indexX, int indexY, LkToAlphaOptions? opts)
    {
        if (mclyRaw.Length == 0 || mcalRaw.Length == 0) return;
        
        int numLayers = mclyRaw.Length / 16;
        if (numLayers == 0) return;
        
        bool hasErrors = false;
        
        // Check each layer's offsetInMCAL value
        for (int i = 0; i < numLayers; i++)
        {
            int layerOffset = i * 16;
            uint textureId = BitConverter.ToUInt32(mclyRaw, layerOffset + 0);
            uint flags = BitConverter.ToUInt32(mclyRaw, layerOffset + 4);
            uint offsetInMCAL = BitConverter.ToUInt32(mclyRaw, layerOffset + 8);
            int effectId = BitConverter.ToInt32(mclyRaw, layerOffset + 12);
            
            // Layer 0 (base) typically has no alpha map
            if (i == 0 && offsetInMCAL == 0) continue;
            
            // Check if offset is within MCAL bounds
            if (offsetInMCAL >= mcalRaw.Length)
            {
                Console.WriteLine($"[ERROR] Tile [{indexY:D2},{indexX:D2}] MCLY layer {i}: offsetInMCAL ({offsetInMCAL}) exceeds MCAL size ({mcalRaw.Length})");
                Console.WriteLine($"        TextureID={textureId}, Flags=0x{flags:X8}, EffectID={effectId}");
                hasErrors = true;
            }
            
            // Warn if offset looks suspicious (e.g., exactly at midpoint, suggesting split data bug)
            if (mcalRaw.Length == 4096 && offsetInMCAL == 2048 && i == 1)
            {
                Console.WriteLine($"[WARNING] Tile [{indexY:D2},{indexX:D2}] MCLY layer {i}: offsetInMCAL (2048) is exactly half of 4096-byte MCAL");
                Console.WriteLine($"          This may indicate the 'gray dot → half-circles' bug!");
            }
            
            // Check typical alpha map sizes
            bool useAlphaMap = (flags & 0x100) != 0; // use_alpha_map flag
            bool isCompressed = (flags & 0x200) != 0; // alpha_map_compressed flag
            
            if (useAlphaMap && offsetInMCAL > 0)
            {
                int remainingBytes = mcalRaw.Length - (int)offsetInMCAL;
                
                // Typical uncompressed sizes: 2048 (4-bit) or 4096 (8-bit)
                // Compressed: 128-4160 bytes
                if (!isCompressed && remainingBytes < 2048)
                {
                    Console.WriteLine($"[WARNING] Tile [{indexY:D2},{indexX:D2}] MCLY layer {i}: Only {remainingBytes} bytes remain for uncompressed alpha map (expected 2048 or 4096)");
                }
            }
        }
        
        // Log success if verbose
        if (!hasErrors && opts?.VerboseLogging == true)
        {
            Console.WriteLine($"[OK] Tile [{indexY:D2},{indexX:D2}] MCLY/MCAL validation passed: {numLayers} layers, {mcalRaw.Length} bytes alpha data");
        }
    }
}
