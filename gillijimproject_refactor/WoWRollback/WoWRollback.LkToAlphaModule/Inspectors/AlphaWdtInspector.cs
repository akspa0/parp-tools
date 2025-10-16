using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace WoWRollback.LkToAlphaModule.Inspectors;

public static class AlphaWdtInspector
{
    private const int ChunkHeader = 8;

    private static string ReadToken(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> hdr = stackalloc byte[4];
        if (fs.Read(hdr) != 4) return "";
        return Encoding.ASCII.GetString(hdr);
    }

    private static int ReadInt32(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> buf = stackalloc byte[4];
        if (fs.Read(buf) != 4) return 0;
        return BitConverter.ToInt32(buf);
    }

    private static (string tokenOnDisk, int size) ReadChunkHeader(FileStream fs, long offset)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        Span<byte> hdr = stackalloc byte[8];
        if (fs.Read(hdr) != 8) return ("", 0);
        string tok = Encoding.ASCII.GetString(hdr.Slice(0, 4));
        int size = BitConverter.ToInt32(hdr.Slice(4, 4));
        return (tok, size);
    }

    private static byte[] ReadBytes(FileStream fs, long offset, int size)
    {
        fs.Seek(offset, SeekOrigin.Begin);
        var buf = new byte[size];
        int read = fs.Read(buf, 0, size);
        if (read < size) Array.Resize(ref buf, read);
        return buf;
    }

    private static List<string> ParseCStringTable(byte[] data)
    {
        var res = new List<string>();
        int start = 0;
        while (start < data.Length)
        {
            int end = Array.IndexOf<byte>(data, 0, start);
            if (end < 0) end = data.Length;
            int len = end - start;
            if (len > 0)
            {
                string s = Encoding.ASCII.GetString(data, start, len);
                if (!string.IsNullOrWhiteSpace(s)) res.Add(s);
            }
            start = end + 1;
        }
        return res;
    }

    private static string ForwardFourCC(string onDisk)
    {
        if (string.IsNullOrEmpty(onDisk) || onDisk.Length != 4) return onDisk;
        return new string(new[] { onDisk[3], onDisk[2], onDisk[1], onDisk[0] });
        }

    /// <summary>
    /// Validates that MHDR offsets point to the correct chunk types
    /// </summary>
    private static bool ValidateOffsets(FileStream fs, long mhdrDataStart, int offsTex, int offsDoo, int offsMob, int sizeTex, int sizeDoo, int sizeMob, out List<string> errors)
    {
        errors = new List<string>();
        bool valid = true;
        
        // Validate MTEX offset (should point to MTEX FourCC - Alpha convention)
        if (offsTex > 0)
        {
            long mtexAbs = mhdrDataStart + offsTex;
            string tokAtOffset = ReadToken(fs, mtexAbs);
            string fwd = ForwardFourCC(tokAtOffset);
            if (fwd != "MTEX")
            {
                errors.Add($"offsTex={offsTex:X} points to offset 0x{mtexAbs:X} with token '{fwd}' not 'MTEX'");
                valid = false;
            }
        }
        
        // Validate MDDF offset (should point to MDDF FourCC for empty chunks)
        if (offsDoo > 0)
        {
            long mddfAbs = mhdrDataStart + offsDoo;
            string tokAtOffset = ReadToken(fs, mddfAbs);
            string fwd = ForwardFourCC(tokAtOffset);
            if (fwd != "MDDF")
            {
                errors.Add($"offsDoo={offsDoo:X} points to offset 0x{mddfAbs:X} with token '{fwd}' not 'MDDF'");
                valid = false;
            }
        }
        
        // Validate MODF offset (should point to MODF FourCC for empty chunks)
        if (offsMob > 0)
        {
            long modfAbs = mhdrDataStart + offsMob;
            string tokAtOffset = ReadToken(fs, modfAbs);
            string fwd = ForwardFourCC(tokAtOffset);
            if (fwd != "MODF")
            {
                errors.Add($"offsMob={offsMob:X} points to offset 0x{modfAbs:X} with token '{fwd}' not 'MODF' - THIS IS THE BUG!");
                valid = false;
            }
        }
        
        return valid;
    }

    public static void Inspect(string wdtPath, int sampleTiles, string? jsonPath = null)
    {
        using var fs = File.OpenRead(wdtPath);
        Console.WriteLine($"[wdt] file: {wdtPath} size={fs.Length}");

        var report = new Report
        {
            File = wdtPath,
            Size = fs.Length,
            TopLevel = new(),
            Tiles = new()
        };

        // Top-level scan
        long off = 0;
        var topOrder = new List<(long off, string onDisk, string fwd, int size)>();
        while (off + ChunkHeader <= fs.Length)
        {
            var (tok, size) = ReadChunkHeader(fs, off);
            if (string.IsNullOrWhiteSpace(tok)) break;
            var fwd = ForwardFourCC(tok);
            topOrder.Add((off, tok, fwd, size));
            report.TopLevel.Add(new ReportTop
            {
                Offset = off,
                OnDisk = tok,
                Fwd = fwd,
                Size = size
            });
            int pad = (size & 1) == 1 ? 1 : 0;
            long next = off + ChunkHeader + size + pad;
            if (next <= off) break; // safety
            off = next;
            if (topOrder.Count > 32) break; // enough to see ordering
        }
        Console.WriteLine("[wdt] top-level order:");
        foreach (var t in topOrder)
            Console.WriteLine($"  @0x{t.off:X8} on-disk='{t.onDisk}' fwd='{t.fwd}' size={t.size}");

        // Locate MPHD and MAIN from order
        long mphdDataStart = -1;
        long mainDataStart = -1;
        foreach (var t in topOrder)
        {
            if (t.fwd == "MPHD") mphdDataStart = t.off + ChunkHeader;
            if (t.fwd == "MAIN") mainDataStart = t.off + ChunkHeader;
        }
        if (mphdDataStart < 0 || mainDataStart < 0)
        {
            Console.WriteLine("[wdt] MPHD or MAIN not found, abort");
            return;
        }

        // MDNM/MONM from MPHD
        int offsMdnm = ReadInt32(fs, mphdDataStart + 4);
        int offsMonm = ReadInt32(fs, mphdDataStart + 12);
        Console.WriteLine($"[wdt] MPHD: offsMdnm=0x{offsMdnm:X}, token='{ReadToken(fs, offsMdnm)}' offsMonm=0x{offsMonm:X}, token='{ReadToken(fs, offsMonm)}'");
        report.Mphd = new ReportMphd
        {
            OffsMdnm = offsMdnm,
            OffsMonm = offsMonm,
            MdnmToken = ReadToken(fs, offsMdnm),
            MonmToken = ReadToken(fs, offsMonm)
        };

        // MAIN: 4096 cells * 16
        int nonZero = 0;
        var tiles = new List<(int idx, int mhdrOff, int sizeToFirstMcnk)>();
        for (int i = 0; i < 4096; i++)
        {
            long cell = mainDataStart + (i * 16);
            int mhdrOff = ReadInt32(fs, cell);
            int sizeToFirst = ReadInt32(fs, cell + 4);
            if (mhdrOff != 0)
            {
                nonZero++;
                if (tiles.Count < sampleTiles)
                    tiles.Add((i, mhdrOff, sizeToFirst));
            }
        }
        Console.WriteLine($"[main] non-zero tiles: {nonZero}");
        report.Main = new ReportMain { NonZeroTiles = nonZero, MainChecks = new List<ReportMainCheck>() };

        // MAIN validation: check first K non-zero cells
        int mainChecksLimit = Math.Min(sampleTiles, tiles.Count);
        Console.WriteLine($"[main] validating first {mainChecksLimit} MAIN cells...");
        for (int k = 0; k < mainChecksLimit; k++)
        {
            var t = tiles[k];
            string startToken = ReadToken(fs, t.mhdrOff);
            string endToken = ReadToken(fs, t.mhdrOff + t.sizeToFirstMcnk);
            var check = new ReportMainCheck
            {
                Index = t.idx,
                X = t.idx % 64,
                Y = t.idx / 64,
                Offset = t.mhdrOff,
                Size = t.sizeToFirstMcnk,
                StartToken = startToken,
                EndToken = endToken
            };
            report.Main.MainChecks.Add(check);
            Console.WriteLine($"  MAIN[{t.idx}] ({check.Y:D2}_{check.X:D2}): offset=0x{t.mhdrOff:X} -> '{startToken}', offset+size=0x{t.mhdrOff + t.sizeToFirstMcnk:X} -> '{endToken}'");
            Console.WriteLine($"    → MHDR letters @0x{t.mhdrOff:X}, data @0x{t.mhdrOff + 8:X}, size={t.sizeToFirstMcnk} (0x{t.sizeToFirstMcnk:X})");
        }

        // For a few tiles, detect MHDR->MCIN base and validate first MCNK
        foreach (var t in tiles)
        {
            int x = t.idx % 64, y = t.idx / 64;
            string mhdrTok = ReadToken(fs, t.mhdrOff);
            long mhdrDataStart = t.mhdrOff + 8;
            int offsInfoDataRel = ReadInt32(fs, mhdrDataStart + 0);
            int offsTex = ReadInt32(fs, mhdrDataStart + 4);
            int sizeTex = ReadInt32(fs, mhdrDataStart + 8);
            int offsDoo = ReadInt32(fs, mhdrDataStart + 0x0C);
            int sizeDoo = ReadInt32(fs, mhdrDataStart + 0x10);
            int offsMob = ReadInt32(fs, mhdrDataStart + 0x14);
            int sizeMob = ReadInt32(fs, mhdrDataStart + 0x18);
            long mcinAtDataRel = t.mhdrOff + 8 + offsInfoDataRel;
            long mcinAtStartRel = t.mhdrOff + offsInfoDataRel;
            string tokDataRel = ReadToken(fs, mcinAtDataRel);
            string tokStartRel = ReadToken(fs, mcinAtStartRel);
            long firstMcnkAbs = t.mhdrOff + t.sizeToFirstMcnk;
            string firstMcnkTok = ReadToken(fs, firstMcnkAbs);
            Console.WriteLine($"[tile {y:D2}_{x:D2}] MHDR @0x{t.mhdrOff:X} token='{mhdrTok}'");
            Console.WriteLine($"  MHDR.data @0x{mhdrDataStart:X}");
            Console.WriteLine($"  offsInfo={offsInfoDataRel} (0x{offsInfoDataRel:X}) -> MCIN @0x{mcinAtDataRel:X} token='{tokDataRel}'");
            Console.WriteLine($"  offsTex={offsTex} (0x{offsTex:X}), sizeTex={sizeTex}");
            Console.WriteLine($"  offsDoo={offsDoo} (0x{offsDoo:X}), sizeDoo={sizeDoo}");
            Console.WriteLine($"  offsMob={offsMob} (0x{offsMob:X}), sizeMob={sizeMob}");
            Console.WriteLine($"  firstMCNK @0x{firstMcnkAbs:X} token='{firstMcnkTok}' (MAIN.size={t.sizeToFirstMcnk})");
            
            // VALIDATE OFFSETS
            if (!ValidateOffsets(fs, mhdrDataStart, offsTex, offsDoo, offsMob, sizeTex, sizeDoo, sizeMob, out var offsetErrors))
            {
                Console.WriteLine($"  ❌ OFFSET VALIDATION FAILED:");
                foreach (var err in offsetErrors)
                {
                    Console.WriteLine($"     {err}");
                }
            }
            else
            {
                Console.WriteLine($"  ✓ Offset validation passed");
            }
            
            // Calculate chunk spacing
            long mcinAbsolute = mcinAtDataRel;
            long mtexAbsolute = offsTex > 0 ? mhdrDataStart + offsTex : 0;
            long mddfAbsolute = offsDoo > 0 ? mhdrDataStart + offsDoo : 0;
            long modfAbsolute = offsMob > 0 ? mhdrDataStart + offsMob : 0;
            Console.WriteLine($"  Chunk order: MHDR→MCIN gap={(mcinAbsolute - mhdrDataStart - 64):+0;-0;0}");
            if (mtexAbsolute > 0) Console.WriteLine($"    MCIN→MTEX gap={(mtexAbsolute - mcinAbsolute - 8 - 4096):+0;-0;0}");
            if (mddfAbsolute > 0) Console.WriteLine($"    MTEX→MDDF gap={(mddfAbsolute - mtexAbsolute - 8 - sizeTex):+0;-0;0}");
            if (modfAbsolute > 0 && mddfAbsolute > 0) Console.WriteLine($"    MDDF→MODF gap={(modfAbsolute - mddfAbsolute - 8 - sizeDoo):+0;-0;0}");
            if (firstMcnkAbs > 0 && modfAbsolute > 0) Console.WriteLine($"    MODF→MCNK[0] gap={(firstMcnkAbs - modfAbsolute - 8 - sizeMob):+0;-0;0}");
            var repTile = new ReportTile
            {
                Index = t.idx,
                X = x,
                Y = y,
                MhdrOffset = t.mhdrOff,
                MhdrToken = mhdrTok,
                OffsInfoDataRel = offsInfoDataRel,
                McinAtDataRelToken = tokDataRel,
                McinAtStartRelToken = tokStartRel,
                FirstMcnkAbs = firstMcnkAbs,
                FirstMcnkToken = firstMcnkTok,
                Mhdr = new ReportMhdr 
                { 
                    OffsInfo = offsInfoDataRel,
                    OffsTex = offsTex, 
                    SizeTex = sizeTex, 
                    OffsDoo = offsDoo, 
                    SizeDoo = sizeDoo, 
                    OffsMob = offsMob, 
                    SizeMob = sizeMob,
                    McinAbsolute = mcinAtDataRel,
                    MtexAbsolute = offsTex > 0 ? mhdrDataStart + offsTex : 0,
                    MddfAbsolute = offsDoo > 0 ? mhdrDataStart + offsDoo : 0,
                    ModfAbsolute = offsMob > 0 ? mhdrDataStart + offsMob : 0
                }
            };

            // Inspect MCIN table and enumerate all 256 MCNKs
            var (mcinTok, mcinSize) = ReadChunkHeader(fs, mcinAtDataRel);
            if (ForwardFourCC(mcinTok) == "MCIN")
            {
                long mcinDataStart = mcinAtDataRel + ChunkHeader;
                repTile.Mcin = new ReportMcin { Size = mcinSize, Entry0 = null };

                var mcins = new List<ReportMcinEntryDetailed>(256);
                var present = new List<int>(256);
                for (int idx = 0; idx < 256; idx++)
                {
                    long entry = mcinDataStart + idx * 16;
                    int offVal = ReadInt32(fs, entry + 0);
                    int sz = ReadInt32(fs, entry + 4);
                    mcins.Add(new ReportMcinEntryDetailed { Index = idx, Offset = offVal, Size = sz });
                    if (idx == 0)
                    {
                        string mcnk0Tok = offVal > 0 ? ReadToken(fs, offVal) : "";
                        repTile.Mcin.Entry0 = new ReportMcinEntry { Offset = offVal, Size = sz, Token = mcnk0Tok };
                    }
                    if (offVal > 0) present.Add(idx);
                }
                repTile.McinEntries = mcins;
                repTile.PresentMcnkCount = present.Count;
                repTile.PresentMcnkIndices = present;
                Console.WriteLine($"  MCIN present: {present.Count}; indices: {string.Join(",", present.GetRange(0, Math.Min(32, present.Count)))}{(present.Count>32?"...":"")}");

                // Parse MTEX strings if present
                if (offsTex > 0 && sizeTex >= 0)
                {
                    long mtexAbs = mhdrDataStart + offsTex;
                    var (mtexTok, mtexSize) = ReadChunkHeader(fs, mtexAbs);
                    if (ForwardFourCC(mtexTok) == "MTEX" && mtexSize >= 0 && mtexSize < 1_000_000)
                    {
                        var table = ReadBytes(fs, mtexAbs + ChunkHeader, mtexSize);
                        var names = ParseCStringTable(table);
                        repTile.Mtex = new ReportMtex { Count = names.Count, First = names.Count > 0 ? names[0] : string.Empty };
                    }
                }

                // Inspect all present MCNKs
                var mcnkDetails = new List<ReportMcnkDetail>(present.Count);
                foreach (int idx in present)
                {
                    int mcnkOffset = mcins[idx].Offset;
                    if (mcnkOffset <= 0) continue;
                    var (tok, _) = ReadChunkHeader(fs, mcnkOffset);
                    if (ForwardFourCC(tok) != "MCNK") continue;

                    long mcnkHdrData = mcnkOffset + ChunkHeader;
                    byte[] hdr = ReadBytes(fs, mcnkHdrData, 128);
                    if (hdr.Length != 128) continue;

                    // Decode ALL MCNK header fields (128 bytes total)
                    uint Flags = BitConverter.ToUInt32(hdr, 0x00);
                    uint IndexX = BitConverter.ToUInt32(hdr, 0x04);
                    uint IndexY = BitConverter.ToUInt32(hdr, 0x08);
                    float Radius = BitConverter.ToSingle(hdr, 0x0C);
                    uint NLayers = BitConverter.ToUInt32(hdr, 0x10);
                    uint NDoodadRefs = BitConverter.ToUInt32(hdr, 0x14);
                    int McvtOffset = BitConverter.ToInt32(hdr, 0x18);
                    int McnrOffset = BitConverter.ToInt32(hdr, 0x1C);
                    int MclyOffset = BitConverter.ToInt32(hdr, 0x20);
                    int McrfOffset = BitConverter.ToInt32(hdr, 0x24);
                    int McalOffset = BitConverter.ToInt32(hdr, 0x28);
                    int McalSize = BitConverter.ToInt32(hdr, 0x2C);
                    int McshOffset = BitConverter.ToInt32(hdr, 0x30);
                    int McshSize = BitConverter.ToInt32(hdr, 0x34);
                    uint AreaId = BitConverter.ToUInt32(hdr, 0x38);
                    uint NMapObjRefs = BitConverter.ToUInt32(hdr, 0x3C);
                    ushort Holes = BitConverter.ToUInt16(hdr, 0x40);
                    ushort Pad0 = BitConverter.ToUInt16(hdr, 0x42);
                    // PredTex: 8 uint16s at 0x44-0x53
                    ushort[] PredTex = new ushort[8];
                    for (int pt = 0; pt < 8; pt++) PredTex[pt] = BitConverter.ToUInt16(hdr, 0x44 + pt * 2);
                    // NoEffectDoodad: 8 bytes at 0x54-0x5B
                    byte[] NoEffectDoodad = new byte[8];
                    Buffer.BlockCopy(hdr, 0x54, NoEffectDoodad, 0, 8);
                    int McseOffset = BitConverter.ToInt32(hdr, 0x5C);
                    uint NSndEmitters = BitConverter.ToUInt32(hdr, 0x60);
                    int MclqOffset = BitConverter.ToInt32(hdr, 0x64);
                    // Pad1: 24 bytes at 0x68-0x7F
                    int McnkChunksSize = McseOffset; // Note: This field appears to be at 0x5C in some docs

                    long subBase = mcnkHdrData + 128;
                    long mcvtAbs = subBase + McvtOffset;
                    long mcnrAbs = subBase + McnrOffset;
                    long mclyAbs = subBase + MclyOffset;
                    long mcrfAbs = subBase + McrfOffset;
                    long mcalAbs = McalOffset > 0 ? subBase + McalOffset : 0;
                    long mcshAbs = McshOffset > 0 ? subBase + McshOffset : 0;

                    // Sample MCVT min/max
                    float minH = float.MaxValue, maxH = float.MinValue;
                    try
                    {
                        var mcvt = ReadBytes(fs, mcvtAbs, 145 * 4);
                        int n = mcvt.Length / 4;
                        for (int k = 0; k < n; k++)
                        {
                            float v = BitConverter.ToSingle(mcvt, k * 4);
                            if (v < minH) minH = v;
                            if (v > maxH) maxH = v;
                        }
                    }
                    catch { }

                    // MCLY layer0
                    int layer0Tex = -1; int layerCount = 0;
                    var (mclyTok, mclySize) = ReadChunkHeader(fs, mclyAbs);
                    if (ForwardFourCC(mclyTok) == "MCLY" && mclySize >= 16)
                    {
                        layerCount = mclySize / 16;
                        byte[] l0 = ReadBytes(fs, mclyAbs + ChunkHeader, 16);
                        if (l0.Length == 16) layer0Tex = BitConverter.ToInt32(l0, 0);
                    }

                    var det = new ReportMcnkDetail
                    {
                        Index = idx,
                        McnkOffset = mcnkOffset,
                        Header = new ReportMcnkHeader
                        {
                            Flags = $"0x{Flags:X8}",
                            IndexX = (int)IndexX,
                            IndexY = (int)IndexY,
                            Radius = Radius,
                            NLayers = (int)NLayers,
                            NDoodadRefs = (int)NDoodadRefs,
                            McvtOffset = McvtOffset,
                            McnrOffset = McnrOffset,
                            MclyOffset = MclyOffset,
                            McrfOffset = McrfOffset,
                            McalOffset = McalOffset,
                            McalSize = McalSize,
                            McshOffset = McshOffset,
                            McshSize = McshSize,
                            AreaId = $"0x{AreaId:X8}",
                            AreaIdZone = (int)(AreaId >> 16),
                            AreaIdSubzone = (int)(AreaId & 0xFFFF),
                            NMapObjRefs = (int)NMapObjRefs,
                            Holes = $"0x{Holes:X4}",
                            PredTex = PredTex,
                            NoEffectDoodad = NoEffectDoodad,
                            McseOffset = McseOffset,
                            NSndEmitters = (int)NSndEmitters,
                            MclqOffset = MclqOffset
                        },
                        NLayers = (int)NLayers,
                        AreaCandidate = (int)AreaId,
                        McnkChunksSize = McnkChunksSize,
                        Mcvt = new ReportSub { Abs = mcvtAbs },
                        Mcnr = new ReportSub { Abs = mcnrAbs },
                        Mcly = new ReportSub { Abs = mclyAbs, Token = ForwardFourCC(mclyTok), Size = mclySize },
                        Mcrf = new ReportSub { Abs = mcrfAbs, Token = ForwardFourCC(ReadChunkHeader(fs, mcrfAbs).tokenOnDisk), Size = ReadChunkHeader(fs, mcrfAbs).size },
                        Mcal = mcalAbs > 0 ? new ReportSub { Abs = mcalAbs, Token = ForwardFourCC(ReadChunkHeader(fs, mcalAbs).tokenOnDisk), Size = ReadChunkHeader(fs, mcalAbs).size } : null,
                        Mcsh = mcshAbs > 0 ? new ReportSub { Abs = mcshAbs, Token = ForwardFourCC(ReadChunkHeader(fs, mcshAbs).tokenOnDisk), Size = ReadChunkHeader(fs, mcshAbs).size } : null,
                        McvtMin = minH,
                        McvtMax = maxH,
                        Layer0Texture = layer0Tex
                    };
                    mcnkDetails.Add(det);
                }

                repTile.McnkDetails = mcnkDetails;
            }

            report.Tiles.Add(repTile);
        }

        if (!string.IsNullOrWhiteSpace(jsonPath))
        {
            var opts = new JsonSerializerOptions { WriteIndented = true, DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull };
            File.WriteAllText(jsonPath, JsonSerializer.Serialize(report, opts));
            Console.WriteLine($"[wdt] wrote JSON report: {jsonPath}");
        }
    }

    private sealed class Report
    {
        public string File { get; set; } = string.Empty;
        public long Size { get; set; }
        public List<ReportTop> TopLevel { get; set; } = new();
        public ReportMphd? Mphd { get; set; }
        public ReportMain? Main { get; set; }
        public List<ReportTile> Tiles { get; set; } = new();
    }
    private sealed class ReportTop { public long Offset { get; set; } public string OnDisk { get; set; } = string.Empty; public string Fwd { get; set; } = string.Empty; public int Size { get; set; } }
    private sealed class ReportMphd { public int OffsMdnm { get; set; } public int OffsMonm { get; set; } public string MdnmToken { get; set; } = string.Empty; public string MonmToken { get; set; } = string.Empty; }
    private sealed class ReportMain { public int NonZeroTiles { get; set; } public List<ReportMainCheck>? MainChecks { get; set; } }
    private sealed class ReportMainCheck
    {
        public int Index { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int Offset { get; set; }
        public int Size { get; set; }
        public string StartToken { get; set; } = string.Empty;
        public string EndToken { get; set; } = string.Empty;
    }
    private sealed class ReportTile
    {
        public int Index { get; set; }
        public int X { get; set; }
        public int Y { get; set; }
        public int MhdrOffset { get; set; }
        public string MhdrToken { get; set; } = string.Empty;
        public int OffsInfoDataRel { get; set; }
        public string McinAtDataRelToken { get; set; } = string.Empty;
        public string McinAtStartRelToken { get; set; } = string.Empty;
        public long FirstMcnkAbs { get; set; }
        public string FirstMcnkToken { get; set; } = string.Empty;
        public ReportMcin? Mcin { get; set; }
        public ReportMhdr? Mhdr { get; set; }
        public ReportMtex? Mtex { get; set; }
        public ReportMcnk? Mcnk { get; set; }
        public List<ReportMcinEntryDetailed>? McinEntries { get; set; }
        public int PresentMcnkCount { get; set; }
        public List<int>? PresentMcnkIndices { get; set; }
        public List<ReportMcnkDetail>? McnkDetails { get; set; }
    }
    private sealed class ReportMcin { public int Size { get; set; } public ReportMcinEntry? Entry0 { get; set; } }
    private sealed class ReportMcinEntry { public int Offset { get; set; } public int Size { get; set; } public string Token { get; set; } = string.Empty; }

    private sealed class ReportMhdr
    {
        public int OffsInfo { get; set; }
        public int OffsTex { get; set; }
        public int SizeTex { get; set; }
        public int OffsDoo { get; set; }
        public int SizeDoo { get; set; }
        public int OffsMob { get; set; }
        public int SizeMob { get; set; }
        public long McinAbsolute { get; set; }
        public long MtexAbsolute { get; set; }
        public long MddfAbsolute { get; set; }
        public long ModfAbsolute { get; set; }
    }
    private sealed class ReportMtex { public int Count { get; set; } public string First { get; set; } = string.Empty; }
    private sealed class ReportSub { public long Abs { get; set; } public string Token { get; set; } = string.Empty; public int Size { get; set; } }
    private sealed class ReportMcnk
    {
        public int NLayers { get; set; }
        public int AreaCandidate { get; set; }
        public int McnkChunksSize { get; set; }
        public ReportSub? Mcvt { get; set; }
        public ReportSub? Mcnr { get; set; }
        public ReportSub? Mcly { get; set; }
        public ReportSub? Mcrf { get; set; }
        public ReportSub? Mcal { get; set; }
        public ReportSub? Mcsh { get; set; }
        public float McvtMin { get; set; }
        public float McvtMax { get; set; }
        public int Layer0Texture { get; set; }
    }

    private sealed class ReportMcinEntryDetailed
    {
        public int Index { get; set; }
        public int Offset { get; set; }
        public int Size { get; set; }
    }

    private sealed class ReportMcnkDetail
    {
        public int Index { get; set; }
        public int McnkOffset { get; set; }
        public ReportMcnkHeader? Header { get; set; }
        public int NLayers { get; set; }
        public int AreaCandidate { get; set; }
        public int McnkChunksSize { get; set; }
        public ReportSub? Mcvt { get; set; }
        public ReportSub? Mcnr { get; set; }
        public ReportSub? Mcly { get; set; }
        public ReportSub? Mcrf { get; set; }
        public ReportSub? Mcal { get; set; }
        public ReportSub? Mcsh { get; set; }
        public float McvtMin { get; set; }
        public float McvtMax { get; set; }
        public int Layer0Texture { get; set; }
    }

    private sealed class ReportMcnkHeader
    {
        public string Flags { get; set; } = string.Empty;
        public int IndexX { get; set; }
        public int IndexY { get; set; }
        public float Radius { get; set; }
        public int NLayers { get; set; }
        public int NDoodadRefs { get; set; }
        public int McvtOffset { get; set; }
        public int McnrOffset { get; set; }
        public int MclyOffset { get; set; }
        public int McrfOffset { get; set; }
        public int McalOffset { get; set; }
        public int McalSize { get; set; }
        public int McshOffset { get; set; }
        public int McshSize { get; set; }
        public string AreaId { get; set; } = string.Empty;
        public int AreaIdZone { get; set; }
        public int AreaIdSubzone { get; set; }
        public int NMapObjRefs { get; set; }
        public string Holes { get; set; } = string.Empty;
        public ushort[] PredTex { get; set; } = Array.Empty<ushort>();
        public byte[] NoEffectDoodad { get; set; } = Array.Empty<byte>();
        public int McseOffset { get; set; }
        public int NSndEmitters { get; set; }
        public int MclqOffset { get; set; }
    }

    public static void CompareFiles(string referenceFile, string testFile, int maxBytes)
    {
        Console.WriteLine($"Comparing Alpha WDT files:");
        Console.WriteLine($"  Reference: {referenceFile}");
        Console.WriteLine($"  Test:      {testFile}");
        Console.WriteLine();

        if (!File.Exists(referenceFile))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"ERROR: Reference file not found: {referenceFile}");
            Console.ResetColor();
            return;
        }

        if (!File.Exists(testFile))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"ERROR: Test file not found: {testFile}");
            Console.ResetColor();
            return;
        }

        var refBytes = File.ReadAllBytes(referenceFile);
        var testBytes = File.ReadAllBytes(testFile);

        Console.WriteLine($"File sizes:");
        Console.WriteLine($"  Reference: {refBytes.Length:N0} bytes");
        Console.WriteLine($"  Test:      {testBytes.Length:N0} bytes");
        
        if (refBytes.Length != testBytes.Length)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"  Difference: {testBytes.Length - refBytes.Length:+#;-#;0} bytes");
            Console.ResetColor();
        }
        Console.WriteLine();

        int compareLength = Math.Min(Math.Min(refBytes.Length, testBytes.Length), maxBytes);
        int differenceCount = 0;
        int firstDifferenceOffset = -1;

        for (int i = 0; i < compareLength; i++)
        {
            if (refBytes[i] != testBytes[i])
            {
                if (firstDifferenceOffset == -1)
                {
                    firstDifferenceOffset = i;
                }
                differenceCount++;
            }
        }

        if (differenceCount == 0)
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"✓ Files are IDENTICAL (first {compareLength:N0} bytes)");
            Console.ResetColor();
            return;
        }

        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"✗ Found {differenceCount:N0} byte differences in first {compareLength:N0} bytes");
        Console.ResetColor();
        Console.WriteLine();

        // Show first difference in detail
        Console.WriteLine($"First difference at offset 0x{firstDifferenceOffset:X} ({firstDifferenceOffset:N0}):");
        
        int contextStart = Math.Max(0, firstDifferenceOffset - 16);
        int contextEnd = Math.Min(compareLength, firstDifferenceOffset + 48);

        Console.WriteLine();
        Console.WriteLine("Reference:");
        PrintHexContext(refBytes, contextStart, contextEnd, firstDifferenceOffset);
        
        Console.WriteLine();
        Console.WriteLine("Test:");
        PrintHexContext(testBytes, contextStart, contextEnd, firstDifferenceOffset);

        // Try to identify what chunk we're in
        Console.WriteLine();
        IdentifyChunkAtOffset(refBytes, firstDifferenceOffset, "Reference");
        IdentifyChunkAtOffset(testBytes, firstDifferenceOffset, "Test");
    }

    private static void PrintHexContext(byte[] data, int start, int end, int highlightOffset)
    {
        for (int i = start; i < end; i += 16)
        {
            Console.Write($"  {i:X8}: ");
            
            // Hex bytes
            for (int j = 0; j < 16 && i + j < end; j++)
            {
                if (i + j == highlightOffset)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                
                Console.Write($"{data[i + j]:X2} ");
                
                if (i + j == highlightOffset)
                    Console.ResetColor();
            }
            
            // ASCII
            Console.Write(" | ");
            for (int j = 0; j < 16 && i + j < end; j++)
            {
                byte b = data[i + j];
                char c = (b >= 32 && b < 127) ? (char)b : '.';
                
                if (i + j == highlightOffset)
                    Console.ForegroundColor = ConsoleColor.Yellow;
                
                Console.Write(c);
                
                if (i + j == highlightOffset)
                    Console.ResetColor();
            }
            Console.WriteLine();
        }
    }

    private static void IdentifyChunkAtOffset(byte[] data, int offset, string label)
    {
        // Search backwards for chunk header (FourCC + size)
        for (int i = offset; i >= 0; i -= 4)
        {
            if (i + 8 <= data.Length)
            {
                string token = Encoding.ASCII.GetString(data, i, 4);
                int size = BitConverter.ToInt32(data, i + 4);
                
                // Check if this looks like a valid chunk header
                if (IsValidFourCC(token) && size >= 0 && size < 100_000_000)
                {
                    int dataStart = i + 8;
                    int dataEnd = dataStart + size;
                    
                    if (offset >= dataStart && offset < dataEnd)
                    {
                        int offsetInChunk = offset - dataStart;
                        Console.WriteLine($"{label}: Inside chunk '{ForwardFourCC(token)}' at offset +0x{offsetInChunk:X} ({offsetInChunk}) from chunk data start");
                        Console.WriteLine($"         Chunk header at 0x{i:X}, size={size}");
                        return;
                    }
                }
            }
        }
        
        Console.WriteLine($"{label}: Could not identify chunk");
    }

    private static bool IsValidFourCC(string token)
    {
        if (token.Length != 4) return false;
        foreach (char c in token)
        {
            if (c < 32 || c > 126) return false;
        }
        return true;
    }
}
