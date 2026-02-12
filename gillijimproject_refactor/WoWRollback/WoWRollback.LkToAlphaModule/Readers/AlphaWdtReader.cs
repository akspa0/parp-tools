using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Readers;

public sealed class AlphaWdtReader
{
    private readonly byte[] _bytes;
    private readonly string _path;

    private AlphaWdtReader(string path)
    {
        _path = path;
        _bytes = File.ReadAllBytes(path);
    }

    public static AlphaWdt Read(string path)
    {
        var r = new AlphaWdtReader(path);
        return r.ReadInternal();
    }

    private AlphaWdt ReadInternal()
    {
        var wdt = new AlphaWdt { Path = _path };

        // Top-level chunk scan
        var top = ScanTopLevelChunks();
        if (top.TryGetValue("DHPM", out var mphd))
        {
            wdt.Mphd = ReadMphd(mphd.offset + 8);
        }

        // MAIN grid (64x64)
        if (!top.TryGetValue("NIAM", out var main))
            return wdt;

        int mainData = main.offset + 8;
        for (int i = 0; i < 4096; i++)
        {
            int pos = mainData + i * 16;
            if (pos + 16 > _bytes.Length) break;
            int mhdrAbs = BitConverter.ToInt32(_bytes, pos + 0);
            int sizeToFirstMcnk = BitConverter.ToInt32(_bytes, pos + 4);
            if (mhdrAbs <= 0) continue;

            var tile = new AlphaTile { Index = i, MhdrOffset = mhdrAbs };
            tile.Mhdr = ReadMhdr(mhdrAbs + 8);

            // Resolve first MCNK by max(end of MCIN, MTEX, MDDF, MODF)
            int mhdrDataStart = mhdrAbs + 8;
            int mcInEnd = mhdrDataStart + tile.Mhdr.OffsInfo + (8 + 4096);
            int mtExEnd = mhdrDataStart + tile.Mhdr.OffsTex + (8 + tile.Mhdr.SizeTex);
            int mdDfEnd = mhdrDataStart + tile.Mhdr.OffsDoo + (8 + tile.Mhdr.SizeDoo);
            int moDfEnd = mhdrDataStart + tile.Mhdr.OffsMob + (8 + tile.Mhdr.SizeMob);
            int firstMcnk = Math.Max(Math.Max(mcInEnd, mtExEnd), Math.Max(mdDfEnd, moDfEnd));
            if (firstMcnk + 8 + 128 <= _bytes.Length)
            {
                tile.FirstMcnk = ReadSmChunk(firstMcnk + 8);

                // Read MCSE if header points to it
                if (tile.FirstMcnk.OffsSndEmitters > 0)
                {
                    int mcsePos = (firstMcnk + 8 + 128) + tile.FirstMcnk.OffsSndEmitters;
                    // Expect named MCSE (reversed 'ESCM')
                    if (mcsePos + 8 <= _bytes.Length)
                    {
                        string fcc = Encoding.ASCII.GetString(_bytes, mcsePos, 4);
                        int size = BitConverter.ToInt32(_bytes, mcsePos + 4);
                        if (fcc == "ESCM" && mcsePos + 8 + size <= _bytes.Length)
                        {
                            var dataStart = mcsePos + 8;
                            int count = 0; int entrySize = 0;
                            if (size % 76 == 0) { entrySize = 76; count = size / 76; }
                            else if (size % 52 == 0) { entrySize = 52; count = size / 52; }
                            for (int n = 0; n < count; n++)
                            {
                                int epos = dataStart + n * entrySize;
                                var e = ReadMcseEntry(epos, entrySize);
                                if (e != null) tile.Mcse.Add(e);
                            }
                        }
                    }
                }
            }

            wdt.Tiles.Add(tile);
        }

        return wdt;
    }

    private AlphaMcseEntry? ReadMcseEntry(int pos, int entrySize)
    {
        try
        {
            if (entrySize == 76)
            {
                var e = new AlphaMcseEntry
                {
                    SoundPointId = BitConverter.ToUInt32(_bytes, pos + 0),
                    SoundNameId = BitConverter.ToUInt32(_bytes, pos + 4),
                    PosX = BitConverter.ToSingle(_bytes, pos + 8),
                    PosY = BitConverter.ToSingle(_bytes, pos + 12),
                    PosZ = BitConverter.ToSingle(_bytes, pos + 16),
                    MinDistance = BitConverter.ToSingle(_bytes, pos + 0x14),
                    MaxDistance = BitConverter.ToSingle(_bytes, pos + 0x18),
                    CutoffDistance = BitConverter.ToSingle(_bytes, pos + 0x1C),
                    StartTime = BitConverter.ToUInt32(_bytes, pos + 0x20),
                    EndTime = BitConverter.ToUInt32(_bytes, pos + 0x24),
                    Mode = BitConverter.ToUInt32(_bytes, pos + 0x28),
                    GroupSilenceMin = BitConverter.ToUInt32(_bytes, pos + 0x2C),
                    GroupSilenceMax = BitConverter.ToUInt32(_bytes, pos + 0x30),
                    PlayInstancesMin = BitConverter.ToUInt32(_bytes, pos + 0x34),
                    PlayInstancesMax = BitConverter.ToUInt32(_bytes, pos + 0x38),
                    LoopCountMin = BitConverter.ToUInt32(_bytes, pos + 0x3C),
                    LoopCountMax = BitConverter.ToUInt32(_bytes, pos + 0x40),
                    InterSoundGapMin = BitConverter.ToUInt32(_bytes, pos + 0x44),
                    InterSoundGapMax = BitConverter.ToUInt32(_bytes, pos + 0x48)
                };
                return e;
            }
            else if (entrySize == 52)
            {
                // Older variant (1.12.1). Map available fields; others remain default.
                var e = new AlphaMcseEntry
                {
                    SoundPointId = BitConverter.ToUInt32(_bytes, pos + 0),
                    SoundNameId = BitConverter.ToUInt32(_bytes, pos + 4),
                    PosX = BitConverter.ToSingle(_bytes, pos + 8),
                    PosY = BitConverter.ToSingle(_bytes, pos + 12),
                    PosZ = BitConverter.ToSingle(_bytes, pos + 16),
                    MinDistance = BitConverter.ToSingle(_bytes, pos + 0x14),
                    MaxDistance = BitConverter.ToSingle(_bytes, pos + 0x18),
                    CutoffDistance = BitConverter.ToSingle(_bytes, pos + 0x1C),
                    // start/end as uint16 in doc; store as uint32 for simplicity
                    StartTime = BitConverter.ToUInt16(_bytes, pos + 0x20),
                    EndTime = BitConverter.ToUInt16(_bytes, pos + 0x22)
                };
                return e;
            }
        }
        catch { /* ignore malformed entries */ }
        return null;
    }

    private AlphaMcnkHeader ReadSmChunk(int mcnkDataStart)
    {
        // Read fields at exact offsets per Alpha.md
        var h = new AlphaMcnkHeader
        {
            Flags = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x00),
            IndexX = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x04),
            IndexY = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x08),
            Radius = BitConverter.ToSingle(_bytes, mcnkDataStart + 0x0C),
            NLayers = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x10),
            NDoodadRefs = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x14),
            OffsHeight = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x18),
            OffsNormal = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x1C),
            OffsLayer = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x20),
            OffsRefs = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x24),
            OffsAlpha = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x28),
            SizeAlpha = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x2C),
            OffsShadow = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x30),
            SizeShadow = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x34),
            AreaId = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x38),
            NMapObjRefs = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x3C),
            Holes = BitConverter.ToUInt16(_bytes, mcnkDataStart + 0x40),
            OffsSndEmitters = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x5C),
            NSndEmitters = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x60),
            OffsLiquid = BitConverter.ToInt32(_bytes, mcnkDataStart + 0x64)
        };
        return h;
    }

    private AlphaMhdr ReadMhdr(int mhdrDataStart)
    {
        return new AlphaMhdr
        {
            OffsInfo = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x00),
            OffsTex  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x04),
            SizeTex  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x08),
            OffsDoo  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x0C),
            SizeDoo  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x10),
            OffsMob  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x14),
            SizeMob  = BitConverter.ToInt32(_bytes, mhdrDataStart + 0x18)
        };
    }

    private AlphaMphd ReadMphd(int mphdDataStart)
    {
        return new AlphaMphd
        {
            NDoodadNames = BitConverter.ToInt32(_bytes, mphdDataStart + 0x00),
            OffsDoodadNames = BitConverter.ToInt32(_bytes, mphdDataStart + 0x04),
            NMapObjNames = BitConverter.ToInt32(_bytes, mphdDataStart + 0x08),
            OffsMapObjNames = BitConverter.ToInt32(_bytes, mphdDataStart + 0x0C)
        };
    }

    private Dictionary<string, (int offset, int size)> ScanTopLevelChunks()
    {
        var dict = new Dictionary<string, (int offset, int size)>(StringComparer.Ordinal);
        for (int i = 0; i + 8 <= _bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(_bytes, i, 4);
            int size = BitConverter.ToInt32(_bytes, i + 4);
            dict[fcc] = (i, size);
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break;
            i = next;
        }
        return dict;
    }
}
