using System;
using System.Collections.Generic;
using System.IO;
using U = GillijimProject.Utilities.Utilities;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Alpha;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of WdtAlpha (see lib/gillijimproject/wowfiles/alpha/WdtAlpha.cpp)
/// Parses Alpha WDT to components and exposes helpers to build LK WDT.
/// </summary>
public class WdtAlpha : WowFiles.WowChunkedFormat
{
    private readonly string _wdtName;
    private readonly Chunk _mver;
    private readonly Alpha.MphdAlpha _mphd;
    private readonly Alpha.MainAlpha _main;
    private readonly Alpha.Mdnm _mdnm;
    private readonly Alpha.Monm _monm;
    private readonly Chunk _modf;

    public WdtAlpha(byte[] fileData)
    {
        _wdtName = ""; // Not needed when parsing from memory

        // Find chunks by scanning
        int mverOffset = U.FindChunkOffset(fileData, "MVER", 0);
        if (mverOffset < 0)
            throw new InvalidDataException("MVER chunk not found in WDT file");
        _mver = new Chunk(fileData, mverOffset);
        
        int mphdOffset = U.FindChunkOffset(fileData, "MPHD", 0);
        if (mphdOffset < 0)
            throw new InvalidDataException("MPHD chunk not found in WDT file");
        _mphd = new Alpha.MphdAlpha(fileData, mphdOffset);
        
        int mainOffset = U.FindChunkOffset(fileData, "MAIN", 0);
        if (mainOffset < 0)
            throw new InvalidDataException("MAIN chunk not found in WDT file");
        _main = new Alpha.MainAlpha(fileData, mainOffset);
        
        // MDNM and MONM offsets are absolute, read from MPHD data.
        int mphdDataStart = mphdOffset + ChunkLettersAndSize;
        int mdnmOffset = BitConverter.ToInt32(fileData, mphdDataStart + 4);
        if (mdnmOffset < 0 || mdnmOffset + 8 > fileData.Length)
            throw new InvalidDataException("Invalid MDNM offset in WDT file");
        _mdnm = new Alpha.Mdnm(fileData, mdnmOffset);

        int monmOffset = BitConverter.ToInt32(fileData, mphdDataStart + 12);
        if (monmOffset < 0 || monmOffset + 8 > fileData.Length)
            throw new InvalidDataException("Invalid MONM offset in WDT file");
        _monm = new Alpha.Monm(fileData, monmOffset);

        // Optional MODF chunk follows MAIN when WMO-based.
        int modfOffset = U.FindChunkOffset(fileData, "MODF", 0);
        if (_mphd.IsWmoBased() && modfOffset >= 0)
        {
            _modf = new Chunk(fileData, modfOffset);
        }
        else
        {
            _modf = new Chunk("MODF", 0, Array.Empty<byte>());
        }
    }

    public Wdt ToWdt()
    {
        var cMphd = _mphd.ToMphd();
        var cMain = _main.ToMain();

        var cMwmo = new Chunk("MWMO", 0, Array.Empty<byte>());
        var cModf = new Chunk("MODF", 0, Array.Empty<byte>());

        if (_mphd.IsWmoBased())
        {
            cMwmo = _monm.ToMwmo();
            cModf = _modf;
        }

        return new Wdt(_wdtName, _mver, cMphd, cMain, cMwmo, cModf);
    }

    public List<int> GetExistingAdtsNumbers()
    {
        var adtsOffsets = _main.GetMhdrOffsets();
        var existing = new List<int>();
        for (int i = 0; i < adtsOffsets.Count; i++)
        {
            if (adtsOffsets[i] != 0) existing.Add(i);
        }
        return existing;
    }

    public List<int> GetAdtOffsetsInMain()
    {
        return _main.GetMhdrOffsets();
    }

    public List<string> GetMdnmFileNames()
    {
        return _mdnm.GetFilesNames();
    }

    public List<string> GetMonmFileNames()
    {
        return _monm.GetFilesNames();
    }

    public LichKing.WdtLk ToWdtLk(List<string> pMdnm, List<string> pMonm)
    {
        var cMphd = _mphd.ToMphd();
        var cMain = _main.ToMain();

        var cMwmo = new Chunk("MWMO", 0, Array.Empty<byte>());
        var cModf = new Chunk("MODF", 0, Array.Empty<byte>());

        if (_mphd.IsWmoBased())
        {
            cMwmo = _monm.ToMwmo();
            cModf = _modf;
        }

        return new LichKing.WdtLk(_wdtName, _mver, cMphd, (LichKing.Main)cMain, cMwmo, cModf);
    }

    public Dictionary<int, int> GetAdtOffsets()
    {
        var offsets = _main.GetMhdrOffsets();
        var adtOffsets = new Dictionary<int, int>();
        for (int i = 0; i < offsets.Count; i++)
        {
            if (offsets[i] != 0)
            {
                adtOffsets[i] = offsets[i];
            }
        }
        return adtOffsets;
    }

    public Dictionary<int, string> GetAdtFileNames()
    {
        return _main.GetAdtFileNames();
    }

    public List<string> GetMphdAndMdnm()
    {
        return _mdnm.GetFilesNames();
    }

    public List<string> GetMphdAndMonm()
    {
        return _monm.GetFilesNames();
    }
}
