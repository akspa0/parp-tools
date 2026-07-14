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

    public WdtAlpha(string wdtAlphaName)
    {
        _wdtName = wdtAlphaName;
        using var fs = File.OpenRead(wdtAlphaName);

        static int NextOffset(int start, Chunk c)
        {
            int pad = (c.GivenSize & 1) == 1 ? 1 : 0;
            return start + ChunkLettersAndSize + c.GivenSize + pad;
        }

        int offsetInFile = 0;

        // MVER
        _mver = new Chunk(fs, offsetInFile);
        offsetInFile = NextOffset(offsetInFile, _mver);

        // MPHD
        int mphdStartOffset = offsetInFile + ChunkLettersAndSize; // start of MPHD data
        _mphd = new Alpha.MphdAlpha(fs, offsetInFile);
        offsetInFile = NextOffset(offsetInFile, _mphd);

        // MAIN
        _main = new Alpha.MainAlpha(fs, offsetInFile);
        offsetInFile = NextOffset(offsetInFile, _main);

        // MDNM and MONM offsets come from MPHD data
        int mdnmOffset = U.GetIntFromFile(fs, mphdStartOffset + 4);
        _mdnm = new Alpha.Mdnm(fs, mdnmOffset);

        int monmOffset = U.GetIntFromFile(fs, mphdStartOffset + 12);
        _monm = new Alpha.Monm(fs, monmOffset);
        offsetInFile = NextOffset(monmOffset, _monm);

        // Optional MODF after MONM when WMO-based
        if (_mphd.IsWmoBased())
        {
            _modf = new Chunk(fs, offsetInFile);
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

    /// <summary>True if this WDT is a WMO-only map (no terrain tiles, just a global WMO).</summary>
    public bool IsWmoBased => _mphd.IsWmoBased();

    /// <summary>Get the raw MODF chunk data from the WDT header (for WMO-only maps).</summary>
    public byte[] GetWdtModfRaw() => _modf.Data ?? Array.Empty<byte>();
}
