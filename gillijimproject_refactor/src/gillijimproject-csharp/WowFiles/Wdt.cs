using System;
using System.IO;

namespace GillijimProject.WowFiles;

/// <summary>
/// [PORT] Skeleton of Wdt (see lib/gillijimproject/wowfiles/Wdt.h)
/// </summary>
public class Wdt : WowChunkedFormat
{
    private readonly string _wdtName;
    private readonly Chunk _mver;
    private readonly Mphd _mphd;
    private readonly Chunk _main;
    private readonly Chunk _mwmo;
    private readonly Chunk _modf;

    /// <summary>
    /// [PORT] Construct from raw file (parsing TBD).
    /// </summary>
    public Wdt(byte[] wdtFile, string wdtFileName)
    {
        _wdtName = wdtFileName;
        // [PORT] Placeholder: real parsing to be implemented.
        _mver = new Chunk("MVER", 0, Array.Empty<byte>());
        _mphd = new Mphd();
        _main = new Chunk("MAIN", 0, Array.Empty<byte>());
        _mwmo = new Chunk("MWMO", 0, Array.Empty<byte>());
        _modf = new Chunk("MODF", 0, Array.Empty<byte>());
    }

    /// <summary>
    /// [PORT] Construct from parts.
    /// </summary>
    public Wdt(string name, Chunk cMver, Mphd cMphd, Chunk cMain, Chunk cMwmo, Chunk cModf)
    {
        _wdtName = name;
        _mver = cMver;
        _mphd = cMphd;
        _main = cMain;
        _mwmo = cMwmo;
        _modf = cModf;
    }

    /// <summary>
    /// [PORT] Write to file (not implemented yet).
    /// </summary>
    public void ToFile()
    {
        throw new NotImplementedException("[PORT] Wdt.ToFile not implemented.");
    }

    public override string ToString() => $"Wdt({_wdtName})";
}
