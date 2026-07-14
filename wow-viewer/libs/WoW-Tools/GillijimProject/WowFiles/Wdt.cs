using System;
using System.IO;
using System.Collections.Generic;

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
    /// [PORT] Write to file: concatenate serialized chunks. Output name: original + "_new" next to input.
    /// </summary>
    public void ToFile()
    {
        var whole = new List<byte>();
        whole.AddRange(_mver.GetWholeChunk());
        whole.AddRange(_mphd.GetWholeChunk());
        whole.AddRange(_main.GetWholeChunk());
        if (!_mwmo.IsEmpty()) whole.AddRange(_mwmo.GetWholeChunk());
        if (!_modf.IsEmpty()) whole.AddRange(_modf.GetWholeChunk());

        var fileName = _wdtName + "_new";
        File.WriteAllBytes(fileName, whole.ToArray());
    }

    /// <summary>
    /// [PORT] Write to a specific directory. File name is basename(input) + "_new".
    /// </summary>
    public void ToFile(string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        var whole = new List<byte>();
        whole.AddRange(_mver.GetWholeChunk());
        whole.AddRange(_mphd.GetWholeChunk());
        whole.AddRange(_main.GetWholeChunk());
        if (!_mwmo.IsEmpty()) whole.AddRange(_mwmo.GetWholeChunk());
        if (!_modf.IsEmpty()) whole.AddRange(_modf.GetWholeChunk());

        var outPath = Path.Combine(outputDir, Path.GetFileName(_wdtName) + "_new");
        File.WriteAllBytes(outPath, whole.ToArray());
    }

    /// <summary>
    /// Write to an exact file path.
    /// </summary>
    public void ToExactFile(string absolutePath)
    {
        var dir = Path.GetDirectoryName(absolutePath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

        var whole = new List<byte>();
        whole.AddRange(_mver.GetWholeChunk());
        whole.AddRange(_mphd.GetWholeChunk());
        whole.AddRange(_main.GetWholeChunk());
        if (!_mwmo.IsEmpty()) whole.AddRange(_mwmo.GetWholeChunk());
        if (!_modf.IsEmpty()) whole.AddRange(_modf.GetWholeChunk());

        File.WriteAllBytes(absolutePath, whole.ToArray());
    }

    public override string ToString() => $"Wdt({_wdtName})";
}
