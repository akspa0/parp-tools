using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles.LichKing;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] Minimal C# port of AdtAlpha (see lib/gillijimproject/wowfiles/alpha/AdtAlpha.{h,cpp})
/// This establishes constructor parity and basic helpers (coords), leaving full parsing for later.
/// </summary>
public class AdtAlpha : WowFiles.WowChunkedFormat
{
    private readonly int _adtNumber;
    private readonly string _adtFileName;
    private readonly int _x;
    private readonly int _y;

    /// <summary>
    /// [PORT] Constructor parity (wdtAlphaName, offsetInFile, adtNum).
    /// Offset not used yet; full parsing is deferred until dependent classes are ported.
    /// </summary>
    public AdtAlpha(string wdtAlphaName, int offsetInFile, int adtNum)
    {
        _adtNumber = adtNum;
        _x = adtNum % 64;
        _y = adtNum / 64;
        _adtFileName = GetAdtFileName(wdtAlphaName, _x, _y);
        // [PORT] Future: parse ADT at offsetInFile as needed when porting Alpha ADT chunks.
    }

    /// <summary>
    /// [PORT] X tile coordinate in the 64x64 grid.
    /// </summary>
    public int GetXCoord() => _x;

    /// <summary>
    /// [PORT] Y tile coordinate in the 64x64 grid.
    /// </summary>
    public int GetYCoord() => _y;

    /// <summary>
    /// [PORT] Convert to LichKing ADT placeholder until full conversion is ported.
    /// </summary>
    public AdtLk ToAdtLk(List<string> mdnmFileNames, List<string> monmFileNames)
    {
        // [PORT] Future: map Alpha ADT data to LK structures using provided file lists.
        return new AdtLk();
    }

    private static string GetAdtFileName(string wdtName, int x, int y)
    {
        // [PORT] Placeholder naming: baseName_x_y.adt next to WDT; refine when porting exact logic.
        var dir = Path.GetDirectoryName(wdtName) ?? string.Empty;
        var baseName = Path.GetFileNameWithoutExtension(wdtName);
        var name = $"{baseName}_{x}_{y}.adt";
        return Path.Combine(dir, name);
    }
}
