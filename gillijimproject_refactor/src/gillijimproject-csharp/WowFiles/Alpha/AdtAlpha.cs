using System;
using System.Collections.Generic;
using System.IO;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] Skeleton of AdtAlpha (see lib/gillijimproject/wowfiles/alpha/AdtAlpha.h)
/// Will be filled after core primitives and dependent types are ported.
/// </summary>
public class AdtAlpha : WowFiles.WowChunkedFormat
{
    private int _adtNumber;
    private string _adtFileName = string.Empty;

    /// <summary>
    /// [PORT] Constructor parity (wdtAlphaName, offsetInFile, adtNum).
    /// Currently throws until parsing is implemented.
    /// </summary>
    public AdtAlpha(string wdtAlphaName, int offsetInFile, int adtNum)
    {
        _adtNumber = adtNum;
        _adtFileName = GetAdtFileName(wdtAlphaName);
        throw new NotImplementedException("[PORT] AdtAlpha parsing not implemented yet.");
    }

    public int GetXCoord() => throw new NotImplementedException();

    public int GetYCoord() => throw new NotImplementedException();

    public LichKing.AdtLk ToAdtLk(List<string> mdnmFileNames, List<string> monmFileNames)
        => throw new NotImplementedException();

    private string GetAdtFileName(string wdtName)
    {
        // [PORT] Placeholder. Exact naming logic to be ported.
        return Path.ChangeExtension(wdtName, ".adt");
    }
}
