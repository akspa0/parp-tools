using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles.Alpha;

namespace AlphaWdtAnalyzer.Core;

public sealed class WdtAlphaScanner
{
    public string WdtPath { get; }

    public string MapName { get; }
    public List<int> AdtNumbers { get; } = new();
    public List<int> AdtMhdrOffsets { get; } = new();

    public List<string> MdnmFiles { get; } = new();
    public List<string> MonmFiles { get; } = new();

    public WdtAlphaScanner(string wdtPath)
    {
        if (!File.Exists(wdtPath)) throw new FileNotFoundException("WDT not found", wdtPath);
        WdtPath = wdtPath;
        MapName = Path.GetFileNameWithoutExtension(wdtPath);

        var wdt = new WdtAlpha(wdtPath);
        AdtNumbers.AddRange(wdt.GetExistingAdtsNumbers());
        AdtMhdrOffsets.AddRange(wdt.GetAdtOffsetsInMain());
        MdnmFiles.AddRange(wdt.GetMdnmFileNames());
        MonmFiles.AddRange(wdt.GetMonmFileNames());
    }
}
