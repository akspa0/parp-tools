using System;
using System.IO;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Alpha;

namespace WoWRollback.LkToAlphaModule.Writers;

public static class AlphaWdtWriter
{
    // Legacy minimal method preserved for compatibility (writes MVER + MAIN only)
    public static void WriteAlphaWdt(string outFile, byte[] mainFlags)
    {
        if (string.IsNullOrWhiteSpace(outFile)) throw new ArgumentException("outFile required", nameof(outFile));
        if (mainFlags is null) throw new ArgumentNullException(nameof(mainFlags));

        Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? ".");

        var mverData = BitConverter.GetBytes(18); // v18
        var mver = new Chunk("MVER", mverData.Length, mverData);
        var main = new Chunk("MAIN", mainFlags.Length, mainFlags);

        using var fs = File.Create(outFile);
        fs.Write(mver.GetWholeChunk());
        fs.Write(main.GetWholeChunk());
    }

    // Full writer from explicit chunks in canonical Alpha order: MVER, MPHD, MAIN, MDNM, MONM, [MODF]
    public static void Write(Chunk mver, MphdAlpha mphd, MainAlpha main, Mdnm mdnm, Monm monm, Chunk? modf, string outputPath)
    {
        if (mver == null) throw new ArgumentNullException(nameof(mver));
        if (mphd == null) throw new ArgumentNullException(nameof(mphd));
        if (main == null) throw new ArgumentNullException(nameof(main));
        if (mdnm == null) throw new ArgumentNullException(nameof(mdnm));
        if (monm == null) throw new ArgumentNullException(nameof(monm));
        if (string.IsNullOrWhiteSpace(outputPath)) throw new ArgumentException("Output path is required", nameof(outputPath));

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }

        EnsureMonmTrailingEmptyString(mphd, monm);

        using var fs = File.Create(outputPath);
        fs.Write(mver.GetWholeChunk());
        fs.Write(mphd.GetWholeChunk());
        fs.Write(main.GetWholeChunk());
        fs.Write(mdnm.GetWholeChunk());
        fs.Write(monm.GetWholeChunk());
        if (modf != null && !modf.IsEmpty())
        {
            fs.Write(modf.GetWholeChunk());
        }
    }

    // Convenience: write WDT directly from a parsed WdtAlpha source (uses ToWdt to get canonical chunks)
    public static void WriteFromSource(WdtAlpha source, string outputPath)
    {
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (string.IsNullOrWhiteSpace(outputPath)) throw new ArgumentException("Output path is required", nameof(outputPath));

        // WdtAlpha.ToWdt returns a Wdt (LK-style container) with MVER/MPHD/MAIN/MWMO/MODF converted.
        // For Alpha direct-write, we rely on source’s native chunks instead.
        // Here we reconstruct from WdtAlpha by re-reading the raw chunks if needed.
        // Callers should prefer the explicit overload with native Alpha chunks for full control.
        throw new NotSupportedException("Use the explicit Write(mver, mphd, main, mdnm, monm, modf, outputPath) overload for Alpha native write.");
    }

    private static void EnsureMonmTrailingEmptyString(MphdAlpha mphd, Monm monm)
    {
        try
        {
            if (!mphd.IsWmoBased()) return;
            // If Monm doesn’t ensure an extra empty string at the end when WMOs exist,
            // the writer should normalize here. Placeholder: assume caller provided a normalized Monm.
        }
        catch
        {
            // Non-fatal safeguard
        }
    }
}
