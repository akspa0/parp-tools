using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaWdtWriter
{
    public void WriteAlphaWdt(string outFile, byte[] mainFlags)
    {
        if (string.IsNullOrWhiteSpace(outFile)) throw new ArgumentException("outFile required", nameof(outFile));
        if (mainFlags is null) throw new ArgumentNullException(nameof(mainFlags));

        Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? ".");

        var mverData = BitConverter.GetBytes(18); // v18
        var mver = new Chunk("MVER", mverData.Length, mverData);
        var main = new Chunk("MAIN", mainFlags.Length, mainFlags);

        var bytes = new List<byte>();
        bytes.AddRange(mver.GetWholeChunk());
        bytes.AddRange(main.GetWholeChunk());

        File.WriteAllBytes(outFile, bytes.ToArray());
    }
}
