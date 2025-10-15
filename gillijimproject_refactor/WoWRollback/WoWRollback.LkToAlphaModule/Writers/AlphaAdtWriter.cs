using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using WoWRollback.LkToAlphaModule.Builders;

namespace WoWRollback.LkToAlphaModule.Writers;

public sealed class AlphaAdtWriter : IAdtWriter
{
    public void WriteAlphaAdt(Models.AlphaAdtData data, string outFile)
    {
        if (data is null) throw new ArgumentNullException(nameof(data));
        if (string.IsNullOrWhiteSpace(outFile)) throw new ArgumentException("outFile required", nameof(outFile));
        // TODO: Serialize Alpha ADT (single-file) using WowFiles helpers
    }

    // WDT helper retained here for convenience
    public void WriteAlphaWdtHeaderOnly(string outFile, byte[] mainFlags)
    {
        if (string.IsNullOrWhiteSpace(outFile)) throw new ArgumentException("outFile required", nameof(outFile));
        if (mainFlags is null) throw new ArgumentNullException(nameof(mainFlags));
        Directory.CreateDirectory(Path.GetDirectoryName(outFile) ?? ".");

        var mverData = BitConverter.GetBytes(18);
        var mver = new Chunk("MVER", mverData.Length, mverData);
        var main = new Chunk("MAIN", mainFlags.Length, mainFlags);
        using var fs = File.Create(outFile);
        fs.Write(mver.GetWholeChunk());
        fs.Write(main.GetWholeChunk());
    }

    // Terrain-only Alpha ADT from LK root ADT: writes 256 MCNKs with MCVT only
    public void WriteTerrainOnlyFromLkRoot(string lkRootAdtPath, string outAlphaAdtPath)
    {
        if (string.IsNullOrWhiteSpace(lkRootAdtPath)) throw new ArgumentException("required", nameof(lkRootAdtPath));
        if (string.IsNullOrWhiteSpace(outAlphaAdtPath)) throw new ArgumentException("required", nameof(outAlphaAdtPath));
        if (!File.Exists(lkRootAdtPath)) throw new FileNotFoundException("LK root ADT not found", lkRootAdtPath);

        var bytes = File.ReadAllBytes(lkRootAdtPath);

        // Locate MHDR
        int mhdrOffset = -1;
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == "RDHM") { mhdrOffset = i; break; }
            if (dataStart + size > bytes.Length) break;
            i = next;
        }
        if (mhdrOffset < 0) throw new InvalidDataException("MHDR not found in LK ADT");

        var mhdr = new GillijimProject.WowFiles.Mhdr(bytes, mhdrOffset);
        int mhdrStart = mhdrOffset + 8;
        int mcinOff = mhdr.GetOffset(GillijimProject.WowFiles.Mhdr.McinOffset);
        if (mcinOff == 0) throw new InvalidDataException("MCIN offset is zero in MHDR");
        var mcin = new GillijimProject.WowFiles.Mcin(bytes, mhdrStart + mcinOff);
        var offsets = mcin.GetMcnkOffsets();

        Directory.CreateDirectory(Path.GetDirectoryName(outAlphaAdtPath) ?? ".");
        using var fs = File.Create(outAlphaAdtPath);

        for (int i = 0; i < 256; i++)
        {
            int off = (i < offsets.Count) ? offsets[i] : 0;
            byte[] alphaMcnk;
            if (off > 0)
            {
                alphaMcnk = AlphaMcnkBuilder.BuildFromLk(bytes, off);
            }
            else
            {
                int indexX = i % 16;
                int indexY = i / 16;
                alphaMcnk = AlphaMcnkBuilder.BuildEmpty(indexX, indexY);
            }
            fs.Write(alphaMcnk, 0, alphaMcnk.Length);
        }
    }
}
