using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.Next.Core.WowFiles;

namespace GillijimProject.Next.Core.WowFiles.Alpha;

/// <summary>
/// Reader for Alpha-era WDT files.
/// Parses MVER, MPHD, MAIN and locates MDNM/MONM using offsets stored in MPHD.
/// </summary>
public static class AlphaWdtReader
{
    public static AlphaWdt Read(string wdtPath)
    {
        if (string.IsNullOrWhiteSpace(wdtPath) || !File.Exists(wdtPath))
            throw new FileNotFoundException("WDT path not found", wdtPath);

        byte[]? mphdData = null;
        long mphdDataStart = 0;
        byte[]? mainData = null;
        var mdnmFiles = Array.Empty<string>();
        var monmFiles = Array.Empty<string>();
        bool wmoBased = false;

        using var fs = File.OpenRead(wdtPath);
        using var br = new BinaryReader(fs, Encoding.ASCII, leaveOpen: true);

        long len = fs.Length;
        while (fs.Position + ChunkIO.HeaderSize <= len)
        {
            if (!ChunkIO.TryReadHeader(fs, br, out string id, out uint size, out long dataStart)) break;

            if (ChunkIO.Matches(id, "MVER"))
            {
                // Read version if present (tolerate absence)
                if (size >= 4 && dataStart + size <= len)
                {
                    _ = br.ReadInt32();
                    fs.Position = dataStart + size;
                }
                else
                {
                    fs.Position = dataStart + size;
                }
                if ((size & 1) == 1) fs.Position++; // pad
                continue;
            }

            if (ChunkIO.Matches(id, "MPHD"))
            {
                mphdDataStart = dataStart;
                if (dataStart + size > len) break;
                mphdData = br.ReadBytes((int)size);
                fs.Position = dataStart + size;
                if ((size & 1) == 1) fs.Position++;
                continue;
            }

            if (ChunkIO.Matches(id, "MAIN"))
            {
                if (dataStart + size > len) break;
                mainData = br.ReadBytes((int)size);
                fs.Position = dataStart + size;
                if ((size & 1) == 1) fs.Position++;
                continue;
            }

            // Skip unknown top-level
            ChunkIO.SkipChunk(fs, size);
        }

        // Parse MPHD flags (WMO-based) and offsets
        if (mphdData is not null && mphdData.Length >= 12)
        {
            // IsWmoBased(): int at offset 8 == 2
            int val = BitConverter.ToInt32(mphdData, 8);
            wmoBased = (val == 2);

            // Offsets within file to MDNM and MONM are stored at (mphdDataStart + 4) and (+12)
            fs.Position = mphdDataStart + 4;
            int mdnmOffset = br.ReadInt32();
            fs.Position = mphdDataStart + 12;
            int monmOffset = br.ReadInt32();

            // MDNM
            if (mdnmOffset > 0 && mdnmOffset + ChunkIO.HeaderSize <= len)
            {
                fs.Position = mdnmOffset;
                if (ChunkIO.TryReadHeader(fs, br, out string id, out uint size, out long dataStart))
                {
                    if (ChunkIO.Matches(id, "MDNM") && dataStart + size <= len)
                    {
                        var data = br.ReadBytes((int)size);
                        var list = new System.Collections.Generic.List<string>(ChunkIO.ParseZeroTerminatedStrings(data));
                        mdnmFiles = list.ToArray();
                        fs.Position = dataStart + size;
                        if ((size & 1) == 1) fs.Position++;
                    }
                }
            }

            // MONM
            if (monmOffset > 0 && monmOffset + ChunkIO.HeaderSize <= len)
            {
                fs.Position = monmOffset;
                if (ChunkIO.TryReadHeader(fs, br, out string id, out uint size, out long dataStart))
                {
                    if (ChunkIO.Matches(id, "MONM") && dataStart + size <= len)
                    {
                        var data = br.ReadBytes((int)size);
                        var list = new System.Collections.Generic.List<string>(ChunkIO.ParseZeroTerminatedStrings(data));
                        monmFiles = list.ToArray();
                        fs.Position = dataStart + size;
                        if ((size & 1) == 1) fs.Position++;
                    }
                }
            }
        }

        // Parse MAIN for ADT offsets
        var adtOffsets = new List<int>(4096);
        if (mainData is not null)
        {
            const int CellSize = 16;
            int current = 0;
            for (int i = 0; i < 4096; i++)
            {
                int off = (current + 4 <= mainData.Length) ? BitConverter.ToInt32(mainData, current) : 0;
                adtOffsets.Add(off);
                current += CellSize;
                if (current >= mainData.Length) break;
            }
            // If MAIN was short, pad to 4096 entries
            while (adtOffsets.Count < 4096) adtOffsets.Add(0);
        }
        else
        {
            // No MAIN — conservative default
            for (int i = 0; i < 4096; i++) adtOffsets.Add(0);
        }

        return new AlphaWdt(
            Path: wdtPath,
            WmoBased: wmoBased,
            AdtOffsets: adtOffsets,
            MdnmFiles: mdnmFiles,
            MonmFiles: monmFiles
        );
    }
}
