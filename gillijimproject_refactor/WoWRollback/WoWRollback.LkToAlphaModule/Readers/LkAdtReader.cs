using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;

namespace WoWRollback.LkToAlphaModule.Readers;

public sealed class LkAdtReader : IAdtReader
{
    public Models.LkAdtData Read(string rootAdtPath, string objAdtPath, string texAdtPath)
    {
        if (string.IsNullOrWhiteSpace(rootAdtPath)) throw new ArgumentException("root ADT path required", nameof(rootAdtPath));
        if (string.IsNullOrWhiteSpace(objAdtPath)) throw new ArgumentException("obj ADT path required", nameof(objAdtPath));
        if (string.IsNullOrWhiteSpace(texAdtPath)) throw new ArgumentException("tex ADT path required", nameof(texAdtPath));
        // TODO: Use WowFiles to parse ADT chunks
        return new Models.LkAdtData();
    }

    public List<McnkLk> ReadRootMcnks(string rootAdtPath)
    {
        if (string.IsNullOrWhiteSpace(rootAdtPath)) throw new ArgumentException("root ADT path required", nameof(rootAdtPath));
        if (!File.Exists(rootAdtPath)) throw new FileNotFoundException("ADT not found", rootAdtPath);

        var bytes = File.ReadAllBytes(rootAdtPath);

        // Find MHDR chunk (on-disk 'RDHM')
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
        if (mhdrOffset < 0) throw new InvalidDataException("MHDR not found in ADT");

        var mhdr = new Mhdr(bytes, mhdrOffset);
        int mhdrStart = mhdrOffset + 8;
        int mcinOff = mhdr.GetOffset(Mhdr.McinOffset);
        if (mcinOff == 0) throw new InvalidDataException("MCIN offset is zero in MHDR");
        var mcin = new Mcin(bytes, mhdrStart + mcinOff);
        var offsets = mcin.GetMcnkOffsets();

        var list = new List<McnkLk>(256);
        for (int i = 0; i < 256; i++)
        {
            int off = (i < offsets.Count) ? offsets[i] : 0;
            if (off > 0)
                list.Add(new McnkLk(bytes, off, 0x80));
            else
                list.Add(McnkLk.CreatePlaceholder());
        }
        return list;
    }
}
