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
        var data = new Models.LkAdtData
        {
            HasMh2o = HasMh2oChunk(rootAdtPath)
        };

        data.MmdxNames.AddRange(ReadM2Names(rootAdtPath));
        data.MwmoNames.AddRange(ReadWmoNames(rootAdtPath));
        data.MmidOffsets.AddRange(ReadMmidOffsets(rootAdtPath));
        data.MwidOffsets.AddRange(ReadMwidOffsets(rootAdtPath));
        data.MddfPlacements.AddRange(ReadMddfEntries(rootAdtPath));
        data.ModfPlacements.AddRange(ReadModfEntries(rootAdtPath));

        return data;
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

    /// <summary>
    /// Read WMO names (MWMO chunk) from a LK root ADT file
    /// </summary>
    public List<string> ReadWmoNames(string rootAdtPath)
    {
        var result = new List<string>();
        if (!File.Exists(rootAdtPath)) return result;

        var bytes = File.ReadAllBytes(rootAdtPath);
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            string fourCC = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;

            if (fourCC == "OMWM") // MWMO reversed
            {
                // Parse null-terminated strings
                int pos = dataStart;
                int end = dataStart + size;
                while (pos < end)
                {
                    int nullPos = Array.IndexOf(bytes, (byte)0, pos, end - pos);
                    if (nullPos == -1) nullPos = end;
                    
                    int len = nullPos - pos;
                    if (len > 0)
                    {
                        string name = Encoding.UTF8.GetString(bytes, pos, len);
                        if (!string.IsNullOrWhiteSpace(name))
                        {
                            result.Add(name);
                        }
                    }
                    
                    pos = nullPos + 1;
                }
                break;
            }

            i = next;
        }

        return result;
    }

    private static bool HasMh2oChunk(string rootAdtPath)
    {
        if (!File.Exists(rootAdtPath)) return false;
        var bytes = File.ReadAllBytes(rootAdtPath);
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == "O2HM") return true;
            if (size < 0 || next <= i || next > bytes.Length) break;
            i = next;
        }
        return false;
    }

    private static IEnumerable<int> ReadMmidOffsets(string rootAdtPath)
        => ReadIntTableChunk(rootAdtPath, "DIMM");

    private static IEnumerable<int> ReadMwidOffsets(string rootAdtPath)
        => ReadIntTableChunk(rootAdtPath, "DIWM");

    private static IEnumerable<int> ReadIntTableChunk(string rootAdtPath, string fourCC)
    {
        var list = new List<int>();
        if (!File.Exists(rootAdtPath)) return list;

        var bytes = File.ReadAllBytes(rootAdtPath);
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (size < 0 || next <= i || next > bytes.Length) break;

            if (fcc == fourCC)
            {
                for (int offset = 0; offset + 4 <= size; offset += 4)
                {
                    list.Add(BitConverter.ToInt32(bytes, dataStart + offset));
                }
                break;
            }

            i = next;
        }

        return list;
    }

    private static IEnumerable<Models.LkMddfPlacement> ReadMddfEntries(string rootAdtPath)
        => ReadPlacementEntries(rootAdtPath, "FDDM", 36, ParseMddfPlacement);

    private static IEnumerable<Models.LkModfPlacement> ReadModfEntries(string rootAdtPath)
        => ReadPlacementEntries(rootAdtPath, "FDOM", 64, ParseModfPlacement);

    private static IEnumerable<T> ReadPlacementEntries<T>(string rootAdtPath, string fourCC, int entrySize, System.Func<byte[], int, T> parser)
    {
        var list = new List<T>();
        if (!File.Exists(rootAdtPath)) return list;

        var bytes = File.ReadAllBytes(rootAdtPath);
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (size < 0 || next <= i || next > bytes.Length) break;

            if (fcc == fourCC)
            {
                for (int offset = 0; offset + entrySize <= size; offset += entrySize)
                {
                    list.Add(parser(bytes, dataStart + offset));
                }
                break;
            }

            i = next;
        }

        return list;
    }

    private static Models.LkMddfPlacement ParseMddfPlacement(byte[] bytes, int offset)
    {
        int nameIndex = BitConverter.ToInt32(bytes, offset + 0);
        int uniqueId = BitConverter.ToInt32(bytes, offset + 4);
        float posX = BitConverter.ToSingle(bytes, offset + 8);
        float posY = BitConverter.ToSingle(bytes, offset + 12);
        float posZ = BitConverter.ToSingle(bytes, offset + 16);
        float rotX = BitConverter.ToSingle(bytes, offset + 20);
        float rotY = BitConverter.ToSingle(bytes, offset + 24);
        float rotZ = BitConverter.ToSingle(bytes, offset + 28);
        ushort scaleRaw = BitConverter.ToUInt16(bytes, offset + 32);
        ushort flags = BitConverter.ToUInt16(bytes, offset + 34);
        float scale = scaleRaw / 1024.0f;
        return new Models.LkMddfPlacement(nameIndex, uniqueId, posX, posY, posZ, rotX, rotY, rotZ, scale, flags);
    }

    private static Models.LkModfPlacement ParseModfPlacement(byte[] bytes, int offset)
    {
        int nameIndex = BitConverter.ToInt32(bytes, offset + 0);
        int uniqueId = BitConverter.ToInt32(bytes, offset + 4);
        float posX = BitConverter.ToSingle(bytes, offset + 8);
        float posY = BitConverter.ToSingle(bytes, offset + 12);
        float posZ = BitConverter.ToSingle(bytes, offset + 16);
        float rotX = BitConverter.ToSingle(bytes, offset + 20);
        float rotY = BitConverter.ToSingle(bytes, offset + 24);
        float rotZ = BitConverter.ToSingle(bytes, offset + 28);
        float extentsX = BitConverter.ToSingle(bytes, offset + 32);
        float extentsY = BitConverter.ToSingle(bytes, offset + 36);
        float extentsZ = BitConverter.ToSingle(bytes, offset + 40);
        ushort flags = BitConverter.ToUInt16(bytes, offset + 44);
        ushort doodadSet = BitConverter.ToUInt16(bytes, offset + 46);
        ushort nameSet = BitConverter.ToUInt16(bytes, offset + 48);
        ushort scale = BitConverter.ToUInt16(bytes, offset + 50);
        return new Models.LkModfPlacement(nameIndex, uniqueId, posX, posY, posZ, rotX, rotY, rotZ,
            extentsX, extentsY, extentsZ, flags, doodadSet, nameSet, scale);
    }

    /// <summary>
    /// Read M2 names (MMDX chunk) from a LK root ADT file
    /// </summary>
    public List<string> ReadM2Names(string rootAdtPath)
    {
        var result = new List<string>();
        if (!File.Exists(rootAdtPath)) return result;

        var bytes = File.ReadAllBytes(rootAdtPath);
        int i = 0;
        while (i + 8 <= bytes.Length)
        {
            string fourCC = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (dataStart + size > bytes.Length) break;

            if (fourCC == "XDMM") // MMDX reversed
            {
                // Parse null-terminated strings
                int pos = dataStart;
                int end = dataStart + size;
                while (pos < end)
                {
                    int nullPos = Array.IndexOf(bytes, (byte)0, pos, end - pos);
                    if (nullPos == -1) nullPos = end;
                    
                    int len = nullPos - pos;
                    if (len > 0)
                    {
                        string name = Encoding.UTF8.GetString(bytes, pos, len);
                        if (!string.IsNullOrWhiteSpace(name))
                        {
                            result.Add(name);
                        }
                    }
                    
                    pos = nullPos + 1;
                }
                break;
            }

            i = next;
        }

        return result;
    }
}
