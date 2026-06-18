using System.Buffers.Binary;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.PM4.Services;

public static class Pm4BinaryWriter
{
    public static byte[] Write(
        Pm4GenerationData data)
    {
        using MemoryStream ms = new();
        using BinaryWriter bw = new(ms);

        WriteMver(bw, data.Version);
        WriteMshd(bw, data.Mshd);
        WriteMspv(bw, data.Mspv);
        WriteMspi(bw, data.Mspi);
        WriteMsvt(bw, data.Msvt);
        WriteMsvi(bw, data.Msvi);
        WriteMsur(bw, data.Msur);
        WriteMscn(bw, data.Mscn);
        WriteMprl(bw, data.Mprl);
        WriteMprr(bw, data.Mprr);
        WriteMslk(bw, data.Mslk);
        WriteMdbh(bw, data.Mdbh);
        WriteMdbi(bw, data.Mdbi);
        WriteMdbf(bw, data.Mdbf);
        WriteMdos(bw, data.Mdos);
        WriteMdsf(bw, data.Mdsf);

        return ms.ToArray();
    }

    private static void WriteChunkHeader(BinaryWriter bw, string signature, int payloadSize)
    {
        byte[] reversed = Encoding.ASCII.GetBytes(signature);
        Array.Reverse(reversed);
        bw.Write(reversed);
        bw.Write(payloadSize);
    }

    private static void WriteMver(BinaryWriter bw, uint version)
    {
        WriteChunkHeader(bw, "MVER", 4);
        bw.Write(version);
    }

    private static void WriteMshd(BinaryWriter bw, Pm4GenerationMshd mshd)
    {
        WriteChunkHeader(bw, "MSHD", 32);
        bw.Write(mshd.Field00);
        bw.Write(mshd.Field04);
        bw.Write(mshd.Field08);
        bw.Write(mshd.Field0C);
        bw.Write(mshd.Field10);
        bw.Write(mshd.Field14);
        bw.Write(mshd.Field18);
        bw.Write(mshd.Field1C);
    }

    private static void WriteMspv(BinaryWriter bw, IReadOnlyList<Vector3> vectors)
    {
        int count = vectors.Count;
        WriteChunkHeader(bw, "MSPV", count * 12);
        for (int i = 0; i < count; i++)
            WriteVector3(bw, vectors[i]);
    }

    private static void WriteMspi(BinaryWriter bw, IReadOnlyList<uint> values)
    {
        int count = values.Count;
        WriteChunkHeader(bw, "MSPI", count * 4);
        for (int i = 0; i < count; i++)
            bw.Write(values[i]);
    }

    private static void WriteMsvt(BinaryWriter bw, IReadOnlyList<Vector3> vectors)
    {
        int count = vectors.Count;
        WriteChunkHeader(bw, "MSVT", count * 12);
        for (int i = 0; i < count; i++)
            WriteVector3(bw, vectors[i]);
    }

    private static void WriteMsvi(BinaryWriter bw, IReadOnlyList<uint> indices)
    {
        int count = indices.Count;
        WriteChunkHeader(bw, "MSVI", count * 4);
        for (int i = 0; i < count; i++)
            bw.Write(indices[i]);
    }

    private static void WriteMsur(BinaryWriter bw, IReadOnlyList<Pm4GenerationMsur> entries)
    {
        int count = entries.Count;
        WriteChunkHeader(bw, "MSUR", count * 32);
        for (int i = 0; i < count; i++)
        {
            Pm4GenerationMsur e = entries[i];
            bw.Write(e.GroupKey);
            bw.Write(e.IndexCount);
            bw.Write(e.AttributeMask);
            bw.Write(e.Padding);
            WriteVector3(bw, e.Normal);
            bw.Write(e.Height);
            bw.Write(e.MsviFirstIndex);
            bw.Write(e.MscnRefIndex);
            bw.Write(e.PackedParams);
        }
    }

    private static void WriteMscn(BinaryWriter bw, IReadOnlyList<Vector3> vectors)
    {
        int count = vectors.Count;
        WriteChunkHeader(bw, "MSCN", count * 12);
        for (int i = 0; i < count; i++)
            WriteVector3(bw, vectors[i]);
    }

    private static void WriteMprl(BinaryWriter bw, IReadOnlyList<Pm4GenerationMprl> entries)
    {
        int count = entries.Count;
        WriteChunkHeader(bw, "MPRL", count * 24);
        for (int i = 0; i < count; i++)
        {
            Pm4GenerationMprl e = entries[i];
            bw.Write(e.Unk00);
            bw.Write(e.Unk02);
            bw.Write(e.Unk04);
            bw.Write(e.Unk06);
            WriteVector3(bw, e.Position);
            bw.Write(e.Unk14);
            bw.Write(e.Unk16);
        }
    }

    private static void WriteMprr(BinaryWriter bw, IReadOnlyList<Pm4GenerationMprr> entries)
    {
        int count = entries.Count;
        WriteChunkHeader(bw, "MPRR", count * 4);
        for (int i = 0; i < count; i++)
        {
            bw.Write(entries[i].Value1);
            bw.Write(entries[i].Value2);
        }
    }

    private static void WriteMslk(BinaryWriter bw, IReadOnlyList<Pm4GenerationMslk> entries)
    {
        int count = entries.Count;
        WriteChunkHeader(bw, "MSLK", count * 20);
        for (int i = 0; i < count; i++)
        {
            Pm4GenerationMslk e = entries[i];
            bw.Write(e.TypeFlags);
            bw.Write(e.Subtype);
            bw.Write(e.Padding);
            bw.Write(e.GroupObjectId);
            WriteInt24(bw, e.MspiFirstIndex);
            bw.Write(e.MspiIndexCount);
            bw.Write(e.LinkId);
            bw.Write(e.RefIndex);
            bw.Write(e.SystemFlag);
        }
    }

    private static void WriteMdbh(BinaryWriter bw, Pm4GenerationMdbh? entry)
    {
        if (entry is null)
            return;
        WriteChunkHeader(bw, "MDBH", 4);
        bw.Write(entry.DestructibleBuildingCount);
    }

    private static void WriteMdbi(BinaryWriter bw, IReadOnlyList<uint> indices)
    {
        if (indices.Count == 0)
            return;
        WriteChunkHeader(bw, "MDBI", indices.Count * 4);
        for (int i = 0; i < indices.Count; i++)
            bw.Write(indices[i]);
    }

    private static void WriteMdbf(BinaryWriter bw, IReadOnlyList<Pm4GenerationMdbf> entries)
    {
        for (int i = 0; i < entries.Count; i++)
        {
            byte[] nameBytes = Encoding.ASCII.GetBytes(entries[i].Filename);
            WriteChunkHeader(bw, "MDBF", nameBytes.Length + 1);
            bw.Write(nameBytes);
            bw.Write((byte)0);
        }
    }

    private static void WriteMdos(BinaryWriter bw, IReadOnlyList<Pm4GenerationMdos> entries)
    {
        if (entries.Count == 0)
            return;
        WriteChunkHeader(bw, "MDOS", entries.Count * 8);
        for (int i = 0; i < entries.Count; i++)
        {
            bw.Write(entries[i].DestructibleBuildingIndex);
            bw.Write(entries[i].DestructionState);
        }
    }

    private static void WriteMdsf(BinaryWriter bw, IReadOnlyList<Pm4GenerationMdsf> entries)
    {
        if (entries.Count == 0)
            return;
        WriteChunkHeader(bw, "MDSF", entries.Count * 8);
        for (int i = 0; i < entries.Count; i++)
        {
            bw.Write(entries[i].MsurIndex);
            bw.Write(entries[i].MdosIndex);
        }
    }

    private static void WriteVector3(BinaryWriter bw, Vector3 v)
    {
        bw.Write(v.X);
        bw.Write(v.Y);
        bw.Write(v.Z);
    }

    private static void WriteInt24(BinaryWriter bw, int value)
    {
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(bytes, value);
        bw.Write(bytes[0..3]);
    }
}

public sealed record Pm4GenerationData(
    uint Version,
    Pm4GenerationMshd Mshd,
    IReadOnlyList<Vector3> Mspv,
    IReadOnlyList<uint> Mspi,
    IReadOnlyList<Vector3> Msvt,
    IReadOnlyList<uint> Msvi,
    IReadOnlyList<Pm4GenerationMsur> Msur,
    IReadOnlyList<Vector3> Mscn,
    IReadOnlyList<Pm4GenerationMprl> Mprl,
    IReadOnlyList<Pm4GenerationMprr> Mprr,
    IReadOnlyList<Pm4GenerationMslk> Mslk,
    Pm4GenerationMdbh? Mdbh,
    IReadOnlyList<uint> Mdbi,
    IReadOnlyList<Pm4GenerationMdbf> Mdbf,
    IReadOnlyList<Pm4GenerationMdos> Mdos,
    IReadOnlyList<Pm4GenerationMdsf> Mdsf);

public sealed record Pm4GenerationMshd(
    uint Field00,
    uint Field04,
    uint Field08,
    uint Field0C,
    uint Field10,
    uint Field14,
    uint Field18,
    uint Field1C);

public sealed record Pm4GenerationMsur(
    byte GroupKey,
    byte IndexCount,
    byte AttributeMask,
    byte Padding,
    Vector3 Normal,
    float Height,
    uint MsviFirstIndex,
    uint MscnRefIndex,
    uint PackedParams);

public sealed record Pm4GenerationMprl(
    ushort Unk00,
    short Unk02,
    ushort Unk04,
    ushort Unk06,
    Vector3 Position,
    short Unk14,
    ushort Unk16);

public sealed record Pm4GenerationMprr(
    ushort Value1,
    ushort Value2);

public sealed record Pm4GenerationMslk(
    byte TypeFlags,
    byte Subtype,
    ushort Padding,
    uint GroupObjectId,
    int MspiFirstIndex,
    byte MspiIndexCount,
    uint LinkId,
    ushort RefIndex,
    ushort SystemFlag);

public sealed record Pm4GenerationMdbh(uint DestructibleBuildingCount);

public sealed record Pm4GenerationMdbf(string Filename);

public sealed record Pm4GenerationMdos(uint DestructibleBuildingIndex, uint DestructionState);

public sealed record Pm4GenerationMdsf(uint MsurIndex, uint MdosIndex);
