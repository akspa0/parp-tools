using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using GillijimProject.WowFiles;
using WoWRollback.LkToAlphaModule.Liquids;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class LkAdtBuilder
{
    private const int McnkPerTile = 256;
    private const int MhdrSize = 64;
    private const int MhdrRelativeStart = 0x14;
    private const int McinSize = 4096;

    public static byte[] Build(LkAdtSource source, LkToAlphaOptions options)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        options ??= new LkToAlphaOptions();

        ValidateSource(source);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        WriteChunk(writer, "MVER", BitConverter.GetBytes(18));

        long mhdrPosition = writer.BaseStream.Position;
        WriteChunk(writer, "RDHM", new byte[MhdrSize]);

        long mcinPosition = writer.BaseStream.Position;
        WriteChunk(writer, "NICM", new byte[McinSize]);

        long mtexPosition = writer.BaseStream.Position;
        WriteChunk(writer, "XTEM", BuildMtex(source));

        long mmdxPosition = writer.BaseStream.Position;
        WriteChunk(writer, "XDMM", BuildStringTable(source.MmdxFilenames));

        long mmidPosition = writer.BaseStream.Position;
        WriteChunk(writer, "DIMM", BuildOffsets(source.MmidOffsets));

        long mwmoPosition = writer.BaseStream.Position;
        WriteChunk(writer, "OMWM", BuildStringTable(source.MwmoFilenames));

        long mwidPosition = writer.BaseStream.Position;
        WriteChunk(writer, "DIWM", BuildOffsets(source.MwidOffsets));

        long mddfPosition = writer.BaseStream.Position;
        WriteChunk(writer, "FDDM", BuildMddf(source.MddfPlacements));

        long modfPosition = writer.BaseStream.Position;
        WriteChunk(writer, "FDOM", BuildModf(source.ModfPlacements));

        long mh2oPosition = writer.BaseStream.Position;
        byte[] mh2o = BuildMh2o(source.Mh2oByChunk);
        if (mh2o.Length > 0)
        {
            WriteChunk(writer, "O2HM", mh2o);
        }

        byte[] mfbo = source.MfboRaw;
        long mfboPosition = writer.BaseStream.Position;
        if (mfbo.Length > 0)
        {
            WriteChunk(writer, "OBFM", mfbo);
        }

        byte[] mtxf = source.MtxfRaw;
        long mtxfPosition = writer.BaseStream.Position;
        if (mtxf.Length > 0)
        {
            WriteChunk(writer, "FXTM", mtxf);
        }

        var mcnkOffsets = new int[McnkPerTile];
        for (int i = 0; i < source.Mcnks.Count; i++)
        {
            mcnkOffsets[i] = (int)writer.BaseStream.Position;
            var chunkBytes = LkMcnkBuilder.BuildFromAlpha(source.Mcnks[i], options);
            writer.Write(chunkBytes);
        }

        writer.BaseStream.Position = mhdrPosition + 8;
        WriteMhdr(writer, source, mcinPosition, mtexPosition, mmdxPosition, mmidPosition, mwmoPosition,
            mwidPosition, mddfPosition, modfPosition, mh2o.Length > 0 ? mh2oPosition : 0,
            mfbo.Length > 0 ? mfboPosition : 0, mtxf.Length > 0 ? mtxfPosition : 0);

        writer.BaseStream.Position = mcinPosition + 8;
        WriteMcin(writer, mcnkOffsets);

        writer.BaseStream.Position = writer.BaseStream.Length;
        return ms.ToArray();
    }

    private static void ValidateSource(LkAdtSource source)
    {
        if (source.Mcnks.Count != McnkPerTile)
        {
            throw new InvalidDataException($"Expected {McnkPerTile} MCNK sources, got {source.Mcnks.Count}.");
        }
    }

    private static void WriteChunk(BinaryWriter writer, string reversedFourCc, byte[] payload)
    {
        writer.Write(Encoding.ASCII.GetBytes(reversedFourCc));
        writer.Write(payload.Length);
        writer.Write(payload);
    }

    private static void WriteMhdr(BinaryWriter writer, LkAdtSource source,
        long mcinPos, long mtexPos, long mmdxPos, long mmidPos, long mwmoPos, long mwidPos,
        long mddfPos, long modfPos, long mh2oPos, long mfboPos, long mtxfPos)
    {
        Span<byte> buffer = stackalloc byte[MhdrSize];
        buffer.Clear();

        BitConverter.TryWriteBytes(buffer[0x00..], 0u);
        BitConverter.TryWriteBytes(buffer[0x04..], 0u);
        BitConverter.TryWriteBytes(buffer[0x08..], 0u);
        BitConverter.TryWriteBytes(buffer[0x0C..], 0u);
        BitConverter.TryWriteBytes(buffer[0x10..], 0u);
        BitConverter.TryWriteBytes(buffer[0x14..], RelativeOffset(mcinPos));
        BitConverter.TryWriteBytes(buffer[0x18..], RelativeOffset(mtexPos));
        BitConverter.TryWriteBytes(buffer[0x1C..], RelativeOffset(mmdxPos));
        BitConverter.TryWriteBytes(buffer[0x20..], RelativeOffset(mmidPos));
        BitConverter.TryWriteBytes(buffer[0x24..], RelativeOffset(mwmoPos));
        BitConverter.TryWriteBytes(buffer[0x28..], RelativeOffset(mwidPos));
        BitConverter.TryWriteBytes(buffer[0x2C..], RelativeOffset(mddfPos));
        BitConverter.TryWriteBytes(buffer[0x30..], RelativeOffset(modfPos));
        BitConverter.TryWriteBytes(buffer[0x34..], mh2oPos == 0 ? 0 : RelativeOffset(mh2oPos));
        BitConverter.TryWriteBytes(buffer[0x38..], mfboPos == 0 ? 0 : RelativeOffset(mfboPos));
        BitConverter.TryWriteBytes(buffer[0x3C..], mtxfPos == 0 ? 0 : RelativeOffset(mtxfPos));

        writer.Write(buffer);

        static int RelativeOffset(long position)
        {
            return position == 0 ? 0 : (int)(position - MhdrRelativeStart);
        }
    }

    private static void WriteMcin(BinaryWriter writer, int[] offsets)
    {
        Span<byte> buffer = stackalloc byte[McinSize];
        buffer.Clear();

        for (int i = 0; i < offsets.Length; i++)
        {
            int baseIndex = i * 16;
            BitConverter.TryWriteBytes(buffer[baseIndex..], offsets[i]);
            BitConverter.TryWriteBytes(buffer[(baseIndex + 4)..], 0);
            BitConverter.TryWriteBytes(buffer[(baseIndex + 8)..], 0);
            BitConverter.TryWriteBytes(buffer[(baseIndex + 12)..], 0);
        }

        writer.Write(buffer);
    }

    private static byte[] BuildMtex(LkAdtSource source)
    {
        var merged = source.Mcnks.SelectMany(m => m.PredictedTextures)
            .Where(tex => tex != 0)
            .Distinct()
            .SelectMany(BitConverter.GetBytes)
            .ToArray();
        return merged;
    }

    private static byte[] BuildStringTable(IEnumerable<string> strings)
    {
        using var ms = new MemoryStream();
        foreach (var s in strings)
        {
            if (string.IsNullOrEmpty(s)) continue;
            var bytes = Encoding.UTF8.GetBytes(s);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0);
        }
        return ms.ToArray();
    }

    private static byte[] BuildOffsets(IEnumerable<int> offsets)
    {
        using var ms = new MemoryStream();
        foreach (int offset in offsets)
        {
            ms.Write(BitConverter.GetBytes(offset));
        }
        return ms.ToArray();
    }

    private static byte[] BuildMddf(IEnumerable<LkMddfPlacement> placements)
    {
        const int entrySize = 36;
        using var ms = new MemoryStream();
        foreach (var p in placements)
        {
            ms.Write(BitConverter.GetBytes(p.NameIndex));
            ms.Write(BitConverter.GetBytes(p.UniqueId));
            ms.Write(BitConverter.GetBytes(p.PositionX));
            ms.Write(BitConverter.GetBytes(p.PositionY));
            ms.Write(BitConverter.GetBytes(p.PositionZ));
            ms.Write(BitConverter.GetBytes(p.RotationX));
            ms.Write(BitConverter.GetBytes(p.RotationY));
            ms.Write(BitConverter.GetBytes(p.RotationZ));
            ushort scale = (ushort)(p.Scale <= 0 ? 1024 : p.Scale * 1024.0f);
            ms.Write(BitConverter.GetBytes(scale));
            ms.Write(BitConverter.GetBytes(p.Flags));
        }

        return ms.ToArray();
    }

    private static byte[] BuildModf(IEnumerable<LkModfPlacement> placements)
    {
        using var ms = new MemoryStream();
        foreach (var p in placements)
        {
            ms.Write(BitConverter.GetBytes(p.NameIndex));
            ms.Write(BitConverter.GetBytes(p.UniqueId));
            ms.Write(BitConverter.GetBytes(p.PositionX));
            ms.Write(BitConverter.GetBytes(p.PositionY));
            ms.Write(BitConverter.GetBytes(p.PositionZ));
            ms.Write(BitConverter.GetBytes(p.RotationX));
            ms.Write(BitConverter.GetBytes(p.RotationY));
            ms.Write(BitConverter.GetBytes(p.RotationZ));
            ms.Write(BitConverter.GetBytes(p.ExtentsX));
            ms.Write(BitConverter.GetBytes(p.ExtentsY));
            ms.Write(BitConverter.GetBytes(p.ExtentsZ));
            ms.Write(BitConverter.GetBytes(p.Flags));
            ms.Write(BitConverter.GetBytes(p.DoodadSet));
            ms.Write(BitConverter.GetBytes(p.NameSet));
            ms.Write(BitConverter.GetBytes(p.Scale));
        }

        return ms.ToArray();
    }

    private static byte[] BuildMh2o(Mh2oChunk?[] chunks) => Mh2oSerializer.Build(chunks);
}
