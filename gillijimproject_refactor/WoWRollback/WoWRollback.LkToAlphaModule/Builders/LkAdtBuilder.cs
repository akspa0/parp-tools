using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Builders;

public static class LkAdtBuilder
{
    private const int McnkPerTile = 256;

    public static byte[] Build(LkAdtSource source, LkToAlphaOptions options)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        options ??= new LkToAlphaOptions();

        using var ms = new MemoryStream();
        WriteMver(ms);
        WriteMcnkBlocks(ms, source, options);
        return ms.ToArray();
    }

    private static void WriteMver(Stream stream)
    {
        // LK ADTs start with MVER chunk version 18
        Span<byte> header = stackalloc byte[8];
        Encoding.ASCII.GetBytes("REVM").CopyTo(header); // 'MVER' reversed
        BitConverter.TryWriteBytes(header[4..], 4);
        stream.Write(header);
        stream.Write(BitConverter.GetBytes(18));
    }

    private static void WriteMcnkBlocks(Stream stream, LkAdtSource source, LkToAlphaOptions options)
    {
        if (source.Mcnks.Count != McnkPerTile)
        {
            throw new InvalidDataException($"Expected {McnkPerTile} MCNK sources, got {source.Mcnks.Count}.");
        }

        foreach (var mcnk in source.Mcnks)
        {
            var bytes = LkMcnkBuilder.BuildFromAlpha(mcnk, options);
            stream.Write(bytes, 0, bytes.Length);
        }
    }
}
