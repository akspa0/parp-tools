using System;
using System.IO;
using System.Text;
using WoWRollback.LkToAlphaModule.Mcal;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Builders;

/// <summary>
/// Builds Lich King MCNK chunks from Alpha intermediate data.
/// </summary>
public static class LkMcnkBuilder
{
    private const int McnkHeaderSize = 0x80;
    private const int ChunkHeaderSize = 8;
    private const int LayerEntrySize = 16;

    public static byte[] BuildFromAlpha(LkMcnkSource source, LkToAlphaOptions options)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        options ??= new LkToAlphaOptions();

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        writer.Write(Encoding.ASCII.GetBytes("KNCM"));
        writer.Write(0); // size placeholder

        Span<byte> header = stackalloc byte[McnkHeaderSize];
        header.Clear();

        BitConverter.TryWriteBytes(header[0x00..], source.Flags);
        BitConverter.TryWriteBytes(header[0x04..], source.IndexX);
        BitConverter.TryWriteBytes(header[0x08..], source.IndexY);
        BitConverter.TryWriteBytes(header[0x0C..], source.Radius);
        BitConverter.TryWriteBytes(header[0x10..], source.LayerCount);
        BitConverter.TryWriteBytes(header[0x14..], source.DoodadRefCount);
        BitConverter.TryWriteBytes(header[0x18..], 0); // will be patched with offsets later
        BitConverter.TryWriteBytes(header[0x38..], source.AreaId);
        BitConverter.TryWriteBytes(header[0x3C..], source.MapObjectRefs);
        BitConverter.TryWriteBytes(header[0x40..], source.HolesLowRes);

        for (int i = 0; i < Math.Min(source.PredictedTextures.Length, 8); i++)
        {
            BitConverter.TryWriteBytes(header[(0x44 + i * 2)..], source.PredictedTextures[i]);
        }

        BitConverter.TryWriteBytes(header[0x54..], (uint)(source.NoEffectDoodadMask & 0xFFFFFFFF));
        BitConverter.TryWriteBytes(header[0x5C..], 0);
        BitConverter.TryWriteBytes(header[0x60..], 0);
        BitConverter.TryWriteBytes(header[0x64..], 0);

        writer.Write(header);

        long dataStart = ms.Position;

        WriteRaw(writer, source.McvtRaw, out int mcvtOffset);
        WriteRaw(writer, source.McnrRaw, out int mcnrOffset);
        
        // Write MCLY with chunk header
        int mclyOffset = (int)writer.BaseStream.Position;
        if (source.MclyRaw.Length > 0)
        {
            writer.Write(Encoding.ASCII.GetBytes("YLCM")); // "MCLY" reversed
            writer.Write(source.MclyRaw.Length);
            writer.Write(source.MclyRaw);
        }
        
        WriteRaw(writer, source.McrfRaw, out int mcrfOffset);
        WriteRaw(writer, source.McshRaw, out int mcshOffset);

        // Prefer passthrough MCAL (raw from Alpha) when available
        byte[] mcalRaw;
        if (source.McalRaw.Length > 0 && source.MclyRaw.Length > 0)
        {
            mcalRaw = source.McalRaw;
        }
        else
        {
            mcalRaw = BuildMcal(source, options, mclyOffset);
        }
        int mcalOffset = (int)writer.BaseStream.Position;
        if (mcalRaw.Length > 0)
        {
            writer.Write(Encoding.ASCII.GetBytes("LACM")); // "MCAL" reversed
            writer.Write(mcalRaw.Length);
            writer.Write(mcalRaw);
        }

        WriteRaw(writer, source.McseRaw, out int mcseOffset);

        long endPos = ms.Position;
        int subChunkSize = (int)(endPos - dataStart);

        ms.Position = 4;
        writer.Write(subChunkSize);

        ms.Position = ChunkHeaderSize;
        writer.Write(header);
        ms.Position = ChunkHeaderSize;

        Span<byte> patch = stackalloc byte[McnkHeaderSize];
        ms.Read(patch);

        BitConverter.TryWriteBytes(patch[0x18..], mcvtOffset);
        BitConverter.TryWriteBytes(patch[0x1C..], mcnrOffset);
        BitConverter.TryWriteBytes(patch[0x20..], mclyOffset);
        BitConverter.TryWriteBytes(patch[0x24..], mcrfOffset);
        BitConverter.TryWriteBytes(patch[0x28..], mcalOffset);
        BitConverter.TryWriteBytes(patch[0x2C..], mcalRaw.Length);
        BitConverter.TryWriteBytes(patch[0x30..], mcshOffset);
        BitConverter.TryWriteBytes(patch[0x34..], source.McshRaw.Length);
        BitConverter.TryWriteBytes(patch[0x5C..], mcseOffset);
        BitConverter.TryWriteBytes(patch[0x60..], 0);

        ms.Position = ChunkHeaderSize;
        writer.Write(patch);

        return ms.ToArray();
    }

    private static void WriteRaw(BinaryWriter writer, byte[] data, out int offset)
    {
        offset = (int)writer.BaseStream.Position;
        if (data.Length > 0)
        {
            writer.Write(data);
        }
    }

    private static byte[] BuildMcal(LkMcnkSource source, LkToAlphaOptions options, int mclyOffset)
    {
        Console.WriteLine($"[BuildMcal] MCNK {source.IndexX},{source.IndexY}: AlphaLayers.Count={source.AlphaLayers.Count}, LayerCount={source.LayerCount}");
        if (source.AlphaLayers.Count == 0)
        {
            Console.WriteLine($"[BuildMcal] MCNK {source.IndexX},{source.IndexY}: No AlphaLayers, returning empty MCAL");
            return Array.Empty<byte>();
        }

        Span<byte> mclySpan = source.MclyRaw.AsSpan();
        if (mclySpan.Length % LayerEntrySize != 0)
        {
            throw new InvalidDataException($"MCLY raw length {mclySpan.Length} is not a multiple of {LayerEntrySize}.");
        }

        using var ms = new MemoryStream();
        foreach (var layer in source.AlphaLayers)
        {
            if (layer.LayerIndex < 0 || layer.LayerIndex >= source.LayerCount)
            {
                throw new InvalidDataException($"Layer index {layer.LayerIndex} out of range (layers={source.LayerCount}).");
            }

            int entryOffset = layer.LayerIndex * LayerEntrySize;
            uint effectiveFlags = layer.OverrideFlags ?? BitConverter.ToUInt32(mclySpan.Slice(entryOffset + 4, 4));
            
            // If ColumnMajorAlpha is already compressed/encoded (from Alpha extraction), use it directly
            // Otherwise encode it (from LK decompressed data)
            byte[] encoded;
            if (layer.ColumnMajorAlpha.Length == 4096)
            {
                // Full 4096 bytes = uncompressed, need to encode
                if (options.ForceCompressedAlpha)
                {
                    effectiveFlags |= 0x200;
                }
                encoded = McalAlphaEncoder.Encode(layer.ColumnMajorAlpha, effectiveFlags, options.AssumeAlphaEdgeFixed);
            }
            else
            {
                // Already compressed/partial data from Alpha, use as-is
                encoded = layer.ColumnMajorAlpha;
            }

            Console.WriteLine($"[BuildMcal]   Layer {layer.LayerIndex}: {encoded.Length} bytes, flags=0x{effectiveFlags:X}");

            BitConverter.TryWriteBytes(mclySpan.Slice(entryOffset + 4, 4), effectiveFlags);
            BitConverter.TryWriteBytes(mclySpan.Slice(entryOffset + 8, 4), (uint)ms.Position);

            ms.Write(encoded, 0, encoded.Length);
        }

        byte[] result = ms.ToArray();
        Console.WriteLine($"[BuildMcal] MCNK {source.IndexX},{source.IndexY}: Built MCAL with {result.Length} bytes total");
        return result;
    }
}
