using System;
using System.IO;
using System.Text;
using System.Runtime.InteropServices;
using GillijimProject.WowFiles;

namespace WoWRollback.LkToAlphaModule.LkInspectors;

public static class LkAdtInspector
{
    private const int McnkHeaderSize = 0x80;

    public static void Dump(string adtPath, int tileIndex, bool verbose)
    {
        if (string.IsNullOrWhiteSpace(adtPath)) throw new ArgumentException("ADT path required", nameof(adtPath));
        if (!File.Exists(adtPath)) throw new FileNotFoundException("ADT not found", adtPath);

        var bytes = File.ReadAllBytes(adtPath);
        int mhdrOffset = FindChunk(bytes, "RDHM");
        if (mhdrOffset < 0) { Console.WriteLine("MHDR chunk not found."); return; }

        var mhdr = new Mhdr(bytes, mhdrOffset);
        int mhdrDataStart = mhdrOffset + 8;
        int mcinOffset = mhdr.GetOffset(Mhdr.McinOffset);
        if (mcinOffset == 0) { Console.WriteLine("MCIN offset is zero in MHDR."); return; }

        var mcin = new Mcin(bytes, mhdrDataStart + mcinOffset);
        var offsets = mcin.GetMcnkOffsets();

        Console.WriteLine($"Inspecting LK ADT: {adtPath}");
        Console.WriteLine($"Tiles parsed: {offsets.Count}");

        if (tileIndex >= 0 && tileIndex < offsets.Count)
            DumpTile(bytes, offsets[tileIndex], tileIndex, verbose);
        else
            for (int i = 0; i < offsets.Count; i++)
                DumpTile(bytes, offsets[i], i, verbose);
    }

    private static void DumpTile(byte[] bytes, int mcNkOffset, int tileIndex, bool verbose)
    {
        if (mcNkOffset <= 0) { Console.WriteLine($"Tile {tileIndex}: placeholder"); return; }
        if (mcNkOffset + 8 + McnkHeaderSize > bytes.Length)
        {
            Console.WriteLine($"Tile {tileIndex}: MCNK header exceeds file bounds (offset=0x{mcNkOffset:X}).");
            return;
        }

        var header = ReadLkMcnkHeader(bytes, mcNkOffset);
        Console.WriteLine($"Tile {tileIndex}: IndexX={header.IndexX} IndexY={header.IndexY} NLayers={header.NLayers} McalSize={header.McalSize}");

        if (header.MclyOffset <= 0) { Console.WriteLine("  No MCLY chunk."); return; }

        int mclyPos = mcNkOffset + header.MclyOffset;
        if (mclyPos + 8 > bytes.Length) { Console.WriteLine("  MCLY offset out of bounds."); return; }

        int mclySize = BitConverter.ToInt32(bytes, mclyPos + 4);
        int mclyDataStart = mclyPos + 8;
        if (mclyDataStart + mclySize > bytes.Length)
            mclySize = Math.Max(0, bytes.Length - mclyDataStart);

        int mclyEntries = mclySize / 16;
        int layerCount = Math.Min(header.NLayers, mclyEntries);

        bool hasMcal = false;
        int mcalDataStart = 0;
        int totalMcalSize = 0;
        if (header.McalOffset > 0)
        {
            int mcalPos = mcNkOffset + header.McalOffset;
            if (mcalPos + 8 <= bytes.Length)
            {
                int chunkSize = BitConverter.ToInt32(bytes, mcalPos + 4);
                mcalDataStart = mcalPos + 8;
                totalMcalSize = Math.Max(0, Math.Min(chunkSize, bytes.Length - mcalDataStart));
                hasMcal = totalMcalSize > 0;
            }
        }

        for (int layer = 0; layer < layerCount; layer++)
        {
            int entryOffset = mclyDataStart + layer * 16;
            uint textureId = BitConverter.ToUInt32(bytes, entryOffset + 0);
            uint flags = BitConverter.ToUInt32(bytes, entryOffset + 4);
            uint rawOffset = BitConverter.ToUInt32(bytes, entryOffset + 8);
            ushort effectId = BitConverter.ToUInt16(bytes, entryOffset + 12);
            bool usesAlpha = (flags & 0x80) != 0;

            Console.WriteLine($"  Layer {layer}: texId={textureId} flags=0x{flags:X} usesAlpha={usesAlpha} rawOffset={rawOffset} effectId={effectId}");

            if (verbose && usesAlpha && hasMcal)
            {
                var (start, end, length) = ComputeAlphaRange(bytes, mclyDataStart, layerCount, layer, totalMcalSize, rawOffset);
                bool isCompressed = (flags & 0x200) != 0;
                bool isBigAlpha = (flags & 0x100) != 0;
                Console.WriteLine($"    alpha: start={start} end={end} len={length} compressed={isCompressed} bigAlpha={isBigAlpha}");
            }
            else if (verbose && usesAlpha && !hasMcal)
            {
                Console.WriteLine("    alpha: MCAL chunk missing");
            }
        }
    }

    private static int FindChunk(byte[] bytes, string reversedFourCC)
    {
        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);
            if (fcc == reversedFourCC) return i;
            if (dataStart + size > bytes.Length || next <= i) break;
            i = next;
        }
        return -1;
    }

    private static McnkHeader ReadLkMcnkHeader(byte[] bytes, int mcNkOffset)
    {
        ReadOnlySpan<byte> headerSpan = new ReadOnlySpan<byte>(bytes, mcNkOffset + 8, McnkHeaderSize);
        return MemoryMarshal.Read<McnkHeader>(headerSpan);
    }

    private static (int start, int end, int length) ComputeAlphaRange(byte[] bytes, int mclyDataStart, int layerCount, int layerIndex, int totalMcalSize, uint rawOffset)
    {
        int start = Math.Min((int)rawOffset, Math.Max(0, totalMcalSize));
        int end = totalMcalSize;
        for (int next = layerIndex + 1; next < layerCount; next++)
        {
            uint nextFlags = BitConverter.ToUInt32(bytes, mclyDataStart + next * 16 + 4);
            if ((nextFlags & 0x80) != 0)
            {
                uint nextOffset = BitConverter.ToUInt32(bytes, mclyDataStart + next * 16 + 8);
                int candidate = (int)Math.Min(nextOffset, (uint)totalMcalSize);
                if (candidate < end) end = candidate;
                break;
            }
        }
        if (end < start) end = start;
        int length = Math.Max(0, end - start);
        return (start, end, length);
    }
}