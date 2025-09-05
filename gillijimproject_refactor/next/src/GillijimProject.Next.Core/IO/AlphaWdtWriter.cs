using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace GillijimProject.Next.Core.IO;

/// <summary>
/// Writer for Alpha-era WDT files embedding raw LK ADT payloads.
/// Produces:
///   MVER (version=18), MPHD (16 bytes), MAIN (4096 x 16B), MDNM (optional), MONM (optional), then appended ADT payloads.
/// MAIN[0..3] holds absolute offset to MHDR within the WDT, MAIN[4..7] holds payload size; remaining bytes are zero.
/// </summary>
public static class AlphaWdtWriter
{
    public sealed record Options(
        bool IncludeEmptyMdnm = true,
        bool IncludeEmptyMonm = true,
        bool WmoBased = false
    );

    public sealed record Result(
        string MapName,
        string LkAdtDir,
        string OutWdtPath,
        int TilesTotal,
        int TilesEmbedded,
        int TilesMissing,
        IReadOnlyList<string> Warnings
    );

    /// <summary>
    /// Generate an Alpha WDT at outWdtPath, embedding raw LK ADT payloads found in lkAdtDir for the given mapName.
    /// Root ADTs must be named like "{mapName}_x_y.adt" (case-insensitive), where x,y in [0..63].
    /// </summary>
    public static Result GenerateFromLkAdts(string mapName, string lkAdtDir, string outWdtPath, Options? options = null)
    {
        if (string.IsNullOrWhiteSpace(mapName)) throw new ArgumentException("mapName is required", nameof(mapName));
        if (string.IsNullOrWhiteSpace(lkAdtDir) || !Directory.Exists(lkAdtDir)) throw new DirectoryNotFoundException(lkAdtDir);
        if (string.IsNullOrWhiteSpace(outWdtPath)) throw new ArgumentException("outWdtPath is required", nameof(outWdtPath));
        options ??= new Options();

        mapName = Path.GetFileNameWithoutExtension(mapName);

        // Scan LK ADTs and collect unique tile indices -> files
        var warnings = new List<string>();
        var tiles = ScanRootAdts(mapName, lkAdtDir, warnings);
        var tilesTotal = 64 * 64;

        // Create directory
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outWdtPath)) ?? ".");

        using var fs = new FileStream(outWdtPath, FileMode.Create, FileAccess.ReadWrite, FileShare.None);

        // 1) MVER (version=18)
        var mverData = new byte[4];
        BitConverter.GetBytes(18).CopyTo(mverData, 0);
        _ = WriteChunk(fs, "MVER", mverData, out _, out _);

        // 2) MPHD (16 bytes)
        var mphdData = new byte[16];
        if (options.WmoBased) BitConverter.GetBytes(2).CopyTo(mphdData, 8); // IsWmoBased() checks data[8]==2
        var mphdHeaderStart = WriteChunk(fs, "MPHD", mphdData, out var mphdDataStart, out _);

        // 3) MAIN (4096 x 16 bytes), pre-zeroed then backfilled
        var mainData = new byte[4096 * 16];
        _ = WriteChunk(fs, "MAIN", mainData, out var mainDataStart, out _);

        // 4) MDNM (optional empty)
        long mdnmHeaderStart = 0;
        if (options.IncludeEmptyMdnm)
        {
            mdnmHeaderStart = WriteChunk(fs, "MDNM", Array.Empty<byte>(), out _, out _);
        }

        // 5) MONM (optional empty)
        long monmHeaderStart = 0;
        if (options.IncludeEmptyMonm)
        {
            monmHeaderStart = WriteChunk(fs, "MONM", Array.Empty<byte>(), out _, out _);
        }

        // Patch MPHD offsets to MDNM (+4) and MONM (+12)
        fs.Seek(mphdDataStart + 4, SeekOrigin.Begin);
        var mdnmOffsetBytes = BitConverter.GetBytes((int)mdnmHeaderStart);
        var monmOffsetBytes = BitConverter.GetBytes((int)monmHeaderStart);
        fs.Write(mdnmOffsetBytes, 0, mdnmOffsetBytes.Length);
        fs.Seek(mphdDataStart + 12, SeekOrigin.Begin);
        fs.Write(monmOffsetBytes, 0, monmOffsetBytes.Length);

        // 6) Append ADT payloads, backfilling MAIN entries (offset,size)
        int tilesEmbedded = 0;
        var ordered = tiles.OrderBy(kv => kv.Key).ToList();
        foreach (var kv in ordered)
        {
            int tileIndex = kv.Key;
            string path = kv.Value;
            byte[] bytes;
            try { bytes = File.ReadAllBytes(path); }
            catch (Exception ex)
            {
                warnings.Add($"Failed to read ADT '{path}': {ex.Message}");
                continue;
            }

            int mhdrWithin = FindMhdrOffset(bytes);
            if (mhdrWithin < 0)
            {
                warnings.Add($"MHDR not found in ADT '{path}', skipping.");
                continue;
            }

            long payloadStart = fs.Seek(0, SeekOrigin.End);
            long mhdrAbsolute = payloadStart + mhdrWithin;

            // Backfill MAIN[tileIndex]
            long cellPos = mainDataStart + (tileIndex * 16);
            fs.Seek(cellPos, SeekOrigin.Begin);
            var offBytes = BitConverter.GetBytes((int)mhdrAbsolute);
            fs.Write(offBytes, 0, offBytes.Length);
            var sizeBytes = BitConverter.GetBytes(bytes.Length);
            fs.Write(sizeBytes, 0, sizeBytes.Length);
            // remaining 8 bytes stay zero

            // Append payload
            fs.Seek(0, SeekOrigin.End);
            fs.Write(bytes, 0, bytes.Length);

            tilesEmbedded++;
        }

        return new Result(
            mapName,
            lkAdtDir,
            outWdtPath,
            TilesTotal: tilesTotal,
            TilesEmbedded: tilesEmbedded,
            TilesMissing: tilesTotal - tilesEmbedded,
            Warnings: warnings
        );
    }

    private static Dictionary<int, string> ScanRootAdts(string mapName, string lkAdtDir, List<string> warnings)
    {
        var dict = new Dictionary<int, string>();
        var files = Directory.EnumerateFiles(lkAdtDir, "*.adt", SearchOption.TopDirectoryOnly);
        foreach (var f in files)
        {
            var name = Path.GetFileNameWithoutExtension(f);
            if (!TryParseTileCoords(name, mapName, out int x, out int y))
                continue;
            if (x < 0 || x > 63 || y < 0 || y > 63) { warnings.Add($"Ignoring out-of-range tile: {name}"); continue; }
            int index = y * 64 + x;
            if (!dict.ContainsKey(index))
                dict[index] = f;
        }
        return dict;
    }

    private static bool TryParseTileCoords(string fileStem, string mapName, out int x, out int y)
    {
        x = y = -1;
        // Accept patterns strictly: {map}_{x}_{y}
        var parts = fileStem.Split('_');
        if (parts.Length != 3) return false;
        if (!parts[0].Equals(mapName, StringComparison.OrdinalIgnoreCase)) return false;
        if (!int.TryParse(parts[1], out x)) return false;
        if (!int.TryParse(parts[2], out y)) return false;
        return true;
    }

    private static int FindMhdrOffset(byte[] adt)
    {
        // Look for either "RDHM" (on-disk reversed) or "MHDR" (safety)
        static int IndexOf(byte[] haystack, byte[] needle)
        {
            var idx = haystack.AsSpan().IndexOf(needle);
            return idx >= 0 ? idx : -1;
        }
        var mhdr = Encoding.ASCII.GetBytes("MHDR");
        var rdhm = Encoding.ASCII.GetBytes("RDHM");
        int a = IndexOf(adt, rdhm);
        if (a >= 0) return a;
        int b = IndexOf(adt, mhdr);
        return b;
    }

    /// <summary>
    /// Writes a chunk: reversed FourCC + size + data + pad (if odd). Returns header start; also outputs data start and total bytes written.
    /// </summary>
    private static long WriteChunk(FileStream fs, string fourCC, byte[] data, out long dataStart, out int totalBytes)
    {
        if (fourCC is null || fourCC.Length != 4) throw new ArgumentException("FourCC must be 4 ASCII characters", nameof(fourCC));
        long headerStart = fs.Position;
        // header: reversed FourCC then 4-byte LE size
        var rev = ReverseFourCC(fourCC);
        var four = Encoding.ASCII.GetBytes(rev);
        fs.Write(four, 0, 4);
        fs.Write(BitConverter.GetBytes(data.Length), 0, 4);
        dataStart = fs.Position;
        if (data.Length > 0)
            fs.Write(data, 0, data.Length);
        // pad if odd size
        if ((data.Length & 1) == 1)
            fs.WriteByte(0);
        totalBytes = (int)(fs.Position - headerStart);
        return headerStart;
    }

    private static string ReverseFourCC(string s)
    {
        return new string(new[] { s[3], s[2], s[1], s[0] });
    }
}
