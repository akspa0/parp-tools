using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using WoWRollback.Core.Services.PM4;

namespace WoWRollback.Cli.Commands;

public static class CompareAdtCommand
{
    public static int Execute(Dictionary<string, string> opts)
    {
        var minePath = opts.GetValueOrDefault("mine");
        var theirsPath = opts.GetValueOrDefault("theirs");

        if (string.IsNullOrEmpty(minePath) || string.IsNullOrEmpty(theirsPath))
        {
            Console.WriteLine("Usage: compare-adt --mine <path> --theirs <path>");
            return 1;
        }

        if (!File.Exists(minePath)) { Console.WriteLine($"[ERROR] My file not found: {minePath}"); return 1; }
        if (!File.Exists(theirsPath)) { Console.WriteLine($"[ERROR] Their file not found: {theirsPath}"); return 1; }

        Console.WriteLine($"Comparing:");
        Console.WriteLine($"  MINE:   {minePath}");
        Console.WriteLine($"  THEIRS: {theirsPath}");
        Console.WriteLine();

        var mineData = File.ReadAllBytes(minePath);
        var theirsData = File.ReadAllBytes(theirsPath);

        var mineChunks = ParseChunks(mineData);
        var theirsChunks = ParseChunks(theirsData);

        var mineLookup = mineChunks.ToDictionary(c => c.Signature);
        var theirsLookup = theirsChunks.ToDictionary(c => c.Signature);

        // 1. Chunk Presence & Size
        Console.WriteLine("=== Chunk Structure ===");
        var allSigs = mineLookup.Keys.Union(theirsLookup.Keys).OrderBy(k => k).ToList();
        
        foreach (var sig in allSigs)
        {
            var inMine = mineLookup.ContainsKey(sig);
            var inTheirs = theirsLookup.ContainsKey(sig);
            
            if (inMine && inTheirs)
            {
                var m = mineLookup[sig];
                var t = theirsLookup[sig];
                var diff = m.DataSize - t.DataSize;
                var status = diff == 0 ? "MATCH" : $"DIFF ({diff:+0;-0} bytes)";
                Console.WriteLine($"  [{sig}] {status,-15} Mine: {m.DataSize,8} | Theirs: {t.DataSize,8}");
            }
            else if (inMine)
            {
                Console.WriteLine($"  [{sig}] EXTRA MINE      Mine: {mineLookup[sig].DataSize,8} | Theirs:   ABSENT");
            }
            else
            {
                Console.WriteLine($"  [{sig}] MISSING MINE    Mine:   ABSENT | Theirs: {theirsLookup[sig].DataSize,8}");
            }
        }
        Console.WriteLine();

        // 2. MCNK Count
        var mineMcnk = mineChunks.Count(c => c.Signature == "MCNK");
        var theirsMcnk = theirsChunks.Count(c => c.Signature == "MCNK");
        Console.WriteLine($"MCNK Count: Mine={mineMcnk}, Theirs={theirsMcnk} {(mineMcnk == theirsMcnk ? "[OK]" : "[FAIL]")}");
        Console.WriteLine();

        // 3. Texture Comparison (MTEX)
        if (mineLookup.ContainsKey("MTEX") && theirsLookup.ContainsKey("MTEX"))
        {
            CompareMtex(mineData, mineLookup["MTEX"], theirsData, theirsLookup["MTEX"]);
        }

        // 4. WMO Name Comparison (MWMO)
        if (mineLookup.ContainsKey("MWMO") && theirsLookup.ContainsKey("MWMO"))
        {
            CompareMwmo(mineData, mineLookup["MWMO"], theirsData, theirsLookup["MWMO"]);
        }

        // 5. Placement Comparison (MODF)
        if (mineLookup.ContainsKey("MODF") && theirsLookup.ContainsKey("MODF"))
        {
            CompareModf(mineData, mineLookup["MODF"], theirsData, theirsLookup["MODF"]);
        }

        return 0;
    }

    private static void CompareMtex(byte[] mData, ChunkInfo mChunk, byte[] tData, ChunkInfo tChunk)
    {
        Console.WriteLine("=== MTEX (Textures) ===");
        var mTex = ParseStringBlock(mData, mChunk.Offset + 8, mChunk.DataSize); // +8 for header
        var tTex = ParseStringBlock(tData, tChunk.Offset + 8, tChunk.DataSize);

        var mSet = new HashSet<string>(mTex);
        var tSet = new HashSet<string>(tTex);

        var missing = tSet.Except(mSet).ToList();
        var extra = mSet.Except(tSet).ToList();

        Console.WriteLine($"  Count: Mine={mTex.Count}, Theirs={tTex.Count}");
        if (missing.Any())
        {
            Console.WriteLine("  MISSING in Mine:");
            foreach (var t in missing.Take(5)) Console.WriteLine($"    - {t}");
            if (missing.Count > 5) Console.WriteLine($"    ... and {missing.Count - 5} more");
        }
        if (extra.Any())
        {
            Console.WriteLine("  EXTRA in Mine:");
            foreach (var t in extra.Take(5)) Console.WriteLine($"    + {t}");
            if (extra.Count > 5) Console.WriteLine($"    ... and {extra.Count - 5} more");
        }
        if (!missing.Any() && !extra.Any()) Console.WriteLine("  [MATCH] Texture sets are identical.");
        Console.WriteLine();
    }

    private static void CompareMwmo(byte[] mData, ChunkInfo mChunk, byte[] tData, ChunkInfo tChunk)
    {
        Console.WriteLine("=== MWMO (WMO Files) ===");
        var mNames = ParseStringBlock(mData, mChunk.Offset + 8, mChunk.DataSize);
        var tNames = ParseStringBlock(tData, tChunk.Offset + 8, tChunk.DataSize);

        var mSet = new HashSet<string>(mNames);
        var tSet = new HashSet<string>(tNames);

        var missing = tSet.Except(mSet).ToList();
        var extra = mSet.Except(tSet).ToList();

        Console.WriteLine($"  Count: Mine={mNames.Count}, Theirs={tNames.Count}");
        if (missing.Any())
        {
            Console.WriteLine("  MISSING in Mine:");
            foreach (var t in missing.Take(5)) Console.WriteLine($"    - {t}");
            if (missing.Count > 5) Console.WriteLine($"    ... and {missing.Count - 5} more");
        }
        if (extra.Any())
        {
            Console.WriteLine("  EXTRA in Mine:");
            foreach (var t in extra.Take(5)) Console.WriteLine($"    + {t}");
            if (extra.Count > 5) Console.WriteLine($"    ... and {extra.Count - 5} more");
        }
        if (!missing.Any() && !extra.Any()) Console.WriteLine("  [MATCH] WMO sets are identical.");
        Console.WriteLine();
    }

    private static void CompareModf(byte[] mData, ChunkInfo mChunk, byte[] tData, ChunkInfo tChunk)
    {
        Console.WriteLine("=== MODF (WMO Placements) ===");
        int mCount = mChunk.DataSize / 64;
        int tCount = tChunk.DataSize / 64;
        Console.WriteLine($"  Count: Mine={mCount}, Theirs={tCount}");
        // Deep comparison requires parsing unique IDs, position, etc.
        // For now, simple count check is a good indicator.
        if (mCount == 0 && tCount > 0) Console.WriteLine("  [FAIL] Mine has NO placements!");
        else if (mCount > 0 && tCount == 0) Console.WriteLine("  [WARN] Mine has placements but theirs does not?");
        else if (mCount != tCount) Console.WriteLine($"  [WARN] Count mismatch: {mCount - tCount:+0;-0}");
        else Console.WriteLine("  [OK] Count matches.");
        Console.WriteLine();
    }

    private static List<string> ParseStringBlock(byte[] data, int start, int length)
    {
        var list = new List<string>();
        if (length <= 0) return list;

        int end = start + length;
        int current = start;
        while (current < end)
        {
            int zero = Array.IndexOf(data, (byte)0, current);
            if (zero < 0 || zero >= end) break;
            
            if (zero > current)
            {
                var s = Encoding.ASCII.GetString(data, current, zero - current);
                list.Add(s);
            }
            current = zero + 1;
        }
        return list;
    }

    private record ChunkInfo(string Signature, int Offset, int DataSize, int TotalSize);

    private static List<ChunkInfo> ParseChunks(byte[] data)
    {
        var chunks = new List<ChunkInfo>();
        int pos = 0;
        while (pos < data.Length - 8)
        {
            // Read signature as string directly
            var sig = Encoding.ASCII.GetString(data, pos, 4);
            // Normalize reversed signatures
            var rSig = new string(sig.Reverse().ToArray());
            var normSig = (rSig == "MVER" || rSig == "MHDR" || rSig == "MCNK" || rSig == "MTEX" || 
                           rSig == "MWMO" || rSig == "MODF" || rSig == "MDDF" || rSig == "MMDX" || rSig == "MMID" || rSig == "MWID" || rSig == "MH2O") 
                           ? rSig : sig;

            var size = BitConverter.ToInt32(data, pos + 4);
            if (size < 0 || pos + 8 + size > data.Length) break;

            chunks.Add(new ChunkInfo(normSig, pos, size, 8 + size));
            
            // For MCNK, we just skip over it
            pos += 8 + size;
        }
        return chunks;
    }
}
