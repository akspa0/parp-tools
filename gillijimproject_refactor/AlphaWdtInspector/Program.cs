using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Security.Cryptography;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length == 0)
        {
            PrintHelp();
            return 1;
        }

        var cmd = args[0].ToLowerInvariant();
        var argv = args.Length > 1 ? args[1..] : Array.Empty<string>();

        return cmd switch
        {
            "export-lk" => ExportLk(argv),
            "pack-alpha" => PackAlpha(argv),
            "pack-alpha-mono" => PackAlphaMono(argv),
            "mtex-audit" => MtexAudit(argv),
            "tile-diff" => TileDiff(argv),
            "dump-adt" => DumpAdt(argv),
            "list-placements" => ListPlacements(argv),
            "dump-mcnk" => DumpMcnk(argv),
            "mcnk-diff" => McnkDiff(argv),
            "placements-sweep-diff" => PlacementsSweepDiff(argv),
            "wdt-sanity" => WdtSanity(argv),
            _ => Unknown(cmd)
        };
    }

    private static int Unknown(string cmd)
    {
        Console.WriteLine($"Unknown command: {cmd}");
        PrintHelp();
        return 2;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("AlphaWdtInspector - Commands:");
        Console.WriteLine("  export-lk        --alpha-wdt <path> [--out <dir>]");
        Console.WriteLine("  pack-alpha       --out <alpha.wdt> [--lk-root <dir>] [--no-coord-xform] [--dest-rebucket] [--emit-mclq] [--scan-mcse]");
        Console.WriteLine("  pack-alpha-mono  --lk-root <dir> --map <Map> [--out <dir>]");
        Console.WriteLine("  mtex-audit       --alpha-wdt <path> [--out <dir>] [--limit <N>]");
        Console.WriteLine("  list-placements  --file <wdt|adt|dir> [--out <dir>]");
        Console.WriteLine("  dump-mcnk        --file <adt> [--out <dir>]");
        Console.WriteLine("  mcnk-diff        --a <adtA> --b <adtB> [--out <dir>]");
        Console.WriteLine("  tile-diff        --original <alpha.wdt> --generated <alpha.wdt> [--tile <X_Y>] [--out <dir>]");
        Console.WriteLine("  dump-adt         --file <adt|wdt|path> [--out <dir>]");
        Console.WriteLine("  placements-sweep-diff --alpha-wdt <path> --lk-dir <dir> [--hex] [--limit <N>] [--out <dir>]");
        Console.WriteLine("  wdt-sanity       --alpha-wdt <path> [--out <dir>] [--limit <N>]");
    }

    private static int ExportLk(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--alpha-wdt", out var alphaWdt) || string.IsNullOrWhiteSpace(alphaWdt))
        {
            Console.WriteLine("Missing required --alpha-wdt <path>");
            return 2;
        }
        if (!File.Exists(alphaWdt))
        {
            Console.WriteLine($"File not found: {alphaWdt}");
            return 3;
        }
        var outDir = GetOutDir(p);
        Console.WriteLine($"[export-lk] alpha-wdt={alphaWdt}\nout={outDir}");

        var mapName = Path.GetFileNameWithoutExtension(alphaWdt);
        if (string.IsNullOrWhiteSpace(mapName))
        {
            Console.WriteLine("Unable to derive map name from --alpha-wdt filename");
            return 4;
        }

        var destDir = Path.Combine(outDir, "lk_out", mapName);
        Directory.CreateDirectory(destDir);

        // Read Alpha WDT bytes and enumerate tiles directly from WDT
        byte[] wdtBytes;
        try { wdtBytes = File.ReadAllBytes(alphaWdt); }
        catch (Exception ex) { Console.WriteLine($"Failed to read WDT: {ex.Message}"); return 6; }

        var tiles = ParseAlphaWdtTiles(wdtBytes);
        var list = new List<string>();
        foreach (var t in tiles)
        {
            if (t.firstMcnk <= 0) continue;
            var x = t.index % 64;
            var y = t.index / 64;
            var name = $"{mapName}_{x:D2}_{y:D2}.adt";
            var dst = Path.Combine(destDir, name);
            try
            {
                var built = BuildLkAdtFromAlphaWdtTile(wdtBytes, t.mhdrAbs);
                if (built.Length > 0)
                {
                    File.WriteAllBytes(dst, built);
                    list.Add(name);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Tile {name}: build failed: {ex.Message}");
            }
        }

        var csv = new List<string> { "filename" };
        csv.AddRange(list);
        var csvPath = Path.Combine(destDir, "export_lk_list.csv");
        try
        {
            File.WriteAllLines(csvPath, csv);
            Console.WriteLine($"Built {list.Count} LK-like ADTs → {destDir}");
            Console.WriteLine($"Wrote {csvPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write list CSV: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static int PackAlphaMono(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--lk-root", out var lkRoot) || string.IsNullOrWhiteSpace(lkRoot))
        {
            Console.WriteLine("Missing required --lk-root <dir>");
            return 2;
        }
        if (!Directory.Exists(lkRoot)) { Console.WriteLine($"Directory not found: {lkRoot}"); return 3; }
        if (!p.TryGetValue("--map", out var mapName) || string.IsNullOrWhiteSpace(mapName))
        {
            Console.WriteLine("Missing required --map <Map>");
            return 2;
        }
        var outDir = GetOutDir(p);
        var outPath = Path.Combine(outDir, mapName + ".wdt");

        var lkMapDir = Path.Combine(lkRoot, "world", "maps", mapName);
        if (!Directory.Exists(lkMapDir))
        {
            var alt = Path.Combine(lkRoot, mapName);
            if (Directory.Exists(alt)) lkMapDir = alt;
            else { Console.WriteLine($"LK map directory not found: {lkMapDir} or {alt}"); return 4; }
        }

        var adtFiles = new List<string>(Directory.EnumerateFiles(lkMapDir, "*.adt", SearchOption.TopDirectoryOnly));
        var present = new HashSet<int>();

        var doodadNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var wmoNames = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        // Store per-ADT name lists for index remapping (key = tile index)
        var perAdtMmdxNames = new Dictionary<int, List<string>>();
        var perAdtMwmoNames = new Dictionary<int, List<string>>();

        foreach (var f in adtFiles)
        {
            var name = Path.GetFileNameWithoutExtension(f) ?? "";
            var parts = name.Split('_');
            int tileIdx = -1;
            if (parts.Length >= 3 &&
                int.TryParse(parts[^2], NumberStyles.Integer, CultureInfo.InvariantCulture, out var x) &&
                int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var y) &&
                x >= 0 && x < 64 && y >= 0 && y < 64)
            {
                tileIdx = y * 64 + x;
                present.Add(tileIdx);
            }

            byte[] data;
            try { data = File.ReadAllBytes(f); } catch { continue; }

            var mmdx = ExtractChunkPayload(data, "MMDX");
            var mwmo = ExtractChunkPayload(data, "MWMO");

            var mmdxList = SplitNullStringsList(mmdx);
            var mwmoList = SplitNullStringsList(mwmo);

            // Store per-ADT name lists for later remapping
            if (tileIdx >= 0)
            {
                perAdtMmdxNames[tileIdx] = mmdxList;
                perAdtMwmoNames[tileIdx] = mwmoList;
            }

            foreach (var s in mmdxList)
                if (!doodadNames.ContainsKey(s)) doodadNames[s] = s;
            foreach (var s in mwmoList)
                if (!wmoNames.ContainsKey(s)) wmoNames[s] = s;
        }

        var doodadList = new List<string>(doodadNames.Values);
        doodadList.Sort(StringComparer.OrdinalIgnoreCase);
        var wmoList = new List<string>(wmoNames.Values);
        wmoList.Sort(StringComparer.OrdinalIgnoreCase);

        static byte[] JoinNames(IEnumerable<string> names, bool extraEmpty)
        {
            using var ms = new MemoryStream();
            foreach (var n in names)
            {
                var b = Encoding.ASCII.GetBytes(n);
                ms.Write(b, 0, b.Length);
                ms.WriteByte(0);
            }
            if (extraEmpty) ms.WriteByte(0);
            return ms.ToArray();
        }

        var mdnmPayload = JoinNames(doodadList, false);
        var monmPayload = JoinNames(wmoList, true);

        using var outMs = new MemoryStream();
        using var bw = new BinaryWriter(outMs, Encoding.ASCII, leaveOpen: true);

        void WriteChunkDisk(string fourccForward, ReadOnlySpan<byte> payload)
        {
            var tag = ReverseFourCcBytes(fourccForward);
            bw.Write(tag);
            bw.Write(payload.Length);
            bw.Write(payload);
            Pad4(bw);
        }

        Span<byte> ver = stackalloc byte[4];
        BitConverter.TryWriteBytes(ver, 18);
        WriteChunkDisk("MVER", ver);

        var mphd = new byte[128];
        WriteChunkDisk("MPHD", mphd);

        var mainPayload = new byte[16 * 4096];
        long mainChunkStart = bw.BaseStream.Position;
        WriteChunkDisk("MAIN", mainPayload);
        int mainDataStart = (int)mainChunkStart + 8;

        WriteChunkDisk("MDNM", mdnmPayload);
        WriteChunkDisk("MONM", monmPayload);

        foreach (var idx in present)
        {
            int x = idx % 64;
            int y = idx / 64;
            string adtPath = Path.Combine(lkMapDir, $"{mapName}_{x:D2}_{y:D2}.adt");
            if (!File.Exists(adtPath))
            {
                string wildcard = $"*_{x:D2}_{y:D2}.adt";
                foreach (var f in Directory.EnumerateFiles(lkMapDir, wildcard, SearchOption.TopDirectoryOnly))
                { adtPath = f; break; }
            }
            if (!File.Exists(adtPath)) continue;

            byte[] adtBytes;
            try { adtBytes = File.ReadAllBytes(adtPath); } catch { continue; }

            byte[] mtexPayload = ExtractChunkPayload(adtBytes, "MTEX");
            byte[] mddfPayload = ExtractChunkPayload(adtBytes, "MDDF");
            byte[] modfPayload = ExtractChunkPayload(adtBytes, "MODF");

            // Remap MDDF/MODF indices from per-ADT (LK) to global (Alpha)
            // LK indices point into per-ADT MMDX/MWMO; Alpha indices point into WDT MDNM/MONM
            if (perAdtMmdxNames.TryGetValue(idx, out var thisMmdx) && mddfPayload.Length > 0)
            {
                mddfPayload = RemapMddfIndicesForAlpha(mddfPayload, thisMmdx, doodadList);
            }
            if (perAdtMwmoNames.TryGetValue(idx, out var thisMwmo) && modfPayload.Length > 0)
            {
                modfPayload = RemapModfIndicesForAlpha(modfPayload, thisMwmo, wmoList);
            }

            var mcnkList = MapMcnk(adtBytes);
            var mcnkBlocks = new List<byte[]>();
            foreach (var m in mcnkList)
            {
                if (m.offset >= 0 && m.size > 0 && m.offset + m.size <= adtBytes.Length)
                {
                    var block = new byte[m.size];
                    Buffer.BlockCopy(adtBytes, m.offset, block, 0, m.size);
                    mcnkBlocks.Add(block);
                }
            }

            int tileMhdrAbs = (int)bw.BaseStream.Position;
            bw.Write(ReverseFourCcBytes("MHDR"));
            int mhdrSize = 0x40;
            bw.Write(mhdrSize);
            long mhdrPayloadStart = bw.BaseStream.Position;
            var mhdrBuf = new byte[mhdrSize];
            bw.Write(mhdrBuf);
            Pad4(bw);

            long mcinChunkStart = bw.BaseStream.Position;
            bw.Write(ReverseFourCcBytes("MCIN"));
            int mcinPayloadSize = 16 * 256;
            bw.Write(mcinPayloadSize);
            long mcinPayloadStart = bw.BaseStream.Position;
            var mcinOut = new byte[mcinPayloadSize];
            bw.Write(mcinOut);
            Pad4(bw);

            long mtexChunkStart = -1;
            if (mtexPayload.Length > 0)
            {
                mtexChunkStart = bw.BaseStream.Position;
                WriteChunkDisk("MTEX", mtexPayload);
            }

            long mddfChunkStart = -1;
            if (mddfPayload.Length > 0)
            {
                mddfChunkStart = bw.BaseStream.Position;
                WriteChunkDisk("MDDF", mddfPayload);
            }

            long modfChunkStart = -1;
            if (modfPayload.Length > 0)
            {
                modfChunkStart = bw.BaseStream.Position;
                WriteChunkDisk("MODF", modfPayload);
            }

            var newMcnk = new List<(int off, int sizePayload)>();
            foreach (var block in mcnkBlocks)
            {
                int abs = (int)bw.BaseStream.Position;
                bw.Write(block);
                Pad4(bw);
                int payload = Math.Max(0, block.Length - 8);
                newMcnk.Add((abs, payload));
            }

            for (int i = 0; i < 256; i++)
            {
                int entryOff = (int)mcinPayloadStart + i * 16;
                bw.BaseStream.Seek(entryOff, SeekOrigin.Begin);
                if (i < newMcnk.Count)
                {
                    var e = newMcnk[i];
                    int rel = e.off - (int)mhdrPayloadStart;
                    bw.Write(rel);
                    bw.Write(e.sizePayload);
                }
                else
                {
                    bw.Write(0);
                    bw.Write(0);
                }
                bw.Write(0);
                bw.Write(0);
            }

            bw.BaseStream.Seek(mhdrPayloadStart + 0x00, SeekOrigin.Begin);
            int offsInfo = (int)(mcinChunkStart - mhdrPayloadStart);
            int offsTex  = mtexChunkStart >= 0 ? (int)(mtexChunkStart - mhdrPayloadStart) : 0;
            int sizeTex  = mtexPayload.Length;
            int offsDoo  = mddfChunkStart >= 0 ? (int)(mddfChunkStart - mhdrPayloadStart) : 0;
            int sizeDoo  = mddfPayload.Length;
            int offsMob  = modfChunkStart >= 0 ? (int)(modfChunkStart - mhdrPayloadStart) : 0;
            int sizeMob  = modfPayload.Length;
            bw.Write(offsInfo);
            bw.Write(offsTex);
            bw.Write(sizeTex);
            bw.Write(offsDoo);
            bw.Write(sizeDoo);
            bw.Write(offsMob);
            bw.Write(sizeMob);
            bw.Write(0);

            int mcInEnd = offsInfo + 8 + mcinPayloadSize;
            int mtExEnd = offsTex  > 0 ? (offsTex + 8 + Math.Max(0, sizeTex)) : 0;
            int mdDfEnd = offsDoo  > 0 ? (offsDoo + 8 + Math.Max(0, sizeDoo)) : 0;
            int moDfEnd = offsMob  > 0 ? (offsMob + 8 + Math.Max(0, sizeMob)) : 0;
            int firstRel = Math.Max(Math.Max(mcInEnd, mtExEnd), Math.Max(mdDfEnd, moDfEnd));

            int mainEntryOff = mainDataStart + idx * 16;
            bw.BaseStream.Seek(mainEntryOff, SeekOrigin.Begin);
            bw.Write(tileMhdrAbs);
            bw.Write(firstRel);

            bw.BaseStream.Seek(0, SeekOrigin.End);
        }

        try
        {
            var final = outMs.ToArray();
            var pairs = new[] { "MVER", "MPHD", "MAIN", "MDNM", "MONM" };
            var sb = new StringBuilder();
            foreach (var k in pairs)
            {
                var fwd = Encoding.ASCII.GetBytes(k);
                var rev = ReverseFourCcBytes(k);
                var (fc, _) = FindOccurrences(final, fwd);
                var (rc, _) = FindOccurrences(final, rev);
                if (sb.Length > 0) sb.Append(' ');
                sb.Append(k).Append(':').Append(rc).Append('/').Append(fc);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath)) ?? ".");
            File.WriteAllBytes(outPath, final);
            Console.WriteLine($"Wrote {outPath}");
            Console.WriteLine($"fourcc_orientation {sb}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write WDT: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static int WdtSanity(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--alpha-wdt", out var alphaWdt) || string.IsNullOrWhiteSpace(alphaWdt))
        {
            Console.WriteLine("Missing required --alpha-wdt <path>");
            return 2;
        }
        if (!File.Exists(alphaWdt)) { Console.WriteLine($"File not found: {alphaWdt}"); return 3; }

        var outDir = EnsureSubOut(p, "wdt_sanity");
        int limit = -1;
        if (p.TryGetValue("--limit", out var limStr) && int.TryParse(limStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var lim) && lim > 0)
            limit = lim;

        byte[] wdt;
        try { wdt = File.ReadAllBytes(alphaWdt); } catch (Exception ex) { Console.WriteLine($"Read failed: {ex.Message}"); return 4; }

        // Find MAIN (NIAM)
        int mainOff = -1; int mainSize = 0;
        for (int i = 0; i + 8 <= wdt.Length; )
        {
            var tok = ReadFcc(wdt, i);
            int size = (int)ReadU32Le(wdt, i + 4);
            if (tok == "NIAM") { mainOff = i; mainSize = size; break; }
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break; i = next;
        }
        if (mainOff < 0) { Console.WriteLine("MAIN not found"); return 5; }

        var rows = new List<string> { "tileX,tileY,mhdrAbs,main_firstMcnkRel,calc_firstMcnkRel,mc_in_tok,mt_ex_tok,md_df_tok,mo_df_tok,errors" };
        int processed = 0;
        int mainData = mainOff + 8;
        for (int idx = 0; idx < 4096; idx++)
        {
            int pos = mainData + idx * 16;
            if (pos + 16 > wdt.Length) break;
            int mhdrAbs = (int)ReadU32Le(wdt, pos + 0);
            int firstRel = (int)ReadU32Le(wdt, pos + 4);
            if (mhdrAbs <= 0) continue;

            int x = idx % 64, y = idx / 64;
            var errs = new List<string>();
            string tMcin = ""; string tMtex = ""; string tMddf = ""; string tModf = "";
            int calcFirst = 0;

            int mhdrData = mhdrAbs + 8;
            if (mhdrData + 64 > wdt.Length) { errs.Add("mhdr_oob"); goto Emit; }

            int offsInfo = (int)ReadU32Le(wdt, mhdrData + 0x00);
            int offsTex  = (int)ReadU32Le(wdt, mhdrData + 0x04);
            int sizeTex  = (int)ReadU32Le(wdt, mhdrData + 0x08);
            int offsDoo  = (int)ReadU32Le(wdt, mhdrData + 0x0C);
            int sizeDoo  = (int)ReadU32Le(wdt, mhdrData + 0x10);
            int offsMob  = (int)ReadU32Le(wdt, mhdrData + 0x14);
            int sizeMob  = (int)ReadU32Le(wdt, mhdrData + 0x18);

            // Tokens
            if (offsInfo > 0 && mhdrData + offsInfo + 8 <= wdt.Length) tMcin = ReadFcc(wdt, mhdrData + offsInfo);
            if (offsTex  > 0 && mhdrData + offsTex  + 8 <= wdt.Length) tMtex = ReadFcc(wdt, mhdrData + offsTex);
            if (offsDoo  > 0 && mhdrData + offsDoo  + 8 <= wdt.Length) tMddf = ReadFcc(wdt, mhdrData + offsDoo);
            if (offsMob  > 0 && mhdrData + offsMob  + 8 <= wdt.Length) tModf = ReadFcc(wdt, mhdrData + offsMob);

            // Calculate first MCNK
            int mcInEnd = offsInfo > 0 ? (offsInfo + 8 + Math.Max(0, Math.Min((int)ReadU32Le(wdt, mhdrData + offsInfo + 4), wdt.Length))) : 0;
            if (mcInEnd == 0) errs.Add("mcin_missing");
            int mtExEnd = offsTex  > 0 ? (offsTex  + 8 + Math.Max(0, sizeTex)) : 0;
            int mdDfEnd = offsDoo  > 0 ? (offsDoo  + 8 + Math.Max(0, sizeDoo)) : 0;
            int moDfEnd = offsMob  > 0 ? (offsMob  + 8 + Math.Max(0, sizeMob)) : 0;
            calcFirst = Math.Max(Math.Max(mcInEnd, mtExEnd), Math.Max(mdDfEnd, moDfEnd));

            if (calcFirst != firstRel) errs.Add($"firstRel_mismatch({firstRel}!={calcFirst})");

        Emit:
            rows.Add(string.Join(',', new[]
            {
                x.ToString(CultureInfo.InvariantCulture),
                y.ToString(CultureInfo.InvariantCulture),
                mhdrAbs.ToString(CultureInfo.InvariantCulture),
                firstRel.ToString(CultureInfo.InvariantCulture),
                calcFirst.ToString(CultureInfo.InvariantCulture),
                Csv(tMcin), Csv(tMtex), Csv(tMddf), Csv(tModf),
                Csv(string.Join('|', errs))
            }));

            processed++;
            if (limit > 0 && processed >= limit) break;
        }

        var csvPath = Path.Combine(outDir, "wdt_sanity.csv");
        try { File.WriteAllLines(csvPath, rows); Console.WriteLine($"Wrote {csvPath}"); }
        catch (Exception ex) { Console.WriteLine($"Failed to write wdt_sanity.csv: {ex.Message}"); return 6; }

        return 0;
    }

    private static void ExtractPlacementsFromWdtTile(byte[] wdt, int mhdrAbs,
        out byte[] mddfPayload, out byte[] modfPayload,
        out string tokMddf, out string tokModf)
    {
        mddfPayload = Array.Empty<byte>();
        modfPayload = Array.Empty<byte>();
        tokMddf = "";
        tokModf = "";

        int mhdrDataStart = mhdrAbs + 8;
        if (mhdrDataStart + 64 > wdt.Length) return;

        int offsDoo = (int)ReadU32Le(wdt, mhdrDataStart + 0x0C);
        int sizeDoo = (int)ReadU32Le(wdt, mhdrDataStart + 0x10);
        int offsMob = (int)ReadU32Le(wdt, mhdrDataStart + 0x14);
        int sizeMob = (int)ReadU32Le(wdt, mhdrDataStart + 0x18);

        if (offsDoo > 0)
        {
            int hdrAbs = mhdrDataStart + offsDoo;
            if (hdrAbs + 8 <= wdt.Length)
                tokMddf = ReadFcc(wdt, hdrAbs);
            int abs = hdrAbs + 8;
            if (sizeDoo > 0 && abs + sizeDoo <= wdt.Length)
            {
                mddfPayload = new byte[sizeDoo];
                Buffer.BlockCopy(wdt, abs, mddfPayload, 0, sizeDoo);
            }
        }

        if (offsMob > 0)
        {
            int hdrAbs = mhdrDataStart + offsMob;
            if (hdrAbs + 8 <= wdt.Length)
                tokModf = ReadFcc(wdt, hdrAbs);
            int abs = hdrAbs + 8;
            if (sizeMob > 0 && abs + sizeMob <= wdt.Length)
            {
                modfPayload = new byte[sizeMob];
                Buffer.BlockCopy(wdt, abs, modfPayload, 0, sizeMob);
            }
        }
    }

    private static byte[] ExtractChunkPayload(byte[] data, string fourcc)
    {
        if (data is null || data.Length == 0) return Array.Empty<byte>();
        var info = FindChunk(data, fourcc);
        if (!info.found) return Array.Empty<byte>();
        int abs = info.offset + 8;
        if (info.size <= 0 || abs + info.size > data.Length) return Array.Empty<byte>();
        var buf = new byte[info.size];
        Buffer.BlockCopy(data, abs, buf, 0, info.size);
        return buf;
    }

    private static string Sha1Hex(byte[] data)
    {
        if (data is null || data.Length == 0) return "";
        using var sha1 = SHA1.Create();
        var hash = sha1.ComputeHash(data);
        var sb = new StringBuilder(hash.Length * 2);
        foreach (var b in hash) sb.Append(b.ToString("X2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    private static int PlacementsSweepDiff(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--alpha-wdt", out var alphaWdt) || string.IsNullOrWhiteSpace(alphaWdt))
        {
            Console.WriteLine("Missing required --alpha-wdt <path>");
            return 2;
        }
        if (!p.TryGetValue("--lk-dir", out var lkDir) || string.IsNullOrWhiteSpace(lkDir))
        {
            Console.WriteLine("Missing required --lk-dir <dir> (folder containing LK-like ADTs)");
            return 2;
        }
        if (!File.Exists(alphaWdt)) { Console.WriteLine($"File not found: {alphaWdt}"); return 3; }
        if (!Directory.Exists(lkDir)) { Console.WriteLine($"Directory not found: {lkDir}"); return 3; }

        var outDir = EnsureSubOut(p, "placements_sweep_diff");
        var hexMismatch = p.ContainsKey("--hex");
        int limit = -1;
        if (p.TryGetValue("--limit", out var limStr) && int.TryParse(limStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var lim) && lim > 0)
            limit = lim;

        Console.WriteLine($"[placements-sweep-diff] alpha-wdt={alphaWdt}\n lk-dir={lkDir}\n out={outDir} hex={(hexMismatch?1:0)} limit={(limit>0?limit:0)}");

        byte[] wdtBytes;
        try { wdtBytes = File.ReadAllBytes(alphaWdt); }
        catch (Exception ex) { Console.WriteLine($"Failed to read WDT: {ex.Message}"); return 4; }

        // Prefer map name from lk-dir leaf; fallback to WDT filename
        var mapName = Path.GetFileName(Path.GetFullPath(lkDir));
        if (string.IsNullOrWhiteSpace(mapName))
            mapName = Path.GetFileNameWithoutExtension(alphaWdt) ?? "map";
        var tiles = ParseAlphaWdtTiles(wdtBytes);

        var rows = new List<string> {
            "tileX,tileY,adt_path,wdt_mddf_size,adt_mddf_size,wdt_mddf_sha1,adt_mddf_sha1,md_df_equal,wdt_modf_size,adt_modf_size,wdt_modf_sha1,adt_modf_sha1,mod_f_equal,wdt_tok_mddf,wdt_tok_modf"
        };

        var payloadDir = Path.Combine(outDir, "payloads");
        if (hexMismatch) Directory.CreateDirectory(payloadDir);

        int processed = 0, compared = 0, missingAdt = 0;

        foreach (var t in tiles)
        {
            if (t.mhdrAbs <= 0) continue;
            var x = t.index % 64;
            var y = t.index / 64;
            var adtName = $"{mapName}_{x:D2}_{y:D2}.adt";
            var adtPath = Path.Combine(lkDir, adtName);
            if (!File.Exists(adtPath))
            {
                // Fallback: accept any map name prefix as long as indices match
                string wildcard = $"*_{x:D2}_{y:D2}.adt";
                foreach (var f in Directory.EnumerateFiles(lkDir, wildcard, SearchOption.TopDirectoryOnly))
                {
                    adtPath = f; break;
                }
            }

            // Extract WDT tile placements
            ExtractPlacementsFromWdtTile(wdtBytes, t.mhdrAbs,
                out var wdtMddf, out var wdtModf,
                out var wdtTokMddf, out var wdtTokModf);

            byte[] adtBytes = Array.Empty<byte>();
            if (File.Exists(adtPath))
            {
                try { adtBytes = File.ReadAllBytes(adtPath); }
                catch (Exception ex) { Console.WriteLine($"Read ADT failed: {adtPath} -> {ex.Message}"); }
            }
            else { missingAdt++; }

            // Extract ADT placements
            var adtMddf = ExtractChunkPayload(adtBytes, "MDDF");
            var adtModf = ExtractChunkPayload(adtBytes, "MODF");

            string wdtMddfHash = Sha1Hex(wdtMddf);
            string adtMddfHash = Sha1Hex(adtMddf);
            string wdtModfHash = Sha1Hex(wdtModf);
            string adtModfHash = Sha1Hex(adtModf);

            bool eqMddf = wdtMddf.Length == adtMddf.Length && wdtMddfHash == adtMddfHash;
            bool eqModf = wdtModf.Length == adtModf.Length && wdtModfHash == adtModfHash;

            rows.Add(string.Join(',', new[]
            {
                x.ToString(CultureInfo.InvariantCulture),
                y.ToString(CultureInfo.InvariantCulture),
                Csv(adtPath),
                wdtMddf.Length.ToString(CultureInfo.InvariantCulture),
                adtMddf.Length.ToString(CultureInfo.InvariantCulture),
                wdtMddfHash,
                adtMddfHash,
                (eqMddf?"1":"0"),
                wdtModf.Length.ToString(CultureInfo.InvariantCulture),
                adtModf.Length.ToString(CultureInfo.InvariantCulture),
                wdtModfHash,
                adtModfHash,
                (eqModf?"1":"0"),
                wdtTokMddf,
                wdtTokModf
            }));

            if (hexMismatch && (!eqMddf || !eqModf))
            {
                var baseName = $"{mapName}_{x:D2}_{y:D2}";
                try { File.WriteAllBytes(Path.Combine(payloadDir, baseName + "_wdt_mddf.bin"), wdtMddf); } catch { }
                try { File.WriteAllBytes(Path.Combine(payloadDir, baseName + "_adt_mddf.bin"), adtMddf); } catch { }
                try { File.WriteAllBytes(Path.Combine(payloadDir, baseName + "_wdt_modf.bin"), wdtModf); } catch { }
                try { File.WriteAllBytes(Path.Combine(payloadDir, baseName + "_adt_modf.bin"), adtModf); } catch { }
            }

            processed++;
            if (adtBytes.Length > 0) compared++;
            if (limit > 0 && processed >= limit) break;
        }

        var csvPath = Path.Combine(outDir, "placements_by_tile.csv");
        try { File.WriteAllLines(csvPath, rows); Console.WriteLine($"Wrote {csvPath}"); }
        catch (Exception ex) { Console.WriteLine($"Failed to write placements_by_tile.csv: {ex.Message}"); return 5; }

        Console.WriteLine($"Tiles processed={processed} compared={compared} missingAdt={missingAdt}");
        return 0;
    }

    private static int MtexAudit(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--alpha-wdt", out var alphaWdt) || string.IsNullOrWhiteSpace(alphaWdt))
        {
            Console.WriteLine("Missing required --alpha-wdt <path>");
            return 2;
        }
        if (!File.Exists(alphaWdt)) { Console.WriteLine($"File not found: {alphaWdt}"); return 3; }

        int limit = -1;
        if (p.TryGetValue("--limit", out var limStr) && int.TryParse(limStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out var lim) && lim > 0)
            limit = lim;

        var outDir = EnsureSubOut(p, "mtex_audit");
        byte[] wdt;
        try { wdt = File.ReadAllBytes(alphaWdt); } catch (Exception ex) { Console.WriteLine($"Read failed: {ex.Message}"); return 4; }

        var tiles = ParseAlphaWdtTiles(wdt);
        var rows = new List<string> { "tileX,tileY,mtex_count,max_mcly_index,violations,first_violation_chunk,first_violation_layer,notes" };

        int processed = 0;
        foreach (var t in tiles)
        {
            if (t.mhdrAbs <= 0) continue;
            int x = t.index % 64, y = t.index / 64;

            int mhdrData = t.mhdrAbs + 8;
            if (mhdrData + 64 > wdt.Length)
            {
                rows.Add(string.Join(',', new[] { x.ToString(CultureInfo.InvariantCulture), y.ToString(CultureInfo.InvariantCulture), "0", "", "", "", "", Csv("mhdr_oob") }));
                continue;
            }

            int offsInfo = (int)ReadU32Le(wdt, mhdrData + 0x00);
            int offsTex  = (int)ReadU32Le(wdt, mhdrData + 0x04);
            int sizeTex  = (int)ReadU32Le(wdt, mhdrData + 0x08);

            // Read MTEX string list (count includes empty entries to match indices behavior)
            int mtexCount = 0;
            if (offsTex > 0 && sizeTex > 0 && mhdrData + offsTex + 8 + sizeTex <= wdt.Length)
            {
                int payloadAbs = mhdrData + offsTex + 8;
                int end = payloadAbs + sizeTex;
                int pos = payloadAbs;
                while (pos <= end)
                {
                    int nul = Array.IndexOf(wdt, (byte)0, pos, end - pos);
                    if (nul < 0) { mtexCount++; break; }
                    mtexCount++;
                    pos = nul + 1;
                }
            }

            // From MCIN, enumerate MCNKs and scan MCLY texture indices
            int violations = 0; string firstChunk = ""; int firstLayer = -1; int maxIndex = -1; string notes = "";
            if (offsInfo > 0 && mhdrData + offsInfo + 8 + 16 * 256 <= wdt.Length)
            {
                int mcinAbs = mhdrData + offsInfo;
                int mcinSize = (int)ReadU32Le(wdt, mcinAbs + 4);
                int entries = Math.Min(256, Math.Max(0, mcinSize) / 16);
                int mcinData = mcinAbs + 8;
                for (int i = 0; i < entries; i++)
                {
                    int moff = (int)ReadU32Le(wdt, mcinData + i * 16 + 0);
                    int msz  = (int)ReadU32Le(wdt, mcinData + i * 16 + 4);
                    if (moff <= 0 || msz <= 0) continue;
                    long mcnkAbsL = (long)mhdrData + (long)moff;
                    long mcnkEndL = mcnkAbsL + 8L + (long)msz;
                    if (mcnkAbsL < 0 || mcnkAbsL + 8L > wdt.LongLength || mcnkEndL > wdt.LongLength)
                    {
                        notes = "mcnk_oob";
                        continue;
                    }
                    int mcnkAbs = (int)mcnkAbsL;
                    int mcnkEnd = (int)mcnkEndL;
                    // Scan subchunks inside MCNK for MCLY
                    int pos = mcnkAbs + 8;
                    while (pos + 8 <= mcnkEnd)
                    {
                        string tok = ReadFcc(wdt, pos);
                        int sz = (int)ReadU32Le(wdt, pos + 4);
                        int data = pos + 8;
                        int next = data + sz + ((sz & 1) == 1 ? 1 : 0);
                        if (next <= pos || next > mcnkEnd) break;
                        if (tok == "YLCM") // 'MCLY'
                        {
                            int layers = Math.Max(0, sz / 16);
                            for (int li = 0; li < layers; li++)
                            {
                                int entry = data + li * 16;
                                if (entry + 4 > wdt.Length) break;
                                int texId = (int)ReadU32Le(wdt, entry + 0);
                                if (texId > maxIndex) maxIndex = texId;
                                if (mtexCount > 0 && texId >= mtexCount)
                                {
                                    violations++;
                                    if (firstLayer < 0) { firstLayer = li; firstChunk = i.ToString(CultureInfo.InvariantCulture); }
                                }
                            }
                        }
                        pos = next;
                    }
                }
            }
            else
            {
                notes = "mcin_missing";
            }

            rows.Add(string.Join(',', new[]
            {
                x.ToString(CultureInfo.InvariantCulture),
                y.ToString(CultureInfo.InvariantCulture),
                mtexCount.ToString(CultureInfo.InvariantCulture),
                (maxIndex >= 0 ? maxIndex.ToString(CultureInfo.InvariantCulture) : "").ToString(CultureInfo.InvariantCulture),
                violations.ToString(CultureInfo.InvariantCulture),
                Csv(firstChunk),
                (firstLayer >= 0 ? firstLayer.ToString(CultureInfo.InvariantCulture) : ""),
                Csv(notes)
            }));

            processed++;
            if (limit > 0 && processed >= limit) break;
        }

        var csvPath = Path.Combine(outDir, "mtex_audit.csv");
        try { File.WriteAllLines(csvPath, rows); Console.WriteLine($"Wrote {csvPath}"); }
        catch (Exception ex) { Console.WriteLine($"Failed to write mtex_audit.csv: {ex.Message}"); return 5; }

        return 0;
    }

    private static int McnkDiff(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--a", out var aPath) || string.IsNullOrWhiteSpace(aPath) ||
            !p.TryGetValue("--b", out var bPath) || string.IsNullOrWhiteSpace(bPath))
        {
            Console.WriteLine("Missing required --a <adtA> and --b <adtB>");
            return 2;
        }
        if (!File.Exists(aPath) || !File.Exists(bPath))
        {
            Console.WriteLine("Input file not found");
            return 3;
        }
        var outDir = EnsureSubOut(p, "mcnk_diff");

        byte[] a, b;
        try { a = File.ReadAllBytes(aPath); b = File.ReadAllBytes(bPath); }
        catch (Exception ex) { Console.WriteLine($"Read failed: {ex.Message}"); return 4; }

        var listA = MapMcnk(a);
        var listB = MapMcnk(b);
        var max = Math.Max(listA.Count, listB.Count);

        string HexPrefix(byte[] bytes, int n)
        {
            var sb = new StringBuilder(n * 2);
            for (int i = 0; i < bytes.Length && i < n; i++) sb.Append(bytes[i].ToString("X2", CultureInfo.InvariantCulture));
            return sb.ToString();
        }

        var lines = new List<string> { "index,a_offset,a_size,b_offset,b_size,size_delta,a_hex32,b_hex32,prefix_equal" };
        for (int i = 0; i < max; i++)
        {
            var hasA = i < listA.Count;
            var hasB = i < listB.Count;
            var aOff = hasA ? listA[i].offset : -1;
            var aSize = hasA ? listA[i].size : -1;
            var bOff = hasB ? listB[i].offset : -1;
            var bSize = hasB ? listB[i].size : -1;
            var sizeDelta = (hasA && hasB) ? (aSize - bSize) : 0;
            var aHex = hasA ? HexPrefix(listA[i].prefix, 32) : "";
            var bHex = hasB ? HexPrefix(listB[i].prefix, 32) : "";
            var eq = (hasA && hasB && aHex == bHex) ? 1 : 0;
            lines.Add(string.Join(',', new[]
            {
                i.ToString(CultureInfo.InvariantCulture),
                aOff.ToString(CultureInfo.InvariantCulture),
                aSize.ToString(CultureInfo.InvariantCulture),
                bOff.ToString(CultureInfo.InvariantCulture),
                bSize.ToString(CultureInfo.InvariantCulture),
                sizeDelta.ToString(CultureInfo.InvariantCulture),
                aHex, bHex, eq.ToString(CultureInfo.InvariantCulture)
            }));
        }

        var csvPath = Path.Combine(outDir, "mcnk_diff.csv");
        try { File.WriteAllLines(csvPath, lines); Console.WriteLine($"Wrote {csvPath}"); }
        catch (Exception ex) { Console.WriteLine($"Failed writing mcnk_diff.csv: {ex.Message}"); return 5; }
        return 0;
    }

    private static int PackAlpha(string[] argv)
    {
        var p = ParseArgs(argv);
        // Placements-only mode: --alpha-wdt, --lk-dir, --out
        if (p.TryGetValue("--alpha-wdt", out var inAlpha) && !string.IsNullOrWhiteSpace(inAlpha)
            && p.TryGetValue("--lk-dir", out var lkDir) && !string.IsNullOrWhiteSpace(lkDir)
            && p.TryGetValue("--out", out var outAlpha) && !string.IsNullOrWhiteSpace(outAlpha))
        {
            if (!File.Exists(inAlpha)) { Console.WriteLine($"File not found: {inAlpha}"); return 2; }
            if (!Directory.Exists(lkDir)) { Console.WriteLine($"Directory not found: {lkDir}"); return 2; }

            Console.WriteLine($"[pack-alpha placements] in={inAlpha}\n lk-dir={lkDir}\n out={outAlpha}");

            byte[] src;
            try { src = File.ReadAllBytes(inAlpha); } catch (Exception ex) { Console.WriteLine($"Read failed: {ex.Message}"); return 3; }

            var mapName = Path.GetFileNameWithoutExtension(inAlpha) ?? "map";
            // Scan top-level to find MAIN (NIAM)
            int mainOff = -1, mainSize = 0;
            for (int i = 0; i + 8 <= src.Length;)
            {
                var tok = ReadFcc(src, i);
                int size = (int)ReadU32Le(src, i + 4);
                if (tok == "NIAM") { mainOff = i; mainSize = size; break; }
                int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
                if (next <= i) break; i = next;
            }
            if (mainOff < 0) { Console.WriteLine("MAIN not found in alpha WDT"); return 4; }
            int mainData = mainOff + 8;

            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);
            bw.Write(src);

            var tiles = ParseAlphaWdtTiles(src);

            foreach (var t in tiles)
            {
                if (t.mhdrAbs <= 0) continue;
                int x = t.index % 64; int y = t.index / 64;
                var adtPath = Path.Combine(lkDir, $"{mapName}_{x:D2}_{y:D2}.adt");
                byte[] adt = Array.Empty<byte>();
                if (File.Exists(adtPath))
                {
                    try { adt = File.ReadAllBytes(adtPath); } catch { }
                }

                ExtractPlacementsFromWdtTile(src, t.mhdrAbs, out var wdtMddf, out var wdtModf, out var tokMddf, out var tokModf);
                var adtMddf = ExtractChunkPayload(adt, "MDDF");
                var adtModf = ExtractChunkPayload(adt, "MODF");

                // Fallback: if ADT missing, keep original payloads
                var newMddf = adtMddf.Length > 0 ? adtMddf : wdtMddf;
                var newModf = adtModf.Length > 0 ? adtModf : wdtModf;

                // Gather original header tokens and MTEX payload
                int mhdrDataStart = t.mhdrAbs + 8;
                string tokMhdr = ReadFcc(src, t.mhdrAbs);
                int offsInfo = (int)ReadU32Le(src, mhdrDataStart + 0x00);
                int offsTex  = (int)ReadU32Le(src, mhdrDataStart + 0x04);
                int sizeTex  = (int)ReadU32Le(src, mhdrDataStart + 0x08);
                int offsDoo  = (int)ReadU32Le(src, mhdrDataStart + 0x0C);
                int offsMob  = (int)ReadU32Le(src, mhdrDataStart + 0x14);

                // Read tokens for subchunks as written in original file
                string tokMcin = ReadFcc(src, mhdrDataStart + offsInfo);
                string tokMtex = offsTex > 0 ? ReadFcc(src, mhdrDataStart + offsTex) : "";
                if (string.IsNullOrEmpty(tokMddf) && offsDoo > 0) tokMddf = ReadFcc(src, mhdrDataStart + offsDoo);
                if (string.IsNullOrEmpty(tokModf) && offsMob > 0) tokModf = ReadFcc(src, mhdrDataStart + offsMob);

                // Preserve original MTEX payload from Alpha WDT tile for compatibility with 0.5.3 client
                byte[] mtexPayload = Array.Empty<byte>();
                if (offsTex > 0 && sizeTex > 0)
                {
                    int abs = mhdrDataStart + offsTex + 8;
                    if (abs + sizeTex <= src.Length)
                    {
                        mtexPayload = new byte[sizeTex];
                        Buffer.BlockCopy(src, abs, mtexPayload, 0, sizeTex);
                    }
                }

                // Read original MCIN entries to copy MCNK blocks and sizes
                int mcinAbs = mhdrDataStart + offsInfo;
                int mcinSize = (int)ReadU32Le(src, mcinAbs + 4);
                int mcinData = mcinAbs + 8;
                int entries = Math.Min(256, mcinSize / 16);
                var mcnkSrc = new List<(int off, int size)>();
                for (int i = 0; i < entries; i++)
                {
                    int moff = (int)ReadU32Le(src, mcinData + i * 16 + 0);
                    int msz  = (int)ReadU32Le(src, mcinData + i * 16 + 4);
                    if (moff > 0 && msz > 0 && moff + 8 + msz <= src.Length)
                        mcnkSrc.Add((moff, msz));
                }

                // Start writing new tile blob at end of file
                int newMhdrAbs = (int)ms.Position;
                void WriteAscii(string s) { bw.Write(Encoding.ASCII.GetBytes(s)); }

                // MHDR header + payload placeholder (0x40)
                WriteAscii(tokMhdr);
                // Use original MHDR payload size for compatibility
                int origMhdrSize = (int)ReadU32Le(src, t.mhdrAbs + 4);
                if (origMhdrSize <= 0) origMhdrSize = 0x40;
                bw.Write(origMhdrSize);
                long mhdrPayloadStart = ms.Position;
                var mhdrBuf = new byte[origMhdrSize];
                // Copy original MHDR payload to preserve flags/unknowns
                int origMhdrPayloadAbs = t.mhdrAbs + 8;
                if (origMhdrPayloadAbs >= 0 && origMhdrPayloadAbs + origMhdrSize <= src.Length)
                {
                    Buffer.BlockCopy(src, origMhdrPayloadAbs, mhdrBuf, 0, origMhdrSize);
                }
                bw.Write(mhdrBuf);
                Pad4(bw);

                // MCIN placeholder (write original token and 4096 payload)
                long mcinChunkStart = ms.Position;
                WriteAscii(tokMcin);
                // Use original MCIN payload size (typically 4096)
                int origMcinSize = (int)ReadU32Le(src, mcinAbs + 4);
                if (origMcinSize <= 0) origMcinSize = 16 * 256;
                bw.Write(origMcinSize);
                long mcinPayloadStart = ms.Position;
                var mcinOut = new byte[origMcinSize];
                bw.Write(mcinOut);
                Pad4(bw);

                // MTEX (preserve original payload)
                long mtexChunkStart = -1;
                if (mtexPayload.Length > 0 && !string.IsNullOrEmpty(tokMtex))
                {
                    mtexChunkStart = ms.Position;
                    WriteAscii(tokMtex);
                    bw.Write(mtexPayload.Length);
                    bw.Write(mtexPayload);
                    Pad4(bw);
                }

                // MDDF from ADT (preserve original token orientation)
                long mddfChunkStart = -1;
                if (!string.IsNullOrEmpty(tokMddf) && newMddf.Length >= 0)
                {
                    mddfChunkStart = ms.Position;
                    WriteAscii(tokMddf);
                    bw.Write(newMddf.Length);
                    if (newMddf.Length > 0) bw.Write(newMddf);
                    Pad4(bw);
                }

                // MODF from ADT
                long modfChunkStart = -1;
                if (!string.IsNullOrEmpty(tokModf) && newModf.Length >= 0)
                {
                    modfChunkStart = ms.Position;
                    WriteAscii(tokModf);
                    bw.Write(newModf.Length);
                    if (newModf.Length > 0) bw.Write(newModf);
                    Pad4(bw);
                }

                // MCNK blocks copied from original, record absolute offsets for MCIN backpatch
                var newMcnkAbs = new List<(int off, int size)>();
                int firstMcnkStartAbs = -1;
                foreach (var e in mcnkSrc)
                {
                    int absOff = (int)ms.Position;
                    // copy MCNK header+payload
                    int total = 8 + e.size;
                    bw.Write(new ReadOnlySpan<byte>(src, e.off, total));
                    Pad4(bw);
                    if (firstMcnkStartAbs < 0) firstMcnkStartAbs = absOff;
                    newMcnkAbs.Add((absOff, e.size));
                }

                // Backpatch MCIN entries (offsets are RELATIVE to MHDR.data)
                bw.BaseStream.Seek(mcinPayloadStart, SeekOrigin.Begin);
                for (int i = 0; i < 256; i++)
                {
                    int entryOff = (int)mcinPayloadStart + i * 16;
                    bw.BaseStream.Seek(entryOff, SeekOrigin.Begin);
                    if (i < newMcnkAbs.Count)
                    {
                        int rel = OffRel(newMcnkAbs[i].off);
                        bw.Write(rel);
                        bw.Write(newMcnkAbs[i].size);
                    }
                    else
                    {
                        bw.Write(0);
                        bw.Write(0);
                    }
                    bw.Write(0); // flags
                    bw.Write(0); // async
                }

                // Backpatch MHDR offsets/sizes
                int OffRel(long chunkStart) => (int)(chunkStart >= 0 ? (chunkStart - mhdrPayloadStart) : 0);
                bw.BaseStream.Seek(mhdrPayloadStart + 0x00, SeekOrigin.Begin); bw.Write(OffRel(mcinChunkStart));
                bw.BaseStream.Seek(mhdrPayloadStart + 0x04, SeekOrigin.Begin); bw.Write(OffRel(mtexChunkStart));
                bw.BaseStream.Seek(mhdrPayloadStart + 0x08, SeekOrigin.Begin); bw.Write(mtexPayload.Length);
                bw.BaseStream.Seek(mhdrPayloadStart + 0x0C, SeekOrigin.Begin); bw.Write(OffRel(mddfChunkStart));
                bw.BaseStream.Seek(mhdrPayloadStart + 0x10, SeekOrigin.Begin); bw.Write(newMddf.Length);
                bw.BaseStream.Seek(mhdrPayloadStart + 0x14, SeekOrigin.Begin); bw.Write(OffRel(modfChunkStart));
                bw.BaseStream.Seek(mhdrPayloadStart + 0x18, SeekOrigin.Begin); bw.Write(newModf.Length);

                // Compute size to first MCNK for MAIN entry using actual position (includes padding)
                int firstMcnkRel;
                if (firstMcnkStartAbs >= 0)
                {
                    firstMcnkRel = (int)(firstMcnkStartAbs - mhdrPayloadStart);
                }
                else
                {
                    // Fallback if no MCNKs: end of last pre-MCNK chunk
                    int endMcin = OffRel(mcinChunkStart) + 8 + origMcinSize;
                    int endMtex = OffRel(mtexChunkStart) + 8 + mtexPayload.Length;
                    int endMddf = OffRel(mddfChunkStart) + 8 + newMddf.Length;
                    int endModf = OffRel(modfChunkStart) + 8 + newModf.Length;
                    firstMcnkRel = Math.Max(Math.Max(endMcin, endMtex), Math.Max(endMddf, endModf));
                }

                // Patch MAIN entry
                int tilePos = mainData + t.index * 16;
                bw.BaseStream.Seek(tilePos + 0, SeekOrigin.Begin); bw.Write(newMhdrAbs);
                bw.BaseStream.Seek(tilePos + 4, SeekOrigin.Begin); bw.Write(firstMcnkRel);

                // Return to end for next tile
                bw.BaseStream.Seek(0, SeekOrigin.End);
            }

            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outAlpha)) ?? ".");
                File.WriteAllBytes(outAlpha, ms.ToArray());
                Console.WriteLine($"Wrote {outAlpha}");
            }
            catch (Exception ex) { Console.WriteLine($"Failed to write out WDT: {ex.Message}"); return 5; }

            return 0;
        }

        // Legacy planning mode (kept for compatibility)
        var lkRoot = ResolveLkRoot(p);
        if (lkRoot is null) { Console.WriteLine("Missing required --lk-root <dir> and no default found"); return 2; }
        if (!p.TryGetValue("--out", out var outAlphaLegacy) || string.IsNullOrWhiteSpace(outAlphaLegacy))
        {
            Console.WriteLine("Missing required --out <alpha.wdt>");
            return 2;
        }

        var noCoordXform = p.ContainsKey("--no-coord-xform");
        var destRebucket = p.ContainsKey("--dest-rebucket");
        var emitMclq = p.ContainsKey("--emit-mclq");
        var scanMcse = p.ContainsKey("--scan-mcse");

        Console.WriteLine("[pack-alpha] \n" +
            $"lk-root={lkRoot}\n" +
            $"out={outAlphaLegacy}\n" +
            $"no-coord-xform={(noCoordXform ? 1 : 0)} dest-rebucket={(destRebucket ? 1 : 0)} emit-mclq={(emitMclq ? 1 : 0)} scan-mcse={(scanMcse ? 1 : 0)}");

        var mapNameLegacy = Path.GetFileNameWithoutExtension(outAlphaLegacy);
        if (string.IsNullOrWhiteSpace(mapNameLegacy))
        {
            Console.WriteLine("Unable to derive map name from --out <alpha.wdt>");
            return 4;
        }

        if (!Directory.Exists(lkRoot))
        {
            Console.WriteLine($"Directory not found: {lkRoot}");
            return 3;
        }

        var lkMapDirLegacy = Path.Combine(lkRoot, "world", "maps", mapNameLegacy);
        if (!Directory.Exists(lkMapDirLegacy))
        {
            Console.WriteLine($"LK map directory not found: {lkMapDirLegacy}");
            return 4;
        }

        var outAlphaDirLegacy = Path.GetDirectoryName(Path.GetFullPath(outAlphaLegacy)) ?? Directory.GetCurrentDirectory();
        Directory.CreateDirectory(outAlphaDirLegacy);

        var planCsvLegacy = new List<string>
        {
            "key,value",
            $"mapName,{mapNameLegacy}",
            $"lkRoot,{Path.GetFullPath(lkRoot)}",
            $"lkMapDir,{lkMapDirLegacy}",
            $"outAlpha,{Path.GetFullPath(outAlphaLegacy)}",
            $"noCoordXform,{(noCoordXform?1:0)}",
            $"destRebucket,{(destRebucket?1:0)}",
            $"emitMclq,{(emitMclq?1:0)}",
            $"scanMcse,{(scanMcse?1:0)}"
        };

        var adtListLegacy = new List<string> { "filename" };
        int adtCountLegacy = 0;
        foreach (var adt in Directory.EnumerateFiles(lkMapDirLegacy, "*.adt", SearchOption.TopDirectoryOnly))
        {
            var name = Path.GetFileName(adt);
            if (name is null) continue;
            adtListLegacy.Add(name);
            adtCountLegacy++;
        }
        planCsvLegacy.Add($"lkAdtCount,{adtCountLegacy}");

        var planCsvPathLegacy = Path.Combine(outAlphaDirLegacy, "pack_alpha_plan.csv");
        var listCsvPathLegacy = Path.Combine(outAlphaDirLegacy, "lk_adt_list.csv");
        try
        {
            File.WriteAllLines(planCsvPathLegacy, planCsvLegacy);
            File.WriteAllLines(listCsvPathLegacy, adtListLegacy);
            Console.WriteLine($"Wrote {planCsvPathLegacy}");
            Console.WriteLine($"Wrote {listCsvPathLegacy}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write planning CSVs: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static int TileDiff(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--original", out var original) || string.IsNullOrWhiteSpace(original))
        {
            Console.WriteLine("Missing required --original <alpha.wdt>");
            return 2;
        }
        if (!p.TryGetValue("--generated", out var generated) || string.IsNullOrWhiteSpace(generated))
        {
            Console.WriteLine("Missing required --generated <alpha.wdt>");
            return 2;
        }
        p.TryGetValue("--tile", out var tile);
        if (!File.Exists(original)) { Console.WriteLine($"File not found: {original}"); return 3; }
        if (!File.Exists(generated)) { Console.WriteLine($"File not found: {generated}"); return 3; }

        var outDir = GetOutDir(p);
        Console.WriteLine($"[tile-diff] original={original} generated={generated} tile={tile ?? "(all)"} out={outDir}");

        byte[] a, b;
        try { a = File.ReadAllBytes(original); } catch (Exception ex) { Console.WriteLine($"Read original failed: {ex.Message}"); return 4; }
        try { b = File.ReadAllBytes(generated); } catch (Exception ex) { Console.WriteLine($"Read generated failed: {ex.Message}"); return 4; }

        var keys = new[] { "MHDR", "MCIN", "MCNK", "MCLY", "MCRF", "MH2O", "MDDF", "MODF" };
        var lines = new List<string> { "feature,orig_forward,gen_forward,orig_reversed,gen_reversed,orig_first_fwd,gen_first_fwd,orig_first_rev,gen_first_rev" };

        foreach (var k in keys)
        {
            var kf = Encoding.ASCII.GetBytes(k);
            var kr = ReverseFourCcBytes(k);
            var (oa, oaf) = FindOccurrences(a, kf);
            var (ob, obf) = FindOccurrences(b, kf);
            var (ra, raf) = FindOccurrences(a, kr);
            var (rb, rbf) = FindOccurrences(b, kr);
            lines.Add(string.Join(',', new[]
            {
                k,
                oa.ToString(CultureInfo.InvariantCulture),
                ob.ToString(CultureInfo.InvariantCulture),
                ra.ToString(CultureInfo.InvariantCulture),
                rb.ToString(CultureInfo.InvariantCulture),
                oaf.ToString(CultureInfo.InvariantCulture),
                obf.ToString(CultureInfo.InvariantCulture),
                raf.ToString(CultureInfo.InvariantCulture),
                rbf.ToString(CultureInfo.InvariantCulture)
            }));
        }

        var csvPath = Path.Combine(outDir, "tile_diff.csv");
        try
        {
            File.WriteAllLines(csvPath, lines);
            Console.WriteLine($"Wrote {csvPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write CSV: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static int ListPlacements(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--file", out var path) || string.IsNullOrWhiteSpace(path))
        {
            Console.WriteLine("Missing required --file <wdt|adt|dir>");
            return 2;
        }
        var outDir = EnsureSubOut(p, "list_placements");

        var files = new List<string>();
        if (Directory.Exists(path))
        {
            foreach (var ext in new[] { "*.wdt", "*.adt" })
                files.AddRange(Directory.EnumerateFiles(path, ext, SearchOption.AllDirectories));
        }
        else if (File.Exists(path))
        {
            files.Add(path);
        }
        else
        {
            Console.WriteLine($"Path not found: {path}");
            return 3;
        }

        var header = "file,format,chunk,kind,offset,size_if_header,payload_path,payload_len,sample_hex";
        var lines = new List<string> { header };
        var payloadRoot = Path.Combine(outDir, "payloads");
        Directory.CreateDirectory(payloadRoot);
        foreach (var f in files)
        {
            byte[] data;
            try { data = File.ReadAllBytes(f); } catch (Exception ex) { Console.WriteLine($"Read failed: {f} -> {ex.Message}"); continue; }
            var format = GuessFormatFromBytes(data);
            foreach (var chunk in new[] { "MDDF", "MODF", "FDDM", "FDOM" })
            {
                foreach (var kind in new[] { "forward", "reversed" })
                {
                    var token = kind == "forward" ? Encoding.ASCII.GetBytes(chunk) : ReverseFourCcBytes(chunk);
                    foreach (var off in FindAllOccurrences(data, token))
                    {
                        int? size = TryReadSizeAfterHeader(data, off + 4);
                        var sample = HexSample(data, off, 64);
                        string payloadPath = "";
                        string payloadLen = "";
                        if (size is int s && s > 0 && off + 8 + s <= data.Length)
                        {
                            var baseName = Path.GetFileName(f) ?? "file";
                            var outName = $"{baseName}_{chunk}_{off}.bin";
                            var outPath = Path.Combine(payloadRoot, outName);
                            try
                            {
                                File.WriteAllBytes(outPath, new ReadOnlySpan<byte>(data, off + 8, s).ToArray());
                                payloadPath = outPath;
                                payloadLen = s.ToString(CultureInfo.InvariantCulture);
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"Failed to write payload for {baseName}:{chunk} at {off}: {ex.Message}");
                            }
                        }
                        lines.Add(string.Join(',', new[]
                        {
                            Csv(f), Csv(format), Csv(chunk), Csv(kind), off.ToString(CultureInfo.InvariantCulture),
                            size?.ToString(CultureInfo.InvariantCulture) ?? "",
                            Csv(payloadPath),
                            payloadLen,
                            Csv(sample)
                        }));
                    }
                }
            }
        }

        var csvPath = Path.Combine(outDir, "placements.csv");
        try { File.WriteAllLines(csvPath, lines); Console.WriteLine($"Wrote {csvPath}"); }
        catch (Exception ex) { Console.WriteLine($"Failed to write placements.csv: {ex.Message}"); return 4; }
        return 0;
    }

    private static int DumpMcnk(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--file", out var file) || string.IsNullOrWhiteSpace(file))
        {
            Console.WriteLine("Missing required --file <adt>");
            return 2;
        }
        if (!File.Exists(file)) { Console.WriteLine($"File not found: {file}"); return 3; }
        var outDir = EnsureSubOut(p, "dump_mcnk");

        byte[] data;
        try { data = File.ReadAllBytes(file); } catch (Exception ex) { Console.WriteLine($"Read failed: {ex.Message}"); return 4; }

        var mcinInfo = FindChunk(data, "MCIN");
        var mcnkList = new List<(int index, int offset, int size)>();
        if (mcinInfo.found && mcinInfo.size >= 16)
        {
            var baseOff = mcinInfo.offset + 8;
            var entryCount = mcinInfo.size / 16;
            for (int i = 0; i < entryCount; i++)
            {
                var o = ReadU32(data, baseOff + i * 16);
                var s = ReadU32(data, baseOff + i * 16 + 4);
                if (o > 0 && s > 0 && o + s <= data.Length)
                    mcnkList.Add((i, (int)o, (int)s));
            }
        }
        else
        {
            foreach (var off in FindAllOccurrences(data, Encoding.ASCII.GetBytes("MCNK")))
            {
                var size = TryReadSizeAfterHeader(data, off + 4) ?? 0;
                mcnkList.Add((mcnkList.Count, off, size));
            }
        }

        var mapCsv = new List<string> { "index,offset,size" };
        foreach (var m in mcnkList)
            mapCsv.Add(string.Join(',', m.index.ToString(CultureInfo.InvariantCulture), m.offset.ToString(CultureInfo.InvariantCulture), m.size.ToString(CultureInfo.InvariantCulture)));

        var hexdumpsDir = Path.Combine(outDir, "hexdumps");
        Directory.CreateDirectory(hexdumpsDir);

        var subscan = new List<string> { "index,MCLY,MCRF,MH2O,MDDF,MODF" };
        foreach (var m in mcnkList)
        {
            var sliceLen = m.size > 0 ? Math.Min(m.size + 8, data.Length - m.offset) : Math.Min(4096, data.Length - m.offset);
            var hex = HexDump(data, m.offset, sliceLen);
            var hexPath = Path.Combine(hexdumpsDir, $"mcnk_{m.index:D3}.hex");
            try { File.WriteAllText(hexPath, hex); } catch { }

            string Scan(string tag)
            {
                var rel = IndexOf(data, Encoding.ASCII.GetBytes(tag), m.offset, sliceLen);
                if (rel >= 0) return rel.ToString(CultureInfo.InvariantCulture);
                rel = IndexOf(data, ReverseFourCcBytes(tag), m.offset, sliceLen);
                if (rel >= 0) return rel.ToString(CultureInfo.InvariantCulture);
                return "";
            }

            subscan.Add(string.Join(',', m.index.ToString(CultureInfo.InvariantCulture), Scan("MCLY"), Scan("MCRF"), Scan("MH2O"), Scan("MDDF"), Scan("MODF")));
        }

        var mcinCsv = new List<string> { "offset,size" };
        if (mcinInfo.found)
            mcinCsv.Add(string.Join(',', mcinInfo.offset.ToString(CultureInfo.InvariantCulture), mcinInfo.size.ToString(CultureInfo.InvariantCulture)));

        try
        {
            File.WriteAllLines(Path.Combine(outDir, "mcin_index.csv"), mcinCsv);
            File.WriteAllLines(Path.Combine(outDir, "mcnk_map.csv"), mapCsv);
            File.WriteAllLines(Path.Combine(outDir, "subchunk_scan.csv"), subscan);
            Console.WriteLine($"Wrote {Path.Combine(outDir, "mcnk_map.csv")}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed writing dump-mcnk CSVs: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static int DumpAdt(string[] argv)
    {
        var p = ParseArgs(argv);
        if (!p.TryGetValue("--file", out var file) || string.IsNullOrWhiteSpace(file))
        {
            Console.WriteLine("Missing required --file <path>");
            return 2;
        }
        if (!File.Exists(file))
        {
            Console.WriteLine($"File not found: {file}");
            return 3;
        }

        var outDir = GetOutDir(p);
        Console.WriteLine($"[dump-adt] file={file} out={outDir}");

        byte[] data;
        try
        {
            data = File.ReadAllBytes(file);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to read file: {ex.Message}");
            return 4;
        }

        var keys = new[] { "MHDR", "MCIN", "MCNK", "MH2O", "MDDF", "MODF" };
        var lines = new List<string> { "feature,forward_count,reversed_count,first_forward_offset,first_reversed_offset" };
        foreach (var k in keys)
        {
            var (fCount, fFirst) = FindOccurrences(data, Encoding.ASCII.GetBytes(k));
            var (rCount, rFirst) = FindOccurrences(data, ReverseFourCcBytes(k));
            lines.Add(string.Join(',', new[]
            {
                k,
                fCount.ToString(CultureInfo.InvariantCulture),
                rCount.ToString(CultureInfo.InvariantCulture),
                fFirst.ToString(CultureInfo.InvariantCulture),
                rFirst.ToString(CultureInfo.InvariantCulture)
            }));
        }

        var csvPath = Path.Combine(outDir, "dump_adt.csv");
        try
        {
            File.WriteAllLines(csvPath, lines);
            Console.WriteLine($"Wrote {csvPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed to write CSV: {ex.Message}");
            return 5;
        }

        return 0;
    }

    private static Dictionary<string, string?> ParseArgs(string[] argv)
    {
        var map = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < argv.Length; i++)
        {
            var a = argv[i];
            if (a.StartsWith("--", StringComparison.Ordinal))
            {
                if (i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    map[a] = argv[++i];
                }
                else
                {
                    map[a] = null;
                }
            }
        }
        return map;
    }

    private static string GetOutDir(Dictionary<string, string?> p)
    {
        if (p.TryGetValue("--out", out var provided) && !string.IsNullOrWhiteSpace(provided))
        {
            Directory.CreateDirectory(provided);
            return Path.GetFullPath(provided);
        }
        var session = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        var dir = Path.Combine("WdtInspector_outputs", $"session_{session}");
        Directory.CreateDirectory(dir);
        return Path.GetFullPath(dir);
    }

    private static string EnsureSubOut(Dictionary<string, string?> p, string subFolder)
    {
        var root = GetOutDir(p);
        var dir = Path.Combine(root, subFolder);
        Directory.CreateDirectory(dir);
        return Path.GetFullPath(dir);
    }

    private static (int count, int firstOffset) FindOccurrences(byte[] haystack, byte[] needle)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length)
            return (0, -1);

        var count = 0;
        var first = -1;
        for (int i = 0; i <= haystack.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (haystack[i + j] != needle[j])
                {
                    match = false;
                    break;
                }
            }
            if (match)
            {
                if (first == -1) first = i;
                count++;
                i += needle.Length - 1;
            }
        }
        return (count, first);
    }

    private static byte[] ReverseFourCcBytes(string fourcc)
    {
        var b = Encoding.ASCII.GetBytes(fourcc);
        Array.Reverse(b);
        return b;
    }

    private static List<int> FindAllOccurrences(byte[] data, byte[] needle)
    {
        var list = new List<int>();
        if (needle.Length == 0 || data.Length < needle.Length) return list;
        for (int i = 0; i <= data.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++)
            {
                if (data[i + j] != needle[j]) { match = false; break; }
            }
            if (match)
            {
                list.Add(i);
                i += needle.Length - 1;
            }
        }
        return list;
    }

    private static int? TryReadSizeAfterHeader(byte[] data, int sizePos)
    {
        if (sizePos + 4 > data.Length) return null;
        var s = (int)ReadU32(data, sizePos);
        if (s < 0 || s > data.Length) return null;
        return s;
    }

    private static string GuessFormatFromBytes(byte[] data)
    {
        var hasMhdr = IndexOf(data, Encoding.ASCII.GetBytes("MHDR")) >= 0 || IndexOf(data, ReverseFourCcBytes("MHDR")) >= 0;
        var hasMver = IndexOf(data, Encoding.ASCII.GetBytes("MVER")) >= 0 || IndexOf(data, ReverseFourCcBytes("MVER")) >= 0;
        if (hasMhdr || hasMver) return "LK";
        return "AlphaOrUnknown";
    }

    private static string HexSample(byte[] data, int offset, int length)
    {
        var len = Math.Max(0, Math.Min(length, data.Length - offset));
        var sb = new StringBuilder(len * 2);
        for (int i = 0; i < len; i++) sb.Append(data[offset + i].ToString("X2", CultureInfo.InvariantCulture));
        return sb.ToString();
    }

    private static string HexDump(byte[] data, int offset, int length)
    {
        var len = Math.Max(0, Math.Min(length, data.Length - offset));
        var sb = new StringBuilder(len * 3);
        int lineAddr = offset;
        for (int i = 0; i < len; i += 16)
        {
            int chunk = Math.Min(16, len - i);
            sb.Append(lineAddr.ToString("X8", CultureInfo.InvariantCulture));
            sb.Append(':');
            for (int j = 0; j < chunk; j++)
            {
                sb.Append(' ');
                sb.Append(data[offset + i + j].ToString("X2", CultureInfo.InvariantCulture));
            }
            sb.AppendLine();
            lineAddr += chunk;
        }
        return sb.ToString();
    }

    private static int IndexOf(byte[] data, byte[] needle, int start, int count)
    {
        var end = Math.Min(data.Length, start + Math.Max(0, count));
        for (int i = start; i <= end - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++) { if (data[i + j] != needle[j]) { match = false; break; } }
            if (match) return i - start;
        }
        return -1;
    }

    private static int IndexOf(byte[] data, byte[] needle)
    {
        for (int i = 0; i <= data.Length - needle.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Length; j++) { if (data[i + j] != needle[j]) { match = false; break; } }
            if (match) return i;
        }
        return -1;
    }

    private static (bool found, int offset, int size) FindChunk(byte[] data, string fourcc)
    {
        var fwd = Encoding.ASCII.GetBytes(fourcc);
        var idx = IndexOf(data, fwd);
        if (idx < 0)
        {
            var rev = ReverseFourCcBytes(fourcc);
            idx = IndexOf(data, rev);
        }
        if (idx < 0 || idx + 8 > data.Length) return (false, -1, 0);
        var size = (int)ReadU32(data, idx + 4);
        if (size < 0 || size > data.Length) size = 0;
        return (true, idx, size);
    }

    private static uint ReadU32(byte[] data, int pos)
    {
        if (pos + 4 > data.Length) return 0;
        return (uint)(data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16) | (data[pos + 3] << 24));
    }

    private static List<(int offset, int size, byte[] prefix)> MapMcnk(byte[] data)
    {
        var list = new List<(int offset, int size, byte[] prefix)>();

        // Prefer MCIN if present
        var mcin = FindChunk(data, "MCIN");
        if (mcin.found && mcin.size >= 16)
        {
            int entries = mcin.size / 16;
            int baseOff = mcin.offset + 8;
            for (int i = 0; i < entries; i++)
            {
                var o = (int)ReadU32(data, baseOff + i * 16);
                var s = (int)ReadU32(data, baseOff + i * 16 + 4);
                if (o <= 0 || s <= 0) continue;
                int total = s;
                // MCIN size is usually payload size (without 8), but ADT headers are MCNK+size+payload.
                // Most LK MCIN encodes payload size; ensure we include header safely if present.
                int start = o;
                int end = Math.Min(data.Length, start + total + 8);
                int len = Math.Max(0, end - start);
                if (len <= 0) continue;
                int prefLen = Math.Min(32, len);
                var pref = new byte[prefLen];
                Buffer.BlockCopy(data, start, pref, 0, prefLen);
                list.Add((start, len, pref));
            }
            return list;
        }

        // Fallback: scan for MCNK (forward and reversed)
        var seen = new HashSet<int>();
        foreach (var off in FindAllOccurrences(data, Encoding.ASCII.GetBytes("MCNK")))
            if (seen.Add(off))
            {
                var sz = TryReadSizeAfterHeader(data, off + 4);
                if (sz is int s && s > 0)
                {
                    int total = Math.Min(data.Length - off, s + 8);
                    int prefLen = Math.Min(32, total);
                    var pref = new byte[prefLen];
                    Buffer.BlockCopy(data, off, pref, 0, prefLen);
                    list.Add((off, total, pref));
                }
            }
        foreach (var off in FindAllOccurrences(data, ReverseFourCcBytes("MCNK")))
            if (seen.Add(off))
            {
                var sz = TryReadSizeAfterHeader(data, off + 4);
                if (sz is int s && s > 0)
                {
                    int total = Math.Min(data.Length - off, s + 8);
                    int prefLen = Math.Min(32, total);
                    var pref = new byte[prefLen];
                    Buffer.BlockCopy(data, off, pref, 0, prefLen);
                    list.Add((off, total, pref));
                }
            }
        return list;
    }

    private static string Csv(string s)
    {
        if (s.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0)
            return "\"" + s.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
        return s;
    }

    private static string? ResolveLkRoot(Dictionary<string, string?> p)
    {
        if (p.TryGetValue("--lk-root", out var lk) && !string.IsNullOrWhiteSpace(lk)) return lk;
        var def = @"G:\\WoW\\WoWArchive-0.X-3.X\\Mount\\3.X_Retail_Windows_enUS_3.3.5.12340\\World of Warcraft";
        return Directory.Exists(def) ? def : null;
    }

    private static byte[] BuildLkAdtFromAlpha(byte[] alphaAdt)
    {
        // Collect MCNK blocks from the alpha ADT by scanning FourCC headers and sizes
        var mcnkOffsets = new List<int>();
        var seen = new HashSet<int>();
        foreach (var off in FindAllOccurrences(alphaAdt, Encoding.ASCII.GetBytes("MCNK")))
            if (seen.Add(off)) mcnkOffsets.Add(off);
        foreach (var off in FindAllOccurrences(alphaAdt, ReverseFourCcBytes("MCNK")))
            if (seen.Add(off)) mcnkOffsets.Add(off);
        var mcnkBlocks = new List<byte[]>();
        foreach (var off in mcnkOffsets)
        {
            var size = TryReadSizeAfterHeader(alphaAdt, off + 4);
            if (size == null) continue;
            var total = 8 + size.Value;
            if (off + total <= alphaAdt.Length)
            {
                var block = new byte[total];
                Buffer.BlockCopy(alphaAdt, off, block, 0, total);
                mcnkBlocks.Add(block);
            }
        }

        // Build minimal LK-like ADT: MVER, MHDR (0x40), MCIN (4096), then all MCNK blocks
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        void WriteChunk(string fourcc, ReadOnlySpan<byte> payload)
        {
            var tag = Encoding.ASCII.GetBytes(fourcc);
            bw.Write(tag);
            bw.Write(payload.Length);
            bw.Write(payload);
            Pad4(bw);
        }

        // MVER (version 18)
        Span<byte> mver = stackalloc byte[4];
        BitConverter.TryWriteBytes(mver, 18);
        WriteChunk("MVER", mver);

        // MHDR (0x40 zeros)
        var mhdr = new byte[0x40];
        WriteChunk("MHDR", mhdr);

        // Prepare MCIN payload
        var mcinPayload = new byte[16 * 256];
        int baseOffset = (int)ms.Position + 8 + mcinPayload.Length; // position after we write MCIN chunk header+payload
        // Since Pad4 keeps alignment and lengths are multiples of 4, baseOffset is correct after MCIN

        int current = baseOffset;
        int count = Math.Min(mcnkBlocks.Count, 256);
        for (int i = 0; i < count; i++)
        {
            var block = mcnkBlocks[i];
            // entry: offset(u32), size(u32), flags(u32)=0, asyncId(u32)=0
            BitConverter.TryWriteBytes(mcinPayload.AsSpan(i * 16 + 0), current);
            BitConverter.TryWriteBytes(mcinPayload.AsSpan(i * 16 + 4), block.Length);
            // flags/async left zero
            // advance with 4-byte alignment
            var padded = (block.Length + 3) & ~3;
            current += padded;
        }

        // MCIN
        WriteChunk("MCIN", mcinPayload);

        // MCNK blocks
        foreach (var block in mcnkBlocks)
        {
            // Blocks already include their own header (MCNK + size)
            bw.Write(block);
            Pad4(bw);
        }

        bw.Flush();
        return ms.ToArray();
    }

    private static void Pad4(BinaryWriter bw)
    {
        var len = (int)bw.BaseStream.Position;
        int pad = ((len + 3) & ~3) - len;
        for (int i = 0; i < pad; i++) bw.Write((byte)0);
    }

    private static List<(int index, int mhdrAbs, int firstMcnk)> ParseAlphaWdtTiles(byte[] wdt)
    {
        // Scan top-level chunks (on-disk tokens are reversed); we look for NIAM (MAIN) and optionally DHPM (MPHD)
        var tops = new Dictionary<string, (int off, int size)>(StringComparer.Ordinal);
        for (int i = 0; i + 8 <= wdt.Length;)
        {
            string tok = ReadFcc(wdt, i);
            int size = (int)ReadU32Le(wdt, i + 4);
            if (!string.IsNullOrWhiteSpace(tok)) tops[tok] = (i, size);
            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break;
            i = next;
        }

        if (!tops.TryGetValue("NIAM", out var main)) return new();
        int mainData = main.off + 8;
        var list = new List<(int index, int mhdrAbs, int firstMcnk)>();
        for (int i = 0; i < 4096; i++)
        {
            int pos = mainData + i * 16;
            if (pos + 16 > wdt.Length) break;
            int mhdrAbs = (int)ReadU32Le(wdt, pos + 0);
            int sizeToFirstMcnk = (int)ReadU32Le(wdt, pos + 4);
            if (mhdrAbs <= 0) continue;

            int mhdrDataStart = mhdrAbs + 8;
            if (mhdrDataStart + 64 > wdt.Length) continue;

            // Read MHDR fields (Alpha layout; offsets to MCIN/MTEX/MDDF/MODF payloads)
            int offsInfo = (int)ReadU32Le(wdt, mhdrDataStart + 0x00);
            int offsTex  = (int)ReadU32Le(wdt, mhdrDataStart + 0x04);
            int sizeTex  = (int)ReadU32Le(wdt, mhdrDataStart + 0x08);
            int offsDoo  = (int)ReadU32Le(wdt, mhdrDataStart + 0x0C);
            int sizeDoo  = (int)ReadU32Le(wdt, mhdrDataStart + 0x10);
            int offsMob  = (int)ReadU32Le(wdt, mhdrDataStart + 0x14);
            int sizeMob  = (int)ReadU32Le(wdt, mhdrDataStart + 0x18);

            int mcInEnd = mhdrDataStart + offsInfo + (8 + 4096);
            int mtExEnd = mhdrDataStart + offsTex + (8 + Math.Max(0, sizeTex));
            int mdDfEnd = mhdrDataStart + offsDoo + (8 + Math.Max(0, sizeDoo));
            int moDfEnd = mhdrDataStart + offsMob + (8 + Math.Max(0, sizeMob));
            int firstMcnk = Math.Max(Math.Max(mcInEnd, mtExEnd), Math.Max(mdDfEnd, moDfEnd));

            if (firstMcnk + 8 <= wdt.Length)
            {
                // Sanity: token should be MCNK reversed (KNCM) or forward
                var tok = ReadFcc(wdt, firstMcnk);
                // Accept anything; downstream extractor will validate
                list.Add((i, mhdrAbs, firstMcnk));
            }
        }

        return list;
    }

    private static byte[] BuildLkAdtFromAlphaWdtTile(byte[] wdt, int mhdrAbs)
    {
        // Parse MHDR-relative offsets
        int mhdrDataStart = mhdrAbs + 8;
        if (mhdrDataStart + 64 > wdt.Length) return Array.Empty<byte>();
        int offsInfo = (int)ReadU32Le(wdt, mhdrDataStart + 0x00);
        int offsTex  = (int)ReadU32Le(wdt, mhdrDataStart + 0x04);
        int sizeTex  = (int)ReadU32Le(wdt, mhdrDataStart + 0x08);
        int offsDoo  = (int)ReadU32Le(wdt, mhdrDataStart + 0x0C);
        int sizeDoo  = (int)ReadU32Le(wdt, mhdrDataStart + 0x10);
        int offsMob  = (int)ReadU32Le(wdt, mhdrDataStart + 0x14);
        int sizeMob  = (int)ReadU32Le(wdt, mhdrDataStart + 0x18);

        int mcinAbs = mhdrAbs + 8 + offsInfo;
        if (mcinAbs + 8 > wdt.Length) return Array.Empty<byte>();
        var mcinTok = ReadFcc(wdt, mcinAbs);
        int mcinSize = (int)ReadU32Le(wdt, mcinAbs + 4);
        if (mcinSize < 0 || mcinAbs + 8 + mcinSize > wdt.Length) return Array.Empty<byte>();

        // Collect MCNK blocks using MCIN absolute offsets
        var blocks = new List<byte[]>();
        int entries = mcinSize / 16;
        int mcinData = mcinAbs + 8;
        for (int i = 0; i < entries && i < 256; i++)
        {
            int offVal = (int)ReadU32Le(wdt, mcinData + i * 16 + 0);
            int sz     = (int)ReadU32Le(wdt, mcinData + i * 16 + 4);
            if (offVal <= 0 || sz <= 0) continue;
            if (offVal + 8 + sz > wdt.Length) continue;
            var block = new byte[8 + sz];
            Buffer.BlockCopy(wdt, offVal, block, 0, 8 + sz);
            blocks.Add(block);
        }

        // Extract top-level payloads from WDT tile
        byte[] mtexPayload = Array.Empty<byte>();
        if (offsTex > 0 && sizeTex > 0)
        {
            int abs = mhdrDataStart + offsTex + 8;
            if (abs + sizeTex <= wdt.Length)
            {
                mtexPayload = new byte[sizeTex];
                Buffer.BlockCopy(wdt, abs, mtexPayload, 0, sizeTex);
            }
        }

        byte[] mddfPayload = Array.Empty<byte>();
        if (offsDoo > 0 && sizeDoo > 0)
        {
            int abs = mhdrDataStart + offsDoo + 8;
            if (abs + sizeDoo <= wdt.Length)
            {
                mddfPayload = new byte[sizeDoo];
                Buffer.BlockCopy(wdt, abs, mddfPayload, 0, sizeDoo);
            }
        }

        byte[] modfPayload = Array.Empty<byte>();
        if (offsMob > 0 && sizeMob > 0)
        {
            int abs = mhdrDataStart + offsMob + 8;
            if (abs + sizeMob <= wdt.Length)
            {
                modfPayload = new byte[sizeMob];
                Buffer.BlockCopy(wdt, abs, modfPayload, 0, sizeMob);
            }
        }

        // Build LK-like ADT with MHDR+MCIN placeholders and backpatch
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, Encoding.ASCII, leaveOpen: true);

        void WriteChunk(string fourcc, ReadOnlySpan<byte> payload)
        {
            var tag = Encoding.ASCII.GetBytes(fourcc);
            bw.Write(tag);
            bw.Write(payload.Length);
            bw.Write(payload);
            Pad4(bw);
        }

        // MVER(18)
        Span<byte> mver = stackalloc byte[4];
        BitConverter.TryWriteBytes(mver, 18);
        WriteChunk("MVER", mver);
        // MHDR placeholder
        long mhdrChunkStart = ms.Position;
        bw.Write(Encoding.ASCII.GetBytes("MHDR"));
        bw.Write(0x40); // size
        long mhdrPayloadStart = ms.Position;
        var mhdrBuf = new byte[0x40];
        bw.Write(mhdrBuf);
        Pad4(bw);

        // MCIN placeholder (4096)
        long mcinChunkStart = ms.Position;
        bw.Write(Encoding.ASCII.GetBytes("MCIN"));
        bw.Write(16 * 256); // size
        long mcinPayloadStart = ms.Position;
        var mcinOut = new byte[16 * 256];
        bw.Write(mcinOut);
        Pad4(bw);

        // MTEX
        long mtexChunkStart = -1;
        if (mtexPayload.Length > 0)
        {
            mtexChunkStart = ms.Position;
            WriteChunk("MTEX", mtexPayload);
        }

        // MDDF
        long mddfChunkStart = -1;
        if (mddfPayload.Length > 0)
        {
            mddfChunkStart = ms.Position;
            WriteChunk("MDDF", mddfPayload);
        }

        // MODF
        long modfChunkStart = -1;
        if (modfPayload.Length > 0)
        {
            modfChunkStart = ms.Position;
            WriteChunk("MODF", modfPayload);
        }

        // MCNK blocks area; record offsets for MCIN entries
        var mcnkOffsets = new List<(int off, int size)>();
        foreach (var block in blocks)
        {
            int off = (int)ms.Position;
            bw.Write(block);
            Pad4(bw);
            mcnkOffsets.Add((off, block.Length));
        }

        // Backpatch MCIN entries (offset=size are ADT-relative file offsets)
        for (int i = 0; i < 256; i++)
        {
            int entryOff = (int)mcinPayloadStart + i * 16;
            if (i < mcnkOffsets.Count)
            {
                var e = mcnkOffsets[i];
                // write offset and size
                bw.BaseStream.Seek(entryOff, SeekOrigin.Begin);
                bw.Write(e.off);
                bw.Write(e.size);
                // flags/async left zero
            }
        }

        // Backpatch MHDR offsets (relative to MHDR payload start)
        bw.BaseStream.Seek(mhdrPayloadStart + 0x00, SeekOrigin.Begin); // OffsInfo
        bw.Write((int)(mcinChunkStart - mhdrPayloadStart));
        bw.BaseStream.Seek(mhdrPayloadStart + 0x04, SeekOrigin.Begin); // OffsTex
        bw.Write(mtexChunkStart > 0 ? (int)(mtexChunkStart - mhdrPayloadStart) : 0);
        bw.BaseStream.Seek(mhdrPayloadStart + 0x08, SeekOrigin.Begin); // SizeTex
        bw.Write(mtexPayload.Length);
        bw.BaseStream.Seek(mhdrPayloadStart + 0x0C, SeekOrigin.Begin); // OffsDoo
        bw.Write(mddfChunkStart > 0 ? (int)(mddfChunkStart - mhdrPayloadStart) : 0);
        bw.BaseStream.Seek(mhdrPayloadStart + 0x10, SeekOrigin.Begin); // SizeDoo
        bw.Write(mddfPayload.Length);
        bw.BaseStream.Seek(mhdrPayloadStart + 0x14, SeekOrigin.Begin); // OffsMob
        bw.Write(modfChunkStart > 0 ? (int)(modfChunkStart - mhdrPayloadStart) : 0);
        bw.BaseStream.Seek(mhdrPayloadStart + 0x18, SeekOrigin.Begin); // SizeMob
        bw.Write(modfPayload.Length);

        // Return buffer
        bw.BaseStream.Seek(0, SeekOrigin.End);
        bw.Flush();
        return ms.ToArray();
    }

    private static uint ReadU32Le(byte[] data, int pos)
    {
        if (pos + 4 > data.Length) return 0;
        return (uint)(data[pos] | (data[pos + 1] << 8) | (data[pos + 2] << 16) | (data[pos + 3] << 24));
    }

    private static string ReadFcc(byte[] data, int pos)
    {
        if (pos + 4 > data.Length) return string.Empty;
        return Encoding.ASCII.GetString(data, pos, 4);
    }

    /// <summary>
    /// Split a null-terminated string table into a list of strings.
    /// </summary>
    private static List<string> SplitNullStringsList(byte[] payload)
    {
        var list = new List<string>();
        if (payload.Length == 0) return list;
        int start = 0;
        for (int i = 0; i < payload.Length; i++)
        {
            if (payload[i] == 0)
            {
                if (i > start)
                {
                    var s = Encoding.ASCII.GetString(payload, start, i - start);
                    list.Add(s);
                }
                start = i + 1;
            }
        }
        if (start < payload.Length)
        {
            var s = Encoding.ASCII.GetString(payload, start, payload.Length - start);
            if (s.Length > 0) list.Add(s);
        }
        return list;
    }

    /// <summary>
    /// Remap MDDF indices from per-ADT (LK MMDX) to global (Alpha MDNM).
    /// Entry size: 36 bytes, index at offset 0.
    /// </summary>
    private static byte[] RemapMddfIndicesForAlpha(byte[] mddfPayload, List<string> perAdtNames, List<string> globalNames)
    {
        const int entrySize = 36;
        var result = (byte[])mddfPayload.Clone();

        // Build lookup: globalName -> globalIndex
        var globalLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < globalNames.Count; i++)
            globalLookup[globalNames[i]] = i;

        for (int start = 0; start + entrySize <= result.Length; start += entrySize)
        {
            int perAdtIndex = BitConverter.ToInt32(result, start);
            if (perAdtIndex < 0 || perAdtIndex >= perAdtNames.Count)
                continue; // invalid index, leave as-is

            var name = perAdtNames[perAdtIndex];
            if (globalLookup.TryGetValue(name, out var globalIndex))
            {
                var bytes = BitConverter.GetBytes(globalIndex);
                Buffer.BlockCopy(bytes, 0, result, start, 4);
            }
        }
        return result;
    }

    /// <summary>
    /// Remap MODF indices from per-ADT (LK MWMO) to global (Alpha MONM).
    /// Entry size: 64 bytes, index at offset 0.
    /// </summary>
    private static byte[] RemapModfIndicesForAlpha(byte[] modfPayload, List<string> perAdtNames, List<string> globalNames)
    {
        const int entrySize = 64;
        var result = (byte[])modfPayload.Clone();

        // Build lookup: globalName -> globalIndex
        var globalLookup = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < globalNames.Count; i++)
            globalLookup[globalNames[i]] = i;

        for (int start = 0; start + entrySize <= result.Length; start += entrySize)
        {
            int perAdtIndex = BitConverter.ToInt32(result, start);
            if (perAdtIndex < 0 || perAdtIndex >= perAdtNames.Count)
                continue; // invalid index, leave as-is

            var name = perAdtNames[perAdtIndex];
            if (globalLookup.TryGetValue(name, out var globalIndex))
            {
                var bytes = BitConverter.GetBytes(globalIndex);
                Buffer.BlockCopy(bytes, 0, result, start, 4);
            }
        }
        return result;
    }
}
