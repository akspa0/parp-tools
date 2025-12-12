using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace WoWRollback.PM4Module;

/// <summary>
/// Patches existing ADT files with new MODF/MDDF placement data from PM4.
/// Reads split ADT files (root + _obj0 + _tex0), patches object chunks,
/// fixes MCIN/MHDR offsets, and writes merged 3.3.5 monolithic ADT.
/// </summary>
public sealed class AdtPatcher
{
    // Optional minimap MCCV data for fallback when no existing MCCV
    // Indexed by MCNK index (0-255), each entry is 580 bytes (145 vertices * 4 bytes BGRA)
    private byte[][]? _minimapMccvData;
    private int _currentMcnkIndex;

    /// <summary>
    /// Represents a parsed chunk that can be modified.
    /// </summary>
    public class AdtChunk
    {
        public string Signature { get; set; } = "";  // 4-char reversed sig as read from disk
        public byte[] Data { get; set; } = Array.Empty<byte>();
        
        public int TotalSize => 8 + Data.Length;  // header + data
        
        /// <summary>Human-readable signature (unreversed)</summary>
        public string ReadableSig => new string(Signature.Reverse().ToArray());
    }

    /// <summary>
    /// Parsed ADT file structure.
    /// </summary>
    public class ParsedAdt
    {
        public List<AdtChunk> Chunks { get; } = new();
        
        public AdtChunk? FindChunk(string reversedSig) => 
            Chunks.Find(c => c.Signature == reversedSig);
        
        public List<AdtChunk> FindAllChunks(string reversedSig) =>
            Chunks.FindAll(c => c.Signature == reversedSig);
        
        public int ReplaceChunk(string reversedSig, byte[] newData)
        {
            var chunk = FindChunk(reversedSig);
            if (chunk != null)
            {
                chunk.Data = newData;
                return 1;
            }
            return 0;
        }
        
        public void InsertChunkAfter(string afterSig, AdtChunk newChunk)
        {
            int idx = Chunks.FindIndex(c => c.Signature == afterSig);
            if (idx >= 0)
                Chunks.Insert(idx + 1, newChunk);
            else
                Chunks.Add(newChunk);
        }
    }

    // Readable signatures (Internal use)
    // On disk they are reversed (little-endian uint read as string), but we reverse them on load.
    private const string SIG_MVER = "MVER";
    private const string SIG_MHDR = "MHDR";
    private const string SIG_MCIN = "MCIN";
    private const string SIG_MTEX = "MTEX";
    private const string SIG_MMDX = "MMDX";
    private const string SIG_MMID = "MMID";
    private const string SIG_MWMO = "MWMO";
    private const string SIG_MWID = "MWID";
    private const string SIG_MDDF = "MDDF";
    private const string SIG_MODF = "MODF";
    private const string SIG_MCNK = "MCNK";
    private const string SIG_MH2O = "MH2O";
    private const string SIG_MFBO = "MFBO"; // Added constant for consistency

    // Subchunk signatures
    private const string SUB_MCVT = "MCVT";
    private const string SUB_MCCV = "MCCV";
    private const string SUB_MCNR = "MCNR";
    private const string SUB_MCLY = "MCLY";
    private const string SUB_MCRF = "MCRF";
    private const string SUB_MCAL = "MCAL";
    private const string SUB_MCSH = "MCSH";
    private const string SUB_MCRD = "MCRD";
    private const string SUB_MCRW = "MCRW";
    private const string SUB_MCSE = "MCSE";
    private const string SUB_MCLQ = "MCLQ";

    /// <summary>
    /// Normalize paths in a chunk containing null-terminated path strings.
    /// Converts backslashes to forward slashes and lowercases all characters.
    /// WoW client expects lowercase paths with forward slashes.
    /// </summary>
    private static byte[] NormalizePathChunk(byte[] data)
    {
        if (data == null || data.Length == 0) return data;
        
        var result = new byte[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            byte b = data[i];
            // Convert backslash to forward slash
            if (b == (byte)'\\')
                result[i] = (byte)'/';
            // Convert uppercase to lowercase (A-Z -> a-z)
            else if (b >= (byte)'A' && b <= (byte)'Z')
                result[i] = (byte)(b + 32);
            else
                result[i] = b;
        }
        return result;
    }

    /// <summary>
    /// Helper to reverse a string (e.g. "REVM" -> "MVER" or vice versa)
    /// </summary>
    private static string ReverseSig(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        char[] charArray = s.ToCharArray();
        Array.Reverse(charArray);
        return new string(charArray);
    }

    /// <summary>
    /// Parse an ADT file into mutable chunks.
    /// </summary>
    public ParsedAdt ParseAdt(byte[] data)
    {
        var result = new ParsedAdt();
        int pos = 0;

        while (pos < data.Length - 8)
        {
            // Read 4-byte signature from disk (e.g. "REVM")
            string rawSig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);

            if (size < 0 || pos + 8 + size > data.Length)
            {
                Console.WriteLine($"[WARN] Invalid chunk at {pos}: sig={rawSig}, size={size}");
                break;
            }

            var chunkData = new byte[size];
            Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);

            // Reverse signature to Readable format (e.g. "REVM" -> "MVER")
            string readableSig = ReverseSig(rawSig);
            
            result.Chunks.Add(new AdtChunk
            {
                Signature = readableSig,
                Data = chunkData
            });

            pos += 8 + size;
        }

        return result;
    }

    /// <summary>
    /// Parse split ADT files and merge into a single ParsedAdt with correct chunk ordering.
    /// 3.3.5 monolithic ADT order: MVER, MHDR, MCIN, MTEX, MMDX, MMID, MWMO, MWID, MDDF, MODF, MH2O, MCNK x256, MFBO
    /// </summary>
    public ParsedAdt ParseSplitAdt(string rootPath, string? obj0Path = null, string? tex0Path = null)
    {
        // ... (File loading logic remains same) ...
        if (!File.Exists(rootPath))
            throw new FileNotFoundException($"Root ADT not found: {rootPath}");

        var root = ParseAdt(File.ReadAllBytes(rootPath));
        ParsedAdt? obj0 = null;
        ParsedAdt? tex0 = null;

        if (obj0Path != null && File.Exists(obj0Path))
        {
            obj0 = ParseAdt(File.ReadAllBytes(obj0Path));
            Console.WriteLine($"[INFO] Loaded _obj0: {obj0.Chunks.Count} chunks");
        }
        
        if (tex0Path != null && File.Exists(tex0Path))
        {
            tex0 = ParseAdt(File.ReadAllBytes(tex0Path));
            Console.WriteLine($"[INFO] Loaded _tex0: {tex0.Chunks.Count} chunks");
        }

        // Build merged ADT with correct chunk ordering
        var merged = new ParsedAdt();

        // Helper to select chunk with logging
        AdtChunk? SelectChunk(string sig, string name, bool preferTex0 = false, bool preferObj0 = false)
        {
            AdtChunk? c = null;
            string source = "none";

            if (preferTex0 && tex0 != null) { c = tex0.FindChunk(sig); if (c != null) source = "tex0"; }
            if (c == null && preferObj0 && obj0 != null) { c = obj0.FindChunk(sig); if (c != null) source = "obj0"; }
            if (c == null) { c = root.FindChunk(sig); if (c != null) source = "root"; }

            if (c != null && c.Data.Length > 0)
                Console.WriteLine($"  {name}: {c.Data.Length} bytes (from {source})");
            
            return c;
        }

        // 1. MVER (from root)
        var mver = root.FindChunk(SIG_MVER);
        if (mver != null) merged.Chunks.Add(mver);

        // 2. MHDR (from root)
        var mhdr = root.FindChunk(SIG_MHDR);
        if (mhdr != null) merged.Chunks.Add(mhdr);

        // 3. MCIN - always include
        var mcin = root.FindChunk(SIG_MCIN);
        if (mcin != null)
            merged.Chunks.Add(mcin);
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MCIN, Data = new byte[256 * 16] });

        // 4. MTEX (from tex0 first, fallback to root)
        var mtex = SelectChunk(SIG_MTEX, "MTEX", preferTex0: true);
        if (mtex != null && mtex.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MTEX, Data = NormalizePathChunk(mtex.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MTEX, Data = Array.Empty<byte>() });

        // 5. MMDX (from obj0 first, fallback to root)
        var mmdx = SelectChunk(SIG_MMDX, "MMDX", preferObj0: true);
        if (mmdx != null && mmdx.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMDX, Data = NormalizePathChunk(mmdx.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMDX, Data = Array.Empty<byte>() });

        // 6. MMID (from obj0 first, fallback to root)
        var mmid = SelectChunk(SIG_MMID, "MMID", preferObj0: true);
        if (mmid != null && mmid.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMID, Data = mmid.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMID, Data = Array.Empty<byte>() });

        // 7. MWMO (from obj0 first, fallback to root)
        var mwmo = SelectChunk(SIG_MWMO, "MWMO", preferObj0: true);
        if (mwmo != null && mwmo.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWMO, Data = NormalizePathChunk(mwmo.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWMO, Data = Array.Empty<byte>() });

        // 8. MWID (from obj0 first, fallback to root)
        var mwid = SelectChunk(SIG_MWID, "MWID", preferObj0: true);
        if (mwid != null && mwid.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWID, Data = mwid.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWID, Data = Array.Empty<byte>() });

        // 9. MDDF (from obj0 first, fallback to root)
        var mddf = SelectChunk(SIG_MDDF, "MDDF", preferObj0: true);
        if (mddf != null && mddf.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MDDF, Data = mddf.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MDDF, Data = Array.Empty<byte>() });

        // 10. MODF (from obj0 first, fallback to root)
        var modf = SelectChunk(SIG_MODF, "MODF", preferObj0: true);
        if (modf != null && modf.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MODF, Data = modf.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MODF, Data = Array.Empty<byte>() });

        // 11. MH2O (from root)
        var mh2o = root.FindChunk(SIG_MH2O);
        if (mh2o != null && mh2o.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MH2O, Data = mh2o.Data });

        // 12. MCNK x 256
        var rootMcnks = root.Chunks.Where(c => c.Signature == SIG_MCNK).ToList();
        var tex0Mcnks = tex0?.Chunks.Where(c => c.Signature == SIG_MCNK).ToList() ?? new List<AdtChunk>();
        var obj0Mcnks = obj0?.Chunks.Where(c => c.Signature == SIG_MCNK).ToList() ?? new List<AdtChunk>();

        Console.WriteLine($"[INFO] MCNK counts - root: {rootMcnks.Count}, tex0: {tex0Mcnks.Count}, obj0: {obj0Mcnks.Count}");

        for (int i = 0; i < rootMcnks.Count; i++)
        {
            _currentMcnkIndex = i; // Track for minimap MCCV fallback
            
            var rootMcnk = rootMcnks[i];
            var tex0Mcnk = i < tex0Mcnks.Count ? tex0Mcnks[i] : null;
            var obj0Mcnk = i < obj0Mcnks.Count ? obj0Mcnks[i] : null;

            var mergedMcnk = MergeMcnkChunks(rootMcnk, tex0Mcnk, obj0Mcnk);
            merged.Chunks.Add(mergedMcnk);
        }

        // 13. MFBO
        var mfbo = root.FindChunk(SIG_MFBO);
        if (mfbo != null && mfbo.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MFBO, Data = mfbo.Data });

        return merged;
    }

    /// <summary>
    /// Parse MCNK data into subchunks.
    /// Root MCNK has 128-byte header before subchunks.
    /// Tex0/Obj0 MCNKs have NO header - subchunks start immediately.
    /// </summary>
    private Dictionary<string, byte[]> ParseMcnkSubchunks(byte[] mcnkData, bool hasHeader = true)
    {
        var result = new Dictionary<string, byte[]>();
        
        int headerSize = hasHeader ? 128 : 0;
        if (mcnkData.Length < headerSize) return result;

        int pos = headerSize;
        while (pos < mcnkData.Length - 8)
        {
            string rawSig = Encoding.ASCII.GetString(mcnkData, pos, 4);
            int size = BitConverter.ToInt32(mcnkData, pos + 4);

            // Validate size - must be non-negative and fit within remaining data
            if (size < 0 || size > 10_000_000 || pos + 8 + size > mcnkData.Length)
            {
                // Console.WriteLine($"[WARN] Malformed subchunk at {pos}: {rawSig} size={size}");
                break;
            }

            try
            {
                var data = new byte[size];
                Buffer.BlockCopy(mcnkData, pos + 8, data, 0, size);
                // Reverse to readable (e.g. "TVCM" -> "MCVT")
                result[ReverseSig(rawSig)] = data;
            }
            catch (Exception)
            {
                break;
            }

            pos += 8 + size;
        }

        return result;
    }

    /// <summary>
    /// Merge MCNK chunks from root, tex0, and obj0 into a single monolithic MCNK.
    /// Root has: MCVT, MCNR, MCLQ (terrain)
    /// Tex0 has: MCLY, MCAL, MCSH (textures)
    /// Obj0 has: MCRD, MCRW (object refs)
    /// </summary>
    private AdtChunk MergeMcnkChunks(AdtChunk rootMcnk, AdtChunk? tex0Mcnk, AdtChunk? obj0Mcnk)
    {
        try
        {
            // Parse subchunks from each source
            // Root MCNK has 128-byte header, tex0/obj0 MCNKs are headerless
            var rootSubs = ParseMcnkSubchunks(rootMcnk.Data, hasHeader: true);
            var tex0Subs = tex0Mcnk != null ? ParseMcnkSubchunks(tex0Mcnk.Data, hasHeader: false) : new Dictionary<string, byte[]>();
            var obj0Subs = obj0Mcnk != null ? ParseMcnkSubchunks(obj0Mcnk.Data, hasHeader: false) : new Dictionary<string, byte[]>();

            // MCNK header is first 128 bytes of root
            byte[] header = new byte[128];
            if (rootMcnk.Data.Length >= 128)
                Buffer.BlockCopy(rootMcnk.Data, 0, header, 0, 128);

            using var ms = new MemoryStream();
            ms.Write(header, 0, header.Length);

            var subchunkOffsets = new Dictionary<string, uint>();

            // Write subchunks in correct order for monolithic ADT
            // Using READABLE signatures
            WriteSubchunkTracked(ms, SUB_MCVT, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCVT
            WriteSubchunkTracked(ms, SUB_MCCV, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCCV
            WriteSubchunkTracked(ms, SUB_MCNR, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCNR
            WriteSubchunkTracked(ms, SUB_MCLY, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCLY
            
            // MCRF - fallback
            if (!obj0Subs.ContainsKey(SUB_MCRF) && !rootSubs.ContainsKey(SUB_MCRF)) { }
            WriteSubchunkTracked(ms, SUB_MCRF, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRF
            
            WriteSubchunkTracked(ms, SUB_MCRD, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRD
            WriteSubchunkTracked(ms, SUB_MCRW, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRW
            
            // MCAL - fallback
            if (!tex0Subs.ContainsKey(SUB_MCAL) && rootSubs.ContainsKey(SUB_MCAL)) { }
            WriteSubchunkTracked(ms, SUB_MCAL, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCAL
            
            WriteSubchunkTracked(ms, SUB_MCSH, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCSH
            WriteSubchunkTracked(ms, SUB_MCSE, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCSE
            WriteSubchunkTracked(ms, SUB_MCLQ, rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCLQ

            var result = ms.ToArray();

            // Update MCNK header offsets
            void WriteOffset(int headerOffset, string sig)
            {
                uint ofs = subchunkOffsets.TryGetValue(sig, out var o) ? o : 0u;
                BitConverter.GetBytes(ofs).CopyTo(result, headerOffset);
            }

            void WriteOffsetAndSize(int headerOffset, int sizeOffset, string sig)
            {
                uint ofs = subchunkOffsets.TryGetValue(sig, out var o) ? o : 0u;
                BitConverter.GetBytes(ofs).CopyTo(result, headerOffset);
                if (ofs > 0)
                {
                    byte[]? subData = null;
                    if (tex0Subs.TryGetValue(sig, out var td)) subData = td;
                    else if (rootSubs.TryGetValue(sig, out var rd)) subData = rd;
                    else if (obj0Subs.TryGetValue(sig, out var od)) subData = od;
                    uint size = subData != null ? (uint)subData.Length : 0;
                    BitConverter.GetBytes(size).CopyTo(result, sizeOffset);
                }
            }

            WriteOffset(0x14, SUB_MCVT);
            WriteOffset(0x18, SUB_MCNR);
            WriteOffset(0x1C, SUB_MCLY);
            WriteOffset(0x20, SUB_MCRF);
            WriteOffsetAndSize(0x24, 0x28, SUB_MCAL);
            WriteOffsetAndSize(0x2C, 0x30, SUB_MCSH);
            WriteOffset(0x58, SUB_MCSE);
            WriteOffset(0x60, SUB_MCLQ);
            WriteOffset(0x74, SUB_MCCV);

            // Set has_mccv flag
            if (subchunkOffsets.TryGetValue(SUB_MCCV, out var mccvOfs) && mccvOfs > 0)
            {
                uint flags = BitConverter.ToUInt32(result, 0);
                flags |= 0x40;
                BitConverter.GetBytes(flags).CopyTo(result, 0);
            }

            return new AdtChunk { Signature = SIG_MCNK, Data = result };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to merge MCNK: {ex.Message}, using root only");
            return rootMcnk;
        }
    }

    /// <summary>
    /// Write a subchunk and track its offset.
    /// </summary>
    private void WriteSubchunkTracked(MemoryStream ms, string sig,
        Dictionary<string, byte[]> rootSubs,
        Dictionary<string, byte[]> tex0Subs,
        Dictionary<string, byte[]> obj0Subs,
        Dictionary<string, uint> offsets)
    {
        byte[]? data = null;

        if (sig == SUB_MCLY || sig == SUB_MCAL || sig == SUB_MCSH)
            data = tex0Subs.GetValueOrDefault(sig) ?? rootSubs.GetValueOrDefault(sig);
        else if (sig == SUB_MCRD || sig == SUB_MCRW)
            data = obj0Subs.GetValueOrDefault(sig) ?? rootSubs.GetValueOrDefault(sig);
        else
            data = rootSubs.GetValueOrDefault(sig) ?? tex0Subs.GetValueOrDefault(sig) ?? obj0Subs.GetValueOrDefault(sig);

        if (sig == SUB_MCCV)
        {
            // Always use minimap MCCV if provided, otherwise fall back to existing or generate default
            if (_minimapMccvData != null && _currentMcnkIndex < _minimapMccvData.Length && _minimapMccvData[_currentMcnkIndex] != null)
                data = _minimapMccvData[_currentMcnkIndex];
            else if (data == null || data.Length == 0)
                data = GenerateDefaultMccv();
        }

        offsets[sig] = (uint)ms.Position + 8;
        
        // Write REVERSED signature to stream (e.g. "MCVT" -> "TVCM")
        ms.Write(Encoding.ASCII.GetBytes(ReverseSig(sig)), 0, 4);
        int size = data?.Length ?? 0;
        ms.Write(BitConverter.GetBytes(size), 0, 4);
        if (data != null && data.Length > 0)
            ms.Write(data, 0, data.Length);
    }

    /// <summary>
    /// Generate default MCCV data with neutral vertex colors.
    /// 145 entries (9*9 + 8*8) of BGRA, each component = 0x7F (1.0 neutral).
    /// </summary>
    private static byte[] GenerateDefaultMccv()
    {
        const int vertexCount = 145; // 9*9 + 8*8
        var data = new byte[vertexCount * 4];
        
        // Fill with neutral color: BGRA = 0x7F, 0x7F, 0x7F, 0x00
        // This represents RGB(1.0, 1.0, 1.0) = white/neutral vertex color
        for (int i = 0; i < vertexCount; i++)
        {
            int offset = i * 4;
            data[offset + 0] = 0x7F; // Blue
            data[offset + 1] = 0x7F; // Green
            data[offset + 2] = 0x7F; // Red
            data[offset + 3] = 0x00; // Alpha (unused)
        }
        
        return data;
    }

    /// <summary>
    /// Build MWMO chunk data (null-terminated WMO path strings).
    /// </summary>
    public byte[] BuildMwmoData(List<string> wmoPaths)
    {
        using var ms = new MemoryStream();
        foreach (var path in wmoPaths)
        {
            var bytes = Encoding.ASCII.GetBytes(path);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0); // null terminator
        }
        return ms.ToArray();
    }

    /// <summary>
    /// Build MWID chunk data (offsets into MWMO).
    /// </summary>
    public byte[] BuildMwidData(List<string> wmoPaths)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        
        uint offset = 0;
        foreach (var path in wmoPaths)
        {
            bw.Write(offset);
            offset += (uint)(Encoding.ASCII.GetByteCount(path) + 1);
        }
        return ms.ToArray();
    }

    /// <summary>
    /// MODF entry for patching.
    /// </summary>
    public struct ModfEntry
    {
        public uint NameId;       // Index into MWMO
        public uint UniqueId;     // Unique placement ID
        public Vector3 Position;
        public Vector3 Rotation;  // Degrees
        public Vector3 BoundsMin;
        public Vector3 BoundsMax;
        public ushort Flags;
        public ushort DoodadSet;
        public ushort NameSet;
        public ushort Scale;      // 1024 = 1.0
    }

    /// <summary>
    /// Build MODF chunk data (64 bytes per entry).
    /// </summary>
    public byte[] BuildModfData(List<ModfEntry> entries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        foreach (var e in entries)
        {
            bw.Write(e.NameId);
            bw.Write(e.UniqueId);
            bw.Write(e.Position.X);
            bw.Write(e.Position.Y);
            bw.Write(e.Position.Z);
            bw.Write(e.Rotation.X);
            bw.Write(e.Rotation.Y);
            bw.Write(e.Rotation.Z);
            bw.Write(e.BoundsMin.X);
            bw.Write(e.BoundsMin.Y);
            bw.Write(e.BoundsMin.Z);
            bw.Write(e.BoundsMax.X);
            bw.Write(e.BoundsMax.Y);
            bw.Write(e.BoundsMax.Z);
            bw.Write(e.Flags);
            bw.Write(e.DoodadSet);
            bw.Write(e.NameSet);
            bw.Write(e.Scale);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// MDDF entry for patching.
    /// </summary>
    public struct MddfEntry
    {
        public uint NameId;       // Index into MMDX
        public uint UniqueId;
        public Vector3 Position;
        public Vector3 Rotation;  // Degrees
        public ushort Scale;      // 1024 = 1.0
        public ushort Flags;
    }

    /// <summary>
    /// Build MDDF chunk data (36 bytes per entry).
    /// </summary>
    public byte[] BuildMddfData(List<MddfEntry> entries)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        foreach (var e in entries)
        {
            bw.Write(e.NameId);
            bw.Write(e.UniqueId);
            bw.Write(e.Position.X);
            bw.Write(e.Position.Y);
            bw.Write(e.Position.Z);
            bw.Write(e.Rotation.X);
            bw.Write(e.Rotation.Y);
            bw.Write(e.Rotation.Z);
            bw.Write(e.Scale);
            bw.Write(e.Flags);
        }

        return ms.ToArray();
    }

    /// <summary>
    /// Build MMDX chunk data (null-terminated M2 path strings).
    /// </summary>
    public byte[] BuildMmdxData(List<string> m2Paths)
    {
        using var ms = new MemoryStream();
        foreach (var path in m2Paths)
        {
            var bytes = Encoding.ASCII.GetBytes(path);
            ms.Write(bytes, 0, bytes.Length);
            ms.WriteByte(0);
        }
        return ms.ToArray();
    }

    /// <summary>
    /// Build MMID chunk data (offsets into MMDX).
    /// </summary>
    public byte[] BuildMmidData(List<string> m2Paths)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);
        
        uint offset = 0;
        foreach (var path in m2Paths)
        {
            bw.Write(offset);
            offset += (uint)(Encoding.ASCII.GetByteCount(path) + 1);
        }
        return ms.ToArray();
    }

    /// <summary>
    /// Patch object placement chunks in a parsed ADT.
    /// </summary>
    public void PatchObjectChunks(
        ParsedAdt adt,
        List<string> wmoPaths,
        List<ModfEntry> modfEntries,
        List<string> m2Paths,
        List<MddfEntry> mddfEntries)
    {
        // Build and replace WMO chunks
        var mwmoData = BuildMwmoData(wmoPaths);
        var mwidData = BuildMwidData(wmoPaths);
        var modfData = BuildModfData(modfEntries);

        EnsureChunkExists(adt, SIG_MWMO, SIG_MMID);
        EnsureChunkExists(adt, SIG_MWID, SIG_MWMO);
        EnsureChunkExists(adt, SIG_MODF, SIG_MDDF);

        adt.ReplaceChunk(SIG_MWMO, mwmoData);
        adt.ReplaceChunk(SIG_MWID, mwidData);
        adt.ReplaceChunk(SIG_MODF, modfData);

        // Build and replace M2 chunks
        var mmdxData = BuildMmdxData(m2Paths);
        var mmidData = BuildMmidData(m2Paths);
        var mddfData = BuildMddfData(mddfEntries);

        EnsureChunkExists(adt, SIG_MMDX, SIG_MTEX);
        EnsureChunkExists(adt, SIG_MMID, SIG_MMDX);
        EnsureChunkExists(adt, SIG_MDDF, SIG_MWID);

        adt.ReplaceChunk(SIG_MMDX, mmdxData);
        adt.ReplaceChunk(SIG_MMID, mmidData);
        adt.ReplaceChunk(SIG_MDDF, mddfData);
    }

    private void EnsureChunkExists(ParsedAdt adt, string sig, string afterSig)
    {
        if (adt.FindChunk(sig) == null)
        {
            adt.InsertChunkAfter(afterSig, new AdtChunk { Signature = sig, Data = Array.Empty<byte>() });
        }
    }

    /// <summary>
    /// Write patched ADT to bytes, fixing MCIN and MHDR offsets.
    /// </summary>
    public byte[] WriteAdt(ParsedAdt adt)
    {
        // First pass: calculate positions of all chunks
        var chunkPositions = new Dictionary<int, long>(); // chunk index -> file offset
        long pos = 0;

        for (int i = 0; i < adt.Chunks.Count; i++)
        {
            chunkPositions[i] = pos;
            pos += adt.Chunks[i].TotalSize;
        }

        // Find MCNK positions for MCIN
        var mcnkIndices = new List<int>();
        for (int i = 0; i < adt.Chunks.Count; i++)
        {
            if (adt.Chunks[i].Signature == SIG_MCNK)
                mcnkIndices.Add(i);
        }

        // Update MCIN if present
        var mcinChunk = adt.FindChunk(SIG_MCIN);
        if (mcinChunk != null && mcnkIndices.Count == 256)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);

            for (int i = 0; i < 256; i++)
            {
                int mcnkIdx = mcnkIndices[i];
                uint offset = (uint)chunkPositions[mcnkIdx];
                uint size = (uint)adt.Chunks[mcnkIdx].TotalSize;
                bw.Write(offset);
                bw.Write(size);
                bw.Write(0u); // flags
                bw.Write(0u); // asyncId
            }

            mcinChunk.Data = ms.ToArray();
        }

        // Update MHDR offsets if present
        var mhdrChunk = adt.FindChunk(SIG_MHDR);
        if (mhdrChunk != null && mhdrChunk.Data.Length >= 64)
        {
            // MHDR contains offsets to various chunks relative to MHDR data start
            // We need to recalculate these based on new chunk positions
            UpdateMhdrOffsets(adt, mhdrChunk, chunkPositions);
        }

        // Second pass: write all chunks
        using var output = new MemoryStream();
        using var writer = new BinaryWriter(output);

        foreach (var chunk in adt.Chunks)
        {
            // Write reversed signature (e.g. "MVER" -> "REVM")
            writer.Write(Encoding.ASCII.GetBytes(ReverseSig(chunk.Signature)));
            writer.Write(chunk.Data.Length);
            writer.Write(chunk.Data);
        }

        return output.ToArray();
    }

    private void UpdateMhdrOffsets(ParsedAdt adt, AdtChunk mhdrChunk, Dictionary<int, long> chunkPositions)
    {
        // ... (logic remains same as it relies on internal Readable signatures) ...
        // MHDR structure (offsets relative to end of MHDR header, i.e., start of MHDR data + 8)
        // 0x00: flags
        // 0x04: ofsMcin
        // 0x08: ofsMtex
        // 0x0C: ofsMmdx
        // 0x10: ofsMmid
        // 0x14: ofsMwmo
        // 0x18: ofsMwid
        // 0x1C: ofsMddf
        // 0x20: ofsModf
        // 0x24: ofsMfbo (optional)
        // 0x28: ofsMh2o
        // 0x2C: ofsMtxf
        // ... more optional fields

        int mhdrIdx = adt.Chunks.FindIndex(c => c.Signature == SIG_MHDR);
        if (mhdrIdx < 0) return;

        long mhdrDataStart = chunkPositions[mhdrIdx] + 8; // After MHDR header

        using var ms = new MemoryStream(mhdrChunk.Data);
        using var br = new BinaryReader(ms);
        using var outMs = new MemoryStream();
        using var bw = new BinaryWriter(outMs);

        // Helper to find chunk offset relative to MHDR data
        uint GetRelativeOffset(string sig)
        {
            int idx = adt.Chunks.FindIndex(c => c.Signature == sig);
            if (idx < 0) return 0;
            return (uint)(chunkPositions[idx] - mhdrDataStart);
        }

        // Read flags, then update them to reflect chunk presence
        uint flags = br.ReadUInt32();
        
        if (GetRelativeOffset(SIG_MCIN) > 0) flags |= 0x1;
        if (GetRelativeOffset(SIG_MTEX) > 0) flags |= 0x2;
        if (GetRelativeOffset(SIG_MMDX) > 0) flags |= 0x4;
        if (GetRelativeOffset(SIG_MMID) > 0) flags |= 0x8;
        if (GetRelativeOffset(SIG_MWMO) > 0) flags |= 0x10;
        if (GetRelativeOffset(SIG_MWID) > 0) flags |= 0x20;
        if (GetRelativeOffset(SIG_MDDF) > 0) flags |= 0x40;
        if (GetRelativeOffset(SIG_MODF) > 0) flags |= 0x80;

        bw.Write(flags);

        bw.Write(GetRelativeOffset(SIG_MCIN));  // 0x04
        bw.Write(GetRelativeOffset(SIG_MTEX));  // 0x08
        bw.Write(GetRelativeOffset(SIG_MMDX));  // 0x0C
        bw.Write(GetRelativeOffset(SIG_MMID));  // 0x10
        bw.Write(GetRelativeOffset(SIG_MWMO));  // 0x14
        bw.Write(GetRelativeOffset(SIG_MWID));  // 0x18
        bw.Write(GetRelativeOffset(SIG_MDDF));  // 0x1C
        bw.Write(GetRelativeOffset(SIG_MODF));  // 0x20

        // Copy remaining bytes (optional fields)
        if (mhdrChunk.Data.Length > 36)
        {
            ms.Position = 36;
            var remaining = br.ReadBytes(mhdrChunk.Data.Length - 36);
            
            // Update MH2O offset if present (at offset 0x28 = 40 bytes from start)
            if (mhdrChunk.Data.Length >= 44)
            {
                // ofsMfbo at 0x24
                bw.Write(0u); // ofsMfbo - usually 0
                bw.Write(GetRelativeOffset(SIG_MH2O)); // 0x28
                
                if (mhdrChunk.Data.Length > 44)
                {
                    ms.Position = 44;
                    bw.Write(br.ReadBytes(mhdrChunk.Data.Length - 44));
                }
            }
            else
            {
                bw.Write(remaining);
            }
        }

        mhdrChunk.Data = outMs.ToArray();
    }

    /// <summary>
    /// Generate a minimal blank ADT template for tiles that don't exist.
    /// Creates valid 3.3.5 monolithic ADT with empty terrain.
    /// </summary>
    public ParsedAdt GenerateBlankAdt(int tileX, int tileY)
    {
        var adt = new ParsedAdt();

        // MVER - version 18 for 3.3.5
        adt.Chunks.Add(new AdtChunk
        {
            Signature = SIG_MVER,
            Data = BitConverter.GetBytes(18u)
        });

        // MHDR - header with offsets (will be fixed up on write)
        var mhdrData = new byte[64];
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MHDR, Data = mhdrData });

        // MCIN - 256 entries, 16 bytes each (will be fixed up on write)
        var mcinData = new byte[256 * 16];
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MCIN, Data = mcinData });

        // MTEX - empty texture list
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MTEX, Data = Array.Empty<byte>() });

        // MMDX - empty M2 list
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MMDX, Data = Array.Empty<byte>() });

        // MMID - empty M2 offsets
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MMID, Data = Array.Empty<byte>() });


        // MWMO - empty WMO list
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MWMO, Data = Array.Empty<byte>() });

        // MWID - empty WMO offsets
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MWID, Data = Array.Empty<byte>() });

        // MDDF - empty M2 placements
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MDDF, Data = Array.Empty<byte>() });

        // MODF - empty WMO placements
        adt.Chunks.Add(new AdtChunk { Signature = SIG_MODF, Data = Array.Empty<byte>() });

        // Generate 256 MCNK chunks (16x16 grid)
        for (int cy = 0; cy < 16; cy++)
        {
            for (int cx = 0; cx < 16; cx++)
            {
                var mcnkData = GenerateBlankMcnk(tileX, tileY, cx, cy);
                adt.Chunks.Add(new AdtChunk { Signature = SIG_MCNK, Data = mcnkData });
            }
        }

        return adt;
    }

    /// <summary>
    /// Generate a minimal blank MCNK chunk.
    /// </summary>
    private byte[] GenerateBlankMcnk(int tileX, int tileY, int chunkX, int chunkY)
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms);

        // MCNK header (128 bytes)
        bw.Write(0u);           // flags
        bw.Write(chunkX);       // indexX
        bw.Write(chunkY);       // indexY
        bw.Write(0u);           // nLayers
        bw.Write(0u);           // nDoodadRefs
        bw.Write(0u);           // ofsHeight (MCVT) - relative to MCNK data
        bw.Write(0u);           // ofsNormal (MCNR)
        bw.Write(0u);           // ofsLayer (MCLY)
        bw.Write(0u);           // ofsRefs (MCRF)
        bw.Write(0u);           // ofsAlpha (MCAL)
        bw.Write(0u);           // sizeAlpha
        bw.Write(0u);           // ofsShadow (MCSH)
        bw.Write(0u);           // sizeShadow
        bw.Write(0u);           // areaid
        bw.Write(0u);           // nMapObjRefs
        bw.Write(0u);           // holes
        
        // Low quality texture map (8 bytes)
        bw.Write(0uL);
        
        bw.Write(0u);           // predTex
        bw.Write(0u);           // noEffectDoodad
        bw.Write(0u);           // ofsSndEmitters (MCSE)
        bw.Write(0u);           // nSndEmitters
        bw.Write(0u);           // ofsLiquid (MCLQ)
        bw.Write(0u);           // sizeLiquid
        
        // Position
        float chunkSize = 33.3333f;
        float tileSize = 533.3333f;
        float baseX = (32 - tileX) * tileSize - chunkX * chunkSize;
        float baseY = (32 - tileY) * tileSize - chunkY * chunkSize;
        bw.Write(baseX);        // position.x
        bw.Write(baseY);        // position.y
        bw.Write(0f);           // position.z
        
        bw.Write(0u);           // ofsMCCV
        bw.Write(0u);           // ofsMCLV
        bw.Write(0u);           // unused

        // Pad to 128 bytes
        while (ms.Position < 128)
            bw.Write((byte)0);

        // MCVT - height map (145 floats = 580 bytes)
        bw.Write(Encoding.ASCII.GetBytes("TVCM")); // Reversed
        bw.Write(145 * 4);
        for (int i = 0; i < 145; i++)
            bw.Write(0f);

        // MCNR - normals (145 * 3 bytes + 13 padding = 448 bytes)
        bw.Write(Encoding.ASCII.GetBytes("RNCM")); // Reversed
        bw.Write(448);
        for (int i = 0; i < 145; i++)
        {
            bw.Write((sbyte)0);   // x
            bw.Write((sbyte)127); // y (up)
            bw.Write((sbyte)0);   // z
        }
        // Padding to 448
        for (int i = 0; i < 13; i++)
            bw.Write((byte)0);

        return ms.ToArray();
    }

    /// <summary>
    /// Full pipeline: read split ADT, patch with PM4 data, write monolithic ADT.
    /// </summary>
    public void PatchAndWrite(
        string rootAdtPath,
        string? obj0Path,
        string? tex0Path,
        string outputPath,
        List<string> wmoPaths,
        List<ModfEntry> modfEntries,
        List<string> m2Paths,
        List<MddfEntry> mddfEntries)
    {
        Console.WriteLine($"[INFO] Parsing ADT: {rootAdtPath}");
        var adt = ParseSplitAdt(rootAdtPath, obj0Path, tex0Path);
        
        Console.WriteLine($"[INFO] Parsed {adt.Chunks.Count} chunks");
        foreach (var c in adt.Chunks.GroupBy(x => x.ReadableSig))
        {
            Console.WriteLine($"  {c.Key}: {c.Count()} chunk(s), {c.Sum(x => x.Data.Length)} bytes");
        }

        Console.WriteLine($"[INFO] Patching object chunks...");
        Console.WriteLine($"  WMOs: {wmoPaths.Count}, MODF entries: {modfEntries.Count}");
        Console.WriteLine($"  M2s: {m2Paths.Count}, MDDF entries: {mddfEntries.Count}");
        
        PatchObjectChunks(adt, wmoPaths, modfEntries, m2Paths, mddfEntries);

        Console.WriteLine($"[INFO] Writing patched ADT: {outputPath}");
        var outputData = WriteAdt(adt);
        
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, outputData);
        
        Console.WriteLine($"[INFO] Written {outputData.Length:N0} bytes");
    }

    /// <summary>
    /// Merge split ADTs into monolithic 3.3.5 ADT and return bytes.
    /// Drop-in replacement for SplitAdtMerger.MergeSplitAdt().
    /// </summary>
    public byte[] MergeSplitAdt(string rootAdtPath, string? obj0Path, string? tex0Path)
    {
        var adt = ParseSplitAdt(rootAdtPath, obj0Path, tex0Path);
        return WriteAdt(adt);
    }

    /// <summary>
    /// Merge split ADTs into monolithic 3.3.5 ADT without modifying object data.
    /// This preserves all existing MWMO/MWID/MODF/MMDX/MMID/MDDF data as-is.
    /// </summary>
    public void MergeAndWrite(
        string rootAdtPath,
        string? obj0Path,
        string? tex0Path,
        string outputPath)
    {
        MergeAndWrite(rootAdtPath, obj0Path, tex0Path, outputPath, null);
    }

    /// <summary>
    /// Merge split ADTs into monolithic 3.3.5 ADT with optional minimap MCCV painting.
    /// For MCNKs without existing MCCV data, uses minimap colors if provided.
    /// </summary>
    /// <param name="minimapMccvData">Optional array of 256 MCCV byte arrays from minimap (one per MCNK).
    /// Each entry should be 580 bytes (145 vertices * 4 bytes BGRA). Pass null to use neutral gray.</param>
    public void MergeAndWrite(
        string rootAdtPath,
        string? obj0Path,
        string? tex0Path,
        string outputPath,
        byte[][]? minimapMccvData)
    {
        Console.WriteLine($"[INFO] Merging ADT: {rootAdtPath}");
        
        // Set minimap MCCV data for fallback during merge
        _minimapMccvData = minimapMccvData;
        
        var adt = ParseSplitAdt(rootAdtPath, obj0Path, tex0Path);
        
        // Clear minimap data after merge
        _minimapMccvData = null;

        Console.WriteLine($"[INFO] Merged {adt.Chunks.Count} chunks (no patching)");
        foreach (var c in adt.Chunks.GroupBy(x => x.ReadableSig))
        {
            Console.WriteLine($"  {c.Key}: {c.Count()} chunk(s), {c.Sum(x => x.Data.Length)} bytes");
        }

        Console.WriteLine($"[INFO] Writing merged ADT: {outputPath}");
        var outputData = WriteAdt(adt);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, outputData);

        Console.WriteLine($"[INFO] Written {outputData.Length:N0} bytes");
    }

    /// <summary>
    /// Full pipeline with fallback to blank ADT if source doesn't exist.
    /// </summary>
    public void PatchOrCreateAndWrite(
        string? rootAdtPath,
        string? obj0Path,
        string? tex0Path,
        string outputPath,
        int tileX,
        int tileY,
        List<string> wmoPaths,
        List<ModfEntry> modfEntries,
        List<string> m2Paths,
        List<MddfEntry> mddfEntries)
    {
        ParsedAdt adt;

        if (rootAdtPath != null && File.Exists(rootAdtPath))
        {
            Console.WriteLine($"[INFO] Parsing existing ADT: {rootAdtPath}");
            adt = ParseSplitAdt(rootAdtPath, obj0Path, tex0Path);
        }
        else
        {
            Console.WriteLine($"[INFO] No source ADT found, generating blank template for tile {tileX}_{tileY}");
            adt = GenerateBlankAdt(tileX, tileY);
        }

        Console.WriteLine($"[INFO] Parsed/generated {adt.Chunks.Count} chunks");
        foreach (var c in adt.Chunks.GroupBy(x => x.ReadableSig))
        {
            Console.WriteLine($"  {c.Key}: {c.Count()} chunk(s), {c.Sum(x => x.Data.Length)} bytes");
        }

        Console.WriteLine($"[INFO] Patching object chunks...");
        Console.WriteLine($"  WMOs: {wmoPaths.Count}, MODF entries: {modfEntries.Count}");
        Console.WriteLine($"  M2s: {m2Paths.Count}, MDDF entries: {mddfEntries.Count}");

        PatchObjectChunks(adt, wmoPaths, modfEntries, m2Paths, mddfEntries);

        Console.WriteLine($"[INFO] Writing patched ADT: {outputPath}");
        var outputData = WriteAdt(adt);

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        File.WriteAllBytes(outputPath, outputData);

        Console.WriteLine($"[INFO] Written {outputData.Length:N0} bytes");
    }

    /// <summary>
    /// Batch process multiple tiles from PM4 data.
    /// </summary>
    public void BatchPatchTiles(
        string sourceAdtDir,
        string outputDir,
        string mapName,
        Dictionary<(int tileX, int tileY), (List<string> wmoPaths, List<ModfEntry> modfEntries, List<string> m2Paths, List<MddfEntry> mddfEntries)> tileData)
    {
        Console.WriteLine($"[INFO] Batch patching {tileData.Count} tiles for map '{mapName}'");

        foreach (var ((tileX, tileY), data) in tileData)
        {
            string baseName = $"{mapName}_{tileX}_{tileY}";
            string rootPath = Path.Combine(sourceAdtDir, $"{baseName}.adt");
            string obj0Path = Path.Combine(sourceAdtDir, $"{baseName}_obj0.adt");
            string tex0Path = Path.Combine(sourceAdtDir, $"{baseName}_tex0.adt");
            string outputPath = Path.Combine(outputDir, $"{baseName}.adt");

            Console.WriteLine($"\n=== Tile {tileX}_{tileY} ===");

            PatchOrCreateAndWrite(
                File.Exists(rootPath) ? rootPath : null,
                File.Exists(obj0Path) ? obj0Path : null,
                File.Exists(tex0Path) ? tex0Path : null,
                outputPath,
                tileX,
                tileY,
                data.wmoPaths,
                data.modfEntries,
                data.m2Paths,
                data.mddfEntries);
        }

        Console.WriteLine($"\n[INFO] Batch complete. Output: {outputDir}");
    }
}
