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

    // Reversed signatures as they appear on disk
    private const string SIG_MVER = "REVM";
    private const string SIG_MHDR = "RDHM";
    private const string SIG_MCIN = "NICM";
    private const string SIG_MTEX = "XETM";
    private const string SIG_MMDX = "XDMM";
    private const string SIG_MMID = "DIMM";
    private const string SIG_MWMO = "OMWM";
    private const string SIG_MWID = "DIWM";
    private const string SIG_MDDF = "FDDM";
    private const string SIG_MODF = "FDOM";
    private const string SIG_MCNK = "KNCM";
    private const string SIG_MH2O = "O2HM";

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
    /// Parse an ADT file into mutable chunks.
    /// </summary>
    public ParsedAdt ParseAdt(byte[] data)
    {
        var result = new ParsedAdt();
        int pos = 0;

        while (pos < data.Length - 8)
        {
            string sig = Encoding.ASCII.GetString(data, pos, 4);
            int size = BitConverter.ToInt32(data, pos + 4);

            if (size < 0 || pos + 8 + size > data.Length)
            {
                Console.WriteLine($"[WARN] Invalid chunk at {pos}: sig={sig}, size={size}");
                break;
            }

            var chunkData = new byte[size];
            Buffer.BlockCopy(data, pos + 8, chunkData, 0, size);

            result.Chunks.Add(new AdtChunk
            {
                Signature = sig,
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

        // 1. MVER (from root)
        var mver = root.FindChunk(SIG_MVER);
        if (mver != null) merged.Chunks.Add(mver);

        // 2. MHDR (from root)
        var mhdr = root.FindChunk(SIG_MHDR);
        if (mhdr != null) merged.Chunks.Add(mhdr);

        // 3. MCIN - always include (will be regenerated on write with correct offsets)
        // Split ADTs don't have MCIN, so create placeholder with 256 entries (16 bytes each = 4096 bytes)
        var mcin = root.FindChunk(SIG_MCIN);
        if (mcin != null)
            merged.Chunks.Add(mcin);
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MCIN, Data = new byte[256 * 16] });

        // 4. MTEX (from tex0 first, fallback to root) - normalize paths
        var mtex = tex0?.FindChunk(SIG_MTEX) ?? root.FindChunk(SIG_MTEX);
        if (mtex != null && mtex.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MTEX, Data = NormalizePathChunk(mtex.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MTEX, Data = Array.Empty<byte>() });

        // 5. MMDX (from obj0 first, fallback to root) - normalize paths
        var mmdx = obj0?.FindChunk(SIG_MMDX) ?? root.FindChunk(SIG_MMDX);
        if (mmdx != null && mmdx.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMDX, Data = NormalizePathChunk(mmdx.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMDX, Data = Array.Empty<byte>() });

        // 6. MMID (from obj0 first, fallback to root)
        var mmid = obj0?.FindChunk(SIG_MMID) ?? root.FindChunk(SIG_MMID);
        if (mmid != null && mmid.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMID, Data = mmid.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MMID, Data = Array.Empty<byte>() });

        // 7. MWMO (from obj0 first, fallback to root) - normalize paths
        var mwmo = obj0?.FindChunk(SIG_MWMO) ?? root.FindChunk(SIG_MWMO);
        if (mwmo != null && mwmo.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWMO, Data = NormalizePathChunk(mwmo.Data) });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWMO, Data = Array.Empty<byte>() });

        // 8. MWID (from obj0 first, fallback to root)
        var mwid = obj0?.FindChunk(SIG_MWID) ?? root.FindChunk(SIG_MWID);
        if (mwid != null && mwid.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWID, Data = mwid.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MWID, Data = Array.Empty<byte>() });

        // 9. MDDF (from obj0 first, fallback to root)
        var mddf = obj0?.FindChunk(SIG_MDDF) ?? root.FindChunk(SIG_MDDF);
        if (mddf != null && mddf.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MDDF, Data = mddf.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MDDF, Data = Array.Empty<byte>() });

        // 10. MODF (from obj0 first, fallback to root)
        var modf = obj0?.FindChunk(SIG_MODF) ?? root.FindChunk(SIG_MODF);
        if (modf != null && modf.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MODF, Data = modf.Data });
        else
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MODF, Data = Array.Empty<byte>() });

        // 11. MH2O (from root, if present)
        var mh2o = root.FindChunk(SIG_MH2O);
        if (mh2o != null && mh2o.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = SIG_MH2O, Data = mh2o.Data });

        // 12. MCNK x 256 - merge from root, tex0, and obj0
        var rootMcnks = root.Chunks.Where(c => c.Signature == SIG_MCNK).ToList();
        var tex0Mcnks = tex0?.Chunks.Where(c => c.Signature == SIG_MCNK).ToList() ?? new List<AdtChunk>();
        var obj0Mcnks = obj0?.Chunks.Where(c => c.Signature == SIG_MCNK).ToList() ?? new List<AdtChunk>();

        Console.WriteLine($"[INFO] MCNK counts - root: {rootMcnks.Count}, tex0: {tex0Mcnks.Count}, obj0: {obj0Mcnks.Count}");

        for (int i = 0; i < rootMcnks.Count; i++)
        {
            var rootMcnk = rootMcnks[i];
            var tex0Mcnk = i < tex0Mcnks.Count ? tex0Mcnks[i] : null;
            var obj0Mcnk = i < obj0Mcnks.Count ? obj0Mcnks[i] : null;

            var mergedMcnk = MergeMcnkChunks(rootMcnk, tex0Mcnk, obj0Mcnk);
            merged.Chunks.Add(mergedMcnk);
        }

        // 13. MFBO (from root, if present)
        var mfbo = root.FindChunk("OBFM"); // MFBO reversed
        if (mfbo != null && mfbo.Data.Length > 0)
            merged.Chunks.Add(new AdtChunk { Signature = "OBFM", Data = mfbo.Data });

        return merged;
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

            // Build merged MCNK with all subchunks, tracking offsets
            using var ms = new MemoryStream();
            ms.Write(header, 0, header.Length);

            // Track subchunk offsets (relative to MCNK chunk start, which is 8 bytes before data)
            // Offsets in header are relative to MCNK chunk start (before the 8-byte chunk header)
            // So offset 0x80 (128) = first subchunk position
            var subchunkOffsets = new Dictionary<string, uint>();

            // Write subchunks in correct order for monolithic ADT
            // Order: MCVT, MCCV, MCNR, MCLY, MCRF, MCAL, MCSH, MCSE, MCLQ
            WriteSubchunkTracked(ms, "TVCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCVT - heights
            WriteSubchunkTracked(ms, "VCCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCCV - vertex colors
            WriteSubchunkTracked(ms, "RNCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCNR - normals
            WriteSubchunkTracked(ms, "YLCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCLY - texture layers (from tex0)
            WriteSubchunkTracked(ms, "FRCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRF - doodad/WMO refs
            WriteSubchunkTracked(ms, "DRCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRD - doodad refs (from obj0)
            WriteSubchunkTracked(ms, "WRCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCRW - WMO refs (from obj0)
            WriteSubchunkTracked(ms, "LACM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCAL - alpha maps (from tex0)
            WriteSubchunkTracked(ms, "HSCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCSH - shadows (from tex0)
            WriteSubchunkTracked(ms, "ESCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCSE - sound emitters
            WriteSubchunkTracked(ms, "QLCM", rootSubs, tex0Subs, obj0Subs, subchunkOffsets); // MCLQ - liquids

            var result = ms.ToArray();

            // Update MCNK header offsets
            // MCNK header structure (offsets relative to MCNK chunk start = 8 bytes before data):
            // 0x14: ofsHeight (MCVT)
            // 0x18: ofsNormal (MCNR)
            // 0x1C: ofsLayer (MCLY)
            // 0x20: ofsRefs (MCRF)
            // 0x24: ofsAlpha (MCAL)
            // 0x28: sizeAlpha
            // 0x2C: ofsShadow (MCSH)
            // 0x30: sizeShadow
            // 0x58: ofsSndEmitters (MCSE)
            // 0x60: ofsLiquid (MCLQ)
            // 0x74: ofsMCCV
            
            void WriteOffset(int headerOffset, string sig)
            {
                uint ofs = subchunkOffsets.TryGetValue(sig, out var o) ? o : 0u;
                BitConverter.GetBytes(ofs).CopyTo(result, headerOffset);
            }

            void WriteOffsetAndSize(int headerOffset, int sizeOffset, string sig)
            {
                uint ofs = subchunkOffsets.TryGetValue(sig, out var o) ? o : 0u;
                BitConverter.GetBytes(ofs).CopyTo(result, headerOffset);
                // Size is the subchunk data size (without 8-byte header)
                if (ofs > 0)
                {
                    // Find size from the subchunk at that offset
                    byte[]? subData = null;
                    if (rootSubs.TryGetValue(sig, out var rd)) subData = rd;
                    else if (tex0Subs.TryGetValue(sig, out var td)) subData = td;
                    else if (obj0Subs.TryGetValue(sig, out var od)) subData = od;
                    uint size = subData != null ? (uint)subData.Length : 0;
                    BitConverter.GetBytes(size).CopyTo(result, sizeOffset);
                }
            }

            WriteOffset(0x14, "TVCM");      // ofsHeight -> MCVT
            WriteOffset(0x18, "RNCM");      // ofsNormal -> MCNR
            WriteOffset(0x1C, "YLCM");      // ofsLayer -> MCLY
            WriteOffset(0x20, "FRCM");      // ofsRefs -> MCRF
            WriteOffsetAndSize(0x24, 0x28, "LACM"); // ofsAlpha, sizeAlpha -> MCAL
            WriteOffsetAndSize(0x2C, 0x30, "HSCM"); // ofsShadow, sizeShadow -> MCSH
            WriteOffset(0x58, "ESCM");      // ofsSndEmitters -> MCSE
            WriteOffset(0x60, "QLCM");      // ofsLiquid -> MCLQ
            WriteOffset(0x74, "VCCM");      // ofsMCCV -> MCCV

            // Set has_mccv flag (bit 6 = 0x40) in MCNK header flags at offset 0x00
            // This tells the client that MCCV vertex colors are present
            if (subchunkOffsets.TryGetValue("VCCM", out var mccvOfs) && mccvOfs > 0)
            {
                uint flags = BitConverter.ToUInt32(result, 0);
                flags |= 0x40; // has_mccv flag
                BitConverter.GetBytes(flags).CopyTo(result, 0);
            }

            return new AdtChunk { Signature = SIG_MCNK, Data = result };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to merge MCNK: {ex.Message}, using root only");
            return rootMcnk; // Fall back to root MCNK if merge fails
        }
    }

    /// <summary>
    /// Write a subchunk and track its offset.
    /// For MCCV, generates default vertex colors (neutral 0x7F7F7F00) if none exist.
    /// </summary>
    private void WriteSubchunkTracked(MemoryStream ms, string sig,
        Dictionary<string, byte[]> rootSubs,
        Dictionary<string, byte[]> tex0Subs,
        Dictionary<string, byte[]> obj0Subs,
        Dictionary<string, uint> offsets)
    {
        byte[]? data = null;

        // Texture-related chunks prefer tex0
        if (sig == "YLCM" || sig == "LACM" || sig == "HSCM") // MCLY, MCAL, MCSH
        {
            data = tex0Subs.GetValueOrDefault(sig) ?? rootSubs.GetValueOrDefault(sig);
        }
        // Object-related chunks prefer obj0
        else if (sig == "DRCM" || sig == "WRCM") // MCRD, MCRW
        {
            data = obj0Subs.GetValueOrDefault(sig) ?? rootSubs.GetValueOrDefault(sig);
        }
        // Everything else from root first
        else
        {
            data = rootSubs.GetValueOrDefault(sig) ?? tex0Subs.GetValueOrDefault(sig) ?? obj0Subs.GetValueOrDefault(sig);
        }

        // MCCV special handling: generate default vertex colors if none exist
        // MCCV format: 145 entries of BGRA (4 bytes each), 0x7F = 1.0 (neutral)
        // WoW uses BGRA order: blue, green, red, alpha
        if (sig == "VCCM" && (data == null || data.Length == 0))
        {
            data = GenerateDefaultMccv();
        }

        // Always write the subchunk (even if empty) so offsets are valid
        // Record offset (position in stream = offset from MCNK data start)
        // MCNK header offsets are relative to chunk start (8 bytes before data)
        // So we add 8 to convert from data offset to chunk offset
        offsets[sig] = (uint)ms.Position + 8;
        
        ms.Write(Encoding.ASCII.GetBytes(sig), 0, 4);
        int size = data?.Length ?? 0;
        ms.Write(BitConverter.GetBytes(size), 0, 4);
        if (data != null && data.Length > 0)
        {
            ms.Write(data, 0, data.Length);
        }
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
            string sig = Encoding.ASCII.GetString(mcnkData, pos, 4);
            int size = BitConverter.ToInt32(mcnkData, pos + 4);

            // Validate size - must be non-negative and fit within remaining data
            if (size < 0 || size > 10_000_000 || pos + 8 + size > mcnkData.Length)
                break;

            try
            {
                var data = new byte[size];
                Buffer.BlockCopy(mcnkData, pos + 8, data, 0, size);
                result[sig] = data;
            }
            catch (Exception)
            {
                // Skip corrupted subchunk
                break;
            }

            pos += 8 + size;
        }

        return result;
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
            writer.Write(Encoding.ASCII.GetBytes(chunk.Signature));
            writer.Write(chunk.Data.Length);
            writer.Write(chunk.Data);
        }

        return output.ToArray();
    }

    private void UpdateMhdrOffsets(ParsedAdt adt, AdtChunk mhdrChunk, Dictionary<int, long> chunkPositions)
    {
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

        // Read flags, keep them
        uint flags = br.ReadUInt32();
        bw.Write(flags);

        // Helper to find chunk offset relative to MHDR data
        uint GetRelativeOffset(string sig)
        {
            int idx = adt.Chunks.FindIndex(c => c.Signature == sig);
            if (idx < 0) return 0;
            return (uint)(chunkPositions[idx] - mhdrDataStart);
        }

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
    /// Merge split ADTs into monolithic 3.3.5 ADT without modifying object data.
    /// This preserves all existing MWMO/MWID/MODF/MMDX/MMID/MDDF data as-is.
    /// </summary>
    public void MergeAndWrite(
        string rootAdtPath,
        string? obj0Path,
        string? tex0Path,
        string outputPath)
    {
        Console.WriteLine($"[INFO] Merging ADT: {rootAdtPath}");
        var adt = ParseSplitAdt(rootAdtPath, obj0Path, tex0Path);

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
