using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace WowViewer.Core.IO.Lk
{
    /// <summary>
    /// Ghidra-verified MCNK parser for WotLK 3.3.5.
    /// Subchunks are discovered by sequential FourCC scanning from offset 0x80
    /// (after the 128-byte header), NOT by header offset fields.
    /// See: terrain-loading-335-mcnk-deep-dive.md
    /// </summary>
    public class Mcnk
    {
        public readonly struct ParseOptions
        {
            public bool UseHeaderAlphaSize { get; init; }
            public bool UseHeaderShadowSize { get; init; }
            /// <summary>
            /// When true, skip the 128-byte MCNK header and start sub-chunk scanning
            /// at offset 0.  Used for Cata+ _tex0.adt MCNK chunks which are headerless.
            /// </summary>
            public bool SkipHeader { get; init; }
        }

        /// <summary>
        /// How the sub-chunk walk advances past a record.
        /// </summary>
        public enum SubchunkAdvanceRule
        {
            /// <summary>
            /// The 5.0.1 native rule (MapChunk.cpp FUN_00ba3050): advance by the declared
            /// size and nothing else.
            /// </summary>
            DeclaredSize,

            /// <summary>
            /// Legacy over-consumption: MCNR occupies 0x1C0 bytes regardless of its declared
            /// size, and MCAL/MCSH may be extended to the MCNK header's declared size. Correct
            /// for 3.3.5-era files, where the declared MCNR size excludes 13 trailing bytes.
            /// </summary>
            LegacyOverConsume,
        }

        public const string Signature = "MCNK";
        public McnkHeader Header;
        public float[] Heightmap; // MCVT
        public Mcal AlphaMaps; // MCAL (3.3.5 flag-based decode)
        public byte[] McalRawData; // MCAL raw bytes (for Alpha-style sequential decode)
        public byte[] MclyRawData; // MCLY raw bytes (for Alpha-style sequential decode)
        public List<MclyEntry> TextureLayers; // MCLY
        public byte[] GeneratedNormals; // To store normals if needed

        // Raw subchunks discovered by FourCC scan
        public byte[] McvtData;
        public byte[] MccvData;  // MCV? vertex color
        public byte[] McnrData;
        public byte[] McshData;
        public byte[] McseData;  // MCSE positional sound emitters
        public byte[] MclqData;  // MCLQ legacy liquid
        public byte[] McrdData;  // MCRD split doodad references
        public byte[] McrwData;  // MCRW split WMO references
        public byte[] McrfData;  // MCRF monolithic references
        public byte[] MclvData;  // MCLV
        public byte[] McbbData;  // MCBB blend batches (20-byte records)
        public byte[] McddData;  // MCDD detail doodad disable mask
        public uint? McmtValue;  // MCMT is dereferenced by the client, not retained as a pointer

        /// <summary>Layer count as the client derives it: MCLY size &gt;&gt; 4 (16-byte records).</summary>
        public int MclyRecordCount { get; private set; }
        /// <summary>MCRD reference count as the client derives it: size &gt;&gt; 2.</summary>
        public int McrdRecordCount { get; private set; }
        /// <summary>MCRW reference count as the client derives it: size &gt;&gt; 2.</summary>
        public int McrwRecordCount { get; private set; }
        /// <summary>MCBB blend-batch count as the client derives it: size / 0x14, asserted &lt;= 0xff.</summary>
        public int McbbRecordCount { get; private set; }

        /// <summary>
        /// FourCCs encountered in the sub-chunk stream that this parser does not model.
        /// Empty is the goal: native asserts the walk consumes the payload exactly, so an
        /// unmodelled token is a gap in our decode even though the walk survives it.
        /// </summary>
        public IReadOnlyList<string> UnknownSubchunks => _unknownSubchunks;

        /// <summary>
        /// True when the chosen advance rule consumed the MCNK payload exactly, which is the
        /// invariant the client asserts (MapChunk.cpp:0x461 "dataSize == 0").
        /// </summary>
        public bool SubchunkWalkConsumedExactly { get; private set; }

        /// <summary>Which advance rule satisfied the native exact-consumption invariant.</summary>
        public SubchunkAdvanceRule AdvanceRuleUsed { get; private set; }

        private readonly List<string> _unknownSubchunks = new();
        private readonly ParseOptions _parseOptions;

        public Mcnk(byte[] data)
            : this(data, default)
        {
        }

        public Mcnk(byte[] data, ParseOptions parseOptions)
        {
            _parseOptions = parseOptions;
            Parse(data);
        }

        private void Parse(byte[] data)
        {
            if (_parseOptions.SkipHeader || IsHeaderlessSubchunkStream(data))
            {
                Header = default;
                ScanSubchunks(data, startOffset: 0, SelectAdvanceRule(data, 0));
            }
            else
            {
                if (data.Length < 128)
                {
                    if (IsHeaderlessSubchunkStream(data))
                    {
                        Header = default;
                        ScanSubchunks(data, startOffset: 0, SelectAdvanceRule(data, 0));
                        return;
                    }
                    throw new InvalidDataException("MCNK data too short for header");
                }

                Header = ReadHeader(data);
                ScanSubchunks(data, startOffset: 0x80, SelectAdvanceRule(data, 0x80));
            }
        }

        private static bool IsHeaderlessSubchunkStream(byte[] data)
        {
            if (data == null || data.Length < 8)
                return false;

            uint fourcc = BitConverter.ToUInt32(data, 0);
            // Tokens this parser handles, so detection matches capability. MCXH was removed:
            // it appears in neither 5.0.1 dispatcher and corresponds to nothing the client
            // reads — the per-texture height/blend parameters it was invented for are MTXP.
            return fourcc is 0x4D434C59 or 0x594C434D   // MCLY
                        or 0x4D43414C or 0x4C41434D   // MCAL
                        or 0x4D435348 or 0x4853434D   // MCSH
                        or 0x4D435654 or 0x5456434D   // MCVT
                        or 0x4D434E52 or 0x524E434D   // MCNR
                        or 0x4D434356 or 0x5643434D   // MCCV
                        or 0x4D434C51 or 0x514C434D   // MCLQ
                        or 0x4D435345 or 0x4553434D   // MCSE
                        or 0x4D434D53 or 0x534D434D   // MCMS
                        or 0x4D435246 or 0x4652434D   // MCRF
                        or 0x4D434C56 or 0x564C434D   // MCLV
                        or 0x4D434D54 or 0x544D434D   // MCMT
                        or 0x4D434242 or 0x4242434D   // MCBB
                        or 0x4D434444 or 0x4444434D   // MCDD
                        or 0x4D435244 or 0x4452434D   // MCRD
                        or 0x4D435257 or 0x5752434D;  // MCRW
        }

        /// <summary>
        /// Sequential FourCC scan starting at offset 0x80.
        /// Matches Ghidra FUN_007c3a10 behavior exactly.
        /// </summary>
        private static int _diagCount = 0;
        private static bool _diagLiquidDone = false;

        /// <summary>
        /// Pick the advance rule by testing each against the invariant the client itself
        /// asserts: the sub-chunk walk must consume the MCNK payload exactly
        /// (MapChunk.cpp:0x461 "dataSize == 0"). The native declared-size rule is tried first;
        /// 3.3.5-era files only satisfy the invariant under legacy over-consumption, because
        /// their declared MCNR size excludes 13 trailing bytes. Nothing here is guessed from
        /// the build — the data decides.
        /// </summary>
        private void RecordUnknownSubchunk(byte[] data, int pos)
        {
            if (pos + 4 > data.Length)
                return;

            string token = System.Text.Encoding.ASCII.GetString(data, pos, 4);
            if (!_unknownSubchunks.Contains(token))
                _unknownSubchunks.Add(token);
        }

        private SubchunkAdvanceRule SelectAdvanceRule(byte[] data, int startOffset)
        {
            if (ProbeWalkConsumesExactly(data, startOffset, SubchunkAdvanceRule.DeclaredSize))
                return SubchunkAdvanceRule.DeclaredSize;

            if (ProbeWalkConsumesExactly(data, startOffset, SubchunkAdvanceRule.LegacyOverConsume))
                return SubchunkAdvanceRule.LegacyOverConsume;

            // Neither rule satisfies the native invariant. Keep the historical behaviour so
            // existing readers are unchanged, and let ScanSubchunks record the shortfall.
            return SubchunkAdvanceRule.LegacyOverConsume;
        }

        /// <summary>
        /// Walk the sub-chunk stream without materialising anything, reporting whether the
        /// walk lands exactly on the end of the payload under the supplied rule.
        /// </summary>
        private bool ProbeWalkConsumesExactly(byte[] data, int startOffset, SubchunkAdvanceRule rule)
        {
            int pos = startOffset;

            while (pos + 8 <= data.Length)
            {
                uint fourcc = BitConverter.ToUInt32(data, pos);
                uint size = BitConverter.ToUInt32(data, pos + 4);
                int dataStart = pos + 8;

                if (size > (uint)(data.Length - dataStart))
                    return false;

                int next = dataStart + (int)ComputeConsumedSize(fourcc, size, dataStart, data.Length, rule);
                if (next <= pos || next > data.Length)
                    return false;

                pos = next;
            }

            return pos == data.Length;
        }

        /// <summary>
        /// Bytes consumed by one sub-chunk record under the supplied rule. Under
        /// <see cref="SubchunkAdvanceRule.DeclaredSize"/> this is always the declared size,
        /// which is what the 5.0.1 dispatcher does for every token it handles.
        /// </summary>
        private uint ComputeConsumedSize(uint fourcc, uint size, int dataStart, int dataLength, SubchunkAdvanceRule rule)
        {
            if (rule == SubchunkAdvanceRule.DeclaredSize)
                return size;

            switch (fourcc)
            {
                case 0x4D434E52: // MCNR
                case 0x524E434D:
                    return Math.Max(size, 0x1C0u);

                case 0x4D43414C: // MCAL
                case 0x4C41434D:
                {
                    uint fromHeader = _parseOptions.UseHeaderAlphaSize && Header.SizeMcal >= 8
                        ? Header.SizeMcal - 8
                        : 0;
                    return fromHeader > 0 && fromHeader <= (uint)(dataLength - dataStart)
                        ? Math.Max(size, fromHeader)
                        : size;
                }

                case 0x4D435348: // MCSH
                case 0x4853434D:
                {
                    uint fromHeader = _parseOptions.UseHeaderShadowSize && Header.SizeMcsh >= 8
                        ? Header.SizeMcsh - 8
                        : 0;
                    return fromHeader > 0 && fromHeader <= (uint)(dataLength - dataStart)
                        ? Math.Max(size, fromHeader)
                        : size;
                }

                default:
                    return size;
            }
        }

        private void ScanSubchunks(byte[] data, int startOffset, SubchunkAdvanceRule rule)
        {
            AdvanceRuleUsed = rule;
            int pos = startOffset;
            int remaining = data.Length - startOffset;
            // Diag: first chunk, AND first chunk with liquid flags
            uint rawFlags = data.Length >= 4 ? BitConverter.ToUInt32(data, 0) : 0;
            bool hasLiquidFlags = (rawFlags & 0x3C) != 0; // River|Ocean|Magma|Slime
            bool diag = (_diagCount++ == 0) || (hasLiquidFlags && !_diagLiquidDone);
            if (hasLiquidFlags && diag) _diagLiquidDone = true;
            var dl = diag ? new System.Collections.Generic.List<string>() : null;
            if (diag && dl != null) dl.Add($"--- MCNK #{_diagCount - 1} flags=0x{rawFlags:X} hasLiquid={hasLiquidFlags} dataLen={data.Length} ---");

            while (remaining > 8)
            {
                uint fourcc = BitConverter.ToUInt32(data, pos);
                uint size = BitConverter.ToUInt32(data, pos + 4);
                int dataStart = pos + 8;
                uint consumedSize = ComputeConsumedSize(fourcc, size, dataStart, data.Length, rule);

                if (diag)
                {
                    string f = (pos + 4 <= data.Length) ? System.Text.Encoding.ASCII.GetString(data, pos, 4) : "????";
                    dl.Add($"pos=0x{pos:X} fcc='{f}' (0x{fourcc:X8}) size={size} rem={remaining}");
                }

                if (size > (uint)remaining - 8)
                {
                    if (diag) dl.Add($"BREAK: size={size} > rem-8={(uint)remaining - 8}");
                    break;
                }

                switch (fourcc)
                {
                    case 0x4D435654: // MCVT
                    case 0x5456434D:
                        Heightmap = ReadFloats(data, dataStart, size);
                        break;

                    case 0x4D434E52: // MCNR
                    case 0x524E434D:
                    {
                        // How far the walk advances past MCNR is decided by the selected rule
                        // (5.0.1 uses the declared size; 3.3.5 files only satisfy the native
                        // exact-consumption invariant when MCNR is treated as 0x1C0 bytes).
                        int readSize = (int)Math.Min(consumedSize, (uint)(data.Length - dataStart));
                        if (readSize > 0)
                        {
                            McnrData = new byte[Math.Min(readSize, 0x1C0)];
                            Array.Copy(data, dataStart, McnrData, 0, McnrData.Length);
                        }
                        break;
                    }

                    case 0x4D434C59: // MCLY
                    case 0x594C434D:
                        TextureLayers = ReadMclyData(data, dataStart, size);
                        MclyRecordCount = (int)(size >> 4); // client: layerCount = size >> 4
                        // Also keep raw bytes for Alpha-style decode path
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MclyRawData = new byte[size];
                            Array.Copy(data, dataStart, MclyRawData, 0, (int)size);
                        }
                        break;

                    case 0x4D43414C: // MCAL
                    case 0x4C41434D:
                    {
                        if (consumedSize > 0 && dataStart + consumedSize <= data.Length)
                        {
                            var mcalBytes = new byte[consumedSize];
                            Array.Copy(data, dataStart, mcalBytes, 0, (int)consumedSize);
                            McalRawData = mcalBytes; // Always keep raw
                            AlphaMaps = new Mcal(mcalBytes); // 3.3.5 flag-based wrapper
                        }
                        break;
                    }

                    case 0x4D435348: // MCSH
                    case 0x4853434D:
                    {
                        if (consumedSize > 0 && dataStart + consumedSize <= data.Length)
                        {
                            McshData = new byte[consumedSize];
                            Array.Copy(data, dataStart, McshData, 0, (int)consumedSize);
                        }
                        break;
                    }

                    case 0x4D434C51: // MCLQ
                    case 0x514C434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MclqData = new byte[size];
                            Array.Copy(data, dataStart, MclqData, 0, (int)size);
                        }
                        break;

                    case 0x4D435345: // MCSE
                    case 0x4553434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McseData = new byte[size];
                            Array.Copy(data, dataStart, McseData, 0, (int)size);
                        }
                        break;

                    case 0x4D434356: // MCCV vertex color
                    case 0x5643434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MccvData = new byte[size];
                            Array.Copy(data, dataStart, MccvData, 0, (int)size);
                        }
                        break;

                    case 0x4D435244: // MCRD split doodad references
                    case 0x4452434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McrdData = new byte[size];
                            Array.Copy(data, dataStart, McrdData, 0, (int)size);
                        }
                        McrdRecordCount = (int)(size >> 2); // client: count = size >> 2
                        break;

                    case 0x4D435257: // MCRW split WMO references
                    case 0x5752434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McrwData = new byte[size];
                            Array.Copy(data, dataStart, McrwData, 0, (int)size);
                        }
                        McrwRecordCount = (int)(size >> 2); // client: count = size >> 2
                        break;

                    case 0x4D435246: // MCRF monolithic references
                    case 0x4652434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McrfData = new byte[size];
                            Array.Copy(data, dataStart, McrfData, 0, (int)size);
                        }
                        break;

                    case 0x4D434C56: // MCLV
                    case 0x564C434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            MclvData = new byte[size];
                            Array.Copy(data, dataStart, MclvData, 0, (int)size);
                        }
                        break;

                    case 0x4D434D54: // MCMT - client dereferences the payload, storing a value
                    case 0x544D434D:
                        if (size >= 4 && dataStart + 4 <= data.Length)
                            McmtValue = BitConverter.ToUInt32(data, dataStart);
                        break;

                    case 0x4D434242: // MCBB blend batches, 20-byte records
                    case 0x4242434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McbbData = new byte[size];
                            Array.Copy(data, dataStart, McbbData, 0, (int)size);
                        }
                        McbbRecordCount = (int)(size / 0x14); // client asserts this is <= 0xff
                        break;

                    case 0x4D434444: // MCDD detail-doodad mask; client asserts size 8 or 32
                    case 0x4444434D:
                        if (size > 0 && dataStart + size <= data.Length)
                        {
                            McddData = new byte[size];
                            Array.Copy(data, dataStart, McddData, 0, (int)size);
                        }
                        break;

                    default:
                        RecordUnknownSubchunk(data, pos);
                        break;
                }

                pos = dataStart + (int)consumedSize;
                remaining = data.Length - pos;
            }

            SubchunkWalkConsumedExactly = pos == data.Length;

            // Fallback: if FourCC scan didn't find MCLQ but header has a valid offset,
            // use the header offset to locate MCLQ data (0.6.0 style).
            // Header offsets are relative to MCNK chunk start (token), but our data
            // starts after the 8-byte chunk header, so subtract 8.
            if (MclqData == null && Header.OfsMclq > 8)
            {
                int mclqPos = (int)Header.OfsMclq - 8; // adjust for missing chunk header
                if (mclqPos >= 0 && mclqPos + 8 <= data.Length)
                {
                    // Validate MCLQ FourCC at the offset
                    uint mclqFcc = BitConverter.ToUInt32(data, mclqPos);
                    if (mclqFcc == 0x4D434C51) // MCLQ
                    {
                        uint mclqDeclaredSize = BitConverter.ToUInt32(data, mclqPos + 4);
                        int mclqDataStart = mclqPos + 8;

                        // Use declared size if > 0, otherwise compute from flags
                        int mclqSize;
                        if (mclqDeclaredSize > 0 && mclqDataStart + mclqDeclaredSize <= data.Length)
                        {
                            mclqSize = (int)mclqDeclaredSize;
                        }
                        else
                        {
                            // Compute from liquid flags: each set bit in 0x3C = one 0x2D4-byte instance
                            uint flags = (uint)Header.Flags;
                            int instanceCount = 0;
                            if ((flags & 0x04) != 0) instanceCount++;
                            if ((flags & 0x08) != 0) instanceCount++;
                            if ((flags & 0x10) != 0) instanceCount++;
                            if ((flags & 0x20) != 0) instanceCount++;
                            mclqSize = instanceCount * 0x2D4;
                        }

                        if (mclqSize > 0 && mclqDataStart + mclqSize <= data.Length)
                        {
                            MclqData = new byte[mclqSize];
                            Array.Copy(data, mclqDataStart, MclqData, 0, mclqSize);
                        }
                    }
                }
            }

            // Some 3.x/4.x MCNKs omit both MCLY and MCAL and place MCCV
            // immediately after a short MCNR payload. The client-side MCNR
            // reader consumes a padded 0x1C0-byte span, so the normal walk can
            // step over the following MCCV header in those sparse chunks.
            // MCCV is independent vertex data; recover it without requiring
            // any texture-layer or alpha subchunk to exist.
            if (MccvData == null)
                RecoverMccvFromRawScan(data, startOffset);

            if (diag && dl != null)
            {
                dl.Add($"RESULT: MCVT={Heightmap != null} MCNR={McnrData != null} MCLY={TextureLayers?.Count ?? -1} MCAL={McalRawData != null}({McalRawData?.Length ?? 0}) MCSH={McshData != null} MCCV={MccvData != null}({MccvData?.Length ?? 0}) MCLQ={MclqData != null}({MclqData?.Length ?? 0}) flags=0x{(uint)Header.Flags:X} ofsMclq=0x{Header.OfsMclq:X}");
                try { File.AppendAllLines(Path.Combine(Path.GetTempPath(), "mcnk_scan.txt"), dl); } catch { }
            }
        }

        private void RecoverMccvFromRawScan(byte[] data, int startOffset)
        {
            const uint MccvFourCc = 0x4D434356;
            const int MccvPayloadSize = 145 * 4;

            // The normal walk deliberately inflates MCNR to the client-sized
            // 0x1C0-byte span. Sparse chunks can omit that padding, so use the
            // declared subchunk sizes for this recovery pass. This keeps the
            // fallback bounded to valid subchunk boundaries instead of
            // searching arbitrary vertex bytes for an MCCV-looking sequence.
            for (int pos = Math.Max(0, startOffset); pos + 8 <= data.Length;)
            {
                uint fourcc = BitConverter.ToUInt32(data, pos);
                uint declaredSize = BitConverter.ToUInt32(data, pos + 4);
                if (declaredSize > (uint)(data.Length - pos - 8))
                {
                    break;
                }

                if (fourcc == MccvFourCc)
                {
                    if (declaredSize < MccvPayloadSize)
                        return;

                    MccvData = new byte[declaredSize];
                    Array.Copy(data, pos + 8, MccvData, 0, (int)declaredSize);
                    return;
                }

                int next = pos + 8 + (int)declaredSize;
                if (next <= pos)
                    break;
                pos = next;
            }
        }

        /// <summary>
        /// Read header fields from raw bytes.
        /// Ghidra-verified offsets from data start (FUN_007c64b0).
        /// </summary>
        private static McnkHeader ReadHeader(byte[] data)
        {
            var h = new McnkHeader();
            h.Flags = (McnkFlags)BitConverter.ToUInt32(data, 0x00);
            h.IndexX = BitConverter.ToUInt32(data, 0x04);
            h.IndexY = BitConverter.ToUInt32(data, 0x08);
            h.Layers = BitConverter.ToUInt32(data, 0x0C);
            h.DoodadRefs = BitConverter.ToUInt32(data, 0x10);
            h.AreaId = BitConverter.ToUInt32(data, 0x34);
            h.MapObjRefs = BitConverter.ToUInt32(data, 0x38);
            h.Holes = BitConverter.ToUInt16(data, 0x3C);

            // Sub-chunk offset table (0.6.0 Ghidra-verified)
            if (data.Length >= 0x68)
            {
                h.OfsMcvt = BitConverter.ToUInt32(data, 0x14);
                h.OfsMcnr = BitConverter.ToUInt32(data, 0x18);
                h.OfsMcly = BitConverter.ToUInt32(data, 0x1C);
                h.OfsMcrf = BitConverter.ToUInt32(data, 0x20);
                h.OfsMcal = BitConverter.ToUInt32(data, 0x24);
                h.SizeMcal = BitConverter.ToUInt32(data, 0x28);
                h.OfsMcsh = BitConverter.ToUInt32(data, 0x2C);
                h.SizeMcsh = BitConverter.ToUInt32(data, 0x30);
                h.OfsMcse = BitConverter.ToUInt32(data, 0x58);
                h.NSndEmitters = BitConverter.ToUInt32(data, 0x5C);
                h.OfsMclq = BitConverter.ToUInt32(data, 0x60);
                h.SizeMclq = BitConverter.ToUInt32(data, 0x64);
            }

            if (data.Length >= 0x7C)
            {
                h.Position = new float[] {
                    BitConverter.ToSingle(data, 0x70), // Z (base height)
                    BitConverter.ToSingle(data, 0x74), // X
                    BitConverter.ToSingle(data, 0x78)  // Y
                };
            }
            else
            {
                h.Position = new float[] { 0f, 0f, 0f };
            }

            return h;
        }

        private static float[] ReadFloats(byte[] data, int offset, uint size)
        {
            int count = (int)(size / 4);
            var floats = new float[count];
            for (int i = 0; i < count; i++)
                floats[i] = BitConverter.ToSingle(data, offset + i * 4);
            return floats;
        }

        private static List<MclyEntry> ReadMclyData(byte[] data, int offset, uint size)
        {
            var list = new List<MclyEntry>();
            int end = offset + (int)size;
            int pos = offset;
            while (pos + 16 <= end && pos + 16 <= data.Length)
            {
                list.Add(new MclyEntry
                {
                    TextureId = BitConverter.ToUInt32(data, pos),
                    Flags = (MclyFlags)BitConverter.ToUInt32(data, pos + 4),
                    AlphaMapOffset = BitConverter.ToUInt32(data, pos + 8),
                    EffectId = BitConverter.ToUInt32(data, pos + 12)
                });
                pos += 16;
            }
            return list;
        }
    }

    public struct McnkHeader
    {
        public McnkFlags Flags;
        public uint IndexX;
        public uint IndexY;
        public uint Layers;
        public uint DoodadRefs;
        public uint AreaId;
        public uint MapObjRefs;
        public ushort Holes;
        public float[] Position; // [0]=Z(height), [1]=X, [2]=Y at data offset 0x70

        // 0.6.0 sub-chunk offset table (offsets relative to MCNK chunk start, i.e. where token lives)
        // Since we receive data AFTER the 8-byte chunk header, these offsets need -8 adjustment.
        public uint OfsMcvt;  // 0x14
        public uint OfsMcnr;  // 0x18
        public uint OfsMcly;  // 0x1c
        public uint OfsMcrf;  // 0x20
        public uint OfsMcal;  // 0x24
        public uint SizeMcal; // 0x28 (sizeAlpha)
        public uint OfsMcsh;  // 0x2c
        public uint SizeMcsh; // 0x30 (sizeShadow)
        public uint OfsMcse;  // 0x58
        public uint NSndEmitters; // 0x5C
        public uint OfsMclq;  // 0x60
        public uint SizeMclq; // 0x64 (size of MCLQ data, from 0.5.3 header)
    }

    [Flags]
    public enum McnkFlags : uint
    {
        HasShadows = 0x1,
        Impassable = 0x2,
        River = 0x4,
        Ocean = 0x8,
        HasMagma = 0x10,
        HasSlime = 0x20,
        HasMccv = 0x40,
        HasBakedShadows = 0x20000
    }
}
