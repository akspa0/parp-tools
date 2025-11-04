using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using GillijimProject.Utilities;
using Util = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] C# port of McnkAlpha (see lib/gillijimproject/wowfiles/alpha/McnkAlpha.h)
/// </summary>
public class McnkAlpha : Mcnk
{
    private new const int McnkTerrainHeaderSize = 128;
    private new const int ChunkLettersAndSize = 8;
    private const int McvtSize = 580;
    private const int McnrSize = 448;
    
    private readonly int _adtNumber;
    private readonly McnkAlphaHeader _mcnkAlphaHeader;
    private readonly McvtAlpha _mcvt;
    private readonly McnrAlpha _mcnrAlpha;
    private readonly Chunk _mcly;
    private readonly Mcrf _mcrf;
    private readonly Chunk _mcsh;
    private readonly Mcal _mcal;
    private readonly Chunk _mclq;

    public McnkAlpha(FileStream adtFile, int offsetInFile, int headerSize, int adtNum) : base(adtFile, offsetInFile)
    {
        _adtNumber = adtNum;
        _ = headerSize; // [PORT] Not used by the Alpha reader; header size is fixed
        
        int headerStartOffset = offsetInFile;
        offsetInFile += ChunkLettersAndSize;

        byte[] dataBuffer = Util.GetByteArrayFromFile(adtFile, offsetInFile, McnkTerrainHeaderSize);
        _mcnkAlphaHeader = Util.ByteArrayToStruct<McnkAlphaHeader>(dataBuffer);

        // Read MCVT
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.McvtOffset;
        byte[] mcvtData = Util.GetByteArrayFromFile(adtFile, offsetInFile, McvtSize);
        _mcvt = new McvtAlpha("MCVT", McvtSize, mcvtData);

        // Read MCNR
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.McnrOffset;
        byte[] mcnrData = Util.GetByteArrayFromFile(adtFile, offsetInFile, McnrSize);
        _mcnrAlpha = new McnrAlpha("MCNR", McnrSize, mcnrData);

        // Read MCLY
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.MclyOffset;
        _mcly = new Chunk(adtFile, offsetInFile);

        // Read MCRF
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.McrfOffset;
        _mcrf = new Mcrf(adtFile, offsetInFile);

        // Read MCSH
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.McshOffset;
        byte[] mcshData = Util.GetByteArrayFromFile(adtFile, offsetInFile, _mcnkAlphaHeader.McshSize);
        _mcsh = new Chunk("MCSH", _mcnkAlphaHeader.McshSize, mcshData);

        // Read MCAL (Alpha MCAL has no chunk header; offset is relative to MCNK payload start)
        int mcnkPayloadStart = headerStartOffset + McnkTerrainHeaderSize;
        int mcalStartAbs = mcnkPayloadStart + _mcnkAlphaHeader.McalOffset;

        // Derive MCAL size from the next known subchunk (prefer MCLQ), else to end of MCNK payload
        int nextAfterMcalRel = (_mcnkAlphaHeader.MclqOffset > _mcnkAlphaHeader.McalOffset && _mcnkAlphaHeader.MclqOffset > 0)
            ? _mcnkAlphaHeader.MclqOffset
            : _mcnkAlphaHeader.McnkChunksSize;
        int mcalSize = Math.Max(0, nextAfterMcalRel - _mcnkAlphaHeader.McalOffset);

        // Clamp to file length if needed
        long remaining = adtFile.Length - mcalStartAbs;
        if (remaining < mcalSize) mcalSize = (int)Math.Max(0, remaining);

        offsetInFile = mcalStartAbs;
        byte[] mcalData = Util.GetByteArrayFromFile(adtFile, offsetInFile, mcalSize);
        _mcal = new Mcal("MCAL", mcalData.Length, mcalData);

        // Read MCLQ
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.MclqOffset;
        int mclqSize = _mcnkAlphaHeader.McnkChunksSize - _mcnkAlphaHeader.MclqOffset;
        byte[] mclqData = Util.GetByteArrayFromFile(adtFile, offsetInFile, mclqSize);
        _mclq = new Chunk("MCLQ", mclqSize, mclqData);
    }

    public McnkAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) 
    { 
        // Initialize non-nullable fields to prevent CS8618 warnings
        _adtNumber = 0;
        _mcnkAlphaHeader = new McnkAlphaHeader();
        _mcvt = new McvtAlpha();
        _mcnrAlpha = new McnrAlpha();
        _mcly = new Chunk();
        _mcrf = new Mcrf();
        _mcsh = new Chunk();
        _mcal = new Mcal();
        _mclq = new Chunk();
    }
    
    public McnkAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) 
    { 
        // Initialize non-nullable fields to prevent CS8618 warnings
        _adtNumber = 0;
        _mcnkAlphaHeader = new McnkAlphaHeader();
        _mcvt = new McvtAlpha();
        _mcnrAlpha = new McnrAlpha();
        _mcly = new Chunk();
        _mcrf = new Mcrf();
        _mcsh = new Chunk();
        _mcal = new Mcal();
        _mclq = new Chunk();
    }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public McnkAlpha() : base("MCNK", 0, Array.Empty<byte>())
    {
        // Initialize non-nullable fields to prevent CS8618 warnings
        _adtNumber = 0;
        _mcnkAlphaHeader = new McnkAlphaHeader();
        _mcvt = new McvtAlpha();
        _mcnrAlpha = new McnrAlpha();
        _mcly = new Chunk();
        _mcrf = new Mcrf();
        _mcsh = new Chunk();
        _mcal = new Mcal();
        _mclq = new Chunk();
    }
    
    public McnkLk ToMcnkLk(List<int> alphaM2Indices, List<int> alphaWmoIndices)
    {
        var cMcnkHeader = new McnkHeader();
        int offsetInHeader = ChunkLettersAndSize + McnkTerrainHeaderSize;

        // Copy over header values from Alpha to LK
        cMcnkHeader.Flags = _mcnkAlphaHeader.Flags;
        cMcnkHeader.IndexX = _mcnkAlphaHeader.IndexX;
        cMcnkHeader.IndexY = _mcnkAlphaHeader.IndexY;
        cMcnkHeader.NLayers = _mcnkAlphaHeader.NLayers;
        cMcnkHeader.M2Number = _mcnkAlphaHeader.M2Number;

        // Calculate offsets for LK format
        cMcnkHeader.McvtOffset = offsetInHeader;
        offsetInHeader = offsetInHeader + ChunkLettersAndSize + _mcvt.GetRealSize();

        cMcnkHeader.McnrOffset = offsetInHeader;
        offsetInHeader = offsetInHeader + ChunkLettersAndSize + _mcnrAlpha.GetRealSize();

        cMcnkHeader.MclyOffset = offsetInHeader;
        offsetInHeader = offsetInHeader + ChunkLettersAndSize + _mcly.GetRealSize();

        cMcnkHeader.McrfOffset = offsetInHeader;
        int mcshOffset = offsetInHeader + ChunkLettersAndSize + _mcrf.GetRealSize();
        offsetInHeader = mcshOffset + ChunkLettersAndSize + _mcsh.GetRealSize();

        cMcnkHeader.McalOffset = offsetInHeader;
        offsetInHeader = offsetInHeader + ChunkLettersAndSize + _mcal.GetRealSize();
        int mclqOffset = offsetInHeader;

        cMcnkHeader.McalSize = _mcal.GetRealSize() + ChunkLettersAndSize;
        offsetInHeader = mcshOffset;

        cMcnkHeader.McshOffset = mcshOffset;
        cMcnkHeader.McshSize = _mcsh.GetRealSize();
        cMcnkHeader.AreaId = _mcnkAlphaHeader.Unknown3; // TODO: I don't know... AreaID should be here, but results are not really convincing
        cMcnkHeader.WmoNumber = _mcnkAlphaHeader.WmoNumber;
        cMcnkHeader.Holes = _mcnkAlphaHeader.Holes;
        cMcnkHeader.GroundEffectsMap1 = _mcnkAlphaHeader.GroundEffectsMap1;
        cMcnkHeader.GroundEffectsMap2 = _mcnkAlphaHeader.GroundEffectsMap2;
        cMcnkHeader.GroundEffectsMap3 = _mcnkAlphaHeader.GroundEffectsMap3;
        cMcnkHeader.GroundEffectsMap4 = _mcnkAlphaHeader.GroundEffectsMap4;
        cMcnkHeader.PredTex = 0;
        cMcnkHeader.NEffectDoodad = 0;
        cMcnkHeader.McseOffset = 0;
        cMcnkHeader.NSndEmitters = 0;
        cMcnkHeader.MclqOffset = mclqOffset;

        if (_mclq.GetRealSize() != 0)
            cMcnkHeader.MclqSize = _mclq.GetRealSize() + ChunkLettersAndSize;
        else
            cMcnkHeader.MclqSize = 0;

        // Calculate positions from ADT coordinates using centralized helper for parity
        var (posX, posY, posZ) = McnkLk.ComputePositionFromAdt(_adtNumber, cMcnkHeader.IndexX, cMcnkHeader.IndexY);
        cMcnkHeader.PosX = posX;
        cMcnkHeader.PosY = posY;
        cMcnkHeader.PosZ = posZ;
        cMcnkHeader.MccvOffset = 0;
        cMcnkHeader.MclvOffset = 0;
        cMcnkHeader.Unused = 0;

        byte[] emptyData = new byte[0];
        Chunk emptyChunk = new Chunk();

        McnrLk cMcnr = _mcnrAlpha.ToMcnrLk();

        Chunk cMcvt = new Chunk("MCVT", 0, emptyData);
        cMcvt = _mcvt.ToMcvt();

        Mcrf cMcrf = _mcrf.UpdateIndicesForLk(alphaM2Indices, _mcnkAlphaHeader.M2Number, alphaWmoIndices, _mcnkAlphaHeader.WmoNumber);

        // Build normalized LK MCLY/MCAL: always emit 8-bit 64x64 alpha per extra layer.
        // 4bpp inputs (2048) are expanded with 63x63 fix (duplicate last col/row) to 4096.
        Chunk cMcly;
        Mcal cMcal;
        if (_mcly.Data != null && _mcly.Data.Length >= _mcnkAlphaHeader.NLayers * 16 && _mcal.Data != null)
        {
            int n = _mcnkAlphaHeader.NLayers;
            var mclySrc = _mcly.Data;
            var mclyOut = new byte[n * 16];
            var mcalOut = new List<byte>(Math.Max(0, (n - 1) * 4096));

            // Read original offsets
            int[] offs = new int[Math.Max(n, 1)];
            for (int i = 0; i < n; i++) offs[i] = BitConverter.ToInt32(mclySrc, i * 16 + 0x08);

            int accum = 0;
            for (int i = 0; i < n; i++)
            {
                int srcBase = i * 16;
                uint textureId = BitConverter.ToUInt32(mclySrc, srcBase + 0x00);
                uint flags = BitConverter.ToUInt32(mclySrc, srcBase + 0x04);
                uint effectId = BitConverter.ToUInt32(mclySrc, srcBase + 0x0C);

                uint outOff = 0;
                if (i > 0)
                {
                    // enforce use_alpha_map and force 8-bit (clear compressed)
                    flags |= 0x100u;
                    int start = offs[i];
                    int end = (i + 1 < n) ? offs[i + 1] : _mcal.Data.Length;
                    int span = Math.Max(0, end - start);

                    flags &= ~0x200u; // not compressed (we emit 8-bit)
                    outOff = (uint)accum;
                    accum += 4096;
                }

                // write out entry
                Array.Copy(BitConverter.GetBytes(textureId), 0, mclyOut, srcBase + 0x00, 4);
                Array.Copy(BitConverter.GetBytes(flags),     0, mclyOut, srcBase + 0x04, 4);
                Array.Copy(BitConverter.GetBytes(outOff),    0, mclyOut, srcBase + 0x08, 4);
                Array.Copy(BitConverter.GetBytes(effectId),  0, mclyOut, srcBase + 0x0C, 4);
            }

            cMcly = new Chunk("MCLY", mclyOut.Length, mclyOut);
            cMcal = new Mcal("MCAL", mcalOut.Count, mcalOut.ToArray());
        }
        else
        {
            // Fallback: pass-through
            cMcly = _mcly;
            cMcal = new Mcal("MCAL", _mcal.Data?.Length ?? 0, _mcal.Data ?? Array.Empty<byte>());
        }

        McnkLk mcnkLk = new McnkLk(cMcnkHeader, cMcvt, emptyChunk, cMcnr, cMcly, cMcrf, _mcsh, cMcal, _mclq, emptyChunk);
        return mcnkLk;
    }

    private static byte[] Expand4BitTo8_Fixed64(byte[] src, int start)
    {
        // Expand 4bpp 64x64 with 63x63 fix:
        // - Rows 0..62: 63 values from data, col63 duplicates col62
        // - Row 63: duplicate entire row 62
        var dst = new byte[64 * 64];
        // helper to read a 4-bit value (LSB nibble first)
        static int ReadNib(byte[] data, int nibIndex)
        {
            int b = data[nibIndex >> 1];
            if ((nibIndex & 1) == 0) return b & 0x0F; else return (b >> 4) & 0x0F;
        }

        // rows 0..62
        for (int y = 0; y < 63; y++)
        {
            int baseNib = y * 64; // layout includes an ignored nibble at col63 which we skip
            int rowOff = y * 64;
            for (int x = 0; x < 63; x++)
            {
                int nib = ReadNib(src, start + baseNib + x);
                byte v = (byte)((nib & 0x0F) * 17); // v | (v << 4)
                dst[rowOff + x] = v;
            }
            // duplicate last col
            dst[rowOff + 63] = dst[rowOff + 62];
        }
        // duplicate last row
        int row62 = 62 * 64;
        int row63 = 63 * 64;
        Buffer.BlockCopy(dst, row62, dst, row63, 64);
        return dst;
    }

    private static void InvertAlpha(byte[] buf)
    {
        for (int i = 0; i < buf.Length; i++) buf[i] = (byte)(255 - buf[i]);
    }

    private static void FlipY64(byte[] buf)
    {
        // In-place vertical flip for 64x64 image
        const int stride = 64;
        byte[] tmp = new byte[stride];
        for (int y = 0; y < 32; y++)
        {
            int top = y * stride;
            int bot = (63 - y) * stride;
            Buffer.BlockCopy(buf, top, tmp, 0, stride);
            Buffer.BlockCopy(buf, bot, buf, top, stride);
            Buffer.BlockCopy(tmp, 0, buf, bot, stride);
        }
    }

    public int GetAlphaAreaId()
    {
        return unchecked((int)_mcnkAlphaHeader.Unknown3);
    }


    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Chunk letters: {Letters}");
        sb.AppendLine($"Chunk givenSize: {GivenSize}");
        
        sb.AppendLine("------------------------------");
        sb.AppendLine("Header:");
        
        sb.AppendLine($"#0x00 Flags\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Flags:X}");
        sb.AppendLine($"#0x04 IndexX\t\t\t\t\t\t: {_mcnkAlphaHeader.IndexX}");
        sb.AppendLine($"#0x08 IndexY\t\t\t\t\t\t: {_mcnkAlphaHeader.IndexY}");
        sb.AppendLine($"#0x0C I don't know\t\t\t\t\t: {_mcnkAlphaHeader.Unknown1}");
        sb.AppendLine($"#0x10 Layers number\t\t\t\t\t: {_mcnkAlphaHeader.NLayers}");
        sb.AppendLine($"#0x14 Doodads number\t\t\t\t: {_mcnkAlphaHeader.M2Number}");
        sb.AppendLine($"#0x18 MCVT offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.McvtOffset:X}");
        sb.AppendLine($"#0x1C MCNR offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.McnrOffset:X}");
        sb.AppendLine($"#0x20 MCLY offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.MclyOffset:X}");
        sb.AppendLine($"#0x24 MCRF offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.McrfOffset:X}");
        sb.AppendLine($"#0x28 MCAL offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.McalOffset:X}");
        sb.AppendLine($"#0x2C MCAL size\t\t\t\t\t\t: {_mcnkAlphaHeader.McalSize}");
        sb.AppendLine($"#0x30 MCSH offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.McshOffset:X}");
        sb.AppendLine($"#0x34 MCSH size\t\t\t\t\t\t: {_mcnkAlphaHeader.McshSize}");
        sb.AppendLine($"#0x38 -> #0x3B\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unknown3:X}");
        sb.AppendLine($"#0x3C Wmo number\t\t\t\t\t: {_mcnkAlphaHeader.WmoNumber}");
        sb.AppendLine($"#0x40 -> Holes #0x43\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Holes:X}");
        sb.AppendLine($"#0x44 -> #0x47 Low Texturing map\t: 0x{_mcnkAlphaHeader.GroundEffectsMap1:X}");
        sb.AppendLine($"#0x48 -> #0x4B Low Texturing map\t: 0x{_mcnkAlphaHeader.GroundEffectsMap2:X}");
        sb.AppendLine($"#0x4C -> #0x4F Low Texturing map\t: 0x{_mcnkAlphaHeader.GroundEffectsMap3:X}");
        sb.AppendLine($"#0x50 -> #0x53 Low Texturing map\t: 0x{_mcnkAlphaHeader.GroundEffectsMap4:X}");
        sb.AppendLine($"#0x54 -> #0x57\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unknown6:X}");
        sb.AppendLine($"#0x58 -> #0x5B\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unknown7:X}");
        sb.AppendLine($"#0x5C MCNK size (minus header)\t\t: {_mcnkAlphaHeader.McnkChunksSize}");
        sb.AppendLine($"#0x60 -> #0x63\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unknown8:X}");
        sb.AppendLine($"#0x64 MCLQ offset\t\t\t\t\t: 0x{_mcnkAlphaHeader.MclqOffset:X}");
        sb.AppendLine($"#0x68 -> #0x6B\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused1:X}");
        sb.AppendLine($"#0x6C -> #0x6F\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused2:X}");
        sb.AppendLine($"#0x70 -> #0x73\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused3:X}");
        sb.AppendLine($"#0x74 -> #0x77\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused4:X}");
        sb.AppendLine($"#0x78 -> #0x7B\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused5:X}");
        sb.AppendLine($"#0x7C -> #0x7F\t\t\t\t\t\t: 0x{_mcnkAlphaHeader.Unused6:X}");
        
        sb.AppendLine("------------------------------");
        sb.AppendLine(_mcvt.ToString());
        sb.AppendLine(_mcnrAlpha.ToString());
        sb.AppendLine(_mcly.ToString());
        sb.AppendLine(_mcrf.ToString());
        
        sb.Append("Doodads indices: ");
        foreach (int index in _mcrf.GetDoodadsIndices(_mcnkAlphaHeader.M2Number))
        {
            sb.Append($"{index} ");
        }
        sb.AppendLine();
        
        sb.Append("Wmos indices: ");
        foreach (int index in _mcrf.GetWmosIndices(_mcnkAlphaHeader.WmoNumber))
        {
            sb.Append($"{index} ");
        }
        sb.AppendLine();
        
        sb.AppendLine("------------------------------");
        sb.AppendLine(_mcsh.ToString());
        sb.AppendLine(_mcal.ToString());
        sb.AppendLine(_mclq.ToString());
        sb.AppendLine("------------------------------");
        
        return sb.ToString();
    }
}
