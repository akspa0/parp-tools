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
    private readonly Mcly _mcly;
    private readonly Mcrf _mcrf;
    private readonly Mcsh _mcsh;
    private readonly Mcal _mcal;
    private readonly Mclq _mclq;

    public McnkAlpha(byte[] wholeFile, int offsetInFile, int adtNum) : base(wholeFile, offsetInFile)
    {
        _adtNumber = adtNum;

        int headerStartOffset = offsetInFile + ChunkLettersAndSize;
        var headerData = new Span<byte>(wholeFile, headerStartOffset, McnkTerrainHeaderSize).ToArray();
        _mcnkAlphaHeader = Util.ByteArrayToStruct<McnkAlphaHeader>(headerData);

        // Sub-chunks are located via offsets relative to the start of the MCNK data (after FourCC and size).
        int dataStartOffset = offsetInFile + ChunkLettersAndSize;

        // Read MCVT
        int mcvtOffset = dataStartOffset + _mcnkAlphaHeader.McvtOffset;
        var mcvtData = new Span<byte>(wholeFile, mcvtOffset, McvtSize).ToArray();
        _mcvt = new McvtAlpha("MCVT", McvtSize, mcvtData);

        // Read MCNR
        int mcnrOffset = dataStartOffset + _mcnkAlphaHeader.McnrOffset;
        var mcnrData = new Span<byte>(wholeFile, mcnrOffset, McnrSize).ToArray();
        _mcnrAlpha = new McnrAlpha("MCNR", McnrSize, mcnrData);

        // Read MCLY
        int mclyOffset = dataStartOffset + _mcnkAlphaHeader.MclyOffset;
        _mcly = new Mcly(wholeFile, mclyOffset);

        // Read MCRF
        int mcrfOffset = dataStartOffset + _mcnkAlphaHeader.McrfOffset;
        _mcrf = new Mcrf(wholeFile, mcrfOffset);

        // Read MCSH
        int mcshOffset = dataStartOffset + _mcnkAlphaHeader.McshOffset;
        var mcshData = new Span<byte>(wholeFile, mcshOffset, _mcnkAlphaHeader.McshSize).ToArray();
        _mcsh = new Mcsh("MCSH", _mcnkAlphaHeader.McshSize, mcshData);

        // Read MCAL
        int mcalOffset = dataStartOffset + _mcnkAlphaHeader.McalOffset;
        _mcal = new Mcal(wholeFile, mcalOffset);

        // Read MCLQ
        int mclqOffset = dataStartOffset + _mcnkAlphaHeader.MclqOffset;
        int mclqSize = _mcnkAlphaHeader.McnkChunksSize - _mcnkAlphaHeader.MclqOffset;
        if (mclqSize > 0)
        {
            var mclqData = new Span<byte>(wholeFile, mclqOffset, mclqSize).ToArray();
            _mclq = new Mclq("MCLQ", mclqSize, mclqData);
        }
        else
        {
            _mclq = new Mclq("MCLQ", 0, Array.Empty<byte>());
        }
    }
    
    public McnkAlpha(byte[] wholeFile, int offsetInFile) : base(wholeFile, offsetInFile) 
    { 
        // Initialize non-nullable fields to prevent CS8618 warnings
        _adtNumber = 0;
        _mcnkAlphaHeader = new McnkAlphaHeader();
        _mcvt = new McvtAlpha();
        _mcnrAlpha = new McnrAlpha();
        _mcly = new Mcly();
        _mcrf = new Mcrf();
        _mcsh = new Mcsh();
        _mcal = new Mcal();
        _mclq = new Mclq();
    }
    
    public McnkAlpha(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) 
    { 
        // Initialize non-nullable fields to prevent CS8618 warnings
        _adtNumber = 0;
        _mcnkAlphaHeader = new McnkAlphaHeader();
        _mcvt = new McvtAlpha();
        _mcnrAlpha = new McnrAlpha();
        _mcly = new Mcly();
        _mcrf = new Mcrf();
        _mcsh = new Mcsh();
        _mcal = new Mcal();
        _mclq = new Mclq();
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
        _mcly = new Mcly();
        _mcrf = new Mcrf();
        _mcsh = new Mcsh();
        _mcal = new Mcal();
        _mclq = new Mclq();
    }
    
    public LichKing.McnkLk ToMcnkLk(Dictionary<int, int> alphaM2Indices, Dictionary<int, int> alphaWmoIndices)
    {
        var cMcnkHeader = _mcnkAlphaHeader.ToMcnkHeader();
        var cMcvt = _mcvt.ToMcvt();
        var emptyMccv = new Chunk("MCCV", 0, Array.Empty<byte>());
        var cMcnr = _mcnrAlpha.ToMcnrLk();
        var cMcrf = _mcrf.UpdateIndicesForLk(alphaM2Indices, (int)_mcnkAlphaHeader.M2Number, alphaWmoIndices, (int)_mcnkAlphaHeader.WmoNumber);
        var emptyMcse = new Mcse("MCSE", 0, Array.Empty<byte>());

        return new LichKing.McnkLk(cMcnkHeader, cMcvt, emptyMccv, cMcnr, _mcly, cMcrf, _mcsh, _mcal, _mclq, emptyMcse);
    }

    public override byte[] GetPayload()
    {
        throw new NotImplementedException();
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
