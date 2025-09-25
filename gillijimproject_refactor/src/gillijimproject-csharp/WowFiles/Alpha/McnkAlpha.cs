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

        // Read MCAL
        offsetInFile = headerStartOffset + McnkTerrainHeaderSize + ChunkLettersAndSize + _mcnkAlphaHeader.McalOffset;
        byte[] mcalData = Util.GetByteArrayFromFile(adtFile, offsetInFile, _mcnkAlphaHeader.McalSize);
        _mcal = new Mcal("MCAL", _mcnkAlphaHeader.McalSize, mcalData);

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

        McnkLk mcnkLk = new McnkLk(cMcnkHeader, cMcvt, emptyChunk, cMcnr, _mcly, cMcrf, _mcsh, _mcal, _mclq, emptyChunk);
        return mcnkLk;
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
