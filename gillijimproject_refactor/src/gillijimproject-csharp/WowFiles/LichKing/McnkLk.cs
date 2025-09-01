using System;
using System.IO;
using System.Text;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;
using Util = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles.LichKing;

/// <summary>
/// [PORT] C# port of McnkLk (see lib/gillijimproject/wowfiles/lichking/McnkLk.h)
/// Handles LichKing format MCNK chunks
/// </summary>
public class McnkLk : Mcnk
{
    private new const int ChunkLettersAndSize = 8; // 4 bytes for letters + 4 bytes for size
    private new const int McnkTerrainHeaderSize = 0x80; // Size of MCNK header in bytes
    
    private McnkHeader _mcnkHeader;
    private Chunk _mcvt;
    private Chunk? _mccv;
    private McnrLk _mcnr;
    private Chunk _mcly;
    private Mcrf _mcrf;
    private Chunk? _mcsh;
    private Mcal _mcal;
    private Chunk? _mclq;
    private Chunk? _mcse;

    /// <summary>
    /// Constructs a McnkLk from a file array at the given offset with the given header size
    /// </summary>
    /// <param name="adtFile">The byte array containing the file data</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    /// <param name="headerSize">Size of the header</param>
    public McnkLk(byte[] adtFile, int offsetInFile, int headerSize) : base(adtFile, offsetInFile, McnkTerrainHeaderSize)
    {
        int headerStartOffset = offsetInFile;
        
        offsetInFile += ChunkLettersAndSize;
        
        byte[] headerContent = new byte[McnkTerrainHeaderSize];
        Array.Copy(adtFile, offsetInFile, headerContent, 0, McnkTerrainHeaderSize);
        _mcnkHeader = Util.ByteArrayToStruct<McnkHeader>(headerContent);
        
        offsetInFile = headerStartOffset + _mcnkHeader.McvtOffset;
        _mcvt = new Chunk(adtFile, offsetInFile);
        
        if (_mcnkHeader.MccvOffset != 0)
        {
            offsetInFile = headerStartOffset + _mcnkHeader.MccvOffset;
            _mccv = new Chunk(adtFile, offsetInFile);
        }
        
        offsetInFile = headerStartOffset + _mcnkHeader.McnrOffset;
        _mcnr = new McnrLk(adtFile, offsetInFile);
        
        offsetInFile = headerStartOffset + _mcnkHeader.MclyOffset;
        _mcly = new Chunk(adtFile, offsetInFile);
        
        offsetInFile = headerStartOffset + _mcnkHeader.McrfOffset;
        _mcrf = new Mcrf(adtFile, offsetInFile);
        
        // Note : We don't check the 0x1 Mcnk header flag since it's not set on some maps, 
        // even though there is a shadow map (e.g. MonasteryInstances)
        if ((_mcnkHeader.McshOffset != 0) && (_mcnkHeader.McshOffset != _mcnkHeader.McalOffset))
        {
            offsetInFile = headerStartOffset + _mcnkHeader.McshOffset;
            _mcsh = new Chunk(adtFile, offsetInFile);
        }
        
        offsetInFile = headerStartOffset + _mcnkHeader.McalOffset;
        int alphaSize = _mcnkHeader.McalSize - ChunkLettersAndSize;
        _mcal = new Mcal(adtFile, offsetInFile, alphaSize);
        
        if (_mcnkHeader.MclqOffset != 0)
        {
            offsetInFile = headerStartOffset + _mcnkHeader.MclqOffset;
            _mclq = new Chunk(adtFile, offsetInFile);
        }
        
        if (_mcnkHeader.McseOffset != 0)
        {
            offsetInFile = headerStartOffset + _mcnkHeader.McseOffset;
            _mcse = new Chunk(adtFile, offsetInFile);
        }
    }
    
    /// <summary>
    /// Constructs a McnkLk from a file stream at the given offset with the given header size
    /// </summary>
    /// <param name="adtFile">The file stream to read from</param>
    /// <param name="offsetInFile">Offset in the file where the chunk starts</param>
    /// <param name="headerSize">Size of the header</param>
    public McnkLk(FileStream adtFile, int offsetInFile, int headerSize) 
        : this(WowChunkedFormat.ReadBytes(adtFile, offsetInFile, (int)adtFile.Length - offsetInFile), 0, headerSize) { }
    
    /// <summary>
    /// Constructs a McnkLk from chunk data
    /// </summary>
    /// <param name="letters">The FourCC code</param>
    /// <param name="givenSize">The size of the chunk</param>
    /// <param name="chunkData">The chunk data</param>
    public McnkLk(string letters, int givenSize, byte[] chunkData) : base(letters, givenSize, chunkData) 
    { 
        // Initialize non-nullable fields to prevent CS8618 warnings
        _mcnkHeader = new McnkHeader();
        _mcvt = new Chunk();
        _mcnr = new McnrLk();
        _mcly = new Chunk();
        _mcrf = new Mcrf();
        _mcsh = new Chunk();
        _mcal = new Mcal();
        _mclq = new Chunk();
        _mcse = new Chunk();
    }
    
    /// <summary>
    /// Constructs a McnkLk from all its components
    /// </summary>
    public McnkLk(
        McnkHeader mcnkHeader,
        Chunk mcvt,
        Chunk? mccv,
        McnrLk mcnr,
        Chunk mcly,
        Mcrf mcrf,
        Chunk? mcsh,
        Mcal mcal,
        Chunk? mclq,
        Chunk? mcse) : base("KNCM", CalculateGivenSize(mcnkHeader, mcvt, mccv, mcnr, mcly, mcrf, mcsh, mcal, mclq, mcse), Array.Empty<byte>())
    {
        _mcnkHeader = mcnkHeader;
        _mcvt = mcvt;
        _mccv = mccv;
        _mcnr = mcnr;
        _mcly = mcly;
        _mcrf = mcrf;
        _mcsh = mcsh;
        _mcal = mcal;
        _mclq = mclq;
        _mcse = mcse;
    }
    
    /// <summary>
    /// [PORT] Default parameterless constructor
    /// </summary>
    public McnkLk() : base("KNCM", 0, Array.Empty<byte>())
    {
        _mcnkHeader = new McnkHeader();
        _mcvt = new Chunk();
        _mccv = new Chunk();
        _mcnr = new McnrLk();
        _mcly = new Chunk();
        _mcrf = new Mcrf();
        _mcsh = new Chunk();
        _mcal = new Mcal();
        _mclq = new Chunk();
        _mcse = new Chunk();
    }
    
    /// <summary>
    /// [PORT] Helper method to calculate the size for the constructor
    /// </summary>
    private static int CalculateGivenSize(
        McnkHeader mcnkHeader,
        Chunk mcvt,
        Chunk? mccv,
        McnrLk mcnr,
        Chunk mcly,
        Mcrf mcrf,
        Chunk? mcsh,
        Mcal mcal,
        Chunk? mclq,
        Chunk? mcse)
    {
        int size = McnkTerrainHeaderSize
            + mcvt.GetRealSize()
            + ChunkLettersAndSize
            + mcnr.GetRealSize()
            + ChunkLettersAndSize
            + mcrf.GetRealSize()
            + ChunkLettersAndSize;
        
        if (mccv != null && !mccv.IsEmpty())
            size += ChunkLettersAndSize + mccv.GetRealSize();
        
        if (mcly != null && !mcly.IsEmpty())
            size += ChunkLettersAndSize + mcly.GetRealSize();
        
        if (mcsh != null && !mcsh.IsEmpty())
            size += ChunkLettersAndSize + mcsh.GetRealSize();
        
        if (mcal != null && !mcal.IsEmpty())
            size += ChunkLettersAndSize + mcal.GetRealSize();
        
        if (mclq != null && !mclq.IsEmpty())
            size += ChunkLettersAndSize + mclq.GetRealSize();
        
        if (mcse != null && !mcse.IsEmpty())
            size += ChunkLettersAndSize + mcse.GetRealSize();
            
        return size;
    }
    
    /// <summary>
    /// Gets the serialized chunk data
    /// </summary>
    /// <returns>Byte array containing the whole chunk</returns>
    public new byte[] GetWholeChunk()
    {
        using MemoryStream ms = new MemoryStream();
        
        // Write chunk letters
        byte[] tempData = Encoding.ASCII.GetBytes(Letters);
        ms.Write(tempData, 0, tempData.Length);
        
        // Write chunk size
        tempData = BitConverter.GetBytes(GivenSize);
        ms.Write(tempData, 0, tempData.Length);
        
        // Write header
        byte[] headerContent = Util.StructToByteArray(_mcnkHeader);
        ms.Write(headerContent, 0, McnkTerrainHeaderSize);
        
        // Write MCVT
        tempData = _mcvt.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        // Write MCCV if not empty
        if (_mccv != null && !_mccv.IsEmpty())
        {
            tempData = _mccv.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        // Write MCNR
        tempData = _mcnr.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        // Write MCLY if not empty
        if (_mcly != null && !_mcly.IsEmpty())
        {
            tempData = _mcly.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        // Write MCRF
        tempData = _mcrf.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        // Write MCSH if not empty
        if (_mcsh != null && !_mcsh.IsEmpty())
        {
            tempData = _mcsh.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        // Write MCAL if not empty
        if (!_mcal.IsEmpty())
        {
            tempData = _mcal.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        // Write MCLQ if not empty
        if (_mclq != null && !_mclq.IsEmpty())
        {
            tempData = _mclq.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        // Write MCSE if not empty
        if (_mcse != null && !_mcse.IsEmpty())
        {
            tempData = _mcse.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        return ms.ToArray();
    }
    
    /// <summary>
    /// Returns a string representation of the McnkLk chunk
    /// </summary>
    /// <returns>String representation</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        
        sb.AppendLine($"Chunk letters: {Letters}");
        sb.AppendLine($"Chunk givenSize: {GivenSize}");
        
        sb.AppendLine("------------------------------");
        
        sb.AppendLine(_mcvt?.ToString() ?? "MCVT: null");
        sb.AppendLine(_mccv?.ToString() ?? "MCCV: null");
        sb.AppendLine(_mcnr?.ToString() ?? "MCNR: null");
        sb.AppendLine(_mcly?.ToString() ?? "MCLY: null");
        sb.AppendLine(_mcrf?.ToString() ?? "MCRF: null");
        sb.AppendLine(_mcsh?.ToString() ?? "MCSH: null");
        sb.AppendLine(_mcal?.ToString() ?? "MCAL: null");
        sb.AppendLine(_mclq?.ToString() ?? "MCLQ: null");
        sb.AppendLine(_mcse?.ToString() ?? "MCSE: null");
        
        sb.AppendLine("------------------------------");
        
        return sb.ToString();
    }
}
