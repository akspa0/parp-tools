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
    
    // [PORT] When true, this MCNK is a zero-byte placeholder used to pad to 256 entries without writing data
    private readonly bool _isPlaceholder;

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
        _isPlaceholder = false;
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
        Chunk? mcse) : base("MCNK", CalculateGivenSize(mcnkHeader, mcvt, mccv, mcnr, mcly, mcrf, mcsh, mcal, mclq, mcse), Array.Empty<byte>())
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
    public McnkLk() : base("MCNK", 0, Array.Empty<byte>())
    {
        _isPlaceholder = false;
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
    /// [PORT] Factory for zero-byte placeholder MCNK entries.
    /// </summary>
    public static McnkLk CreatePlaceholder()
    {
        var m = new McnkLk();
        // flip private readonly via constructor pattern
        // Since fields are readonly, use dedicated private ctor
        return new McnkLk(true);
    }

    // Private ctor for placeholders
    private McnkLk(bool placeholder) : base("MCNK", 0, Array.Empty<byte>())
    {
        _isPlaceholder = placeholder;
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
    /// Exposes whether this MCNK is a placeholder (zero-byte) entry.
    /// </summary>
    public bool IsPlaceholder => _isPlaceholder;

    /// <summary>
    /// [PORT] Compute LK world-space position from ADT number and MCNK local indices.
    /// Mirrors the existing Alpha -> LK conversion formula for PosX/PosY/PosZ.
    /// </summary>
    /// <param name="adtNumber">Absolute ADT tile index (0..4095, row-major 64x64)</param>
    /// <param name="indexX">MCNK X index within ADT (0..15)</param>
    /// <param name="indexY">MCNK Y index within ADT (0..15)</param>
    /// <returns>Tuple (posX, posY, posZ)</returns>
    public static (float posX, float posY, float posZ) ComputePositionFromAdt(int adtNumber, int indexX, int indexY)
    {
        int adtX = adtNumber % 64;
        int adtY = adtNumber / 64;

        // Note: Keep exact constants and sign to preserve parity with original logic
        float posY = (((533.33333f / 16f) * indexX) + (533.33333f * adtX) - (533.33333f * 32f)) * -1f;
        float posX = (((533.33333f / 16f) * indexY) + (533.33333f * adtY) - (533.33333f * 32f)) * -1f;
        float posZ = 0f;
        return (posX, posY, posZ);
    }
    
    /// <summary>
    /// Helper method to calculate the size for the constructor
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
        int size = McnkTerrainHeaderSize;

        // Always-present chunks
        size += mcvt.GetRealSize();
        size += mcnr.GetRealSize();
        size += mcrf.GetRealSize();

        // Optional chunks
        if (mccv != null && !mccv.IsEmpty())
            size += mccv.GetRealSize();

        if (mcly != null && !mcly.IsEmpty())
            size += mcly.GetRealSize();

        if (mcsh != null && !mcsh.IsEmpty())
            size += mcsh.GetRealSize();

        if (mcal != null && !mcal.IsEmpty())
            size += mcal.GetRealSize();

        if (mclq != null && !mclq.IsEmpty())
            size += mclq.GetRealSize();

        if (mcse != null && !mcse.IsEmpty())
            size += mcse.GetRealSize();

        return size;
    }
    
    /// <summary>
    /// Gets the serialized chunk data
    /// </summary>
    /// <returns>Byte array containing the whole chunk</returns>
    public new byte[] GetWholeChunk()
    {
        if (_isPlaceholder) return Array.Empty<byte>();
        using MemoryStream ms = new MemoryStream();
        
        // Write chunk letters
        var reversedLetters = new string(new[] { Letters[3], Letters[2], Letters[1], Letters[0] });
        byte[] tempData = Encoding.ASCII.GetBytes(reversedLetters);
        ms.Write(tempData, 0, tempData.Length);
        
        // Write chunk size
        tempData = BitConverter.GetBytes(GivenSize);
        ms.Write(tempData, 0, tempData.Length);
        
        // [PORT] Recompute header subchunk offsets/sizes before writing header.
        // Offsets are relative to the start of the MCNK chunk (including letters).
        var hdr = _mcnkHeader;
        int offset = ChunkLettersAndSize + McnkTerrainHeaderSize; // first subchunk starts after letters+size+header

        // Reset optional fields
        hdr.MccvOffset = 0;
        hdr.MclyOffset = 0;
        hdr.McshOffset = 0; hdr.McshSize = 0;
        hdr.McalOffset = 0; hdr.McalSize = 0;
        hdr.MclqOffset = 0; hdr.MclqSize = 0;
        hdr.McseOffset = 0;

        // Required/present chunks in write order
        hdr.McvtOffset = offset; offset += _mcvt.GetWholeChunk().Length;

        if (_mccv != null && !_mccv.IsEmpty()) { hdr.MccvOffset = offset; offset += _mccv.GetWholeChunk().Length; }

        hdr.McnrOffset = offset; offset += _mcnr.GetWholeChunk().Length;

        if (_mcly != null && !_mcly.IsEmpty()) { hdr.MclyOffset = offset; offset += _mcly.GetWholeChunk().Length; }

        hdr.McrfOffset = offset; offset += _mcrf.GetWholeChunk().Length;

        if (_mcsh != null && !_mcsh.IsEmpty()) { hdr.McshOffset = offset; hdr.McshSize = _mcsh.GetWholeChunk().Length; offset += _mcsh.GetWholeChunk().Length; }

        if (_mcal != null && !_mcal.IsEmpty()) { hdr.McalOffset = offset; hdr.McalSize = _mcal.GetWholeChunk().Length; offset += _mcal.GetWholeChunk().Length; }

        if (_mclq != null && !_mclq.IsEmpty()) { hdr.MclqOffset = offset; hdr.MclqSize = _mclq.GetWholeChunk().Length; offset += _mclq.GetWholeChunk().Length; }

        if (_mcse != null && !_mcse.IsEmpty()) { hdr.McseOffset = offset; offset += _mcse.GetWholeChunk().Length; }

        // Serialize updated header
        byte[] headerContent = Util.StructToByteArray(hdr);
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
        if (_mcal != null && !_mcal.IsEmpty())
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
    
    /// <summary>
    /// [PORT] Indicates whether this MCNK contains an MCLQ subchunk
    /// </summary>
    public bool HasMclq()
    {
        if (_isPlaceholder) return false;
        return _mclq != null && !_mclq.IsEmpty();
    }
}
