using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;

namespace GillijimProject.WowFiles.LichKing;

/// <summary>
/// [PORT] C# port of AdtLk (see lib/gillijimproject/wowfiles/lichking/AdtLk.h)
/// Handles LichKing format ADT files
/// </summary>
public class AdtLk
{
    private const int ChunkLettersAndSize = 8; // 4 bytes for letters + 4 bytes for size
    private const int McnkTerrainHeaderSize = 0x80; // Size of MCNK header in bytes
    
    private string _adtName;
    private Chunk _mver;
    private Mhdr _mhdr;
    private Mcin _mcin;
    private Mh2o _mh2o;
    private Chunk _mtex;
    private Mmdx _mmdx;
    private Mmid _mmid;
    private Mwmo _mwmo;
    private Mwid _mwid;
    private Mddf _mddf;
    private Modf _modf;
    private List<McnkLk> _mcnks;
    private Chunk _mfbo;
    private Chunk _mtxf;
    
    /// <summary>
    /// Constructs an AdtLk from file data and filename
    /// </summary>
    /// <param name="adtFile">The byte array containing the ADT file data</param>
    /// <param name="adtFileName">The name of the ADT file</param>
    public AdtLk(byte[] adtFile, string adtFileName)
    {
        _adtName = adtFileName;
        _mcnks = new List<McnkLk>(256);
        _mh2o = new Mh2o();
        _mfbo = new Chunk("OBFM", 0, Array.Empty<byte>());
        _mtxf = new Chunk("FXTM", 0, Array.Empty<byte>());
        
        int offsetInFile = 0;
        int currentChunkSize;
        
        _mver = new Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = BitConverter.ToInt32(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        
        int mhdrStartOffset = offsetInFile + ChunkLettersAndSize;
        
        _mhdr = new Mhdr(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.McinOffset);
        
        _mcin = new Mcin(adtFile, offsetInFile);
        
        int mh2oSizeInFile = 0;
        
        if (_mhdr.GetOffset(Mhdr.Mh2oOffset) != 0)
        {
            const int lettersSize = 4;
            offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.Mh2oOffset);
            mh2oSizeInFile = BitConverter.ToInt32(adtFile, offsetInFile + lettersSize);
            _mh2o = new Mh2o(adtFile, offsetInFile);
        }
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MtexOffset);
        _mtex = new Chunk(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MmdxOffset);
        _mmdx = new Mmdx(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MmidOffset);
        _mmid = new Mmid(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MwmoOffset);
        _mwmo = new Mwmo(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MwidOffset);
        _mwid = new Mwid(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MddfOffset);
        _mddf = new Mddf(adtFile, offsetInFile);
        
        offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.ModfOffset);
        _modf = new Modf(adtFile, offsetInFile);
        
        List<int> mcnkOffsets = _mcin.GetMcnkOffsets();
        
        for (int currentMcnk = 0; currentMcnk < 256; ++currentMcnk)
        {
            offsetInFile = mcnkOffsets[currentMcnk];
            _mcnks.Add(new McnkLk(adtFile, offsetInFile, McnkTerrainHeaderSize));
        }
        
        if (_mhdr.GetOffset(Mhdr.MfboOffset) != 0)
        {
            offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MfboOffset);
            _mfbo = new Chunk(adtFile, offsetInFile);
        }
        
        if (_mhdr.GetOffset(Mhdr.MtxfOffset) != 0)
        {
            offsetInFile = mhdrStartOffset + BitConverter.ToInt32(adtFile, mhdrStartOffset + Mhdr.MtxfOffset);
            _mtxf = new Chunk(adtFile, offsetInFile);
        }
        
        if (!CheckIntegrity())
        {
            UpdateOrCreateMhdrAndMcin();
        }
    }
    
    /// <summary>
    /// Constructs an AdtLk from file stream and filename
    /// </summary>
    /// <param name="adtFile">The file stream containing the ADT file data</param>
    /// <param name="adtFileName">The name of the ADT file</param>
    public AdtLk(FileStream adtFile, string adtFileName)
        : this(WowChunkedFormat.ReadBytes(adtFile, 0, (int)adtFile.Length), adtFileName) { }
    
    /// <summary>
    /// Constructs an AdtLk from its component chunks
    /// </summary>
    public AdtLk(
        string name,
        Chunk mver,
        int mhdrFlags,
        Mh2o mh2o,
        Chunk mtex,
        Mmdx mmdx,
        Mmid mmid,
        Mwmo mwmo,
        Mwid mwid,
        Mddf mddf,
        Modf modf,
        List<McnkLk> mcnks,
        Chunk mfbo,
        Chunk mtxf)
    {
        _adtName = name;
        _mver = mver;
        _mh2o = mh2o;
        _mtex = mtex;
        _mmdx = mmdx;
        _mmid = mmid;
        _mwmo = mwmo;
        _mwid = mwid;
        _mddf = mddf;
        _modf = modf;
        _mcnks = mcnks;
        _mfbo = mfbo;
        _mtxf = mtxf;
        
        // Ensure non-null defaults to satisfy definite assignment (will be replaced below)
        _mhdr = new Mhdr();
        _mcin = new Mcin("NICM", 0, Array.Empty<byte>());
        
        UpdateOrCreateMhdrAndMcin();
    }
    
    /// <summary>
    /// Writes the ADT to a file using the stored name
    /// </summary>
    public void ToFile()
    {
        string fileName = _adtName;
        ToFile(fileName);
    }
 
    /// <summary>
    /// Writes the ADT to the specified file name
    /// </summary>
    /// <param name="fileName">The file name to write to</param>
    public void ToFile(string fileName)
    {
        // Treat a directory argument as an output directory
        string outPath = fileName;
        if (Directory.Exists(fileName))
        {
            outPath = Path.Combine(fileName, Path.GetFileName(_adtName));
        }
        using var ms = new MemoryStream();
        
        byte[] tempData = _mver.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mhdr.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mcin.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mtex.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mmdx.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mmid.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mwmo.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mwid.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _mddf.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        tempData = _modf.GetWholeChunk();
        ms.Write(tempData, 0, tempData.Length);
        
        if (!_mh2o.IsEmpty())
        {
            tempData = _mh2o.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        foreach (var mcnk in _mcnks)
        {
            tempData = mcnk.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        if (!_mfbo.IsEmpty())
        {
            tempData = _mfbo.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        if (!_mtxf.IsEmpty())
        {
            tempData = _mtxf.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
        
        File.WriteAllBytes(outPath, ms.ToArray());
    }
    
    /// <summary>
    /// Gets the total size of all MCNK chunks
    /// </summary>
    /// <returns>The total size in bytes</returns>
    private int GetMcnksWholeSize()
    {
        int wholeSize = 0;
        
        foreach (var mcnk in _mcnks)
        {
            wholeSize += mcnk.GetWholeChunk().Length;
        }
        
        return wholeSize;
    }
    
    /// <summary>
    /// Checks if the ADT's internal offsets are correct
    /// </summary>
    /// <returns>True if the offsets are correct, false otherwise</returns>
    private bool CheckIntegrity()
    {
        return CheckMhdrOffsets() && CheckMcinOffsets();
    }
    
    /// <summary>
    /// Checks if the MCIN offsets are correct
    /// </summary>
    /// <returns>True if the offsets are correct, false otherwise</returns>
    private bool CheckMcinOffsets()
    {
        List<int> mcnkOffsets = _mcin.GetMcnkOffsets();
        
        int mcnkFoundOffset = ChunkLettersAndSize + _mver.GetRealSize()
            + ChunkLettersAndSize + _mhdr.GetRealSize()
            + ChunkLettersAndSize + _mcin.GetRealSize()
            + ChunkLettersAndSize + _mtex.GetRealSize()
            + ChunkLettersAndSize + _mmdx.GetRealSize()
            + ChunkLettersAndSize + _mmid.GetRealSize()
            + ChunkLettersAndSize + _mwmo.GetRealSize()
            + ChunkLettersAndSize + _mwid.GetRealSize()
            + ChunkLettersAndSize + _mddf.GetRealSize()
            + ChunkLettersAndSize + _modf.GetRealSize();
        
        if (!_mh2o.IsEmpty())
            mcnkFoundOffset += ChunkLettersAndSize + _mh2o.GetRealSize();
        
        bool offsetsOk = true;
        
        for (int currentMcnk = 0; currentMcnk < mcnkOffsets.Count; ++currentMcnk)
        {
            offsetsOk = mcnkOffsets[currentMcnk] == mcnkFoundOffset;
            
            if (!offsetsOk)
                break;
            
            mcnkFoundOffset += _mcnks[currentMcnk].GetWholeChunk().Length;
        }
        
        return offsetsOk;
    }
    
    /// <summary>
    /// Checks if the MHDR offsets are correct
    /// </summary>
    /// <returns>True if the offsets are correct, false otherwise</returns>
    private bool CheckMhdrOffsets()
    {
        const int mhdrStartOffset = 0x14;
        
        int offsetInFile = ChunkLettersAndSize + _mver.GetRealSize()
            + ChunkLettersAndSize + _mhdr.GetRealSize();
        
        bool offsetsOk = _mhdr.GetOffset(Mhdr.McinOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mcin.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MtexOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mtex.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MmdxOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mmdx.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MmidOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mmid.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MwmoOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mwmo.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MwidOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mwid.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.MddfOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _mddf.GetRealSize();
        
        offsetsOk = _mhdr.GetOffset(Mhdr.ModfOffset) + mhdrStartOffset == offsetInFile;
        if (!offsetsOk) return false;
        offsetInFile += ChunkLettersAndSize + _modf.GetRealSize();
        
        if (!_mh2o.IsEmpty())
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.Mh2oOffset) + mhdrStartOffset == offsetInFile;
            if (!offsetsOk) return false;
            offsetInFile += ChunkLettersAndSize + _mh2o.GetRealSize() + GetMcnksWholeSize();
        }
        else
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.Mh2oOffset) == 0;
            if (!offsetsOk) return false;
            offsetInFile += GetMcnksWholeSize();
        }
        
        if (!_mfbo.IsEmpty())
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.MfboOffset) + mhdrStartOffset == offsetInFile;
            if (!offsetsOk) return false;
            offsetInFile += ChunkLettersAndSize + _mfbo.GetRealSize();
        }
        else
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.MfboOffset) == 0;
            if (!offsetsOk) return false;
        }
        
        if (!_mtxf.IsEmpty())
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.MtxfOffset) + mhdrStartOffset == offsetInFile;
            if (!offsetsOk) return false;
        }
        else
        {
            offsetsOk = _mhdr.GetOffset(Mhdr.MtxfOffset) == 0;
            if (!offsetsOk) return false;
        }
        
        return offsetsOk;
    }
    
    /// <summary>
    /// Updates or creates the MHDR and MCIN chunks with correct offsets
    /// </summary>
    private void UpdateOrCreateMhdrAndMcin()
    {
        const int mhdrFixedSize = 64;
        const int mcinFixedSize = 4096;
        const int relativeMhdrStart = 0x14;
        
        List<byte> mhdrData = new List<byte>();
        byte[] emptyData = new byte[4];
        
        // Add flags
        if (_mhdr != null && _mhdr.GetRealSize() != 0)
        {
            byte[] flags = BitConverter.GetBytes(_mhdr.GetFlags());
            mhdrData.AddRange(flags);
        }
        else
        {
            mhdrData.AddRange(emptyData);
        }
        
        // Calculate all offsets
        int offsetInFile = ChunkLettersAndSize + _mver.GetRealSize() + ChunkLettersAndSize + mhdrFixedSize;
        byte[] mcinOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mcinOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + mcinFixedSize;
        byte[] mtexOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mtexOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mtex.GetRealSize();
        byte[] mmdxOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mmdxOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mmdx.GetRealSize();
        byte[] mmidOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mmidOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mmid.GetRealSize();
        byte[] mwmoOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mwmoOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mwmo.GetRealSize();
        byte[] mwidOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mwidOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mwid.GetRealSize();
        byte[] mddfOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(mddfOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _mddf.GetRealSize();
        byte[] modfOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        mhdrData.AddRange(modfOffset);
        
        offsetInFile = offsetInFile + ChunkLettersAndSize + _modf.GetRealSize();
        
        byte[] mh2oOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        
        if (!_mh2o.IsEmpty())
        {
            offsetInFile = offsetInFile + ChunkLettersAndSize + _mh2o.GetRealSize();
        }
        
        // Create MCIN data with MCNK offsets
        List<byte> mcinData = new List<byte>();
        const int unusedMcinBytes = 8;
        
        for (int currentMcnk = 0; currentMcnk < 256; ++currentMcnk)
        {
            byte[] mcnkOffset = BitConverter.GetBytes(offsetInFile);
            mcinData.AddRange(mcnkOffset);
            
            byte[] mcnkSize = _mcnks != null && currentMcnk < _mcnks.Count
                ? BitConverter.GetBytes(_mcnks[currentMcnk].GetWholeChunk().Length)
                : BitConverter.GetBytes(0);
                
            mcinData.AddRange(mcnkSize);
            
            // Add unused bytes
            for (int i = 0; i < unusedMcinBytes; ++i)
            {
                mcinData.Add(0);
            }
            
            if (_mcnks != null && currentMcnk < _mcnks.Count)
            {
                offsetInFile += _mcnks[currentMcnk].GetWholeChunk().Length;
            }
        }
        
        // Create updated MCIN
        _mcin = new Mcin("NICM", mcinFixedSize, mcinData.ToArray());
        
        // Add remaining offsets to MHDR
        byte[] emptyOffset = new byte[4];
        
        byte[] mfboOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        if (_mfbo != null && _mfbo.GetRealSize() != 0)
        {
            mhdrData.AddRange(mfboOffset);
        }
        else
        {
            mhdrData.AddRange(emptyOffset);
        }
        
        if (_mh2o != null && _mh2o.GetRealSize() != 0)
        {
            mhdrData.AddRange(mh2oOffset);
        }
        else
        {
            mhdrData.AddRange(emptyOffset);
        }
        
        offsetInFile = offsetInFile + (_mfbo != null ? ChunkLettersAndSize + _mfbo.GetRealSize() : 0);
        byte[] mtxfOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        if (_mtxf != null && _mtxf.GetRealSize() != 0)
        {
            mhdrData.AddRange(mtxfOffset);
        }
        else
        {
            mhdrData.AddRange(emptyOffset);
        }
        
        // Add unused bytes at the end
        const int unused = 16;
        for (int i = 0; i < unused; ++i)
        {
            mhdrData.Add(0);
        }
        
        // Create updated MHDR
        _mhdr = new Mhdr("RDHM", mhdrFixedSize, mhdrData.ToArray());
    }
    
    /// <summary>
    /// Gets the MHDR flags
    /// </summary>
    /// <returns>The MHDR flags as int</returns>
    public int GetMhdrFlags()
    {
        return _mhdr.GetOffset(0);
    }
    
    /// <summary>
    /// Gets a list of all M2 model names
    /// </summary>
    /// <returns>List of M2 model names</returns>
    public List<string> GetAllM2Names()
    {
        return _mmdx.GetM2Names();
    }
    
    /// <summary>
    /// Gets a list of all WMO names
    /// </summary>
    /// <returns>List of WMO names</returns>
    public List<string> GetAllWmoNames()
    {
        return _mwmo.GetWmoNames();
    }
    
    /// <summary>
    /// Returns a string representation of the AdtLk
    /// </summary>
    /// <returns>String representation</returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();
        
        sb.AppendLine(_adtName);
        sb.AppendLine("------------------------------");
        sb.AppendLine(_mver.ToString());
        sb.AppendLine(_mhdr.ToString());
        sb.AppendLine(_mcin.ToString());
        sb.AppendLine(_mtex.ToString());
        sb.AppendLine(_mmdx.ToString());
        sb.AppendLine(_mmid.ToString());
        sb.AppendLine(_mwmo.ToString());
        sb.AppendLine(_mwid.ToString());
        sb.AppendLine(_mddf.ToString());
        sb.AppendLine(_modf.ToString());
        sb.AppendLine(_mh2o.ToString());
        
        for (int i = 0; i < _mcnks.Count; ++i)
        {
            sb.AppendLine($"MCNK #{i} : ");
            sb.AppendLine(_mcnks[i].ToString());
        }
        
        sb.AppendLine(_mfbo.ToString());
        sb.AppendLine(_mtxf.ToString());
        
        return sb.ToString();
    }
}
