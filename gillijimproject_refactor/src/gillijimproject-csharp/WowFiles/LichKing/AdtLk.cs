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
    private int _mhdrFlags;
    
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
        
        int offsetInFile = 0;
        int currentChunkSize;
        
        _mver = new Chunk(adtFile, offsetInFile);
        offsetInFile += 4;
        currentChunkSize = BitConverter.ToInt32(adtFile, offsetInFile);
        offsetInFile = 4 + offsetInFile + currentChunkSize;
        
        int mhdrStartOffset = offsetInFile + ChunkLettersAndSize;
        
        _mhdr = new Mhdr(adtFile, offsetInFile);
        _mhdrFlags = _mhdr.GetFlags();
        
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
        List<McnkLk> mcnks)
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
        _mhdrFlags = mhdrFlags;
        
        // Ensure non-null defaults to satisfy definite assignment (will be replaced below)
        _mhdr = new Mhdr();
        _mcin = new Mcin("MCIN", 0, Array.Empty<byte>());
        
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
        
        // Determine if any MCNK has an MCLQ subchunk; MH2O is mutually exclusive with MCLQ
        bool hasAnyMclq = false;
        if (_mcnks != null)
        {
            foreach (var k in _mcnks)
            {
                if (k.GetMclqRealSize() != 0) { hasAnyMclq = true; break; }
            }
        }
        
        // Emit MH2O only if there are no MCLQ chunks in any MCNK
        if (!hasAnyMclq)
        {
            tempData = _mh2o.GetWholeChunk();
            ms.Write(tempData, 0, tempData.Length);
        }
         
        if (_mcnks != null) {
            foreach (var mcnk in _mcnks)
            {
                tempData = mcnk.GetWholeChunk();
                ms.Write(tempData, 0, tempData.Length);
            }
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
        
        if (_mcnks != null) {
            foreach (var mcnk in _mcnks)
            {
                wholeSize += mcnk.GetWholeChunk().Length;
            }
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
        
        // MH2O appears only if there are no MCLQ subchunks
        bool hasAnyMclq = false;
        if (_mcnks != null)
        {
            foreach (var k in _mcnks)
            {
                if (k.GetMclqRealSize() != 0) { hasAnyMclq = true; break; }
            }
        }
        if (!hasAnyMclq)
            mcnkFoundOffset += ChunkLettersAndSize + _mh2o.GetRealSize();
        
        bool offsetsOk = true;
        
        if (_mcnks != null) {
            for (int currentMcnk = 0; currentMcnk < mcnkOffsets.Count; ++currentMcnk)
            {
                offsetsOk = mcnkOffsets[currentMcnk] == mcnkFoundOffset;
                
                if (!offsetsOk)
                    break;
                
                mcnkFoundOffset += _mcnks[currentMcnk].GetWholeChunk().Length;
            }
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
        
        // MH2O appears only if there are no MCLQ subchunks
        bool hasAnyMclq = false;
        if (_mcnks != null)
        {
            foreach (var k in _mcnks)
            {
                if (k.GetMclqRealSize() != 0) { hasAnyMclq = true; break; }
            }
        }
        if (!hasAnyMclq)
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
        
        // Add flags (from stored _mhdrFlags)
        mhdrData.AddRange(BitConverter.GetBytes(_mhdrFlags));
        
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
        
        // MH2O offset written only if MH2O is present; otherwise zero
        byte[] mh2oOffset = BitConverter.GetBytes(offsetInFile - relativeMhdrStart);
        
        // MH2O only when there is no MCLQ in any MCNK
        bool hasAnyMclq = false;
        if (_mcnks != null)
        {
            foreach (var k in _mcnks)
            {
                if (k.GetMclqRealSize() != 0) { hasAnyMclq = true; break; }
            }
        }
        if (!hasAnyMclq)
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
            
            byte[] mcnkSize = (_mcnks != null && currentMcnk < _mcnks.Count)
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
        _mcin = new Mcin("MCIN", mcinFixedSize, mcinData.ToArray());

        // [DEBUG] Temporary diagnostics to validate MCIN/MCNK sizes and offsets
        try
        {
            if (_mcin.Data.Length != mcinFixedSize)
            {
                Console.WriteLine($"[DBG] MCIN data length unexpected: {_mcin.Data.Length} (expected {mcinFixedSize})");
            }
            else
            {
                Console.WriteLine($"[DBG] MCIN data length OK: {_mcin.Data.Length}");
            }

            // Dump first 4 MCIN entries: offset & size
            for (int i = 0; i < 4; i++)
            {
                int baseOff = i * 16;
                int mOff = BitConverter.ToInt32(_mcin.Data, baseOff + 0);
                int mSize = BitConverter.ToInt32(_mcin.Data, baseOff + 4);
                Console.WriteLine($"[DBG] MCIN[{i}] off={mOff} size={mSize}");
            }

            if (_mcnks != null && _mcnks.Count > 0)
            {
                int given = _mcnks[0].GivenSize;
                int serialized = _mcnks[0].GetWholeChunk().Length;
                Console.WriteLine($"[DBG] MCNK[0] GivenSize={given}, SerializedLen={serialized}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[DBG] Exception while dumping MCIN/MCNK debug: {ex.Message}");
        }

        // Add remaining offsets to MHDR
        byte[] emptyOffset = new byte[4];
        
        // MH2O offset written only if MH2O is present; otherwise zero
        if (!hasAnyMclq)
            mhdrData.AddRange(mh2oOffset);
        else
            mhdrData.AddRange(emptyOffset);

        byte[] mfboOffset = BitConverter.GetBytes(0);
        mhdrData.AddRange(mfboOffset);
        
        byte[] mtxfOffset = BitConverter.GetBytes(0);
        mhdrData.AddRange(mtxfOffset);
         
        // Add unused bytes at the end
        const int unused = 16;
        for (int i = 0; i < unused; ++i)
        {
            mhdrData.Add(0);
        }
        
        // Create updated MHDR
        _mhdr = new Mhdr("MHDR", mhdrFixedSize, mhdrData.ToArray());
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
        
        if (_mcnks != null) {
            for (int i = 0; i < _mcnks.Count; ++i)
            {
                sb.AppendLine($"MCNK #{i} : ");
                sb.AppendLine(_mcnks[i].ToString());
            }
        }
        
        return sb.ToString();
    }
}
