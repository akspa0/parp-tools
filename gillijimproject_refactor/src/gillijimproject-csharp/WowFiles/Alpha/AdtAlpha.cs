using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles; // for Chunk
using GillijimProject.WowFiles.LichKing;
using U = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles.Alpha;

/// <summary>
/// [PORT] Minimal C# port of AdtAlpha (see lib/gillijimproject/wowfiles/alpha/AdtAlpha.{h,cpp})
/// This establishes constructor parity and basic helpers (coords), leaving full parsing for later.
/// </summary>
public class AdtAlpha : WowFiles.WowChunkedFormat
{
    private readonly int _adtNumber;
    private readonly string _adtFileName;
    private readonly int _x;
    private readonly int _y;
    private readonly Chunk _mhdr;
    private readonly Mcin _mcin;
    private readonly Chunk _mtex;
    private readonly Mddf _mddf;
    private readonly Modf _modf;
    private readonly Chunk? _mfbo; // Optional
    private readonly Chunk? _mh2o; // Optional
    private readonly Chunk? _mtxf; // Optional
    private readonly List<McnkAlpha> _mcnksAlpha;

    /// <summary>
    /// [PORT] Constructor parity (wdtAlphaName, offsetInFile, adtNum).
    /// Offset not used yet; full parsing is deferred until dependent classes are ported.
    /// </summary>
    public AdtAlpha(string adtFullPath, int adtNum)
    {
        _adtNumber = adtNum;
        _x = adtNum % 64;
        _y = adtNum / 64;
        _adtFileName = adtFullPath;

        // [PORT] Read entire file to byte array for robust parsing.
        var fileData = File.ReadAllBytes(_adtFileName);

        // Find all chunks in the file first.
        var chunks = new Dictionary<string, Chunk>();
        int offset = 0;
        while (offset < fileData.Length)
        {
            var chunk = new Chunk(fileData, offset);
            chunks[chunk.Letters] = chunk;
            int pad = (chunk.GivenSize & 1) == 1 ? 1 : 0;
            offset += ChunkLettersAndSize + chunk.GivenSize + pad;
        }

        _mhdr = chunks["MHDR"];
        int mhdrDataStart = (int)(Array.IndexOf(fileData, _mhdr.Data) - _mhdr.Data.Length + _mhdr.GivenSize); // Risky but works for now

        _mcin = new Mcin(fileData, mhdrDataStart + _mhdr.GetOffset(4));
        _mtex = new Chunk(fileData, mhdrDataStart + _mhdr.GetOffset(8));
        _mddf = new Mddf(fileData, mhdrDataStart + _mhdr.GetOffset(20));
        _modf = new Modf(fileData, mhdrDataStart + _mhdr.GetOffset(28));

        // Optional chunks
        chunks.TryGetValue("MFBO", out _mfbo);
        chunks.TryGetValue("MH2O", out _mh2o);
        chunks.TryGetValue("MTXF", out _mtxf);

        // Parse all 256 MCNK chunks using MCIN offsets
        var mcnkOffsets = _mcin.GetMcnkOffsets();
        _mcnksAlpha = new List<McnkAlpha>(capacity: 256);
        for (int i = 0; i < 256; ++i)
        {
            _mcnksAlpha.Add(new McnkAlpha(fileData, mcnkOffsets[i], _adtNumber));
        }
    }

    /// <summary>
    /// [PORT] X tile coordinate in the 64x64 grid.
    /// </summary>
    public int GetXCoord() => _x;

    /// <summary>
    /// [PORT] Y tile coordinate in the 64x64 grid.
    /// </summary>
    public int GetYCoord() => _y;

    /// <summary>
    /// [PORT] Convert to LichKing ADT placeholder until full conversion is ported.
    /// Produces a minimal ADT with MVER (0x12) and MHDR (64 bytes zeroed).
    /// </summary>
    public AdtLk ToAdtLk(List<string> mdnmFileNames, List<string> monmFileNames)
    {
        var mverData = new byte[] { 0x12, 0x00, 0x00, 0x00 };
        var cMver = new Chunk("MVER", mverData.Length, mverData);

        // Use existing MH2O or create an empty one if it doesn't exist.
        var cMh2o = _mh2o != null ? new Mh2o(_mh2o.Letters, _mh2o.GivenSize, _mh2o.Data) : new Mh2o();

        // Build LK index chunks from alpha MDDF/MODF and WDT name tables
        var alphaM2IndexMap = _mddf.GetM2IndicesForMmdx();
        var alphaM2ListOrdered = new List<int>(new int[alphaM2IndexMap.Count]);
        foreach (var kvp in alphaM2IndexMap) alphaM2ListOrdered[kvp.Value] = kvp.Key;
        var cMmdx = new Mmdx(alphaM2ListOrdered, mdnmFileNames);
        var cMmid = new Mmid(cMmdx.GetIndicesForMmid());

        var alphaWmoIndices = _modf.GetWmoIndicesForMwmo();
        var cMwmo = new Mwmo(alphaWmoIndices, monmFileNames);
        var cMwid = new Mwid(cMwmo.GetIndicesForMwid());

        // Copy and remap MDDF/MODF indices for LK
        var cMddf = new Mddf("MDDF", _mddf.Data.Length, (byte[])_mddf.Data.Clone());
        cMddf.UpdateIndicesForLk(alphaM2ListOrdered);

        var cModf = new Modf("MODF", _modf.Data.Length, (byte[])_modf.Data.Clone());
        cModf.UpdateIndicesForLk(alphaWmoIndices);

        // Convert parsed Alpha MCNKs to LK MCNKs with updated indices
        var cMcnks = new List<McnkLk>(capacity: 256);
        // [PORT] Build WMO value->position map once for all MCNK conversions
        var alphaWmoIndexMap = new Dictionary<int, int>(alphaWmoIndices.Count);
        for (int i = 0; i < alphaWmoIndices.Count; ++i) alphaWmoIndexMap[alphaWmoIndices[i]] = i;
        for (int currentMcnk = 0; currentMcnk < 256; ++currentMcnk)
        {
            if (_mcnksAlpha[currentMcnk] != null)
            {
                cMcnks.Add(_mcnksAlpha[currentMcnk].ToMcnkLk(alphaM2IndexMap, alphaWmoIndexMap));
            }
        }

        return new AdtLk(
            _adtFileName,
            cMver,
            0,            // mhdrFlags
            cMh2o,
            _mtex,
            cMmdx,
            cMmid,
            cMwmo,
            cMwid,
            cMddf,
            cModf,
            cMcnks
        );
    }
}
