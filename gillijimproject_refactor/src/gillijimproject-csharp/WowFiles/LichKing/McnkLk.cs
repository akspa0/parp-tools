using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;
using Util = GillijimProject.Utilities.Utilities;

namespace GillijimProject.WowFiles.LichKing;

public class McnkLk : Mcnk
{
    private McnkHeader _mcnkHeader;
    private readonly Mcvt _mcvt;
    private readonly Chunk _mccv; // Remains a generic chunk as it has no unique fields
    private readonly McnrLk _mcnr;
    private readonly Mcly _mcly;
    private readonly Mcrf _mcrf;
    private readonly Mcsh _mcsh;
    private readonly Mcal _mcal;
    private readonly Mclq _mclq;
    private readonly Mcse _mcse;

    public McnkLk(byte[] adtFile, int offsetInFile) : base(adtFile, offsetInFile)
    {
        int headerStartOffset = offsetInFile;
        offsetInFile += 8; // Skip MCNK FourCC and size

        byte[] headerContent = new byte[128];
        Array.Copy(adtFile, offsetInFile, headerContent, 0, 128);
        _mcnkHeader = Util.ByteArrayToStruct<McnkHeader>(headerContent);

        _mcvt = new Mcvt(adtFile, headerStartOffset + _mcnkHeader.McvtOffset);
        _mccv = new Chunk(adtFile, headerStartOffset + _mcnkHeader.MccvOffset);
        _mcnr = new McnrLk(adtFile, headerStartOffset + _mcnkHeader.McnrOffset);
        _mcly = new Mcly(adtFile, headerStartOffset + _mcnkHeader.MclyOffset);
        _mcrf = new Mcrf(adtFile, headerStartOffset + _mcnkHeader.McrfOffset);
        _mcsh = new Mcsh(adtFile, headerStartOffset + _mcnkHeader.McshOffset);
        _mcal = new Mcal(adtFile, headerStartOffset + _mcnkHeader.McalOffset, _mcnkHeader.McalSize - 8);
        _mclq = new Mclq(adtFile, headerStartOffset + _mcnkHeader.MclqOffset);
        _mcse = new Mcse(adtFile, headerStartOffset + _mcnkHeader.McseOffset);
    }

    // [PORT] Compatibility overload; C++ passes header size (0x80). We compute sizes from data; value unused.
    public McnkLk(byte[] adtFile, int offsetInFile, int headerSize) : this(adtFile, offsetInFile) { }

    public McnkLk(McnkHeader mcnkHeader, Mcvt mcvt, Chunk mccv, McnrLk mcnr, Mcly mcly, Mcrf mcrf, Mcsh mcsh, Mcal mcal, Mclq mclq, Mcse mcse)
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

    public override byte[] GetPayload()
    {
        using var ms = new MemoryStream();

        // Define fixed order for sub-chunk emission
        var subChunks = new List<Chunk> { _mcvt, _mccv, _mcnr, _mcly, _mcrf, _mcsh, _mcal, _mclq, _mcse };

        // Recompute header offsets/sizes based on fixed emission order and current subchunk sizes
        int currentOffset = 128; // Start after the header itself

        _mcnkHeader.McvtOffset = currentOffset; currentOffset += _mcvt.GetSize();
        _mcnkHeader.MccvOffset = currentOffset; currentOffset += _mccv.GetSize();
        _mcnkHeader.McnrOffset = currentOffset; currentOffset += _mcnr.GetSize();
        _mcnkHeader.MclyOffset = currentOffset; currentOffset += _mcly.GetSize();
        _mcnkHeader.McrfOffset = currentOffset; currentOffset += _mcrf.GetSize();

        _mcnkHeader.McshOffset = currentOffset;
        _mcnkHeader.McshSize = _mcsh.GetRealSize(); // Mcsh size is payload only
        currentOffset += _mcsh.GetSize();

        _mcnkHeader.McalOffset = currentOffset;
        _mcnkHeader.McalSize = _mcal.GetSize(); // Mcal size includes its header
        currentOffset += _mcal.GetSize();

        _mcnkHeader.MclqOffset = currentOffset;
        _mcnkHeader.MclqSize = _mclq.GetRealSize() > 0 ? _mclq.GetSize() : 0;
        currentOffset += _mclq.GetSize();

        _mcnkHeader.McseOffset = currentOffset;

        // Write header
        byte[] headerBytes = Util.StructToByteArray(_mcnkHeader);
        ms.Write(headerBytes, 0, headerBytes.Length);

        // Write sub-chunks
        foreach (var chunk in subChunks)
        {
            byte[] chunkBytes = chunk.GetWholeChunk();
            ms.Write(chunkBytes, 0, chunkBytes.Length);
        }

        return ms.ToArray();
    }

    // [PORT] Wrapper used by AdtLk integrity decisions (MH2O vs MCLQ)
    public int GetMclqRealSize() => _mclq.GetRealSize();

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine(base.ToString()); // Includes FourCC and size
        sb.AppendLine("------------------------------");
        sb.AppendLine(_mcvt.ToString());
        sb.AppendLine(_mccv.ToString());
        sb.AppendLine(_mcnr.ToString());
        sb.AppendLine(_mcly.ToString());
        sb.AppendLine(_mcrf.ToString());
        sb.AppendLine(_mcsh.ToString());
        sb.AppendLine(_mcal.ToString());
        sb.AppendLine(_mclq.ToString());
        sb.AppendLine(_mcse.ToString());
        sb.AppendLine("------------------------------");
        return sb.ToString();
    }
}
