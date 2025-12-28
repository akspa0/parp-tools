using System.Text;
using WoWMapConverter.Core.Converters;
using WoWMapConverter.Core.Formats.Liquids;

namespace WoWMapConverter.Core.Builders;

/// <summary>
/// Builds Alpha MCNK chunks from LK MCNK data.
/// Handles terrain, textures, shadows, and MH2O→MCLQ liquid conversion.
/// </summary>
public static class AlphaMcnkBuilder
{
    private const int McnkHeaderSize = 0x80; // 128 bytes
    private const int ChunkHeaderSize = 8;   // FourCC + size

    /// <summary>
    /// Build Alpha MCNK from LK MCNK data.
    /// </summary>
    public static byte[] BuildFromLk(
        byte[] lkBytes,
        int mcnkOffset,
        byte[]? texBytes,
        int texMcnkOffset,
        IReadOnlyList<int> doodadRefs,
        IReadOnlyList<int> wmoRefs,
        LkToAlphaOptions? opts = null)
    {
        // Read LK MCNK header
        var lkHeader = ReadLkMcnkHeader(lkBytes, mcnkOffset);
        int mcnkSize = BitConverter.ToInt32(lkBytes, mcnkOffset + 4);
        int subStart = mcnkOffset + ChunkHeaderSize + McnkHeaderSize;
        int subEnd = Math.Min(mcnkOffset + ChunkHeaderSize + mcnkSize, lkBytes.Length);

        // Extract subchunks from LK MCNK
        byte[]? mcvtData = ExtractSubchunk(lkBytes, subStart, subEnd, "MCVT");
        byte[]? mcnrData = ExtractSubchunk(lkBytes, subStart, subEnd, "MCNR");
        byte[]? mclyData = ExtractSubchunkFromOffset(lkBytes, mcnkOffset, lkHeader.MclyOffset, subEnd) 
                       ?? ExtractSubchunk(lkBytes, subStart, subEnd, "MCLY");
        byte[]? mcalData = ExtractSubchunkFromOffset(lkBytes, mcnkOffset, lkHeader.McalOffset, subEnd)
                       ?? ExtractSubchunk(lkBytes, subStart, subEnd, "MCAL");
        byte[]? mcshData = ExtractSubchunkFromOffset(lkBytes, mcnkOffset, lkHeader.McshOffset, subEnd)
                       ?? ExtractSubchunk(lkBytes, subStart, subEnd, "MCSH");

        // Try to get MCLY/MCAL from _tex.adt if not in root
        if ((mclyData == null || mclyData.Length == 0) && texBytes != null && texMcnkOffset > 0)
        {
            int texMcnkSize = BitConverter.ToInt32(texBytes, texMcnkOffset + 4);
            int texSubStart = texMcnkOffset + ChunkHeaderSize + McnkHeaderSize;
            int texSubEnd = Math.Min(texMcnkOffset + ChunkHeaderSize + texMcnkSize, texBytes.Length);
            mclyData = ExtractSubchunk(texBytes, texSubStart, texSubEnd, "MCLY");
            mcalData = ExtractSubchunk(texBytes, texSubStart, texSubEnd, "MCAL");
        }

        // Build MCLQ from MH2O if liquids conversion enabled
        byte[]? mclqData = null;
        if (opts?.ConvertLiquids != false)
        {
            mclqData = TryBuildMclqFromMh2o(lkBytes, lkHeader.IndexX, lkHeader.IndexY);
        }

        // Build Alpha MCNK
        using var ms = new MemoryStream();

        // Calculate subchunk sizes
        int mcvtSize = mcvtData?.Length ?? 580;  // Default MCVT size
        int mcnrSize = mcnrData?.Length ?? 448;  // Default MCNR size (435 + 13 padding)
        int mclySize = mclyData?.Length ?? 0;
        int mcalSize = mcalData?.Length ?? 0;
        int mcshSize = mcshData?.Length ?? 0;
        int mclqSize = mclqData?.Length ?? 0;

        // Build MCRF from doodad/WMO refs
        var mcrfData = BuildMcrfData(doodadRefs, wmoRefs);
        int mcrfSize = mcrfData.Length;

        // Calculate offsets (relative to MCNK payload start, after 128-byte header)
        int currentOffset = 0;

        int mcvtOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mcvtSize;

        int mcnrOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mcnrSize;

        int mclyOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mclySize;

        int mcrfOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mcrfSize;

        int mcshOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mcshSize;

        int mcalOffset = currentOffset;
        currentOffset += mcalSize; // MCAL has no chunk header in Alpha

        int mclqOffset = currentOffset;
        currentOffset += ChunkHeaderSize + mclqSize;

        int mcnkChunksSize = currentOffset;

        // Build Alpha MCNK header
        var alphaHeader = new byte[McnkHeaderSize];
        BitConverter.GetBytes(lkHeader.Flags).CopyTo(alphaHeader, 0x00);
        BitConverter.GetBytes(lkHeader.IndexX).CopyTo(alphaHeader, 0x04);
        BitConverter.GetBytes(lkHeader.IndexY).CopyTo(alphaHeader, 0x08);
        // 0x0C: Unknown1 (float) - leave as 0
        BitConverter.GetBytes(lkHeader.NLayers).CopyTo(alphaHeader, 0x10);
        BitConverter.GetBytes(doodadRefs.Count).CopyTo(alphaHeader, 0x14);
        BitConverter.GetBytes(mcvtOffset).CopyTo(alphaHeader, 0x18);
        BitConverter.GetBytes(mcnrOffset).CopyTo(alphaHeader, 0x1C);
        BitConverter.GetBytes(mclyOffset).CopyTo(alphaHeader, 0x20);
        BitConverter.GetBytes(mcrfOffset).CopyTo(alphaHeader, 0x24);
        BitConverter.GetBytes(mcalOffset).CopyTo(alphaHeader, 0x28);
        BitConverter.GetBytes(mcalSize).CopyTo(alphaHeader, 0x2C);
        BitConverter.GetBytes(mcshOffset).CopyTo(alphaHeader, 0x30);
        BitConverter.GetBytes(mcshSize).CopyTo(alphaHeader, 0x34);
        BitConverter.GetBytes(lkHeader.AreaId).CopyTo(alphaHeader, 0x38); // Unknown3 in Alpha = AreaID
        BitConverter.GetBytes(wmoRefs.Count).CopyTo(alphaHeader, 0x3C);
        BitConverter.GetBytes(lkHeader.Holes).CopyTo(alphaHeader, 0x40);
        BitConverter.GetBytes(lkHeader.GroundEffectsMap1).CopyTo(alphaHeader, 0x44);
        BitConverter.GetBytes(lkHeader.GroundEffectsMap2).CopyTo(alphaHeader, 0x48);
        BitConverter.GetBytes(lkHeader.GroundEffectsMap3).CopyTo(alphaHeader, 0x4C);
        BitConverter.GetBytes(lkHeader.GroundEffectsMap4).CopyTo(alphaHeader, 0x50);
        // 0x54-0x5B: Unknown6, Unknown7 - leave as 0
        BitConverter.GetBytes(mcnkChunksSize).CopyTo(alphaHeader, 0x5C);
        // 0x60: Unknown8 - leave as 0
        BitConverter.GetBytes(mclqOffset).CopyTo(alphaHeader, 0x64);
        // 0x68-0x7F: Unused - leave as 0

        // Write MCNK chunk
        ms.Write(Encoding.ASCII.GetBytes("KNCM")); // MCNK reversed
        ms.Write(BitConverter.GetBytes(McnkHeaderSize + mcnkChunksSize));
        ms.Write(alphaHeader);

        // Write subchunks
        WriteSubchunk(ms, "MCVT", mcvtData ?? new byte[580]);
        WriteSubchunk(ms, "MCNR", mcnrData ?? new byte[448]);
        WriteSubchunk(ms, "MCLY", mclyData ?? Array.Empty<byte>());
        WriteSubchunk(ms, "MCRF", mcrfData);
        WriteSubchunk(ms, "MCSH", mcshData ?? Array.Empty<byte>());
        
        // MCAL has no chunk header in Alpha - write raw data
        if (mcalData != null && mcalData.Length > 0)
            ms.Write(mcalData);

        WriteSubchunk(ms, "MCLQ", mclqData ?? Array.Empty<byte>());

        return ms.ToArray();
    }

    private static LkMcnkHeader ReadLkMcnkHeader(byte[] bytes, int offset)
    {
        int dataStart = offset + ChunkHeaderSize;
        return new LkMcnkHeader
        {
            Flags = BitConverter.ToInt32(bytes, dataStart + 0x00),
            IndexX = BitConverter.ToInt32(bytes, dataStart + 0x04),
            IndexY = BitConverter.ToInt32(bytes, dataStart + 0x08),
            NLayers = BitConverter.ToInt32(bytes, dataStart + 0x0C),
            M2Number = BitConverter.ToInt32(bytes, dataStart + 0x10),
            McvtOffset = BitConverter.ToInt32(bytes, dataStart + 0x14),
            McnrOffset = BitConverter.ToInt32(bytes, dataStart + 0x18),
            MclyOffset = BitConverter.ToInt32(bytes, dataStart + 0x1C),
            McrfOffset = BitConverter.ToInt32(bytes, dataStart + 0x20),
            McalOffset = BitConverter.ToInt32(bytes, dataStart + 0x24),
            McalSize = BitConverter.ToInt32(bytes, dataStart + 0x28),
            McshOffset = BitConverter.ToInt32(bytes, dataStart + 0x2C),
            McshSize = BitConverter.ToInt32(bytes, dataStart + 0x30),
            AreaId = BitConverter.ToInt32(bytes, dataStart + 0x34),
            WmoNumber = BitConverter.ToInt32(bytes, dataStart + 0x38),
            Holes = BitConverter.ToInt32(bytes, dataStart + 0x3C),
            GroundEffectsMap1 = BitConverter.ToInt32(bytes, dataStart + 0x40),
            GroundEffectsMap2 = BitConverter.ToInt32(bytes, dataStart + 0x44),
            GroundEffectsMap3 = BitConverter.ToInt32(bytes, dataStart + 0x48),
            GroundEffectsMap4 = BitConverter.ToInt32(bytes, dataStart + 0x4C),
            MclqOffset = BitConverter.ToInt32(bytes, dataStart + 0x58),
            MclqSize = BitConverter.ToInt32(bytes, dataStart + 0x5C)
        };
    }

    private static byte[]? ExtractSubchunk(byte[] bytes, int start, int end, string fourCC)
    {
        string reversed = new string(fourCC.Reverse().ToArray());

        for (int i = start; i + 8 <= end;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);
            int dataStart = i + 8;
            int next = dataStart + size + ((size & 1) == 1 ? 1 : 0);

            if (fcc == reversed && size > 0 && dataStart + size <= bytes.Length)
            {
                var data = new byte[size];
                Buffer.BlockCopy(bytes, dataStart, data, 0, size);
                return data;
            }

            if (next <= i || dataStart + size > end) break;
            i = next;
        }

        return null;
    }

    private static byte[]? ExtractSubchunkFromOffset(byte[] bytes, int mcnkOffset, int relOffset, int maxEnd)
    {
        if (relOffset <= 0) return null;

        int absOffset = mcnkOffset + relOffset;
        if (absOffset < 0 || absOffset + 8 > bytes.Length) return null;

        // Verify it's a valid chunk
        int size = BitConverter.ToInt32(bytes, absOffset + 4);
        if (size <= 0 || absOffset + 8 + size > bytes.Length) return null;

        var data = new byte[size];
        Buffer.BlockCopy(bytes, absOffset + 8, data, 0, size);
        return data;
    }

    private static byte[] BuildMcrfData(IReadOnlyList<int> doodadRefs, IReadOnlyList<int> wmoRefs)
    {
        // MCRF: doodad indices followed by WMO indices (all uint32)
        int totalCount = doodadRefs.Count + wmoRefs.Count;
        var data = new byte[totalCount * 4];
        int pos = 0;

        foreach (var idx in doodadRefs)
        {
            BitConverter.GetBytes(idx).CopyTo(data, pos);
            pos += 4;
        }

        foreach (var idx in wmoRefs)
        {
            BitConverter.GetBytes(idx).CopyTo(data, pos);
            pos += 4;
        }

        return data;
    }

    private static byte[]? TryBuildMclqFromMh2o(byte[] lkBytes, int chunkX, int chunkY)
    {
        // Find MH2O chunk in ADT
        int mh2oOffset = FindChunk(lkBytes, "MH2O");
        if (mh2oOffset < 0) return null;

        int mh2oSize = BitConverter.ToInt32(lkBytes, mh2oOffset + 4);
        if (mh2oSize <= 0) return null;

        // Parse MH2O chunk
        var mh2oData = new byte[mh2oSize];
        Buffer.BlockCopy(lkBytes, mh2oOffset + 8, mh2oData, 0, mh2oSize);

        var mh2o = Mh2oChunk.Parse(mh2oData);
        int chunkIndex = chunkY * 16 + chunkX;

        // Get instances for this chunk
        var instances = mh2o.GetInstancesForChunk(chunkIndex).ToList();
        if (instances.Count == 0) return null;

        // Use first instance (most common case)
        var instance = instances[0];
        var attributes = mh2o.Attributes[chunkIndex];

        // Convert MH2O to MCLQ
        var mclq = LiquidConverter.Mh2oToMclq(instance, attributes);
        var mclqType = LiquidConverter.MapLiquidTypeIdToMclqType(instance.LiquidTypeId);

        return mclq.ToBytes(mclqType);
    }

    private static int FindChunk(byte[] bytes, string fourCC)
    {
        string reversed = new string(fourCC.Reverse().ToArray());

        for (int i = 0; i + 8 <= bytes.Length;)
        {
            string fcc = Encoding.ASCII.GetString(bytes, i, 4);
            int size = BitConverter.ToInt32(bytes, i + 4);

            if (fcc == reversed)
                return i;

            int next = i + 8 + size + ((size & 1) == 1 ? 1 : 0);
            if (next <= i) break;
            i = next;
        }

        return -1;
    }

    private static void WriteSubchunk(MemoryStream ms, string fourCC, byte[] data)
    {
        ms.Write(Encoding.ASCII.GetBytes(new string(fourCC.Reverse().ToArray())));
        ms.Write(BitConverter.GetBytes(data.Length));
        ms.Write(data);
    }

    private struct LkMcnkHeader
    {
        public int Flags;
        public int IndexX;
        public int IndexY;
        public int NLayers;
        public int M2Number;
        public int McvtOffset;
        public int McnrOffset;
        public int MclyOffset;
        public int McrfOffset;
        public int McalOffset;
        public int McalSize;
        public int McshOffset;
        public int McshSize;
        public int AreaId;
        public int WmoNumber;
        public int Holes;
        public int GroundEffectsMap1;
        public int GroundEffectsMap2;
        public int GroundEffectsMap3;
        public int GroundEffectsMap4;
        public int MclqOffset;
        public int MclqSize;
    }
}
