using System.Buffers.Binary;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4BinaryAdtPatcher
{
    private const float MapOrigin = 17066.666f;
    private const int MddfEntrySize = 36;
    private const int ModfEntrySize = 64;
    private const float TileSize = 533.33333f;
    private const float ChunkSize = TileSize / 16f;

    public static byte[] Patch(byte[] sourceBytes, int tileX, int tileY,
        IReadOnlyList<Pm4AdtM2Placement> m2Placements)
    {
        using var ms = new MemoryStream(sourceBytes);
        using var br = new BinaryReader(ms);
        var chunks = new List<(long offset, int size)>();
        while (ms.Position + 8 <= ms.Length)
        {
            br.ReadBytes(4);
            int size = br.ReadInt32();
            if (size < 0 || ms.Position + size > ms.Length) break;
            long off = ms.Position - 8;
            chunks.Add((off, size));
            ms.Position = off + 8 + size;
        }

        using var outMs = new MemoryStream();

        long mhdrPos = -1;
        long mcinStart = -1;
        var mcnkPoses = new List<(long pos, int size)>();

        foreach (var (off, size) in chunks)
        {
            string tag = Encoding.ASCII.GetString(sourceBytes, (int)off, 4);
            byte[] payload = new byte[size];
            Buffer.BlockCopy(sourceBytes, (int)off + 8, payload, 0, size);

            switch (tag)
            {
                case "REVM":
                    WriteChunk(outMs, "REVM", payload);
                    break;
                case "RDHM":
                    // Preserve original MHDR as-is
                    WriteChunk(outMs, "RDHM", payload);
                    break;
                case "NICM":
                    // Preserve original MCIN as-is
                    WriteChunk(outMs, "NICM", payload);
                    break;
                case "XETM":
                    // Preserve original MTEX as-is
                    WriteChunk(outMs, "XETM", payload);
                    break;
                case "XDMM":
                case "DIMM":
                case "OMWM":
                case "DIWM":
                case "FDDM":
                case "FDOM":
                    // Preserve all original placement chunks as-is
                    WriteChunk(outMs, tag, payload);
                    break;
                case "KNCM":
                    // Preserve original MCNK as-is
                    WriteChunk(outMs, "KNCM", payload);
                    break;
                case "OBFM":
                    WriteChunk(outMs, "OBFM", payload);
                    break;
                default:
                    WriteChunk(outMs, tag, payload);
                    break;
            }
        }

        byte[] result = outMs.ToArray();

        // Since we're preserving all original chunks, no need to patch MHDR or MCIN
        // The original offsets are already correct

        return result;
    }

    private static void PatchAndWriteMcnk(MemoryStream ms, byte[] source, int mcnkOff, int mcnkSize,
        int tileX, int tileY,
        IReadOnlyList<Pm4AdtM2Placement> m2,
        IReadOnlyList<Pm4AdtWmoPlacement> wmo)
    {
        int cx = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(mcnkOff + 8 + 0x04));
        int cy = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(mcnkOff + 8 + 0x08));
        int ofsMcal = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(mcnkOff + 8 + 0x24));
        int szMcal = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(mcnkOff + 8 + 0x28));

        float ax = tileX * TileSize + cx * ChunkSize;
        float ay = tileY * TileSize + cy * ChunkSize;
        float wminX = MapOrigin - ay - ChunkSize, wmaxX = MapOrigin - ay;
        float wminY = MapOrigin - ax - ChunkSize, wmaxY = MapOrigin - ax;

        var m2i = new List<int>();
        for (int i = 0; i < m2.Count; i++) { var p = m2[i].Position; if (p.X >= wminX && p.X < wmaxX && p.Y >= wminY && p.Y < wmaxY) m2i.Add(i); }
        var wmoi = new List<int>();
        for (int i = 0; i < wmo.Count; i++) { var p = wmo[i].Position; if (p.X >= wminX && p.X < wmaxX && p.Y >= wminY && p.Y < wmaxY) wmoi.Add(i); }

        long chunkPos = ms.Position;
        ms.Write(source, mcnkOff, 8 + mcnkSize);
        var buf = ms.GetBuffer();
        long hdr = chunkPos + 8;

        if (ofsMcal > 0) BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)hdr + 0x28), szMcal);
        BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)hdr + 0x10), m2i.Count);
        BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)hdr + 0x38), wmoi.Count);

        int plStart = (int)chunkPos + 8, plEnd = plStart + mcnkSize;
        int frcmOff = -1, frcmSize = -1;
        for (int p = plStart + 128; p + 8 <= plEnd;)
        {
            string t = Encoding.ASCII.GetString(buf, p, 4);
            int s = BitConverter.ToInt32(buf, p + 4);
            if (t == "FRCM") { frcmOff = p; frcmSize = 8 + s; break; }
            if (s < 0) break;
            p += 8 + s;
        }

        byte[] newFrcm;
        if (m2i.Count > 0 || wmoi.Count > 0)
        {
            using var mem = new MemoryStream();
            mem.Write(Encoding.ASCII.GetBytes("FRCM"));
            int ds = 8 + m2i.Count * 4 + wmoi.Count * 4;
            mem.Write(BitConverter.GetBytes(ds));
            mem.Write(BitConverter.GetBytes(m2i.Count));
            mem.Write(BitConverter.GetBytes(wmoi.Count));
            foreach (int x in m2i) mem.Write(BitConverter.GetBytes(x));
            foreach (int x in wmoi) mem.Write(BitConverter.GetBytes(x));
            newFrcm = mem.ToArray();
        }
        else newFrcm = [];

        int delta = 0;
        if (newFrcm.Length == 0 && frcmOff >= 0)
        {
            int after = frcmOff + frcmSize;
            Array.Copy(buf, after, buf, frcmOff, plEnd - after);
            delta = -frcmSize;
            ms.SetLength(ms.Length + delta);
        }
        else if (newFrcm.Length > 0 && frcmOff >= 0)
        {
            delta = newFrcm.Length - frcmSize;
            if (delta != 0)
            {
                Array.Copy(buf, frcmOff + frcmSize, buf, frcmOff + newFrcm.Length, plEnd - frcmOff - frcmSize);
                ms.SetLength(ms.Length + delta);
            }
            newFrcm.CopyTo(buf, frcmOff);
        }
        else if (newFrcm.Length > 0 && frcmOff < 0) { }

        int newSize = mcnkSize + delta;
        BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)chunkPos + 4), newSize);
        if (delta != 0 && ofsMcal > 0)
        {
            BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)hdr + 0x24), ofsMcal + delta);
            int ofsMcsh = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(mcnkOff + 8 + 0x2C));
            if (ofsMcsh > 0)
                BinaryPrimitives.WriteInt32LittleEndian(buf.AsSpan((int)hdr + 0x2C), ofsMcsh + delta);
        }
    }

    private static (byte[] mmdx, byte[] mmid, byte[] mddf) BuildM2PlacementChunks(
        IReadOnlyList<Pm4AdtM2Placement> m2Placements)
    {
        var modelNames = new List<string>();
        var modelNameOffsets = new List<int>();
        int stringOffset = 0;
        foreach (var p in m2Placements)
        {
            string path = p.ModelPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant();
            if (!modelNames.Contains(path))
            {
                modelNameOffsets.Add(stringOffset);
                modelNames.Add(path);
                stringOffset += Encoding.UTF8.GetByteCount(path) + 1;
            }
        }
        byte[] mmdxData = new byte[stringOffset];
        int off = 0;
        for (int i = 0; i < modelNames.Count; i++)
        {
            int bytes = Encoding.UTF8.GetBytes(modelNames[i], 0, modelNames[i].Length, mmdxData, off);
            mmdxData[off + bytes] = 0;
            off += bytes + 1;
        }
        byte[] mmidData = new byte[modelNames.Count * 4];
        for (int i = 0; i < modelNames.Count; i++)
            BinaryPrimitives.WriteInt32LittleEndian(mmidData.AsSpan(i * 4), modelNameOffsets[i]);

        byte[] mddfData = new byte[m2Placements.Count * MddfEntrySize];
        for (int i = 0; i < m2Placements.Count; i++)
        {
            var p = m2Placements[i];
            int nameId = modelNames.IndexOf(p.ModelPath.Replace('\\', '/').TrimStart('/').ToLowerInvariant());
            int eo = i * MddfEntrySize;
            BinaryPrimitives.WriteInt32LittleEndian(mddfData.AsSpan(eo), nameId);
            BinaryPrimitives.WriteInt32LittleEndian(mddfData.AsSpan(eo + 4), p.UniqueId);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 8), MapOrigin - p.Position.Y);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 12), p.Position.Z);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 16), MapOrigin - p.Position.X);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 20), p.Rotation.X);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 24), p.Rotation.Z);
            BinaryPrimitives.WriteSingleLittleEndian(mddfData.AsSpan(eo + 28), p.Rotation.Y);
            BinaryPrimitives.WriteUInt16LittleEndian(mddfData.AsSpan(eo + 32), (ushort)MathF.Round(p.Scale * 1024f));
            BinaryPrimitives.WriteUInt16LittleEndian(mddfData.AsSpan(eo + 34), 0);
        }

        return (mmdxData, mmidData, mddfData);
    }

    private static void WriteChunk(MemoryStream ms, string tag, byte[] payload)
    {
        ms.Write(Encoding.ASCII.GetBytes(tag));
        ms.Write(BitConverter.GetBytes(payload.Length));
        ms.Write(payload);
    }
}
