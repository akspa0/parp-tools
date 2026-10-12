using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Maps.AdtAhdr;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Spec 237 (experimental): serializes an <see cref="AdtAhdrTile"/> as a DAT v26 file. The chunk order is the one
/// observed in every corpus file: MVER AHDR ALOC AOCH AVTX ANRM ATEX×n ADOO×n ACNK×256 ADST×n ACVT, and inside ACNK:
/// header, ALYR×n (each with its AMAP), ASHD, ACDO×n. Object records are written from their decoded fields, not from
/// <see cref="AdtAhdrObjectDefinition.RawRecord"/>, so a byte-exact rewrite of a real file proves the decode keeps
/// every byte. No padding (chunks are unpadded in v26).
/// </summary>
public static class AdtAhdrWriter
{
    private const int AlyrFixedSize = 0x20;
    private const int AochSize = 0x800;

    public static byte[] Write(AdtAhdrTile tile)
    {
        ArgumentNullException.ThrowIfNull(tile);
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        if (tile.MverVersion is uint mver)
            WriteChunk(writer, "MVER", w => w.Write(mver));

        WriteChunk(writer, "AHDR", w =>
        {
            w.Write(tile.Version);
            w.Write((uint)tile.VerticesX);
            w.Write((uint)tile.VerticesY);
            w.Write((uint)tile.ChunksX);
            w.Write((uint)tile.ChunksY);
            for (int i = 0; i < 11; i++)
                w.Write(i < tile.HeaderReserved.Length ? tile.HeaderReserved[i] : 0u);
        });

        if (tile.Aloc is { } aloc)
            WriteChunk(writer, "ALOC", w => { foreach (uint value in aloc) w.Write(value); });

        WriteChunk(writer, "AOCH", w => w.Write(tile.AochRaw ?? new byte[AochSize]));
        WriteChunk(writer, "AVTX", w =>
        {
            foreach (float h in tile.OuterHeights) w.Write(h);
            foreach (float h in tile.InnerHeights) w.Write(h);
        });

        if (tile.NormalsRaw is { } normals)
            WriteChunk(writer, "ANRM", w => w.Write(normals));

        foreach (string texture in tile.TextureNames)
            WriteChunk(writer, "ATEX", w => WriteCString(w, texture));
        foreach (string model in tile.ModelNames)
            WriteChunk(writer, "ADOO", w => WriteCString(w, model));

        foreach (AdtAhdrChunk chunk in tile.Chunks)
            WriteChunk(writer, "ACNK", w => WriteAcnk(w, chunk));

        foreach (AdtAhdrModelFileReference reference in tile.ModelFileReferences)
        {
            WriteChunk(writer, "ADST", w =>
            {
                w.Write(reference.UniqueId);
                w.Write(reference.FileDataId);
                w.Write(reference.Field8);
            });
        }

        if (tile.VertexShadingRaw is { } shading)
            WriteChunk(writer, "ACVT", w => w.Write(shading));

        writer.Flush();
        return stream.ToArray();
    }

    private static void WriteAcnk(BinaryWriter w, AdtAhdrChunk chunk)
    {
        var header = new byte[0x40];
        chunk.HeaderRaw.AsSpan(0, Math.Min(0x40, chunk.HeaderRaw.Length)).CopyTo(header);
        BinaryPrimitives.WriteInt32LittleEndian(header, chunk.IndexX);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(4), chunk.IndexY);
        w.Write(header);

        foreach (AdtAhdrLayer layer in chunk.Layers)
        {
            WriteChunk(w, "ALYR", lw =>
            {
                lw.Write(layer.TextureIndex);
                lw.Write(layer.Flags);
                lw.Write(new byte[AlyrFixedSize - 8]);
                if (layer.AlphaMap is { } alpha)
                    WriteChunk(lw, "AMAP", aw => aw.Write(alpha));
            });
        }

        if (chunk.ShadowRaw is { } shadow)
            WriteChunk(w, "ASHD", sw => sw.Write(shadow));

        foreach (AdtAhdrObjectDefinition obj in chunk.Objects)
            WriteChunk(w, "ACDO", ow => WriteAcdo(ow, obj));
    }

    private static void WriteAcdo(BinaryWriter w, AdtAhdrObjectDefinition obj)
    {
        w.Write(obj.ModelIndex);
        WriteVector(w, obj.LocalPositionInches);
        WriteVector(w, obj.RotationDegrees);
        w.Write(obj.Scale);
        w.Write(obj.Field20);
        w.Write(obj.Field24);
        w.Write(obj.Field28);
        w.Write(obj.UniqueId);
        w.Write((uint)obj.TrailingValues.Length);
        w.Write(obj.Field34);
        foreach (uint value in obj.TrailingValues)
            w.Write(value);
    }

    private static void WriteChunk(BinaryWriter writer, string id, Action<BinaryWriter> body)
    {
        using var payload = new MemoryStream();
        using (var payloadWriter = new BinaryWriter(payload, Encoding.ASCII, leaveOpen: true))
            body(payloadWriter);

        // Chunk ids are stored reversed on disk ("REVM" for MVER).
        writer.Write((byte)id[3]);
        writer.Write((byte)id[2]);
        writer.Write((byte)id[1]);
        writer.Write((byte)id[0]);
        writer.Write((uint)payload.Length);
        payload.Position = 0;
        payload.CopyTo(writer.BaseStream);
    }

    // MEASURED: ATEX (216 chunks) and ADOO (159,600 chunks) hold the name with no NUL terminator; the chunk size delimits it.
    private static void WriteCString(BinaryWriter w, string value) => w.Write(Encoding.Latin1.GetBytes(value));

    private static void WriteVector(BinaryWriter w, Vector3 v)
    {
        w.Write(v.X);
        w.Write(v.Y);
        w.Write(v.Z);
    }
}
