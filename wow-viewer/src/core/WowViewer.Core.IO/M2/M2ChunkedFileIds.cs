using System.Buffers.Binary;

namespace WowViewer.Core.IO.M2;

/// <summary>
/// Spec 239: FileDataID tables of a chunked (MD21) M2. Chunk ids are stored in reading order
/// ("MD21", "SFID", "TXID"). Measured on wow_classic_beta 1.60.1: SILVERPINETREE03.M2 carries
/// SFID [508760] and TXID [203980, 399526].
/// </summary>
public sealed record M2ChunkedFileIds(uint[] SkinFileDataIds, uint[] TextureFileDataIds)
{
    private const uint Md21 = 0x3132444D; // 'MD21'
    private const uint Sfid = 0x44494653; // 'SFID'
    private const uint Txid = 0x44495854; // 'TXID'

    public static bool TryRead(ReadOnlySpan<byte> data, out M2ChunkedFileIds ids)
    {
        ids = new M2ChunkedFileIds([], []);
        if (data.Length < 8 || BinaryPrimitives.ReadUInt32LittleEndian(data) != Md21)
            return false;

        uint[] skins = [], textures = [];
        int position = 0;
        while (position + 8 <= data.Length)
        {
            uint id = BinaryPrimitives.ReadUInt32LittleEndian(data[position..]);
            uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[(position + 4)..]);
            if (position + 8L + size > data.Length)
                break;

            ReadOnlySpan<byte> payload = data.Slice(position + 8, (int)size);
            if (id == Sfid)
                skins = ReadIds(payload);
            else if (id == Txid)
                textures = ReadIds(payload);

            position += 8 + (int)size;
        }

        ids = new M2ChunkedFileIds(skins, textures);
        return true;
    }

    /// <summary>
    /// The MD20 blob inside a chunked M2's MD21 chunk (offsets inside it are relative to its start),
    /// or the input unchanged when the file is not chunked. Readers that only understand a bare MD20
    /// root (M2ModelReader / M2GeometryReader) must be given this payload.
    /// </summary>
    public static byte[] GetMd20Payload(byte[] data)
    {
        if (data.Length < 8 || BinaryPrimitives.ReadUInt32LittleEndian(data) != Md21)
            return data;

        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(4));
        return data.AsSpan(8, (int)Math.Min(size, (uint)(data.Length - 8))).ToArray();
    }

    /// <summary>
    /// Returns <paramref name="geometry"/> with each empty texture filename replaced by the TXID entry
    /// at the same index, resolved through <see cref="Files.FileDataIdPaths"/>.
    /// </summary>
    public WowViewer.Core.M2.M2GeometryDocument ApplyTextureNames(WowViewer.Core.M2.M2GeometryDocument geometry)
    {
        if (TextureFileDataIds.Length == 0)
            return geometry;

        var textures = new List<WowViewer.Core.M2.M2GeometryTexture>(geometry.Textures.Count);
        for (int i = 0; i < geometry.Textures.Count; i++)
        {
            WowViewer.Core.M2.M2GeometryTexture texture = geometry.Textures[i];
            uint fileDataId = i < TextureFileDataIds.Length ? TextureFileDataIds[i] : 0;
            textures.Add(string.IsNullOrWhiteSpace(texture.Filename) && fileDataId != 0
                ? new WowViewer.Core.M2.M2GeometryTexture(Files.FileDataIdPaths.Resolve(fileDataId), texture.ReplaceableId, texture.Flags)
                : texture);
        }

        return new WowViewer.Core.M2.M2GeometryDocument(
            geometry.Model,
            geometry.Vertices,
            textures,
            geometry.RenderFlags,
            geometry.TextureLookup,
            geometry.TextureUnitLookup,
            geometry.TransparencyLookup,
            geometry.TextureAnimationLookup,
            geometry.BoneLookup);
    }

    private static uint[] ReadIds(ReadOnlySpan<byte> payload)
    {
        var values = new uint[payload.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadUInt32LittleEndian(payload[(i * 4)..]);
        return values;
    }
}
