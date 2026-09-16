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

    private static uint[] ReadIds(ReadOnlySpan<byte> payload)
    {
        var values = new uint[payload.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadUInt32LittleEndian(payload[(i * 4)..]);
        return values;
    }
}
