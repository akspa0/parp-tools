using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.M2;

namespace WowViewer.Core.IO.M2;

public static class M2GeometryReader
{
    private const int VertexCountOffset = 0x3C;
    private const int VertexOffsetOffset = 0x40;
    private const int TextureCountOffset = 0x50;
    private const int TextureOffsetOffset = 0x54;
    private const int RenderFlagCountOffset = 0x70;
    private const int RenderFlagOffsetOffset = 0x74;
    private const int BoneLookupCountOffset = 0x78;
    private const int BoneLookupOffsetOffset = 0x7C;
    private const int TextureLookupCountOffset = 0x80;
    private const int TextureLookupOffsetOffset = 0x84;
    private const int TextureUnitLookupCountOffset = 0x88;
    private const int TextureUnitLookupOffsetOffset = 0x8C;
    private const int TransparencyLookupCountOffset = 0x90;
    private const int TransparencyLookupOffsetOffset = 0x94;
    private const int TextureAnimationLookupCountOffset = 0x98;
    private const int TextureAnimationLookupOffsetOffset = 0x9C;

    private const int VertexStride = 0x30;
    private const int TextureStride = 0x10;
    private const int RenderFlagStride = 0x04;
    private const int LookupStride = sizeof(ushort);

    public static M2GeometryDocument Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static M2GeometryDocument Read(Stream stream, string sourcePath)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        M2ModelDocument model = M2ModelReader.Read(stream, sourcePath);
        byte[] data = ReadAllBytes(stream);
        List<M2GeometryVertex> vertices = ReadVertices(data, sourcePath);
        List<M2GeometryTexture> textures = ReadTextures(data, sourcePath);
        List<M2GeometryRenderFlag> renderFlags = ReadRenderFlags(data, sourcePath);
        List<M2GeometryTextureLookup> textureLookup = ReadTextureLookup(data, sourcePath);
        List<M2GeometryTextureUnitLookup> textureUnitLookup = ReadTextureUnitLookup(data, sourcePath);
        List<M2GeometryTransparencyLookup> transparencyLookup = ReadTransparencyLookup(data, sourcePath);
        List<M2GeometryTextureAnimationLookup> textureAnimationLookup = ReadTextureAnimationLookup(data, sourcePath);
        List<M2GeometryBoneLookup> boneLookup = ReadBoneLookup(data, sourcePath);

        return new M2GeometryDocument(
            model,
            vertices,
            textures,
            renderFlags,
            textureLookup,
            textureUnitLookup,
            transparencyLookup,
            textureAnimationLookup,
            boneLookup);
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            byte[] data = new byte[checked((int)stream.Length)];
            stream.ReadExactly(data);
            return data;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<M2GeometryVertex> ReadVertices(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, VertexCountOffset));
        uint offset = ReadUInt32At(data, VertexOffsetOffset);
        ValidateSpan(count, offset, VertexStride, data.Length, sourcePath, "vertices");

        List<M2GeometryVertex> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * VertexStride));
            values.Add(new M2GeometryVertex(
                ReadFiniteVector3At(data, entryOffset + 0x00, sourcePath, $"vertex[{index}].position"),
                ReadFiniteVector3At(data, entryOffset + 0x14, sourcePath, $"vertex[{index}].normal"),
                ReadFiniteVector2At(data, entryOffset + 0x20, sourcePath, $"vertex[{index}].uv0"),
                ReadFiniteVector2At(data, entryOffset + 0x28, sourcePath, $"vertex[{index}].uv1"),
                new Vector4(data[entryOffset + 0x10], data[entryOffset + 0x11], data[entryOffset + 0x12], data[entryOffset + 0x13]),
                new Vector4(data[entryOffset + 0x0C] / 255f, data[entryOffset + 0x0D] / 255f, data[entryOffset + 0x0E] / 255f, data[entryOffset + 0x0F] / 255f)));
        }

        return values;
    }

    private static List<M2GeometryTexture> ReadTextures(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, TextureCountOffset));
        uint offset = ReadUInt32At(data, TextureOffsetOffset);
        ValidateSpan(count, offset, TextureStride, data.Length, sourcePath, "textures");

        List<M2GeometryTexture> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * TextureStride));
            uint replaceableId = ReadUInt32At(data, entryOffset + 0x00);
            uint flags = ReadUInt32At(data, entryOffset + 0x04);
            uint nameLength = ReadUInt32At(data, entryOffset + 0x08);
            uint nameOffset = ReadUInt32At(data, entryOffset + 0x0C);
            values.Add(new M2GeometryTexture(ReadStringAt(data, sourcePath, $"texture[{index}].name", nameLength, nameOffset), replaceableId, flags));
        }

        return values;
    }

    private static List<M2GeometryRenderFlag> ReadRenderFlags(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, RenderFlagCountOffset));
        uint offset = ReadUInt32At(data, RenderFlagOffsetOffset);
        ValidateSpan(count, offset, RenderFlagStride, data.Length, sourcePath, "renderFlags");

        List<M2GeometryRenderFlag> values = new(count);
        for (int index = 0; index < count; index++)
        {
            int entryOffset = checked((int)offset + (index * RenderFlagStride));
            values.Add(new M2GeometryRenderFlag(
                ReadUInt16At(data, entryOffset + 0x00),
                ReadUInt16At(data, entryOffset + 0x02)));
        }

        return values;
    }

    private static List<M2GeometryTextureLookup> ReadTextureLookup(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, TextureLookupCountOffset));
        uint offset = ReadUInt32At(data, TextureLookupOffsetOffset);
        ValidateSpan(count, offset, LookupStride, data.Length, sourcePath, "textureLookup");
        List<M2GeometryTextureLookup> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(new M2GeometryTextureLookup(ReadUInt16At(data, checked((int)offset + (index * LookupStride)))));

        return values;
    }

    private static List<M2GeometryTextureUnitLookup> ReadTextureUnitLookup(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, TextureUnitLookupCountOffset));
        uint offset = ReadUInt32At(data, TextureUnitLookupOffsetOffset);
        ValidateSpan(count, offset, LookupStride, data.Length, sourcePath, "textureUnitLookup");
        List<M2GeometryTextureUnitLookup> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(new M2GeometryTextureUnitLookup(ReadUInt16At(data, checked((int)offset + (index * LookupStride)))));

        return values;
    }

    private static List<M2GeometryTransparencyLookup> ReadTransparencyLookup(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, TransparencyLookupCountOffset));
        uint offset = ReadUInt32At(data, TransparencyLookupOffsetOffset);
        ValidateSpan(count, offset, LookupStride, data.Length, sourcePath, "transparencyLookup");
        List<M2GeometryTransparencyLookup> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(new M2GeometryTransparencyLookup(ReadUInt16At(data, checked((int)offset + (index * LookupStride)))));

        return values;
    }

    private static List<M2GeometryTextureAnimationLookup> ReadTextureAnimationLookup(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, TextureAnimationLookupCountOffset));
        uint offset = ReadUInt32At(data, TextureAnimationLookupOffsetOffset);
        ValidateSpan(count, offset, LookupStride, data.Length, sourcePath, "textureAnimationLookup");
        List<M2GeometryTextureAnimationLookup> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(new M2GeometryTextureAnimationLookup(ReadUInt16At(data, checked((int)offset + (index * LookupStride)))));

        return values;
    }

    private static List<M2GeometryBoneLookup> ReadBoneLookup(byte[] data, string sourcePath)
    {
        int count = checked((int)ReadUInt32At(data, BoneLookupCountOffset));
        uint offset = ReadUInt32At(data, BoneLookupOffsetOffset);
        ValidateSpan(count, offset, LookupStride, data.Length, sourcePath, "boneLookup");
        List<M2GeometryBoneLookup> values = new(count);
        for (int index = 0; index < count; index++)
            values.Add(new M2GeometryBoneLookup(ReadUInt16At(data, checked((int)offset + (index * LookupStride)))));

        return values;
    }

    private static string? ReadStringAt(byte[] data, string sourcePath, string label, uint count, uint offset)
    {
        if (count == 0)
            return null;

        ValidateSpan(checked((int)count), offset, sizeof(byte), data.Length, sourcePath, label);
        ReadOnlySpan<byte> bytes = data.AsSpan(checked((int)offset), checked((int)count));
        int terminator = bytes.IndexOf((byte)0);
        int length = terminator >= 0 ? terminator : bytes.Length;
        if (length == 0)
            return null;

        return Encoding.UTF8.GetString(bytes[..length]);
    }

    private static uint ReadUInt32At(byte[] data, int offset)
    {
        return BinaryPrimitives.ReadUInt32LittleEndian(data.AsSpan(offset, sizeof(uint)));
    }

    private static ushort ReadUInt16At(byte[] data, int offset)
    {
        return BinaryPrimitives.ReadUInt16LittleEndian(data.AsSpan(offset, sizeof(ushort)));
    }

    private static float ReadFiniteSingleAt(byte[] data, int offset, string sourcePath, string label)
    {
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset, sizeof(float))));
        if (!float.IsFinite(value))
            return 0f;

        return value;
    }

    private static Vector2 ReadFiniteVector2At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector2(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"));
    }

    private static Vector3 ReadFiniteVector3At(byte[] data, int offset, string sourcePath, string label)
    {
        return new Vector3(
            ReadFiniteSingleAt(data, offset + 0x00, sourcePath, $"{label}.x"),
            ReadFiniteSingleAt(data, offset + 0x04, sourcePath, $"{label}.y"),
            ReadFiniteSingleAt(data, offset + 0x08, sourcePath, $"{label}.z"));
    }

    private static void ValidateSpan(int count, uint offset, int stride, int length, string sourcePath, string label)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        if (count == 0)
            return;

        if (offset == 0)
            throw new InvalidDataException($"M2 geometry payload for '{sourcePath}' has a zero offset for non-empty span '{label}'.");

        ulong total = checked((ulong)count * (ulong)stride);
        ulong end = (ulong)offset + total;
        if ((ulong)offset >= (ulong)length || end > (ulong)length || end < offset)
        {
            throw new InvalidDataException(
                $"M2 geometry payload for '{sourcePath}' has an out-of-range span for '{label}': count={count}, offset=0x{offset:X}, stride=0x{stride:X}, length=0x{length:X}.");
        }
    }
}