using System;
using System.IO;
using System.Text;
using ArcaneFileParser.Core.Common.Types;

namespace ArcaneFileParser.Core.IO;

/// <summary>
/// Provides extension methods for BinaryReader to read WoW file format specific types.
/// </summary>
public static class BinaryReaderExtensions
{
    // Basic type extensions
    public static string ReadCString(this BinaryReader reader)
    {
        var bytes = new List<byte>();
        byte b;
        while ((b = reader.ReadByte()) != 0)
            bytes.Add(b);
        return Encoding.UTF8.GetString(bytes.ToArray());
    }

    public static string ReadStringFixed(this BinaryReader reader, int length)
    {
        var bytes = reader.ReadBytes(length);
        int nullIndex = Array.IndexOf<byte>(bytes, 0);
        if (nullIndex != -1)
            Array.Resize(ref bytes, nullIndex);
        return Encoding.UTF8.GetString(bytes);
    }

    // Vector types
    public static Vector2F ReadVector2F(this BinaryReader reader) =>
        new(reader.ReadSingle(), reader.ReadSingle());

    public static Vector3F ReadVector3F(this BinaryReader reader) =>
        new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

    // Color types
    public static ColorBGRA ReadColorBGRA(this BinaryReader reader) =>
        new(reader.ReadByte(), reader.ReadByte(), reader.ReadByte(), reader.ReadByte());

    public static ColorRGBA ReadColorRGBA(this BinaryReader reader) =>
        new(reader.ReadByte(), reader.ReadByte(), reader.ReadByte(), reader.ReadByte());

    // Matrix type
    public static Matrix4x4F ReadMatrix4x4F(this BinaryReader reader) =>
        new(
            reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(),
            reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(),
            reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(),
            reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle()
        );

    // Quaternion type
    public static QuaternionF ReadQuaternionF(this BinaryReader reader) =>
        new(reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle(), reader.ReadSingle());

    // Bounding types
    public static BoundingBox ReadBoundingBox(this BinaryReader reader) =>
        new(reader.ReadVector3F(), reader.ReadVector3F());

    // Array reading helpers
    public static T[] ReadArray<T>(this BinaryReader reader, int count, Func<BinaryReader, T> elementReader)
    {
        var array = new T[count];
        for (int i = 0; i < count; i++)
            array[i] = elementReader(reader);
        return array;
    }

    // Chunk reading helpers
    public static uint ReadChunkSignature(this BinaryReader reader) =>
        reader.ReadUInt32();

    public static bool TryReadChunkSignature(this BinaryReader reader, out uint signature)
    {
        try
        {
            signature = reader.ReadUInt32();
            return true;
        }
        catch (EndOfStreamException)
        {
            signature = 0;
            return false;
        }
    }

    public static (uint Signature, uint Size) ReadChunkHeader(this BinaryReader reader)
    {
        uint signature = reader.ReadChunkSignature();
        uint size = reader.ReadUInt32();
        return (signature, size);
    }

    // Padding and alignment
    public static void SkipPadding(this BinaryReader reader, int alignment)
    {
        long position = reader.BaseStream.Position;
        long padding = (alignment - (position % alignment)) % alignment;
        if (padding > 0)
            reader.BaseStream.Seek(padding, SeekOrigin.Current);
    }

    public static void SeekToChunkEnd(this BinaryReader reader, long chunkStart, uint chunkSize)
    {
        long expectedEnd = chunkStart + chunkSize;
        if (reader.BaseStream.Position != expectedEnd)
            reader.BaseStream.Seek(expectedEnd, SeekOrigin.Begin);
    }

    // Versioned chunk reading
    public static bool IsValidChunkVersion(this BinaryReader reader, uint expectedVersion)
    {
        uint version = reader.ReadUInt32();
        return version == expectedVersion;
    }

    public static bool TryReadVersionedChunk(this BinaryReader reader, uint expectedSignature, uint expectedVersion, out uint size)
    {
        long startPosition = reader.BaseStream.Position;
        try
        {
            var (signature, chunkSize) = reader.ReadChunkHeader();
            if (signature != expectedSignature)
            {
                reader.BaseStream.Seek(startPosition, SeekOrigin.Begin);
                size = 0;
                return false;
            }

            if (!reader.IsValidChunkVersion(expectedVersion))
            {
                reader.BaseStream.Seek(startPosition, SeekOrigin.Begin);
                size = 0;
                return false;
            }

            size = chunkSize;
            return true;
        }
        catch (EndOfStreamException)
        {
            reader.BaseStream.Seek(startPosition, SeekOrigin.Begin);
            size = 0;
            return false;
        }
    }
} 