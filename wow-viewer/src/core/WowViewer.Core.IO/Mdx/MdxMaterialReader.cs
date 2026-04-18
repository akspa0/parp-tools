using System.Buffers.Binary;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxMaterialReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;

    public static MdxMaterialFile Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxMaterialFile Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX material reading requires a seekable stream.", nameof(stream));

        if (stream.Length < SignatureSizeBytes)
            throw new InvalidDataException($"MDX file '{sourcePath}' is too small to contain a signature.");

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signatureBytes = stackalloc byte[SignatureSizeBytes];
            stream.ReadExactly(signatureBytes);

            string signature = Encoding.ASCII.GetString(signatureBytes);
            if (!string.Equals(signature, "MDLX", StringComparison.Ordinal))
                throw new InvalidDataException($"File '{sourcePath}' does not contain an MDLX signature. Found '{signature}'.");

            uint? version = null;
            string? modelName = null;
            List<MdxMaterial> materials = [];
            Span<byte> headerBytes = stackalloc byte[ChunkHeader.SizeInBytes];

            while (stream.Position <= stream.Length - ChunkHeader.SizeInBytes)
            {
                long headerOffset = stream.Position;
                stream.ReadExactly(headerBytes);
                if (!TryReadMdxChunkHeader(headerBytes, out ChunkHeader header))
                    throw new InvalidDataException($"Could not decode MDX chunk header at offset {headerOffset}.");

                long dataOffset = stream.Position;
                long endOffset = checked(dataOffset + header.Size);
                if (endOffset > stream.Length)
                    throw new InvalidDataException($"MDX chunk {header.Id} at offset {headerOffset} overruns the stream length.");

                if (header.Id == MdxChunkIds.Vers && header.Size >= sizeof(uint))
                {
                    version = ReadUInt32At(stream, dataOffset);
                }
                else if (header.Id == MdxChunkIds.Modl && header.Size >= ModlNameSizeBytes)
                {
                    modelName = ReadFixedAsciiAt(stream, dataOffset, ModlNameSizeBytes);
                }
                else if (header.Id == MdxChunkIds.Mtls)
                {
                    materials = ReadClassicMaterials(stream, dataOffset, header.Size, version);
                }

                stream.Position = endOffset;
            }

            return new MdxMaterialFile(sourcePath, signature, version, modelName, materials);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static List<MdxMaterial> ReadClassicMaterials(Stream stream, long dataOffset, uint size, uint? version)
    {
        if (version is not null and not 1300u and not 1400u)
            return [];

        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            if (size < 8)
                return [];

            uint materialCount = ReadUInt32(stream);
            _ = ReadUInt32(stream);

            List<MdxMaterial> materials = new(checked((int)materialCount));
            long chunkEnd = checked(dataOffset + size);
            for (int materialIndex = 0; materialIndex < materialCount; materialIndex++)
            {
                long materialSizeOffset = stream.Position;
                uint materialSize = ReadUInt32(stream);
                long materialEnd = checked(materialSizeOffset + materialSize);
                if (materialEnd > chunkEnd || materialEnd <= materialSizeOffset)
                    throw new InvalidDataException($"MTLS(v1300): invalid material size 0x{materialSize:X} at index {materialIndex}.");

                int priorityPlane = ReadInt32(stream);
                uint layerCount = ReadUInt32(stream);
                List<MdxMaterialLayer> layers = new(checked((int)layerCount));

                for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                {
                    long layerSizeOffset = stream.Position;
                    uint layerSize = ReadUInt32(stream);
                    long layerEnd = checked(layerSizeOffset + layerSize);
                    if (layerEnd > materialEnd || layerEnd <= layerSizeOffset)
                        throw new InvalidDataException($"MTLS(v1300): invalid layer size 0x{layerSize:X} in material {materialIndex} layer {layerIndex}.");

                    if (layerEnd - stream.Position < 24)
                        throw new InvalidDataException($"MTLS(v1300): missing fixed layer fields in material {materialIndex} layer {layerIndex}.");

                    uint blendMode = ReadUInt32(stream);
                    uint flags = ReadUInt32(stream);
                    int textureId = ReadInt32(stream);
                    int transformId = ReadInt32(stream);
                    int coordId = ReadInt32(stream);
                    float staticAlpha = ReadSingle(stream);
                    float staticEmissiveGain = layerEnd - stream.Position >= 4
                        ? ReadSingle(stream)
                        : 0.0f;

                    MdxScalarTrack? emissiveTrack = null;
                    MdxScalarTrack? alphaTrack = null;
                    MdxIntTrack? textureLayerTrack = null;

                    while (stream.Position <= layerEnd - 4)
                    {
                        string tag = ReadTag(stream);
                        switch (tag)
                        {
                            case "KMTE":
                                emissiveTrack = MdxTrackReader.ReadScalarTrack(stream, layerEnd, tag, "MTLS(v1300)", $"MTLS(v1300): {tag} payload overran material {materialIndex} layer {layerIndex}.");
                                break;
                            case "KMTA":
                                alphaTrack = MdxTrackReader.ReadScalarTrack(stream, layerEnd, tag, "MTLS(v1300)", $"MTLS(v1300): {tag} payload overran material {materialIndex} layer {layerIndex}.");
                                break;
                            case "KMTF":
                                textureLayerTrack = MdxTrackReader.ReadIntTrack(stream, layerEnd, tag, "MTLS(v1300)", $"MTLS(v1300): {tag} payload overran material {materialIndex} layer {layerIndex}.");
                                break;
                            default:
                                stream.Position = layerEnd;
                                break;
                        }
                    }

                    layers.Add(new MdxMaterialLayer(
                        layerIndex,
                        blendMode,
                        flags,
                        textureId,
                        transformId,
                        coordId,
                        staticAlpha,
                        staticEmissiveGain,
                        emissiveTrack,
                        alphaTrack,
                        textureLayerTrack));
                    stream.Position = layerEnd;
                }

                materials.Add(new MdxMaterial(materialIndex, priorityPlane, layers));
                stream.Position = materialEnd;
            }

            return materials;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadMdxChunkHeader(ReadOnlySpan<byte> bytes, out ChunkHeader header)
    {
        header = default;
        if (bytes.Length < ChunkHeader.SizeInBytes)
            return false;

        FourCC id = FourCC.FromString(Encoding.ASCII.GetString(bytes[..4]));
        header = new ChunkHeader(id, BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(4, 4)));
        return true;
    }

    private static uint ReadUInt32At(Stream stream, long offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            return ReadUInt32(stream);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadFixedAsciiAt(Stream stream, long offset, int length)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            byte[] bytes = new byte[length];
            stream.ReadExactly(bytes);
            int end = Array.IndexOf(bytes, (byte)0);
            if (end < 0)
                end = bytes.Length;

            return Encoding.ASCII.GetString(bytes, 0, end);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadTag(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return Encoding.ASCII.GetString(bytes);
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadInt32LittleEndian(bytes);
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[4];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }
}