using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

public static class MdxSummaryReader
{
    private const int SignatureSizeBytes = 4;
    private const int ModlNameSizeBytes = 0x50;
    private const int ModlBoundsAndBlendSizeBytes = 0x18 + sizeof(uint);
    private const int ModlSummarySizeBytes = ModlNameSizeBytes + ModlBoundsAndBlendSizeBytes;
    private const int TexsEntrySizeLegacy = 0x108;
    private const int TexsEntrySizeExtended = 0x10C;
    private const int TexsPathSizeLegacy = 0x100;
    private const int TexsPathSizeExtended = 0x104;

    private static readonly HashSet<FourCC> KnownChunkIds =
    [
        MdxChunkIds.Vers,
        MdxChunkIds.Modl,
        MdxChunkIds.Seqs,
        MdxChunkIds.Glbs,
        MdxChunkIds.Mtls,
        MdxChunkIds.Texs,
        MdxChunkIds.Geos,
        MdxChunkIds.Geoa,
        MdxChunkIds.Bone,
        MdxChunkIds.Help,
        MdxChunkIds.Pivt,
        MdxChunkIds.Atch,
        MdxChunkIds.Lite,
        MdxChunkIds.Prem,
        MdxChunkIds.Pre2,
        MdxChunkIds.Ribb,
        MdxChunkIds.Evts,
        MdxChunkIds.Cams,
        MdxChunkIds.Clid,
        MdxChunkIds.Htst,
        MdxChunkIds.Txan,
        MdxChunkIds.Corn,
    ];

    public static MdxSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static MdxSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("MDX summary reading requires a seekable stream.", nameof(stream));

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

            List<MdxChunkSummary> chunks = [];
            uint? version = null;
            string? modelName = null;
            uint? blendTime = null;
            Vector3? boundsMin = null;
            Vector3? boundsMax = null;
            List<MdxTextureSummary> textures = [];
            List<MdxMaterialSummary> materials = [];
            int knownChunkCount = 0;
            int unknownChunkCount = 0;
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

                bool isKnownChunk = KnownChunkIds.Contains(header.Id);
                if (isKnownChunk)
                    knownChunkCount++;
                else
                    unknownChunkCount++;

                chunks.Add(new MdxChunkSummary(header.Id, header.Size, headerOffset, dataOffset, isKnownChunk));

                if (header.Id == MdxChunkIds.Vers && header.Size >= sizeof(uint))
                {
                    version = ReadUInt32At(stream, dataOffset);
                }
                else if (header.Id == MdxChunkIds.Modl)
                {
                    ReadModlSummary(stream, dataOffset, header.Size, out modelName, out blendTime, out boundsMin, out boundsMax);
                }
                else if (header.Id == MdxChunkIds.Texs)
                {
                    textures = ReadTexsSummary(stream, dataOffset, header.Size);
                }
                else if (header.Id == MdxChunkIds.Mtls)
                {
                    materials = ReadMtlsSummary(stream, dataOffset, header.Size);
                }

                stream.Position = endOffset;
            }

            return new MdxSummary(sourcePath, signature, version, modelName, blendTime, boundsMin, boundsMax, textures, materials, chunks, knownChunkCount, unknownChunkCount);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32At(Stream stream, long offset)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = offset;
            Span<byte> bytes = stackalloc byte[sizeof(uint)];
            stream.ReadExactly(bytes);
            return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static bool TryReadMdxChunkHeader(ReadOnlySpan<byte> data, out ChunkHeader header)
    {
        if (data.Length < ChunkHeader.SizeInBytes)
        {
            header = default;
            return false;
        }

        string idText = Encoding.ASCII.GetString(data[..4]);
        FourCC id = FourCC.FromString(idText);
        uint size = BinaryPrimitives.ReadUInt32LittleEndian(data[4..]);
        header = new ChunkHeader(id, size);
        return true;
    }

    private static List<MdxTextureSummary> ReadTexsSummary(Stream stream, long dataOffset, uint size)
    {
        long previousPosition = stream.Position;
        try
        {
            (int entrySize, int pathSize) = ResolveTexsLayout(size);
            int count = checked((int)(size / entrySize));
            List<MdxTextureSummary> textures = new(count);
            byte[] replaceableBytes = new byte[sizeof(uint)];
            byte[] flagsBytes = new byte[sizeof(uint)];

            stream.Position = dataOffset;
            for (int index = 0; index < count; index++)
            {
                stream.ReadExactly(replaceableBytes);
                uint replaceableId = BinaryPrimitives.ReadUInt32LittleEndian(replaceableBytes);

                byte[] pathBytes = new byte[pathSize];
                stream.ReadExactly(pathBytes);
                string path = ReadNullTerminatedAscii(pathBytes);

                stream.ReadExactly(flagsBytes);
                uint flags = BinaryPrimitives.ReadUInt32LittleEndian(flagsBytes);

                textures.Add(new MdxTextureSummary(index, replaceableId, path, flags));
            }

            return textures;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static (int EntrySize, int PathSize) ResolveTexsLayout(uint size)
    {
        if (size % TexsEntrySizeExtended == 0)
            return (TexsEntrySizeExtended, TexsPathSizeExtended);

        if (size % TexsEntrySizeLegacy == 0)
            return (TexsEntrySizeLegacy, TexsPathSizeLegacy);

        throw new InvalidDataException($"Invalid TEXS size 0x{size:X}: expected divisibility by 0x{TexsEntrySizeLegacy:X} or 0x{TexsEntrySizeExtended:X}.");
    }

    private static List<MdxMaterialSummary> ReadMtlsSummary(Stream stream, long dataOffset, uint size)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            if (size < 8)
                return [];

            uint materialCount = ReadUInt32(stream);
            _ = ReadUInt32(stream);

            List<MdxMaterialSummary> materials = new(checked((int)materialCount));
            for (int materialIndex = 0; materialIndex < materialCount; materialIndex++)
            {
                long materialSizeOffset = stream.Position;
                uint materialSize = ReadUInt32(stream);
                long materialEnd = checked(materialSizeOffset + materialSize);
                if (materialEnd > dataOffset + size)
                    throw new InvalidDataException($"MTLS material {materialIndex} overruns the MTLS payload.");

                int priorityPlane = ReadInt32(stream);
                uint layerCount = ReadUInt32(stream);
                List<MdxMaterialLayerSummary> layers = new(checked((int)layerCount));

                for (int layerIndex = 0; layerIndex < layerCount; layerIndex++)
                {
                    long layerSizeOffset = stream.Position;
                    uint layerSize = ReadUInt32(stream);
                    long layerEnd = checked(layerSizeOffset + layerSize);
                    if (layerEnd > materialEnd)
                        throw new InvalidDataException($"MTLS layer {layerIndex} in material {materialIndex} overruns the material payload.");

                    uint blendMode = ReadUInt32(stream);
                    uint flags = ReadUInt32(stream);
                    int textureId = ReadInt32(stream);
                    int transformId = ReadInt32(stream);
                    int coordId = ReadInt32(stream);
                    float staticAlpha = ReadSingle(stream);

                    layers.Add(new MdxMaterialLayerSummary(layerIndex, blendMode, flags, textureId, transformId, coordId, staticAlpha));
                    stream.Position = layerEnd;
                }

                materials.Add(new MdxMaterialSummary(materialIndex, priorityPlane, layers));
                stream.Position = materialEnd;
            }

            return materials;
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadInt32LittleEndian(bytes);
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static void ReadModlSummary(Stream stream, long dataOffset, uint size, out string? modelName, out uint? blendTime, out Vector3? boundsMin, out Vector3? boundsMax)
    {
        long previousPosition = stream.Position;
        try
        {
            stream.Position = dataOffset;
            int nameBytesToRead = checked((int)Math.Min(ModlNameSizeBytes, size));
            byte[] nameBytes = new byte[nameBytesToRead];
            if (nameBytesToRead > 0)
                stream.ReadExactly(nameBytes);

            string rawName = ReadNullTerminatedAscii(nameBytes);
            modelName = string.IsNullOrWhiteSpace(rawName) ? null : rawName;

            blendTime = null;
            boundsMin = null;
            boundsMax = null;
            if (size < ModlSummarySizeBytes)
                return;

            Span<byte> boundsAndBlendBytes = stackalloc byte[ModlBoundsAndBlendSizeBytes];
            stream.ReadExactly(boundsAndBlendBytes);
            boundsMin = new Vector3(
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x00..0x04]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x04..0x08]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x08..0x0C]));
            boundsMax = new Vector3(
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x0C..0x10]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x10..0x14]),
                BinaryPrimitives.ReadSingleLittleEndian(boundsAndBlendBytes[0x14..0x18]));
            blendTime = BinaryPrimitives.ReadUInt32LittleEndian(boundsAndBlendBytes[0x18..0x1C]);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    private static string ReadNullTerminatedAscii(byte[] bytes)
    {
        int length = Array.IndexOf(bytes, (byte)0);
        if (length < 0)
            length = bytes.Length;

        return Encoding.ASCII.GetString(bytes, 0, length);
    }
}