using System.Buffers.Binary;
using WowViewer.Core.IO.M2;
using WowViewer.Core.IO.Mdx;
using WowViewer.Core.M2;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.M2Chunked;

public static class M2ChunkedModelReader
{
    public static M2ModelDocument Read(string path)
        => ReadDetailed(path).Model;

    public static M2ModelDocument Read(Stream stream, string sourcePath, Func<string, byte[]?>? companionReader = null)
        => ReadDetailed(stream, sourcePath, companionReader).Model;

    public static M2ChunkedReadResult ReadDetailed(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return ReadDetailed(stream, Path.GetFullPath(path));
    }

    public static M2ChunkedReadResult ReadDetailed(Stream stream, string sourcePath, Func<string, byte[]?>? companionReader = null)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        _ = companionReader;

        byte[] bytes = ReadAllBytes(stream);
        ValidateChunkedMagic(bytes, sourcePath);

        IReadOnlyList<M2ChunkedChunkHeader> chunks;
        using (MemoryStream walkerStream = new(bytes, writable: false))
        using (BinaryReader walkerReader = new(walkerStream))
            chunks = new M2ChunkedChunkWalker(walkerReader).Walk();

        using MemoryStream summaryStream = new(bytes, writable: false);
        MdxSummary summary = MdxSummaryReader.Read(summaryStream, sourcePath);

        using MemoryStream geometryStream = new(bytes, writable: false);
        MdxGeometryFile geometry = MdxGeometryReader.Read(geometryStream, sourcePath);

        string canonicalModelPath = M2ModelIdentity.FromPath(sourcePath).CanonicalModelPath;
        MdxToM2ConversionResult conversion = MdxToM2Converter.Convert(summary, geometry, canonicalModelPath);

        using MemoryStream modelStream = new(conversion.ModelBytes, writable: false);
        M2ModelDocument model = M2ModelReader.Read(modelStream, conversion.ModelPath);

        return new M2ChunkedReadResult(model, summary, geometry, conversion, chunks);
    }

    private static void ValidateChunkedMagic(byte[] bytes, string sourcePath)
    {
        if (bytes.Length < sizeof(uint))
            throw new InvalidDataException($"File '{sourcePath}' is too small to be a chunked MDX file.");

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(0, sizeof(uint)));
        if (magic != MdxMagic.Mdlx)
            throw new InvalidDataException($"File '{sourcePath}' is not a chunked MDX file. Expected MDLX magic.");
    }

    private static byte[] ReadAllBytes(Stream stream)
    {
        if (!stream.CanSeek)
            throw new ArgumentException("Chunked MDX reading requires a seekable stream.", nameof(stream));

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
}
