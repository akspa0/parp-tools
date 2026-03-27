using WowViewer.Core.Chunks;
using WowViewer.Core.Files;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Files;

public static class WowFileDetector
{
    private static readonly FourCC Mprl = FourCC.FromString("MPRL");
    private static readonly FourCC Momo = FourCC.FromString("MOMO");
    private static readonly FourCC Mohd = FourCC.FromString("MOHD");
    private static readonly FourCC Mogp = FourCC.FromString("MOGP");
    private static readonly FourCC Mdid = FourCC.FromString("MDID");

    public static WowFileDetection Detect(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Detect(stream, Path.GetFullPath(path));
    }

    public static WowFileDetection Detect(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        if (!stream.CanSeek)
            throw new ArgumentException("File detection requires a seekable stream.", nameof(stream));

        if (stream.Length < 4)
            return new WowFileDetection(sourcePath, WowFileKind.Unknown, null);

        long previousPosition = stream.Position;
        try
        {
            stream.Position = 0;
            Span<byte> signature = stackalloc byte[4];
            stream.ReadExactly(signature);

            if (MatchesAscii(signature, "BLP1") || MatchesAscii(signature, "BLP2"))
                return new WowFileDetection(sourcePath, WowFileKind.Blp, null);

            if (MatchesAscii(signature, "WDBC"))
                return new WowFileDetection(sourcePath, WowFileKind.Dbc, null);

            if (MatchesAscii(signature, "WDB2") || MatchesAscii(signature, "WDB5") || MatchesAscii(signature, "WDB6") || MatchesAscii(signature, "WDB7"))
                return new WowFileDetection(sourcePath, WowFileKind.Db2, null);

            if (MatchesAscii(signature, "MD20") || MatchesAscii(signature, "MD21"))
                return new WowFileDetection(sourcePath, WowFileKind.M2, null);

            if (MatchesAscii(signature, "MDLX"))
                return new WowFileDetection(sourcePath, WowFileKind.Mdx, null);

            if (MatchesAscii(signature, "M3DT") || MatchesAscii(signature, "33DM"))
                return new WowFileDetection(sourcePath, WowFileKind.M3, null);

            if (FourCC.FromFileBytes(signature) == Mogp)
            {
                IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
                return Detect(sourcePath, chunks, version: null);
            }

            if (FourCC.FromFileBytes(signature) == MapChunkIds.Mver)
            {
                IReadOnlyList<ChunkSpan> chunks = ChunkedFileReader.ReadTopLevelChunks(stream);
                uint? version = TryReadVersion(stream, chunks);
                return Detect(sourcePath, chunks, version);
            }

            return new WowFileDetection(sourcePath, WowFileKind.Unknown, null);
        }
        finally
        {
            stream.Position = previousPosition;
        }
    }

    internal static WowFileDetection Detect(string sourcePath, IReadOnlyList<ChunkSpan> chunks, uint? version)
    {
        string fileName = Path.GetFileName(sourcePath);

        if (fileName.EndsWith("_lod.adt", StringComparison.OrdinalIgnoreCase))
            return new WowFileDetection(sourcePath, WowFileKind.AdtLod, version);

        if (fileName.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase))
            return new WowFileDetection(sourcePath, WowFileKind.AdtTex, version);

        if (fileName.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase))
            return new WowFileDetection(sourcePath, WowFileKind.AdtObj, version);

        if (chunks.Count == 0)
            return new WowFileDetection(sourcePath, WowFileKind.Unknown, version);

        if (chunks[0].Header.Id == Mogp)
            return new WowFileDetection(sourcePath, WowFileKind.WmoGroup, version);

        if (chunks[0].Header.Id != MapChunkIds.Mver)
            return new WowFileDetection(sourcePath, WowFileKind.Unknown, version);

        FourCC? secondChunkId = chunks.Count > 1 ? chunks[1].Header.Id : null;

        if (secondChunkId == Mprl)
            return new WowFileDetection(sourcePath, WowFileKind.Pm4, version);

        if (secondChunkId == MapChunkIds.Mphd || chunks.Any(chunk => chunk.Header.Id == MapChunkIds.Main))
            return new WowFileDetection(sourcePath, WowFileKind.Wdt, version);

        if (secondChunkId == Mohd)
            return new WowFileDetection(sourcePath, WowFileKind.Wmo, version);

        if (secondChunkId == Momo)
            return new WowFileDetection(sourcePath, WowFileKind.Wmo, version);

        if (secondChunkId == Mogp)
            return new WowFileDetection(sourcePath, WowFileKind.WmoGroup, version);

        if (secondChunkId == MapChunkIds.Mhdr || chunks.Any(chunk => chunk.Header.Id == MapChunkIds.Mcnk))
            return new WowFileDetection(sourcePath, WowFileKind.Adt, version);

        if (secondChunkId == MapChunkIds.Mtex || secondChunkId == Mdid)
            return new WowFileDetection(sourcePath, WowFileKind.AdtTex, version);

        if (secondChunkId == MapChunkIds.Mmdx || secondChunkId == MapChunkIds.Mwmo)
            return new WowFileDetection(sourcePath, WowFileKind.AdtObj, version);

        return new WowFileDetection(sourcePath, WowFileKind.Unknown, version);
    }

    private static uint? TryReadVersion(Stream stream, IReadOnlyList<ChunkSpan> chunks)
    {
        if (chunks.Count == 0 || chunks[0].Header.Id != MapChunkIds.Mver)
            return null;

        return ChunkedFileReader.TryReadUInt32(stream, chunks[0]);
    }

    private static bool MatchesAscii(ReadOnlySpan<byte> bytes, string text)
    {
        return bytes.Length >= 4
            && bytes[0] == text[0]
            && bytes[1] == text[1]
            && bytes[2] == text[2]
            && bytes[3] == text[3];
    }
}
