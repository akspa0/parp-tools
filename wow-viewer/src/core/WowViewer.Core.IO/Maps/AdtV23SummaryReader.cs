using System.Buffers.Binary;
using WowViewer.Core.Maps;

namespace WowViewer.Core.IO.Maps;

public static class AdtV23SummaryReader
{
    public static AdtV23Summary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, Path.GetFullPath(path));
        return Read(stream, fileSummary);
    }

    public static AdtV23Summary Read(Stream stream, MapFileSummary fileSummary)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(fileSummary);

        if (fileSummary.Kind is not (MapFileKind.AdtV23 or MapFileKind.AdtV23Error))
            throw new InvalidDataException($"ADT/v23 semantic summary requires an ADT/v23 file, but found {fileSummary.Kind}.");

        byte[] ahdr = MapSummaryReaderCommon.ReadChunkPayload(stream, fileSummary, MapChunkIds.Ahdr)
            ?? throw new InvalidDataException("ADT/v23 summary requires an AHDR chunk.");

        if (ahdr.Length < 20)
            throw new InvalidDataException($"ADT/v23 AHDR chunk is too short: expected at least 20 bytes, found {ahdr.Length}.");

        return new AdtV23Summary(
            fileSummary.SourcePath,
            fileSummary.Kind,
            BinaryPrimitives.ReadUInt32LittleEndian(ahdr.AsSpan(0, 4)),
            checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ahdr.AsSpan(4, 4))),
            checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ahdr.AsSpan(8, 4))),
            checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ahdr.AsSpan(12, 4))),
            checked((int)BinaryPrimitives.ReadUInt32LittleEndian(ahdr.AsSpan(16, 4))),
            terrainChunkCount: fileSummary.CountChunks(MapChunkIds.Acnk),
            textureNameCount: fileSummary.CountChunks(MapChunkIds.Atex),
            objectNameCount: fileSummary.CountChunks(MapChunkIds.Adoo),
            hasVertexHeights: fileSummary.HasChunk(MapChunkIds.Avtx),
            hasNormals: fileSummary.HasChunk(MapChunkIds.Anrm),
            hasFlightBounds: fileSummary.HasChunk(MapChunkIds.Afbo),
            hasVertexShading: fileSummary.HasChunk(MapChunkIds.Acvt));
    }
}