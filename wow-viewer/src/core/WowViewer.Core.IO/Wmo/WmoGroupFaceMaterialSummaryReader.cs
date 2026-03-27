using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupFaceMaterialSummaryReader
{
    public static WmoGroupFaceMaterialSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupFaceMaterialSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, byte[] mogp) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogp);
        byte[]? mopyPayload = null;
        foreach ((var header, int dataOffset) in WmoGroupReaderCommon.EnumerateSubchunks(mogp, headerSizeBytes))
        {
            if (header.Id != WmoChunkIds.Mopy)
                continue;

            mopyPayload = mogp.AsSpan(dataOffset, checked((int)header.Size)).ToArray();
            break;
        }

        if (mopyPayload is null)
            throw new InvalidDataException("WMO group face-material summary requires an MOPY subchunk.");

        int entrySizeBytes = WmoGroupReaderCommon.InferMopyEntrySize(mopyPayload.Length, version);
        if (entrySizeBytes <= 0 || mopyPayload.Length % entrySizeBytes != 0)
            throw new InvalidDataException($"MOPY payload size {mopyPayload.Length} is not compatible with inferred entry size {entrySizeBytes}.");

        int faceCount = mopyPayload.Length / entrySizeBytes;
        HashSet<byte> materialIds = [];
        int hiddenFaceCount = 0;
        int flaggedFaceCount = 0;
        for (int index = 0; index < faceCount; index++)
        {
            int offset = index * entrySizeBytes;
            byte flags = mopyPayload[offset];
            byte materialId = mopyPayload[offset + 1];
            if (flags != 0)
                flaggedFaceCount++;

            if (materialId == byte.MaxValue)
            {
                hiddenFaceCount++;
                continue;
            }

            materialIds.Add(materialId);
        }

        return new WmoGroupFaceMaterialSummary(
            sourcePath,
            version,
            mopyPayload.Length,
            entrySizeBytes,
            faceCount,
            distinctMaterialIdCount: materialIds.Count,
            highestMaterialId: materialIds.Count > 0 ? materialIds.Max() : 0,
            hiddenFaceCount,
            flaggedFaceCount);
    }
}
