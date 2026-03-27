using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoGroupNameReferenceSummaryReader
{
    public static WmoGroupNameReferenceSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoGroupNameReferenceSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] mohd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        byte[] mogn = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogn);
        byte[] mogi = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mogi);
        int entrySize = WmoRootReaderCommon.InferMogiEntrySize(mogi, WmoRootReaderCommon.ReadReportedGroupCount(mohd));
        if (entrySize <= 0 || mogi.Length % entrySize != 0)
            throw new InvalidDataException($"MOGI payload size {mogi.Length} is not compatible with inferred entry size {entrySize}.");

        int entryCount = mogi.Length / entrySize;
        int resolved = 0;
        int unresolved = 0;
        int maxLength = 0;
        HashSet<string> names = new(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < entryCount; i++)
        {
            int offset = i * entrySize + (entrySize == 40 ? 36 : 28);
            uint nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(mogi.AsSpan(offset, 4));
            string name = WmoRootReaderCommon.ResolveStringAtOffset(mogn, nameOffset);
            if (string.IsNullOrEmpty(name))
            {
                unresolved++;
                continue;
            }

            resolved++;
            names.Add(name);
            maxLength = Math.Max(maxLength, name.Length);
        }

        return new WmoGroupNameReferenceSummary(sourcePath, version, entryCount, resolved, unresolved, names.Count, maxLength);
    }
}
