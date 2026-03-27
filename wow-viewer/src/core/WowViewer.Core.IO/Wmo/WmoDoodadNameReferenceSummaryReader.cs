using System.Buffers.Binary;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoDoodadNameReferenceSummaryReader
{
    private const int ModdEntrySize = 40;

    public static WmoDoodadNameReferenceSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoDoodadNameReferenceSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        byte[] modn = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modn);
        byte[] modd = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Modd);
        if (modd.Length % ModdEntrySize != 0)
            throw new InvalidDataException($"MODD payload size {modd.Length} is not divisible by {ModdEntrySize}.");

        int entryCount = modd.Length / ModdEntrySize;
        int resolved = 0;
        int unresolved = 0;
        int maxLength = 0;
        HashSet<string> names = new(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < entryCount; i++)
        {
            uint nameOffset = BinaryPrimitives.ReadUInt32LittleEndian(modd.AsSpan(i * ModdEntrySize, 4)) & 0x00FFFFFFu;
            string name = WmoRootReaderCommon.ResolveStringAtOffset(modn, nameOffset);
            if (string.IsNullOrEmpty(name))
            {
                unresolved++;
                continue;
            }

            resolved++;
            names.Add(name);
            maxLength = Math.Max(maxLength, name.Length);
        }

        return new WmoDoodadNameReferenceSummary(sourcePath, version, entryCount, resolved, unresolved, names.Count, maxLength);
    }
}
