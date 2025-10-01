using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.LichKing;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class AdtPlacementAnalyzer
{
    // LK defaults; earlier (e.g., 0.6.0) commonly use smaller structs
    private const int MddfEntrySizeLk = 36;
    private const int MddfEntrySize060 = 32;
    private const int ModfEntrySizeLk = 64;
    private const int ModfEntrySize060 = 56;

    public static IEnumerable<PlacementRange> AnalyzeAdt(string adtPath)
    {
        var fileBytes = File.ReadAllBytes(adtPath);

        // Walk MVER, then MHDR like AdtLk does
        int offset = 0;
        // Skip first chunk (MVER)
        int mverSize = BitConverter.ToInt32(fileBytes, offset + 4);
        offset = offset + 8 + mverSize;
        if ((mverSize & 1) == 1) offset += 1; // pad

        // MHDR chunk now at 'offset'
        var mhdr = new Mhdr(fileBytes, offset);
        int mhdrStart = offset + 8;

        var (map, row, col) = ParseMapTile(adtPath);

        // MDDF (M2 placements)
        int mddfRel = mhdr.GetOffset(Mhdr.MddfOffset);
        if (mddfRel != 0)
        {
            int mddfAbs = mhdrStart + mddfRel;
            var mddf = new Mddf(fileBytes, mddfAbs);
            var (count, minId, maxId) = ReadUniqueIdRangeAuto(mddf.Data, isMddf: true);
            yield return new PlacementRange(map, row, col, PlacementKind.M2, count, minId, maxId, adtPath);
        }

        // MODF (WMO placements)
        int modfRel = mhdr.GetOffset(Mhdr.ModfOffset);
        if (modfRel != 0)
        {
            int modfAbs = mhdrStart + modfRel;
            var modf = new Modf(fileBytes, modfAbs);
            var (count, minId, maxId) = ReadUniqueIdRangeAuto(modf.Data, isMddf: false);
            yield return new PlacementRange(map, row, col, PlacementKind.WMO, count, minId, maxId, adtPath);
        }
    }

    public static IEnumerable<PlacementEntry> EnumeratePlacements(string adtPath)
    {
        var fileBytes = File.ReadAllBytes(adtPath);

        int offset = 0;
        int mverSize = BitConverter.ToInt32(fileBytes, offset + 4);
        offset = offset + 8 + mverSize;
        if ((mverSize & 1) == 1) offset += 1;

        var mhdr = new Mhdr(fileBytes, offset);
        int mhdrStart = offset + 8;
        var (map, row, col) = ParseMapTile(adtPath);

        int mddfRel = mhdr.GetOffset(Mhdr.MddfOffset);
        if (mddfRel != 0)
        {
            int mddfAbs = mhdrStart + mddfRel;
            var mddf = new Mddf(fileBytes, mddfAbs);
            foreach (var id in EnumerateUniqueIdsAuto(mddf.Data, isMddf: true))
            {
                yield return new PlacementEntry(map, row, col, PlacementKind.M2, id, adtPath);
            }
        }

        int modfRel = mhdr.GetOffset(Mhdr.ModfOffset);
        if (modfRel != 0)
        {
            int modfAbs = mhdrStart + modfRel;
            var modf = new Modf(fileBytes, modfAbs);
            foreach (var id in EnumerateUniqueIdsAuto(modf.Data, isMddf: false))
            {
                yield return new PlacementEntry(map, row, col, PlacementKind.WMO, id, adtPath);
            }
        }
    }

    private static int DetermineEntrySize(byte[] data, bool isMddf)
    {
        if (data == null || data.Length < 8) return 0;
        var candidates = isMddf
            ? new[] { MddfEntrySizeLk, MddfEntrySize060 }
            : new[] { ModfEntrySizeLk, ModfEntrySize060 };
        foreach (var size in candidates)
        {
            if (size > 0 && data.Length % size == 0) return size;
        }
        // Fallback: prefer the larger known size if at least one full record fits
        foreach (var size in candidates)
        {
            if (size > 0 && data.Length >= size) return size;
        }
        return 0;
    }

    private static (int count, uint min, uint max) ReadUniqueIdRangeAuto(byte[] data, bool isMddf)
    {
        int entrySize = DetermineEntrySize(data, isMddf);
        if (entrySize <= 0 || data.Length < entrySize) return (0, 0, 0);
        int count = 0;
        uint min = uint.MaxValue;
        uint max = 0u;
        for (int start = 0; start + entrySize <= data.Length; start += entrySize)
        {
            // UniqueId is commonly the second field (offset +4) for both MDDF and MODF in LK
            int uidOffset = 4; // holds across 0.5.x/0.6.0/LK
            if (start + uidOffset + 4 > data.Length) break;
            uint uniqueId = BitConverter.ToUInt32(data, start + uidOffset);
            count++;
            if (uniqueId < min) min = uniqueId;
            if (uniqueId > max) max = uniqueId;
        }
        if (count == 0) return (0, 0, 0);
        return (count, min, max);
    }

    private static IEnumerable<uint> EnumerateUniqueIdsAuto(byte[] data, bool isMddf)
    {
        int entrySize = DetermineEntrySize(data, isMddf);
        if (entrySize <= 0) yield break;
        for (int start = 0; start + entrySize <= data.Length; start += entrySize)
        {
            int uidOffset = 4;
            if (start + uidOffset + 4 > data.Length) yield break;
            yield return BitConverter.ToUInt32(data, start + uidOffset);
        }
    }

    private static (string map, int row, int col) ParseMapTile(string adtPath)
    {
        var name = Path.GetFileNameWithoutExtension(adtPath);
        // Expect pattern: <map>_<row>_<col>
        int lastUnderscore = name.LastIndexOf('_');
        int prevUnderscore = lastUnderscore > 0 ? name.LastIndexOf('_', lastUnderscore - 1) : -1;
        if (lastUnderscore > 0 && prevUnderscore > 0)
        {
            var map = name.Substring(0, prevUnderscore);
            var rowStr = name.Substring(prevUnderscore + 1, lastUnderscore - prevUnderscore - 1);
            var colStr = name.Substring(lastUnderscore + 1);
            if (int.TryParse(rowStr, out var r) && int.TryParse(colStr, out var c))
            {
                return (map, r, c);
            }
        }
        return ("unknown", -1, -1);
    }
}
