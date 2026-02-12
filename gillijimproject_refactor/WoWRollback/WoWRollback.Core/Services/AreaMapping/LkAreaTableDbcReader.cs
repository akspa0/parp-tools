using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using WoWRollback.Core.Services.Archive;

namespace WoWRollback.Core.Services.AreaMapping
{
    /// <summary>
    /// Minimal reader for LK 3.3.5 AreaTable.dbc (WDBC) from a client root.
    /// Loads IDs only and returns a set of LK Area IDs.
    /// Uses PrioritizedArchiveSource to respect loose files over MPQs.
    /// </summary>
    public static class LkAreaTableDbcReader
    {
        public static HashSet<int> LoadLkAreaIdsFromClient(string lkClientRoot)
        {
            var result = new HashSet<int>();
            if (string.IsNullOrWhiteSpace(lkClientRoot) || !Directory.Exists(lkClientRoot))
            {
                return result;
            }

            var mpqs = ArchiveLocator.LocateMpqs(lkClientRoot);
            using var src = new PrioritizedArchiveSource(lkClientRoot, mpqs);

            // Try common path casings
            var candidates = new[]
            {
                "DBFilesClient/AreaTable.dbc",
                "dbfilesclient/areatable.dbc"
            };

            Stream? s = null;
            string? usedPath = null;
            foreach (var p in candidates)
            {
                if (src.FileExists(p))
                {
                    s = src.OpenFile(p);
                    usedPath = p;
                    break;
                }
            }

            if (s is null)
            {
                return result;
            }

            using (s)
            using (var br = new BinaryReader(s, Encoding.ASCII, leaveOpen: false))
            {
                // WDBC header: 'WDBC' + 4 x int32 (records, fields, record_size, string_block_size)
                var magic = br.ReadBytes(4);
                var magicStr = Encoding.ASCII.GetString(magic);
                if (!string.Equals(magicStr, "WDBC", StringComparison.Ordinal))
                {
                    // Not a classic WDBC file; unsupported here
                    return result;
                }

                int recordCount = br.ReadInt32();
                int fieldCount = br.ReadInt32();
                int recordSize = br.ReadInt32();
                int stringBlockSize = br.ReadInt32();

                // Safety checks
                if (recordCount <= 0 || fieldCount <= 0 || recordSize <= 0)
                {
                    return result;
                }

                var bytes = br.ReadBytes(recordCount * recordSize);
                if (bytes.Length < recordCount * recordSize)
                {
                    return result;
                }

                // First field (int32) is AreaID
                for (int i = 0; i < recordCount; i++)
                {
                    int offset = i * recordSize;
                    if (offset + 4 > bytes.Length) break;
                    int id = BitConverter.ToInt32(bytes, offset);
                    if (id > 0) result.Add(id);
                }
            }

            return result;
        }

        public static Dictionary<int, int> LoadLkAreaToMapIdFromClient(string lkClientRoot)
        {
            var mapByArea = new Dictionary<int, int>();
            if (string.IsNullOrWhiteSpace(lkClientRoot) || !Directory.Exists(lkClientRoot))
            {
                return mapByArea;
            }

            var mpqs = ArchiveLocator.LocateMpqs(lkClientRoot);
            using var src = new PrioritizedArchiveSource(lkClientRoot, mpqs);

            var candidates = new[] { "DBFilesClient/AreaTable.dbc", "dbfilesclient/areatable.dbc" };
            Stream? s = null;
            foreach (var p in candidates)
            {
                if (src.FileExists(p)) { s = src.OpenFile(p); break; }
            }
            if (s is null) return mapByArea;

            using (s)
            using (var br = new BinaryReader(s, Encoding.ASCII, leaveOpen: false))
            {
                var magic = br.ReadBytes(4);
                var magicStr = Encoding.ASCII.GetString(magic);
                if (!string.Equals(magicStr, "WDBC", StringComparison.Ordinal)) return mapByArea;
                int recordCount = br.ReadInt32();
                int fieldCount = br.ReadInt32();
                int recordSize = br.ReadInt32();
                int stringBlockSize = br.ReadInt32();
                var bytes = br.ReadBytes(recordCount * recordSize);
                _ = br.ReadBytes(stringBlockSize);

                for (int i = 0; i < recordCount; i++)
                {
                    int offset = i * recordSize;
                    if (offset + 8 > bytes.Length) break;
                    int id = BitConverter.ToInt32(bytes, offset);
                    int contId = BitConverter.ToInt32(bytes, offset + 4);
                    if (id > 0) mapByArea[id] = contId;
                }
            }
            return mapByArea;
        }
    }
}
