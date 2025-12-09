using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using GillijimProject.Utilities;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Alpha;

namespace AlphaLkToAlphaStandalone.Core
{
    internal static class AreaIdRoundtripWriter
    {
        public static void WriteAreaIdRoundtripCsv(string alphaWdtPath, string mapName, string lkExportRoot, string outputCsvPath)
        {
            var worldMapsDir = Path.Combine(lkExportRoot, "World", "Maps", mapName);
            if (!File.Exists(alphaWdtPath) || !Directory.Exists(worldMapsDir))
            {
                return;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(outputCsvPath)!);

            var wdtAlpha = new WdtAlpha(alphaWdtPath);
            var adtOffsets = wdtAlpha.GetAdtOffsetsInMain();

            using var writer = new StreamWriter(outputCsvPath, false);
            writer.WriteLine("tileX,tileY,mcnkIndex,alpha_area_id,lk_area_id");

            foreach (var adtPath in Directory.EnumerateFiles(worldMapsDir, "*.adt", SearchOption.TopDirectoryOnly))
            {
                var fileName = Path.GetFileNameWithoutExtension(adtPath);
                if (string.IsNullOrEmpty(fileName)) continue;
                var parts = fileName.Split('_');
                if (parts.Length < 3) continue;
                if (!string.Equals(parts[0], mapName, StringComparison.OrdinalIgnoreCase)) continue;
                if (!int.TryParse(parts[^2], out var tileX)) continue;
                if (!int.TryParse(parts[^1], out var tileY)) continue;

                var tileIndex = tileY * 64 + tileX;
                if (tileIndex < 0 || tileIndex >= adtOffsets.Count) continue;

                var alphaOffset = adtOffsets[tileIndex];
                List<int> alphaAreaIds;
                if (alphaOffset > 0)
                {
                    var adtAlpha = new AdtAlpha(alphaWdtPath, alphaOffset, tileIndex);
                    alphaAreaIds = adtAlpha.GetAlphaMcnkAreaIds();
                }
                else
                {
                    alphaAreaIds = new List<int>(capacity: 256);
                    for (int i = 0; i < 256; i++) alphaAreaIds.Add(-1);
                }

                var lkAreaIds = ReadLkMcnkAreaIds(adtPath);

                for (int mcnkIndex = 0; mcnkIndex < 256; mcnkIndex++)
                {
                    var alphaAreaId = mcnkIndex < alphaAreaIds.Count ? alphaAreaIds[mcnkIndex] : -1;
                    if (!lkAreaIds.TryGetValue(mcnkIndex, out var lkAreaId)) lkAreaId = -1;
                    writer.WriteLine(string.Join(',', tileX, tileY, mcnkIndex, alphaAreaId, lkAreaId));
                }
            }
        }

        private static Dictionary<int, int> ReadLkMcnkAreaIds(string adtPath)
        {
            var result = new Dictionary<int, int>();
            using var fs = new FileStream(adtPath, FileMode.Open, FileAccess.Read, FileShare.Read);

            var mver = new Chunk(fs, 0);
            var mhdrChunkOffset = (int)fs.Position;
            var mhdr = new Mhdr(fs, mhdrChunkOffset);
            var mhdrStartOffset = mhdrChunkOffset + WowChunkedFormat.ChunkLettersAndSize;
            var mcinOffsetAbs = mhdrStartOffset + mhdr.GetOffset(Mhdr.McinOffset);
            var mcin = new Mcin(fs, mcinOffsetAbs);
            var offsets = mcin.GetMcnkOffsets();

            var headerSize = Marshal.SizeOf<McnkHeader>();

            for (int i = 0; i < offsets.Count && i < 256; i++)
            {
                var off = offsets[i];
                if (off <= 0) continue;
                var headerBytes = Utilities.GetByteArrayFromFile(fs, off + WowChunkedFormat.ChunkLettersAndSize, headerSize);
                var header = Utilities.ByteArrayToStruct<McnkHeader>(headerBytes);
                result[i] = header.AreaId;
            }

            return result;
        }
    }
}
