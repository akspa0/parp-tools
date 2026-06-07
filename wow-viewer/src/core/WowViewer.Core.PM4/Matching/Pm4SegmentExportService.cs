using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4SegmentExportService
{
    public static Pm4SegmentExportRun Export(string inputPath)
    {
        string resolvedInputPath = Path.GetFullPath(inputPath);
        List<string> warnings = [];
        List<Pm4SegmentExportFile> files = [];

        foreach (string pm4Path in EnumeratePm4Paths(resolvedInputPath))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
            {
                warnings.Add($"Skipped '{pm4Path}' because tile coordinates could not be parsed from the file name.");
                continue;
            }

            IReadOnlyList<Pm4BuiltObjectSegment> segments = Pm4ObjectSegmentBuilder.Build(pm4Path);
            files.Add(new Pm4SegmentExportFile(Path.GetFullPath(pm4Path), tileX, tileY, segments));
        }

        int segmentCount = files.Sum(static file => file.SegmentCount);
        string runId = BuildRunId(resolvedInputPath, files);
        return new Pm4SegmentExportRun(runId, resolvedInputPath, files.Count, segmentCount, files, warnings);
    }

    private static IEnumerable<string> EnumeratePm4Paths(string resolvedInputPath)
    {
        if (File.Exists(resolvedInputPath))
        {
            yield return resolvedInputPath;
            yield break;
        }

        if (!Directory.Exists(resolvedInputPath))
            throw new DirectoryNotFoundException($"PM4 export input '{resolvedInputPath}' was not found.");

        foreach (string path in Directory.EnumerateFiles(resolvedInputPath, "*.pm4", SearchOption.TopDirectoryOnly).OrderBy(Path.GetFileName, StringComparer.OrdinalIgnoreCase))
            yield return path;
    }

    private static string BuildRunId(string resolvedInputPath, IReadOnlyList<Pm4SegmentExportFile> files)
    {
        StringBuilder builder = new();
        builder.Append(resolvedInputPath);

        foreach (Pm4SegmentExportFile file in files)
        {
            builder.Append('|');
            builder.Append(file.SourcePath);
            builder.Append('|');
            builder.Append(file.TileX);
            builder.Append('_');
            builder.Append(file.TileY);
            builder.Append('|');
            foreach (string segmentId in file.Segments.Select(static segment => segment.Segment.SegmentId).OrderBy(static id => id, StringComparer.Ordinal))
            {
                builder.Append(segmentId);
                builder.Append(',');
            }
        }

        byte[] bytes = SHA256.HashData(Encoding.UTF8.GetBytes(builder.ToString()));
        return $"pm4-export-{Convert.ToHexString(bytes.AsSpan(0, 8)).ToLowerInvariant()}";
    }
}
