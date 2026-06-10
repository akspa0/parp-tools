using WowViewer.Core.PM4.Matching;

namespace WowViewer.Core.PM4.Tests;

public sealed class Pm4SegmentExportServiceTests
{
    [Fact]
    public void Export_DevelopmentTile_ProducesDeterministicSingleFileRun()
    {
        Pm4SegmentExportRun first = Pm4SegmentExportService.Export(Pm4TestPaths.DevelopmentTilePath);
        Pm4SegmentExportRun second = Pm4SegmentExportService.Export(Pm4TestPaths.DevelopmentTilePath);

        Assert.Equal(1, first.FileCount);
        Assert.Equal(first.RunId, second.RunId);
        Assert.Equal(first.SegmentCount, second.SegmentCount);
        Assert.Empty(first.Warnings);

        Pm4SegmentExportFile file = Assert.Single(first.Files);
        Assert.Equal(0, file.TileX);
        Assert.Equal(0, file.TileY);
        Assert.EndsWith("development_00_00.pm4", file.SourcePath, StringComparison.OrdinalIgnoreCase);
        Assert.True(file.SegmentCount > 0);
    }

    [Fact]
    public void Export_DevelopmentDirectory_ProducesStableOrderedCorpusRun()
    {
        using BoundedPm4Directory boundedDirectory = BoundedPm4Directory.Create(Pm4TestPaths.DevelopmentDirectoryPath, 4);
        Pm4SegmentExportRun first = Pm4SegmentExportService.Export(boundedDirectory.Path);
        Pm4SegmentExportRun second = Pm4SegmentExportService.Export(boundedDirectory.Path);

        Assert.Equal(4, first.FileCount);
        Assert.True(first.SegmentCount > 0);
        Assert.Empty(first.Warnings);
        Assert.Equal(first.RunId, second.RunId);
        Assert.Equal(first.FileCount, second.FileCount);
        Assert.Equal(first.SegmentCount, second.SegmentCount);
        Assert.Equal(
            first.Files.SelectMany(static file => file.Segments.Select(static segment => segment.Segment.SegmentId)),
            second.Files.SelectMany(static file => file.Segments.Select(static segment => segment.Segment.SegmentId)));

        List<string> orderedNames = first.Files.Select(static file => Path.GetFileName(file.SourcePath)).ToList();
        List<string> sortedNames = orderedNames.OrderBy(static name => name, StringComparer.OrdinalIgnoreCase).ToList();
        Assert.Equal(sortedNames, orderedNames);
    }

    private sealed class BoundedPm4Directory : IDisposable
    {
        private BoundedPm4Directory(string path)
        {
            Path = path;
        }

        public string Path { get; }

        public static BoundedPm4Directory Create(string sourceDirectory, int maxFiles)
        {
            string tempDirectory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"pm4-export-tests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDirectory);

            foreach (string sourcePath in Directory.EnumerateFiles(sourceDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
                .OrderBy(System.IO.Path.GetFileName, StringComparer.OrdinalIgnoreCase)
                .Take(maxFiles))
            {
                string targetPath = System.IO.Path.Combine(tempDirectory, System.IO.Path.GetFileName(sourcePath));
                File.Copy(sourcePath, targetPath, overwrite: true);
            }

            return new BoundedPm4Directory(tempDirectory);
        }

        public void Dispose()
        {
            if (Directory.Exists(Path))
                Directory.Delete(Path, recursive: true);
        }
    }
}
