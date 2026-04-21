using WowViewer.Core.Audio;
using WowViewer.Core.IO.Audio;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class AlphaAreaAudioAssetResolverTests
{
    [Fact]
    public void Resolve_PrefersDiskAssetsBeforeArchiveAndMarksMissingReferences()
    {
        string root = Path.Combine(Path.GetTempPath(), $"{nameof(AlphaAreaAudioAssetResolverTests)}_{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);

        try
        {
            string dayPath = Path.Combine(root, "Data", "Sound", "Ambience", "MIDI");
            Directory.CreateDirectory(dayPath);
            string dayFile = Path.Combine(dayPath, "ElwynnDay.mid");
            File.WriteAllBytes(dayFile, [1, 2, 3]);

            AlphaAreaAudioBinding binding = new(
                new AlphaAreaRecord(1, 0, 0, "Elwynn Forest", 7, 8, 0, 0, 0),
                new AlphaAreaMidiAmbience(
                    7,
                    "Sound\\Ambience\\MIDI\\ElwynnDay.mid",
                    "Sound\\Ambience\\MIDI\\ElwynnNight.mid",
                    "Sound\\Ambience\\MIDI\\ForestNormal.dls",
                    0.75f),
                new AlphaAreaMidiAmbience(
                    8,
                    "Sound\\Ambience\\MIDI\\UnderWater.mid",
                    string.Empty,
                    "Sound\\Ambience\\MIDI\\UnderWater.dls",
                    0.50f));

            FakeArchiveReader archiveReader = new(
                new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
                {
                    ["Sound\\Ambience\\MIDI\\ElwynnDay.mid"] = [9],
                    ["Sound\\Ambience\\MIDI\\ForestNormal.dls"] = [8],
                    ["Sound\\Ambience\\MIDI\\UnderWater.mid"] = [7],
                });

            AlphaAreaAudioAssetResolver resolver = new();
            AlphaAreaAudioBindingAssetReport report = resolver.Resolve(binding, [root], archiveReader);

            Assert.True(report.DaySequence.Exists);
            Assert.Equal(AlphaAreaAudioAssetSource.Disk, report.DaySequence.Source);
            Assert.Equal(Path.GetFullPath(dayFile), report.DaySequence.ResolvedPath);

            Assert.False(report.NightSequence.Exists);
            Assert.Equal(AlphaAreaAudioAssetSource.None, report.NightSequence.Source);

            Assert.True(report.DlsFile.Exists);
            Assert.Equal(AlphaAreaAudioAssetSource.Archive, report.DlsFile.Source);
            Assert.Equal("Sound\\Ambience\\MIDI\\ForestNormal.dls", report.DlsFile.ResolvedPath);

            Assert.True(report.UnderwaterDaySequence.Exists);
            Assert.Equal(AlphaAreaAudioAssetSource.Archive, report.UnderwaterDaySequence.Source);

            Assert.False(report.UnderwaterNightSequence.IsReferenced);
            Assert.False(report.UnderwaterNightSequence.Exists);

            Assert.False(report.UnderwaterDlsFile.Exists);
            Assert.Equal(AlphaAreaAudioAssetSource.None, report.UnderwaterDlsFile.Source);
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }
    }

    private sealed class FakeArchiveReader : IArchiveReader
    {
        private readonly IReadOnlyDictionary<string, byte[]> _files;

        public FakeArchiveReader(IReadOnlyDictionary<string, byte[]> files)
        {
            _files = files;
        }

        public bool FileExists(string virtualPath)
        {
            return _files.ContainsKey(virtualPath);
        }

        public byte[]? ReadFile(string virtualPath)
        {
            return _files.TryGetValue(virtualPath, out byte[]? data) ? data : null;
        }
    }
}
