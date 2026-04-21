using System.Text;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.Tests;

public sealed class AlphaAreaAudioCatalogReaderTests
{
    [Fact]
    public void Load_ReadsArchiveAreaMidiBindingsAndResolvesSequences()
    {
        Dictionary<string, uint> strings = BuildStringOffsets(
            "Elwynn Forest",
            "Sound\\Ambience\\MIDI\\ElwynnDay.mid",
            "Sound\\Ambience\\MIDI\\ElwynnNight.mid",
            "Sound\\Ambience\\MIDI\\ForestNormal.dls",
            "Sound\\Ambience\\MIDI\\UnderWater.mid",
            "Sound\\Ambience\\MIDI\\UnderWater.dls");

        FakeArchiveReader archiveReader = new(
            new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["DBFilesClient\\AreaTable.dbc"] = BuildDbc(
                    fieldCount: 14,
                    rows:
                    [
                        [1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 7u, 8u, 0u, 0u, 0u, strings["Elwynn Forest"]],
                    ],
                    stringBlockEntries: strings.Keys),
                ["DBFilesClient\\AreaMIDIAmbiences.dbc"] = BuildDbc(
                    fieldCount: 5,
                    rows:
                    [
                        [7u, strings["Sound\\Ambience\\MIDI\\ElwynnDay.mid"], strings["Sound\\Ambience\\MIDI\\ElwynnNight.mid"], strings["Sound\\Ambience\\MIDI\\ForestNormal.dls"], FloatBits(0.75f)],
                        [8u, strings["Sound\\Ambience\\MIDI\\UnderWater.mid"], 0u, strings["Sound\\Ambience\\MIDI\\UnderWater.dls"], FloatBits(0.5f)],
                    ],
                    stringBlockEntries: strings.Keys),
            });

        AlphaAreaAudioCatalogReader reader = new();
        var catalog = reader.Load(Array.Empty<string>(), archiveReader, "0.5.3.3368");

        Assert.NotNull(catalog);
        Assert.Single(catalog.Areas);
        Assert.Equal(2, catalog.MidiAmbiences.Count);

        var binding = catalog.TryResolve(1);
        Assert.NotNull(binding);
        Assert.Equal("Elwynn Forest", binding.Area.AreaName);
        Assert.Equal(7, binding.Area.MidiAmbienceId);
        Assert.Equal(8, binding.Area.MidiAmbienceUnderwaterId);
        Assert.Equal("Sound\\Ambience\\MIDI\\ElwynnDay.mid", binding.MidiAmbience?.DaySequence);
        Assert.Equal("Sound\\Ambience\\MIDI\\ElwynnNight.mid", binding.MidiAmbience?.NightSequence);
        Assert.Equal("Sound\\Ambience\\MIDI\\ForestNormal.dls", binding.MidiAmbience?.DlsFile);
        Assert.Equal(0.75f, binding.MidiAmbience?.Volume);
        Assert.Equal("Sound\\Ambience\\MIDI\\UnderWater.mid", binding.UnderwaterMidiAmbience?.DaySequence);
        Assert.Equal("Sound\\Ambience\\MIDI\\UnderWater.dls", binding.UnderwaterMidiAmbience?.DlsFile);
    }

    private static uint FloatBits(float value)
    {
        return BitConverter.ToUInt32(BitConverter.GetBytes(value), 0);
    }

    private static Dictionary<string, uint> BuildStringOffsets(params string[] values)
    {
        Dictionary<string, uint> offsets = new(StringComparer.Ordinal);
        uint offset = 1;

        foreach (string value in values)
        {
            if (offsets.ContainsKey(value))
            {
                continue;
            }

            offsets[value] = offset;
            offset += checked((uint)Encoding.UTF8.GetByteCount(value) + 1);
        }

        return offsets;
    }

    private static byte[] BuildDbc(uint fieldCount, IReadOnlyList<uint[]> rows, IEnumerable<string> stringBlockEntries)
    {
        using MemoryStream stringStream = new();
        stringStream.WriteByte(0);

        foreach (string entry in stringBlockEntries)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(entry);
            stringStream.Write(bytes, 0, bytes.Length);
            stringStream.WriteByte(0);
        }

        using MemoryStream stream = new();
        using BinaryWriter writer = new(stream, Encoding.UTF8, leaveOpen: true);

        writer.Write(0x43424457u);
        writer.Write(checked((uint)rows.Count));
        writer.Write(fieldCount);
        writer.Write(fieldCount * 4u);
        writer.Write(checked((uint)stringStream.Length));

        foreach (uint[] row in rows)
        {
            Assert.Equal((int)fieldCount, row.Length);
            foreach (uint value in row)
            {
                writer.Write(value);
            }
        }

        stringStream.Position = 0;
        stringStream.CopyTo(stream);
        writer.Flush();
        return stream.ToArray();
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