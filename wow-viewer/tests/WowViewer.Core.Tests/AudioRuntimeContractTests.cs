using System.Buffers.Binary;
using WowViewer.Core.Audio;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class AudioRuntimeContractTests
{
    [Fact]
    public void OpenAlProbe_MissingOptionalLibraryFailsClosed()
    {
        bool available = OpenAlNativeLibraryProbe.TryFind(
            ["wowviewer-openal-library-that-does-not-exist"],
            out string? libraryName);

        Assert.False(available);
        Assert.Null(libraryName);
    }

    [Fact]
    public void Alpha053Mcse_DecodesSoundIdentityPositionAndRanges()
    {
        const int relativeEmitterOffset = 0x20;
        byte[] payload = new byte[0x80 + relativeEmitterOffset + AdtMcseReader.Alpha053EntrySize];
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x5C, 4), relativeEmitterOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x60, 4), 1);

        Span<byte> emitter = payload.AsSpan(0x80 + relativeEmitterOffset, AdtMcseReader.Alpha053EntrySize);
        BinaryPrimitives.WriteUInt32LittleEndian(emitter.Slice(0x00, 4), 7);
        BinaryPrimitives.WriteUInt32LittleEndian(emitter.Slice(0x04, 4), 1234);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x08, 4), 1.5f);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x0C, 4), 2.5f);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x10, 4), 3.5f);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x14, 4), 4f);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x18, 4), 25f);
        BinaryPrimitives.WriteSingleLittleEndian(emitter.Slice(0x1C, 4), 30f);

        AdtMcseData result = AdtMcseReader.ReadAlpha053Mcnk(payload);

        AdtMcseEmitter decoded = Assert.Single(result.Emitters);
        Assert.Equal((uint)7, decoded.SoundPointId);
        Assert.Equal((uint)1234, decoded.SoundNameId);
        Assert.Equal(1.5f, decoded.Position.X);
        Assert.Equal(2.5f, decoded.Position.Y);
        Assert.Equal(3.5f, decoded.Position.Z);
        Assert.Equal(4f, decoded.MinDistance);
        Assert.Equal(25f, decoded.MaxDistance);
        Assert.Equal(30f, decoded.CutoffDistance);
        Assert.Equal(AdtMcseReader.Alpha053EntrySize, decoded.RawEntry.Length);
    }

    [Fact]
    public void PcmWavDecoder_ReadsMono16Pcm()
    {
        byte[] wav = CreatePcmWav(sampleRate: 22050, channels: 1, bitsPerSample: 16, pcm: [0, 0, 0xFF, 0x7F]);

        Assert.True(PcmWavDecoder.TryDecode(wav, out PcmWavData? decoded, out string reason), reason);
        Assert.NotNull(decoded);
        Assert.Equal(1, decoded!.Channels);
        Assert.Equal(16, decoded.BitsPerSample);
        Assert.Equal(22050, decoded.SampleRate);
        Assert.Equal(4, decoded.PcmBytes.Length);
    }

    [Fact]
    public void StandardMidiFileProbe_RecognizesSmfHeader()
    {
        byte[] midi = [
            (byte)'M', (byte)'T', (byte)'h', (byte)'d', 0, 0, 0, 6,
            0, 1, 0, 1, 0, 96
        ];

        Assert.True(StandardMidiFileProbe.TryRead(midi, out StandardMidiFileInfo? info, out string reason), reason);
        Assert.NotNull(info);
        Assert.Equal((ushort)1, info!.Format);
        Assert.Equal((ushort)1, info.TrackCount);
        Assert.Equal((ushort)96, info.Division);
    }

    [Fact]
    public void DlsFileProbe_RequiresDlsFormAndCoreLists()
    {
        byte[] dls = [
            (byte)'R', (byte)'I', (byte)'F', (byte)'F', 0x0C, 0, 0, 0,
            (byte)'D', (byte)'L', (byte)'S', (byte)' ',
            (byte)'l', (byte)'i', (byte)'n', (byte)'s',
            (byte)'w', (byte)'v', (byte)'p', (byte)'l'
        ];

        Assert.True(DlsFileProbe.TryRead(dls, out DlsFileInfo? info, out string reason), reason);
        Assert.NotNull(info);
        Assert.True(info!.HasInstrumentList);
        Assert.True(info.HasWavePool);
    }

    [Fact]
    public void MidiDlsAssetPairProbe_RequiresTheMatchingClientBank()
    {
        Assert.True(
            MidiDlsAssetPair.TryCreate(
                "Sound\\Ambience\\MIDI\\ElwynnDay.mid",
                "Sound\\Ambience\\MIDI\\ForestNormal.dls",
                out MidiDlsAssetPair? pair,
                out string pairReason),
            pairReason);

        byte[] midi = [
            (byte)'M', (byte)'T', (byte)'h', (byte)'d', 0, 0, 0, 6,
            0, 1, 0, 1, 0, 96
        ];
        byte[] dls = [
            (byte)'R', (byte)'I', (byte)'F', (byte)'F', 0x0C, 0, 0, 0,
            (byte)'D', (byte)'L', (byte)'S', (byte)' ',
            (byte)'l', (byte)'i', (byte)'n', (byte)'s',
            (byte)'w', (byte)'v', (byte)'p', (byte)'l'
        ];

        Assert.True(
            MidiDlsAssetPairProbe.TryRead(pair!, midi, dls, out MidiDlsAssetPairProbeResult? result, out string reason),
            reason);
        Assert.NotNull(result);
        Assert.Equal(pair, result!.Pair);
        Assert.Equal((ushort)1, result.Midi.Format);
        Assert.True(result.Dls.HasWavePool);

        Assert.False(
            MidiDlsAssetPair.TryCreate("ElwynnDay.mid", null, out _, out string missingBankReason));
        Assert.Contains("DLS", missingBankReason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void DlsFileProbe_RejectsTruncatedFormWithoutThrowing()
    {
        byte[] truncated = [
            (byte)'R', (byte)'I', (byte)'F', (byte)'F', 0, 0, 0, 0,
            (byte)'D', (byte)'L', (byte)'S', (byte)' '
        ];

        Assert.False(DlsFileProbe.TryRead(truncated, out _, out string reason));
        Assert.NotEmpty(reason);
    }

    private static byte[] CreatePcmWav(int sampleRate, ushort channels, ushort bitsPerSample, byte[] pcm)
    {
        const int fmtSize = 16;
        int byteRate = sampleRate * channels * (bitsPerSample / 8);
        short blockAlign = checked((short)(channels * (bitsPerSample / 8)));
        byte[] wav = new byte[12 + 8 + fmtSize + 8 + pcm.Length];
        wav.AsSpan(0, 4).Fill((byte)'R');
        "RIFF"u8.CopyTo(wav.AsSpan(0, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(4, 4), checked((uint)(wav.Length - 8)));
        "WAVE"u8.CopyTo(wav.AsSpan(8, 4));
        "fmt "u8.CopyTo(wav.AsSpan(12, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(16, 4), fmtSize);
        BinaryPrimitives.WriteUInt16LittleEndian(wav.AsSpan(20, 2), 1);
        BinaryPrimitives.WriteUInt16LittleEndian(wav.AsSpan(22, 2), channels);
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(24, 4), checked((uint)sampleRate));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(28, 4), checked((uint)byteRate));
        BinaryPrimitives.WriteUInt16LittleEndian(wav.AsSpan(32, 2), checked((ushort)blockAlign));
        BinaryPrimitives.WriteUInt16LittleEndian(wav.AsSpan(34, 2), bitsPerSample);
        "data"u8.CopyTo(wav.AsSpan(36, 4));
        BinaryPrimitives.WriteUInt32LittleEndian(wav.AsSpan(40, 4), checked((uint)pcm.Length));
        pcm.CopyTo(wav.AsSpan(44));
        return wav;
    }
}
