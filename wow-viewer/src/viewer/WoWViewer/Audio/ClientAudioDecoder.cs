using System.Buffers.Binary;
using System.Text;
using NLayer;
using NVorbis;
using WowViewer.Core.Audio;

namespace WoWViewer.Audio;

/// <summary>
/// Decodes compressed client audio into the one PCM shape consumed by the
/// active OpenAL runtime. MIDI is intentionally not handled here: it requires
/// a sequence renderer and its matching instrument bank.
/// </summary>
internal static class ClientAudioDecoder
{
    private const int OutputBitsPerSample = 16;
    private const int SamplesPerRead = 4096;
    private const int MaxDecodedPcmBytes = 256 * 1024 * 1024;

    public static bool TryDecode(
        ReadOnlySpan<byte> bytes,
        string? virtualPath,
        out PcmAudioData? data,
        out string reason)
    {
        data = null;
        reason = string.Empty;
        if (bytes.IsEmpty)
        {
            reason = "audio file is empty";
            return false;
        }

        string extension = Path.GetExtension(virtualPath ?? string.Empty).ToLowerInvariant();
        try
        {
            if (extension is ".wav" or ".wave" || IsRiffWave(bytes))
            {
                if (!PcmWavDecoder.TryDecode(bytes, out PcmWavData? wav, out reason) || wav is null)
                    return false;

                data = new PcmAudioData(wav.Channels, wav.BitsPerSample, wav.SampleRate, wav.PcmBytes);
                return true;
            }

            if (extension is ".ogg" or ".oga" || IsOgg(bytes))
                return TryDecodeVorbis(bytes, out data, out reason);

            if (extension is ".mp3" or ".mp2" or ".mpeg" || LooksLikeMpegAudio(bytes))
                return TryDecodeMpeg(bytes, out data, out reason);

            reason = $"unsupported audio extension '{extension}'";
            return false;
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or ArgumentException or InvalidOperationException or EndOfStreamException or OverflowException)
        {
            data = null;
            reason = $"decoder failed: {ex.Message}";
            return false;
        }
    }

    private static bool TryDecodeMpeg(ReadOnlySpan<byte> bytes, out PcmAudioData? data, out string reason)
    {
        data = null;
        reason = string.Empty;
        using MemoryStream stream = new(bytes.ToArray(), writable: false);
        using MpegFile mpeg = new(stream);
        if (mpeg.Channels is not (1 or 2) || mpeg.SampleRate <= 0)
        {
            reason = $"unsupported MP3 shape channels={mpeg.Channels}, rate={mpeg.SampleRate}";
            return false;
        }

        byte[] pcm = DecodeFloatSamples(
            mpeg.Channels,
            mpeg.SampleRate,
            (buffer, offset, count) => mpeg.ReadSamples(buffer, offset, count),
            out reason);
        if (pcm.Length == 0)
            return false;

        data = new PcmAudioData(mpeg.Channels, OutputBitsPerSample, mpeg.SampleRate, pcm);
        return true;
    }

    private static bool TryDecodeVorbis(ReadOnlySpan<byte> bytes, out PcmAudioData? data, out string reason)
    {
        data = null;
        reason = string.Empty;
        using MemoryStream stream = new(bytes.ToArray(), writable: false);
        using VorbisReader vorbis = new(stream, closeOnDispose: false);
        if (vorbis.Channels is not (1 or 2) || vorbis.SampleRate <= 0)
        {
            reason = $"unsupported OGG/Vorbis shape channels={vorbis.Channels}, rate={vorbis.SampleRate}";
            return false;
        }

        byte[] pcm = DecodeFloatSamples(
            vorbis.Channels,
            vorbis.SampleRate,
            (buffer, offset, count) => vorbis.ReadSamples(buffer, offset, count),
            out reason);
        if (pcm.Length == 0)
            return false;

        data = new PcmAudioData(vorbis.Channels, OutputBitsPerSample, vorbis.SampleRate, pcm);
        return true;
    }

    private static byte[] DecodeFloatSamples(
        int channels,
        int sampleRate,
        Func<float[], int, int, int> readSamples,
        out string reason)
    {
        reason = string.Empty;
        using MemoryStream output = new();
        float[] samples = new float[Math.Max(channels, 1) * SamplesPerRead];
        while (true)
        {
            int read = readSamples(samples, 0, samples.Length);
            if (read <= 0)
                break;

            long nextLength = output.Length + (long)read * sizeof(short);
            if (nextLength > MaxDecodedPcmBytes)
            {
                reason = $"decoded PCM exceeds {MaxDecodedPcmBytes / (1024 * 1024)} MiB limit";
                return Array.Empty<byte>();
            }

            byte[] chunk = new byte[read * sizeof(short)];
            for (int index = 0; index < read; index++)
            {
                short sample = (short)Math.Clamp(
                    (int)MathF.Round(samples[index] * short.MaxValue),
                    short.MinValue,
                    short.MaxValue);
                BinaryPrimitives.WriteInt16LittleEndian(chunk.AsSpan(index * sizeof(short)), sample);
            }

            output.Write(chunk);
        }

        if (output.Length == 0)
        {
            reason = $"decoder produced no samples at {sampleRate} Hz";
            return Array.Empty<byte>();
        }

        return output.ToArray();
    }

    private static bool IsRiffWave(ReadOnlySpan<byte> bytes)
        => bytes.Length >= 12
            && Encoding.ASCII.GetString(bytes[..4]) == "RIFF"
            && Encoding.ASCII.GetString(bytes[8..12]) == "WAVE";

    private static bool IsOgg(ReadOnlySpan<byte> bytes)
        => bytes.Length >= 4 && Encoding.ASCII.GetString(bytes[..4]) == "OggS";

    private static bool LooksLikeMpegAudio(ReadOnlySpan<byte> bytes)
        => (bytes.Length >= 3 && Encoding.ASCII.GetString(bytes[..3]) == "ID3")
            || (bytes.Length >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0);
}
