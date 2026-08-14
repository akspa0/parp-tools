using System.Buffers.Binary;
using System.Text;

namespace WowViewer.Core.Audio;

public sealed record PcmWavData(int Channels, int BitsPerSample, int SampleRate, byte[] PcmBytes)
{
    public int FrameCount => Channels <= 0 || BitsPerSample <= 0
        ? 0
        : PcmBytes.Length / (Channels * (BitsPerSample / 8));
}

/// <summary>
/// Small, dependency-free PCM WAV decoder for the first interactive viewer
/// backend. Compressed WAV variants remain explicitly unsupported.
/// </summary>
public static class PcmWavDecoder
{
    public static bool TryDecode(ReadOnlySpan<byte> bytes, out PcmWavData? data, out string reason)
    {
        data = null;
        reason = string.Empty;
        if (bytes.Length < 12 || Encoding.ASCII.GetString(bytes[..4]) != "RIFF" || Encoding.ASCII.GetString(bytes[8..12]) != "WAVE")
        {
            reason = "not a RIFF/WAVE file";
            return false;
        }

        ushort format = 0;
        ushort channels = 0;
        ushort bits = 0;
        int sampleRate = 0;
        ReadOnlySpan<byte> pcm = default;
        int position = 12;
        while (position + 8 <= bytes.Length)
        {
            string chunkId = Encoding.ASCII.GetString(bytes.Slice(position, 4));
            uint declaredSize = BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(position + 4, 4));
            position += 8;
            if (declaredSize > int.MaxValue || position + declaredSize > bytes.Length)
            {
                reason = $"truncated {chunkId} chunk";
                return false;
            }

            ReadOnlySpan<byte> chunk = bytes.Slice(position, (int)declaredSize);
            if (chunkId == "fmt " && chunk.Length >= 16)
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(chunk.Slice(0, 2));
                channels = BinaryPrimitives.ReadUInt16LittleEndian(chunk.Slice(2, 2));
                sampleRate = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(chunk.Slice(4, 4)));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(chunk.Slice(14, 2));
            }
            else if (chunkId == "data")
            {
                pcm = chunk;
            }

            position += (int)declaredSize + ((declaredSize & 1) == 1 ? 1 : 0);
        }

        if (format != 1)
        {
            reason = $"WAVE format {format} is not uncompressed PCM";
            return false;
        }

        if (channels is not (1 or 2) || bits is not (8 or 16) || sampleRate <= 0 || pcm.IsEmpty)
        {
            reason = $"unsupported PCM shape channels={channels}, bits={bits}, rate={sampleRate}";
            return false;
        }

        data = new PcmWavData(channels, bits, sampleRate, pcm.ToArray());
        return true;
    }
}
