namespace WowViewer.Core.Audio;

/// <summary>
/// Interleaved PCM samples ready for a runtime audio backend.
/// Compressed client formats are normalized to signed 16-bit PCM here.
/// </summary>
public sealed record PcmAudioData(int Channels, int BitsPerSample, int SampleRate, byte[] PcmBytes)
{
    public int FrameCount => Channels <= 0 || BitsPerSample <= 0
        ? 0
        : PcmBytes.Length / (Channels * (BitsPerSample / 8));
}
