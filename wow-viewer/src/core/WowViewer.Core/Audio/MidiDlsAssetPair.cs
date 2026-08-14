namespace WowViewer.Core.Audio;

/// <summary>
/// The two client assets that form one historical MIDI ambience program.
/// A sequence without its declared DLS collection is not a playable ambience
/// binding for the Alpha clients.
/// </summary>
public sealed record MidiDlsAssetPair(string MidiPath, string DlsPath)
{
    public static bool TryCreate(
        string? midiPath,
        string? dlsPath,
        out MidiDlsAssetPair? pair,
        out string reason)
    {
        pair = null;
        reason = string.Empty;
        string midi = Normalize(midiPath);
        string dls = Normalize(dlsPath);
        if (string.IsNullOrEmpty(midi))
        {
            reason = "MIDI sequence path is missing";
            return false;
        }

        string midiExtension = Path.GetExtension(midi);
        if (!midiExtension.Equals(".mid", StringComparison.OrdinalIgnoreCase)
            && !midiExtension.Equals(".midi", StringComparison.OrdinalIgnoreCase))
        {
            reason = $"MIDI asset has unsupported extension '{midiExtension}'";
            return false;
        }

        if (string.IsNullOrEmpty(dls))
        {
            reason = "matching DLS soundbank path is missing";
            return false;
        }

        string dlsExtension = Path.GetExtension(dls);
        if (!dlsExtension.Equals(".dls", StringComparison.OrdinalIgnoreCase))
        {
            reason = $"instrument bank has unsupported extension '{dlsExtension}'";
            return false;
        }

        pair = new MidiDlsAssetPair(midi, dls);
        return true;
    }

    private static string Normalize(string? path)
        => string.IsNullOrWhiteSpace(path)
            ? string.Empty
            : path.Trim().TrimStart('\\', '/').Replace('/', '\\');
}

public sealed record MidiDlsAssetPairProbeResult(
    MidiDlsAssetPair Pair,
    StandardMidiFileInfo Midi,
    DlsFileInfo Dls);

/// <summary>
/// Validates the exact MIDI+DLS pair before a future sequencer/synth consumes it.
/// This deliberately does not decode the DLS into PCM.
/// </summary>
public static class MidiDlsAssetPairProbe
{
    public static bool TryRead(
        MidiDlsAssetPair pair,
        ReadOnlySpan<byte> midiBytes,
        ReadOnlySpan<byte> dlsBytes,
        out MidiDlsAssetPairProbeResult? result,
        out string reason)
    {
        ArgumentNullException.ThrowIfNull(pair);
        result = null;
        reason = string.Empty;

        if (!StandardMidiFileProbe.TryRead(midiBytes, out StandardMidiFileInfo? midi, out reason) || midi is null)
            return false;

        if (!DlsFileProbe.TryRead(dlsBytes, out DlsFileInfo? dls, out reason) || dls is null)
            return false;

        result = new MidiDlsAssetPairProbeResult(pair, midi, dls);
        return true;
    }
}
