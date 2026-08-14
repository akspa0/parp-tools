using System.Buffers.Binary;
using System.Text;

namespace WowViewer.Core.Audio;

public sealed record StandardMidiFileInfo(ushort Format, ushort TrackCount, ushort Division, int HeaderLength);

/// <summary>Validates the SMF header before a platform or synth backend is asked to play it.</summary>
public static class StandardMidiFileProbe
{
    public static bool TryRead(ReadOnlySpan<byte> bytes, out StandardMidiFileInfo? info, out string reason)
    {
        info = null;
        reason = string.Empty;
        if (bytes.Length < 14 || Encoding.ASCII.GetString(bytes[..4]) != "MThd")
        {
            reason = "not a standard MIDI file (missing MThd header)";
            return false;
        }

        uint headerLength = BinaryPrimitives.ReadUInt32BigEndian(bytes.Slice(4, 4));
        if (headerLength < 6 || headerLength > bytes.Length - 8)
        {
            reason = $"invalid MIDI header length {headerLength}";
            return false;
        }

        ushort format = BinaryPrimitives.ReadUInt16BigEndian(bytes.Slice(8, 2));
        ushort trackCount = BinaryPrimitives.ReadUInt16BigEndian(bytes.Slice(10, 2));
        ushort division = BinaryPrimitives.ReadUInt16BigEndian(bytes.Slice(12, 2));
        if (format > 2 || trackCount == 0 || division == 0)
        {
            reason = $"unsupported MIDI header format={format}, tracks={trackCount}, division={division}";
            return false;
        }

        info = new StandardMidiFileInfo(format, trackCount, division, checked((int)headerLength));
        return true;
    }
}
