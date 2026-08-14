using System.Buffers.Binary;
using System.Text;

namespace WowViewer.Core.Audio;

public sealed record DlsFileInfo(uint FormSize, string FormType, bool HasInstrumentList, bool HasWavePool);

/// <summary>Confirms a client file is a RIFF DLS collection without decoding it as PCM.</summary>
public static class DlsFileProbe
{
    public static bool TryRead(ReadOnlySpan<byte> bytes, out DlsFileInfo? info, out string reason)
    {
        info = null;
        reason = string.Empty;
        if (bytes.Length < 12 || Encoding.ASCII.GetString(bytes[..4]) != "RIFF")
        {
            reason = "not a RIFF DLS collection";
            return false;
        }

        uint formSize = BinaryPrimitives.ReadUInt32LittleEndian(bytes.Slice(4, 4));
        string formType = Encoding.ASCII.GetString(bytes.Slice(8, 4));
        if (formType != "DLS " || formSize < 4 || formSize > bytes.Length - 8)
        {
            reason = $"unsupported RIFF form '{formType}'";
            return false;
        }

        ReadOnlySpan<byte> body = bytes.Slice(12, checked((int)formSize - 4));
        info = new DlsFileInfo(formSize, formType, ContainsAscii(body, "lins"), ContainsAscii(body, "wvpl"));
        if (!info.HasInstrumentList || !info.HasWavePool)
        {
            reason = "DLS collection is missing lins or wvpl data";
            info = null;
            return false;
        }

        return true;
    }

    private static bool ContainsAscii(ReadOnlySpan<byte> bytes, string value)
    {
        byte[] needle = Encoding.ASCII.GetBytes(value);
        return bytes.IndexOf(needle) >= 0;
    }
}
