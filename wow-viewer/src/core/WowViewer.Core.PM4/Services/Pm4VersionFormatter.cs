namespace WowViewer.Core.PM4.Services;

public static class Pm4VersionFormatter
{
    public static string Format(uint rawVersion)
    {
        if (rawVersion == 0)
            return "v0 (unknown)";

        byte b0 = (byte)(rawVersion & 0xFF);
        byte b1 = (byte)((rawVersion >> 8) & 0xFF);

        // Cataclysm Beta (2010): Byte 8 = 0x10 (Version 16)
        if (rawVersion == 0x3010 || (b0 == 0x10 && b1 == 0x30))
            return "v16 (Cataclysm Beta, 0x3010 / 12304)";

        // WoD (2014): Byte 8 = 0x30 (Version 48)
        if (rawVersion == 0x30 || (b0 == 0x30 && b1 == 0x00))
            return "v48 (WoD, 0x0030 / 48)";

        if (b1 != 0)
            return $"v{b0} (flags=0x{b1:X2}, 0x{rawVersion:X4})";

        return $"v{b0} (0x{rawVersion:X4})";
    }

    public static string FormatShort(uint rawVersion)
    {
        if (rawVersion == 0)
            return "v0";

        byte b0 = (byte)(rawVersion & 0xFF);
        byte b1 = (byte)((rawVersion >> 8) & 0xFF);

        if (rawVersion == 0x3010 || (b0 == 0x10 && b1 == 0x30))
            return "v16";

        if (rawVersion == 0x30 || (b0 == 0x30 && b1 == 0x00))
            return "v48";

        return $"v{b0}";
    }
}
