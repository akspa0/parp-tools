namespace WowViewer.Core.IO.M2Era1121;

public enum M2Era1121Version
{
    Unknown = 0,
    V100 = 0x100,
    V101 = 0x101,
}

public static class M2Era1121VersionExtensions
{
    public static bool Is1121(this M2Era1121Version version)
        => version is M2Era1121Version.V100 or M2Era1121Version.V101;

    public static M2Era1121Version FromUInt(uint raw)
        => raw switch
        {
            0x100u => M2Era1121Version.V100,
            0x101u => M2Era1121Version.V101,
            _ => M2Era1121Version.Unknown,
        };
}
