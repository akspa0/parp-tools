namespace WowViewer.Core.IO.M2Era1121;

public enum M2Era1121EraTag
{
    Unknown = 99,
    Mdlx = 0,
    Md20_1X_V100 = 1,
    Md20_1X_V101 = 2,
    Md20_3X_V108 = 3,
}

public static class M2Era1121EraTagExtensions
{
    public static string ToDisplayString(this M2Era1121EraTag era)
        => era switch
        {
            M2Era1121EraTag.Mdlx => "MDLX (chunked)",
            M2Era1121EraTag.Md20_1X_V100 => "1.12.1 (MD20 v0x100)",
            M2Era1121EraTag.Md20_1X_V101 => "1.12.1 (MD20 v0x101)",
            M2Era1121EraTag.Md20_3X_V108 => "3.3.5 (MD20 v0x108)",
            _ => "Unknown",
        };
}
