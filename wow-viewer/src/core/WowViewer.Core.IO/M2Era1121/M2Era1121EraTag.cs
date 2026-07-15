namespace WowViewer.Core.IO.M2Era1121;

public enum M2Era1121EraTag
{
    Unknown = 99,
    Mdlx = 0,
    Md20_1X_V100 = 1,
    Md20_1X_V101 = 2,
    Md20_3X_V108 = 3,
    Md20_4X_V109 = 4,
    /// <summary>WoW 1.0.0 (build 3980, beta-3) — version 0x100 with the classic M2Vertex + M2Division layout.</summary>
    Md20_1X_V100_Era100 = 5,
}

public static class M2Era1121EraTagExtensions
{
    public static string ToDisplayString(this M2Era1121EraTag era)
        => era switch
        {
            M2Era1121EraTag.Mdlx => "MDLX (chunked)",
            M2Era1121EraTag.Md20_1X_V100 => "1.12.1 (MD20 v0x100)",
            M2Era1121EraTag.Md20_1X_V101 => "1.12.1 (MD20 v0x101)",
            M2Era1121EraTag.Md20_1X_V100_Era100 => "1.0.0 (MD20 v0x100, classic layout)",
            M2Era1121EraTag.Md20_3X_V108 => "3.3.5 (MD20 v0x108)",
            M2Era1121EraTag.Md20_4X_V109 => "4.x / Cata+ (MD20 v0x109)",
            _ => "Unknown",
        };
}
