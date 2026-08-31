namespace WowViewer.Core.Maps;

public enum MapFileKind
{
    Unknown,
    Wdt,
    Adt,
    AdtV23,
    AdtV23Error,
    AdtTex,
    AdtObj,
    AdtLod,
    AdtTex1,
    AdtObj1,
}

public static class MapFileKindExtensions
{
    public static bool IsTextureCompanion(this MapFileKind kind)
        => kind is MapFileKind.AdtTex or MapFileKind.AdtTex1;

    public static bool IsObjectCompanion(this MapFileKind kind)
        => kind is MapFileKind.AdtObj or MapFileKind.AdtObj1;

    public static bool IsAdtFamily(this MapFileKind kind)
        => kind is MapFileKind.Adt
            or MapFileKind.AdtTex
            or MapFileKind.AdtObj
            or MapFileKind.AdtTex1
            or MapFileKind.AdtObj1;

    public static bool IsRecognizedAdt(this MapFileKind kind)
        => kind.IsAdtFamily()
            || kind is MapFileKind.AdtV23
            or MapFileKind.AdtV23Error
            or MapFileKind.AdtLod;
}
