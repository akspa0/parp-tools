namespace WoWRollback.LkToAlphaModule.Utils;

public static class Validation
{
    public static bool BasicTileStructureOk(Models.AlphaAdtData data)
    {
        // TODO: ensure 256 MCNK, expected chunk ordering and counts
        return data is not null;
    }
}
