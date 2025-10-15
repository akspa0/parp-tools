namespace WoWRollback.LkToAlphaModule.Converters;

public sealed class LiquidConverter
{
    public Models.AlphaAdtData ApplyLiquids(Models.LkAdtData src, Models.AlphaAdtData dst, bool skipLiquids)
    {
        if (skipLiquids) return dst; // Phase 1: skip by default
        // TODO: Convert MH2O -> MCLQ when enabled
        return dst;
    }
}
