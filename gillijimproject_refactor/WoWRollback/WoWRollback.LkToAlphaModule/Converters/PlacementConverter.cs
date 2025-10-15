namespace WoWRollback.LkToAlphaModule.Converters;

public sealed class PlacementConverter
{
    public Models.AlphaAdtData ApplyPlacements(Models.LkAdtData src, Models.AlphaAdtData dst, bool skipWmos)
    {
        // TODO: Copy MMDX/MMID/MDDF; MWMO/MWID/MODF (respect skipWmos), rename .m2 -> .mdx
        return dst;
    }
}
