namespace WoWRollback.LkToAlphaModule.Converters;

public sealed class PlacementConverter
{
    public Models.AlphaAdtData ApplyPlacements(Models.LkAdtData src, Models.AlphaAdtData dst, bool skipWmos)
    {
        if (src is null) throw new System.ArgumentNullException(nameof(src));
        if (dst is null) throw new System.ArgumentNullException(nameof(dst));

        CopyDoodads(src, dst);
        if (!skipWmos)
        {
            CopyWmos(src, dst);
        }

        return dst;
    }

    private static void CopyDoodads(Models.LkAdtData src, Models.AlphaAdtData dst)
    {
        dst.MmdxNames.Clear();
        dst.MmidOffsets.Clear();
        dst.MddfPlacements.Clear();

        for (int i = 0; i < src.MmdxNames.Count; i++)
        {
            string name = src.MmdxNames[i];
            if (name.EndsWith(".m2", System.StringComparison.OrdinalIgnoreCase))
            {
                name = name[..^3] + ".mdx";
            }
            dst.MmdxNames.Add(name);
        }

        dst.MmidOffsets.AddRange(src.MmidOffsets);

        for (int i = 0; i < src.MddfPlacements.Count; i++)
        {
            var p = src.MddfPlacements[i];
            int nameIndex = p.NameIndex < dst.MmdxNames.Count ? p.NameIndex : 0;
            dst.MddfPlacements.Add(p with { NameIndex = nameIndex });
        }
    }

    private static void CopyWmos(Models.LkAdtData src, Models.AlphaAdtData dst)
    {
        dst.MwmoNames.Clear();
        dst.MwidOffsets.Clear();
        dst.ModfPlacements.Clear();

        for (int i = 0; i < src.MwmoNames.Count; i++)
        {
            dst.MwmoNames.Add(src.MwmoNames[i]);
        }

        dst.MwidOffsets.AddRange(src.MwidOffsets);

        for (int i = 0; i < src.ModfPlacements.Count; i++)
        {
            var p = src.ModfPlacements[i];
            int nameIndex = p.NameIndex < dst.MwmoNames.Count ? p.NameIndex : 0;
            dst.ModfPlacements.Add(p with { NameIndex = nameIndex });
        }
    }
}
