using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Compares the <c>MSUR._0x1C == 0</c> population against the rest, to find what identifies it.
/// </summary>
/// <remarks>
/// <c>_0x1C</c> is the producing placement's Z, and it is zero for a large population that prior work
/// showed is <b>M2 doodad collision</b> - geometric components of it land within 24 units of an MDDF
/// placement 95.1% of the time. Calling it a "remainder" is therefore wrong: it is a second body of
/// object data that simply does not carry a placement height.
///
/// <para>If that is right, its objects must be identified some other way. The candidate is
/// <c>MSLK.GroupObjectId</c>, which earlier work measured at 99.9% distinctness over pure components.
/// This analyzer tests the two populations side by side: if GroupObjectId carries identity for the
/// zero bucket specifically, it should behave differently there than in the WMO population, and the
/// comparison is the evidence - a figure from the zero bucket alone would prove nothing.</para>
/// </remarks>
public static class Pm4ZeroBucketAnalyzer
{
    public static Pm4ZeroBucketReport AnalyzeDirectory(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        var zero = new BucketAccumulator("_0x1C == 0 (no placement height)");
        var placed = new BucketAccumulator("_0x1C != 0 (carries a placement height)");
        int files = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msur.Count == 0)
                continue;

            files++;
            var zeroGroups = new HashSet<uint>();
            var placedGroups = new HashSet<uint>();

            foreach (Pm4MsurEntry s in c.Msur)
            {
                BucketAccumulator bucket = s.PackedParams == 0 ? zero : placed;
                HashSet<uint> groups = s.PackedParams == 0 ? zeroGroups : placedGroups;

                bucket.Surfaces++;
                long start = s._0x18;
                long end = start + s.AttributeMask;
                if (end > c.Mslk.Count)
                    continue;

                for (long j = start; j < end; j++)
                {
                    Pm4MslkEntry link = c.Mslk[(int)j];
                    bucket.Links++;
                    groups.Add(link.GroupObjectId);
                    if (link.GroupObjectId == 0)
                        bucket.ZeroGroupLinks++;
                    if (link.MspiFirstIndex >= 0 && link.MspiIndexCount > 0)
                        bucket.LinksWithWall++;
                }
            }

            zero.DistinctGroups += zeroGroups.Count;
            placed.DistinctGroups += placedGroups.Count;
            if (zeroGroups.Count > 0)
                zero.FilesPresent++;
            if (placedGroups.Count > 0)
                placed.FilesPresent++;
        }

        return new Pm4ZeroBucketReport(resolved, files, zero.ToResult(), placed.ToResult());
    }

    private sealed class BucketAccumulator(string name)
    {
        public string Name { get; } = name;
        public long Surfaces;
        public long Links;
        public long LinksWithWall;
        public long ZeroGroupLinks;
        public long DistinctGroups;
        public int FilesPresent;

        public Pm4BucketResult ToResult() => new(
            Name, Surfaces, Links, DistinctGroups, FilesPresent,
            Surfaces == 0 ? 0 : (double)Links / Surfaces,
            DistinctGroups == 0 ? 0 : (double)Links / DistinctGroups,
            Links == 0 ? 0 : (double)LinksWithWall / Links,
            Links == 0 ? 0 : (double)ZeroGroupLinks / Links);
    }
}

public sealed record Pm4BucketResult(
    string Name,
    long Surfaces,
    long Links,
    long DistinctGroupObjectIds,
    int FilesPresent,
    double LinksPerSurface,
    double LinksPerDistinctGroup,
    double WallFraction,
    double ZeroGroupIdFraction);

public sealed record Pm4ZeroBucketReport(
    string InputDirectory,
    int Files,
    Pm4BucketResult Zero,
    Pm4BucketResult Placed);
