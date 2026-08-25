using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests what <c>MSLK._0x04</c> actually is, treating its inherited name as unexamined.
/// </summary>
/// <remarks>
/// It is called a group/object id, and the sweep contradicts that directly: 32,484 distinct values over
/// 486,819 records with a per-file distinct ratio of 0.676, which is a per-RECORD identifier, not a
/// group key. A per-record identifier inside a chunk already proven to be an adjacency graph is a
/// candidate <b>node id</b>, and that has never been tested.
///
/// <para>Hypotheses tested, each against what the data would have to look like:</para>
/// <list type="number">
/// <item><b>Array position.</b> If it were the record's own index it would equal it. Trivial to check
/// and worth checking, because a field that duplicates the index carries no information.</item>
/// <item><b>Dense id space.</b> A node id assigned by a builder is usually a permutation of
/// <c>0..N-1</c>, or close. Sparse or wildly out-of-range values mean it is something else.</item>
/// <item><b>Referent of <c>RefIndex</c>.</b> If <c>RefIndex</c> names a neighbour by NODE ID rather
/// than by array position, then every <c>RefIndex</c> should appear in the set of <c>_0x04</c> values.
/// The control is whether it instead falls in <c>0..recordCount</c>, which is what an array position
/// does.</item>
/// <item><b>Group key.</b> If it really were a group id, records sharing a value should be numerous.
/// Group-size distribution settles it.</item>
/// </list>
///
/// <para>Reported together so no single number can be read as an answer on its own.</para>
/// </remarks>
internal static class Pm4MslkNodeIdSupport
{
    public static Pm4MslkNodeIdReport Analyze(string inputDirectory, int maxFiles = 200)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        int files = 0;
        long records = 0;
        long equalsOwnIndex = 0;
        long densePermutationFiles = 0;
        long withinRecordCount = 0;
        long refIndexInNodeIdSet = 0;
        long refIndexWithinRecordCount = 0;
        long refIndexTested = 0;
        var groupSizes = new Dictionary<int, long>();

        // A value shared by exactly two records is the signature of an EDGE id joining two half-edges.
        // If that is what it is, the pair should be structurally related: adjacent in the array, or
        // pointing at each other, or sharing a target. Measured against a control of arbitrary pairs.
        long pairsTested = 0, pairsAdjacent = 0, pairsSameRefIndex = 0, pairsPointAtEachOther = 0;
        long pairsSameTypeFlags = 0, controlSameRefIndex = 0;
        long singletonGroups = 0, totalGroups = 0;
        double distinctRatioSum = 0;
        long maxValueSeen = 0;
        var samples = new List<string>();

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (chunks.Mslk.Count == 0)
                continue;

            files++;
            int count = chunks.Mslk.Count;
            records += count;

            var ids = new HashSet<uint>();
            var byId = new Dictionary<uint, int>();

            for (int i = 0; i < count; i++)
            {
                uint id = chunks.Mslk[i].GroupObjectId;
                ids.Add(id);
                byId[id] = byId.GetValueOrDefault(id) + 1;

                if (id == (uint)i)
                    equalsOwnIndex++;
                if (id < (uint)count)
                    withinRecordCount++;
                if (id > maxValueSeen)
                    maxValueSeen = id;
            }

            distinctRatioSum += (double)ids.Count / count;

            // Dense permutation of 0..N-1?
            bool dense = ids.Count == count;
            if (dense)
            {
                for (uint v = 0; v < count; v++)
                {
                    if (!ids.Contains(v))
                    {
                        dense = false;
                        break;
                    }
                }
            }

            if (dense)
                densePermutationFiles++;

            // Does RefIndex name a node id or an array position?
            for (int i = 0; i < count; i++)
            {
                int refIndex = chunks.Mslk[i].RefIndex;
                if (refIndex < 0)
                    continue;

                refIndexTested++;
                if (ids.Contains((uint)refIndex))
                    refIndexInNodeIdSet++;
                if (refIndex < count)
                    refIndexWithinRecordCount++;
            }

            var membersById = new Dictionary<uint, List<int>>();
            for (int i = 0; i < count; i++)
            {
                uint id = chunks.Mslk[i].GroupObjectId;
                if (!membersById.TryGetValue(id, out List<int>? members))
                {
                    members = [];
                    membersById[id] = members;
                }

                members.Add(i);
            }

            foreach ((uint _, List<int> members) in membersById)
            {
                if (members.Count != 2)
                    continue;

                int a = members[0], b = members[1];
                pairsTested++;
                if (b == a + 1)
                    pairsAdjacent++;
                if (chunks.Mslk[a].RefIndex == chunks.Mslk[b].RefIndex)
                    pairsSameRefIndex++;
                if (chunks.Mslk[a].RefIndex == b || chunks.Mslk[b].RefIndex == a)
                    pairsPointAtEachOther++;
                if (chunks.Mslk[a].TypeFlags == chunks.Mslk[b].TypeFlags)
                    pairsSameTypeFlags++;

                // Control: an arbitrary other record against a, same count of comparisons.
                int c = (a + count / 2) % count;
                if (chunks.Mslk[a].RefIndex == chunks.Mslk[c].RefIndex)
                    controlSameRefIndex++;
            }

            foreach ((uint _, int size) in byId)
            {
                totalGroups++;
                if (size == 1)
                    singletonGroups++;
                groupSizes[size] = groupSizes.GetValueOrDefault(size) + 1;
            }

            if (samples.Count < 5)
            {
                var firstFew = chunks.Mslk.Take(6).Select(static e => e.GroupObjectId.ToString()).ToList();
                samples.Add($"{Path.GetFileNameWithoutExtension(path)}  records={count} distinct={ids.Count} dense={dense}  first={string.Join(",", firstFew)}");
            }
        }

        IReadOnlyList<Pm4ValueFrequency> sizes = groupSizes
            .OrderByDescending(static kv => kv.Value)
            .Take(10)
            .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), (int)kv.Value))
            .ToList();

        return new Pm4MslkNodeIdReport(
            resolved, files, records,
            files == 0 ? 0 : distinctRatioSum / files,
            records == 0 ? 0 : (double)equalsOwnIndex / records,
            files == 0 ? 0 : (double)densePermutationFiles / files,
            records == 0 ? 0 : (double)withinRecordCount / records,
            maxValueSeen,
            refIndexTested,
            refIndexTested == 0 ? 0 : (double)refIndexInNodeIdSet / refIndexTested,
            refIndexTested == 0 ? 0 : (double)refIndexWithinRecordCount / refIndexTested,
            totalGroups == 0 ? 0 : (double)singletonGroups / totalGroups,
            sizes,
            pairsTested,
            pairsTested == 0 ? 0 : (double)pairsAdjacent / pairsTested,
            pairsTested == 0 ? 0 : (double)pairsSameRefIndex / pairsTested,
            pairsTested == 0 ? 0 : (double)pairsPointAtEachOther / pairsTested,
            pairsTested == 0 ? 0 : (double)pairsSameTypeFlags / pairsTested,
            pairsTested == 0 ? 0 : (double)controlSameRefIndex / pairsTested,
            samples);
    }
}

internal sealed record Pm4MslkNodeIdReport(
    string InputDirectory,
    int Files,
    long Records,
    double MeanPerFileDistinctRatio,
    double EqualsOwnIndexFraction,
    double DensePermutationFileFraction,
    double WithinRecordCountFraction,
    long MaxValueSeen,
    long RefIndexTested,
    double RefIndexInNodeIdSetFraction,
    double RefIndexWithinRecordCountFraction,
    double SingletonGroupFraction,
    IReadOnlyList<Pm4ValueFrequency> GroupSizes,
    long PairsTested,
    double PairsAdjacentFraction,
    double PairsSameRefIndexFraction,
    double PairsPointAtEachOtherFraction,
    double PairsSameTypeFlagsFraction,
    double ControlSameRefIndexFraction,
    IReadOnlyList<string> Samples);
