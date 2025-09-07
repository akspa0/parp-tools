using System;
using System.Collections.Generic;
using System.Linq;

namespace AlphaWdtAnalyzer.Core;

public static class UniqueIdClusterer
{
    public sealed record Cluster(int MinId, int MaxId, int Count);

    public static List<Cluster> FindClusters(IEnumerable<int> ids, int minCount, int maxGap)
    {
        var sorted = ids.Distinct().OrderBy(x => x).ToList();
        var clusters = new List<Cluster>();
        if (sorted.Count == 0) return clusters;

        int start = sorted[0];
        int prev = start;
        int count = 1;

        for (int i = 1; i < sorted.Count; i++)
        {
            var cur = sorted[i];
            if (cur - prev <= maxGap)
            {
                count++;
            }
            else
            {
                if (count >= minCount)
                {
                    clusters.Add(new Cluster(start, prev, count));
                }
                start = cur;
                count = 1;
            }
            prev = cur;
        }

        if (count >= minCount)
        {
            clusters.Add(new Cluster(start, prev, count));
        }

        return clusters;
    }
}
