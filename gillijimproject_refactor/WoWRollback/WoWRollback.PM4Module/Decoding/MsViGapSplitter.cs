using WoWRollback.PM4Module.Pipeline;

namespace WoWRollback.PM4Module.Decoding;

/// <summary>
/// Splits aggregated PM4 surfaces (grouped by CK24) into individual object instances
/// using MSVI index gaps as the primary separator.
/// </summary>
public static class MsViGapSplitter
{
    /// <summary>
    /// Splits a list of surfaces into separate object instances based on MSVI index gaps.
    /// Large gaps in the MSVI index sequence indicate separate object instances.
    /// </summary>
    /// <param name="surfaces">Surfaces to split (all from same CK24)</param>
    /// <param name="gapThreshold">Minimum gap size to trigger a split (default: 50)</param>
    /// <returns>List of surface groups, each representing one object instance</returns>
    public static List<List<MsurChunk>> SplitByMsviGaps(List<MsurChunk> surfaces, int gapThreshold = 50)
    {
        if (surfaces == null || surfaces.Count == 0)
            return new List<List<MsurChunk>>();

        // Sort surfaces by MsviFirstIndex
        var sorted = surfaces.OrderBy(s => s.MsviFirstIndex).ToList();
        
        var instances = new List<List<MsurChunk>>();
        var currentInstance = new List<MsurChunk> { sorted[0] };
        
        for (int i = 1; i < sorted.Count; i++)
        {
            var prev = sorted[i - 1];
            var curr = sorted[i];
            
            // Calculate the gap between the end of the previous surface and the start of the current
            uint prevEnd = prev.MsviFirstIndex + (uint)prev.IndexCount;
            uint currStart = curr.MsviFirstIndex;
            
            int gap = (int)(currStart - prevEnd);
            
            if (gap > gapThreshold)
            {
                // Large gap detected - start a new instance
                instances.Add(currentInstance);
                currentInstance = new List<MsurChunk> { curr };
            }
            else
            {
                // Contiguous or small gap - same instance
                currentInstance.Add(curr);
            }
        }
        
        // Add the final instance
        if (currentInstance.Count > 0)
            instances.Add(currentInstance);
        
        return instances;
    }
    
    /// <summary>
    /// Analyzes MSVI gaps for a CK24 group and returns statistics.
    /// Useful for debugging and threshold tuning.
    /// </summary>
    public static GapAnalysis AnalyzeGaps(List<MsurChunk> surfaces)
    {
        if (surfaces == null || surfaces.Count < 2)
            return new GapAnalysis(0, 0, 0, new List<int>());

        var sorted = surfaces.OrderBy(s => s.MsviFirstIndex).ToList();
        var gaps = new List<int>();
        
        for (int i = 1; i < sorted.Count; i++)
        {
            var prev = sorted[i - 1];
            var curr = sorted[i];
            
            uint prevEnd = prev.MsviFirstIndex + (uint)prev.IndexCount;
            uint currStart = curr.MsviFirstIndex;
            
            int gap = (int)(currStart - prevEnd);
            gaps.Add(gap);
        }
        
        int maxGap = gaps.Any() ? gaps.Max() : 0;
        int largeGapCount = gaps.Count(g => g > 50);
        double avgGap = gaps.Any() ? gaps.Average() : 0;
        
        return new GapAnalysis(maxGap, largeGapCount, avgGap, gaps);
    }
    
    public record GapAnalysis(int MaxGap, int LargeGapCount, double AverageGap, List<int> AllGaps);
}
