using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Tests what tile address, if any, <c>MSLK.LinkId</c> carries, and in which order.
/// </summary>
/// <remarks>
/// The open question is only the byte ORDER - whether the packed pair reads as the filename's numbers
/// in the order they appear or reversed. Those two hypotheses are indistinguishable on any tile whose
/// two numbers are equal, so diagonal tiles are excluded from scoring and counted separately; leaving
/// them in inflates both candidates by the same amount and makes a tie look like agreement.
///
/// <para>A control is included: the same two decodes scored against a DIFFERENT file's tile numbers.
/// A field that encodes nothing spatial still matches its own tile at whatever rate the value
/// distribution allows, so the real hypothesis has to beat the mismatched pairing, not zero.</para>
/// </remarks>
internal static class Pm4LinkIdOrderSupport
{
    public static Pm4LinkIdOrderReport Analyze(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        long total = 0, diagonalSkipped = 0;
        long firstSecond = 0, secondFirst = 0, controlFirstSecond = 0, controlSecondFirst = 0;
        long low16Zero = 0;
        var files = new List<(int First, int Second, List<uint> Ids)>();

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (!Pm4CoordinateService.TryParseTileCoordinates(path, out int first, out int second))
                continue;

            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (chunks.Mslk.Count == 0)
                continue;

            files.Add((first, second, chunks.Mslk.Select(static e => e.LinkId).ToList()));
        }

        for (int i = 0; i < files.Count; i++)
        {
            (int first, int second, List<uint> ids) = files[i];

            // Control pairing: the next file's tile numbers, which have nothing to do with these ids.
            (int cf, int cs) = files[(i + 1) % files.Count] is var next ? (next.First, next.Second) : (0, 0);

            foreach (uint id in ids)
            {
                uint low = id & 0xFFFFu;
                if (low == 0)
                    low16Zero++;

                byte hi = (byte)((low >> 8) & 0xFF);
                byte lo = (byte)(low & 0xFF);

                if (first == second)
                {
                    diagonalSkipped++;
                    continue;
                }

                total++;
                if (hi == first && lo == second) firstSecond++;
                if (hi == second && lo == first) secondFirst++;
                if (hi == cf && lo == cs) controlFirstSecond++;
                if (hi == cs && lo == cf) controlSecondFirst++;
            }
        }

        return new Pm4LinkIdOrderReport(
            resolved, files.Count, total, diagonalSkipped, low16Zero,
            total == 0 ? 0 : (double)firstSecond / total,
            total == 0 ? 0 : (double)secondFirst / total,
            total == 0 ? 0 : (double)controlFirstSecond / total,
            total == 0 ? 0 : (double)controlSecondFirst / total);
    }
}

internal sealed record Pm4LinkIdOrderReport(
    string InputDirectory,
    int Files,
    long LinksScored,
    long DiagonalLinksSkipped,
    long Low16ZeroCount,
    double FirstSecondFraction,
    double SecondFirstFraction,
    double ControlFirstSecondFraction,
    double ControlSecondFirstFraction);
