using System.Globalization;
using System.Text;

namespace WowViewer.Tools.Shared.Pm4Matching;

public static class Pm4MatchReportFormatter
{
    public static string ToMarkdown(Pm4MatchRunManifest manifest)
    {
        var sb = new StringBuilder();
        string tileLabel = string.Join(", ", manifest.Segments
            .SelectMany(static s => s.TileCoordinates)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(static t => t));

        sb.AppendLine("# PM4 Asset Match Report");
        sb.AppendLine();
        sb.AppendLine($"**Run**: `{manifest.RunId}`");
        sb.AppendLine($"**Input**: `{manifest.InputPm4Root}`");
        sb.AppendLine($"**Tiles**: {tileLabel}");
        sb.AppendLine($"**Total segments**: {manifest.SegmentCount}");
        sb.AppendLine();
        sb.AppendLine("## Summary");
        sb.AppendLine();

        int matched = manifest.MatchedCount ?? 0;
        int ambiguous = manifest.AmbiguousCount ?? 0;
        int unresolved = manifest.UnresolvedCount ?? 0;
        int ineligible = manifest.IneligibleCount ?? 0;
        int proposalCount = manifest.Segments.Count(static s => s.PlacementProposal is not null);

        sb.AppendLine("| Status      | Count     |");
        sb.AppendLine("|-------------|-----------|");
        sb.AppendLine($"| Matched     | {matched,8} |");
        sb.AppendLine($"| Ambiguous   | {ambiguous,8} |");
        sb.AppendLine($"| Unresolved  | {unresolved,8} |");
        sb.AppendLine($"| Ineligible  | {ineligible,8} |");
        sb.AppendLine($"| Proposals   | {proposalCount,8} |");
        sb.AppendLine();

        var matchedSegments = manifest.Segments
            .Where(static s => string.Equals(s.Status, "matched", StringComparison.Ordinal))
            .OrderByDescending(static s => s.Candidates is { Count: > 0 } ? s.Candidates[0].OverallScore : 0d)
            .Take(15)
            .ToList();

        if (matchedSegments.Count > 0)
        {
            sb.AppendLine("## Top Matched Segments");
            sb.AppendLine();
            foreach (Pm4MatchReportSegment seg in matchedSegments)
            {
                Pm4MatchReportCandidate? top = seg.Candidates is { Count: > 0 } ? seg.Candidates[0] : null;
                Pm4MatchReportPlacementProposal? prop = seg.PlacementProposal;

                sb.AppendLine($"### Segment `{seg.SegmentId}`");
                sb.AppendLine($"- **CK24**: `{seg.Ck24}` (type=0x{seg.Ck24Type:X2}, objectId={seg.Ck24ObjectId})");
                sb.AppendLine($"- **Tiles**: {string.Join(", ", seg.TileCoordinates)}");
                sb.AppendLine($"- **Expected kind**: {seg.ExpectedAssetKind ?? "n/a"}");
                sb.AppendLine($"- **Footprint area**: {seg.FootprintArea?.ToString("F1", CultureInfo.InvariantCulture) ?? "n/a"}");
                sb.AppendLine($"- **Center**: ({FormatVector3(seg)})");

                if (top is not null)
                {
                    string scoreStr = top.OverallScore.ToString("F4", CultureInfo.InvariantCulture);
                    sb.AppendLine($"- **Top Candidate** (rank={top.Rank}, score={scoreStr})");
                    sb.AppendLine($"  - **Asset**: `{top.AssetPath}`");
                    sb.AppendLine($"  - **Kind**: {top.AssetKind}");
                    if (top.ScoreBreakdown is { Count: > 0 })
                    {
                        sb.AppendLine($"  - **Breakdown**:");
                        foreach (KeyValuePair<string, double> kv in top.ScoreBreakdown)
                            sb.AppendLine($"    - {kv.Key}: {kv.Value:F4}");
                    }
                }

                if (prop is not null)
                {
                    sb.AppendLine($"- **Placement Proposal**: `{prop.ProposalId}`");
                    sb.AppendLine($"  - **Position**: ({prop.WorldPosition?.X:F2}, {prop.WorldPosition?.Y:F2}, {prop.WorldPosition?.Z:F2})");
                    sb.AppendLine($"  - **Rotation yaw**: {prop.WorldRotation?.Yaw?.ToString("F1", CultureInfo.InvariantCulture) ?? "n/a"}°");
                    sb.AppendLine($"  - **Scale**: {prop.WorldScale?.ToString("F2", CultureInfo.InvariantCulture) ?? "n/a"}");
                    sb.AppendLine($"  - **Confidence**: {prop.Confidence:F4}");
                    sb.AppendLine($"  - **Review required**: {prop.ReviewRequired}");
                    if (prop.Provenance is { Count: > 0 })
                        sb.AppendLine($"  - **Provenance**: {string.Join("; ", prop.Provenance)}");
                }

                sb.AppendLine();
            }
        }

        var ambiguousSegments = manifest.Segments
            .Where(static s => string.Equals(s.Status, "ambiguous", StringComparison.Ordinal))
            .OrderByDescending(static s => s.Candidates is { Count: > 0 } ? s.Candidates[0].OverallScore : 0d)
            .Take(10)
            .ToList();

        if (ambiguousSegments.Count > 0)
        {
            sb.AppendLine("## Ambiguous Segments (top by score)");
            sb.AppendLine();
            foreach (Pm4MatchReportSegment seg in ambiguousSegments)
            {
                Pm4MatchReportCandidate? first = seg.Candidates is { Count: > 0 } ? seg.Candidates[0] : null;
                Pm4MatchReportCandidate? second = seg.Candidates is { Count: > 1 } ? seg.Candidates[1] : null;

                sb.AppendLine($"### Segment `{seg.SegmentId}`");
                sb.AppendLine($"- **CK24**: `{seg.Ck24}` kind={seg.ExpectedAssetKind ?? "n/a"}");
                sb.AppendLine($"- **Tiles**: {string.Join(", ", seg.TileCoordinates)}");

                if (first is not null && second is not null)
                {
                    sb.AppendLine($"- **rank=1**: `{first.AssetPath}` score={first.OverallScore:F4} ({first.AssetKind})");
                    sb.AppendLine($"- **rank=2**: `{second.AssetPath}` score={second.OverallScore:F4} ({second.AssetKind})");
                    sb.AppendLine($"- **gap**: {(first.OverallScore - second.OverallScore).ToString("F4", CultureInfo.InvariantCulture)}");
                }

                if (seg.PlacementProposal is not null)
                    sb.AppendLine($"- **proposal**: pos=({seg.PlacementProposal.WorldPosition?.X:F1},{seg.PlacementProposal.WorldPosition?.Y:F1},{seg.PlacementProposal.WorldPosition?.Z:F1}) review={seg.PlacementProposal.ReviewRequired}");

                sb.AppendLine();
            }
        }

        if (manifest.Warnings is { Count: > 0 })
        {
            sb.AppendLine("## Warnings");
            sb.AppendLine();
            foreach (string warning in manifest.Warnings)
                sb.AppendLine($"- {warning}");
            sb.AppendLine();
        }

        sb.AppendLine("---");
        sb.AppendLine($"*Generated by WowViewer.Tool.Inspect — {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC*");

        return sb.ToString();
    }

    public static void WriteMarkdownToFile(Pm4MatchRunManifest manifest, string outputPath)
    {
        string? directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(outputPath, ToMarkdown(manifest));
    }

    private static string FormatVector3(Pm4MatchReportSegment seg)
    {
        if (seg.Center is not null)
            return $"({seg.Center.X:F2}, {seg.Center.Y:F2}, {seg.Center.Z:F2})";
        if (seg.Bounds is not null)
            return $"center of ({seg.Bounds.Min.X:F1}, {seg.Bounds.Min.Y:F1}, {seg.Bounds.Min.Z:F1})..({seg.Bounds.Max.X:F1}, {seg.Bounds.Max.Y:F1}, {seg.Bounds.Max.Z:F1})";
        return "n/a";
    }
}
