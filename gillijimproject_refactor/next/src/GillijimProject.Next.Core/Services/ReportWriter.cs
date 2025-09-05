using System.Text;
using System.IO;
using GillijimProject.Next.Core.Domain;

namespace GillijimProject.Next.Core.Services;

/// <summary>
/// Writes Markdown reports for analysis outputs.
/// </summary>
public static class ReportWriter
{
    public static string WriteMarkdown(UniqueIdReport report, string outputDir)
    {
        Directory.CreateDirectory(outputDir);
        var path = Path.Combine(outputDir, "uniqueid_report.md");
        var sb = new StringBuilder();
        sb.AppendLine("# UniqueID Report");
        sb.AppendLine($"Total Entries: {report.TotalEntries}");
        sb.AppendLine($"Missing Assets: {report.MissingAssets}");
        sb.AppendLine($"Duplicate IDs: {report.DuplicateIds}");
        sb.AppendLine();
        sb.AppendLine(report.Notes);
        File.WriteAllText(path, sb.ToString());
        return path;
    }
}
