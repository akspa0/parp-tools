using System;
using System.IO;
using System.Linq;
using System.Xml.Linq;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.ProjectAudit
{
    /// <summary>
    /// Generates a CSV report of every *.csproj in the repository (excluding legacy WCAnalyzer proof-of-concept projects)
    /// and asserts that at least one active test project references WoWToolbox.Core.v2.
    /// </summary>
    public class ProjectAuditTests
    {
        [Fact(DisplayName = "Generate project audit CSV and basic assertions")]
        public void GenerateProjectAuditCsv()
        {
            // Determine repo root relative to the compiled test assembly (…/PM4Tool/bin/<cfg>/<tfm>/)
            var repoRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));

            // Enumerate all csproj files, skipping deprecated WCAnalyzer buckets
            var projects = Directory.EnumerateFiles(repoRoot, "*.csproj", SearchOption.AllDirectories)
                                    .Where(p => !p.Contains("WCAnalyzer", StringComparison.OrdinalIgnoreCase))
                                    .ToList();

            var rows = projects.Select(p =>
            {
                var doc = XDocument.Load(p);
                XNamespace ns = doc.Root?.Name.Namespace ?? XNamespace.None;

                string tfm = doc.Descendants(ns + "TargetFramework").FirstOrDefault()?.Value ?? string.Empty;
                bool isTest = doc.Descendants(ns + "PackageReference")
                                 .Any(pr => (pr.Attribute("Include")?.Value ?? string.Empty)
                                            .Contains("Microsoft.NET.Test.Sdk", StringComparison.OrdinalIgnoreCase));
                var refs = doc.Descendants(ns + "ProjectReference")
                              .Select(r => r.Attribute("Include")?.Value)
                              .Where(v => !string.IsNullOrWhiteSpace(v))
                              .ToArray();

                return new AuditRow
                {
                    Path = p,
                    TargetFramework = tfm,
                    IsTest = isTest,
                    ProjectReferences = string.Join(';', refs)
                };
            }).ToList();

            // Ensure project_output directory exists
            var outputDir = Path.Combine(repoRoot, "project_output");
            Directory.CreateDirectory(outputDir);
            var csvPath = Path.Combine(outputDir, "test_audit.csv");

            // Write CSV
            using (var writer = new StreamWriter(csvPath, false))
            {
                writer.WriteLine("Path,TargetFramework,IsTest,ProjectReferences");
                foreach (var r in rows)
                {
                    writer.WriteLine($"\"{r.Path}\",\"{r.TargetFramework}\",{r.IsTest},\"{r.ProjectReferences}\"");
                }
            }

            // Basic sanity assertion: we should have at least one test project referencing Core.v2
            Assert.Contains(rows, r => r.IsTest && r.ProjectReferences.Contains("WoWToolbox.Core.v2", StringComparison.OrdinalIgnoreCase));
        }

        private class AuditRow
        {
            public string Path { get; set; } = string.Empty;
            public string TargetFramework { get; set; } = string.Empty;
            public bool IsTest { get; set; }
            public string ProjectReferences { get; set; } = string.Empty;
        }
    }
}
