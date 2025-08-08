using System.IO;

namespace PM4NextExporter.Services
{
    public static class DiagnosticsService
    {
        public static void WriteSnapshotCsv(string outDir, string name)
        {
            // TODO: real diagnostics (CSV snapshots, histograms, correlations)
            var path = Path.Combine(outDir, $"{name}.csv");
            File.WriteAllText(path, "metric,value\nstub,1\n");
        }
    }
}
