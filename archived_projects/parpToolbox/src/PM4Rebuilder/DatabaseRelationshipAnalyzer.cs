using System;
using System.Data;
using System.IO;
using System.Text;
using Microsoft.Data.Sqlite;

namespace PM4Rebuilder
{
    /// <summary>
    /// Simple ad-hoc analyzer that inspects the SQLite database produced by
    /// <see cref="ParpToolbox.Services.PM4.Database.Pm4DatabaseExporter"/> and writes
    /// CSV and Graphviz DOT files that visualise chunk-field relationships.
    /// </summary>
    internal static class DatabaseRelationshipAnalyzer
    {
        public static int Analyze(string dbPath, string outputDir)
        {
            if (!File.Exists(dbPath))
            {
                Console.WriteLine($"ERROR: Database file '{dbPath}' does not exist.");
                return 1;
            }

            Directory.CreateDirectory(outputDir);

            Console.WriteLine($"[DB-ANALYZE] Opening database: {dbPath}");
            using var conn = new SqliteConnection($"Data Source={dbPath};Mode=ReadOnly");
            conn.Open();

            // 1. Relationship: MPRL.Unknown4 ↔ MSLK.ParentIndex (fan-out via multiple MSLK rows)
            string relCsv = Path.Combine(outputDir, "MprlUnknown4_to_MslkParentIndex.csv");
            string relDot = Path.Combine(outputDir, "MprlUnknown4_to_MslkParentIndex.dot");
            AnalyzeUnknown4ToParentIndex(conn, relCsv, relDot);

            Console.WriteLine("[DB-ANALYZE] Analysis complete. Artifacts written to " + outputDir);
            return 0;
        }

        private static void AnalyzeUnknown4ToParentIndex(SqliteConnection conn, string csvPath, string dotPath)
        {
            // Pull all Unknown4 values from Placements and ParentIndex from Links
            const string sql = @"
                SELECT p.Unknown4 AS ObjectId,
                       l.rowid     AS GroupId
                FROM   Placements p
                JOIN   Links l ON l.ParentIndex = p.Unknown4
                ORDER  BY ObjectId, GroupId
            ";

            using var cmd = new SqliteCommand(sql, conn);
            using var reader = cmd.ExecuteReader();

            var sbCsvPairs = new StringBuilder();
            sbCsvPairs.AppendLine("ObjectId,GroupId");

            var sbCsvSummary = new StringBuilder();
            sbCsvSummary.AppendLine("ObjectId,ChildGroupCount");

            var sbDot = new StringBuilder();
            sbDot.AppendLine("digraph Unknown4ParentIndex {\n  rankdir=LR;");

            int currentObject = -1;
            int groupCounter  = 0;

            void FlushSummary()
            {
                if (currentObject >= 0)
                {
                    sbCsvSummary.AppendLine($"{currentObject},{groupCounter}");
                }
            }

            while (reader.Read())
            {
                int objId   = reader.GetInt32(0);
                int groupId = reader.GetInt32(1);

                if (objId != currentObject)
                {
                    FlushSummary();
                    currentObject = objId;
                    groupCounter  = 0;
                }

                groupCounter++;

                sbCsvPairs.AppendLine($"{objId},{groupId}");
                sbDot.AppendLine($"  \"B_{objId}\" -> \"L_{groupId}\";");
            }
            FlushSummary();

            sbDot.AppendLine("}");

            File.WriteAllText(csvPath, sbCsvPairs.ToString());
            // write summary alongside
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(csvPath)!, "Unknown4_summary.csv"), sbCsvSummary.ToString());
            File.WriteAllText(dotPath, sbDot.ToString());

            // also emit self-contained HTML for browser viewing
            var htmlPath = Path.ChangeExtension(dotPath, ".html");
            var html = $@"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Unknown4→ParentIndex</title>
<script src='https://cdn.jsdelivr.net/npm/viz.js@3.0.0/viz.js' type='application/javascript'></script>
<script src='https://cdn.jsdelivr.net/npm/viz.js@3.0.0/full.render.js' type='application/javascript'></script>
<style>body,html{{margin:0;padding:0;}}</style></head><body>
<div id='graph'></div>
<script>
const src = `{sbDot.ToString().Replace("`", "\u0060").Replace("\\", "\\\\")}`;
const viz = new Viz();
(async () => {{
  const svg = await viz.renderString(src);
  document.getElementById('graph').innerHTML = svg;
}})();
</script></body></html>";
            File.WriteAllText(htmlPath, html);
        }
    }
}
