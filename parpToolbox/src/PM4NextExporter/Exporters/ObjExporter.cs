using System.Collections.Generic;
using System.IO;
using System.Linq;
using System;
using System.Globalization;
using PM4NextExporter.Model;
using PM4NextExporter.Services;

namespace PM4NextExporter.Exporters
{
    public static class ObjExporter
    {
        public static void Export(IEnumerable<AssembledObject> objects, string outDir, bool legacyParity = false)
        {
            // Back-compat wrapper
            Export(objects, outDir, legacyParity, projectLocal: false, alignWithMscn: false);
        }

        public static void Export(IEnumerable<AssembledObject> objects, string outDir, bool legacyParity, bool projectLocal, bool alignWithMscn)
        {
            Directory.CreateDirectory(outDir);
            var objectsDir = Path.Combine(outDir, "objects");
            Directory.CreateDirectory(objectsDir);

            var mirrorX = TransformConfig.ShouldMirrorX(ExporterKind.Object, legacyParity, alignWithMscn);
            var ci = CultureInfo.InvariantCulture;

            var summaryLines = new List<string> {
                "name,vertices,triangles_written,triangles_skipped,filepath"
            };

            foreach (var obj in objects ?? Enumerable.Empty<AssembledObject>())
            {
                var name = string.IsNullOrWhiteSpace(obj.Name) ? "object" : obj.Name;
                var safeName = SanitizeFileName(name);
                var objPath = Path.Combine(objectsDir, safeName + ".obj");

                var verts = obj.Vertices ?? new List<System.Numerics.Vector3>();
                var tris = obj.Triangles ?? new List<(int A, int B, int C)>();

                // Delegate to shared writer preserving TransformConfig precedence
                var centered = alignWithMscn && mirrorX;
                var (written, skipped) = ObjWriterCompat.Write(
                    objPath,
                    verts,
                    tris,
                    legacyParity: false, // flip decision driven solely by TransformConfig
                    projectLocal: projectLocal,
                    forceFlipX: mirrorX,
                    centeredMirrorX: centered);

                summaryLines.Add($"{EscapeCsv(name)},{verts.Count},{written},{skipped},{EscapeCsv(Path.GetRelativePath(outDir, objPath))}");
            }

            // Write summary CSV
            File.WriteAllLines(Path.Combine(outDir, "export_summary.csv"), summaryLines);
        }

        private static string SanitizeFileName(string name)
        {
            var invalid = Path.GetInvalidFileNameChars();
            var s = string.Join("_", name.Split(invalid, StringSplitOptions.RemoveEmptyEntries)).TrimEnd('.');
            return string.IsNullOrWhiteSpace(s) ? "object" : s;
        }

        private static string EscapeCsv(string s)
        {
            if (s.Contains(',') || s.Contains('"') || s.Contains('\n'))
            {
                return '"' + s.Replace("\"", "\"\"") + '"';
            }
            return s;
        }
    }
}
