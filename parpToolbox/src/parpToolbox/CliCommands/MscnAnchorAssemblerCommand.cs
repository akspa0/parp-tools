using System.Globalization;
using System.Linq;
using System.Text;
using ParpToolbox.Utils;

namespace ParpToolbox.CliCommands
{
    /// <summary>
    /// CLI command: mscn-anchor <inputDir> [--radius <r>] [--flip-x] [--export-objs]
    /// Auto-exports PM4→OBJ if only .pm4 files are present, then groups PM4 OBJs around MSCN anchors.
    /// This trimmed implementation avoids System.CommandLine dependencies for rapid build-out.
    /// </summary>
    internal static class MscnAnchorAssemblerCommand
    {
        private const string MscnPattern = "*.mscn.obj";
        private const string Pm4Pattern = "*.pm4.obj";

        public static async Task<int> Run(string[] args)
        {
            if (args.Length < 2)
            {
                ConsoleLogger.WriteLine("Usage: mscn-anchor <inputDir> [--radius <r>] [--flip-x] [--export-objs]");
                return 1;
            }

            string root = args[1];
            double radius = 120.0;
            bool flipX = false;
            bool exportObjs = false;

            for (int i = 2; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--radius":
                        if (i + 1 < args.Length && double.TryParse(args[i + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out var r))
                        { radius = r; i++; }
                        else { ConsoleLogger.WriteLine("Invalid --radius value"); return 1; }
                        break;
                    case "--flip-x":
                        flipX = true;
                        break;
                    case "--export-objs":
                        exportObjs = true;
                        break;
                    default:
                        ConsoleLogger.WriteLine($"Unknown flag {args[i]}");
                        return 1;
                }
            }

            var dirInfo = new DirectoryInfo(root);

            // Look for ready-made OBJ exports first
            var mscnFiles = dirInfo.GetFiles(MscnPattern, SearchOption.TopDirectoryOnly).OrderBy(f => f.Name).ToList();
            var pm4ObjFiles = dirInfo.GetFiles(Pm4Pattern, SearchOption.TopDirectoryOnly).OrderBy(f => f.Name).ToList();

            // If OBJs are absent but PM4 files are present, perform export first
            if (pm4ObjFiles.Count == 0)
            {
                var pm4Files = dirInfo.GetFiles("*.pm4", SearchOption.TopDirectoryOnly).ToList();
                if (pm4Files.Count > 0)
                {
                    ConsoleLogger.WriteLine("No MSCN anchor files found; extracting MSCN anchors from PM4 files...");
                    
                    var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"), "mscn_anchors");
                    Directory.CreateDirectory(outputDir);
                    
                    foreach (var pm4File in pm4Files)
                    {
                        try
                        {
                            // Load PM4 file to extract MSCN anchor points only
                            var pm4Adapter = new Services.PM4.Pm4Adapter();
                            var scene = pm4Adapter.Load(pm4File.FullName);
                            
                            if (scene != null)
                            {
                                // Extract MSCN anchor points only (not full building data)
                                var mscnChunk = scene.ExtraChunks?.OfType<ParpToolbox.Formats.P4.Chunks.Common.MscnChunk>().FirstOrDefault();
                                if (mscnChunk != null && mscnChunk.Vertices.Count > 0)
                                {
                                    var mscnObjPath = Path.Combine(outputDir, $"{Path.GetFileNameWithoutExtension(pm4File.Name)}.mscn.obj");
                                    ExportMscnAnchorsToObj(mscnChunk, mscnObjPath, flipX);
                                    ConsoleLogger.WriteLine($"  → Extracted {mscnChunk.Vertices.Count} MSCN anchors from {pm4File.Name}");
                                }
                                else
                                {
                                    ConsoleLogger.WriteLine($"  → No MSCN data found in {pm4File.Name}");
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            ConsoleLogger.WriteLine($"  → Error extracting MSCN from {pm4File.Name}: {ex.Message}");
                        }
                    }
                    
                    // Refresh file lists after export
                    mscnFiles = dirInfo.GetFiles(MscnPattern, SearchOption.AllDirectories).OrderBy(f => f.Name).ToList();
                    pm4ObjFiles = dirInfo.GetFiles(Pm4Pattern, SearchOption.AllDirectories).OrderBy(f => f.Name).ToList();
                    
                    if (mscnFiles.Count == 0 && pm4ObjFiles.Count == 0)
                    {
                        ConsoleLogger.WriteLine("Export completed but no OBJ files found. Check export directory.");
                        return 1;
                    }
                }
            }

            if (mscnFiles.Count == 0)
            {
                ConsoleLogger.WriteLine("No MSCN files found.");
                return 1;
            }

            return await Process(mscnFiles, pm4ObjFiles, radius, flipX, exportObjs);
        }

        private static async Task<int> Process(List<FileInfo> mscnFiles, List<FileInfo> pm4ObjFiles, double radius, bool flipX, bool exportObjs)
        {
            var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "project_output", DateTime.Now.ToString("yyyyMMdd_HHmmss"), "mscn_anchor_grouping");
            Directory.CreateDirectory(outputDir);

            var anchors = new List<(FileInfo File, (double X, double Y, double Z) Centroid)>();
            foreach (var mfile in mscnFiles)
            {
                var mCentroid = Centroid(LoadVertices(mfile.FullName, flipX));
                anchors.Add((mfile, mCentroid));
            }

            var attached = new List<(FileInfo Pm4File, FileInfo MscnFile, double Distance)>();
            foreach (var anchor in anchors)
            {
                var nearby = pm4ObjFiles
                    .Select(p => (File: p, Centroid: Centroid(LoadVertices(p.FullName, flipX))))
                    .Where(p => Distance(p.Centroid, anchor.Centroid) <= radius)
                    .OrderBy(p => Distance(p.Centroid, anchor.Centroid))
                    .ToList();

                foreach (var near in nearby)
                {
                    attached.Add((near.File, anchor.File, Distance(near.Centroid, anchor.Centroid)));
                }
            }

            var mapping = new StringBuilder();
            mapping.AppendLine("MSCN_File,PM4_File,Distance");
            foreach (var (pm4, mscn, dist) in attached.OrderBy(x => x.Distance))
            {
                mapping.AppendLine($"{mscn.Name},{pm4.Name},{dist:F3}");
            }

            var mappingFile = Path.Combine(outputDir, "mscn_anchor_mapping.csv");
            await File.WriteAllTextAsync(mappingFile, mapping.ToString());
            ConsoleLogger.WriteLine($"Mapping written to {mappingFile}");

            if (exportObjs)
            {
                await WriteMergedObj(attached, outputDir, flipX);
            }

            return 0;
        }

        private static List<(double X, double Y, double Z)> LoadVertices(string objPath, bool flipX)
        {
            var vertices = new List<(double X, double Y, double Z)>();
            foreach (var line in File.ReadLines(objPath))
            {
                if (line.StartsWith("v "))
                {
                    var parts = line.Substring(2).Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 3 &&
                        double.TryParse(parts[0], NumberStyles.Float, CultureInfo.InvariantCulture, out var x) &&
                        double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var y) &&
                        double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var z))
                    {
                        vertices.Add((flipX ? -x : x, y, z));
                    }
                }
            }
            return vertices;
        }

        private static (double X, double Y, double Z) Centroid(List<(double X, double Y, double Z)> verts)
        {
            if (verts.Count == 0) return (0, 0, 0);
            double sx = 0, sy = 0, sz = 0;
            foreach (var (x, y, z) in verts)
            {
                sx += x; sy += y; sz += z;
            }
            return (sx / verts.Count, sy / verts.Count, sz / verts.Count);
        }

        private static double Distance((double X, double Y, double Z) a, (double X, double Y, double Z) b)
        {
            var dx = a.X - b.X;
            var dy = a.Y - b.Y;
            var dz = a.Z - b.Z;
            return Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }

        private static async Task WriteMergedObj(List<(FileInfo Pm4File, FileInfo MscnFile, double Distance)> attached, string outputDir, bool flipX)
        {
            var grouped = attached.GroupBy(a => a.MscnFile.Name).OrderBy(g => g.Key);
            foreach (var group in grouped)
            {
                var merged = new StringBuilder();
                var vertexOffset = 1;
                foreach (var (pm4, mscn, _) in group.OrderBy(x => x.Distance))
                {
                    var vertices = LoadVertices(pm4.FullName, flipX);
                    foreach (var v in vertices)
                    {
                        merged.AppendLine($"v {v.X:F6} {v.Y:F6} {v.Z:F6}");
                    }

                    var faces = new List<string>();
                    foreach (var line in File.ReadLines(pm4.FullName))
                    {
                        if (line.StartsWith("f "))
                        {
                            var parts = line.Substring(2).Split(' ', StringSplitOptions.RemoveEmptyEntries);
                            var sb = new StringBuilder("f");
                            foreach (var part in parts)
                            {
                                if (int.TryParse(part, out var idx))
                                {
                                    sb.Append($" {idx + vertexOffset}");
                                }
                            }
                            faces.Add(sb.ToString());
                        }
                    }
                    faces.ForEach(f => merged.AppendLine(f));
                    vertexOffset += vertices.Count;
                }

                var outPath = Path.Combine(outputDir, $"{group.Key}_merged.obj");
                await File.WriteAllTextAsync(outPath, merged.ToString());
                ConsoleLogger.WriteLine($"Merged OBJ written to {outPath}");
            }
        }
        
        /// <summary>
        /// Export MSCN anchor points as OBJ vertices (no faces - just anchor points)
        /// </summary>
        private static void ExportMscnAnchorsToObj(ParpToolbox.Formats.P4.Chunks.Common.MscnChunk mscnChunk, string objPath, bool flipX)
        {
            using var writer = new StreamWriter(objPath);
            
            writer.WriteLine("# MSCN Anchor Points");
            writer.WriteLine($"# {mscnChunk.Vertices.Count} anchor vertices");
            writer.WriteLine();
            
            foreach (var vertex in mscnChunk.Vertices)
            {
                // Apply coordinate transformation: MSCN uses Y,X,Z ordering
                // Flip X-axis if requested (common for WoW coordinate system)
                float x = flipX ? -vertex.Y : vertex.Y;
                float y = vertex.X;
                float z = vertex.Z;
                
                writer.WriteLine($"v {x:F6} {y:F6} {z:F6}");
            }
            
            writer.WriteLine();
            writer.WriteLine("# No faces - anchor points only");
        }
    }
}
