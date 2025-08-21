using System.IO;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.PM4;

namespace WoWToolbox.Core.v2.Utilities
{
    /// <summary>
    /// Quick-and-dirty raw dump of every primitive field we can access for exploratory analysis.
    /// Currently only handles the MSHD chunk (single-structure dump). Extend as needed.
    /// </summary>
    public static class RawDumpHelper
    {
        public static void DumpAll(PM4File file, string outputDir)
        {
            if (!Directory.Exists(outputDir))
                Directory.CreateDirectory(outputDir);

            DumpMshd(file, outputDir);
            DumpMslk(file, outputDir);
            DumpMsur(file, outputDir);
            DumpMprl(file, outputDir);
            DumpMprr(file, outputDir);
            DumpMspv(file, outputDir);
            DumpMsrn(file, outputDir);
            DumpMsvi(file, outputDir);
            DumpMsvt(file, outputDir);
            DumpMdos(file, outputDir);
            DumpMscn(file, outputDir);
        }

        private static void DumpMshd(PM4File file, string outputDir)
        {
            if (file.MSHD == null) return;
            var row = new
            {
                file.MSHD.Unknown_0x00,
                file.MSHD.Unknown_0x04,
                file.MSHD.Unknown_0x08,
                file.MSHD.Unknown_0x0C,
                file.MSHD.Unknown_0x10,
                file.MSHD.Unknown_0x14,
                file.MSHD.Unknown_0x18,
                file.MSHD.Unknown_0x1C
            };
            var path = Path.Combine(outputDir, "mshd.csv");
            CsvDumpWriter.WriteDump(new[] { row }, path);
                }

        private static void DumpMslk(PM4File file, string dir)
        {
            if (file.MSLK?.Entries == null || file.MSLK.Entries.Count == 0) return;
            var path = Path.Combine(dir, "mslk.csv");
            CsvDumpWriter.WriteDump(file.MSLK.Entries, path);
        }

        private static void DumpMsur(PM4File file, string dir)
        {
            if (file.MSUR?.Entries == null || file.MSUR.Entries.Count == 0) return;
            var path = Path.Combine(dir, "msur.csv");
            CsvDumpWriter.WriteDump(file.MSUR.Entries, path);
        }

        private static void DumpMprl(PM4File file, string dir)
        {
            if (file.MPRL?.Entries == null || file.MPRL.Entries.Count == 0) return;
            var path = Path.Combine(dir, "mprl.csv");
            CsvDumpWriter.WriteDump(file.MPRL.Entries, path);
        }

        private static void DumpMprr(PM4File file, string dir)
        {
            if (file.MPRR?.Sequences == null || file.MPRR.Sequences.Count == 0) return;
            var rows = new System.Collections.Generic.List<dynamic>();
            for (int seqIdx = 0; seqIdx < file.MPRR.Sequences.Count; seqIdx++)
            {
                var seq = file.MPRR.Sequences[seqIdx];
                for (int valIdx = 0; valIdx < seq.Count; valIdx++)
                {
                    rows.Add(new { Sequence = seqIdx, Index = valIdx, Value = seq[valIdx] });
                }
            }
            var path = Path.Combine(dir, "mprr.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }

        private static void DumpMspv(PM4File file, string dir)
        {
            if (file.MSPV?.Vertices == null || file.MSPV.Vertices.Count == 0) return;
            var rows = file.MSPV.Vertices.Select((v, i) => new { Index = i, v.X, v.Y, v.Z }).ToList();
            var path = Path.Combine(dir, "mspv.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }

        private static void DumpMsvi(PM4File file, string dir)
        {
            if (file.MSVI?.Indices == null || file.MSVI.Indices.Count == 0) return;
            var rows = new System.Collections.Generic.List<dynamic>();
            for (int i = 0; i + 2 < file.MSVI.Indices.Count; i += 3)
            {
                rows.Add(new { Triangle = i / 3, V0 = file.MSVI.Indices[i], V1 = file.MSVI.Indices[i + 1], V2 = file.MSVI.Indices[i + 2] });
            }
            var path = Path.Combine(dir, "msvi.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }

        private static void DumpMdos(PM4File file, string dir)
        {
            if (file.MDOS?.Entries == null || file.MDOS.Entries.Count == 0) return;
            var path = Path.Combine(dir, "mdos.csv");
            CsvDumpWriter.WriteDump(file.MDOS.Entries, path);
        }

        private static void DumpMsrn(PM4File file, string dir)
        {
            if (file.MSRN?.Normals == null || file.MSRN.Normals.Count == 0) return;
            var rows = file.MSRN.Normals.Select((n, i) => new
            {
                Index = i,
                n.X,
                n.Y,
                n.Z,
                NormX = n.X / 8192f,
                NormY = n.Y / 8192f,
                NormZ = n.Z / 8192f
            }).ToList();
            var path = Path.Combine(dir, "msrn.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }

        private static void DumpMsvt(PM4File file, string dir)
        {
            if (file.MSVT?.Vertices == null || file.MSVT.Vertices.Count == 0) return;
            var rows = file.MSVT.Vertices.Select((v, i) => new { Index = i, v.X, v.Y, v.Z }).ToList();
            var path = Path.Combine(dir, "msvt.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }

        private static void DumpMscn(PM4File file, string dir)
        {
            if (file.MSCN?.ExteriorVertices == null || file.MSCN.ExteriorVertices.Count == 0) return;
            var rows = file.MSCN.ExteriorVertices.Select((v, i) => new { Index = i, v.X, v.Y, v.Z }).ToList();
            var path = Path.Combine(dir, "mscn.csv");
            CsvDumpWriter.WriteDump(rows, path);
        }
    }
}
