using System;
using System.IO;
using Xunit;
using WoWToolbox.Core.WMO;

namespace WoWToolbox.Tests.WMO
{
    public class WmoBatchObjExportTests
    {
        [Fact]
        public void ExportAllWmoFilesToObj_CacheToOutputFolder()
        {
            // Set input and output roots
            string inputRoot = Path.GetFullPath("test_data/335_wmo");
            string outputRoot = Path.GetFullPath("output/wmo");
            Directory.CreateDirectory(outputRoot);

            var wmoFiles = Directory.GetFiles(inputRoot, "*.wmo", SearchOption.AllDirectories);
            int exported = 0, skipped = 0;
            foreach (var wmoPath in wmoFiles)
            {
                // Compute relative path and output OBJ path
                string relPath = Path.GetRelativePath(inputRoot, wmoPath);
                string outDir = Path.Combine(outputRoot, Path.GetDirectoryName(relPath) ?? "");
                Directory.CreateDirectory(outDir);
                string objPath = Path.Combine(outDir, Path.GetFileNameWithoutExtension(wmoPath) + ".obj");
                if (File.Exists(objPath))
                {
                    skipped++;
                    continue;
                }
                try
                {
                    Console.WriteLine($"[EXPORT] {wmoPath} -> {objPath}");
                    var mergedMesh = WoWToolbox.Core.WMO.WmoMeshExporter.LoadMergedWmoMesh(wmoPath);
                    if (mergedMesh == null)
                    {
                        Console.WriteLine($"[WARN] No valid group meshes found in {wmoPath}, skipping OBJ export.");
                        continue;
                    }
                    WoWToolbox.Core.WMO.WmoMeshExporter.SaveMergedWmoToObj(mergedMesh, objPath);
                    exported++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[WARN] Failed to export {wmoPath}: {ex.Message}");
                }
            }
            Console.WriteLine($"[SUMMARY] Exported {exported} WMO OBJs, skipped {skipped} (already exist)");
        }

        private void ListMomoSubChunks(string v14WmoPath, string logPath)
        {
            using var log = new StreamWriter(logPath, false);
            using var stream = File.OpenRead(v14WmoPath);
            using var reader = new BinaryReader(stream, System.Text.Encoding.UTF8, leaveOpen: true);
            long fileLen = stream.Length;
            // Find MOMO
            stream.Position = 0;
            long momoStart = -1, momoEnd = -1;
            while (stream.Position + 8 <= fileLen)
            {
                long chunkStart = stream.Position;
                var chunkIdBytes = reader.ReadBytes(4);
                if (chunkIdBytes.Length < 4) break;
                string chunkIdStr = new string(chunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint chunkSize = reader.ReadUInt32();
                long chunkDataPos = stream.Position;
                long chunkEnd = chunkDataPos + chunkSize;
                if (chunkIdStr == "MOMO")
                {
                    momoStart = chunkDataPos;
                    momoEnd = chunkEnd;
                    break;
                }
                stream.Position = chunkEnd;
            }
            if (momoStart < 0 || momoEnd < 0)
            {
                log.WriteLine("[DEBUG] MOMO chunk not found.");
                return;
            }
            stream.Position = momoStart;
            log.WriteLine($"[DEBUG] MOMO chunk found: 0x{momoStart:X} - 0x{momoEnd:X}");
            while (stream.Position + 8 <= momoEnd)
            {
                long subChunkStart = stream.Position;
                var subChunkIdBytes = reader.ReadBytes(4);
                if (subChunkIdBytes.Length < 4) break;
                string subChunkIdStr = new string(subChunkIdBytes.Reverse().Select(b => (char)b).ToArray());
                uint subChunkSize = reader.ReadUInt32();
                long subChunkDataPos = stream.Position;
                long subChunkEnd = subChunkDataPos + subChunkSize;
                log.WriteLine($"[DEBUG] MOMO sub-chunk: {subChunkIdStr} at 0x{subChunkStart:X} size {subChunkSize} (0x{subChunkSize:X})");
                stream.Position = subChunkEnd;
            }
        }

        [Fact]
        public void ExportWmoV14ToObj_ProducesObjInV14OutputFolder()
        {
            string inputPath = Path.GetFullPath("test_data/053_wmo/Ironforge_053.wmo");
            string outputDir = Path.GetFullPath("output/wmo_v14");
            Directory.CreateDirectory(outputDir);
            string objPath = Path.Combine(outputDir, "Ironforge_053.obj");
            string logPath = Path.Combine(outputDir, "Ironforge_053_momo_chunks.txt");
            string debugPath = Path.Combine(outputDir, "Ironforge_053_mesh_debug.txt");

            ListMomoSubChunks(inputPath, logPath); // Write MOMO sub-chunk listing to file

            var mesh = WoWToolbox.Core.WMO.WmoMeshExporter.LoadAllMomoChunksV14Mesh(inputPath);
            using (var debug = new StreamWriter(debugPath, false))
            {
                if (mesh == null)
                {
                    debug.WriteLine("[ERROR] Mesh is null");
                }
                else
                {
                    debug.WriteLine($"[INFO] Vertices: {mesh.Vertices.Count}");
                    debug.WriteLine($"[INFO] Triangles: {mesh.Triangles.Count}");
                    for (int i = 0; i < Math.Min(3, mesh.Vertices.Count); i++)
                        debug.WriteLine($"  Vertex[{i}]: {mesh.Vertices[i].Position}");
                    for (int i = 0; i < Math.Min(3, mesh.Triangles.Count); i++)
                        debug.WriteLine($"  Triangle[{i}]: {mesh.Triangles[i].Index0}, {mesh.Triangles[i].Index1}, {mesh.Triangles[i].Index2}");
                }
            }
            Assert.NotNull(mesh);
            WoWToolbox.Core.WMO.WmoMeshExporter.SaveMergedWmoToObj(mesh, objPath);
            Console.WriteLine($"[V14 EXPORT] {inputPath} -> {objPath}");
            Assert.True(File.Exists(objPath));
        }
    }
} 