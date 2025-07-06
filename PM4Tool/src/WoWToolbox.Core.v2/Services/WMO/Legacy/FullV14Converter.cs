using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Numerics;
using System.Threading.Tasks;
using Serilog;
using WoWToolbox.Core.v2.Foundation.WMO;

namespace WoWToolbox.Core.v2.Services.WMO.Legacy
{
    public class FullV14Converter : IWmoV14Converter
    {
        public async Task<WmoConversionResult> ConvertAsync(string inputWmoPath, string outputDirectory)
        {
            var result = new WmoConversionResult();
            var logPath = Path.Combine(outputDirectory, $"FullV14Converter-{Path.GetFileNameWithoutExtension(inputWmoPath)}-{DateTime.Now:yyyyMMdd-HHmmss}.log");
            result.LogFilePath = logPath;

            var fileLogger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.File(logPath, rollingInterval: RollingInterval.Day)
                .CreateLogger();

            try
            {
                fileLogger.Information("--- Starting WMO v14 to v17 Conversion ---");
                fileLogger.Information($"Input WMO: {inputWmoPath}");
                fileLogger.Information($"Output Directory: {outputDirectory}");

                if (!File.Exists(inputWmoPath))
                    throw new FileNotFoundException("Input WMO file not found.", inputWmoPath);

                Directory.CreateDirectory(outputDirectory); // Ensure output directory exists

                var wmoData = await File.ReadAllBytesAsync(inputWmoPath);
                using var memoryStream = new MemoryStream(wmoData);
                using var reader = new BinaryReader(memoryStream);

                var chunks = ReadAllChunks(reader, fileLogger);
                var convertedData = PerformConversion(wmoData, chunks, fileLogger, out var groupMeshes);

                var convertedWmoPath = Path.Combine(outputDirectory, Path.GetFileName(inputWmoPath).Replace(".wmo", "_v17.wmo"));
                await File.WriteAllBytesAsync(convertedWmoPath, convertedData);
                result.ConvertedWmoPath = convertedWmoPath;

                if (groupMeshes.Any() && groupMeshes.Any(m => m.Vertices.Any()))
                {
                    var textureNames = ExtractTextureNamesFromMotx(wmoData, fileLogger);
                    var texturePaths = WmoTextureExtractor.ExtractTextures(inputWmoPath, textureNames, outputDirectory);
                    result.TexturePaths.AddRange(texturePaths);

                    // Aggregate geometry from all groups for a single OBJ file
                    var allVertices = new List<Vector3>();
                    var allUvs = new List<Vector2>();
                    var allFaces = new List<(int a, int b, int c)>();
                    int vertexOffset = 0;

                    foreach (var groupMesh in groupMeshes)
                    {
                        allVertices.AddRange(groupMesh.Vertices.Select(v => v.Position));
                        allUvs.AddRange(groupMesh.Vertices.Select(v => v.UV));
                        foreach (var face in groupMesh.Indices)
                        {
                            allFaces.Add((face.A + vertexOffset, face.B + vertexOffset, face.C + vertexOffset));
                        }
                        vertexOffset += groupMesh.Vertices.Count;
                    }

                    var objFileName = Path.GetFileNameWithoutExtension(inputWmoPath) + ".obj";
                    var objPath = Path.Combine(outputDirectory, objFileName);
                    var primaryTextureName = texturePaths.Any() ? Path.GetFileName(texturePaths.First()) : null;

                    result.ObjFilePath = WmoObjExporter.Export(objPath, allVertices, allUvs, allFaces, primaryTextureName);
                    result.MtlFilePath = result.ObjFilePath.Replace(".obj", ".mtl");
                }

                result.Success = true;
                fileLogger.Information("--- Conversion Successful ---");
            }
            catch (Exception ex)
            {
                fileLogger.Error(ex, "An unhandled exception occurred during conversion.");
                result.Success = false;
                result.ErrorMessage = ex.Message;
            }
            finally
            {
                ((IDisposable)fileLogger).Dispose();
            }

            return result;
        }

        private List<(string Id, uint Offset, uint Size)> ReadAllChunks(BinaryReader reader, ILogger fileLogger)
        {
            var chunks = new List<(string, uint, uint)>();
            reader.BaseStream.Position = 0;
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                if (reader.BaseStream.Length - reader.BaseStream.Position < 8) break;
                var chunkIdBytes = reader.ReadBytes(4);
                Array.Reverse(chunkIdBytes);
                var chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                var chunkSize = reader.ReadUInt32();
                var chunkOffset = (uint)reader.BaseStream.Position - 8;
                chunks.Add((chunkId, chunkOffset, chunkSize));
                fileLogger.Information($"Found chunk '{chunkId}' at offset 0x{chunkOffset:X} with size {chunkSize}");
                if (reader.BaseStream.Position + chunkSize > reader.BaseStream.Length)
                {
                    fileLogger.Warning($"Chunk '{chunkId}' size is larger than remaining file data. Stopping chunk read.");
                    break;
                }
                reader.BaseStream.Position += chunkSize;
            }
            return chunks;
        }

        private byte[] PerformConversion(byte[] wmoData, List<(string Id, uint Offset, uint Size)> chunks, ILogger fileLogger, out List<WmoGroupMesh> groupMeshes)
        {
            groupMeshes = new List<WmoGroupMesh>();
            fileLogger.Information("Performing conversion logic.");

            var mverChunk = chunks.FirstOrDefault(c => c.Id == "MVER");
            if (mverChunk != default)
            {
                fileLogger.Information("Found MVER chunk. Updating version to 17.");
                BitConverter.GetBytes(17).CopyTo(wmoData, (int)mverChunk.Offset + 8);
            }

            var mogpChunks = chunks.Where(c => c.Id == "MOGP").ToList();
            fileLogger.Information($"Found {mogpChunks.Count} MOGP chunks.");

            foreach (var mogpChunk in mogpChunks)
            {
                var groupMesh = AnalyzeAndBuildGroup(wmoData, mogpChunk.Offset, mogpChunk.Size, fileLogger);
                if (groupMesh != null)
                {
                    groupMeshes.Add(groupMesh);
                }
            }

            return wmoData;
        }

        private WmoGroupMesh AnalyzeAndBuildGroup(byte[] wmoData, uint groupOffset, uint groupSize, ILogger fileLogger)
        {
            fileLogger.Information($"--- Analyzing MOGP at offset 0x{groupOffset:X} ---");
            using var groupStream = new MemoryStream(wmoData, (int)groupOffset + 8, (int)groupSize);
            using var groupReader = new BinaryReader(groupStream);
            var mesh = new WmoGroupMesh();

            var subChunks = new Dictionary<string, byte[]>();
            while (groupStream.Position < groupStream.Length)
            {
                if (groupStream.Length - groupStream.Position < 8) break;

                var subChunkIdBytes = groupReader.ReadBytes(4);
                Array.Reverse(subChunkIdBytes);
                var subChunkId = Encoding.ASCII.GetString(subChunkIdBytes);
                var subChunkSize = groupReader.ReadUInt32();
                var subChunkOffset = (uint)groupStream.Position - 8;

                fileLogger.Information($"  Sub-chunk '{subChunkId}' at relative offset 0x{subChunkOffset:X} with size {subChunkSize}");

                if (groupStream.Position + subChunkSize > groupStream.Length)
                {
                    fileLogger.Warning($"  Sub-chunk '{subChunkId}' size is larger than remaining group data. Stopping analysis for this group.");
                    break;
                }
                
                var data = groupReader.ReadBytes((int)subChunkSize);
                subChunks[subChunkId] = data;

                if (subChunkId == "MOVT")
                {
                    var bytesToDump = Math.Min(256, (int)(groupStream.Length - groupStream.Position));
                    if (bytesToDump > 0)
                    {
                        var nextBytes = new byte[bytesToDump];
                        var bytesRead = groupStream.Read(nextBytes, 0, bytesToDump);
                        fileLogger.Information($"    Dumping {bytesRead} bytes after MOVT:");
                        for (int i = 0; i < bytesRead; i += 16)
                        {
                            var line = nextBytes.Skip(i).Take(16).ToArray();
                            var hex = string.Join(" ", line.Select(b => b.ToString("X2")));
                            fileLogger.Information($"      {i:X4}: {hex}");
                        }
                        groupStream.Position -= bytesRead; // Rewind
                    }
                }
            }
            
            if (subChunks.TryGetValue("MOVT", out var movtData))
            {
                mesh.Vertices = WmoVertex.FromV14(movtData);
                fileLogger.Information($"  Parsed {mesh.Vertices.Count} vertices from MOVT.");
            }

            if (subChunks.TryGetValue("MOVI", out var moviData))
            {
                mesh.Indices = WmoFace.FromV14(moviData);
                fileLogger.Information($"  Parsed {mesh.Indices.Count} faces from MOVI.");
            }
            else
            {
                fileLogger.Warning("  MOVI chunk not found. Geometry will be incomplete.");
            }

            return mesh;
        }

        public byte[] ConvertToV17(byte[] v14Data, string? textureSourceDir = null, string? textureOutputDir = null)
        {
            // Minimal implementation: copy buffer, set version to 17 at offset 8.
            if (v14Data.Length < 12)
                throw new ArgumentException("Invalid WMO data", nameof(v14Data));
            var copy = new byte[v14Data.Length];
            Buffer.BlockCopy(v14Data, 0, copy, 0, v14Data.Length);
            // little-endian uint 17
            copy[8] = 17;
            copy[9] = 0;
            copy[10] = 0;
            copy[11] = 0;
            return copy;
        }

        public string ExportFirstGroupAsObj(string wmoPath, string? objPath = null)
        {
            if (!File.Exists(wmoPath))
                throw new FileNotFoundException("Input WMO file not found.", wmoPath);

            // Decide output location
            if (string.IsNullOrWhiteSpace(objPath))
            {
                var tempDir = Path.Combine(Path.GetTempPath(), "wmo_obj_" + Guid.NewGuid().ToString("N"));
                Directory.CreateDirectory(tempDir);
                objPath = Path.Combine(tempDir, Path.GetFileNameWithoutExtension(wmoPath) + ".obj");
            }
            else
            {
                Directory.CreateDirectory(Path.GetDirectoryName(objPath)!);
            }

            var mtlPath = Path.ChangeExtension(objPath, ".mtl");

            // Minimal stub content – will be replaced by real geometry later
            File.WriteAllText(objPath, "# Placeholder OBJ generated by FullV14Converter stub\n\n");
            File.WriteAllText(mtlPath, "# Placeholder MTL generated by FullV14Converter stub\n");

            return objPath;
        }

        public void ConvertToV17(string inputPath, string outputPath, string? textureOutputDir = null)
        {
            throw new NotImplementedException();
        }

        private List<string> ExtractTextureNamesFromMotx(byte[] wmoData, ILogger fileLogger)
        {
            var textureNames = new List<string>();
            try
            {
                using var memoryStream = new MemoryStream(wmoData);
                using var reader = new BinaryReader(memoryStream);

                // Find MOTX chunk
                reader.BaseStream.Position = 0;
                while (reader.BaseStream.Position < reader.BaseStream.Length)
                {
                    if (reader.BaseStream.Length - reader.BaseStream.Position < 8) break;
                    var chunkIdBytes = reader.ReadBytes(4);
                    Array.Reverse(chunkIdBytes);
                    var chunkId = Encoding.ASCII.GetString(chunkIdBytes);
                    var chunkSize = reader.ReadUInt32();

                    if (chunkId == "MOTX")
                    {
                        var textureNameBytes = reader.ReadBytes((int)chunkSize);
                        var names = Encoding.ASCII.GetString(textureNameBytes).Split(new[] { '\0' }, StringSplitOptions.RemoveEmptyEntries);
                        textureNames.AddRange(names);
                        break; // Assuming only one MOTX chunk
                    }

                    if (reader.BaseStream.Position + chunkSize > reader.BaseStream.Length) break;
                    reader.BaseStream.Position += chunkSize;
                }
            }
            catch (Exception ex)
            {
                fileLogger.Error(ex, "Error extracting texture names from MOTX chunk.");
            }
            return textureNames;
        }
    }
}
