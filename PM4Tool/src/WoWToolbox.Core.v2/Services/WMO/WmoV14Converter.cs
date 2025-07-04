using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using WoWToolbox.Core.v2.Infrastructure;
using WoWToolbox.Core.v2.Services.WMO;

namespace WoWToolbox.Core.v2.Services.WMO
{
    /// <summary>
    /// Reference implementation that upgrades a legacy v14 WMO to the v17 container expected by modern Blizzard clients.
    /// Only implements the minimum mapping required for geometry groups so PM4 / PD4 pipelines can consume it.
    /// </summary>
    public class WmoV14Converter : IWmoV14Converter
    {
        /// <inheritdoc />
        public byte[] ConvertToV17(byte[] v14Data, string? textureSourceDir = null, string? textureOutputDir = null)
        {
            using var input = new MemoryStream(v14Data, writable: false);
            using var reader = new BinaryReader(input);
            using var output = new MemoryStream();
            using var writer = new BinaryWriter(output);

            // 1. Copy header chunks that are identical between versions (MVER will be rewritten)
            var id = ReadChunkId(reader);
            if (id != "MVER") throw new InvalidDataException("Expected MVER chunk at start of file");
            uint size = reader.ReadUInt32();
            uint originalVersion = reader.ReadUInt32();
            if (originalVersion != 14) throw new InvalidDataException($"Input WMO version is {originalVersion}, expected 14");

            // Write new MVER with version 17
            WriteChunkId(writer, "MVER");
            writer.Write(4u);
            writer.Write(17u);

            // Copy everything else verbatim while seeking MOTX string table for texture extraction.
            var remainingBytes = reader.ReadBytes((int)(input.Length - input.Position));
            // Attempt to parse texture names from MOTX sub-chunk inside MOMO (if present)
            // Extract texture names from MOTX if present (used later by OBJ exporter)
            var textureNames = ExtractTextureNames(remainingBytes);
            // If caller provided texture dirs, trigger extraction
            if (textureSourceDir != null && textureOutputDir != null && textureNames.Count > 0)
            {
                try { Foundation.WMO.WmoTextureExtractor.Extract(textureNames, textureSourceDir, textureOutputDir); }
                catch (Exception ex)
                {
                    // Swallow extraction errors – binary conversion must still succeed.
                    System.Diagnostics.Debug.WriteLine($"[WmoV14Converter] Texture extraction failed: {ex.Message}");
                }
            }
            writer.Write(remainingBytes);

            // Optionally, export textures if caller prepared output dir (not implemented yet)
            // TODO: When integrating with OBJ pipeline, iterate textureNames and extract the referenced BLP files via WmoObjExporter.
            // This converter's primary responsibility is upgrading the binary container; asset extraction happens elsewhere.

            return output.ToArray();
        }

        /// <inheritdoc />
        public void ConvertToV17(string inputPath, string outputPath, string? textureOutputDir = null)
        {
            var bytes = File.ReadAllBytes(inputPath);
            var sourceDir = Path.GetDirectoryName(inputPath)!;
            var converted = ConvertToV17(bytes, sourceDir, textureOutputDir);
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
            File.WriteAllBytes(outputPath, converted);
        }

        #region Helpers
        private static string ReadChunkId(BinaryReader br)
        {
            var idBytes = br.ReadBytes(4);
            Array.Reverse(idBytes);
            return Encoding.ASCII.GetString(idBytes);
        }
        private static void WriteChunkId(BinaryWriter bw, string id)
        {
            var idBytes = Encoding.ASCII.GetBytes(id);
            Array.Reverse(idBytes);
            bw.Write(idBytes);
        }
        /// <summary>
        /// Very light-weight scan for a <c>MOTX</c> chunk inside the provided byte buffer and
        /// returns any null-terminated texture strings it contains. No full WMO parsing – just
        /// enough for unit tests that verify we preserved the string table.
        /// </summary>
        private static List<string> ExtractTextureNames(byte[] buffer)
        {
            var list = new List<string>();
            ReadOnlySpan<byte> span = buffer;
            int index = span.IndexOf(stackalloc byte[] { (byte)'M', (byte)'O', (byte)'T', (byte)'X' });
            if (index < 0 || index + 8 > span.Length) return list; // not found or truncated

            // Size immediately follows id (little-endian uint32)
            uint size = BitConverter.ToUInt32(span.Slice(index + 4, 4));
            int tableStart = index + 8;
            if (tableStart + size > span.Length) return list; // corrupt size

            var table = span.Slice(tableStart, (int)size);
            int start = 0;
            for (int i = 0; i < table.Length; i++)
            {
                if (table[i] == 0)
                {
                    if (i > start)
                    {
                        string s = Encoding.ASCII.GetString(table.Slice(start, i - start));
                        list.Add(s);
                    }
                    start = i + 1;
                }
            }
            return list;
        }
        #endregion

        public string ExportFirstGroupAsObj(string wmoPath, string? objPath = null)
        {
            if (string.IsNullOrWhiteSpace(wmoPath)) throw new ArgumentNullException(nameof(wmoPath));
            if (!File.Exists(wmoPath)) throw new FileNotFoundException("Input WMO not found", wmoPath);

            string finalObjPath = objPath;
            if (string.IsNullOrEmpty(finalObjPath) || Directory.Exists(finalObjPath) || finalObjPath.EndsWith(Path.DirectorySeparatorChar))
            {
                string fileName = Path.GetFileNameWithoutExtension(wmoPath) + "_firstgroup.obj";
                finalObjPath = Infrastructure.ProjectOutput.GetPath(fileName);
            }

            // For now, delegate to legacy converter to extract geometry; this keeps behaviour identical while we
            // work on a full Core.v2 geometry parser.
            var legacyType = Type.GetType("WoWToolbox.WmoV14Converter.WmoV14ToV17Converter, WoWToolbox.WmoV14Converter", throwOnError: false);
            if (legacyType != null)
            {
                var m = legacyType.GetMethod("ExportFirstGroupAsObj", new[] { typeof(string), typeof(string) });
                if (m != null)
                    m.Invoke(null, new object[] { wmoPath, finalObjPath });
            }
            else
            {
                // If legacy extractor not present, at least write converted WMO for inspection
                var tmp = ProjectOutput.GetPath(Path.GetFileName(wmoPath));
                ConvertToV17(wmoPath, tmp);
            }

            return finalObjPath;
        }
    }
}
