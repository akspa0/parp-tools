using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using WmoBspConverter.Textures;

namespace WmoBspConverter.Wmo
{
    public class WmoObjExporter
    {
        private class Group
        {
            public List<Vector3> Vertices { get; } = new();
            public List<ushort> Indices { get; } = new();
            public List<(byte Flags, byte Material)> Mopy { get; } = new();
            public List<Vector2> UVs { get; } = new();
            public List<Batch> Batches { get; } = new();
            public string Name { get; set; } = string.Empty;
        }

        private class Batch
        {
            public int FirstFace;
            public int NumFaces;
            public int MaterialId;
        }

        public void Export(string objPath, WmoV14Parser.WmoV14Data data, bool allowFallback, bool includeNonRender = false, bool extractTextures = false)
        {
            // 1) Prefer parsed groups from WmoV14Parser (already robust to header quirks)
            var groups = BuildGroupsFromParsed(data);
            if (groups.Count == 0 || groups.All(g => g.Vertices.Count == 0))
            {
                // 2) Fallback to raw file rescan of MOGP regions
                if (data.FileBytes == null || data.FileBytes.Length == 0)
                    throw new InvalidDataException("No file bytes available to re-scan for groups.");
                groups = ExtractGroupsFromFileBytes(data.FileBytes);
            }
            if (groups.Count == 0)
                throw new InvalidDataException("No geometry groups parsed from WMO v14 file.");

            Directory.CreateDirectory(Path.GetDirectoryName(objPath) ?? ".");
            var objDir = Path.GetDirectoryName(objPath) ?? ".";
            var mtlPath = Path.ChangeExtension(objPath, ".mtl");

            // Prepare textures and MTL — collect materials that will actually be used
            var usedMaterialIds = new HashSet<int>();
            foreach (var g in groups)
            {
                int triCount = g.Indices.Count / 3;
                if (triCount <= 0) continue;
                var mats = GetFaceMaterialsForGroup(g, triCount, includeNonRender);
                foreach (var m in mats)
                {
                    if (m >= 0) usedMaterialIds.Add(m);
                }
            }

            // Map material id -> texture name (from MOMT->MOTX), then process textures if requested
            var matIdToTex = new Dictionary<int, string>();
            var texSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var mid in usedMaterialIds)
            {
                if (mid >= 0 && mid < data.MaterialTextureIndices.Count)
                {
                    var texIdx = (int)data.MaterialTextureIndices[mid];
                    if (texIdx >= 0 && texIdx < data.Textures.Count)
                    {
                        var texName = data.Textures[texIdx];
                        matIdToTex[mid] = texName;
                        texSet.Add(texName);
                    }
                }
            }

            var textureProcessor = new TextureProcessor(objDir, extractTextures);
            var processedTextures = textureProcessor.ProcessTexturesAsync(texSet.ToList()).GetAwaiter().GetResult();
            var texNameToRelPath = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var ti in processedTextures)
            {
                string rel = ti.OutputPath != null ? MakeRelative(objDir, ti.OutputPath) : ti.ShaderName;
                // If ShaderName (no file), try add .tga for viewers
                if (ti.OutputPath == null) rel = ti.ShaderName + ".tga";
                texNameToRelPath[ti.OriginalName] = rel.Replace('\\','/');
            }

            // Write MTL
            using (var mtlWriter = new StreamWriter(mtlPath))
            {
                mtlWriter.WriteLine("# WMO materials");
                foreach (var mid in usedMaterialIds.OrderBy(x => x))
                {
                    var mtlName = $"wow_mat_{mid}";
                    mtlWriter.WriteLine($"newmtl {mtlName}");
                    mtlWriter.WriteLine("Kd 1 1 1");
                    if (matIdToTex.TryGetValue(mid, out var texName) && texNameToRelPath.TryGetValue(texName, out var relPath))
                    {
                        mtlWriter.WriteLine($"map_Kd {relPath}");
                    }
                    mtlWriter.WriteLine();
                }
            }

            using var writer = new StreamWriter(objPath);
            writer.WriteLine("# WMO v14 OBJ export (raw WMO coords)");
            writer.WriteLine($"mtllib {Path.GetFileName(mtlPath)}");

            int vBase = 1;
            int vtBase = 1;
            for (int gi = 0; gi < groups.Count; gi++)
            {
                var g = groups[gi];
                writer.WriteLine($"o {(!string.IsNullOrWhiteSpace(g.Name) ? g.Name : $"group_{gi}")}");

                Console.WriteLine($"[OBJ] Group {gi}: verts={g.Vertices.Count}, idx={g.Indices.Count}, mopy={g.Mopy.Count}");
                if (g.Batches != null && g.Batches.Count > 0)
                {
                    var bsum = string.Join(", ", g.Batches.Select(b => $"[{b.FirstFace}+{b.NumFaces} m={b.MaterialId}]").Take(8));
                    Console.WriteLine($"[OBJ] Group {gi}: MOBA {g.Batches.Count} batches {bsum}{(g.Batches.Count>8?" ...":"")}");
                }

                // Material histogram for diagnostics (from chosen mapping)
                var matHist = new Dictionary<int,int>();
                int triCountG = g.Indices.Count / 3;
                if (triCountG > 0)
                {
                    var faceMats = GetFaceMaterialsForGroup(g, triCountG, includeNonRender);
                    foreach (var m in faceMats)
                    {
                        if (m < 0) continue;
                        if (!matHist.ContainsKey(m)) matHist[m] = 0;
                        matHist[m]++;
                    }
                    var summary = string.Join(", ", matHist.OrderBy(k=>k.Key).Select(k=>$"{k.Key}:{k.Value}"));
                    string mapping = (g.Mopy.Count>=triCountG*2?"MOPYx2":(g.Mopy.Count>=triCountG?"MOPY":(g.Batches.Count>0?"MOBA":"Default")));
                    Console.WriteLine($"[OBJ] Group {gi}: mat histogram = [{summary}] (mapping={mapping})");
                }

                foreach (var v in g.Vertices)
                {
                    writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "v {0:F6} {1:F6} {2:F6}", v.X, v.Y, v.Z));
                }

                bool haveUVs = g.UVs.Count == g.Vertices.Count && g.UVs.Count > 0;
                if (haveUVs)
                {
                    foreach (var uv in g.UVs)
                    {
                        writer.WriteLine(string.Format(CultureInfo.InvariantCulture, "vt {0:F6} {1:F6}", uv.X, 1.0f - uv.Y));
                    }
                }

                int usable = g.Indices.Count - (g.Indices.Count % 3);
                if (usable >= 3)
                {
                    int currentMat = -1;
                    int triCount = usable / 3;
                    var faceMats = GetFaceMaterialsForGroup(g, triCount, includeNonRender);
                    for (int t = 0; t < triCount; t++)
                    {
                        int triBase = t * 3;
                        int a = vBase + g.Indices[triBase + 0];
                        int b = vBase + g.Indices[triBase + 1];
                        int c = vBase + g.Indices[triBase + 2];
                        int mat = faceMats[t];
                        if (mat >= 0 && currentMat != mat)
                        {
                            currentMat = mat;
                            writer.WriteLine($"usemtl wow_mat_{mat}");
                        }
                        if (haveUVs)
                        {
                            int ta = vtBase + g.Indices[triBase + 0];
                            int tb = vtBase + g.Indices[triBase + 1];
                            int tc = vtBase + g.Indices[triBase + 2];
                            writer.WriteLine($"f {a}/{ta} {b}/{tb} {c}/{tc}");
                        }
                        else
                        {
                            writer.WriteLine($"f {a} {b} {c}");
                        }
                    }
                }
                else if (allowFallback && g.Vertices.Count >= 3)
                {
                    int faceCount = g.Mopy.Count;
                    int maxTris = Math.Min(faceCount > 0 ? faceCount : int.MaxValue, g.Vertices.Count / 3);
                    int currentMat = -1;
                    for (int t = 0; t < maxTris; t++)
                    {
                        var mat = (int)(t < g.Mopy.Count ? g.Mopy[t].Material : 0);
                        var flags = (byte)(t < g.Mopy.Count ? g.Mopy[t].Flags : (byte)0x20);
                        if (!includeNonRender && !IsRenderable(flags))
                            continue;
                        if (currentMat != mat)
                        {
                            currentMat = mat;
                            writer.WriteLine($"usemtl wow_mat_{mat}");
                        }
                        int a = vBase + t * 3 + 0;
                        int b = vBase + t * 3 + 1;
                        int c = vBase + t * 3 + 2;
                        if (haveUVs)
                        {
                            int ta = vtBase + t * 3 + 0;
                            int tb = vtBase + t * 3 + 1;
                            int tc = vtBase + t * 3 + 2;
                            writer.WriteLine($"f {a}/{ta} {b}/{tb} {c}/{tc}");
                        }
                        else
                        {
                            writer.WriteLine($"f {a} {b} {c}");
                        }
                    }
                    Console.WriteLine($"[OBJ] Group {gi}: MOVI missing, wrote {maxTris} fallback faces");
                }
                else
                {
                    Console.WriteLine($"[OBJ] Group {gi}: no faces emitted (no MOVI, fallback={allowFallback}, verts={g.Vertices.Count})");
                }

                vBase += g.Vertices.Count;
                if (haveUVs) vtBase += g.UVs.Count;
            }
        }

        private static List<Group> BuildGroupsFromParsed(WmoV14Parser.WmoV14Data data)
        {
            var list = new List<Group>();
            if (data.Groups == null || data.Groups.Count == 0) return list;
            foreach (var gd in data.Groups)
            {
                var g = new Group { Name = gd.Name };
                if (gd.Vertices != null && gd.Vertices.Count > 0)
                    g.Vertices.AddRange(gd.Vertices);
                if (gd.Indices != null && gd.Indices.Count > 0)
                    g.Indices.AddRange(gd.Indices);
                if (gd.FaceMaterials != null && gd.FaceMaterials.Count > 0)
                {
                    for (int i = 0; i < gd.FaceMaterials.Count; i++)
                    {
                        byte flags = (gd.FaceFlags != null && gd.FaceFlags.Count > i) ? gd.FaceFlags[i] : (byte)0;
                        byte mat = gd.FaceMaterials[i];
                        g.Mopy.Add((flags, mat));
                    }
                }
                // Copy UVs when available
                if (gd.UVs != null && gd.UVs.Count > 0)
                {
                    g.UVs.AddRange(gd.UVs);
                }
                // Copy batches if present
                if (gd.Batches != null && gd.Batches.Count > 0)
                {
                    foreach (var b in gd.Batches)
                    {
                        g.Batches.Add(new Batch
                        {
                            FirstFace = (int)b.FirstFace,
                            NumFaces = b.NumFaces,
                            MaterialId = b.MaterialId
                        });
                    }
                }
                list.Add(g);
            }
            return list;
        }

        private static bool IsRenderable(byte flags)
        {
            // Heuristic from community docs: render face when (flags & 0x24) == 0x20
            return (flags & 0x24) == 0x20;
        }

        // Build face materials for a group using MOPY when counts match; otherwise use MOBA; return -1 for faces to skip (non-render and filtering)
        private static int[] GetFaceMaterialsForGroup(Group g, int triCount, bool includeNonRender)
        {
            var result = new int[triCount];
            for (int i = 0; i < triCount; i++) result[i] = -1;

            if (g.Mopy.Count >= triCount * 2)
            {
                // Prefer MOPYx2 when we have at least two entries per triangle
                for (int i = 0; i < triCount; i++)
                {
                    var a = g.Mopy[i * 2 + 0];
                    var b = g.Mopy[i * 2 + 1];
                    (byte Flags, byte Material) chosen = a;
                    bool aRenderable = IsRenderable(a.Flags);
                    bool bRenderable = IsRenderable(b.Flags);
                    if (!includeNonRender)
                    {
                        if (!aRenderable && bRenderable) chosen = b;
                        else if (!aRenderable && !bRenderable) { result[i] = -1; continue; }
                    }
                    else
                    {
                        if (a.Material == 255 && b.Material != 255) chosen = b;
                    }
                    int mat = chosen.Material == 255 ? 0 : chosen.Material;
                    result[i] = mat;
                }
            }
            else if (g.Mopy.Count >= triCount)
            {
                for (int i = 0; i < triCount; i++)
                {
                    var (flags, matb) = g.Mopy[i];
                    if (!includeNonRender && !IsRenderable(flags)) { result[i] = -1; continue; }
                    int mat = matb == 255 ? 0 : matb;
                    result[i] = mat;
                }
            }
            else if (g.Mopy.Count > 0)
            {
                // Partial MOPY: use up to triCount entries
                int mcount = Math.Min(triCount, g.Mopy.Count);
                for (int i = 0; i < mcount; i++)
                {
                    var (flags, matb) = g.Mopy[i];
                    if (!includeNonRender && !IsRenderable(flags)) { result[i] = -1; continue; }
                    int mat = matb == 255 ? 0 : matb;
                    result[i] = mat;
                }
                // Fill remainder with default 0; avoid MOBA since we don't trust our parse yet
                for (int i = mcount; i < triCount; i++) if (result[i] < 0) result[i] = 0;
            }
            else if (g.Batches != null && g.Batches.Count > 0)
            {
                var mats = BuildFaceMaterialsFromMoba(g, triCount);
                for (int i = 0; i < triCount; i++) result[i] = mats[i] < 0 ? 0 : mats[i];
            }
            else
            {
                // Default to 0
                for (int i = 0; i < triCount; i++) result[i] = 0;
            }

            return result;
        }

        private static int[] BuildFaceMaterialsFromMoba(Group g, int triCount)
        {
            var result = new int[triCount];
            for (int i = 0; i < triCount; i++) result[i] = -1;
            if (g == null || g.Batches == null || g.Batches.Count == 0) { for (int i=0;i<triCount;i++) result[i]=0; return result; }
            foreach (var b in g.Batches)
            {
                int start = Math.Max(0, b.FirstFace);
                int end = Math.Min(triCount, b.FirstFace + b.NumFaces);
                int mat = (b.MaterialId == 255) ? 0 : b.MaterialId;
                for (int f = start; f < end; f++)
                {
                    result[f] = mat;
                }
            }
            for (int i = 0; i < triCount; i++) if (result[i] < 0) result[i] = 0;
            return result;
        }

        private static string MakeRelative(string baseDir, string path)
        {
            try
            {
                var rel = Path.GetRelativePath(baseDir, path);
                return string.IsNullOrEmpty(rel) ? path : rel;
            }
            catch
            {
                return path;
            }
        }

        private static List<Group> ExtractGroupsFromFileBytes(byte[] wmoBytes)
        {
            var groups = new List<Group>();
            using var ms = new MemoryStream(wmoBytes, writable: false);
            using var br = new BinaryReader(ms);

            ms.Position = 0;
            while (ms.Position + 8 <= ms.Length)
            {
                var idBytes = br.ReadBytes(4);
                if (idBytes.Length < 4) break;
                Array.Reverse(idBytes);
                string id = System.Text.Encoding.ASCII.GetString(idBytes);
                uint size = br.ReadUInt32();
                long dataPos = ms.Position;
                long next = dataPos + size;
                if (next > ms.Length) break;

                if (id == "MOGP")
                {
                    var g = ReadMogpGroup(ms, br, dataPos, size);
                    groups.Add(g);
                }

                ms.Position = next;
            }

            return groups;
        }

        private static Group ReadMogpGroup(Stream file, BinaryReader br, long dataPos, uint size)
        {
            var g = new Group();
            long subStart = dataPos + 0x40; // typical header size
            long end = dataPos + size;
            if (subStart > end) subStart = dataPos; // defensive

            file.Position = subStart;
            while (file.Position + 8 <= end)
            {
                long hdrPos = file.Position;
                var sidBytes = br.ReadBytes(4);
                if (sidBytes.Length < 4) break;
                Array.Reverse(sidBytes);
                string sid = System.Text.Encoding.ASCII.GetString(sidBytes);
                uint ssz = br.ReadUInt32();
                long sdata = file.Position;
                long snext = sdata + ssz;
                if (snext > end) break;

                if (sid == "MOVT")
                {
                    int count = (int)(ssz / 12);
                    for (int i = 0; i < count; i++)
                    {
                        float x = br.ReadSingle();
                        float y = br.ReadSingle();
                        float z = br.ReadSingle();
                        g.Vertices.Add(new Vector3(x, y, z));
                    }
                }
                else if (sid == "MOVI")
                {
                    int count = (int)(ssz / 2);
                    for (int i = 0; i < count; i++) g.Indices.Add(br.ReadUInt16());
                }
                else if (sid == "MOPY")
                {
                    int count = (int)(ssz / 2);
                    for (int i = 0; i < count; i++)
                    {
                        byte flags = br.ReadByte();
                        byte mat = br.ReadByte();
                        g.Mopy.Add((flags, mat));
                    }
                }
                else
                {
                    // skip
                    file.Position = snext;
                    continue;
                }

                file.Position = snext;
            }

            return g;
        }
    }
}
