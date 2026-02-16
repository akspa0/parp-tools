using MdxLTool.Formats.Mdx;
using MdxViewer.Logging;
using Warcraft.NET.Files.M2;
using Warcraft.NET.Files.M2.Chunks;
using Warcraft.NET.Files.M2.Entries;
using Warcraft.NET.Files.Skin;

namespace MdxViewer.Rendering;

internal static class WarcraftNetM2Adapter
{
    private const uint Md20Magic = 0x3032444D;
    private const uint SkinMagic = 0x4E494B53;

    public static bool IsMd20(byte[] data)
    {
        return data.Length >= 4 && BitConverter.ToUInt32(data, 0) == Md20Magic;
    }

    public static IReadOnlyList<string> BuildSkinCandidates(string modelPath)
    {
        var baseName = Path.GetFileNameWithoutExtension(modelPath);
        var dir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? string.Empty;
        var candidates = new List<string>(8);

        for (int i = 0; i < 4; i++)
        {
            string suffix = i.ToString("D2");
            candidates.Add(Path.ChangeExtension(modelPath, $"{suffix}.skin"));
            candidates.Add(string.IsNullOrEmpty(dir)
                ? $"{baseName}{suffix}.skin"
                : $"{dir}\\{baseName}{suffix}.skin");
        }

        return candidates;
    }

    public static string? FindSkinInFileList(string modelPath, IReadOnlyList<string> files)
    {
        if (files.Count == 0) return null;

        string modelName = Path.GetFileNameWithoutExtension(modelPath).ToLowerInvariant();
        string modelDir = (Path.GetDirectoryName(modelPath) ?? string.Empty).Replace('/', '\\').ToLowerInvariant();

        string? bestPath = null;
        int bestScore = int.MinValue;

        foreach (var file in files)
        {
            if (!file.EndsWith(".skin", StringComparison.OrdinalIgnoreCase))
                continue;

            string normalized = file.Replace('/', '\\');
            string fileName = Path.GetFileName(normalized).ToLowerInvariant();
            string fileBase = Path.GetFileNameWithoutExtension(normalized).ToLowerInvariant();
            string fileDir = (Path.GetDirectoryName(normalized) ?? string.Empty).ToLowerInvariant();

            int score = 0;
            if (fileBase.StartsWith(modelName, StringComparison.OrdinalIgnoreCase)) score += 50;
            if (fileBase.Equals(modelName + "00", StringComparison.OrdinalIgnoreCase)) score += 50;
            if (!string.IsNullOrEmpty(modelDir) && fileDir.Equals(modelDir, StringComparison.OrdinalIgnoreCase)) score += 100;
            if (fileName.Equals(modelName + "00.skin", StringComparison.OrdinalIgnoreCase)) score += 20;

            if (score > bestScore)
            {
                bestScore = score;
                bestPath = normalized;
            }
        }

        return bestScore > 0 ? bestPath : null;
    }

    public static MdxFile BuildRuntimeModel(byte[] m2Bytes, byte[] skinBytes, string modelPath)
    {
        MD21 md21;
        try
        {
            var m2Model = new Model(m2Bytes);
            md21 = m2Model.ModelInformation ?? throw new InvalidDataException("M2 is missing MD21 model information.");
        }
        catch (Exception) when (IsMd20(m2Bytes))
        {
            // Some clients/assets use raw MD20 without MD21 chunk container.
            // Warcraft.NET's top-level Model loader expects MD21 chunking; fallback to direct MD20 parser.
            try
            {
                md21 = new MD21(m2Bytes);
            }
            catch (Exception parseEx)
            {
                throw new InvalidDataException($"Failed to parse raw MD20 model for '{Path.GetFileName(modelPath)}'.", parseEx);
            }
        }

        var skin = ParseSkinData(skinBytes, modelPath);

        var mdx = new MdxFile
        {
            Version = 900,
            Model = new MdlModel
            {
                Name = string.IsNullOrWhiteSpace(md21.Name) ? Path.GetFileNameWithoutExtension(modelPath) : md21.Name,
                Bounds = ToMdlBounds(md21.BoundingBox, md21.BoundingBoxRadius),
            }
        };

        foreach (var texture in md21.Textures)
            mdx.Textures.Add(ToMdlTexture(texture));

        if (mdx.Textures.Count == 0)
            mdx.Textures.Add(new MdlTexture { Path = string.Empty, ReplaceableId = 0, Flags = 0 });

        var sectionMaterialIds = BuildMaterialsFromBatches(mdx, md21, skin);

        if (mdx.Materials.Count == 0)
            mdx.Materials.Add(CreateFallbackMaterial());

        foreach (var geoset in BuildGeosets(md21, skin, sectionMaterialIds, mdx.Materials.Count))
            mdx.Geosets.Add(geoset);

        if (mdx.Geosets.Count == 0)
        {
            var fallback = BuildWholeSkinGeoset(md21, skin, 0);
            if (fallback != null)
                mdx.Geosets.Add(fallback);
        }

        if (mdx.Geosets.Count == 0)
            throw new InvalidDataException("M2 adapter produced no renderable geosets.");

        return mdx;
    }

    private static CMdlBounds ToMdlBounds(Warcraft.NET.Files.Structures.BoundingBox box, float radius)
    {
        return new CMdlBounds
        {
            Extent = new CAaBox
            {
                Min = new C3Vector(box.Minimum.X, box.Minimum.Y, box.Minimum.Z),
                Max = new C3Vector(box.Maximum.X, box.Maximum.Y, box.Maximum.Z),
            },
            Radius = radius,
        };
    }

    private static MdlTexture ToMdlTexture(TextureStruct texture)
    {
        string path = texture.Type == TextureType.None ? texture.Filename ?? string.Empty : string.Empty;
        return new MdlTexture
        {
            Path = path.Replace('/', '\\'),
            ReplaceableId = 0,
            Flags = 0,
        };
    }

    private static int[] BuildMaterialsFromBatches(MdxFile mdx, Warcraft.NET.Files.M2.Chunks.MD21 md21, SkinData skin)
    {
        var sectionMaterialIds = Enumerable.Repeat(-1, skin.Submeshes.Count).ToArray();

        for (int batchIndex = 0; batchIndex < skin.TextureUnits.Count; batchIndex++)
        {
            var batch = skin.TextureUnits[batchIndex];
            ushort renderFlagBits = 0;
            ushort blendMode = 0;

            if (batch.MaterialIndex < md21.RenderFlags.Count)
            {
                renderFlagBits = md21.RenderFlags[batch.MaterialIndex].Flags;
                blendMode = md21.RenderFlags[batch.MaterialIndex].BlendingMode;
            }

            int textureId = ResolveTextureId(md21, batch);
            var textureFlags = (textureId >= 0 && textureId < md21.Textures.Count)
                ? md21.Textures[textureId].Flags
                : 0;

            var material = new MdlMaterial { PriorityPlane = batch.PriorityPlane };
            material.Layers.Add(new MdlTexLayer
            {
                BlendMode = MapBlendMode(blendMode),
                TextureId = textureId,
                CoordId = 0,
                TransformId = -1,
                StaticAlpha = 1.0f,
                Flags = MapLayerFlags(renderFlagBits, textureFlags),
            });

            int materialId = mdx.Materials.Count;
            mdx.Materials.Add(material);

            if (batch.SkinSectionIndex < sectionMaterialIds.Length && sectionMaterialIds[batch.SkinSectionIndex] < 0)
                sectionMaterialIds[batch.SkinSectionIndex] = materialId;
        }

        return sectionMaterialIds;
    }

    private static int ResolveTextureId(Warcraft.NET.Files.M2.Chunks.MD21 md21, SkinTextureUnitData batch)
    {
        int lookupIndex = batch.TextureComboIndex;
        if (lookupIndex >= 0 && lookupIndex < md21.TextureLookup.Count)
        {
            int textureId = md21.TextureLookup[lookupIndex].TextureID;
            if (textureId >= 0 && textureId < md21.Textures.Count)
                return textureId;
        }

        return md21.Textures.Count > 0 ? 0 : -1;
    }

    private static MdlMaterial CreateFallbackMaterial()
    {
        var material = new MdlMaterial { PriorityPlane = 0 };
        material.Layers.Add(new MdlTexLayer
        {
            BlendMode = MdlTexOp.Load,
            TextureId = 0,
            CoordId = 0,
            TransformId = -1,
            StaticAlpha = 1.0f,
            Flags = MdlGeoFlags.None,
        });
        return material;
    }

    private static IEnumerable<MdlGeoset> BuildGeosets(Warcraft.NET.Files.M2.Chunks.MD21 md21, SkinData skin, int[] sectionMaterialIds, int materialCount)
    {
        var flatIndices = skin.TriangleIndices;

        for (int sectionIndex = 0; sectionIndex < skin.Submeshes.Count; sectionIndex++)
        {
            var section = skin.Submeshes[sectionIndex];
            if (section.IndexCount < 3) continue;

            int start = section.IndexStart;
            int indexCount = section.IndexCount;
            int endExclusive = Math.Min(flatIndices.Count, start + indexCount);
            if (start < 0 || start >= endExclusive) continue;

            var geoset = new MdlGeoset
            {
                MaterialId = (sectionIndex < sectionMaterialIds.Length && sectionMaterialIds[sectionIndex] >= 0)
                    ? sectionMaterialIds[sectionIndex]
                    : Math.Min(0, materialCount - 1),
            };

            var remap = new Dictionary<ushort, ushort>();

            for (int indexPos = start; indexPos < endExclusive; indexPos++)
            {
                ushort localSkinVertexIndex = flatIndices[indexPos];
                if (localSkinVertexIndex >= skin.Vertices.Count)
                    continue;

                if (!TryGetVertex(md21, skin, localSkinVertexIndex, out var vertex))
                    continue;

                if (!remap.TryGetValue(localSkinVertexIndex, out ushort mappedIndex))
                {
                    mappedIndex = (ushort)geoset.Vertices.Count;
                    remap[localSkinVertexIndex] = mappedIndex;

                    geoset.Vertices.Add(new C3Vector(vertex.Position.X, vertex.Position.Y, vertex.Position.Z));
                    geoset.Normals.Add(new C3Vector(vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z));
                    geoset.TexCoords.Add(new C2Vector(vertex.TextureCoordX, vertex.TextureCoordY));
                    geoset.VertexGroups.Add(0);
                }

                geoset.Indices.Add(mappedIndex);
            }

            int trimmed = geoset.Indices.Count - (geoset.Indices.Count % 3);
            if (trimmed != geoset.Indices.Count)
                geoset.Indices.RemoveRange(trimmed, geoset.Indices.Count - trimmed);

            if (geoset.Vertices.Count > 0 && geoset.Indices.Count >= 3)
                yield return geoset;
        }
    }

    private static MdlGeoset? BuildWholeSkinGeoset(Warcraft.NET.Files.M2.Chunks.MD21 md21, SkinData skin, int materialId)
    {
        var flatIndices = skin.TriangleIndices;
        if (flatIndices.Count < 3) return null;

        var geoset = new MdlGeoset { MaterialId = materialId };
        var remap = new Dictionary<ushort, ushort>();

        for (int indexPos = 0; indexPos < flatIndices.Count; indexPos++)
        {
            ushort localSkinVertexIndex = flatIndices[indexPos];
            if (!TryGetVertex(md21, skin, localSkinVertexIndex, out var vertex))
                continue;

            if (!remap.TryGetValue(localSkinVertexIndex, out ushort mappedIndex))
            {
                mappedIndex = (ushort)geoset.Vertices.Count;
                remap[localSkinVertexIndex] = mappedIndex;

                geoset.Vertices.Add(new C3Vector(vertex.Position.X, vertex.Position.Y, vertex.Position.Z));
                geoset.Normals.Add(new C3Vector(vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z));
                geoset.TexCoords.Add(new C2Vector(vertex.TextureCoordX, vertex.TextureCoordY));
                geoset.VertexGroups.Add(0);
            }

            geoset.Indices.Add(mappedIndex);
        }

        int trimmed = geoset.Indices.Count - (geoset.Indices.Count % 3);
        if (trimmed != geoset.Indices.Count)
            geoset.Indices.RemoveRange(trimmed, geoset.Indices.Count - trimmed);

        return geoset.Vertices.Count > 0 && geoset.Indices.Count >= 3 ? geoset : null;
    }

    private static bool TryGetVertex(Warcraft.NET.Files.M2.Chunks.MD21 md21, SkinData skin, ushort localSkinVertexIndex, out VerticeStruct vertex)
    {
        vertex = default;
        if (localSkinVertexIndex >= skin.Vertices.Count)
            return false;

        int globalIndex = skin.Vertices[localSkinVertexIndex] + (int)skin.GlobalVertexOffset;
        if (globalIndex < 0 || globalIndex >= md21.Vertices.Count)
        {
            globalIndex = skin.Vertices[localSkinVertexIndex];
            if (globalIndex < 0 || globalIndex >= md21.Vertices.Count)
                return false;
        }

        vertex = md21.Vertices[globalIndex];
        return true;
    }

    private static SkinData ParseSkinData(byte[] skinBytes, string modelPath)
    {
        try
        {
            var skin = new Skin(skinBytes);
            return SkinData.FromWarcraftSkin(skin);
        }
        catch (Exception ex)
        {
            try
            {
                var parsed = ParseLegacySkin(skinBytes);
                ViewerLog.Info(ViewerLog.Category.Mdx,
                    $"[M2] Using legacy skin parser fallback for {Path.GetFileName(modelPath)}");
                return parsed;
            }
            catch (Exception fallbackEx)
            {
                throw new InvalidDataException(
                    $"Failed to parse skin for '{Path.GetFileName(modelPath)}' with both Warcraft.NET and legacy fallback parsers.",
                    new AggregateException(ex, fallbackEx));
            }
        }
    }

    private static SkinData ParseLegacySkin(byte[] skinBytes)
    {
        using var ms = new MemoryStream(skinBytes);
        using var br = new BinaryReader(ms);

        uint maybeMagic = br.ReadUInt32();
        if (maybeMagic != SkinMagic)
            br.BaseStream.Position = 0;

        uint nIndices = br.ReadUInt32();
        uint ofsIndices = br.ReadUInt32();
        uint nTriangles = br.ReadUInt32();
        uint ofsTriangles = br.ReadUInt32();
        uint nBones = br.ReadUInt32();
        uint ofsBones = br.ReadUInt32();
        uint nSubmeshes = br.ReadUInt32();
        uint ofsSubmeshes = br.ReadUInt32();
        uint nTextureUnits = br.ReadUInt32();
        uint ofsTextureUnits = br.ReadUInt32();

        _ = nBones;
        _ = ofsBones;

        if (br.BaseStream.Position + 4 <= br.BaseStream.Length)
            _ = br.ReadUInt32(); // boneCountMax (optional)

        var data = new SkinData();

        if (ofsIndices + (nIndices * 2) <= skinBytes.Length)
        {
            br.BaseStream.Position = ofsIndices;
            for (uint i = 0; i < nIndices; i++)
                data.Vertices.Add(br.ReadUInt16());
        }

        if (ofsTriangles + (nTriangles * 2) <= skinBytes.Length)
        {
            br.BaseStream.Position = ofsTriangles;
            for (uint i = 0; i < nTriangles; i++)
                data.TriangleIndices.Add(br.ReadUInt16());
        }

        int submeshStride = InferStride(ofsSubmeshes, nSubmeshes, ofsTextureUnits, skinBytes.Length, 48, 24);
        if (submeshStride >= 12 && ofsSubmeshes < skinBytes.Length)
        {
            for (uint i = 0; i < nSubmeshes; i++)
            {
                long entryPos = ofsSubmeshes + (i * (uint)submeshStride);
                if (entryPos + 12 > skinBytes.Length) break;

                br.BaseStream.Position = entryPos;
                _ = br.ReadUInt32();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                ushort startTriangle = br.ReadUInt16();
                ushort triangleCount = br.ReadUInt16();

                data.Submeshes.Add(new SkinSubmeshData
                {
                    IndexStart = startTriangle,
                    IndexCount = triangleCount,
                });
            }
        }

        int textureUnitStride = InferStride(ofsTextureUnits, nTextureUnits, (uint)skinBytes.Length, skinBytes.Length, 24, 24);
        if (textureUnitStride >= 18 && ofsTextureUnits < skinBytes.Length)
        {
            for (uint i = 0; i < nTextureUnits; i++)
            {
                long entryPos = ofsTextureUnits + (i * (uint)textureUnitStride);
                if (entryPos + 18 > skinBytes.Length) break;

                br.BaseStream.Position = entryPos;
                byte flags = br.ReadByte();
                byte priority = br.ReadByte();
                _ = br.ReadUInt16();
                ushort submeshIndex = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadInt16();
                ushort materialIndex = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                ushort textureComboIndex = br.ReadUInt16();

                _ = flags;

                data.TextureUnits.Add(new SkinTextureUnitData
                {
                    PriorityPlane = priority,
                    SkinSectionIndex = submeshIndex,
                    MaterialIndex = materialIndex,
                    TextureComboIndex = textureComboIndex,
                });
            }
        }

        if (data.Submeshes.Count == 0 && data.TriangleIndices.Count > 0)
        {
            data.Submeshes.Add(new SkinSubmeshData
            {
                IndexStart = 0,
                IndexCount = data.TriangleIndices.Count,
            });
        }

        if (data.TextureUnits.Count == 0 && data.Submeshes.Count > 0)
        {
            data.TextureUnits.Add(new SkinTextureUnitData
            {
                PriorityPlane = 0,
                SkinSectionIndex = 0,
                MaterialIndex = 0,
                TextureComboIndex = 0,
            });
        }

        if (data.Vertices.Count == 0 || data.TriangleIndices.Count == 0)
            throw new InvalidDataException("Parsed skin data is missing vertices or triangle indices.");

        return data;
    }

    private static int InferStride(uint sectionOffset, uint count, uint nextOffset, int streamLength, int preferred, int minimum)
    {
        if (count == 0) return preferred;
        if (sectionOffset >= streamLength) return preferred;

        long end = nextOffset > sectionOffset && nextOffset <= streamLength ? nextOffset : streamLength;
        long available = end - sectionOffset;
        if (available <= 0) return preferred;

        int stride = (int)(available / count);
        if (stride < minimum) return preferred;
        return stride;
    }

    private sealed class SkinData
    {
        public List<ushort> Vertices { get; } = new();
        public List<ushort> TriangleIndices { get; } = new();
        public List<SkinSubmeshData> Submeshes { get; } = new();
        public List<SkinTextureUnitData> TextureUnits { get; } = new();
        public uint GlobalVertexOffset { get; init; }

        public static SkinData FromWarcraftSkin(Skin skin)
        {
            var data = new SkinData
            {
                GlobalVertexOffset = skin.GlobalVertexOffset,
            };

            data.Vertices.AddRange(skin.Vertices);

            foreach (var t in skin.Triangles)
            {
                data.TriangleIndices.Add(t.Vertex1);
                data.TriangleIndices.Add(t.Vertex2);
                data.TriangleIndices.Add(t.Vertex3);
            }

            foreach (var s in skin.Submeshes)
            {
                data.Submeshes.Add(new SkinSubmeshData
                {
                    IndexStart = s.IndexStart,
                    IndexCount = s.IndexCount,
                });
            }

            foreach (var tu in skin.TextureUnits)
            {
                data.TextureUnits.Add(new SkinTextureUnitData
                {
                    PriorityPlane = tu.PriorityPlane < 0 ? (ushort)0 : (ushort)tu.PriorityPlane,
                    SkinSectionIndex = tu.SkinSectionIndex,
                    MaterialIndex = tu.MaterialIndex,
                    TextureComboIndex = tu.TextureComboIndex,
                });
            }

            return data;
        }
    }

    private sealed class SkinSubmeshData
    {
        public int IndexStart { get; init; }
        public int IndexCount { get; init; }
    }

    private sealed class SkinTextureUnitData
    {
        public int MaterialIndex { get; init; }
        public int TextureComboIndex { get; init; }
        public int SkinSectionIndex { get; init; }
        public ushort PriorityPlane { get; init; }
    }

    private static MdlTexOp MapBlendMode(ushort blendMode)
    {
        // WoW M2 blend modes differ from MDX's texop names; we map them to the closest MDX renderer behavior.
        // Reference semantics (wowdev): 0=Opaque, 1=AlphaKey (cutout), 2=Alpha (blend), 3=Add, 4=Mod, 5=Mod2x.
        // Modes beyond that are engine-/version-specific; default to Blend.
        return blendMode switch
        {
            0 => MdlTexOp.Load,          // Opaque
            1 => MdlTexOp.Transparent,   // AlphaKey (cutout)
            2 => MdlTexOp.Blend,         // Alpha blend
            3 => MdlTexOp.Add,           // Add
            4 => MdlTexOp.Modulate,      // Modulate
            5 => MdlTexOp.Modulate2X,    // Mod2x
            6 => MdlTexOp.AddAlpha,      // Commonly treated as additive-with-alpha
            _ => MdlTexOp.Blend,
        };
    }

    private static MdlGeoFlags MapLayerFlags(ushort renderFlags, TextureFlags textureFlags)
    {
        var flags = MdlGeoFlags.None;

        if ((renderFlags & 0x1) != 0)
            flags |= MdlGeoFlags.Unshaded;
        if ((renderFlags & 0x2) != 0)
            flags |= MdlGeoFlags.Unfogged;
        if ((renderFlags & 0x4) != 0)
            flags |= MdlGeoFlags.TwoSided;

        // Depth flags:
        // Some client/version combinations don't reliably populate these bits.
        // Default to depth-test/write ON unless the file explicitly provides depth bits.
        // This prevents common world doodads (trees, rocks) from rendering "on top" of terrain.
        const ushort M2DepthTest = 0x8;
        const ushort M2DepthWrite = 0x10;
        bool hasExplicitDepthBits = (renderFlags & (M2DepthTest | M2DepthWrite)) != 0;
        if (hasExplicitDepthBits)
        {
            // MDX renderer expects inverse flags (disables)
            if ((renderFlags & M2DepthTest) == 0)
                flags |= MdlGeoFlags.NoDepthTest;
            if ((renderFlags & M2DepthWrite) == 0)
                flags |= MdlGeoFlags.NoDepthSet;
        }

        if (textureFlags.HasFlag(TextureFlags.Flag_0x1_WrapX))
            flags |= MdlGeoFlags.WrapWidth;
        if (textureFlags.HasFlag(TextureFlags.Flag_0x2_WrapY))
            flags |= MdlGeoFlags.WrapHeight;

        return flags;
    }
}
