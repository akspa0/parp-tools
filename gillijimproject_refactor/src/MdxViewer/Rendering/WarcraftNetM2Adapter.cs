using System.Numerics;
using MdxLTool.Formats.Mdx;
using MdxViewer.Logging;
using MdxViewer.Terrain;
using Warcraft.NET.Files.M2;
using Warcraft.NET.Files.M2.Chunks;
using Warcraft.NET.Files.M2.Entries;
using Warcraft.NET.Files.Skin;
using WnBoundingBox = Warcraft.NET.Files.Structures.BoundingBox;

namespace MdxViewer.Rendering;

internal static class WarcraftNetM2Adapter
{
    private const uint Md20Magic = 0x3032444D;
    private const uint Md21Magic = 0x3132444D;
    private const uint SkinMagic = 0x4E494B53;

    public static bool IsMd20(byte[] data)
    {
        return data.Length >= 4 && BitConverter.ToUInt32(data, 0) == Md20Magic;
    }

    public static bool IsMd21(byte[] data)
    {
        return data.Length >= 4 && BitConverter.ToUInt32(data, 0) == Md21Magic;
    }

    public static void ValidateModelProfile(byte[] modelBytes, string modelPath, string? buildVersion)
    {
        var profile = FormatProfileRegistry.ResolveModelProfile(buildVersion);
        if (profile == null)
            return;

        ValidateModelProfile(modelBytes, modelPath, profile, buildVersion);
    }

    public static void ValidateModelProfile(byte[] modelBytes, string modelPath, M2Profile profile, string? buildVersion = null)
    {
        string fileName = Path.GetFileName(modelPath);
        string buildLabel = string.IsNullOrWhiteSpace(buildVersion) ? "unknown" : buildVersion;

        if (modelBytes.Length < 4)
            throw new InvalidDataException($"Model '{fileName}' is too short to validate against {profile.ProfileId}.");

        uint rootMagic = BitConverter.ToUInt32(modelBytes, 0);
        uint requiredMagic = profile.RequiredRootMagic switch
        {
            ModelRootMagic.MD20 => Md20Magic,
            _ => throw new InvalidOperationException($"Unsupported model root magic policy for {profile.ProfileId}.")
        };

        if (rootMagic == Md21Magic && profile.AllowMd21Container)
        {
            ViewerLog.Trace($"[M2] Allowing MD21 container for {fileName} under {profile.ProfileId} (build={buildLabel})");
            return;
        }

        if (rootMagic != requiredMagic)
        {
            throw new InvalidDataException(
                $"Model '{fileName}' is incompatible with {profile.ProfileId} (build={buildLabel}): expected {GetMagicLabel(requiredMagic)} root, found {GetMagicLabel(rootMagic)}.");
        }

        if (modelBytes.Length < 8)
            throw new InvalidDataException($"Model '{fileName}' is too short to read an MD20 version for {profile.ProfileId}.");

        int version = unchecked((int)BitConverter.ToUInt32(modelBytes, 4));
        if (version < profile.MinSupportedVersion || version > profile.MaxSupportedVersion)
        {
            throw new InvalidDataException(
                $"Model '{fileName}' is incompatible with {profile.ProfileId} (build={buildLabel}): MD20 version 0x{version:X} is outside supported range 0x{profile.MinSupportedVersion:X}-0x{profile.MaxSupportedVersion:X}.");
        }

        ViewerLog.Trace(
            $"[M2] Using {profile.ProfileId} for {fileName} (build={buildLabel}, magic={GetMagicLabel(rootMagic)}, version=0x{version:X})");
    }

    public static bool HasRenderableGeometry(MdxFile mdx)
    {
        return mdx.Geosets.Any(geoset => geoset.Vertices.Count > 0 && geoset.Indices.Count >= 3);
    }

    public static string SummarizeGeometry(MdxFile mdx)
    {
        int validGeosets = mdx.Geosets.Count(geoset => geoset.Vertices.Count > 0 && geoset.Indices.Count >= 3);
        int totalVertices = mdx.Geosets.Sum(geoset => geoset.Vertices.Count);
        int totalTriangles = mdx.Geosets.Sum(geoset => geoset.Indices.Count / 3);
        return $"geosets={mdx.Geosets.Count}, validGeosets={validGeosets}, vertices={totalVertices}, triangles={totalTriangles}";
    }

    private static string GetMagicLabel(uint magic)
    {
        return magic switch
        {
            Md20Magic => "MD20",
            Md21Magic => "MD21",
            _ => $"0x{magic:X8}"
        };
    }

    public static IReadOnlyList<string> BuildSkinCandidates(string modelPath)
    {
        var baseName = Path.GetFileNameWithoutExtension(modelPath);
        var dir = Path.GetDirectoryName(modelPath)?.Replace('/', '\\') ?? string.Empty;
        var candidates = new List<string>(10);

        candidates.Add(Path.ChangeExtension(modelPath, ".skin"));
        candidates.Add(string.IsNullOrEmpty(dir)
            ? $"{baseName}.skin"
            : $"{dir}\\{baseName}.skin");

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

    public static MdxFile BuildRuntimeModel(byte[] m2Bytes, byte[]? skinBytes, string modelPath, string? buildVersion = null)
    {
        M2Profile? profile = FormatProfileRegistry.ResolveModelProfile(buildVersion);
        ParsedModelData model = ParseModelInformation(m2Bytes, modelPath, profile, buildVersion);
        SkinData skin = ResolveSkinData(model, skinBytes, modelPath, profile, buildVersion);

        var mdx = new MdxFile
        {
            Version = 900,
            Model = new MdlModel
            {
                Name = string.IsNullOrWhiteSpace(model.Name) ? Path.GetFileNameWithoutExtension(modelPath) : model.Name,
                Bounds = ToMdlBounds(model.BoundingBox, model.BoundingBoxRadius),
            }
        };

        foreach (var texture in model.Textures)
            mdx.Textures.Add(ToMdlTexture(texture));

        if (mdx.Textures.Count == 0)
            mdx.Textures.Add(new MdlTexture { Path = string.Empty, ReplaceableId = 0, Flags = 0 });

        var sectionMaterialIds = BuildMaterialsFromBatches(mdx, model, skin);

        if (mdx.Materials.Count == 0)
            mdx.Materials.Add(CreateFallbackMaterial());

        foreach (var geoset in BuildGeosets(model, skin, sectionMaterialIds, mdx.Materials.Count))
            mdx.Geosets.Add(geoset);

        if (mdx.Geosets.Count == 0)
        {
            var fallback = BuildWholeSkinGeoset(model, skin, 0);
            if (fallback != null)
                mdx.Geosets.Add(fallback);
        }

        if (mdx.Geosets.Count == 0)
            throw new InvalidDataException("M2 adapter produced no renderable geosets.");

        return mdx;
    }

    private static SkinData ResolveSkinData(ParsedModelData model, byte[]? skinBytes, string modelPath, M2Profile? profile, string? buildVersion)
    {
        if (skinBytes != null && skinBytes.Length > 0)
            return ParseSkinData(skinBytes, modelPath, profile);

        if (model.EmbeddedSkin != null)
        {
            ViewerLog.Info(ViewerLog.Category.Mdx,
                $"[M2] Using embedded root profile geometry for {Path.GetFileName(modelPath)} (build={buildVersion ?? "unknown"})");
            return model.EmbeddedSkin;
        }

        throw new InvalidDataException($"No external or embedded skin/profile geometry was available for '{Path.GetFileName(modelPath)}'.");
    }

    private static ParsedModelData ParseModelInformation(byte[] m2Bytes, string modelPath, M2Profile? profile, string? buildVersion)
    {
        string fileName = Path.GetFileName(modelPath);

        if (profile != null && IsMd20(m2Bytes) && !profile.AllowMd21Container)
        {
            try
            {
                ViewerLog.Trace($"[M2] Parsing profiled pre-release MD20: {fileName} ({profile.ProfileId})");
                return ParseProfiledMd20Model(m2Bytes, fileName, profile, buildVersion);
            }
            catch (Exception profiledEx)
            {
                try
                {
                    ViewerLog.Trace($"[M2] Profiled MD20 parse failed, trying Warcraft.NET fallback: {fileName}");
                    return ParseWarcraftNetModel(m2Bytes, modelPath);
                }
                catch (Exception fallbackEx)
                {
                    throw new InvalidDataException(
                        $"Failed to parse profiled MD20 model for '{fileName}'.",
                        new AggregateException(profiledEx, fallbackEx));
                }
            }
        }

        if (IsMd20(m2Bytes))
        {
            try
            {
                ViewerLog.Trace($"[M2] Parsing raw MD20 directly: {fileName}");
                return ParseWarcraftNetModel(m2Bytes, modelPath);
            }
            catch (Exception rawMd20Ex)
            {
                try
                {
                    ViewerLog.Trace($"[M2] Raw MD20 parse failed, trying Warcraft.NET Model wrapper: {fileName}");
                    return ParseWarcraftNetModel(m2Bytes, modelPath);
                }
                catch (Exception wrappedEx)
                {
                    throw new InvalidDataException(
                        $"Failed to parse raw MD20 model for '{fileName}'.",
                        new AggregateException(rawMd20Ex, wrappedEx));
                }
            }
        }

        try
        {
            return ParseWarcraftNetModel(m2Bytes, modelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidDataException($"Failed to parse M2 model '{fileName}'.", ex);
        }
    }

    private static ParsedModelData ParseWarcraftNetModel(byte[] m2Bytes, string modelPath)
    {
        string fileName = Path.GetFileName(modelPath);

        if (IsMd20(m2Bytes))
        {
            try
            {
                return ParsedModelData.FromWarcraftNet(new MD21(m2Bytes));
            }
            catch (Exception rawMd20Ex)
            {
                var wrapped = new Model(m2Bytes);
                if (wrapped.ModelInformation != null)
                    return ParsedModelData.FromWarcraftNet(wrapped.ModelInformation);

                throw new InvalidDataException($"M2 is missing MD21 model information for '{fileName}'.", rawMd20Ex);
            }
        }

        var m2Model = new Model(m2Bytes);
        if (m2Model.ModelInformation != null)
        {
            if (IsMd21(m2Bytes))
                ViewerLog.Trace($"[M2] Parsed MD21 container via Warcraft.NET Model wrapper: {fileName}");
            return ParsedModelData.FromWarcraftNet(m2Model.ModelInformation);
        }

        throw new InvalidDataException("M2 is missing MD21 model information.");
    }

    private static CMdlBounds ToMdlBounds(WnBoundingBox box, float radius)
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

    private static MdlTexture ToMdlTexture(ParsedTextureData texture)
    {
        string path = texture.Type == TextureType.None ? texture.Filename ?? string.Empty : string.Empty;
        return new MdlTexture
        {
            Path = path.Replace('/', '\\'),
            ReplaceableId = 0,
            Flags = 0,
        };
    }

    private static int[] BuildMaterialsFromBatches(MdxFile mdx, ParsedModelData model, SkinData skin)
    {
        var sectionMaterialIds = Enumerable.Repeat(-1, skin.Submeshes.Count).ToArray();

        for (int batchIndex = 0; batchIndex < skin.TextureUnits.Count; batchIndex++)
        {
            var batch = skin.TextureUnits[batchIndex];
            ushort renderFlagBits = 0;
            ushort blendMode = 0;

            if (batch.MaterialIndex < model.RenderFlags.Count)
            {
                renderFlagBits = model.RenderFlags[batch.MaterialIndex].Flags;
                blendMode = model.RenderFlags[batch.MaterialIndex].BlendingMode;
            }

            int textureId = ResolveTextureId(model, batch);
            var textureFlags = (textureId >= 0 && textureId < model.Textures.Count)
                ? model.Textures[textureId].Flags
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

    private static int ResolveTextureId(ParsedModelData model, SkinTextureUnitData batch)
    {
        int lookupIndex = batch.TextureComboIndex;
        if (lookupIndex >= 0 && lookupIndex < model.TextureLookup.Count)
        {
            int textureId = model.TextureLookup[lookupIndex].TextureId;
            if (textureId >= 0 && textureId < model.Textures.Count)
                return textureId;
        }

        return model.Textures.Count > 0 ? 0 : -1;
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

    private static IEnumerable<MdlGeoset> BuildGeosets(ParsedModelData model, SkinData skin, int[] sectionMaterialIds, int materialCount)
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

                if (!TryGetVertex(model, skin, localSkinVertexIndex, out var vertex))
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

    private static MdlGeoset? BuildWholeSkinGeoset(ParsedModelData model, SkinData skin, int materialId)
    {
        var flatIndices = skin.TriangleIndices;
        if (flatIndices.Count < 3) return null;

        var geoset = new MdlGeoset { MaterialId = materialId };
        var remap = new Dictionary<ushort, ushort>();

        for (int indexPos = 0; indexPos < flatIndices.Count; indexPos++)
        {
            ushort localSkinVertexIndex = flatIndices[indexPos];
            if (!TryGetVertex(model, skin, localSkinVertexIndex, out var vertex))
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

    private static bool TryGetVertex(ParsedModelData model, SkinData skin, ushort localSkinVertexIndex, out ParsedVertexData vertex)
    {
        vertex = default;
        if (localSkinVertexIndex >= skin.Vertices.Count)
            return false;

        int globalIndex = skin.Vertices[localSkinVertexIndex] + (int)skin.GlobalVertexOffset;
        if (globalIndex < 0 || globalIndex >= model.Vertices.Count)
        {
            globalIndex = skin.Vertices[localSkinVertexIndex];
            if (globalIndex < 0 || globalIndex >= model.Vertices.Count)
                return false;
        }

        vertex = model.Vertices[globalIndex];
        return true;
    }

    private static SkinData ParseSkinData(byte[] skinBytes, string modelPath, M2Profile? profile = null)
    {
        _ = profile;

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

    private static ParsedModelData ParseProfiledMd20Model(byte[] modelBytes, string fileName, M2Profile profile, string? buildVersion)
    {
        const int NameCountOffset = 0x08;
        const int NameOffsetOffset = 0x0C;
        const int VertexCountOffset = 0x44;
        const int VertexDataOffset = 0x48;
        const int RootProfileCountOffset = 0x4C;
        const int RootProfileDataOffset = 0x50;
        const int BoundingBoxOffset = 0xB4;
        const int BoundingRadiusOffset = 0xCC;

        using var ms = new MemoryStream(modelBytes);
        using var br = new BinaryReader(ms);

        uint magic = br.ReadUInt32();
        if (magic != Md20Magic)
            throw new InvalidDataException($"Model '{fileName}' is not a raw MD20 file.");

        uint version = br.ReadUInt32();
        if (version < profile.MinSupportedVersion || version > profile.MaxSupportedVersion)
        {
            throw new InvalidDataException(
                $"Model '{fileName}' is incompatible with {profile.ProfileId}: MD20 version 0x{version:X} is outside supported range 0x{profile.MinSupportedVersion:X}-0x{profile.MaxSupportedVersion:X}.");
        }

        uint nameLength = ReadUInt32(modelBytes, NameCountOffset);
        uint nameOffset = ReadUInt32(modelBytes, NameOffsetOffset);
        uint vertexCount = ReadUInt32(modelBytes, VertexCountOffset);
        uint vertexOffset = ReadUInt32(modelBytes, VertexDataOffset);
        uint rootProfileCount = ReadUInt32(modelBytes, RootProfileCountOffset);
        uint rootProfileOffset = ReadUInt32(modelBytes, RootProfileDataOffset);

        WnBoundingBox boundingBox = ReadBoundingBox(modelBytes, BoundingBoxOffset);
        float boundingRadius = ReadSingle(modelBytes, BoundingRadiusOffset);

        ValidateSpan(vertexCount, vertexOffset, 0x30, modelBytes.Length, fileName, "vertices");
        bool hasNameSpan = TryValidateOptionalSpan(nameLength, nameOffset, 1, modelBytes.Length, fileName, "name");

        string modelName = string.Empty;
        if (hasNameSpan && nameLength > 0 && nameOffset > 0)
        {
            br.BaseStream.Position = nameOffset;
            modelName = ReadNullTerminatedString(br, (int)nameLength);
        }

        var data = new ParsedModelData
        {
            Name = modelName,
            BoundingBox = boundingBox,
            BoundingBoxRadius = boundingRadius,
            EmbeddedSkin = TryParseEmbeddedRootProfile(modelBytes, rootProfileCount, rootProfileOffset, fileName),
        };

        if (vertexCount > 0)
        {
            for (uint i = 0; i < vertexCount; i++)
            {
                long entryPos = vertexOffset + (i * 0x30u);
                br.BaseStream.Position = entryPos;
                Vector3 position = ReadVector3(br);
                br.BaseStream.Position = entryPos + 20;
                Vector3 normal = ReadVector3(br);
                br.BaseStream.Position = entryPos + 32;
                float textureCoordX = br.ReadSingle();
                float textureCoordY = br.ReadSingle();

                data.Vertices.Add(new ParsedVertexData
                {
                    Position = position,
                    Normal = normal,
                    TextureCoordX = textureCoordX,
                    TextureCoordY = textureCoordY,
                });
            }
        }

        TrySupplementMetadataFromWarcraftNet(modelBytes, fileName, data);

        ViewerLog.Info(ViewerLog.Category.Mdx,
            $"[M2] Using profiled MD20 parser for {fileName} (profile={profile.ProfileId}, build={buildVersion ?? "unknown"}, version=0x{version:X}, vertices={data.Vertices.Count}, textures={data.Textures.Count}, embeddedProfile={(data.EmbeddedSkin != null ? "yes" : "no")})");

        return data;
    }

    private static void TrySupplementMetadataFromWarcraftNet(byte[] modelBytes, string fileName, ParsedModelData data)
    {
        try
        {
            ParsedModelData supplement = ParseWarcraftNetModel(modelBytes, fileName);

            if (data.Textures.Count == 0)
                data.Textures.AddRange(supplement.Textures);
            if (data.RenderFlags.Count == 0)
                data.RenderFlags.AddRange(supplement.RenderFlags);
            if (data.TextureLookup.Count == 0)
                data.TextureLookup.AddRange(supplement.TextureLookup);
            if (string.IsNullOrWhiteSpace(data.Name) && !string.IsNullOrWhiteSpace(supplement.Name))
                data.Name = supplement.Name;
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Warcraft.NET metadata supplement skipped for {fileName}: {ex.Message}");
        }
    }

    private static SkinData? TryParseEmbeddedRootProfile(byte[] modelBytes, uint rootProfileCount, uint rootProfileOffset, string fileName)
    {
        if (rootProfileCount == 0 || rootProfileOffset == 0)
            return null;

        if (!TryValidateOptionalSpan(rootProfileCount, rootProfileOffset, 0x2C, modelBytes.Length, fileName, "rootProfiles"))
            return null;

        EmbeddedProfileHeader? selected = null;
        const uint SelectionLimit = 0x100;

        for (uint i = 0; i < rootProfileCount; i++)
        {
            int entryOffset = checked((int)(rootProfileOffset + (i * 0x2Cu)));
            EmbeddedProfileHeader candidate = ReadEmbeddedProfileHeader(modelBytes, entryOffset);

            bool requiredSpansValid = TryValidateOptionalSpan(candidate.VertexMapCount, candidate.VertexMapOffset, 2, modelBytes.Length, fileName, $"rootProfile[{i}].vertexMap")
                && TryValidateOptionalSpan(candidate.IndexCount, candidate.IndexOffset, 2, modelBytes.Length, fileName, $"rootProfile[{i}].indices")
                && TryValidateOptionalSpan(candidate.SubmeshCount, candidate.SubmeshOffset, 0x30, modelBytes.Length, fileName, $"rootProfile[{i}].submeshes");

            if (!requiredSpansValid)
                continue;

            bool withinLimit = candidate.Selector <= SelectionLimit;
            if (selected == null
                || (withinLimit && selected.Value.Selector > SelectionLimit)
                || (withinLimit == selected.Value.Selector <= SelectionLimit && candidate.Selector > selected.Value.Selector))
            {
                selected = candidate;
            }
        }

        if (selected == null)
            return null;

        var skin = new SkinData();

        using var ms = new MemoryStream(modelBytes);
        using var br = new BinaryReader(ms);

        br.BaseStream.Position = selected.Value.VertexMapOffset;
        for (uint i = 0; i < selected.Value.VertexMapCount; i++)
            skin.Vertices.Add(br.ReadUInt16());

        br.BaseStream.Position = selected.Value.IndexOffset;
        for (uint i = 0; i < selected.Value.IndexCount; i++)
            skin.TriangleIndices.Add(br.ReadUInt16());

        br.BaseStream.Position = selected.Value.SubmeshOffset;
        for (uint i = 0; i < selected.Value.SubmeshCount; i++)
        {
            long recordStart = br.BaseStream.Position;
            _ = br.ReadUInt16();
            ushort vertexStart = br.ReadUInt16();
            ushort vertexCount = br.ReadUInt16();
            _ = br.ReadUInt16();
            _ = br.ReadUInt16();
            ushort indexCount = br.ReadUInt16();
            ushort indexStart = br.ReadUInt16();

            if (indexCount >= 3
                && indexStart < skin.TriangleIndices.Count
                && indexStart + indexCount <= skin.TriangleIndices.Count
                && vertexStart + vertexCount <= skin.Vertices.Count)
            {
                skin.Submeshes.Add(new SkinSubmeshData
                {
                    IndexStart = indexStart,
                    IndexCount = indexCount,
                });
            }

            br.BaseStream.Position = recordStart + 0x30;
        }

        if (selected.Value.BatchCount > 0
            && TryValidateOptionalSpan(selected.Value.BatchCount, selected.Value.BatchOffset, 0x18, modelBytes.Length, fileName, "rootProfile.batches"))
        {
            for (uint i = 0; i < selected.Value.BatchCount; i++)
            {
                int batchOffset = checked((int)(selected.Value.BatchOffset + (i * 0x18u)));
                int submeshIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x04);
                if (submeshIndex < skin.Submeshes.Count)
                {
                    skin.TextureUnits.Add(new SkinTextureUnitData
                    {
                        PriorityPlane = modelBytes[batchOffset + 0x01],
                        SkinSectionIndex = submeshIndex,
                        MaterialIndex = 0,
                        TextureComboIndex = 0,
                    });
                }
            }
        }

        if (skin.Submeshes.Count == 0 && skin.TriangleIndices.Count > 0)
        {
            skin.Submeshes.Add(new SkinSubmeshData
            {
                IndexStart = 0,
                IndexCount = skin.TriangleIndices.Count,
            });
        }

        if (skin.TextureUnits.Count == 0 && skin.Submeshes.Count > 0)
        {
            for (int i = 0; i < skin.Submeshes.Count; i++)
            {
                skin.TextureUnits.Add(new SkinTextureUnitData
                {
                    PriorityPlane = 0,
                    SkinSectionIndex = i,
                    MaterialIndex = 0,
                    TextureComboIndex = 0,
                });
            }
        }

        if (skin.Vertices.Count == 0 || skin.TriangleIndices.Count == 0)
            return null;

        return skin;
    }

    private static EmbeddedProfileHeader ReadEmbeddedProfileHeader(byte[] modelBytes, int entryOffset)
    {
        return new EmbeddedProfileHeader
        {
            VertexMapCount = ReadUInt32(modelBytes, entryOffset + 0x00),
            VertexMapOffset = ReadUInt32(modelBytes, entryOffset + 0x04),
            IndexCount = ReadUInt32(modelBytes, entryOffset + 0x08),
            IndexOffset = ReadUInt32(modelBytes, entryOffset + 0x0C),
            BoneComboCount = ReadUInt32(modelBytes, entryOffset + 0x10),
            BoneComboOffset = ReadUInt32(modelBytes, entryOffset + 0x14),
            SubmeshCount = ReadUInt32(modelBytes, entryOffset + 0x18),
            SubmeshOffset = ReadUInt32(modelBytes, entryOffset + 0x1C),
            BatchCount = ReadUInt32(modelBytes, entryOffset + 0x20),
            BatchOffset = ReadUInt32(modelBytes, entryOffset + 0x24),
            Selector = ReadUInt32(modelBytes, entryOffset + 0x28),
        };
    }

    private static uint ReadUInt32(byte[] data, int offset)
    {
        return BitConverter.ToUInt32(data, offset);
    }

    private static float ReadSingle(byte[] data, int offset)
    {
        return BitConverter.ToSingle(data, offset);
    }

    private static WnBoundingBox ReadBoundingBox(byte[] data, int offset)
    {
        Vector3 min = new(BitConverter.ToSingle(data, offset + 0x00), BitConverter.ToSingle(data, offset + 0x04), BitConverter.ToSingle(data, offset + 0x08));
        Vector3 max = new(BitConverter.ToSingle(data, offset + 0x0C), BitConverter.ToSingle(data, offset + 0x10), BitConverter.ToSingle(data, offset + 0x14));
        return new WnBoundingBox(min, max);
    }

    private static void ValidateSpan(uint count, uint offset, uint stride, int length, string fileName, string label)
    {
        if (count == 0 || offset == 0)
            return;

        ulong total = (ulong)count * stride;
        ulong end = (ulong)offset + total;
        if (offset >= length || end > (ulong)length || end < offset)
        {
            throw new InvalidDataException(
                $"Profiled MD20 span '{label}' is out of range for '{fileName}': count={count}, offset=0x{offset:X}, stride=0x{stride:X}, length=0x{length:X}.");
        }
    }

    private static bool TryValidateOptionalSpan(uint count, uint offset, uint stride, int length, string fileName, string label)
    {
        if (count == 0 || offset == 0)
            return false;

        try
        {
            ValidateSpan(count, offset, stride, length, fileName, label);
            return true;
        }
        catch (InvalidDataException ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] Skipping optional profiled MD20 span for {fileName}: {ex.Message}");
            return false;
        }
    }

    private static WnBoundingBox ReadBoundingBox(BinaryReader br)
    {
        Vector3 min = ReadVector3(br);
        Vector3 max = ReadVector3(br);
        return new WnBoundingBox(min, max);
    }

    private static Vector3 ReadVector3(BinaryReader br)
    {
        return new Vector3(br.ReadSingle(), br.ReadSingle(), br.ReadSingle());
    }

    private static string ReadNullTerminatedString(BinaryReader br, int maxLength)
    {
        byte[] bytes = br.ReadBytes(maxLength);
        int end = Array.IndexOf(bytes, (byte)0);
        if (end < 0)
            end = bytes.Length;
        return System.Text.Encoding.UTF8.GetString(bytes, 0, end);
    }

    private static SkinData ParseProfiledSkin(byte[] skinBytes, M2Profile profile)
    {
        using var ms = new MemoryStream(skinBytes);
        using var br = new BinaryReader(ms);

        if (skinBytes.Length < 44)
            throw new InvalidDataException($"Skin data is too short for {profile.ProfileId} header parsing.");

        uint magic = br.ReadUInt32();
        if (magic != SkinMagic)
            throw new InvalidDataException($"Skin data does not start with SKIN magic for {profile.ProfileId}.");

        uint nVertices = br.ReadUInt32();
        uint ofsVertices = br.ReadUInt32();
        uint nIndices = br.ReadUInt32();
        uint ofsIndices = br.ReadUInt32();
        uint nBones = br.ReadUInt32();
        uint ofsBones = br.ReadUInt32();
        uint nSubmeshes = br.ReadUInt32();
        uint ofsSubmeshes = br.ReadUInt32();
        uint nBatches = br.ReadUInt32();
        uint ofsBatches = br.ReadUInt32();
        uint globalVertexOffset = br.ReadUInt32();

        uint nShadowBatches = 0;
        uint ofsShadowBatches = 0;
        if (br.BaseStream.Position + 16 <= br.BaseStream.Length)
        {
            nShadowBatches = br.ReadUInt32();
            ofsShadowBatches = br.ReadUInt32();
            br.ReadBytes(8);
        }

        _ = nBones;
        _ = ofsBones;
        _ = nShadowBatches;
        _ = ofsShadowBatches;

        var data = new SkinData { GlobalVertexOffset = globalVertexOffset };

        if (ofsVertices + (nVertices * 2) <= skinBytes.Length)
        {
            br.BaseStream.Position = ofsVertices;
            for (uint i = 0; i < nVertices; i++)
                data.Vertices.Add(br.ReadUInt16());
        }

        if (ofsIndices + (nIndices * 2) <= skinBytes.Length)
        {
            br.BaseStream.Position = ofsIndices;
            for (uint i = 0; i < nIndices; i++)
                data.TriangleIndices.Add(br.ReadUInt16());
        }

        if (ofsSubmeshes + (nSubmeshes * (uint)profile.SkinLikeAStride) <= skinBytes.Length)
        {
            for (uint i = 0; i < nSubmeshes; i++)
            {
                long entryPos = ofsSubmeshes + (i * (uint)profile.SkinLikeAStride);
                br.BaseStream.Position = entryPos;

                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                ushort indexStart = br.ReadUInt16();
                ushort indexCount = br.ReadUInt16();

                data.Submeshes.Add(new SkinSubmeshData
                {
                    IndexStart = indexStart,
                    IndexCount = indexCount,
                });
            }
        }

        if (ofsBatches + (nBatches * (uint)profile.SkinLikeBStride) <= skinBytes.Length)
        {
            for (uint i = 0; i < nBatches; i++)
            {
                long entryPos = ofsBatches + (i * (uint)profile.SkinLikeBStride);
                br.BaseStream.Position = entryPos;

                byte flags = br.ReadByte();
                byte priority = br.ReadByte();
                _ = br.ReadUInt16();
                ushort submeshIndex = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
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
            throw new InvalidDataException($"Profiled skin parse for {profile.ProfileId} produced no vertices or indices.");

        return data;
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

    private sealed class ParsedModelData
    {
        public string Name { get; set; } = string.Empty;
        public WnBoundingBox BoundingBox { get; init; }
        public float BoundingBoxRadius { get; init; }
        public SkinData? EmbeddedSkin { get; init; }
        public List<ParsedVertexData> Vertices { get; } = new();
        public List<ParsedTextureData> Textures { get; } = new();
        public List<ParsedRenderFlagData> RenderFlags { get; } = new();
        public List<ParsedTextureLookupData> TextureLookup { get; } = new();

        public static ParsedModelData FromWarcraftNet(MD21 md21)
        {
            var data = new ParsedModelData
            {
                Name = md21.Name ?? string.Empty,
                BoundingBox = md21.BoundingBox,
                BoundingBoxRadius = md21.BoundingBoxRadius,
            };

            foreach (var vertex in md21.Vertices)
            {
                data.Vertices.Add(new ParsedVertexData
                {
                    Position = vertex.Position,
                    Normal = vertex.Normal,
                    TextureCoordX = vertex.TextureCoordX,
                    TextureCoordY = vertex.TextureCoordY,
                });
            }

            foreach (var texture in md21.Textures)
            {
                data.Textures.Add(new ParsedTextureData
                {
                    Type = texture.Type,
                    Flags = texture.Flags,
                    Filename = texture.Filename ?? string.Empty,
                });
            }

            foreach (var renderFlag in md21.RenderFlags)
            {
                data.RenderFlags.Add(new ParsedRenderFlagData
                {
                    Flags = renderFlag.Flags,
                    BlendingMode = renderFlag.BlendingMode,
                });
            }

            foreach (var lookup in md21.TextureLookup)
            {
                data.TextureLookup.Add(new ParsedTextureLookupData
                {
                    TextureId = lookup.TextureID,
                });
            }

            return data;
        }
    }

    private struct ParsedVertexData
    {
        public Vector3 Position;
        public Vector3 Normal;
        public float TextureCoordX;
        public float TextureCoordY;
    }

    private sealed class ParsedTextureData
    {
        public TextureType Type { get; init; }
        public TextureFlags Flags { get; init; }
        public string Filename { get; init; } = string.Empty;
    }

    private struct ParsedRenderFlagData
    {
        public ushort Flags;
        public ushort BlendingMode;
    }

    private struct ParsedTextureLookupData
    {
        public int TextureId;
    }

    private struct EmbeddedProfileHeader
    {
        public uint VertexMapCount;
        public uint VertexMapOffset;
        public uint IndexCount;
        public uint IndexOffset;
        public uint BoneComboCount;
        public uint BoneComboOffset;
        public uint SubmeshCount;
        public uint SubmeshOffset;
        public uint BatchCount;
        public uint BatchOffset;
        public uint Selector;
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

        // Do not infer NoDepthTest / NoDepthSet from Warcraft.NET M2 render flags here.
        // The bit layout is not stable across client versions, and treating 0x8/0x10
        // as depth disables causes reflective or fog-like world models to render through terrain.

        if (textureFlags.HasFlag(TextureFlags.Flag_0x1_WrapX))
            flags |= MdlGeoFlags.WrapWidth;
        if (textureFlags.HasFlag(TextureFlags.Flag_0x2_WrapY))
            flags |= MdlGeoFlags.WrapHeight;

        return flags;
    }
}
