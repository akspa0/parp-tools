using System.Numerics;
using System.Reflection;
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
            },
            RawParticleEmitterCount = model.RawParticleEmitterCount,
            RawRibbonEmitterCount = model.RawRibbonEmitterCount,
        };

        foreach (var sequence in model.Sequences)
        {
            mdx.Sequences.Add(new MdlSequence
            {
                Name = sequence.Name,
                Time = new CiRange { Start = sequence.StartFrame, End = sequence.EndFrame },
            });
        }

        foreach (uint globalSequence in model.GlobalSequences)
            mdx.GlobalSequences.Add(globalSequence);

        foreach (var texture in model.Textures)
            mdx.Textures.Add(ToMdlTexture(texture));

        foreach (var textureAnimation in model.TextureAnimations)
            mdx.TextureAnimations.Add(textureAnimation);

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
                var md21 = new MD21(m2Bytes);
                ParsedModelData data = ParsedModelData.FromWarcraftNet(md21);
                TrySupplementAnimationMetadataFromWarcraftNet(md21, m2Bytes, fileName, data);
                TrySupplementRawModelMetadata(m2Bytes, fileName, data);
                return data;
            }
            catch (Exception rawMd20Ex)
            {
                var wrapped = new Model(m2Bytes);
                if (wrapped.ModelInformation != null)
                {
                    ParsedModelData data = ParsedModelData.FromWarcraftNet(wrapped.ModelInformation);
                    TrySupplementAnimationMetadataFromWarcraftNet(wrapped.ModelInformation, m2Bytes, fileName, data);
                    TrySupplementRawModelMetadata(m2Bytes, fileName, data);
                    return data;
                }

                throw new InvalidDataException($"M2 is missing MD21 model information for '{fileName}'.", rawMd20Ex);
            }
        }

        var m2Model = new Model(m2Bytes);
        if (m2Model.ModelInformation != null)
        {
            if (IsMd21(m2Bytes))
                ViewerLog.Trace($"[M2] Parsed MD21 container via Warcraft.NET Model wrapper: {fileName}");
            ParsedModelData data = ParsedModelData.FromWarcraftNet(m2Model.ModelInformation);
            TrySupplementAnimationMetadataFromWarcraftNet(m2Model.ModelInformation, m2Bytes, fileName, data);
            TrySupplementRawModelMetadata(m2Bytes, fileName, data);
            return data;
        }

        throw new InvalidDataException("M2 is missing MD21 model information.");
    }

    private static void TrySupplementAnimationMetadataFromWarcraftNet(MD21 md21, byte[] modelBytes, string fileName, ParsedModelData data)
    {
        try
        {
            PopulateSequenceMetadata(md21, data);
            PopulateTransparencyMetadata(md21, modelBytes, data);
            PopulateColorMetadata(md21, modelBytes, data);
            PopulateUvAnimationMetadata(md21, modelBytes, data);
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Warcraft.NET animation metadata supplement skipped for {fileName}: {ex.Message}");
        }
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
        uint replaceableId = MapReplaceableId(texture.Type);
        string path = texture.Filename ?? string.Empty;
        return new MdlTexture
        {
            Path = path.Replace('/', '\\'),
            ReplaceableId = replaceableId,
            Flags = MapTextureFlags(texture.Flags),
        };
    }

    private static uint MapReplaceableId(TextureType textureType)
    {
        return textureType == TextureType.None ? 0u : (uint)textureType;
    }

    private static uint MapTextureFlags(TextureFlags textureFlags)
    {
        uint flags = 0;

        if (textureFlags.HasFlag(TextureFlags.Flag_0x1_WrapX))
            flags |= (uint)MdlGeoFlags.WrapWidth;
        if (textureFlags.HasFlag(TextureFlags.Flag_0x2_WrapY))
            flags |= (uint)MdlGeoFlags.WrapHeight;

        return flags;
    }

    private static int[] BuildMaterialsFromBatches(MdxFile mdx, ParsedModelData model, SkinData skin)
    {
        var sectionMaterialIds = Enumerable.Repeat(-1, skin.Submeshes.Count).ToArray();

        for (int batchIndex = 0; batchIndex < skin.TextureUnits.Count; batchIndex++)
        {
            var batch = skin.TextureUnits[batchIndex];
            if (batch.SkinSectionIndex < 0 || batch.SkinSectionIndex >= sectionMaterialIds.Length)
                continue;

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

            int materialId = sectionMaterialIds[batch.SkinSectionIndex];
            MdlMaterial material;
            if (materialId >= 0)
            {
                material = mdx.Materials[materialId];
                if (batch.PriorityPlane > material.PriorityPlane)
                    material.PriorityPlane = batch.PriorityPlane;
            }
            else
            {
                material = new MdlMaterial { PriorityPlane = batch.PriorityPlane };
                materialId = mdx.Materials.Count;
                mdx.Materials.Add(material);
                sectionMaterialIds[batch.SkinSectionIndex] = materialId;
            }

            int coordId = ResolveTextureCoordId(model, batch);

            material.Layers.Add(new MdlTexLayer
            {
                BlendMode = MapBlendMode(blendMode),
                TextureId = textureId,
                CoordId = coordId,
                TransformId = -1,
                StaticAlpha = 1.0f,
                StaticColor = new C3Color(1.0f, 1.0f, 1.0f),
                StaticColorAlpha = 1.0f,
                Flags = MapLayerFlags(renderFlagBits, textureFlags, coordId),
            });

            ApplyLayerAnimationMetadata(material.Layers[^1], mdx, model, batch);
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

    private static int ResolveTextureCoordId(ParsedModelData model, SkinTextureUnitData batch)
    {
        int lookupIndex = batch.TextureCoordComboIndex;
        if (lookupIndex >= 0 && lookupIndex < model.TextureCoordLookup.Count)
            return model.TextureCoordLookup[lookupIndex].CoordId;

        return 0;
    }

    private static void ApplyLayerAnimationMetadata(MdlTexLayer layer, MdxFile mdx, ParsedModelData model, SkinTextureUnitData batch)
    {
        if (batch.ColorIndex >= 0 && batch.ColorIndex < model.Colors.Count)
        {
            ParsedColorData color = model.Colors[batch.ColorIndex];
            layer.StaticColor = color.StaticColor;
            layer.StaticColorAlpha = color.StaticAlpha;
            layer.ColorInterpolation = color.ColorInterpolation;
            layer.ColorGlobalSeqId = color.ColorGlobalSeqId;
            layer.ColorAlphaInterpolation = color.AlphaInterpolation;
            layer.ColorAlphaGlobalSeqId = color.AlphaGlobalSeqId;
            layer.ColorKeys.AddRange(color.ColorKeys);
            layer.ColorAlphaKeys.AddRange(color.AlphaKeys);
        }

        int transparencyId = ResolveTransparencyId(model, batch.TransparencyComboIndex);
        if (transparencyId >= 0 && transparencyId < model.Transparency.Count)
        {
            ParsedTransparencyData transparency = model.Transparency[transparencyId];
            layer.AlphaInterpolation = transparency.Interpolation;
            layer.AlphaGlobalSeqId = transparency.GlobalSeqId;
            layer.AlphaKeys.AddRange(transparency.Keys);
        }

        int textureAnimationId = ResolveTextureAnimationId(model, batch.TextureAnimationLookupIndex);
        if (textureAnimationId >= 0 && textureAnimationId < mdx.TextureAnimations.Count)
            layer.TransformId = textureAnimationId;
    }

    private static int ResolveTransparencyId(ParsedModelData model, int lookupIndex)
    {
        if (lookupIndex >= 0 && lookupIndex < model.TransparencyLookup.Count)
            return model.TransparencyLookup[lookupIndex].TransparencyId;

        return -1;
    }

    private static int ResolveTextureAnimationId(ParsedModelData model, int lookupIndex)
    {
        if (lookupIndex >= 0 && lookupIndex < model.UvAnimationLookup.Count)
            return model.UvAnimationLookup[lookupIndex].TextureAnimationId;

        return -1;
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
            StaticColor = new C3Color(1.0f, 1.0f, 1.0f),
            StaticColorAlpha = 1.0f,
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

                    AddVertexToGeoset(geoset, vertex);
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

                AddVertexToGeoset(geoset, vertex);
            }

            geoset.Indices.Add(mappedIndex);
        }

        int trimmed = geoset.Indices.Count - (geoset.Indices.Count % 3);
        if (trimmed != geoset.Indices.Count)
            geoset.Indices.RemoveRange(trimmed, geoset.Indices.Count - trimmed);

        return geoset.Vertices.Count > 0 && geoset.Indices.Count >= 3 ? geoset : null;
    }

    private static void AddVertexToGeoset(MdlGeoset geoset, ParsedVertexData vertex)
    {
        int vertexIndex = geoset.Vertices.Count;
        geoset.Vertices.Add(new C3Vector(vertex.Position.X, vertex.Position.Y, vertex.Position.Z));
        geoset.Normals.Add(new C3Vector(vertex.Normal.X, vertex.Normal.Y, vertex.Normal.Z));
        // ModelRenderer expects UV sets packed as [all uv0][all uv1], not per-vertex interleaving.
        geoset.TexCoords.Insert(vertexIndex, new C2Vector(vertex.TextureCoord0X, vertex.TextureCoord0Y));
        geoset.TexCoords.Add(new C2Vector(vertex.TextureCoord1X, vertex.TextureCoord1Y));
        geoset.VertexGroups.Add(0);
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
            var parsed = SkinData.FromWarcraftSkin(skin);
            TrySupplementSkinBatchMetadata(skinBytes, modelPath, parsed, profile);
            return parsed;
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

    private static void TrySupplementSkinBatchMetadata(byte[] skinBytes, string modelPath, SkinData parsedSkin, M2Profile? profile)
    {
        try
        {
            SkinData? supplement = TryParseSupplementalSkinData(skinBytes, profile);
            if (supplement == null)
                return;

            int mergeCount = Math.Min(parsedSkin.TextureUnits.Count, supplement.TextureUnits.Count);
            for (int i = 0; i < mergeCount; i++)
            {
                parsedSkin.TextureUnits[i].ColorIndex = supplement.TextureUnits[i].ColorIndex;
                parsedSkin.TextureUnits[i].TextureCoordComboIndex = supplement.TextureUnits[i].TextureCoordComboIndex;
                parsedSkin.TextureUnits[i].TransparencyComboIndex = supplement.TextureUnits[i].TransparencyComboIndex;
                parsedSkin.TextureUnits[i].TextureAnimationLookupIndex = supplement.TextureUnits[i].TextureAnimationLookupIndex;
            }

            if (parsedSkin.TextureUnits.Count != supplement.TextureUnits.Count)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx,
                    $"[M2] Skin batch metadata supplement count mismatch for {Path.GetFileName(modelPath)}: parsed={parsedSkin.TextureUnits.Count}, raw={supplement.TextureUnits.Count}");
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx,
                $"[M2] Skin batch metadata supplement skipped for {Path.GetFileName(modelPath)}: {ex.Message}");
        }
    }

    private static SkinData? TryParseSupplementalSkinData(byte[] skinBytes, M2Profile? profile)
    {
        if (profile != null)
        {
            try
            {
                return ParseProfiledSkin(skinBytes, profile);
            }
            catch
            {
            }
        }

        try
        {
            return ParseLegacySkin(skinBytes);
        }
        catch
        {
            return null;
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
                float textureCoord0X = br.ReadSingle();
                float textureCoord0Y = br.ReadSingle();
                float textureCoord1X = br.ReadSingle();
                float textureCoord1Y = br.ReadSingle();

                data.Vertices.Add(new ParsedVertexData
                {
                    Position = position,
                    Normal = normal,
                    TextureCoord0X = textureCoord0X,
                    TextureCoord0Y = textureCoord0Y,
                    TextureCoord1X = textureCoord1X,
                    TextureCoord1Y = textureCoord1Y,
                });
            }
        }

        TrySupplementMetadataFromWarcraftNet(modelBytes, fileName, data);
        TrySupplementRawModelMetadata(modelBytes, fileName, data);
        TryParseProfiledMaterialMetadata(modelBytes, fileName, data);

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

    private static void TrySupplementRawModelMetadata(byte[] modelBytes, string fileName, ParsedModelData data)
    {
        if (!IsMd20(modelBytes))
            return;

        TrySupplementRawVertexTextureCoords(modelBytes, fileName, data);

        if (data.TextureCoordLookup.Count == 0)
        {
            var textureCoordLookup = ReadRawTextureCoordLookup(modelBytes, fileName);
            if (textureCoordLookup != null)
                data.TextureCoordLookup.AddRange(textureCoordLookup);
        }
    }

    private static void TrySupplementRawVertexTextureCoords(byte[] modelBytes, string fileName, ParsedModelData data)
    {
        const int VertexCountOffset = 0x44;
        const int VertexDataOffset = 0x48;

        if (data.Vertices.Count == 0 || modelBytes.Length < VertexDataOffset + 4)
            return;

        uint vertexCount = ReadUInt32(modelBytes, VertexCountOffset);
        uint vertexOffset = ReadUInt32(modelBytes, VertexDataOffset);
        if (!TryValidateOptionalSpan(vertexCount, vertexOffset, 0x30, modelBytes.Length, fileName, "raw.vertices"))
            return;

        int count = Math.Min(data.Vertices.Count, (int)vertexCount);
        using var ms = new MemoryStream(modelBytes);
        using var br = new BinaryReader(ms);

        for (int i = 0; i < count; i++)
        {
            long entryPos = vertexOffset + (i * 0x30L);
            br.BaseStream.Position = entryPos + 32;

            var vertex = data.Vertices[i];
            vertex.TextureCoord0X = br.ReadSingle();
            vertex.TextureCoord0Y = br.ReadSingle();
            vertex.TextureCoord1X = br.ReadSingle();
            vertex.TextureCoord1Y = br.ReadSingle();
            data.Vertices[i] = vertex;
        }
    }

    private static List<ParsedTextureCoordLookupData>? ReadRawTextureCoordLookup(byte[] modelBytes, string fileName)
    {
        const int TextureCoordLookupCountOffset = 0x88;
        const int TextureCoordLookupDataOffset = 0x8C;

        if (modelBytes.Length < TextureCoordLookupDataOffset + 4)
            return null;

        uint count = ReadUInt32(modelBytes, TextureCoordLookupCountOffset);
        uint dataOffset = ReadUInt32(modelBytes, TextureCoordLookupDataOffset);
        return TryReadTextureCoordLookup(modelBytes, fileName, count, dataOffset, out List<ParsedTextureCoordLookupData>? lookup)
            ? lookup
            : null;
    }

    private static void TryParseProfiledMaterialMetadata(byte[] modelBytes, string fileName, ParsedModelData data)
    {
        bool changed = false;

        List<ParsedTextureData>? parsedTextures = DiscoverProfiledTextureTable(modelBytes, fileName);
        if (parsedTextures != null && ShouldPreferProfiledTextures(parsedTextures, data.Textures))
        {
            data.Textures.Clear();
            data.Textures.AddRange(parsedTextures);
            changed = true;
        }

        List<ParsedRenderFlagData>? parsedRenderFlags = DiscoverProfiledRenderFlags(modelBytes, fileName);
        if (parsedRenderFlags != null && ShouldPreferProfiledRenderFlags(parsedRenderFlags, data.RenderFlags))
        {
            data.RenderFlags.Clear();
            data.RenderFlags.AddRange(parsedRenderFlags);
            changed = true;
        }

        List<ParsedTextureLookupData>? parsedTextureLookup = null;
        if (data.Textures.Count > 0)
            parsedTextureLookup = DiscoverProfiledTextureLookup(modelBytes, fileName, data.Textures.Count);

        if (parsedTextureLookup != null && ShouldPreferProfiledTextureLookup(parsedTextureLookup, data.TextureLookup, data.Textures.Count))
        {
            data.TextureLookup.Clear();
            data.TextureLookup.AddRange(parsedTextureLookup);
            changed = true;
        }

        if (changed)
        {
            ViewerLog.Info(ViewerLog.Category.Mdx,
                $"[M2] Parsed direct profiled metadata for {fileName}: textures={data.Textures.Count}, renderFlags={data.RenderFlags.Count}, textureLookup={data.TextureLookup.Count}");
        }
    }

    private static bool ShouldPreferProfiledTextures(IReadOnlyList<ParsedTextureData> candidate, IReadOnlyList<ParsedTextureData> current)
    {
        if (candidate.Count == 0)
            return false;

        if (current.Count == 0)
            return true;

        return EvaluateTextureTableQuality(candidate) > EvaluateTextureTableQuality(current);
    }

    private static int EvaluateTextureTableQuality(IReadOnlyList<ParsedTextureData> textures)
    {
        int namedCount = 0;
        int replaceableCount = 0;

        foreach (ParsedTextureData texture in textures)
        {
            if (!string.IsNullOrWhiteSpace(texture.Filename))
                namedCount++;
            if (texture.Type != TextureType.None)
                replaceableCount++;
        }

        return textures.Count + (namedCount * 32) + (replaceableCount * 2);
    }

    private static bool ShouldPreferProfiledRenderFlags(IReadOnlyList<ParsedRenderFlagData> candidate, IReadOnlyList<ParsedRenderFlagData> current)
    {
        if (candidate.Count == 0)
            return false;

        if (current.Count == 0)
            return true;

        return EvaluateRenderFlagQuality(candidate) > EvaluateRenderFlagQuality(current);
    }

    private static int EvaluateRenderFlagQuality(IReadOnlyList<ParsedRenderFlagData> renderFlags)
    {
        int plausibleBlendModes = 0;
        foreach (ParsedRenderFlagData renderFlag in renderFlags)
        {
            if (renderFlag.BlendingMode <= 6)
                plausibleBlendModes++;
        }

        return (plausibleBlendModes * 8) + renderFlags.Count;
    }

    private static bool ShouldPreferProfiledTextureLookup(
        IReadOnlyList<ParsedTextureLookupData> candidate,
        IReadOnlyList<ParsedTextureLookupData> current,
        int textureCount)
    {
        if (candidate.Count == 0)
            return false;

        if (current.Count == 0)
            return true;

        return EvaluateTextureLookupQuality(candidate, textureCount) > EvaluateTextureLookupQuality(current, textureCount);
    }

    private static int EvaluateTextureLookupQuality(IReadOnlyList<ParsedTextureLookupData> lookup, int textureCount)
    {
        int validEntries = 0;
        foreach (ParsedTextureLookupData entry in lookup)
        {
            if (entry.TextureId >= 0 && entry.TextureId < textureCount)
                validEntries++;
        }

        return (validEntries * 4) + lookup.Count;
    }

    private static List<ParsedTextureData>? DiscoverProfiledTextureTable(byte[] modelBytes, string fileName)
    {
        const int HeaderStart = 0x48;
        const int HeaderEnd = 0xA0;
        List<ParsedTextureData>? best = null;
        int bestScore = int.MinValue;

        for (int headerOffset = HeaderStart; headerOffset <= HeaderEnd; headerOffset += 4)
        {
            uint count = ReadUInt32(modelBytes, headerOffset);
            uint dataOffset = ReadUInt32(modelBytes, headerOffset + 4);
            if (!TryReadTextureTable(modelBytes, fileName, count, dataOffset, out List<ParsedTextureData>? candidate, out int score))
                continue;

            if (score > bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }

        return best;
    }

    private static bool TryReadTextureTable(byte[] modelBytes, string fileName, uint count, uint dataOffset, out List<ParsedTextureData>? textures, out int score)
    {
        textures = null;
        score = 0;

        if (count == 0 || count > 4096 || dataOffset == 0)
            return false;

        if (!TryValidateOptionalSpan(count, dataOffset, 0x10, modelBytes.Length, fileName, "profiled.textures"))
            return false;

        var parsed = new List<ParsedTextureData>((int)count);
        int namedTextures = 0;

        for (uint i = 0; i < count; i++)
        {
            int entryOffset = checked((int)(dataOffset + (i * 0x10u)));
            uint rawType = ReadUInt32(modelBytes, entryOffset + 0x00);
            uint rawFlags = ReadUInt32(modelBytes, entryOffset + 0x04);
            uint nameLength = ReadUInt32(modelBytes, entryOffset + 0x08);
            uint nameOffset = ReadUInt32(modelBytes, entryOffset + 0x0C);

            if (rawType > (uint)TextureType.Unk2)
                return false;
            if ((rawFlags & ~0x3u) != 0)
                return false;

            string filename = string.Empty;
            if (nameLength > 0 && nameOffset > 0)
            {
                if (!TryValidateOptionalSpan(nameLength, nameOffset, 1, modelBytes.Length, fileName, $"profiled.texture[{i}].name"))
                    return false;

                filename = ReadUtf8String(modelBytes, (int)nameOffset, (int)nameLength);
                if (!string.IsNullOrWhiteSpace(filename))
                    namedTextures++;
            }

            parsed.Add(new ParsedTextureData
            {
                Type = (TextureType)rawType,
                Flags = (TextureFlags)rawFlags,
                Filename = filename,
            });
        }

        score = (int)Math.Min(count, 256) + (namedTextures * 32);
        textures = parsed;
        return parsed.Count > 0;
    }

    private static List<ParsedRenderFlagData>? DiscoverProfiledRenderFlags(byte[] modelBytes, string fileName)
    {
        const int HeaderStart = 0x48;
        const int HeaderEnd = 0xA0;
        List<ParsedRenderFlagData>? best = null;
        int bestScore = int.MinValue;

        for (int headerOffset = HeaderStart; headerOffset <= HeaderEnd; headerOffset += 4)
        {
            uint count = ReadUInt32(modelBytes, headerOffset);
            uint dataOffset = ReadUInt32(modelBytes, headerOffset + 4);
            if (!TryReadRenderFlags(modelBytes, fileName, count, dataOffset, out List<ParsedRenderFlagData>? candidate, out int score))
                continue;

            if (score > bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }

        return best;
    }

    private static bool TryReadRenderFlags(byte[] modelBytes, string fileName, uint count, uint dataOffset, out List<ParsedRenderFlagData>? renderFlags, out int score)
    {
        renderFlags = null;
        score = 0;

        if (count == 0 || count > 2048 || dataOffset == 0)
            return false;

        if (!TryValidateOptionalSpan(count, dataOffset, 0x04, modelBytes.Length, fileName, "profiled.renderFlags"))
            return false;

        var parsed = new List<ParsedRenderFlagData>((int)count);
        int plausibleBlendModes = 0;

        for (uint i = 0; i < count; i++)
        {
            int entryOffset = checked((int)(dataOffset + (i * 0x04u)));
            ushort flags = BitConverter.ToUInt16(modelBytes, entryOffset + 0x00);
            ushort blendMode = BitConverter.ToUInt16(modelBytes, entryOffset + 0x02);
            if (blendMode <= 6)
                plausibleBlendModes++;

            parsed.Add(new ParsedRenderFlagData
            {
                Flags = flags,
                BlendingMode = blendMode,
            });
        }

        if (plausibleBlendModes == 0)
            return false;

        score = (plausibleBlendModes * 8) + (int)Math.Min(count, 128);
        renderFlags = parsed;
        return true;
    }

    private static List<ParsedTextureLookupData>? DiscoverProfiledTextureLookup(byte[] modelBytes, string fileName, int textureCount)
    {
        const int HeaderStart = 0x48;
        const int HeaderEnd = 0xA0;
        List<ParsedTextureLookupData>? best = null;
        int bestScore = int.MinValue;

        for (int headerOffset = HeaderStart; headerOffset <= HeaderEnd; headerOffset += 4)
        {
            uint count = ReadUInt32(modelBytes, headerOffset);
            uint dataOffset = ReadUInt32(modelBytes, headerOffset + 4);
            if (!TryReadTextureLookup(modelBytes, fileName, count, dataOffset, textureCount, out List<ParsedTextureLookupData>? candidate, out int score))
                continue;

            if (score > bestScore)
            {
                best = candidate;
                bestScore = score;
            }
        }

        return best;
    }

    private static bool TryReadTextureLookup(byte[] modelBytes, string fileName, uint count, uint dataOffset, int textureCount, out List<ParsedTextureLookupData>? lookup, out int score)
    {
        lookup = null;
        score = 0;

        if (count == 0 || count > 8192 || dataOffset == 0 || textureCount <= 0)
            return false;

        if (!TryValidateOptionalSpan(count, dataOffset, 0x02, modelBytes.Length, fileName, "profiled.textureLookup"))
            return false;

        var parsed = new List<ParsedTextureLookupData>((int)count);
        int validEntries = 0;

        for (uint i = 0; i < count; i++)
        {
            int entryOffset = checked((int)(dataOffset + (i * 0x02u)));
            ushort textureId = BitConverter.ToUInt16(modelBytes, entryOffset);
            if (textureId == ushort.MaxValue)
            {
                parsed.Add(new ParsedTextureLookupData { TextureId = -1 });
                continue;
            }

            if (textureId < textureCount)
                validEntries++;
            else
                return false;

            parsed.Add(new ParsedTextureLookupData { TextureId = textureId });
        }

        if (validEntries == 0)
            return false;

        score = (validEntries * 4) + (int)Math.Min(count, 256);
        lookup = parsed;
        return true;
    }

    private static bool TryReadTextureCoordLookup(byte[] modelBytes, string fileName, uint count, uint dataOffset, out List<ParsedTextureCoordLookupData>? lookup)
    {
        lookup = null;

        if (count == 0 || count > 8192 || dataOffset == 0)
            return false;

        if (!TryValidateOptionalSpan(count, dataOffset, 0x02, modelBytes.Length, fileName, "raw.textureCoordLookup"))
            return false;

        var parsed = new List<ParsedTextureCoordLookupData>((int)count);
        for (uint i = 0; i < count; i++)
        {
            int entryOffset = checked((int)(dataOffset + (i * 0x02u)));
            short coordId = BitConverter.ToInt16(modelBytes, entryOffset);
            if (coordId < -1 || coordId > 1)
                return false;

            parsed.Add(new ParsedTextureCoordLookupData { CoordId = coordId });
        }

        lookup = parsed;
        return parsed.Count > 0;
    }

    private static string ReadUtf8String(byte[] data, int offset, int length)
    {
        int safeLength = Math.Max(0, Math.Min(length, data.Length - offset));
        if (safeLength == 0)
            return string.Empty;

        int end = offset;
        int limit = offset + safeLength;
        while (end < limit && data[end] != 0)
            end++;

        return System.Text.Encoding.UTF8.GetString(data, offset, end - offset);
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
            ushort skinSectionId = br.ReadUInt16();
            ushort level = br.ReadUInt16();
            ushort vertexStart = br.ReadUInt16();
            ushort vertexCount = br.ReadUInt16();
            ushort indexStart = br.ReadUInt16();
            ushort indexCount = br.ReadUInt16();
            _ = br.ReadUInt16();
            _ = br.ReadUInt16();
            _ = br.ReadUInt16();
            _ = br.ReadUInt16();

            if (indexCount >= 3
                && indexStart < skin.TriangleIndices.Count
                && indexStart + indexCount <= skin.TriangleIndices.Count
                && vertexStart + vertexCount <= skin.Vertices.Count)
            {
                skin.Submeshes.Add(new SkinSubmeshData
                {
                    SkinSectionId = skinSectionId,
                    Level = level,
                    VertexStart = vertexStart,
                    VertexCount = vertexCount,
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
                short colorIndex = BitConverter.ToInt16(modelBytes, batchOffset + 0x06);
                ushort materialIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x08);
                ushort textureComboIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x0E);
                ushort textureCoordComboIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x10);
                ushort transparencyComboIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x12);
                ushort textureAnimationLookupIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x14);
                int submeshIndex = BitConverter.ToUInt16(modelBytes, batchOffset + 0x04);
                if (submeshIndex < skin.Submeshes.Count)
                {
                    skin.TextureUnits.Add(new SkinTextureUnitData
                    {
                        PriorityPlane = modelBytes[batchOffset + 0x01],
                        SkinSectionIndex = submeshIndex,
                        ColorIndex = colorIndex,
                        MaterialIndex = materialIndex,
                        TextureComboIndex = textureComboIndex,
                        TextureCoordComboIndex = textureCoordComboIndex,
                        TransparencyComboIndex = transparencyComboIndex,
                        TextureAnimationLookupIndex = textureAnimationLookupIndex,
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
                short colorIndex = br.ReadInt16();
                ushort materialIndex = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                ushort textureComboIndex = br.ReadUInt16();
                ushort textureCoordComboIndex = profile.SkinLikeBStride >= 20 && entryPos + 20 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;
                ushort transparencyComboIndex = profile.SkinLikeBStride >= 22 && entryPos + 22 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;
                ushort textureAnimationLookupIndex = profile.SkinLikeBStride >= 24 && entryPos + 24 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;

                _ = flags;

                data.TextureUnits.Add(new SkinTextureUnitData
                {
                    PriorityPlane = priority,
                    SkinSectionIndex = submeshIndex,
                    ColorIndex = colorIndex,
                    MaterialIndex = materialIndex,
                    TextureComboIndex = textureComboIndex,
                    TextureCoordComboIndex = textureCoordComboIndex,
                    TransparencyComboIndex = transparencyComboIndex,
                    TextureAnimationLookupIndex = textureAnimationLookupIndex,
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
                short colorIndex = br.ReadInt16();
                ushort materialIndex = br.ReadUInt16();
                _ = br.ReadUInt16();
                _ = br.ReadUInt16();
                ushort textureComboIndex = br.ReadUInt16();
                ushort textureCoordComboIndex = textureUnitStride >= 20 && entryPos + 20 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;
                ushort transparencyComboIndex = textureUnitStride >= 22 && entryPos + 22 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;
                ushort textureAnimationLookupIndex = textureUnitStride >= 24 && entryPos + 24 <= skinBytes.Length
                    ? br.ReadUInt16()
                    : (ushort)0;

                _ = flags;

                data.TextureUnits.Add(new SkinTextureUnitData
                {
                    PriorityPlane = priority,
                    SkinSectionIndex = submeshIndex,
                    ColorIndex = colorIndex,
                    MaterialIndex = materialIndex,
                    TextureComboIndex = textureComboIndex,
                    TextureCoordComboIndex = textureCoordComboIndex,
                    TransparencyComboIndex = transparencyComboIndex,
                    TextureAnimationLookupIndex = textureAnimationLookupIndex,
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
                    SkinSectionId = s.SkinSectionId,
                    Level = s.Level,
                    VertexStart = s.VertexStart,
                    VertexCount = s.VertexCount,
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
                    ColorIndex = -1,
                    TextureCoordComboIndex = 0,
                    TransparencyComboIndex = -1,
                    TextureAnimationLookupIndex = -1,
                });
            }

            return data;
        }
    }

    private sealed class SkinSubmeshData
    {
        public int SkinSectionId { get; init; }
        public int Level { get; init; }
        public int VertexStart { get; init; }
        public int VertexCount { get; init; }
        public int IndexStart { get; init; }
        public int IndexCount { get; init; }
    }

    private sealed class SkinTextureUnitData
    {
        public int ColorIndex { get; set; } = -1;
        public int MaterialIndex { get; set; }
        public int TextureComboIndex { get; set; }
        public int TextureCoordComboIndex { get; set; }
        public int TransparencyComboIndex { get; set; } = -1;
        public int TextureAnimationLookupIndex { get; set; } = -1;
        public int SkinSectionIndex { get; set; }
        public ushort PriorityPlane { get; set; }
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
        public List<ParsedTextureCoordLookupData> TextureCoordLookup { get; } = new();
        public List<ParsedSequenceData> Sequences { get; } = new();
        public List<uint> GlobalSequences { get; } = new();
        public List<ParsedTransparencyData> Transparency { get; } = new();
        public List<ParsedTransparencyLookupData> TransparencyLookup { get; } = new();
        public List<ParsedColorData> Colors { get; } = new();
        public List<MdlTextureAnimation> TextureAnimations { get; } = new();
        public List<ParsedUvAnimationLookupData> UvAnimationLookup { get; } = new();
        public int RawParticleEmitterCount { get; init; }
        public int RawRibbonEmitterCount { get; init; }

        public static ParsedModelData FromWarcraftNet(MD21 md21)
        {
            var data = new ParsedModelData
            {
                Name = md21.Name ?? string.Empty,
                BoundingBox = md21.BoundingBox,
                BoundingBoxRadius = md21.BoundingBoxRadius,
                RawParticleEmitterCount = md21.ParticleEmitters?.Count ?? 0,
                RawRibbonEmitterCount = md21.RibbonEmitters?.Count ?? 0,
            };

            foreach (var vertex in md21.Vertices)
            {
                data.Vertices.Add(new ParsedVertexData
                {
                    Position = vertex.Position,
                    Normal = vertex.Normal,
                    TextureCoord0X = vertex.TextureCoordX,
                    TextureCoord0Y = vertex.TextureCoordY,
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

    private readonly record struct RawTrackKeyFrame(int Time, object Value, object InTangent, object OutTangent);

    private static void PopulateSequenceMetadata(MD21 md21, ParsedModelData data)
    {
        data.Sequences.Clear();
        data.GlobalSequences.Clear();

        int sequenceStart = 0;
        int animationIndex = 0;
        foreach (object animation in EnumerateMember(md21, "Animations"))
        {
            int animationId = ReadIntMember(animation, "AnimationID");
            int subAnimationId = ReadIntMember(animation, "SubAnimationID");
            int length = Math.Max(0, ReadIntMember(animation, "Length"));
            int sequenceEnd = sequenceStart + length;

            data.Sequences.Add(new ParsedSequenceData
            {
                Name = $"Anim_{animationIndex}_{animationId}_{subAnimationId}",
                StartFrame = sequenceStart,
                EndFrame = sequenceEnd,
            });

            sequenceStart = sequenceEnd + 1;
            animationIndex++;
        }

        foreach (object sequence in EnumerateMember(md21, "Sequences"))
        {
            int timestamp = Math.Max(0, ReadIntMember(sequence, "Timestamp"));
            data.GlobalSequences.Add((uint)timestamp);
        }
    }

    private static void PopulateTransparencyMetadata(MD21 md21, byte[] rawBytes, ParsedModelData data)
    {
        data.Transparency.Clear();
        data.TransparencyLookup.Clear();

        foreach (object transparency in EnumerateMember(md21, "Transparency"))
        {
            object block = GetRequiredMemberValue(transparency, "Alpha");
            var parsed = new ParsedTransparencyData
            {
                Interpolation = ReadAnimationInterpolation(block),
                GlobalSeqId = ReadGlobalSequenceId(block),
            };
            parsed.Keys.AddRange(ReadFloatKeys(block, rawBytes, data.Sequences, ConvertAlpha));
            data.Transparency.Add(parsed);
        }

        foreach (object lookup in EnumerateMember(md21, "TransparencyLookup"))
        {
            data.TransparencyLookup.Add(new ParsedTransparencyLookupData
            {
                TransparencyId = NormalizeLookupIndex(ReadIntMember(lookup, "TransparencyID")),
            });
        }
    }

    private static void PopulateColorMetadata(MD21 md21, byte[] rawBytes, ParsedModelData data)
    {
        data.Colors.Clear();

        foreach (object color in EnumerateMember(md21, "Colors"))
        {
            object colorBlock = GetRequiredMemberValue(color, "Color");
            object alphaBlock = GetRequiredMemberValue(color, "Alpha");

            var parsed = new ParsedColorData
            {
                ColorInterpolation = ReadAnimationInterpolation(colorBlock),
                ColorGlobalSeqId = ReadGlobalSequenceId(colorBlock),
                AlphaInterpolation = ReadAnimationInterpolation(alphaBlock),
                AlphaGlobalSeqId = ReadGlobalSequenceId(alphaBlock),
            };

            parsed.ColorKeys.AddRange(ReadColorKeys(colorBlock, rawBytes, data.Sequences));
            parsed.AlphaKeys.AddRange(ReadFloatKeys(alphaBlock, rawBytes, data.Sequences, ConvertAlpha));
            data.Colors.Add(parsed);
        }
    }

    private static void PopulateUvAnimationMetadata(MD21 md21, byte[] rawBytes, ParsedModelData data)
    {
        data.TextureAnimations.Clear();
        data.UvAnimationLookup.Clear();

        foreach (object uvAnimation in EnumerateMember(md21, "UVAnimations"))
        {
            var parsed = new MdlTextureAnimation
            {
                TranslationTrack = ReadVectorTrack(GetRequiredMemberValue(uvAnimation, "Translation"), rawBytes, data.Sequences),
                RotationTrack = ReadQuaternionTrack(GetRequiredMemberValue(uvAnimation, "Rotation"), rawBytes, data.Sequences),
                ScalingTrack = ReadVectorTrack(GetRequiredMemberValue(uvAnimation, "Scaling"), rawBytes, data.Sequences),
            };

            data.TextureAnimations.Add(parsed);
        }

        foreach (object lookup in EnumerateMember(md21, "UVAnimLookup"))
        {
            data.UvAnimationLookup.Add(new ParsedUvAnimationLookupData
            {
                TextureAnimationId = NormalizeLookupIndex(ReadIntMember(lookup, "AnimatedTextureID")),
            });
        }
    }

    private static IEnumerable<object> EnumerateMember(object instance, string memberName)
    {
        if (!TryGetMemberValue(instance, memberName, out object? value) || value is not System.Collections.IEnumerable enumerable)
            yield break;

        foreach (object? item in enumerable)
        {
            if (item != null)
                yield return item;
        }
    }

    private static List<MdlAnimKey<float>> ReadFloatKeys(object block, byte[] rawBytes, IReadOnlyList<ParsedSequenceData> sequences, Func<object, float> convert)
    {
        var keys = new List<MdlAnimKey<float>>();
        foreach (var keyFrame in ReadTrackKeyFrames(block, rawBytes, sequences))
        {
            keys.Add(new MdlAnimKey<float>
            {
                Time = keyFrame.Time,
                Value = convert(keyFrame.Value),
                TangentIn = convert(keyFrame.InTangent),
                TangentOut = convert(keyFrame.OutTangent),
            });
        }

        return keys;
    }

    private static List<MdlAnimKey<C3Color>> ReadColorKeys(object block, byte[] rawBytes, IReadOnlyList<ParsedSequenceData> sequences)
    {
        var keys = new List<MdlAnimKey<C3Color>>();
        foreach (var keyFrame in ReadTrackKeyFrames(block, rawBytes, sequences))
        {
            keys.Add(new MdlAnimKey<C3Color>
            {
                Time = keyFrame.Time,
                Value = ConvertColor(keyFrame.Value),
                ColorTangentIn = ConvertColor(keyFrame.InTangent),
                ColorTangentOut = ConvertColor(keyFrame.OutTangent),
            });
        }

        return keys;
    }

    private static MdlAnimTrack<C3Vector>? ReadVectorTrack(object block, byte[] rawBytes, IReadOnlyList<ParsedSequenceData> sequences)
    {
        var track = new MdlAnimTrack<C3Vector>
        {
            InterpolationType = ReadTrackInterpolation(block),
            GlobalSeqId = ReadGlobalSequenceId(block),
        };

        foreach (var keyFrame in ReadTrackKeyFrames(block, rawBytes, sequences))
        {
            track.Keys.Add(new MdlTrackKey<C3Vector>
            {
                Frame = keyFrame.Time,
                Value = ConvertVector(keyFrame.Value),
                InTan = ConvertVector(keyFrame.InTangent),
                OutTan = ConvertVector(keyFrame.OutTangent),
            });
        }

        return track.Keys.Count > 0 ? track : null;
    }

    private static MdlAnimTrack<C4Quaternion>? ReadQuaternionTrack(object block, byte[] rawBytes, IReadOnlyList<ParsedSequenceData> sequences)
    {
        var track = new MdlAnimTrack<C4Quaternion>
        {
            InterpolationType = ReadTrackInterpolation(block),
            GlobalSeqId = ReadGlobalSequenceId(block),
        };

        foreach (var keyFrame in ReadTrackKeyFrames(block, rawBytes, sequences))
        {
            track.Keys.Add(new MdlTrackKey<C4Quaternion>
            {
                Frame = keyFrame.Time,
                Value = ConvertQuaternion(keyFrame.Value),
                InTan = ConvertQuaternion(keyFrame.InTangent),
                OutTan = ConvertQuaternion(keyFrame.OutTangent),
            });
        }

        return track.Keys.Count > 0 ? track : null;
    }

    private static List<RawTrackKeyFrame> ReadTrackKeyFrames(object block, byte[] rawBytes, IReadOnlyList<ParsedSequenceData> sequences)
    {
        var keys = new List<RawTrackKeyFrame>();
        using var ms = new MemoryStream(rawBytes, writable: false);
        using var br = new BinaryReader(ms);

        Array timestampTracks = GetArrayReferenceElements(GetRequiredMemberValue(block, "Timestamps"), br);
        Array valueTracks = GetArrayReferenceElements(GetRequiredMemberValue(block, "Values"), br);
        int trackCount = Math.Min(timestampTracks.Length, valueTracks.Length);
        bool isGlobalSequence = ReadGlobalSequenceId(block) >= 0;

        for (int trackIndex = 0; trackIndex < trackCount; trackIndex++)
        {
            Array timestamps = GetArrayReferenceElements(timestampTracks.GetValue(trackIndex), br);
            Array values = GetArrayReferenceElements(valueTracks.GetValue(trackIndex), br);
            int keyCount = Math.Min(timestamps.Length, values.Length);
            int timeOffset = !isGlobalSequence && trackIndex < sequences.Count ? sequences[trackIndex].StartFrame : 0;

            for (int keyIndex = 0; keyIndex < keyCount; keyIndex++)
            {
                object rawValue = values.GetValue(keyIndex) ?? throw new InvalidDataException("Track key value was null.");
                object value = TryGetTrackSubValue(rawValue, "Value") ?? rawValue;
                object inTangent = TryGetTrackSubValue(rawValue, "InTangent") ?? TryGetTrackSubValue(rawValue, "InTan") ?? value;
                object outTangent = TryGetTrackSubValue(rawValue, "OutTangent") ?? TryGetTrackSubValue(rawValue, "OutTan") ?? value;

                keys.Add(new RawTrackKeyFrame(
                    Time: Convert.ToInt32(timestamps.GetValue(keyIndex) ?? 0) + timeOffset,
                    Value: value,
                    InTangent: inTangent,
                    OutTangent: outTangent));
            }
        }

        keys.Sort(static (left, right) => left.Time.CompareTo(right.Time));
        return keys;
    }

    private static Array GetArrayReferenceElements(object? arrayReference, BinaryReader br)
    {
        if (arrayReference == null)
            return Array.Empty<object>();

        if (arrayReference is Array array)
            return array;

        MethodInfo? getElements = arrayReference.GetType().GetMethod("GetElements", BindingFlags.Instance | BindingFlags.Public);
        if (getElements == null)
            throw new InvalidDataException($"Array reference type '{arrayReference.GetType().FullName}' did not expose GetElements(BinaryReader).");

        object? elements = getElements.Invoke(arrayReference, new object[] { br });
        return elements as Array ?? Array.Empty<object>();
    }

    private static object? TryGetTrackSubValue(object instance, string memberName)
    {
        return TryGetMemberValue(instance, memberName, out object? value) ? value : null;
    }

    private static bool TryGetMemberValue(object instance, string memberName, out object? value)
    {
        const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.IgnoreCase;

        PropertyInfo? property = instance.GetType().GetProperty(memberName, Flags);
        if (property != null)
        {
            value = property.GetValue(instance);
            return true;
        }

        FieldInfo? field = instance.GetType().GetField(memberName, Flags);
        if (field != null)
        {
            value = field.GetValue(instance);
            return true;
        }

        value = null;
        return false;
    }

    private static object GetRequiredMemberValue(object instance, string memberName)
    {
        if (!TryGetMemberValue(instance, memberName, out object? value) || value == null)
            throw new InvalidDataException($"Required member '{memberName}' was missing on '{instance.GetType().FullName}'.");

        return value;
    }

    private static int ReadIntMember(object instance, string memberName)
    {
        object value = GetRequiredMemberValue(instance, memberName);
        int converted = Convert.ToInt32(value);
        return converted == ushort.MaxValue ? -1 : converted;
    }

    private static int NormalizeLookupIndex(int value)
    {
        return value == ushort.MaxValue ? -1 : value;
    }

    private static int ReadGlobalSequenceId(object block)
    {
        return NormalizeLookupIndex(ReadIntMember(block, "GlobalSequence"));
    }

    private static MdlAnimInterpolation ReadAnimationInterpolation(object block)
    {
        int interpolation = ReadIntMember(block, "InterpolationType");
        return interpolation switch
        {
            0 => MdlAnimInterpolation.None,
            1 => MdlAnimInterpolation.Linear,
            2 => MdlAnimInterpolation.Hermite,
            3 => MdlAnimInterpolation.Bezier,
            _ => MdlAnimInterpolation.Linear,
        };
    }

    private static MdlTrackType ReadTrackInterpolation(object block)
    {
        int interpolation = ReadIntMember(block, "InterpolationType");
        return interpolation switch
        {
            0 => MdlTrackType.NoInterp,
            1 => MdlTrackType.Linear,
            2 => MdlTrackType.Hermite,
            3 => MdlTrackType.Bezier,
            _ => MdlTrackType.Linear,
        };
    }

    private static float ConvertAlpha(object value)
    {
        return value switch
        {
            short alphaShort => Math.Clamp(alphaShort / 32767.0f, 0.0f, 1.0f),
            ushort alphaUShort => Math.Clamp(alphaUShort / 65535.0f, 0.0f, 1.0f),
            float alphaFloat => Math.Clamp(alphaFloat, 0.0f, 1.0f),
            _ => Math.Clamp(Convert.ToSingle(value), 0.0f, 1.0f),
        };
    }

    private static C3Color ConvertColor(object value)
    {
        if (value is C3Color color)
            return color;

        if (value is Vector3 vector)
            return new C3Color(vector.X, vector.Y, vector.Z);

        return new C3Color(
            ReadSingleComponent(value, "R", "X"),
            ReadSingleComponent(value, "G", "Y"),
            ReadSingleComponent(value, "B", "Z"));
    }

    private static C3Vector ConvertVector(object value)
    {
        if (value is C3Vector vector)
            return vector;

        if (value is Vector3 numerics)
            return new C3Vector(numerics.X, numerics.Y, numerics.Z);

        return new C3Vector(
            ReadSingleComponent(value, "X"),
            ReadSingleComponent(value, "Y"),
            ReadSingleComponent(value, "Z"));
    }

    private static C4Quaternion ConvertQuaternion(object value)
    {
        if (value is C4Quaternion quaternion)
            return quaternion;

        if (value is Quaternion numerics)
            return new C4Quaternion(numerics.X, numerics.Y, numerics.Z, numerics.W);

        return new C4Quaternion(
            ReadSingleComponent(value, "X"),
            ReadSingleComponent(value, "Y"),
            ReadSingleComponent(value, "Z"),
            ReadSingleComponent(value, "W"));
    }

    private static float ReadSingleComponent(object instance, string primary, string? fallback = null)
    {
        if (TryGetMemberValue(instance, primary, out object? primaryValue) && primaryValue != null)
            return Convert.ToSingle(primaryValue);

        if (fallback != null && TryGetMemberValue(instance, fallback, out object? fallbackValue) && fallbackValue != null)
            return Convert.ToSingle(fallbackValue);

        throw new InvalidDataException($"Could not read '{primary}' from '{instance.GetType().FullName}'.");
    }

    private struct ParsedVertexData
    {
        public Vector3 Position;
        public Vector3 Normal;
        public float TextureCoord0X;
        public float TextureCoord0Y;
        public float TextureCoord1X;
        public float TextureCoord1Y;
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

    private struct ParsedTextureCoordLookupData
    {
        public int CoordId;
    }

    private sealed class ParsedSequenceData
    {
        public string Name { get; init; } = string.Empty;
        public int StartFrame { get; init; }
        public int EndFrame { get; init; }
    }

    private sealed class ParsedTransparencyData
    {
        public MdlAnimInterpolation Interpolation { get; init; }
        public int GlobalSeqId { get; init; } = -1;
        public List<MdlAnimKey<float>> Keys { get; } = new();
    }

    private struct ParsedTransparencyLookupData
    {
        public int TransparencyId;
    }

    private sealed class ParsedColorData
    {
        public C3Color StaticColor { get; init; } = new C3Color(1.0f, 1.0f, 1.0f);
        public float StaticAlpha { get; init; } = 1.0f;
        public MdlAnimInterpolation ColorInterpolation { get; init; }
        public int ColorGlobalSeqId { get; init; } = -1;
        public List<MdlAnimKey<C3Color>> ColorKeys { get; } = new();
        public MdlAnimInterpolation AlphaInterpolation { get; init; }
        public int AlphaGlobalSeqId { get; init; } = -1;
        public List<MdlAnimKey<float>> AlphaKeys { get; } = new();
    }

    private struct ParsedUvAnimationLookupData
    {
        public int TextureAnimationId;
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
        // Reference semantics (wowdev):
        //   0=Opaque, 1=AlphaKey, 2=Alpha, 3=NoAlphaAdd, 4=Add, 5=Mod, 6=Mod2x, 7=BlendAdd.
        // The local MDX renderer does not expose distinct NoAlphaAdd / BlendAdd states, so collapse those to
        // the nearest additive family deliberately instead of shifting every later mode by one.
        return blendMode switch
        {
            0 => MdlTexOp.Load,          // Opaque
            1 => MdlTexOp.Transparent,   // AlphaKey (cutout)
            2 => MdlTexOp.Blend,         // Alpha blend
            3 => MdlTexOp.Add,           // NoAlphaAdd
            4 => MdlTexOp.Add,           // Add
            5 => MdlTexOp.Modulate,      // Mod
            6 => MdlTexOp.Modulate2X,    // Mod2x
            7 => MdlTexOp.AddAlpha,      // BlendAdd
            _ => MdlTexOp.Blend,
        };
    }

    private static MdlGeoFlags MapLayerFlags(ushort renderFlags, TextureFlags textureFlags, int coordId)
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

        if (coordId < 0)
            flags |= MdlGeoFlags.SphereEnvMap;

        if (textureFlags.HasFlag(TextureFlags.Flag_0x1_WrapX))
            flags |= MdlGeoFlags.WrapWidth;
        if (textureFlags.HasFlag(TextureFlags.Flag_0x2_WrapY))
            flags |= MdlGeoFlags.WrapHeight;

        return flags;
    }
}
