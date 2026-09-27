using SereniaBLPLib;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WoWViewer.DataSources;
using WoWViewer.Logging;
using WowViewer.Core.Runtime.M2;

namespace WoWViewer.Rendering;

// Spec 256 P2b: off-render-thread texture reads and decodes for world-streamed native M2s. The resolution
// and decode steps are the synchronous loader's own (TryResolveImagePath, TryReadTextureBytes, BLP/PNG
// decode), called from worker threads; everything touching GL or the caches stays on the render thread.
public sealed partial class M2Renderer
{
    internal IDataSource? TextureDataSource => _dataSource;

    // Render thread (constructor / reload). Per section: the first candidate is taken from the caches when
    // it is already loaded (as the synchronous path would); otherwise the section's candidates go to a worker
    // in the same order the synchronous loop tries them. No candidate, or none loads → missing texture.
    private void StreamSectionTextures()
    {
        for (int sectionIndex = 0; sectionIndex < _sections.Count; sectionIndex++)
        {
            SectionBuffers section = _sections[sectionIndex];
            section.AlphaCutout = section.Material.BlendMode == WowViewer.Core.M2.M2BlendMode.AlphaKey;
            List<M2TextureStreamer.Candidate> candidates = BuildStreamingCandidates(section.Material);
            if (candidates.Count == 0)
            {
                ApplyMissingTexture(section);
                continue;
            }

            M2TextureStreamer.Candidate first = candidates[0];
            string cacheKey = BuildTextureCacheKey(first.TexturePath, first.ClampS, first.ClampT);
            if (_loadedTextureCache.TryGetValue(cacheKey, out uint cachedId) && cachedId != 0)
            {
                ApplySectionTexture(section, first, cachedId);
                continue;
            }

            if (M2TextureCache.TryAcquire(_dataSource, BuildSharedRequestKey(cacheKey), out uint sharedId))
            {
                _loadedTextureCache[cacheKey] = sharedId;
                KeepOneReference(sharedId);
                ApplySectionTexture(section, first, sharedId);
                continue;
            }

            section.TexturePending = true;
            M2TextureStreamer.Enqueue(new M2TextureStreamer.Request
            {
                Owner = this,
                Generation = _textureGeneration,
                SectionListIndex = sectionIndex,
                Candidates = candidates,
            });
        }
    }

    // Same candidates, order and clamp rules as TryLoadMaterialTexture.
    private List<M2TextureStreamer.Candidate> BuildStreamingCandidates(M2StaticRenderMaterial material)
    {
        var candidates = new List<M2TextureStreamer.Candidate>();
        foreach ((string? TexturePath, uint ReplaceableId, uint TextureFlags, int UvSet, bool GeneratedTexCoord) candidate in EnumerateTextureCandidates(material))
        {
            string? resolvedPath = null;
            if (candidate.ReplaceableId != 0)
                resolvedPath = ResolveReplaceableTexture(candidate.ReplaceableId);
            if (string.IsNullOrWhiteSpace(resolvedPath))
                resolvedPath = candidate.TexturePath;
            if (string.IsNullOrWhiteSpace(resolvedPath))
                continue;

            bool clampS = (candidate.TextureFlags & 0x1u) == 0;
            bool clampT = (candidate.TextureFlags & 0x2u) == 0;
            candidates.Add(new M2TextureStreamer.Candidate(resolvedPath, clampS, clampT, candidate.UvSet, candidate.GeneratedTexCoord));
        }

        return candidates;
    }

    /// <summary>Render thread: applies a finished request. Returns true when it uploaded new pixels.</summary>
    internal bool CompleteStreamedTexture(M2TextureStreamer.Result result)
    {
        M2TextureStreamer.Request request = result.Request;
        bool stale = _disposed || _gl == null || request.Generation != _textureGeneration || request.SectionListIndex >= _sections.Count;
        if (stale)
        {
            if ((result.Pixels != null || result.Compressed != null) && result.CandidateIndex >= 0)
            {
                M2TextureStreamer.Candidate dropped = request.Candidates[result.CandidateIndex];
                M2TextureStreamer.ForgetDecoded(M2TextureStreamer.DecodedKey(_dataSource, result.ResolvedPath, dropped.ClampS, dropped.ClampT));
            }

            return false;
        }

        SectionBuffers section = _sections[request.SectionListIndex];
        if (result.CandidateIndex < 0)
        {
            // No candidate loaded: the missing-texture placeholder, as in TryLoadMaterialTexture.
            ApplyMissingTexture(section);
            return false;
        }

        M2TextureStreamer.Candidate candidate = request.Candidates[result.CandidateIndex];
        string cacheKey = BuildTextureCacheKey(candidate.TexturePath, candidate.ClampS, candidate.ClampT);
        string requestKey = BuildSharedRequestKey(cacheKey);

        if (TryAcquireResolved(result.ResolvedPath, candidate.ClampS, candidate.ClampT, requestKey, out uint sharedId))
        {
            FinishSectionTexture(section, candidate, cacheKey, result.ResolvedPath, sharedId);
            return false;
        }

        if (result.ResolvedOnly)
        {
            // Another model is decoding this file; wait for its upload a bounded number of frames, then
            // decode it ourselves so a dropped or released upload cannot leave this section pending.
            if (result.WaitFrames < 60)
            {
                M2TextureStreamer.DeferShared(result);
            }
            else
            {
                M2TextureStreamer.Enqueue(new M2TextureStreamer.Request
                {
                    Owner = this,
                    Generation = request.Generation,
                    SectionListIndex = request.SectionListIndex,
                    Candidates = request.Candidates,
                    ForceDecode = true,
                });
            }

            return false;
        }

        uint textureId = result.Compressed is { } compressed
            ? UploadCompressedTexture(compressed, candidate.ClampS, candidate.ClampT)
            : UploadTexture(result.Pixels!, (uint)result.Width, (uint)result.Height, candidate.ClampS, candidate.ClampT);
        if (textureId == 0)
        {
            ApplyMissingTexture(section);
            return false;
        }

        M2TextureCache.Add(_dataSource, textureId, requestKey, BuildSharedResolvedKey(result.ResolvedPath, candidate.ClampS, candidate.ClampT));
        FinishSectionTexture(section, candidate, cacheKey, result.ResolvedPath, textureId);
        return true;
    }

    private void FinishSectionTexture(SectionBuffers section, M2TextureStreamer.Candidate candidate, string cacheKey, string resolvedPath, uint textureId)
    {
        _loadedTextureCache[cacheKey] = textureId;
        _loadedTextureCache[BuildTextureCacheKey(resolvedPath, candidate.ClampS, candidate.ClampT)] = textureId;
        KeepOneReference(textureId);
        ApplySectionTexture(section, candidate, textureId);
    }

    private static void ApplySectionTexture(SectionBuffers section, M2TextureStreamer.Candidate candidate, uint textureId)
    {
        section.TextureId = textureId;
        section.HasTexture = true;
        section.UvSet = candidate.UvSet;
        section.GeneratedTexCoord = candidate.GeneratedTexCoord;
        section.TexturePending = false;
    }

    // ---- worker-thread helpers: read-only use of _dataSource / _modelDir, no GL, no caches ----

    internal bool TryResolvePngOffThread(string texturePath, out string pngPath)
        => TryResolveImagePath(texturePath, ".png", out pngPath);

    internal bool TryReadTextureBytesOffThread(string texturePath, out byte[]? bytes, out string resolvedPath)
        => TryReadTextureBytes(texturePath, out bytes, out resolvedPath) && bytes != null && bytes.Length > 0;

    internal bool TryDecodeTextureOffThread(string resolvedPath, bool isPng, byte[]? blpBytes, out byte[]? pixels, out int width, out int height, out BlpCompressedTexture? compressed)
    {
        pixels = null;
        width = 0;
        height = 0;
        compressed = null;

        // Spec 256 P2c: DXT BLPs need no decode at all.
        if (!isPng && blpBytes != null && BlpCompressedTexture.TryCreate(blpBytes, resolvedPath) is { } dxt)
        {
            compressed = dxt;
            return true;
        }

        try
        {
            using Image<Rgba32> image = isPng
                ? SixLabors.ImageSharp.Image.Load<Rgba32>(resolvedPath)
                : DecodeBlpImage(blpBytes!);
            pixels = new byte[image.Width * image.Height * 4];
            image.CopyPixelDataTo(pixels);
            width = image.Width;
            height = image.Height;
            return true;
        }
        catch (Exception ex)
        {
            ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] Failed to decode texture '{resolvedPath}': {ex.Message}");
            return false;
        }
    }

    private static Image<Rgba32> DecodeBlpImage(byte[] blpBytes)
    {
        using MemoryStream memoryStream = new(blpBytes, writable: false);
        using BlpFile blp = new(memoryStream);
        return blp.GetImage(0);
    }
}
