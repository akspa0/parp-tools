using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Channels;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Spec 256 P2b: native M2 textures are read and decoded on background threads; the render thread only
/// uploads (or takes an already-uploaded shared texture from <see cref="M2TextureCache"/>). Used for
/// world-streamed models only (<c>deferInitialTextureLoads</c>); other callers keep synchronous loads.
/// </summary>
internal static class M2TextureStreamer
{
    /// <summary>One texture candidate of a section, in the order the synchronous loader would try it.</summary>
    internal readonly record struct Candidate(string TexturePath, bool ClampS, bool ClampT, int UvSet, bool GeneratedTexCoord, uint FallbackSlot);

    /// <summary>A section's remaining candidates, handed to a worker.</summary>
    internal sealed class Request
    {
        public required M2Renderer Owner { get; init; }
        public required int Generation { get; init; }
        public required int SectionListIndex { get; init; }
        public required IReadOnlyList<Candidate> Candidates { get; init; }
        public bool ForceDecode { get; init; }

        /// <summary>
        /// 0 = the material's own candidates; 1 = replaceable fallback slot 11; 2 = slot 1. Fallback slots are
        /// resolved only after the previous stage failed, as in the synchronous loader: resolving them is
        /// expensive (naming-convention probes and a scan of every .blp in the data source).
        /// </summary>
        public int Stage { get; init; }
    }

    /// <summary>The first candidate that resolved (and, unless another model already decoded it, its pixels).</summary>
    internal sealed class Result
    {
        public required Request Request { get; init; }
        public int CandidateIndex { get; init; } = -1;
        public string ResolvedPath { get; init; } = string.Empty;
        public byte[]? Pixels { get; init; }

        /// <summary>Spec 256 P2c: DXT levels to upload as-is (no pixels decoded).</summary>
        public BlpCompressedTexture? Compressed { get; init; }
        public int Width { get; init; }
        public int Height { get; init; }
        public bool ResolvedOnly { get; init; }

        /// <summary>Frames a shared result has waited for the other model's upload (bounded by the owner).</summary>
        public int WaitFrames { get; init; }
    }

    private static readonly Channel<Request> Queue = Channel.CreateUnbounded<Request>(new UnboundedChannelOptions { SingleReader = false });
    private static readonly ConcurrentQueue<Result> Completed = new();
    private static readonly ConcurrentQueue<Result> Deferred = new();

    // Resolved files already decoded (or being decoded) by some request, so a second model sharing the
    // texture skips the decode and takes the uploaded texture instead. A stale entry only costs one retry.
    private static readonly ConcurrentDictionary<string, byte> DecodedKeys = new(StringComparer.OrdinalIgnoreCase);

    private static int _workersStarted;
    private static long _queued;
    private static long _decoded;
    private static long _sharedHits;
    private static long _uploaded;
    private static double _decodeMsTotal;
    private static double _uploadMsTotal;
    private static readonly object StatsLock = new();

    public static int PendingCount => (int)(Interlocked.Read(ref _queued) - Interlocked.Read(ref _decoded) - Interlocked.Read(ref _sharedHits)) + Completed.Count;

    /// <summary>Counters for Runtime Stats: queued, decoded off-thread, shared hits, uploaded, average ms.</summary>
    public static (long Queued, long Decoded, long SharedHits, long Uploaded, double AvgDecodeMs, double AvgUploadMs) Stats
    {
        get
        {
            lock (StatsLock)
            {
                long decoded = Interlocked.Read(ref _decoded);
                long uploaded = Interlocked.Read(ref _uploaded);
                return (Interlocked.Read(ref _queued), decoded, Interlocked.Read(ref _sharedHits), uploaded,
                    decoded == 0 ? 0 : _decodeMsTotal / decoded,
                    uploaded == 0 ? 0 : _uploadMsTotal / uploaded);
            }
        }
    }

    public static void Enqueue(Request request)
    {
        EnsureWorkers();
        Interlocked.Increment(ref _queued);
        Queue.Writer.TryWrite(request);
    }

    /// <summary>
    /// Render thread, once per frame: hands finished results to their renderers within the time budget
    /// (at least one per call, so textures always progress).
    /// </summary>
    public static int ProcessCompleted(double budgetMs)
    {
        // Results deferred last frame (waiting for another model's upload) get another look now.
        while (Deferred.TryDequeue(out Result? waiting))
            Completed.Enqueue(waiting);

        if (Completed.IsEmpty)
            return 0;

        long start = Stopwatch.GetTimestamp();
        int processed = 0;
        while (Completed.TryDequeue(out Result? result))
        {
            long uploadStart = Stopwatch.GetTimestamp();
            bool uploaded = result.Request.Owner.CompleteStreamedTexture(result);
            if (uploaded)
            {
                double ms = Stopwatch.GetElapsedTime(uploadStart).TotalMilliseconds;
                lock (StatsLock)
                    _uploadMsTotal += ms;
                Interlocked.Increment(ref _uploaded);
            }

            processed++;
            if (Stopwatch.GetElapsedTime(start).TotalMilliseconds >= budgetMs)
                break;
        }

        return processed;
    }

    internal static string DecodedKey(object? dataSource, string resolvedPath, bool clampS, bool clampT)
        => $"{(dataSource is null ? 0 : System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(dataSource))}|{resolvedPath.Replace('/', '\\').ToLowerInvariant()}|{(clampS ? 1 : 0)}{(clampT ? 1 : 0)}";

    internal static void ForgetDecoded(string decodedKey) => DecodedKeys.TryRemove(decodedKey, out _);

    /// <summary>Retry a shared result next frame (another model's decode of the same file is not uploaded yet).</summary>
    internal static void DeferShared(Result result)
        => Deferred.Enqueue(new Result
        {
            Request = result.Request,
            CandidateIndex = result.CandidateIndex,
            ResolvedPath = result.ResolvedPath,
            ResolvedOnly = true,
            WaitFrames = result.WaitFrames + 1,
        });

    private static void EnsureWorkers()
    {
        if (Interlocked.Exchange(ref _workersStarted, 1) == 1)
            return;

        int workers = Math.Clamp(Environment.ProcessorCount / 4, 2, 4);
        for (int i = 0; i < workers; i++)
            _ = Task.Run(WorkerLoopAsync);
    }

    private static async Task WorkerLoopAsync()
    {
        await foreach (Request request in Queue.Reader.ReadAllAsync().ConfigureAwait(false))
        {
            Result result;
            try
            {
                result = Resolve(request);
            }
            catch (Exception ex)
            {
                ViewerLog.Debug(ViewerLog.Category.Mdx, $"[M2] Streamed texture failed for {request.Owner.SourceModelPath}: {ex.Message}");
                result = new Result { Request = request };
            }

            Completed.Enqueue(result);
        }
    }

    // Same order as M2Renderer.TryLoadTexture: per candidate, a loose PNG override first, then the BLP
    // bytes; a candidate whose source fails to decode falls through, exactly like a failed load.
    private static Result Resolve(Request request)
    {
        M2Renderer owner = request.Owner;
        for (int index = 0; index < request.Candidates.Count; index++)
        {
            Candidate candidate = request.Candidates[index];
            if (owner.TryResolvePngOffThread(candidate.TexturePath, out string pngPath)
                && TryShareOrDecode(request, index, candidate, pngPath, isPng: true, blpBytes: null, out Result? pngResult))
            {
                return pngResult!;
            }

            if (owner.TryReadTextureBytesOffThread(candidate.TexturePath, out byte[]? blpBytes, out string resolvedPath)
                && TryShareOrDecode(request, index, candidate, resolvedPath, isPng: false, blpBytes, out Result? blpResult))
            {
                return blpResult!;
            }
        }

        Interlocked.Increment(ref _decoded);
        return new Result { Request = request };
    }

    private static bool TryShareOrDecode(Request request, int index, Candidate candidate, string resolvedPath, bool isPng, byte[]? blpBytes, out Result? result)
    {
        string key = DecodedKey(request.Owner.TextureDataSource, resolvedPath, candidate.ClampS, candidate.ClampT);
        if (!request.ForceDecode && !DecodedKeys.TryAdd(key, 0))
        {
            Interlocked.Increment(ref _sharedHits);
            result = new Result { Request = request, CandidateIndex = index, ResolvedPath = resolvedPath, ResolvedOnly = true };
            return true;
        }

        DecodedKeys.TryAdd(key, 0);
        long decodeStart = Stopwatch.GetTimestamp();
        if (request.Owner.TryDecodeTextureOffThread(resolvedPath, isPng, blpBytes, out byte[]? pixels, out int width, out int height, out BlpCompressedTexture? compressed))
        {
            double ms = Stopwatch.GetElapsedTime(decodeStart).TotalMilliseconds;
            lock (StatsLock)
                _decodeMsTotal += ms;
            Interlocked.Increment(ref _decoded);
            result = new Result { Request = request, CandidateIndex = index, ResolvedPath = resolvedPath, Pixels = pixels, Width = width, Height = height, Compressed = compressed };
            return true;
        }

        DecodedKeys.TryRemove(key, out _);
        result = null;
        return false;
    }
}
