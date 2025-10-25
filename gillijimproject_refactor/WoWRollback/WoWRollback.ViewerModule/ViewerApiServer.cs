using System.Diagnostics;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace WoWRollback.ViewerModule;

public sealed class ViewerApiServer : IDisposable
{
    private HttpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _serverTask;

    private readonly object _jobsLock = new();
    private readonly Dictionary<string, Job> _jobs = new();

    public void Start(int port = 8081)
    {
        if (_listener != null) throw new InvalidOperationException("API server already running");

        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
        try
        {
            _listener.Start();
        }
        catch (HttpListenerException ex)
        {
            throw new InvalidOperationException($"Failed to start API server on port {port}: {ex.Message}", ex);
        }

        _cts = new CancellationTokenSource();
        _serverTask = Task.Run(() => ServeAsync(_cts.Token));
    }

    public void Stop()
    {
        if (_listener == null) return;
        _cts?.Cancel();
        try { _serverTask?.Wait(TimeSpan.FromSeconds(5)); } catch { }
        _listener.Stop();
        _listener.Close();
        _listener = null;
        _serverTask = null;
    }

    public void Dispose()
    {
        Stop();
        _cts?.Dispose();
        _cts = null;
    }

    private async Task ServeAsync(CancellationToken token)
    {
        if (_listener == null) return;
        while (!token.IsCancellationRequested && _listener.IsListening)
        {
            try
            {
                var ctx = await _listener.GetContextAsync();
                _ = Task.Run(() => HandleAsync(ctx), token);
            }
            catch (HttpListenerException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ViewerApiServer] Accept error: {ex.Message}");
            }
        }
    }

    private async Task HandleAsync(HttpListenerContext ctx)
    {
        var req = ctx.Request;
        var res = ctx.Response;

        // CORS
        SetCors(res);
        if (req.HttpMethod.Equals("OPTIONS", StringComparison.OrdinalIgnoreCase))
        {
            res.StatusCode = 200;
            res.Close();
            return;
        }

        try
        {
            var path = req.Url?.AbsolutePath ?? "/";

            if (req.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) && path.Equals("/api/defaults", StringComparison.OrdinalIgnoreCase))
            {
                var d = DefaultPathsService.Discover();
                await JsonAsync(res, new { wdt = d.Wdt, crosswalkDir = d.CrosswalkDir, lkDbcDir = d.LkDbcDir });
                return;
            }

            if (req.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) && path.Equals("/api/presets", StringComparison.OrdinalIgnoreCase))
            {
                var payload = BuildSeedPresets();
                await JsonAsync(res, payload);
                return;
            }

            if (req.HttpMethod.Equals("POST", StringComparison.OrdinalIgnoreCase) && path.Equals("/api/build/alpha-to-lk", StringComparison.OrdinalIgnoreCase))
            {
                using var sr = new StreamReader(req.InputStream, req.ContentEncoding);
                var body = await sr.ReadToEndAsync();
                var buildReq = JsonSerializer.Deserialize<BuildRequest>(body) ?? new BuildRequest();
                var job = StartAlphaToLkJob(buildReq);
                await JsonAsync(res, new { jobId = job.Id });
                return;
            }

            if (req.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) && path.StartsWith("/api/jobs/", StringComparison.OrdinalIgnoreCase) && !path.EndsWith("/events", StringComparison.OrdinalIgnoreCase))
            {
                var id = path.Split('/', StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
                if (string.IsNullOrWhiteSpace(id)) { await TextAsync(res, 400, "Bad Request"); return; }
                Job? job; lock (_jobsLock) _jobs.TryGetValue(id, out job);
                if (job == null) { await TextAsync(res, 404, "Not Found"); return; }
                await JsonAsync(res, new { id = job.Id, status = job.Status.ToString(), startedAt = job.StartedAt, finishedAt = job.FinishedAt, output = new { wdt = job.OutputWdt, lkOut = job.OutputLkDir } });
                return;
            }

            if (req.HttpMethod.Equals("GET", StringComparison.OrdinalIgnoreCase) && path.EndsWith("/events", StringComparison.OrdinalIgnoreCase))
            {
                var parts = path.Split('/', StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 3) { await TextAsync(res, 400, "Bad Request"); return; }
                var id = parts[^2];
                Job? job; lock (_jobsLock) _jobs.TryGetValue(id, out job);
                if (job == null) { await TextAsync(res, 404, "Not Found"); return; }

                res.StatusCode = 200;
                res.ContentType = "text/event-stream";
                res.SendChunked = true;
                res.Headers["Cache-Control"] = "no-cache";
                SetCors(res);

                await SseAsync(res, new { type = "status", status = job.Status.ToString() });

                var next = 0;
                while (job.IsActive)
                {
                    List<string>? batch = null;
                    lock (job.LogsLock)
                    {
                        if (next < job.Logs.Count)
                        {
                            batch = job.Logs.Skip(next).ToList();
                            next = job.Logs.Count;
                        }
                        else
                        {
                            job.LogsEvent.Reset();
                        }
                    }

                    if (batch != null && batch.Count > 0)
                    {
                        foreach (var line in batch) await SseAsync(res, new { type = "log", message = line });
                    }
                    else
                    {
                        job.LogsEvent.Wait(TimeSpan.FromSeconds(2));
                        await RawAsync(res, ":keep-alive\n\n");
                    }
                }

                await SseAsync(res, new { type = "status", status = job.Status.ToString() });
                return;
            }

            await TextAsync(res, 404, "Not Found");
        }
        catch (Exception ex)
        {
            await TextAsync(res, 500, $"Internal Server Error: {ex.Message}");
        }
    }

    private Job StartAlphaToLkJob(BuildRequest request)
    {
        var defaults = DefaultPathsService.Discover();
        request ??= new BuildRequest();
        request.Paths ??= new BuildRequest.PathsModel();
        request.UniqueId ??= new BuildRequest.UniqueIdModel();
        request.Terrain ??= new BuildRequest.TerrainModel();
        request.Mapping ??= new BuildRequest.MappingModel();

        if (request.UseDefaults)
        {
            if (string.IsNullOrWhiteSpace(request.Paths.Wdt)) request.Paths.Wdt = defaults.Wdt;
            if (string.IsNullOrWhiteSpace(request.Paths.CrosswalkDir)) request.Paths.CrosswalkDir = defaults.CrosswalkDir;
            if (string.IsNullOrWhiteSpace(request.Paths.LkDbcDir)) request.Paths.LkDbcDir = defaults.LkDbcDir;
        }

        var repoRoot = defaults.RepoRoot ?? Directory.GetCurrentDirectory();
        var cliDir = Path.Combine(repoRoot, "WoWRollback", "WoWRollback.Cli");
        var cliProj = FindFirstCsproj(cliDir) ?? Path.Combine(cliDir, "WoWRollback.Cli.csproj");

        request.Paths.OutDir ??= Path.Combine(repoRoot, "rollback_out");
        var mapNameGuess = GuessMapNameFromWdt(request.Paths.Wdt) ?? "Map";
        request.Paths.LkOutDir ??= Path.Combine(request.Paths.OutDir, "lk_adts", "World", "Maps", mapNameGuess);

        var job = new Job
        {
            Id = Guid.NewGuid().ToString("N"),
            StartedAt = DateTimeOffset.UtcNow,
            Status = JobStatus.Queued,
            OutputWdt = request.Paths.Wdt,
            OutputLkDir = request.Paths.LkOutDir
        };

        lock (_jobsLock) _jobs[job.Id] = job;

        var psi = new ProcessStartInfo
        {
            FileName = "dotnet",
            Arguments = BuildDotnetRunArgs(cliProj, BuildCliArgs(request)),
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        var proc = new Process { StartInfo = psi, EnableRaisingEvents = true };
        job.Process = proc;

        proc.OutputDataReceived += (_, e) => { if (e.Data != null) job.AddLog(e.Data); };
        proc.ErrorDataReceived += (_, e) => { if (e.Data != null) job.AddLog(e.Data); };
        proc.Exited += (_, __) =>
        {
            job.Status = proc.ExitCode == 0 ? JobStatus.Succeeded : JobStatus.Failed;
            job.FinishedAt = DateTimeOffset.UtcNow;
            job.SignalLogs();
        };

        job.Status = JobStatus.Running;
        proc.Start();
        proc.BeginOutputReadLine();
        proc.BeginErrorReadLine();

        return job;
    }

    private static string BuildDotnetRunArgs(string cliProj, string cliArgs)
    {
        var sb = new StringBuilder();
        sb.Append("run --project ");
        sb.Append('"').Append(cliProj).Append('"');
        sb.Append(' ');
        sb.Append("-- ");
        sb.Append(cliArgs);
        return sb.ToString();
    }

    private static string BuildCliArgs(BuildRequest r)
    {
        var args = new List<string> { "alpha-to-lk" };

        void Add(string name, string? value)
        {
            if (string.IsNullOrWhiteSpace(value)) return;
            args.Add(name); args.Add(value);
        }

        args.Add("--input"); args.Add(r.Paths!.Wdt!);
        args.Add("--max-uniqueid"); args.Add((r.UniqueId!.Max <= 0 ? 125000 : r.UniqueId!.Max).ToString());
        Add("--out", r.Paths!.OutDir);
        args.Add("--export-lk-adts");
        Add("--lk-out", r.Paths!.LkOutDir);
        Add("--crosswalk-dir", r.Paths!.CrosswalkDir);
        Add("--lk-dbc-dir", r.Paths!.LkDbcDir);

        if (r.Terrain!.FixHoles) args.Add("--fix-holes");
        if (r.Terrain!.DisableMcsh) args.Add("--disable-mcsh");
        if (r.Mapping!.StrictAreaId) args.Add("--strict-areaid");
        if (r.Mapping!.ChainVia060) args.Add("--chain-via-060");

        for (int i = 0; i < args.Count; i++)
        {
            if (i > 0 && args[i - 1].StartsWith("--") && args[i][0] != '-')
            {
                if (args[i].IndexOf(' ') >= 0 || args[i].Contains('\\') || args[i].Contains('/'))
                    args[i] = '"' + args[i] + '"';
            }
        }

        return string.Join(' ', args);
    }

    private static async Task JsonAsync(HttpListenerResponse res, object payload)
    {
        var json = JsonSerializer.Serialize(payload);
        var data = Encoding.UTF8.GetBytes(json);
        res.StatusCode = 200;
        res.ContentType = "application/json";
        res.ContentLength64 = data.Length;
        await res.OutputStream.WriteAsync(data, 0, data.Length);
    }

    private static async Task TextAsync(HttpListenerResponse res, int code, string text)
    {
        res.StatusCode = code;
        res.ContentType = "text/plain";
        var data = Encoding.UTF8.GetBytes(text);
        res.ContentLength64 = data.Length;
        await res.OutputStream.WriteAsync(data, 0, data.Length);
    }

    private static async Task SseAsync(HttpListenerResponse res, object payload)
    {
        var json = JsonSerializer.Serialize(payload);
        await RawAsync(res, $"data: {json}\n\n");
    }

    private static async Task RawAsync(HttpListenerResponse res, string text)
    {
        var data = Encoding.UTF8.GetBytes(text);
        await res.OutputStream.WriteAsync(data, 0, data.Length);
        await res.OutputStream.FlushAsync();
    }

    private static void SetCors(HttpListenerResponse res)
    {
        res.Headers["Access-Control-Allow-Origin"] = "http://localhost:8080";
        res.Headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS";
        res.Headers["Access-Control-Allow-Headers"] = "Content-Type";
    }

    private static string? GuessMapNameFromWdt(string? wdtPath)
    {
        if (string.IsNullOrWhiteSpace(wdtPath)) return null;
        var name = Path.GetFileNameWithoutExtension(wdtPath);
        return string.IsNullOrWhiteSpace(name) ? null : name;
    }

    private static string? FindFirstCsproj(string dir)
    {
        try
        {
            var files = Directory.EnumerateFiles(dir, "*.csproj", SearchOption.TopDirectoryOnly);
            return files.FirstOrDefault();
        }
        catch { return null; }
    }

    private object BuildSeedPresets()
    {
        var d = DefaultPathsService.Discover();
        var list = new List<object>();

        object Make(string name, string? wdt)
        {
            return new
            {
                name,
                request = new
                {
                    useDefaults = false,
                    uniqueId = new { max = 125000, buryDepth = -5000 },
                    terrain = new { fixHoles = true, disableMcsh = true },
                    mapping = new { strictAreaId = true, chainVia060 = false },
                    paths = new { wdt = wdt, crosswalkDir = d.CrosswalkDir, lkDbcDir = d.LkDbcDir, outDir = "rollback_out", lkOutDir = (string?)null }
                }
            };
        }

        if (!string.IsNullOrWhiteSpace(d.Wdt)) list.Add(Make("Auto (Discovered)", d.Wdt));
        foreach (var map in new[] { "Kalimdor", "Azeroth" })
        {
            var wdt = TrySiblingMapWdt(d.Wdt, map);
            if (!string.IsNullOrWhiteSpace(wdt)) list.Add(Make($"Alpha 0.5.3 {map} (Defaults)", wdt));
        }
        return list;
    }

    private static string? TrySiblingMapWdt(string? discoveredWdt, string mapName)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(discoveredWdt)) return null;
            var dir = Path.GetDirectoryName(discoveredWdt);
            if (string.IsNullOrWhiteSpace(dir)) return null;
            var mapsDir = Directory.GetParent(dir)?.Parent;
            if (mapsDir == null) return null;
            var target = Path.Combine(mapsDir.FullName, mapName, mapName + ".wdt");
            return File.Exists(target) ? target : null;
        }
        catch { return null; }
    }

    private enum JobStatus { Queued, Running, Succeeded, Failed }

    private sealed class Job
    {
        public string Id { get; set; } = string.Empty;
        public JobStatus Status { get; set; } = JobStatus.Queued;
        public DateTimeOffset StartedAt { get; set; }
        public DateTimeOffset? FinishedAt { get; set; }
        public Process? Process { get; set; }
        public string? OutputWdt { get; set; }
        public string? OutputLkDir { get; set; }

        public bool IsActive => Status == JobStatus.Queued || Status == JobStatus.Running;

        public readonly List<string> Logs = new();
        public readonly object LogsLock = new();
        public readonly System.Threading.ManualResetEventSlim LogsEvent = new(false);

        public void AddLog(string line)
        {
            lock (LogsLock)
            {
                Logs.Add(line);
                LogsEvent.Set();
            }
        }

        public void SignalLogs() => LogsEvent.Set();
    }

    private sealed class BuildRequest
    {
        [JsonPropertyName("useDefaults")] public bool UseDefaults { get; set; }
        [JsonPropertyName("uniqueId")] public UniqueIdModel? UniqueId { get; set; }
        [JsonPropertyName("terrain")] public TerrainModel? Terrain { get; set; }
        [JsonPropertyName("mapping")] public MappingModel? Mapping { get; set; }
        [JsonPropertyName("paths")] public PathsModel? Paths { get; set; }

        public sealed class UniqueIdModel { public int Max { get; set; } public int BuryDepth { get; set; } = -5000; }
        public sealed class TerrainModel { public bool FixHoles { get; set; } public bool DisableMcsh { get; set; } }
        public sealed class MappingModel { public bool StrictAreaId { get; set; } = true; public bool ChainVia060 { get; set; } }
        public sealed class PathsModel { public string? Wdt { get; set; } public string? CrosswalkDir { get; set; } public string? LkDbcDir { get; set; } public string? OutDir { get; set; } public string? LkOutDir { get; set; } }
    }
}
