namespace MdxViewer.DataSources;

/// <summary>
/// Downloads and caches the community listfile from wowdev/wow-listfile GitHub releases.
/// Cached in the user's local app data folder to avoid re-downloading every launch.
/// </summary>
public static class ListfileDownloader
{
    private const string DownloadUrl =
        "https://github.com/wowdev/wow-listfile/releases/latest/download/community-listfile-withcapitals.csv";

    private static readonly string CacheDir =
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer");

    private static readonly string CachedPath =
        Path.Combine(CacheDir, "community-listfile-withcapitals.csv");

    /// <summary>
    /// Get the path to the community listfile, downloading if not cached or stale (>7 days).
    /// </summary>
    public static async Task<string?> GetListfilePathAsync(bool forceDownload = false)
    {
        Directory.CreateDirectory(CacheDir);

        bool needsDownload = forceDownload || !File.Exists(CachedPath);

        // Re-download if older than 7 days
        if (!needsDownload && File.Exists(CachedPath))
        {
            var age = DateTime.UtcNow - File.GetLastWriteTimeUtc(CachedPath);
            if (age.TotalDays > 7)
                needsDownload = true;
        }

        if (needsDownload)
        {
            Console.WriteLine($"[Listfile] Downloading community listfile...");
            try
            {
                using var http = new HttpClient();
                http.Timeout = TimeSpan.FromMinutes(5);
                http.DefaultRequestHeaders.UserAgent.ParseAdd("MdxViewer/1.0");

                using var response = await http.GetAsync(DownloadUrl, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();

                var totalBytes = response.Content.Headers.ContentLength;
                using var stream = await response.Content.ReadAsStreamAsync();
                using var fileStream = File.Create(CachedPath);

                var buffer = new byte[81920];
                long downloaded = 0;
                int bytesRead;
                int lastPercent = -1;

                while ((bytesRead = await stream.ReadAsync(buffer)) > 0)
                {
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead));
                    downloaded += bytesRead;

                    if (totalBytes > 0)
                    {
                        int percent = (int)(downloaded * 100 / totalBytes.Value);
                        if (percent != lastPercent && percent % 10 == 0)
                        {
                            Console.WriteLine($"[Listfile] {percent}% ({downloaded / 1024 / 1024}MB / {totalBytes.Value / 1024 / 1024}MB)");
                            lastPercent = percent;
                        }
                    }
                }

                Console.WriteLine($"[Listfile] Downloaded: {CachedPath} ({downloaded / 1024 / 1024}MB)");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Listfile] Download failed: {ex.Message}");
                if (File.Exists(CachedPath))
                {
                    Console.WriteLine("[Listfile] Using cached version.");
                    return CachedPath;
                }
                return null;
            }
        }
        else
        {
            Console.WriteLine($"[Listfile] Using cached: {CachedPath}");
        }

        return File.Exists(CachedPath) ? CachedPath : null;
    }

    /// <summary>
    /// Synchronous wrapper for use in non-async contexts.
    /// </summary>
    public static string? GetListfilePath(bool forceDownload = false)
    {
        return GetListfilePathAsync(forceDownload).GetAwaiter().GetResult();
    }
}
