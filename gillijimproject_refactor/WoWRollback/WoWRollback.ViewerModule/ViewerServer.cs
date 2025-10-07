using System.Net;

namespace WoWRollback.ViewerModule;

/// <summary>
/// Embedded HTTP server for serving static viewer files.
/// Uses HttpListener for simple, dependency-free static file serving.
/// </summary>
public sealed class ViewerServer : IDisposable
{
    private HttpListener? _listener;
    private CancellationTokenSource? _cts;
    private Task? _serverTask;
    private string? _viewerDir;

    /// <summary>
    /// Starts the HTTP server on the specified port, serving files from viewerDir.
    /// </summary>
    /// <param name="viewerDir">Root directory containing static files to serve</param>
    /// <param name="port">HTTP port to listen on (default: 8080)</param>
    public void Start(string viewerDir, int port = 8080)
    {
        if (_listener != null)
            throw new InvalidOperationException("Server is already running");

        if (string.IsNullOrWhiteSpace(viewerDir))
            throw new ArgumentException("Viewer directory is required", nameof(viewerDir));

        if (!Directory.Exists(viewerDir))
            throw new DirectoryNotFoundException($"Viewer directory not found: {viewerDir}");

        _viewerDir = Path.GetFullPath(viewerDir);
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
        
        try
        {
            _listener.Start();
        }
        catch (HttpListenerException ex)
        {
            throw new InvalidOperationException(
                $"Failed to start HTTP server on port {port}. " +
                $"Port may be in use or require administrator privileges. Error: {ex.Message}", 
                ex);
        }

        _cts = new CancellationTokenSource();
        _serverTask = Task.Run(() => ServeFilesAsync(_cts.Token));
    }

    /// <summary>
    /// Stops the HTTP server and waits for pending requests to complete.
    /// </summary>
    public void Stop()
    {
        if (_listener == null)
            return;

        _cts?.Cancel();
        
        if (_serverTask != null)
        {
            try
            {
                _serverTask.Wait(TimeSpan.FromSeconds(5));
            }
            catch (AggregateException)
            {
                // Expected when task is cancelled
            }
        }

        _listener?.Stop();
        _listener?.Close();
        _listener = null;
        _serverTask = null;
    }

    /// <summary>
    /// Disposes the server, stopping it if still running.
    /// </summary>
    public void Dispose()
    {
        Stop();
        _cts?.Dispose();
        _cts = null;
    }

    private async Task ServeFilesAsync(CancellationToken cancellationToken)
    {
        if (_listener == null || _viewerDir == null)
            return;

        while (!cancellationToken.IsCancellationRequested && _listener.IsListening)
        {
            try
            {
                var context = await _listener.GetContextAsync();
                
                // Handle request asynchronously to allow concurrent requests
                _ = Task.Run(() => HandleRequestAsync(context), cancellationToken);
            }
            catch (HttpListenerException)
            {
                // Listener was stopped
                break;
            }
            catch (ObjectDisposedException)
            {
                // Listener was disposed
                break;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[ViewerServer] Error accepting request: {ex.Message}");
            }
        }
    }

    private async Task HandleRequestAsync(HttpListenerContext context)
    {
        var request = context.Request;
        var response = context.Response;

        try
        {
            var urlPath = request.Url?.AbsolutePath ?? "/";
            
            // Default to index.html for root
            if (urlPath == "/")
                urlPath = "/index.html";

            // Remove leading slash and sanitize path
            var relativePath = urlPath.TrimStart('/');
            
            // Prevent directory traversal attacks
            if (relativePath.Contains(".."))
            {
                response.StatusCode = 400;
                await SendTextResponseAsync(response, "Bad Request: Invalid path");
                return;
            }

            var filePath = Path.Combine(_viewerDir!, relativePath);

            if (!File.Exists(filePath))
            {
                response.StatusCode = 404;
                await SendTextResponseAsync(response, $"Not Found: {urlPath}");
                return;
            }

            // Determine content type
            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            response.ContentType = GetContentType(extension);

            // Serve file
            using var fileStream = File.OpenRead(filePath);
            response.ContentLength64 = fileStream.Length;
            await fileStream.CopyToAsync(response.OutputStream);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[ViewerServer] Error handling request: {ex.Message}");
            
            if (!response.OutputStream.CanWrite)
                return;

            try
            {
                response.StatusCode = 500;
                await SendTextResponseAsync(response, "Internal Server Error");
            }
            catch
            {
                // Ignore errors during error response
            }
        }
        finally
        {
            try
            {
                response.Close();
            }
            catch
            {
                // Ignore close errors
            }
        }
    }

    private static async Task SendTextResponseAsync(HttpListenerResponse response, string text)
    {
        var buffer = System.Text.Encoding.UTF8.GetBytes(text);
        response.ContentLength64 = buffer.Length;
        response.ContentType = "text/plain";
        await response.OutputStream.WriteAsync(buffer);
    }

    private static string GetContentType(string extension)
    {
        return extension switch
        {
            ".html" => "text/html; charset=utf-8",
            ".htm" => "text/html; charset=utf-8",
            ".css" => "text/css; charset=utf-8",
            ".js" => "application/javascript; charset=utf-8",
            ".json" => "application/json; charset=utf-8",
            ".png" => "image/png",
            ".jpg" => "image/jpeg",
            ".jpeg" => "image/jpeg",
            ".gif" => "image/gif",
            ".svg" => "image/svg+xml",
            ".ico" => "image/x-icon",
            ".woff" => "font/woff",
            ".woff2" => "font/woff2",
            ".ttf" => "font/ttf",
            ".txt" => "text/plain; charset=utf-8",
            ".xml" => "application/xml; charset=utf-8",
            _ => "application/octet-stream"
        };
    }
}
