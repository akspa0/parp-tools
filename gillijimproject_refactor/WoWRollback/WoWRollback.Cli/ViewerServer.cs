using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace WoWRollback.Cli;

/// <summary>
/// Self-contained HTTP server for hosting the viewer.
/// No external dependencies (Python, Node, etc.) required.
/// </summary>
public static class ViewerServer
{
    public static void Serve(string viewerPath, int port = 8080, bool openBrowser = true)
    {
        if (!Directory.Exists(viewerPath))
        {
            Console.Error.WriteLine($"[error] Viewer directory not found: {viewerPath}");
            return;
        }

        var absolutePath = Path.GetFullPath(viewerPath);
        Console.WriteLine($"[info] Starting HTTP server...");
        Console.WriteLine($"[info] Serving: {absolutePath}");
        Console.WriteLine($"[info] URL: http://localhost:{port}");
        Console.WriteLine($"[info] Press Ctrl+C to stop the server");
        Console.WriteLine();

        var builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            WebRootPath = absolutePath
        });

        // Disable most logging for cleaner output
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        builder.Logging.SetMinimumLevel(LogLevel.Warning);

        // Configure Kestrel
        builder.WebHost.UseKestrel(options =>
        {
            options.ListenLocalhost(port);
        });

        var app = builder.Build();

        // Serve static files with proper MIME types
        var fileProvider = new PhysicalFileProvider(absolutePath);
        var options = new StaticFileOptions
        {
            FileProvider = fileProvider,
            RequestPath = "",
            ServeUnknownFileTypes = true,
            DefaultContentType = "application/octet-stream"
        };

        // Add custom MIME types for viewer assets
        var provider = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
        provider.Mappings[".webp"] = "image/webp";
        provider.Mappings[".json"] = "application/json";
        provider.Mappings[".geojson"] = "application/geo+json";
        options.ContentTypeProvider = provider;

        app.UseStaticFiles(options);

        // Fallback to index.html for root
        app.MapGet("/", async context =>
        {
            var indexPath = Path.Combine(absolutePath, "index.html");
            if (File.Exists(indexPath))
            {
                context.Response.ContentType = "text/html";
                await context.Response.SendFileAsync(indexPath);
            }
            else
            {
                context.Response.StatusCode = 404;
                await context.Response.WriteAsync("index.html not found");
            }
        });

        // Log requests for debugging
        app.Use(async (context, next) =>
        {
            var path = context.Request.Path;
            await next();
            
            // Only log errors (500+), not 404s (terrain overlays are sparse)
            if (context.Response.StatusCode >= 500)
            {
                Console.WriteLine($"[{context.Response.StatusCode}] {path}");
            }
        });

        // Open browser after server starts
        if (openBrowser)
        {
            var url = $"http://localhost:{port}";
            _ = Task.Run(async () =>
            {
                await Task.Delay(1000); // Wait for server to start
                OpenBrowser(url);
            });
        }

        try
        {
            app.Run();
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[error] Server failed: {ex.Message}");
        }
    }

    private static void OpenBrowser(string url)
    {
        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
            {
                Process.Start("xdg-open", url);
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                Process.Start("open", url);
            }
            
            Console.WriteLine($"[ok] Browser opened: {url}");
        }
        catch
        {
            Console.WriteLine($"[info] Please open manually: {url}");
        }
    }
}
