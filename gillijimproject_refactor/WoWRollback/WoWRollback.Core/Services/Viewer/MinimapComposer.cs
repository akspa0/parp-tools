using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using BLPSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace WoWRollback.Core.Services.Viewer;

/// <summary>
/// Generates per-tile minimap JPGs from source imagery (BLP or pre-rendered raster).
/// </summary>
public sealed class MinimapComposer
{
    /// <summary>
    /// Composes a minimap image for a tile and writes it to <paramref name="destinationPath"/>.
    /// </summary>
    /// <param name="source">Stream containing BLP or PNG data.</param>
    /// <param name="destinationPath">Destination JPG path.</param>
    /// <param name="options">Viewer output options.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    public async Task ComposeAsync(
        Stream source,
        string destinationPath,
        ViewerOptions options,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destinationPath);
        ArgumentNullException.ThrowIfNull(options);

        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        if (!source.CanRead)
            throw new InvalidOperationException("Minimap source stream is not readable.");

        if (source.CanSeek) source.Seek(0, SeekOrigin.Begin);

        await using var buffer = new MemoryStream();
        await source.CopyToAsync(buffer, cancellationToken).ConfigureAwait(false);

        if (buffer.Length == 0)
        {
            await WritePlaceholderAsync(destinationPath, options, cancellationToken).ConfigureAwait(false);
            return;
        }

        Image<Rgba32>? image = null;
        try
        {
            if (IsBlp(buffer))
            {
                buffer.Seek(0, SeekOrigin.Begin);
                using var blpStream = new MemoryStream(buffer.ToArray());
                using var blp = new BLPFile(blpStream);
                var pixels = blp.GetPixels(0, out var width, out var height);
                using var bgraImage = Image.LoadPixelData<Bgra32>(pixels, width, height);
                image = bgraImage.CloneAs<Rgba32>();
            }
            else
            {
                buffer.Seek(0, SeekOrigin.Begin);
                image = Image.Load<Rgba32>(buffer.ToArray());
            }

            if (image.Width != options.MinimapWidth || image.Height != options.MinimapHeight)
            {
                // Preserve original pixel structure: nearest-neighbor scaling
                image.Mutate(ctx =>
                    ctx.Resize(new ResizeOptions
                    {
                        Size = new Size(options.MinimapWidth, options.MinimapHeight),
                        Sampler = KnownResamplers.NearestNeighbor,
                        Mode = ResizeMode.Stretch
                    }));
            }

            var encoder = new JpegEncoder
            {
                Quality = 85
            };

            await using var fileStream = File.Open(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
            await image.SaveAsJpegAsync(fileStream, encoder, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            image?.Dispose();
            await WritePlaceholderAsync(destinationPath, options, cancellationToken).ConfigureAwait(false);
            return;
        }
        finally
        {
            image?.Dispose();
        }
    }

    private static bool IsBlp(Stream buffer)
    {
        if (!buffer.CanSeek) return false;
        var initial = buffer.Position;
        buffer.Seek(0, SeekOrigin.Begin);
        Span<byte> header = stackalloc byte[4];
        var read = buffer.Read(header);
        buffer.Seek(initial, SeekOrigin.Begin);
        if (read < 4) return false;
        return header.SequenceEqual("BLP2"u8) || header.SequenceEqual("BLP1"u8);
    }

    public async Task WritePlaceholderAsync(
        string destinationPath,
        ViewerOptions options,
        CancellationToken cancellationToken = default)
    {
        var directory = Path.GetDirectoryName(destinationPath);
        if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);

        using var image = new Image<Rgba32>(options.MinimapWidth, options.MinimapHeight);
        var background = new Rgba32(0x22, 0x25, 0x2B, 255);
        image.Mutate(ctx => ctx.BackgroundColor(background));

        image.ProcessPixelRows(accessor =>
        {
            var midX = accessor.Width / 2;
            var midY = accessor.Height / 2;
            for (var y = 0; y < accessor.Height; y++)
            {
                var rowSpan = accessor.GetRowSpan(y);
                for (var x = 0; x < accessor.Width; x++)
                {
                    if (x == midX || y == midY)
                    {
                        rowSpan[x] = new Rgba32(0x3C, 0x42, 0x4D, 255);
                    }
                }
            }
        });

        var encoder = new JpegEncoder
        {
            Quality = 85
        };

        await using var fileStream = File.Open(destinationPath, FileMode.Create, FileAccess.Write, FileShare.None);
        await image.SaveAsJpegAsync(fileStream, encoder, cancellationToken).ConfigureAwait(false);
    }
}
