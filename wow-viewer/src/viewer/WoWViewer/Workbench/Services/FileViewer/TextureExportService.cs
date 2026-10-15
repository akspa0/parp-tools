using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Wc3;
using WoWViewer.DataSources;

namespace WoWViewer;

/// <summary>Result of a PNG export run.</summary>
internal sealed record TextureExportResult(int Written, int Skipped, int Failed, string? FirstError, string? LastPath);

/// <summary>
/// Decodes textures from any data source and writes PNG files under an operator-supplied output root, mirroring the
/// source-relative folder layout. Mip level 0 is full resolution; BLP0 external mips can optionally be exported too.
/// </summary>
internal static class TextureExportService
{
    private static readonly char[] Invalid = Path.GetInvalidFileNameChars();

    internal static TextureExportResult Export(IDataSource dataSource, IEnumerable<string> paths, string outputRoot,
        bool allMips, bool overwrite, IProgress<string>? progress, CancellationToken token)
    {
        if (string.IsNullOrWhiteSpace(outputRoot))
            throw new ArgumentException("Choose an output folder first.");
        string root = Path.GetFullPath(outputRoot);
        Directory.CreateDirectory(root);
        int written = 0, skipped = 0, failed = 0;
        string? firstError = null, last = null;
        int index = 0;
        foreach (string path in paths)
        {
            token.ThrowIfCancellationRequested();
            index++;
            try
            {
                int[] counts = ExportOne(dataSource, path, root, allMips, overwrite, ref last);
                written += counts[0];
                skipped += counts[1];
            }
            catch (OperationCanceledException) { throw; }
            catch (Exception ex)
            {
                failed++;
                firstError ??= $"{path}: {ex.Message}";
            }
            if (index % 25 == 0) progress?.Report($"PNG export: {index} files processed ({written} written)...");
        }
        return new TextureExportResult(written, skipped, failed, firstError, last);
    }

    private static int[] ExportOne(IDataSource ds, string path, string root, bool allMips, bool overwrite, ref string? last)
    {
        byte[] bytes = DataFileViewerService.ReadRaw(ds, path) ?? throw new FileNotFoundException("Source entry is missing.");
        Func<string, byte[]?> companion = p => DataFileViewerService.ReadRaw(ds, p);
        bool isMipEntry = Wc3Blp0Decoder.IsMipPath(path);
        Wc3AssetPreview main = Wc3AssetPreviewReader.Read(bytes, path, companion);
        if (main.Rgba == null) throw new InvalidDataException(main.Error ?? "No pixels decoded.");

        int written = 0, skipped = 0;
        string suffix0 = isMipEntry ? "_mip" + Path.GetExtension(path)[2..] : "";
        (string? dest, bool wrote) = Write(root, path, suffix0, main, overwrite);
        if (wrote) { written++; last = dest; } else skipped++;

        // BLP0 beta textures keep each mip level in its own .bNN file; level 0 is full resolution.
        bool blp0 = !isMipEntry && bytes.AsSpan().StartsWith("BLP0"u8);
        if (allMips && blp0)
        {
            for (int level = 1; level < 16; level++)
            {
                string mipPath = Path.ChangeExtension(path, $".b{level:D2}");
                byte[]? mipBytes = companion(mipPath);
                if (mipBytes == null) break;
                Wc3AssetPreview mip = Wc3Blp0Decoder.Read(mipBytes, mipPath, companion);
                if (mip.Rgba == null) break;
                (string? mipDest, bool mipWrote) = Write(root, path, $"_mip{level}", mip, overwrite);
                if (mipWrote) { written++; last = mipDest; } else skipped++;
            }
        }
        return [written, skipped];
    }

    private static (string? Dest, bool Written) Write(string root, string sourcePath, string suffix, Wc3AssetPreview pixels, bool overwrite)
    {
        string dest = BuildDestination(root, sourcePath, suffix);
        Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
        if (File.Exists(dest) && !overwrite) return (dest, false);
        using Image<Rgba32> image = SixLabors.ImageSharp.Image.LoadPixelData<Rgba32>(pixels.Rgba!, pixels.Width, pixels.Height);
        using FileStream stream = new(dest, overwrite ? FileMode.Create : FileMode.CreateNew, FileAccess.Write, FileShare.None);
        image.SaveAsPng(stream);
        return (dest, true);
    }

    /// <summary>Safe mirrored destination: no rooted paths, no traversal, extension replaced with .png.</summary>
    internal static string BuildDestination(string root, string sourcePath, string suffix)
    {
        string[] segments = sourcePath.Replace('/', '\\').Split('\\', StringSplitOptions.RemoveEmptyEntries)
            .Where(s => s != "." && s != "..")
            .Select(s => new string(s.Select(c => Array.IndexOf(Invalid, c) >= 0 ? '_' : c).ToArray()))
            .ToArray();
        if (segments.Length == 0) throw new ArgumentException("Empty source path.");
        string file = segments[^1];
        string stem = Path.GetFileNameWithoutExtension(file);
        segments[^1] = stem + suffix + ".png";
        string full = Path.GetFullPath(Path.Combine([root, .. segments]));
        string rootWithSeparator = root.EndsWith(Path.DirectorySeparatorChar) ? root : root + Path.DirectorySeparatorChar;
        if (!full.StartsWith(rootWithSeparator, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Resolved export path escapes the output folder.");
        return full;
    }
}
