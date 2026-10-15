using System.Text;
using Silk.NET.OpenGL;
using WowViewer.Core.IO.Wc3;
using WoWViewer.DataSources;
using WoWViewer.Rendering;

namespace WoWViewer;

/// <summary>Result of opening an arbitrary data-source file in the main viewer.</summary>
internal readonly record struct DataFileView(ISceneRenderer? Renderer, string Info, string Status);

/// <summary>
/// Generic "open any file" support for the main File Browser: images (BLP incl. WC3 BLP0 + external mips, TGA, PNG ...)
/// render in the main viewport via <see cref="TextureViewRenderer"/>; text shows in the info panel; anything else
/// shows size + hex head. Never touches world state; the caller owns renderer lifetime.
/// </summary>
internal static class DataFileViewerService
{
    private const int HexBytes = 512;
    private const int MaxInfoTextChars = 64 * 1024;

    private static readonly HashSet<string> ImageExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".blp", ".tga", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".qoi", ".pbm", ".pgm", ".ppm"
    };

    internal static bool IsImagePath(string path)
        => ImageExtensions.Contains(Path.GetExtension(path)) || Wc3Blp0Decoder.IsMipPath(path);

    /// <summary>True for extensions the model/WMO/WDT loaders own.</summary>
    internal static bool IsHandledByModelLoader(string extension)
        => extension is ".mdx" or ".mdl" or ".m2" or ".wmo" or ".wdt";

    /// <summary>Original bytes (BLP0 textures are NOT pre-converted).</summary>
    internal static byte[]? ReadRaw(IDataSource dataSource, string path)
        => dataSource is Wc3DataSource wc3 ? wc3.ReadRaw(path) : dataSource.ReadFile(path);

    internal static DataFileView Open(IDataSource dataSource, string path, GL gl)
    {
        string name = Path.GetFileName(path);
        byte[]? bytes;
        try { bytes = ReadRaw(dataSource, path); }
        catch (Exception ex) { return new(null, $"Path: {path}\nRead failed: {ex.Message}", $"Failed to read {name}: {ex.Message}"); }
        if (bytes == null || bytes.Length == 0)
            return new(null, $"Path: {path}\nThe file is missing or empty in this source.", $"Failed to read: {path}");

        bool blpMagic = bytes.Length >= 4 && bytes[0] == (byte)'B' && bytes[1] == (byte)'L' && bytes[2] == (byte)'P';
        if (IsImagePath(path) || blpMagic)
        {
            Wc3AssetPreview preview = Wc3AssetPreviewReader.Read(bytes, path, p => ReadRaw(dataSource, p));
            if (preview.Rgba != null)
            {
                TextureViewRenderer renderer = new(gl, preview.Rgba, preview.Width, preview.Height);
                int mips = CountExternalMips(dataSource, path);
                string info = $"Path: {path}\nType: Image ({preview.Format})\nSize: {preview.Width} x {preview.Height}\n" +
                              $"File bytes: {bytes.Length:N0}\n" +
                              (mips > 0 ? $"External mip levels: {mips} (.b00 = full resolution)\n" : "") +
                              "\nMouse wheel: zoom   Drag: pan   Double-click: reset";
                return new(renderer, info, $"Loaded image: {name} ({preview.Width}x{preview.Height}, {preview.Format})");
            }
            string error = preview.Error ?? "No preview available.";
            return new(null, $"Path: {path}\nType: Image ({preview.Format})\n{error}", $"Cannot display {name}: {error}");
        }

        if (LooksLikeText(bytes))
        {
            Wc3AssetPreview text = Wc3AssetPreviewReader.ReadText(bytes, Path.GetExtension(path).TrimStart('.').ToUpperInvariant());
            if (text.Text != null)
            {
                string body = text.Text.Length > MaxInfoTextChars ? text.Text[..MaxInfoTextChars] + "\n[... truncated for display ...]" : text.Text;
                return new(null, $"Path: {path}\nType: Text ({text.Format})\nFile bytes: {bytes.Length:N0}\n\n{body}",
                    $"Viewing text: {name} ({bytes.Length:N0} bytes)");
            }
        }

        return new(null, BuildHexInfo(path, bytes), $"Binary file: {name} ({bytes.Length:N0} bytes) - hex head shown in Model Info.");
    }

    /// <summary>Number of consecutive <c>.b00, .b01 ...</c> companions beside a BLP header (0 when none).</summary>
    internal static int CountExternalMips(IDataSource dataSource, string path)
    {
        string header = Path.ChangeExtension(path, ".blp");
        int count = 0;
        for (; count < 16; count++)
            if (!dataSource.FileExists(Path.ChangeExtension(header, $".b{count:D2}"))) break;
        return count;
    }

    private static bool LooksLikeText(byte[] bytes)
    {
        int n = Math.Min(bytes.Length, 8192);
        int controls = 0;
        for (int i = 0; i < n; i++)
        {
            byte b = bytes[i];
            if (b == 0) return bytes.Length >= 2 && ((bytes[0] == 0xFF && bytes[1] == 0xFE) || (bytes[0] == 0xFE && bytes[1] == 0xFF));
            if (b < 9 || (b > 13 && b < 32)) controls++;
        }
        return controls * 100 <= n;
    }

    private static string BuildHexInfo(string path, byte[] bytes)
    {
        StringBuilder sb = new();
        sb.AppendLine($"Path: {path}");
        sb.AppendLine($"Type: Binary ({Path.GetExtension(path)})");
        sb.AppendLine($"File bytes: {bytes.Length:N0}");
        sb.AppendLine();
        int count = Math.Min(bytes.Length, HexBytes);
        for (int row = 0; row < count; row += 16)
        {
            sb.Append(row.ToString("X6")).Append("  ");
            for (int i = 0; i < 16; i++)
                sb.Append(row + i < count ? bytes[row + i].ToString("X2") + " " : "   ");
            sb.Append(' ');
            for (int i = 0; i < 16 && row + i < count; i++)
                sb.Append(bytes[row + i] is >= 32 and < 127 ? (char)bytes[row + i] : '.');
            sb.AppendLine();
        }
        if (bytes.Length > count) sb.AppendLine($"... {bytes.Length - count:N0} more bytes");
        return sb.ToString();
    }
}
