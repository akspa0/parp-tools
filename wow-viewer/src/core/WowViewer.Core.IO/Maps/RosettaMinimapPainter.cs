using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Blp;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Generates crisp, high-contrast 256x256 minimap tiles for Rosetta calibration maps.
/// Renders cell boundaries, museum pedestals, checkered testing pads, razor-sharp readable text plaques, and asset center markers.
/// </summary>
public static class RosettaMinimapPainter
{
    public const int MinimapResolution = 256;
    private const float TileWorldSize = RosettaGeneratorOptions.TileSize; // 533.33333m

    // Natural Warm Sand & High-Contrast Museum Palette
    private static readonly Rgba32 ColorGround = new(208, 192, 160, 255); // Warm Westfall / Westwood Sand
    private static readonly Rgba32 ColorCellBorder = new(145, 128, 98, 255);
    private static readonly Rgba32 ColorDivider = new(160, 142, 112, 255);
    private static readonly Rgba32 ColorPedestalBevel = new(175, 158, 128, 255);
    private static readonly Rgba32 ColorPedestalFlat = new(192, 178, 148, 255);
    private static readonly Rgba32 ColorPedestalHighlight = new(238, 228, 202, 255);
    private static readonly Rgba32 ColorCheckersDark = new(125, 110, 85, 255);
    private static readonly Rgba32 ColorCheckersLight = new(228, 218, 192, 255);
    private static readonly Rgba32 ColorPlaque = new(248, 245, 238, 255); // Bright clean parchment plaque
    private static readonly Rgba32 ColorInk = new(15, 15, 20, 255); // Crisp solid deep black ink
    private static readonly Rgba32 ColorModelDot = new(70, 215, 255, 255);
    private static readonly Rgba32 ColorModelCenter = new(240, 250, 255, 255);
    private static readonly Rgba32 ColorWmoBox = new(255, 175, 45, 255);
    private static readonly Rgba32 ColorWmoBorder = new(180, 105, 15, 255);

    private static readonly Dictionary<char, byte[]> Glyphs3x5 = BuildGlyphs3x5();

    /// <summary>
    /// Renders a complete 256x256 minimap image for a single Rosetta tile.
    /// </summary>
    public static Image<Rgba32> RenderTileImage(
        RosettaTilePlan tile,
        IReadOnlyList<RosettaPedestal> pedestals,
        byte[]? alphaCanvas)
    {
        ArgumentNullException.ThrowIfNull(tile);

        var image = new Image<Rgba32>(MinimapResolution, MinimapResolution);

        // 1. Fill warm sand ground base
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < MinimapResolution; y++)
            {
                var row = accessor.GetRowSpan(y);
                row.Fill(ColorGround);
            }
        });

        // 2. Render Cell Outlines & Object Band Dividers
        foreach (RosettaPlacementRecord placement in tile.Placements)
        {
            int cellPx0 = Math.Clamp((int)(placement.CellU / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int cellPy0 = Math.Clamp((int)(placement.CellV / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int cellPx1 = Math.Clamp((int)((placement.CellU + placement.CellSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int cellPy1 = Math.Clamp((int)((placement.CellV + placement.CellSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int bandPy0 = Math.Clamp((int)((placement.CellV + placement.ObjectBandSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);

            // Perimeter outline
            for (int x = cellPx0; x <= cellPx1; x++)
            {
                image[x, cellPy0] = ColorCellBorder;
                image[x, cellPy1] = ColorCellBorder;
            }
            for (int y = cellPy0; y <= cellPy1; y++)
            {
                image[cellPx0, y] = ColorCellBorder;
                image[cellPx1, y] = ColorCellBorder;
            }

            // Divider line separating 3D exhibit from label plaque
            for (int x = cellPx0; x <= cellPx1; x++)
            {
                image[x, bandPy0] = ColorDivider;
            }
        }

        // 3. Render Pedestals with Checkers Pattern
        if (pedestals is { Count: > 0 })
        {
            foreach (RosettaPedestal pedestal in pedestals)
            {
                int px0 = Math.Clamp((int)(pedestal.U0 / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
                int py0 = Math.Clamp((int)(pedestal.V0 / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
                int px1 = Math.Clamp((int)(pedestal.U1 / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
                int py1 = Math.Clamp((int)(pedestal.V1 / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);

                // Bevel outer ring
                for (int y = py0; y <= py1; y++)
                {
                    for (int x = px0; x <= px1; x++)
                    {
                        image[x, y] = ColorPedestalBevel;
                    }
                }

                // Flat top plateau with diagnostic checkered pattern
                int insetX = Math.Min(2, (px1 - px0) / 4);
                int insetY = Math.Min(2, (py1 - py0) / 4);
                for (int y = py0 + insetY; y <= py1 - insetY; y++)
                {
                    for (int x = px0 + insetX; x <= px1 - insetX; x++)
                    {
                        bool checker = (((x - (px0 + insetX)) / 3) + ((y - (py0 + insetY)) / 3)) % 2 == 0;
                        image[x, y] = checker ? ColorCheckersDark : ColorCheckersLight;
                    }
                }

                // Top & left bevel highlight
                for (int x = px0; x <= px1; x++)
                    image[x, py0] = ColorPedestalHighlight;
                for (int y = py0; y <= py1; y++)
                    image[px0, y] = ColorPedestalHighlight;
            }
        }

        // 4. Render Razor-Sharp Label Plaques & 3x5 Bitmap Text Directly
        foreach (RosettaPlacementRecord placement in tile.Placements)
        {
            int cellPx0 = Math.Clamp((int)(placement.CellU / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int cellPx1 = Math.Clamp((int)((placement.CellU + placement.CellSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int plaquePy0 = Math.Clamp((int)((placement.CellV + placement.ObjectBandSize) / TileWorldSize * MinimapResolution) + 1, 0, MinimapResolution - 1);
            int plaquePy1 = Math.Clamp((int)((placement.CellV + placement.CellSize) / TileWorldSize * MinimapResolution) - 1, 0, MinimapResolution - 1);

            int availWidth = cellPx1 - cellPx0 - 2;
            int availHeight = plaquePy1 - plaquePy0 - 2;

            if (availWidth < 12 || availHeight < 5)
                continue;

            // Fill clean parchment plaque
            for (int y = plaquePy0; y <= plaquePy1; y++)
            {
                for (int x = cellPx0 + 1; x <= cellPx1 - 1; x++)
                {
                    image[x, y] = ColorPlaque;
                }
            }

            // Calculate exact character capacity in 3x5 font (3px wide + 1px spacing = 4px/char, 5px high + 1px spacing = 6px/line)
            int charsPerLine = Math.Max(1, (availWidth + 1) / 4);
            int maxLines = Math.Max(1, (availHeight + 1) / 6);

            // Format label for minimap: filename with extension
            string label = Path.GetFileName(placement.Asset.AssetPath.Replace('/', '\\'));
            IReadOnlyList<string> lines = WrapMinimapLabel(label, charsPerLine, maxLines);

            int totalTextHeight = (lines.Count * 6) - 1;
            int startY = plaquePy0 + 1 + Math.Max(0, (availHeight - totalTextHeight) / 2);

            for (int lineIdx = 0; lineIdx < lines.Count; lineIdx++)
            {
                string lineText = lines[lineIdx];
                int lineWidth = (lineText.Length * 4) - 1;
                int startX = cellPx0 + 1 + Math.Max(1, (availWidth - lineWidth) / 2);
                int lineY = startY + (lineIdx * 6);

                if (lineY + 5 > plaquePy1)
                    break;

                Draw3x5Text(image, lineText, startX, lineY, cellPx0 + 1, cellPx1 - 1, plaquePy0, plaquePy1, ColorInk);
            }
        }

        // 5. Render Placement Pins (Cyan dot for M2, Orange box for WMO)
        foreach (RosettaPlacementRecord placement in tile.Placements)
        {
            (float centerU, float centerV) = RosettaTilesetGenerator.GetObjectBandCenter(placement);
            int cx = Math.Clamp((int)(centerU / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
            int cy = Math.Clamp((int)(centerV / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);

            if (placement.Asset.Kind == RosettaAssetKind.WorldModel)
            {
                // 5x5 WMO building box
                for (int dy = -2; dy <= 2; dy++)
                {
                    for (int dx = -2; dx <= 2; dx++)
                    {
                        int px = cx + dx;
                        int py = cy + dy;
                        if (px >= 0 && px < MinimapResolution && py >= 0 && py < MinimapResolution)
                        {
                            bool border = Math.Abs(dx) == 2 || Math.Abs(dy) == 2;
                            image[px, py] = border ? ColorWmoBorder : ColorWmoBox;
                        }
                    }
                }
            }
            else
            {
                // 3x3 Model Diamond
                int[] dxs = [0, -1, 1, 0, 0];
                int[] dys = [0, 0, 0, -1, 1];
                for (int i = 0; i < dxs.Length; i++)
                {
                    int px = cx + dxs[i];
                    int py = cy + dys[i];
                    if (px >= 0 && px < MinimapResolution && py >= 0 && py < MinimapResolution)
                    {
                        image[px, py] = i == 0 ? ColorModelCenter : ColorModelDot;
                    }
                }
            }
        }

        // 6. Subtle 1px outer tile border
        for (int i = 0; i < MinimapResolution; i++)
        {
            image[i, 0] = ColorCellBorder;
            image[i, MinimapResolution - 1] = ColorCellBorder;
            image[0, i] = ColorCellBorder;
            image[MinimapResolution - 1, i] = ColorCellBorder;
        }

        return image;
    }

    private static void Draw3x5Text(
        Image<Rgba32> image,
        string text,
        int startX,
        int startY,
        int minX,
        int maxX,
        int minY,
        int maxY,
        Rgba32 color)
    {
        int curX = startX;
        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];
            if (Glyphs3x5.TryGetValue(ch, out byte[]? rows))
            {
                for (int r = 0; r < 5; r++)
                {
                    int drawY = startY + r;
                    if (drawY < minY || drawY > maxY)
                        continue;

                    byte mask = rows[r];
                    if ((mask & 0b100) != 0 && curX >= minX && curX <= maxX)
                        image[curX, drawY] = color;
                    if ((mask & 0b010) != 0 && (curX + 1) >= minX && (curX + 1) <= maxX)
                        image[curX + 1, drawY] = color;
                    if ((mask & 0b001) != 0 && (curX + 2) >= minX && (curX + 2) <= maxX)
                        image[curX + 2, drawY] = color;
                }
            }
            curX += 4; // 3px glyph width + 1px character spacing
        }
    }

    private static IReadOnlyList<string> WrapMinimapLabel(string text, int charsPerLine, int maxLines)
    {
        if (string.IsNullOrWhiteSpace(text) || charsPerLine <= 0 || maxLines <= 0)
            return [];

        int capacity = charsPerLine * maxLines;
        if (text.Length > capacity)
        {
            int tail = (capacity - 1) / 2;
            int head = capacity - 1 - tail;
            text = string.Concat(text.AsSpan(0, head), "-", text.AsSpan(text.Length - tail));
        }

        var lines = new List<string>(maxLines);
        for (int i = 0; i < text.Length && lines.Count < maxLines; i += charsPerLine)
            lines.Add(text.Substring(i, Math.Min(charsPerLine, text.Length - i)));

        return lines;
    }

    private static Dictionary<char, byte[]> BuildGlyphs3x5()
    {
        var dict = new Dictionary<char, byte[]>();
        void Add(char c, byte r0, byte r1, byte r2, byte r3, byte r4)
        {
            dict[c] = [r0, r1, r2, r3, r4];
            dict[char.ToUpperInvariant(c)] = [r0, r1, r2, r3, r4];
            dict[char.ToLowerInvariant(c)] = [r0, r1, r2, r3, r4];
        }

        // A-Z
        Add('A', 0b010, 0b101, 0b111, 0b101, 0b101);
        Add('B', 0b110, 0b101, 0b110, 0b101, 0b110);
        Add('C', 0b011, 0b100, 0b100, 0b100, 0b011);
        Add('D', 0b110, 0b101, 0b101, 0b101, 0b110);
        Add('E', 0b111, 0b100, 0b110, 0b100, 0b111);
        Add('F', 0b111, 0b100, 0b110, 0b100, 0b100);
        Add('G', 0b011, 0b100, 0b101, 0b101, 0b011);
        Add('H', 0b101, 0b101, 0b111, 0b101, 0b101);
        Add('I', 0b111, 0b010, 0b010, 0b010, 0b111);
        Add('J', 0b001, 0b001, 0b001, 0b101, 0b010);
        Add('K', 0b101, 0b110, 0b100, 0b110, 0b101);
        Add('L', 0b100, 0b100, 0b100, 0b100, 0b111);
        Add('M', 0b101, 0b111, 0b101, 0b101, 0b101);
        Add('N', 0b110, 0b101, 0b101, 0b101, 0b101);
        Add('O', 0b010, 0b101, 0b101, 0b101, 0b010);
        Add('P', 0b110, 0b101, 0b110, 0b100, 0b100);
        Add('Q', 0b010, 0b101, 0b101, 0b011, 0b001);
        Add('R', 0b110, 0b101, 0b110, 0b101, 0b101);
        Add('S', 0b011, 0b100, 0b010, 0b001, 0b110);
        Add('T', 0b111, 0b010, 0b010, 0b010, 0b010);
        Add('U', 0b101, 0b101, 0b101, 0b101, 0b010);
        Add('V', 0b101, 0b101, 0b101, 0b101, 0b010);
        Add('W', 0b101, 0b101, 0b101, 0b111, 0b101);
        Add('X', 0b101, 0b101, 0b010, 0b101, 0b101);
        Add('Y', 0b101, 0b101, 0b010, 0b010, 0b010);
        Add('Z', 0b111, 0b001, 0b010, 0b100, 0b111);

        // 0-9
        Add('0', 0b111, 0b101, 0b101, 0b101, 0b111);
        Add('1', 0b010, 0b110, 0b010, 0b010, 0b111);
        Add('2', 0b110, 0b001, 0b010, 0b100, 0b111);
        Add('3', 0b110, 0b001, 0b010, 0b001, 0b110);
        Add('4', 0b101, 0b101, 0b111, 0b001, 0b001);
        Add('5', 0b111, 0b100, 0b110, 0b001, 0b110);
        Add('6', 0b011, 0b100, 0b110, 0b101, 0b010);
        Add('7', 0b111, 0b001, 0b010, 0b010, 0b010);
        Add('8', 0b010, 0b101, 0b010, 0b101, 0b010);
        Add('9', 0b010, 0b101, 0b011, 0b001, 0b110);

        // Punctuation
        Add('.', 0b000, 0b000, 0b000, 0b000, 0b010);
        Add('-', 0b000, 0b000, 0b111, 0b000, 0b000);
        Add('_', 0b000, 0b000, 0b000, 0b000, 0b111);
        Add('/', 0b001, 0b001, 0b010, 0b100, 0b100);
        Add('\\', 0b100, 0b100, 0b010, 0b001, 0b001);
        Add(':', 0b000, 0b010, 0b000, 0b010, 0b000);
        Add(' ', 0b000, 0b000, 0b000, 0b000, 0b000);

        return dict;
    }

    /// <summary>
    /// Renders a 256x256 minimap tile into raw 24-bit RGB bytes (256 * 256 * 3 = 196,608 bytes)
    /// for direct Zarr tensor storage without re-compression loss.
    /// </summary>
    public static byte[] RenderTileRgb24(
        RosettaTilePlan tile,
        IReadOnlyList<RosettaPedestal> pedestals,
        byte[]? alphaCanvas)
    {
        using Image<Rgba32> image = RenderTileImage(tile, pedestals, alphaCanvas);
        byte[] rgb = new byte[MinimapResolution * MinimapResolution * 3];
        int idx = 0;
        for (int y = 0; y < MinimapResolution; y++)
        {
            for (int x = 0; x < MinimapResolution; x++)
            {
                Rgba32 pixel = image[x, y];
                rgb[idx++] = pixel.R;
                rgb[idx++] = pixel.G;
                rgb[idx++] = pixel.B;
            }
        }
        return rgb;
    }

    /// <summary>
    /// Renders and encodes a 256x256 minimap tile directly into standalone BLP2 DXT1 bytes.
    /// </summary>
    public static byte[] RenderTileBlp(
        RosettaTilePlan tile,
        IReadOnlyList<RosettaPedestal> pedestals,
        byte[]? alphaCanvas)
    {
        using Image<Rgba32> image = RenderTileImage(tile, pedestals, alphaCanvas);
        return Blp2Writer.EncodeDxt1(image);
    }
}
