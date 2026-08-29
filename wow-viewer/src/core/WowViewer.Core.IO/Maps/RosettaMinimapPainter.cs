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

        // 3. Render Pedestals with Checkers Border Ring & Clean Neutral Center Plaza
        var effectivePedestals = (pedestals is { Count: > 0 })
            ? pedestals
            : tile.Placements.Select(static p => new RosettaPedestal(
                p.CellU, p.CellV, p.CellU + p.CellSize, p.CellV + p.ObjectBandSize, 2.5f)).ToList();

        foreach (RosettaPedestal pedestal in effectivePedestals)
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

            // Flat top plateau with decorative checkered border ring
            int insetX = Math.Max(1, (px1 - px0) / 8);
            int insetY = Math.Max(1, (py1 - py0) / 8);
            int innerCenterX0 = px0 + (px1 - px0) / 3;
            int innerCenterX1 = px1 - (px1 - px0) / 3;
            int innerCenterY0 = py0 + (py1 - py0) / 3;
            int innerCenterY1 = py1 - (py1 - py0) / 3;

            for (int y = py0 + insetY; y <= py1 - insetY; y++)
            {
                for (int x = px0 + insetX; x <= px1 - insetX; x++)
                {
                    bool isCenterPlaza = x >= innerCenterX0 && x <= innerCenterX1 && y >= innerCenterY0 && y <= innerCenterY1;
                    if (isCenterPlaza)
                    {
                        // Clean neutral smooth marble/stone floor under object
                        image[x, y] = new Rgba32(236, 232, 222, 255);
                    }
                    else
                    {
                        // Diagnostic checker border frame
                        bool checker = (((x - (px0 + insetX)) / 3) + ((y - (py0 + insetY)) / 3)) % 2 == 0;
                        image[x, y] = checker ? ColorCheckersDark : ColorCheckersLight;
                    }
                }
            }

            // Top & left bevel highlight
            for (int x = px0; x <= px1; x++)
                image[x, py0] = ColorPedestalHighlight;
            for (int y = py0; y <= py1; y++)
                image[px0, y] = ColorPedestalHighlight;
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

        // 5. Render Proportional 3D-Shaded Object Footprints & Silhouettes
        foreach (RosettaPlacementRecord placement in tile.Placements)
        {
            DrawObjectFootprint(image, placement);
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

    private static void DrawObjectFootprint(Image<Rgba32> image, RosettaPlacementRecord placement)
    {
        (float centerU, float centerV) = RosettaTilesetGenerator.GetObjectBandCenter(placement);
        int cx = Math.Clamp((int)(centerU / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
        int cy = Math.Clamp((int)(centerV / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);

        int cellPx0 = Math.Clamp((int)(placement.CellU / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
        int cellPx1 = Math.Clamp((int)((placement.CellU + placement.CellSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
        int cellPy0 = Math.Clamp((int)(placement.CellV / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);
        int bandPy0 = Math.Clamp((int)((placement.CellV + placement.ObjectBandSize) / TileWorldSize * MinimapResolution), 0, MinimapResolution - 1);

        float maxAllowedRadX = Math.Max(2f, (cellPx1 - cellPx0) * 0.44f);
        float maxAllowedRadY = Math.Max(2f, (bandPy0 - cellPy0) * 0.44f);

        float extentX = MathF.Abs(placement.Asset.BoundsMax.X - placement.Asset.BoundsMin.X) * placement.Scale;
        float extentY = MathF.Abs(placement.Asset.BoundsMax.Y - placement.Asset.BoundsMin.Y) * placement.Scale;
        float extentZ = MathF.Abs(placement.Asset.BoundsMax.Z - placement.Asset.BoundsMin.Z) * placement.Scale;

        if (!float.IsFinite(extentX) || extentX <= 0.1f) extentX = 4f * placement.Scale;
        if (!float.IsFinite(extentY) || extentY <= 0.1f) extentY = 4f * placement.Scale;

        float radX = Math.Clamp((extentX / TileWorldSize * MinimapResolution) * 0.5f, 2.0f, maxAllowedRadX);
        float radY = Math.Clamp((extentY / TileWorldSize * MinimapResolution) * 0.5f, 2.0f, maxAllowedRadY);

        int rxInt = (int)MathF.Ceiling(radX);
        int ryInt = (int)MathF.Ceiling(radY);

        // 1. Drop shadow behind object
        for (int dy = -ryInt; dy <= ryInt; dy++)
        {
            for (int dx = -rxInt; dx <= rxInt; dx++)
            {
                float dsq = (dx * dx) / (radX * radX) + (dy * dy) / (radY * radY);
                if (dsq <= 1.0f)
                {
                    int sx = cx + dx + 1;
                    int sy = cy + dy + 1;
                    if (sx >= 0 && sx < MinimapResolution && sy >= 0 && sy < MinimapResolution)
                    {
                        Rgba32 baseCol = image[sx, sy];
                        image[sx, sy] = BlendColors(baseCol, new Rgba32(30, 24, 18, 255), 0.45f);
                    }
                }
            }
        }

        if (placement.Asset.Kind == RosettaAssetKind.WorldModel)
        {
            // WMO Building: Rectangular architectural structure
            Rgba32 wallColor = new(75, 45, 20, 255);
            Rgba32 roofLight = new(235, 115, 45, 255);
            Rgba32 roofDark = new(150, 60, 20, 255);
            Rgba32 ridgeColor = new(255, 205, 130, 255);

            for (int dy = -ryInt; dy <= ryInt; dy++)
            {
                for (int dx = -rxInt; dx <= rxInt; dx++)
                {
                    int px = cx + dx;
                    int py = cy + dy;
                    if (px < 0 || px >= MinimapResolution || py < 0 || py >= MinimapResolution)
                        continue;

                    bool isBorder = Math.Abs(dx) == rxInt || Math.Abs(dy) == ryInt;
                    if (isBorder)
                    {
                        image[px, py] = wallColor;
                    }
                    else
                    {
                        float normX = (float)dx / Math.Max(1, rxInt);
                        float normY = (float)dy / Math.Max(1, ryInt);
                        float shade = Math.Clamp(0.5f - (normX * 0.35f + normY * 0.35f), 0f, 1f);
                        Rgba32 roof = LerpColor(roofDark, roofLight, shade);
                        if (rxInt >= 4 && ryInt >= 4 && (dx == 0 || dy == 0))
                            roof = LerpColor(roof, ridgeColor, 0.6f);
                        image[px, py] = roof;
                    }
                }
            }

            // Doorway indicator at South edge
            if (ryInt >= 2)
            {
                int doorY = cy + ryInt;
                if (doorY >= 0 && doorY < MinimapResolution)
                {
                    image[cx, doorY] = new Rgba32(20, 10, 5, 255);
                    if (cx + 1 < MinimapResolution) image[cx + 1, doorY] = new Rgba32(20, 10, 5, 255);
                }
            }
        }
        else
        {
            // M2 / MDX Model: Organic/proportional shaded entity
            string pathLower = placement.Asset.AssetPath.ToLowerInvariant();
            (Rgba32 bodyLight, Rgba32 bodyDark, Rgba32 borderColor, Rgba32 coreGlow) = ResolveModelPalette(pathLower);

            for (int dy = -ryInt; dy <= ryInt; dy++)
            {
                for (int dx = -rxInt; dx <= rxInt; dx++)
                {
                    float dsq = (dx * dx) / (radX * radX) + (dy * dy) / (radY * radY);
                    if (dsq > 1.0f)
                        continue;

                    int px = cx + dx;
                    int py = cy + dy;
                    if (px < 0 || px >= MinimapResolution || py < 0 || py >= MinimapResolution)
                        continue;

                    if (dsq > 0.72f)
                    {
                        image[px, py] = borderColor;
                    }
                    else
                    {
                        float normX = dx / radX;
                        float normY = dy / radY;
                        float normZ = MathF.Sqrt(MathF.Max(0f, 1f - normX * normX - normY * normY));
                        float light = Math.Clamp(normZ * 0.65f - (normX * 0.35f + normY * 0.35f), 0f, 1f);
                        Rgba32 body = LerpColor(bodyDark, bodyLight, light);

                        // Center focal specular point
                        if (dsq < 0.15f && normX < 0.1f && normY < 0.1f)
                            body = LerpColor(body, coreGlow, 0.7f);

                        image[px, py] = body;
                    }
                }
            }
        }
    }

    private static (Rgba32 BodyLight, Rgba32 BodyDark, Rgba32 Border, Rgba32 Core) ResolveModelPalette(string pathLower)
    {
        if (pathLower.Contains("creature") || pathLower.Contains("character"))
        {
            // Vibrant Emerald / Teal for living creatures & characters
            return (
                new Rgba32(65, 235, 195, 255),
                new Rgba32(18, 125, 105, 255),
                new Rgba32(10, 60, 50, 255),
                new Rgba32(230, 255, 250, 255));
        }

        if (pathLower.Contains("item") || pathLower.Contains("spells"))
        {
            // Royal Indigo / Violet for items, weapons, armor, spell effects
            return (
                new Rgba32(150, 140, 255, 255),
                new Rgba32(75, 65, 190, 255),
                new Rgba32(35, 25, 95, 255),
                new Rgba32(245, 240, 255, 255));
        }

        // Amber / Gold for Doodads, Props, Environments, World structures
        return (
            new Rgba32(255, 185, 50, 255),
            new Rgba32(175, 110, 20, 255),
            new Rgba32(85, 50, 10, 255),
            new Rgba32(255, 245, 190, 255));
    }

    private static Rgba32 LerpColor(Rgba32 a, Rgba32 b, float t)
    {
        t = Math.Clamp(t, 0f, 1f);
        byte r = (byte)(a.R + (b.R - a.R) * t);
        byte g = (byte)(a.G + (b.G - a.G) * t);
        byte bl = (byte)(a.B + (b.B - a.B) * t);
        byte al = (byte)(a.A + (b.A - a.A) * t);
        return new Rgba32(r, g, bl, al);
    }

    private static Rgba32 BlendColors(Rgba32 baseCol, Rgba32 overCol, float alpha)
    {
        alpha = Math.Clamp(alpha, 0f, 1f);
        byte r = (byte)(baseCol.R * (1f - alpha) + overCol.R * alpha);
        byte g = (byte)(baseCol.G * (1f - alpha) + overCol.G * alpha);
        byte b = (byte)(baseCol.B * (1f - alpha) + overCol.B * alpha);
        return new Rgba32(r, g, b, 255);
    }

    /// <summary>
    /// Stitches a seamless high-contrast bird's-eye map overview PNG containing all tiles and objects.
    /// </summary>
    public static void RenderAndSaveMapOverview(
        RosettaMapPlan map,
        string outputPath,
        int tileResolution = MinimapResolution)
    {
        ArgumentNullException.ThrowIfNull(map);
        if (map.Tiles.Count == 0)
            return;

        int minX = map.Tiles.Min(static t => t.TileX);
        int minY = map.Tiles.Min(static t => t.TileY);
        int maxX = map.Tiles.Max(static t => t.TileX);
        int maxY = map.Tiles.Max(static t => t.TileY);

        int totalWidth = (maxX - minX + 1) * tileResolution;
        int totalHeight = (maxY - minY + 1) * tileResolution;

        using var canvas = new Image<Rgba32>(totalWidth, totalHeight, ColorGround);

        foreach (RosettaTilePlan tile in map.Tiles)
        {
            using Image<Rgba32> tileImg = RenderTileImage(tile, tile.Pedestals, tile.AlphaCanvas);
            int destX = (tile.TileX - minX) * tileResolution;
            int destY = (tile.TileY - minY) * tileResolution;

            tileImg.ProcessPixelRows(canvas, (srcAcc, dstAcc) =>
            {
                for (int y = 0; y < tileResolution; y++)
                {
                    var srcRow = srcAcc.GetRowSpan(y);
                    var dstRow = dstAcc.GetRowSpan(destY + y);
                    for (int x = 0; x < tileResolution; x++)
                    {
                        dstRow[destX + x] = srcRow[x];
                    }
                }
            });
        }

        string? dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrWhiteSpace(dir))
            Directory.CreateDirectory(dir);

        canvas.SaveAsPng(outputPath);
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

    /// <summary>
    /// Generates the contents of a <c>minimap.trs</c> / <c>md5translate.trs</c> file mapping
    /// map directory tile requests to the corresponding generated minimap BLP files.
    /// </summary>
    public static string GenerateMinimapTrs(
        IReadOnlyList<RosettaMapPlan> maps,
        IReadOnlyList<string>? extraAliases = null)
    {
        ArgumentNullException.ThrowIfNull(maps);

        var sb = new System.Text.StringBuilder();

        foreach (RosettaMapPlan map in maps)
        {
            var directories = new List<string> { map.MapName, map.MapName.ToLowerInvariant() };
            if (extraAliases is not null)
            {
                foreach (string alias in extraAliases)
                {
                    if (!directories.Contains(alias, StringComparer.OrdinalIgnoreCase))
                        directories.Add(alias);
                }
            }

            foreach (string dir in directories)
            {
                sb.AppendLine($"dir: {dir}");
                foreach (RosettaTilePlan tile in map.Tiles)
                {
                    string unpaddedXY = $"map{tile.TileX}_{tile.TileY}.blp";
                    string paddedXY = $"map{tile.TileX:D2}_{tile.TileY:D2}.blp";
                    string unpaddedYX = $"map{tile.TileY}_{tile.TileX}.blp";
                    string paddedYX = $"map{tile.TileY:D2}_{tile.TileX:D2}.blp";

                    // 1. Direct relative entries under active dir:
                    sb.AppendLine($"{unpaddedXY}\t{paddedXY}");
                    if (!string.Equals(unpaddedXY, paddedXY, StringComparison.OrdinalIgnoreCase))
                        sb.AppendLine($"{paddedXY}\t{paddedXY}");
                    sb.AppendLine($"{unpaddedYX}\t{paddedYX}");
                    if (!string.Equals(unpaddedYX, paddedYX, StringComparison.OrdinalIgnoreCase))
                        sb.AppendLine($"{paddedYX}\t{paddedYX}");

                    // 2. Prefixed entries for tools that do not parse 'dir:' state
                    sb.AppendLine($@"{dir}\{unpaddedXY}	{dir}\{paddedXY}");
                    if (!string.Equals(unpaddedXY, paddedXY, StringComparison.OrdinalIgnoreCase))
                        sb.AppendLine($@"{dir}\{paddedXY}	{dir}\{paddedXY}");
                    sb.AppendLine($@"{dir}\{unpaddedYX}	{dir}\{paddedYX}");
                    if (!string.Equals(unpaddedYX, paddedYX, StringComparison.OrdinalIgnoreCase))
                        sb.AppendLine($@"{dir}\{paddedYX}	{dir}\{paddedYX}");
                }
                sb.AppendLine();
            }
        }

        return sb.ToString();
    }

    /// <summary>
    /// Writes <c>minimap.trs</c> and <c>md5translate.trs</c> to canonical Textures\Minimap directories.
    /// </summary>
    public static void WriteMinimapTrs(
        string outputRoot,
        IReadOnlyList<RosettaMapPlan> maps,
        IReadOnlyList<string>? extraAliases = null)
    {
        string trsContent = GenerateMinimapTrs(maps, extraAliases);

        var targets = new List<string>
        {
            Path.Combine(outputRoot, "Textures", "Minimap", "minimap.trs"),
            Path.Combine(outputRoot, "Textures", "Minimap", "md5translate.trs"),
        };

        foreach (string target in targets)
        {
            string? dir = Path.GetDirectoryName(target);
            if (!string.IsNullOrWhiteSpace(dir))
                Directory.CreateDirectory(dir);

            File.WriteAllText(target, trsContent, System.Text.Encoding.UTF8);
        }
    }
}
