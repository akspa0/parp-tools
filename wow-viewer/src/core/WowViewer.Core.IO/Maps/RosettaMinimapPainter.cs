using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Blp;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Generates crisp, high-contrast 256x256 minimap tiles for Rosetta calibration maps.
/// Renders cell boundaries, museum pedestals, antialiased text labels, and asset center markers.
/// </summary>
public static class RosettaMinimapPainter
{
    public const int MinimapResolution = 256;
    private const float TileWorldSize = RosettaGeneratorOptions.TileSize; // 533.33333m

    private static readonly Rgba32 ColorGround = new(36, 52, 58, 255);
    private static readonly Rgba32 ColorCellBorder = new(48, 70, 80, 255);
    private static readonly Rgba32 ColorPedestalBevel = new(64, 84, 94, 255);
    private static readonly Rgba32 ColorPedestalFlat = new(112, 126, 134, 255);
    private static readonly Rgba32 ColorPedestalHighlight = new(150, 168, 178, 255);
    private static readonly Rgba32 ColorInk = new(12, 16, 20, 255);
    private static readonly Rgba32 ColorModelDot = new(70, 215, 255, 255);
    private static readonly Rgba32 ColorModelCenter = new(240, 250, 255, 255);
    private static readonly Rgba32 ColorWmoBox = new(255, 175, 45, 255);
    private static readonly Rgba32 ColorWmoBorder = new(180, 105, 15, 255);

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

        // 1. Fill ground base
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < MinimapResolution; y++)
            {
                var row = accessor.GetRowSpan(y);
                row.Fill(ColorGround);
            }
        });

        // 2. Render Pedestals
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
                        image[x, y] = checker ? ColorPedestalFlat : ColorPedestalBevel;
                    }
                }

                // Top & left bevel highlight
                for (int x = px0; x <= px1; x++)
                    image[x, py0] = ColorPedestalHighlight;
                for (int y = py0; y <= py1; y++)
                    image[px0, y] = ColorPedestalHighlight;
            }
        }

        // 3. Render Text from downsampled 1024x1024 MCAL canvas
        byte[]? effectiveCanvas = alphaCanvas;
        if (effectiveCanvas == null && tile.Placements.Any(static p => p.LabelLines.Count > 0))
        {
            effectiveCanvas = RosettaAlphaPainter.CreateCanvas();
            foreach (RosettaPlacementRecord placement in tile.Placements)
            {
                if (placement.LabelLines.Count == 0)
                    continue;

                float pixel = placement.LabelPixelMeters;
                float bandV0 = placement.CellV + placement.ObjectBandSize;
                float lineAdvance = RosettaAlphaPainter.LineAdvanceMeters(pixel);
                for (int line = 0; line < placement.LabelLines.Count; line++)
                {
                    string text = placement.LabelLines[line];
                    if (text.Length == 0)
                        continue;

                    float width = RosettaAlphaPainter.MeasureWidthMeters(text, pixel);
                    float inset = MathF.Floor(MathF.Max(0f, (placement.CellSize - width) / 2f) / pixel) * pixel;
                    RosettaAlphaPainter.DrawText(
                        effectiveCanvas, text, placement.CellU + inset, bandV0 + (line * lineAdvance),
                        pixel, ink: 255, RosettaGeneratorOptions.ChunkSize);
                }
            }
        }

        if (effectiveCanvas != null && effectiveCanvas.Length >= RosettaAlphaPainter.TexelsPerTile * RosettaAlphaPainter.TexelsPerTile)
        {
            int factor = RosettaAlphaPainter.TexelsPerTile / MinimapResolution; // 1024 / 256 = 4

            for (int y = 0; y < MinimapResolution; y++)
            {
                int srcY0 = y * factor;
                for (int x = 0; x < MinimapResolution; x++)
                {
                    int srcX0 = x * factor;
                    int sum = 0;
                    for (int dy = 0; dy < factor; dy++)
                    {
                        int rowOffset = (srcY0 + dy) * RosettaAlphaPainter.TexelsPerTile;
                        for (int dx = 0; dx < factor; dx++)
                        {
                            sum += effectiveCanvas[rowOffset + srcX0 + dx];
                        }
                    }

                    int avgAlpha = sum / (factor * factor);
                    if (avgAlpha > 0)
                    {
                        float t = Math.Min(1f, (avgAlpha / 255f) * 1.5f); // slight contrast boost
                        Rgba32 current = image[x, y];
                        byte r = (byte)((current.R * (1f - t)) + (ColorInk.R * t));
                        byte g = (byte)((current.G * (1f - t)) + (ColorInk.G * t));
                        byte b = (byte)((current.B * (1f - t)) + (ColorInk.B * t));
                        image[x, y] = new Rgba32(r, g, b, 255);
                    }
                }
            }
        }

        // 4. Render Placement Pins (Cyan dot for M2, Orange box for WMO)
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

        // 5. Subtle 1px outer tile border
        for (int i = 0; i < MinimapResolution; i++)
        {
            image[i, 0] = ColorCellBorder;
            image[i, MinimapResolution - 1] = ColorCellBorder;
            image[0, i] = ColorCellBorder;
            image[MinimapResolution - 1, i] = ColorCellBorder;
        }

        return image;
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
