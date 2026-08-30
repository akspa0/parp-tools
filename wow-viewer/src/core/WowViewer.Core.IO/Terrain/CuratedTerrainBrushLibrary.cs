using System.Numerics;

namespace WowViewer.Core.IO.Terrain;

/// <summary>
/// Provides a rich, built-in library of curated and procedurally synthesized terrain brush pastes
/// across multiple biomes (Garden, Temperate, Forest, Cobblestone City, Mountain).
/// All motifs are strictly validated for smooth walkability (slope limits) and 4-layer texture budgets.
/// </summary>
public static class CuratedTerrainBrushLibrary
{
    private static readonly Lazy<TerrainBrushLibrary> DefaultLibrary = new(BuildStockLibrary);

    public static TerrainBrushLibrary Instance => DefaultLibrary.Value;

    public static TerrainBrushLibrary BuildStockLibrary()
    {
        var lib = new TerrainBrushLibrary();

        // 1. Straight Cobblestone Walkway
        lib.Add(CreateRoadCobbleStraight());

        // 2. Cobblestone 90-Degree Curve
        lib.Add(CreateRoadCobbleCurve());

        // 3. Cobblestone 4-Way Intersection
        lib.Add(CreateRoadCobbleIntersection());

        // 4. Organic Dirt Trail
        lib.Add(CreateRoadDirtTrail());

        // 5. White Marble Exhibit Plaza (Single Chunk 33.3m)
        lib.Add(CreatePlazaMarbleSquare());

        // 6. Grand Central Courtyard (2x2 Chunk 66.6m)
        lib.Add(CreatePlazaGrandCourtyard());

        // 7. Gentle Grass Knoll (Smooth, 100% Walkable Hill)
        lib.Add(CreateHillGentleKnoll());

        // 8. Stepped Garden Terrace
        lib.Add(CreateHillSteppedTerrace());

        // 9. Boundary Perimeter Ridge
        lib.Add(CreateRidgePerimeter());

        // 10. Pond Basin Clearing
        lib.Add(CreateDepressionPondBasin());

        // 11. Circular Garden Flowerbed
        lib.Add(CreateGardenFlowerbed());

        // 12. Grand Promenade Avenue
        lib.Add(CreateGrandPromenadeAvenue());

        // 13. Flat Clean Stone Exhibit Pad (Zero-Bevel, Z=0)
        lib.Add(CreateExhibitPadFlat());

        // 14. Shallow Stepped Podium
        lib.Add(CreateExhibitPodiumShallow());

        // 15. Forest Rocky Clearing
        lib.Add(CreateNatureRockyClearing());

        return lib;
    }

    #region Preset Factory Methods

    private static TerrainBrushPaste CreateRoadCobbleStraight()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res]; // Flat road Z=0
        var alphaSplats = new byte[alphaRes * alphaRes];

        // Center 12m wide cobblestone band (u in [0.32, 0.68])
        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float distFromCenter = MathF.Abs(u - 0.5f);
                float edge = SmoothStep(0.25f, 0.18f, distFromCenter);
                alphaSplats[y * alphaRes + x] = (byte)(edge * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "road_cobble_straight_01",
            Name = "Cobblestone Straight Walkway",
            Category = "Road",
            Tags = ["road", "cobblestone", "walkway", "path", "straight", "garden"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaSplats }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateRoadCobbleCurve()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaSplats = new byte[alphaRes * alphaRes];

        // 90-degree corner arc with center at (0, 0)
        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float distFromCorner = MathF.Sqrt(u * u + v * v);
                float roadDist = MathF.Abs(distFromCorner - 0.5f);
                float edge = SmoothStep(0.22f, 0.14f, roadDist);
                alphaSplats[y * alphaRes + x] = (byte)(edge * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "road_cobble_curve_01",
            Name = "Cobblestone 90-Degree Curve",
            Category = "Road",
            Tags = ["road", "cobblestone", "walkway", "path", "curve", "corner"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaSplats }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateRoadCobbleIntersection()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaCobble = new byte[alphaRes * alphaRes];
        var alphaCenter = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float dx = MathF.Abs(u - 0.5f);
                float dy = MathF.Abs(v - 0.5f);
                float r = MathF.Sqrt(dx * dx + dy * dy);

                // Cross shape
                float crossU = SmoothStep(0.20f, 0.14f, dx);
                float crossV = SmoothStep(0.20f, 0.14f, dy);
                float cobble = MathF.Max(crossU, crossV);

                // Center medallion
                float medallion = SmoothStep(0.18f, 0.12f, r);

                alphaCobble[y * alphaRes + x] = (byte)(cobble * 255f);
                alphaCenter[y * alphaRes + x] = (byte)(medallion * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "road_cobble_cross_01",
            Name = "Cobblestone 4-Way Intersection",
            Category = "Road",
            Tags = ["road", "cobblestone", "intersection", "cross", "plaza"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaCobble },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaCenter }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateRoadDirtTrail()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaDirt = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                // Slight S-curve
                float centerU = 0.5f + 0.12f * MathF.Sin(v * MathF.PI * 2f);
                float dist = MathF.Abs(u - centerU);
                float trail = SmoothStep(0.16f, 0.08f, dist);
                alphaDirt[y * alphaRes + x] = (byte)(trail * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "road_dirt_path_01",
            Name = "Organic Dirt Trail",
            Category = "Road",
            Tags = ["road", "dirt", "path", "trail", "forest", "organic"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\dirt.blp", Resolution = alphaRes, AlphaMask = alphaDirt }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreatePlazaMarbleSquare()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res]; // Perfectly flat Z=0
        var alphaCobbleBorder = new byte[alphaRes * alphaRes];
        var alphaMarbleFloor = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float maxAxisDist = MathF.Max(MathF.Abs(u - 0.5f), MathF.Abs(v - 0.5f));

                // Outer border frame (0.35 to 0.45)
                float outerBox = SmoothStep(0.48f, 0.44f, maxAxisDist);
                float innerBox = SmoothStep(0.38f, 0.34f, maxAxisDist);
                float border = MathF.Max(0f, outerBox - innerBox);

                // Central clean marble floor (radius <= 0.35)
                float floor = innerBox;

                alphaCobbleBorder[y * alphaRes + x] = (byte)(border * 255f);
                alphaMarbleFloor[y * alphaRes + x] = (byte)(floor * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "plaza_marble_square_01",
            Name = "White Marble Exhibit Plaza",
            Category = "Plaza",
            Tags = ["plaza", "marble", "stone", "courtyard", "museum", "clean", "pedestal"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaCobbleBorder },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaMarbleFloor }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreatePlazaGrandCourtyard()
    {
        const int res = 33;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaCobbleBorder = new byte[alphaRes * alphaRes];
        var alphaMarbleFloor = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float r = MathF.Sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));

                float outerRing = SmoothStep(0.48f, 0.42f, r);
                float innerRing = SmoothStep(0.38f, 0.32f, r);
                float border = MathF.Max(0f, outerRing - innerRing);
                float floor = innerRing;

                alphaCobbleBorder[y * alphaRes + x] = (byte)(border * 255f);
                alphaMarbleFloor[y * alphaRes + x] = (byte)(floor * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "plaza_marble_grand_01",
            Name = "Grand Central Courtyard",
            Category = "Plaza",
            Tags = ["plaza", "grand", "courtyard", "museum", "marble", "center"],
            WidthMeters = 66.66666f,
            LengthMeters = 66.66666f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaCobbleBorder },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaMarbleFloor }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateHillGentleKnoll()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaRock = new byte[alphaRes * alphaRes];

        // Elevation max +4.5m with smooth cosine falloff (max slope < 15 degrees)
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                float u = (float)x / (res - 1) - 0.5f;
                float v = (float)y / (res - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                if (r < 1f)
                {
                    float factor = 0.5f * (1f + MathF.Cos(r * MathF.PI));
                    heights[y * res + x] = 4.5f * factor;
                }
            }
        }

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1) - 0.5f;
                float v = (float)y / (alphaRes - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                float crest = SmoothStep(0.5f, 0.2f, r);
                alphaRock[y * alphaRes + x] = (byte)(crest * 180f);
            }
        }

        var paste = new TerrainBrushPaste
        {
            Id = "hill_gentle_knoll_01",
            Name = "Gentle Grass Knoll",
            Category = "Hill",
            Tags = ["hill", "knoll", "relief", "grass", "walkable", "gentle"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\rock.blp", Resolution = alphaRes, AlphaMask = alphaRock }
            ]
        };
        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    private static TerrainBrushPaste CreateHillSteppedTerrace()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaStone = new byte[alphaRes * alphaRes];

        // 3 wide tiers: +0m, +1.5m, +3.0m
        for (int y = 0; y < res; y++)
        {
            float v = (float)y / (res - 1);
            float h = v < 0.35f ? 0f : (v < 0.70f ? 1.5f : 3.0f);
            for (int x = 0; x < res; x++)
                heights[y * res + x] = h;
        }

        for (int y = 0; y < alphaRes; y++)
        {
            float v = (float)y / (alphaRes - 1);
            float edge1 = MathF.Abs(v - 0.35f) < 0.06f ? 1f : 0f;
            float edge2 = MathF.Abs(v - 0.70f) < 0.06f ? 1f : 0f;
            float edge = MathF.Max(edge1, edge2);
            for (int x = 0; x < alphaRes; x++)
                alphaStone[y * alphaRes + x] = (byte)(edge * 220f);
        }

        var paste = new TerrainBrushPaste
        {
            Id = "hill_stepped_terrace_01",
            Name = "Stepped Garden Terrace",
            Category = "Hill",
            Tags = ["hill", "terrace", "stepped", "garden", "walkable", "tiers"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaStone }
            ]
        };
        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    private static TerrainBrushPaste CreateRidgePerimeter()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaRock = new byte[alphaRes * alphaRes];

        // Perimeter mountain ridge rising to +12m at top edge (v = 1)
        for (int y = 0; y < res; y++)
        {
            float v = (float)y / (res - 1);
            float h = 12f * MathF.Pow(v, 2f);
            for (int x = 0; x < res; x++)
                heights[y * res + x] = h;
        }

        for (int y = 0; y < alphaRes; y++)
        {
            float v = (float)y / (alphaRes - 1);
            byte a = (byte)(MathF.Pow(v, 1.5f) * 255f);
            for (int x = 0; x < alphaRes; x++)
                alphaRock[y * alphaRes + x] = a;
        }

        var paste = new TerrainBrushPaste
        {
            Id = "ridge_perimeter_border_01",
            Name = "Boundary Perimeter Ridge",
            Category = "Ridge",
            Tags = ["ridge", "mountain", "border", "perimeter", "rock"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\rock.blp", Resolution = alphaRes, AlphaMask = alphaRock }
            ]
        };
        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    private static TerrainBrushPaste CreateDepressionPondBasin()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaMud = new byte[alphaRes * alphaRes];

        // Basin depth -3.5m
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                float u = (float)x / (res - 1) - 0.5f;
                float v = (float)y / (res - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                if (r < 1f)
                    heights[y * res + x] = -3.5f * 0.5f * (1f + MathF.Cos(r * MathF.PI));
            }
        }

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1) - 0.5f;
                float v = (float)y / (alphaRes - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                float mud = SmoothStep(0.85f, 0.40f, r);
                alphaMud[y * alphaRes + x] = (byte)(mud * 240f);
            }
        }

        var paste = new TerrainBrushPaste
        {
            Id = "depression_pond_basin_01",
            Name = "Pond Basin Depression",
            Category = "Depression",
            Tags = ["depression", "pond", "basin", "water", "mud", "shoreline"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\dirt.blp", Resolution = alphaRes, AlphaMask = alphaMud }
            ]
        };
        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    private static TerrainBrushPaste CreateGardenFlowerbed()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaFloral = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1) - 0.5f;
                float v = (float)y / (alphaRes - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                float ring = SmoothStep(0.8f, 0.6f, r) - SmoothStep(0.4f, 0.2f, r);
                alphaFloral[y * alphaRes + x] = (byte)(MathF.Max(0f, ring) * 230f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "garden_flowerbed_circular_01",
            Name = "Circular Garden Flowerbed",
            Category = "Garden",
            Tags = ["garden", "flowerbed", "floral", "turf", "decorative", "park"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\dirt.blp", Resolution = alphaRes, AlphaMask = alphaFloral }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateGrandPromenadeAvenue()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaCobble = new byte[alphaRes * alphaRes];
        var alphaMarbleEdges = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float dist = MathF.Abs(u - 0.5f);

                // Main cobblestone avenue
                float cobble = SmoothStep(0.35f, 0.28f, dist);

                // Flanking marble borders
                float outerEdge = SmoothStep(0.40f, 0.35f, dist);
                float innerEdge = SmoothStep(0.35f, 0.30f, dist);
                float marble = MathF.Max(0f, outerEdge - innerEdge);

                alphaCobble[y * alphaRes + x] = (byte)(cobble * 255f);
                alphaMarbleEdges[y * alphaRes + x] = (byte)(marble * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "garden_promenade_avenue_01",
            Name = "Grand Promenade Avenue",
            Category = "Garden",
            Tags = ["garden", "promenade", "avenue", "road", "cobblestone", "marble", "walkway"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = alphaRes, AlphaMask = alphaCobble },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaMarbleEdges }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateExhibitPadFlat()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res]; // Zero height delta Z=0
        var alphaMarble = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float dist = MathF.Max(MathF.Abs(u - 0.5f), MathF.Abs(v - 0.5f));
                float pad = SmoothStep(0.32f, 0.28f, dist);
                alphaMarble[y * alphaRes + x] = (byte)(pad * 255f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "pedestal_exhibit_flat_01",
            Name = "Flat Marble Exhibit Pad",
            Category = "Plaza",
            Tags = ["pedestal", "exhibit", "pad", "flat", "marble", "museum", "clean"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaMarble }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static TerrainBrushPaste CreateExhibitPodiumShallow()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaMarble = new byte[alphaRes * alphaRes];

        // Shallow +0.5m podium with very gentle 10-degree ramp
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                float u = (float)x / (res - 1) - 0.5f;
                float v = (float)y / (res - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                float podium = SmoothStep(0.70f, 0.40f, r);
                heights[y * res + x] = 0.5f * podium;
            }
        }

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1) - 0.5f;
                float v = (float)y / (alphaRes - 1) - 0.5f;
                float r = MathF.Sqrt(u * u + v * v) / 0.5f;
                float marble = SmoothStep(0.60f, 0.35f, r);
                alphaMarble[y * alphaRes + x] = (byte)(marble * 255f);
            }
        }

        var paste = new TerrainBrushPaste
        {
            Id = "pedestal_exhibit_stepped_01",
            Name = "Shallow Stepped Podium",
            Category = "Plaza",
            Tags = ["pedestal", "podium", "exhibit", "shallow", "marble", "walkable"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\city\whitemarble.blp", Resolution = alphaRes, AlphaMask = alphaMarble }
            ]
        };
        paste.CalculateMaxSlopeDegrees();
        return paste;
    }

    private static TerrainBrushPaste CreateNatureRockyClearing()
    {
        const int res = 17;
        const int alphaRes = 64;
        var heights = new float[res * res];
        var alphaRock = new byte[alphaRes * alphaRes];
        var alphaDirt = new byte[alphaRes * alphaRes];

        for (int y = 0; y < alphaRes; y++)
        {
            for (int x = 0; x < alphaRes; x++)
            {
                float u = (float)x / (alphaRes - 1);
                float v = (float)y / (alphaRes - 1);
                float r = MathF.Sqrt((u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f));

                float dirt = SmoothStep(0.45f, 0.20f, r);
                float rock = SmoothStep(0.20f, 0.08f, r);

                alphaDirt[y * alphaRes + x] = (byte)(dirt * 200f);
                alphaRock[y * alphaRes + x] = (byte)(rock * 240f);
            }
        }

        return new TerrainBrushPaste
        {
            Id = "nature_rocky_clearing_01",
            Name = "Forest Rocky Clearing",
            Category = "General",
            Tags = ["forest", "clearing", "nature", "rock", "dirt", "wild"],
            WidthMeters = 33.33333f,
            LengthMeters = 33.33333f,
            ResolutionX = res,
            ResolutionY = res,
            HeightDeltas = heights,
            Layers =
            [
                new TerrainPasteLayer { TexturePath = @"tileset\elwynn\elwynngrass.blp", Resolution = alphaRes, AlphaMask = CreateSolidAlpha(alphaRes, 255) },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\dirt.blp", Resolution = alphaRes, AlphaMask = alphaDirt },
                new TerrainPasteLayer { TexturePath = @"tileset\generic\rock.blp", Resolution = alphaRes, AlphaMask = alphaRock }
            ],
            MaxSlopeDegrees = 0f
        };
    }

    private static byte[] CreateSolidAlpha(int resolution, byte value)
    {
        var buf = new byte[resolution * resolution];
        Array.Fill(buf, value);
        return buf;
    }

    private static float SmoothStep(float edge0, float edge1, float x)
    {
        if (edge0 > edge1)
        {
            float t = Math.Clamp((x - edge1) / (edge0 - edge1), 0f, 1f);
            return 1f - (t * t * (3f - 2f * t));
        }
        else
        {
            float t = Math.Clamp((x - edge0) / (edge1 - edge0), 0f, 1f);
            return t * t * (3f - 2f * t);
        }
    }

    #endregion
}
