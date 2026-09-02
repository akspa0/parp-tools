using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

/// <summary>
/// Measures the three per-vertex inputs to the terrain lighting equation, per client, so
/// "terrain is darker on non-0.5.3 eras" can be attributed to a term instead of guessed at.
/// </summary>
/// <remarks>
/// <para>
/// The terrain fragment shader computes
/// <c>lighting = uAmbientColor + uLightColor * vDiffuse * shadowVisibility</c> and then
/// <c>result *= clamp(vVertexColor.rgb * 2.0, 0.0, 2.0)</c>. Only three of those terms vary with the
/// client:
/// </para>
/// <list type="number">
/// <item><b>MCNR normals</b> feed <c>vDiffuse = max(dot(N, L), 0)</c>. Inverted normals drive it to
/// zero and leave only ambient. The signature is a <b>negative mean normal Z</b>: terrain mostly
/// faces up, so a correct tile has mean Z close to +1.</item>
/// <item><b>MCCV vertex colours</b> multiply the result. 0.5.3 has none and gets a neutral 1.0;
/// later eras have them, so a mean below 127/255 darkens those eras <b>and only those</b>.</item>
/// <item><b>MCSH shadows</b> subtract from the directional term.</item>
/// </list>
/// <para>
/// The mean-normal-Z check doubles as the detector's own proof: run it on a client whose terrain
/// looks right and it must come back strongly positive. A detector that cannot tell a correct tile
/// from an inverted one cannot support a null result either.
/// </para>
/// </remarks>
internal static class AdtTerrainShadingSupport
{
    public static void Run(string[] args)
    {
        string? clientRoot = GetOption(args, "--client");
        if (string.IsNullOrWhiteSpace(clientRoot))
        {
            Console.Error.WriteLine("Usage: adt terrain-shading --client <client-dir> [--map <name>] [--limit <n>] [--build <version>]");
            Environment.ExitCode = 1;
            return;
        }

        string? mapFilter = GetOption(args, "--map");
        string? buildVersion = GetOption(args, "--build");
        int limit = int.TryParse(GetOption(args, "--limit"), out int parsed) ? parsed : 20;

        using IArchiveCatalog archiveCatalog = new MpqArchiveCatalogFactory().Create();
        ArchiveCatalogBootstrapResult bootstrap = ArchiveCatalogBootstrapper.Bootstrap(
            archiveCatalog, [clientRoot], new ArchiveCatalogBootstrapOptions());

        IEnumerable<string> candidates = bootstrap.AllFiles
            .Where(static path => path.EndsWith(".adt", StringComparison.OrdinalIgnoreCase))
            .Where(static path =>
                !path.EndsWith("_obj0.adt", StringComparison.OrdinalIgnoreCase)
                && !path.EndsWith("_obj1.adt", StringComparison.OrdinalIgnoreCase)
                && !path.EndsWith("_tex0.adt", StringComparison.OrdinalIgnoreCase)
                && !path.EndsWith("_tex1.adt", StringComparison.OrdinalIgnoreCase));

        if (!string.IsNullOrWhiteSpace(mapFilter))
            candidates = candidates.Where(path => path.Contains(mapFilter, StringComparison.OrdinalIgnoreCase));

        var normalX = new Accumulator();
        var normalY = new Accumulator();
        var normalZ = new Accumulator();
        var normalLength = new Accumulator();
        var mccv = new Accumulator();
        var shadow = new Accumulator();
        var agreeXYZ = new Accumulator();   // slots as stored: (b0, b2, b1)
        var agreeRaw = new Accumulator();   // raw byte order:  (b0, b1, b2)
        var agreeRenderer = new Accumulator(); // what the terrain renderer actually feeds the GPU
        int scanned = 0;
        int withNormals = 0;
        int withMccv = 0;
        int withShadow = 0;
        int negativeZTiles = 0;

        void Accumulate(TerrainTileTensorPack pack)
        {
            // Only ~56% of the dense 257x257 grid is ever written (145 samples x 256 chunks), and
            // the arrays are zero-initialised. Counting the unwritten zeros makes any mean a
            // detector artifact rather than a measurement, so gate every statistic on the mask.
            bool[,]? written = pack.McnrMask257;

            if (pack.McnrNormalXyz is { } normals)
            {
                withNormals++;
                var tileZ = new Accumulator();
                int size0 = normals.GetLength(0);
                int size1 = normals.GetLength(1);
                for (int y = 0; y < size0; y++)
                {
                    for (int x = 0; x < size1; x++)
                    {
                        if (written is not null && !written[y, x])
                            continue;

                        float nx = normals[y, x, 0];
                        float ny = normals[y, x, 1];
                        float nz = normals[y, x, 2];
                        float length = MathF.Sqrt((nx * nx) + (ny * ny) + (nz * nz));
                        if (length < 1e-6f)
                            continue;

                        normalX.Add(nx / length);
                        normalY.Add(ny / length);
                        normalZ.Add(nz / length);
                        tileZ.Add(nz / length);
                        normalLength.Add(length);
                    }
                }

                if (tileZ.Count > 0 && tileZ.Mean < 0)
                    negativeZTiles++;
            }

            if (pack.MccvRgb is { } colours)
            {
                withMccv++;
                for (int y = 0; y < colours.GetLength(0); y++)
                {
                    for (int x = 0; x < colours.GetLength(1); x++)
                    {
                        if (written is not null && !written[y, x])
                            continue;

                        for (int channel = 0; channel < 3; channel++)
                            mccv.Add(colours[y, x, channel]);
                    }
                }
            }

            // Ground truth: derive the surface normal from the heightmap itself. This depends on no
            // byte-order convention at all, so it can arbitrate between the candidate decodes
            // instead of assuming one of them. A correct decode agrees strongly (mean dot near 1).
            if (pack.Height257 is { } heights && pack.McnrNormalXyz is { } stored && written is not null)
            {
                // The 257 grid interleaves outer and inner samples, so a +/-1 neighbour is always an
                // unwritten position. Step by 2 to stay on the outer lattice, whose spacing is one
                // MCVT cell: 533.333 world units per tile / 128 cells.
                const float sampleSpacing = 533.33333f / 128f;
                int size = heights.GetLength(0);
                for (int y = 2; y < size - 2; y += 2)
                {
                    for (int x = 2; x < size - 2; x += 2)
                    {
                        if (!written[y, x])
                            continue;

                        float dzdx = (heights[y, x + 2] - heights[y, x - 2]) / (2f * sampleSpacing);
                        float dzdy = (heights[y + 2, x] - heights[y - 2, x]) / (2f * sampleSpacing);
                        Vector3 geometric = Vector3.Normalize(new Vector3(-dzdx, -dzdy, 1f));

                        // Label by SLOT, not by a byte name: the storage order is exactly what is
                        // under test here, so any label that assumes one inverts its own meaning
                        // the moment the decode changes.
                        float s0 = stored[y, x, 0];
                        float s1 = stored[y, x, 1];
                        float s2 = stored[y, x, 2];

                        agreeXYZ.Add(AbsDot(new Vector3(s0, s1, s2), geometric));
                        agreeRaw.Add(AbsDot(new Vector3(s0, s2, s1), geometric));

                        // TransformAdtNormalToRenderer maps (x,y,z) -> (-y, -x, z).
                        agreeRenderer.Add(AbsDot(new Vector3(-s1, -s0, s2), geometric));
                    }
                }
            }

            if (pack.McshShadowMask256 is { } shadowMask)
            {
                withShadow++;
                for (int y = 0; y < shadowMask.GetLength(0); y++)
                {
                    for (int x = 0; x < shadowMask.GetLength(1); x++)
                        shadow.Add(shadowMask[y, x]);
                }
            }
        }

        // 0.5.3 keeps its terrain INSIDE the WDT rather than in loose .adt files, so an
        // .adt-only enumeration silently reports "0 scanned" on exactly the era we most need to
        // compare against. Read those tiles through the existing alpha path instead.
        var alphaCandidates = bootstrap.AllFiles
            .Where(static path => path.EndsWith(".wdt", StringComparison.OrdinalIgnoreCase))
            .Where(path => string.IsNullOrWhiteSpace(mapFilter) || path.Contains(mapFilter, StringComparison.OrdinalIgnoreCase))
            .OrderBy(static p => p, StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (string wdtPath in alphaCandidates)
        {
            if (scanned >= limit)
                break;

            byte[]? wdtBytes = archiveCatalog.ReadFile(wdtPath);
            if (wdtBytes is null || !AlphaWdtReader.IsAlphaWdt(wdtBytes))
                continue;

            foreach ((int tileX, int tileY) in AlphaWdtReader.ReadExistingTiles(wdtBytes).OrderBy(static t => (t.X, t.Y)))
            {
                if (scanned >= limit)
                    break;

                if (!AlphaWdtReader.TryReadTile(wdtBytes, tileX, tileY, wdtPath, out AlphaTileData? tileData) || tileData is null)
                    continue;

                TerrainTileTensorPack alphaPack;
                try
                {
                    alphaPack = AlphaTensorPackBuilder.Build(tileData, tileX, tileY);
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"  skip {wdtPath} ({tileX},{tileY}): {ex.Message}");
                    continue;
                }

                scanned++;
                Accumulate(alphaPack);
            }
        }

        foreach (string path in candidates.OrderBy(static p => p, StringComparer.OrdinalIgnoreCase))
        {
            if (scanned >= limit)
                break;

            byte[]? bytes = archiveCatalog.ReadFile(path);
            if (bytes is null)
                continue;

            TerrainTileTensorPack pack;
            try
            {
                // Split Cata+/MoP ADTs can carry MCCV on the texture companion rather than the
                // root, so a root-only read would under-report it on exactly the eras in question.
                string texPath = path[..^4] + "_tex0.adt";
                byte[]? texBytes = archiveCatalog.ReadFile(texPath);

                pack = AdtTensorPackBuilder.BuildFromBytes(
                    path,
                    bytes,
                    texBytes,
                    placementSourceBytes: null,
                    buildVersion,
                    texBytes is null ? null : texPath,
                    placementSourcePath: null,
                    archiveCatalog.ReadFile);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"  skip {path}: {ex.Message}");
                continue;
            }

            scanned++;

            Accumulate(pack);
        }

        Console.WriteLine("WowViewer.Tool.Inspect terrain shading inputs");
        Console.WriteLine($"client={clientRoot} map={mapFilter ?? "(all)"}");
        Console.WriteLine($"root ADTs scanned={scanned} withMCNR={withNormals} withMCCV={withMccv} withMCSH={withShadow}");
        Console.WriteLine();

        Console.WriteLine("--- 1. MCNR normals -> vDiffuse = max(dot(N, L), 0)");
        if (normalZ.Count == 0)
        {
            Console.WriteLine("    NO NORMALS DECODED. vDiffuse is 0 everywhere and only ambient lights the terrain.");
        }
        else
        {
            Console.WriteLine($"    mean component X = {normalX.Mean:0.0000}");
            Console.WriteLine($"    mean component Y = {normalY.Mean:0.0000}");
            Console.WriteLine($"    mean component Z = {normalZ.Mean:0.0000}  <- the UP axis; correct terrain faces up, expect near +1");
            Console.WriteLine("    (whichever component is near +1 is the real up axis. If it is not Z, the decode swaps axes.)");
            Console.WriteLine($"    mean |N|      = {normalLength.Mean:0.0000}  (expect ~1.0; far from 1 means a decode/scale fault)");
            Console.WriteLine($"    tiles with mean Z < 0: {negativeZTiles} of {withNormals}");
            Console.WriteLine(normalZ.Mean < 0
                ? "    VERDICT: normals point DOWN on average -> INVERTED. This alone would kill the directional term."
                : normalZ.Mean < 0.5
                    ? "    VERDICT: normals are not inverted but are unexpectedly flat/scrambled; inspect the interleave."
                    : "    VERDICT: normals are upward-facing and normalized. NOT inverted.");
        }

        Console.WriteLine();
        Console.WriteLine("--- 2. MCCV vertex colours -> result *= clamp(rgb * 2, 0, 2)");
        if (mccv.Count == 0)
        {
            Console.WriteLine("    NO MCCV. The shader substitutes a neutral 1.0 tint, so MCCV cannot darken this client.");
        }
        else
        {
            // The builder stores MCCV normalized to 0..1, so 0.5 is the neutral byte 127/255.
            double tint = mccv.Mean * 2.0;
            Console.WriteLine($"    mean MCCV channel = {mccv.Mean:0.0000} (neutral is 0.4980 = byte 127)");
            Console.WriteLine($"    => shader tint multiplier = {tint:0.0000}");
            Console.WriteLine(tint < 0.95
                ? $"    VERDICT: MCCV DARKENS this client by {100.0 * (1.0 - tint):0.0}%. Eras without MCCV are unaffected, which is exactly an era-scoped darkness."
                : "    VERDICT: MCCV is neutral or brightening; it is not the source of darkness.");
        }

        Console.WriteLine();
        Console.WriteLine("--- 1b. Which decode agrees with normals derived from the HEIGHTMAP? (convention-free ground truth)");
        if (agreeRaw.Count == 0)
        {
            Console.WriteLine("    no comparable samples.");
        }
        else
        {
            Console.WriteLine($"    samples compared           : {agreeRaw.Count}");
            Console.WriteLine($"    slots as stored (0,1,2)    : mean dot {agreeXYZ.Mean:0.0000}   <- the decode in force");
            Console.WriteLine($"    slots swapped   (0,2,1)    : mean dot {agreeRaw.Mean:0.0000}");
            Console.WriteLine($"    AS THE RENDERER FEEDS IT   : mean dot {agreeRenderer.Mean:0.0000}   <- this is what lights the terrain");
            Console.WriteLine("    (mean dot near 1.0 = agrees with the real surface; near 0 = perpendicular to it)");
        }

        Console.WriteLine();
        Console.WriteLine("--- 3. MCSH shadow -> multiplies the directional term only");
        Console.WriteLine(shadow.Count == 0
            ? "    NO MCSH decoded."
            : $"    mean shadow mask = {shadow.Mean:0.0000} (1.0 = fully shadowed); covers {withShadow} of {scanned} tiles");
    }

    private static double AbsDot(Vector3 candidate, Vector3 geometric)
    {
        float length = candidate.Length();
        if (length < 1e-6f)
            return 0.0;

        return Math.Abs(Vector3.Dot(candidate / length, geometric));
    }

    private sealed class Accumulator
    {
        private double _sum;

        public int Count { get; private set; }

        public double Mean => Count == 0 ? 0.0 : _sum / Count;

        public void Add(double value)
        {
            _sum += value;
            Count++;
        }
    }

    private static string? GetOption(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }

        return null;
    }
}
