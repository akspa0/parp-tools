using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

/// <summary>
/// Reports the per-channel distribution of MCCV vertex colours in an ADT.
/// </summary>
/// <remarks>
/// Exists to settle a specific rendering question: the terrain shader derives its tint STRENGTH from
/// the MCCV alpha byte (<c>clamp(a * 2 - 1, 0, 1)</c>), so a file whose alpha sits at the neutral 127
/// renders with zero tint no matter what its RGB holds. Distinguishing "the colours are absent" from
/// "the colours are present but the strength gate zeroes them" needs the alpha distribution, not a
/// presence count.
/// </remarks>
internal static class AdtMccvStatsSupport
{
    public static void Run(string[] args)
    {
        string? input = GetOpt(args, "--input") ?? args.FirstOrDefault(static a => !a.StartsWith('-'));
        if (string.IsNullOrWhiteSpace(input) || !File.Exists(input))
        {
            Console.Error.WriteLine("Error: --input <file.adt> is required.");
            Environment.ExitCode = 1;
            return;
        }

        string? tex0 = ProbeCompanion(input, "_tex0");
        string? obj0 = ProbeCompanion(input, "_obj0");

        LkAdtData adt = LkAdtReader.Read(
            File.ReadAllBytes(input),
            tex0 is null ? null : File.ReadAllBytes(tex0),
            obj0 is null ? null : File.ReadAllBytes(obj0),
            0, 0);

        long[] min = [255, 255, 255, 255];
        long[] max = [0, 0, 0, 0];
        long[] sum = [0, 0, 0, 0];
        long samples = 0;
        int chunksWithMccv = 0;
        var alphaHistogram = new SortedDictionary<byte, long>();

        foreach (LkMcnkData chunk in adt.Chunks)
        {
            byte[]? c = chunk.MccvColors;
            if (c is null || c.Length < 145 * 4)
                continue;

            chunksWithMccv++;
            for (int v = 0; v + 3 < c.Length; v += 4)
            {
                samples++;
                for (int ch = 0; ch < 4; ch++)
                {
                    byte b = c[v + ch];
                    if (b < min[ch]) min[ch] = b;
                    if (b > max[ch]) max[ch] = b;
                    sum[ch] += b;
                }
                alphaHistogram[c[v + 3]] = alphaHistogram.GetValueOrDefault(c[v + 3]) + 1;
            }
        }

        Console.WriteLine("WowViewer.Tool.Inspect ADT MCCV statistics");
        Console.WriteLine($"Input: {input}");
        Console.WriteLine($"  tex0 companion: {tex0 ?? "(none)"}");
        Console.WriteLine($"Chunks={adt.Chunks.Count} withMccv={chunksWithMccv} vertexSamples={samples}");
        if (samples == 0)
        {
            Console.WriteLine("  no MCCV payloads found.");
            return;
        }

        string[] names = ["B (byte 0)", "G (byte 1)", "R (byte 2)", "A (byte 3)"];
        Console.WriteLine();
        Console.WriteLine("  channel        min    max    mean");
        for (int ch = 0; ch < 4; ch++)
            Console.WriteLine($"  {names[ch],-12} {min[ch],5} {max[ch],6}  {(double)sum[ch] / samples,6:F2}");

        Console.WriteLine();
        double neutralTint = alphaHistogram.Where(static kv => kv.Key <= 127).Sum(static kv => kv.Value);
        Console.WriteLine($"  alpha <= 127 (shader tintStrength == 0): {neutralTint} / {samples} ({neutralTint / samples:P2})");
        Console.WriteLine("  top alpha values:");
        foreach ((byte value, long count) in alphaHistogram.OrderByDescending(static kv => kv.Value).Take(6))
            Console.WriteLine($"    a={value,-4} {count,10}  tintStrength={Math.Clamp(value / 255.0 * 2.0 - 1.0, 0.0, 1.0):F3}");
    }

    private static string? ProbeCompanion(string rootPath, string suffix)
    {
        string dir = Path.GetDirectoryName(rootPath) ?? string.Empty;
        string stem = Path.GetFileNameWithoutExtension(rootPath);
        string candidate = Path.Combine(dir, stem + suffix + ".adt");
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? GetOpt(string[] args, string name)
    {
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (string.Equals(args[i], name, StringComparison.OrdinalIgnoreCase))
                return args[i + 1];
        }
        return null;
    }
}
