namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Spec 237: DAT v26 <c>AMAP</c> stores per-layer blend <b>weights</b> (layer 0 included) that sum to 255 per pixel,
/// not the ADT's sequential alpha. MEASURED on the corpus: per-pixel sums average 254.6 (range 252.6..255.0 per chunk,
/// exactly 255 for 74% of pixels), and the highest-weight layer per 8×8 cell matches the ACNK predominant-layer map in
/// 99.93% of 239,808 cells versus 95.03% when AMAP is treated as sequential alpha (script acnk_predominant_v26.py).
/// <para>
/// ADT blending: result = layer0; for i = 1..n-1: result = mix(result, layer_i, a_i). The final weight of layer i is
/// a_i × Π(1 − a_j, j &gt; i), layer 0's is Π(1 − a_j). These helpers convert between the two representations.
/// </para>
/// </summary>
public static class AdtAhdrAlpha
{
    public const int Pixels = 64 * 64;

    /// <summary>Converts per-layer weights (one 64×64 map per layer, layer 0 first) to sequential alpha maps for layers 1..n-1.</summary>
    public static byte[][] WeightsToSequentialAlpha(IReadOnlyList<byte[]> weights)
    {
        int layers = weights.Count;
        var alpha = new byte[Math.Max(0, layers - 1)][];
        for (int i = 0; i < alpha.Length; i++)
            alpha[i] = new byte[Pixels];

        for (int p = 0; p < Pixels; p++)
        {
            // a_i = w_i / (w_0 + ... + w_i): the share of layer i in everything drawn up to and including it.
            int running = weights[0][p];
            for (int i = 1; i < layers; i++)
            {
                int w = weights[i][p];
                running += w;
                alpha[i - 1][p] = running <= 0 ? (byte)0 : (byte)Math.Clamp((int)Math.Round(w * 255.0 / running), 0, 255);
            }
        }

        return alpha;
    }

    /// <summary>
    /// Converts sequential alpha maps for layers 1..<paramref name="layerCount"/>-1 (null = fully opaque, as for a layer
    /// without an alpha map) to per-layer weights for all layers, rounded so each pixel sums to 255.
    /// </summary>
    public static byte[][] SequentialAlphaToWeights(int layerCount, IReadOnlyList<byte[]?> alpha)
    {
        var weights = new byte[layerCount][];
        for (int i = 0; i < layerCount; i++)
            weights[i] = new byte[Pixels];

        Span<double> exact = stackalloc double[Math.Max(1, layerCount)];
        for (int p = 0; p < Pixels; p++)
        {
            double remaining = 1.0;
            for (int i = layerCount - 1; i >= 1; i--)
            {
                byte[]? map = i - 1 < alpha.Count ? alpha[i - 1] : null;
                double a = map is null ? 1.0 : map[p] / 255.0;
                exact[i] = a * remaining;
                remaining *= 1.0 - a;
            }

            exact[0] = remaining;
            int sum = 0, largest = 0;
            for (int i = 0; i < layerCount; i++)
            {
                int value = (int)Math.Round(exact[i] * 255.0);
                weights[i][p] = (byte)Math.Clamp(value, 0, 255);
                sum += weights[i][p];
                if (exact[i] > exact[largest])
                    largest = i;
            }

            // Put the rounding remainder on the dominant layer so the pixel sums to 255, as in the observed files.
            weights[largest][p] = (byte)Math.Clamp(weights[largest][p] + (255 - sum), 0, 255);
        }

        return weights;
    }

    /// <summary>2-bit 8×8 predominant-layer map (ACNK +0x12, LSB first, row-major) from per-layer weights.</summary>
    public static byte[] PredominantLayerMap(IReadOnlyList<byte[]> weights)
    {
        var map = new byte[16];
        for (int cellY = 0; cellY < 8; cellY++)
        {
            for (int cellX = 0; cellX < 8; cellX++)
            {
                int best = 0;
                long bestSum = -1;
                for (int layer = 0; layer < Math.Min(4, weights.Count); layer++)
                {
                    long sum = 0;
                    for (int y = 0; y < 8; y++)
                        for (int x = 0; x < 8; x++)
                            sum += weights[layer][(cellY * 8 + y) * 64 + cellX * 8 + x];
                    if (sum > bestSum)
                    {
                        bestSum = sum;
                        best = layer;
                    }
                }

                int bit = (cellY * 8 + cellX) * 2;
                map[bit / 8] |= (byte)(best << (bit % 8));
            }
        }

        return map;
    }
}
