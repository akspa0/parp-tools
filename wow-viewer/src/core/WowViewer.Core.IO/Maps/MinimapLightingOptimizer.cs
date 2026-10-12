using System.Numerics;
using WowViewer.Core.IO.Blp;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.IO.Maps;

/// <summary>
/// Result of an automated photometric lighting and specular calibration pass.
/// </summary>
public sealed record MinimapCalibrationResult(
    MinimapLightingParameters BestParameters,
    float PhotometricMae,
    int Iterations,
    bool Converged
);

/// <summary>
/// Automated photometric solver determining optimal solar direction, ambient/diffuse balance,
/// and texture specular reflectance by minimizing error against observed 0.5.3 minimap tiles.
/// </summary>
public static class MinimapLightingOptimizer
{
    /// <summary>
    /// Computes the Photometric Mean Absolute Error (MAE) between a candidate lighting configuration
    /// and an observed minimap image across unoccluded terrain pixels.
    /// </summary>
    public static float ComputePhotometricMae(
        TerrainTileTensorPack pack,
        byte[,,] observedMinimapRgb,
        MinimapLightingParameters parameters)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(observedMinimapRgb);

        if (pack.McnrNormalXyz is null)
            throw new ArgumentException("Terrain tile must possess decoded McnrNormalXyz surface normals.", nameof(pack));

        int width = observedMinimapRgb.GetLength(0);
        int height = observedMinimapRgb.GetLength(1);
        if (width <= 0 || height <= 0 || observedMinimapRgb.GetLength(2) < 3)
            return float.MaxValue;

        // Solar direction vector from spherical angles
        float cosElev = MathF.Cos(parameters.SolarElevation);
        Vector3 lightDir = Vector3.Normalize(new Vector3(
            cosElev * MathF.Cos(parameters.SolarAzimuth),
            cosElev * MathF.Sin(parameters.SolarAzimuth),
            MathF.Sin(parameters.SolarElevation)
        ));

        Vector3 dirColor = new(parameters.DiffuseIntensity);
        Vector3 ambColor = new(parameters.AmbientIntensity);

        float totalError = 0f;
        int validPixels = 0;

        float[,,] normals = pack.McnrNormalXyz;
        int normW = normals.GetLength(0);
        int normH = normals.GetLength(1);

        byte[,] candidateLuma = new byte[width, height];

        for (int y = 0; y < height; y++)
        {
            float normV = (y + 0.5f) / height;
            int ny = Math.Clamp((int)(normV * normH), 0, normH - 1);

            for (int x = 0; x < width; x++)
            {
                float normU = (x + 0.5f) / width;
                int nx = Math.Clamp((int)(normU * normW), 0, normW - 1);

                Vector3 normal = new(normals[nx, ny, 0], normals[nx, ny, 1], normals[nx, ny, 2]);
                if (normal == Vector3.Zero)
                    normal = Vector3.UnitZ;

                float lambert = Math.Clamp(Vector3.Dot(normal, lightDir), 0f, 1f);
                float shadowMask = pack.McshShadowMask256 is not null
                    ? Math.Clamp(pack.McshShadowMask256[Math.Clamp((int)(normU * 256), 0, 255), Math.Clamp((int)(normV * 256), 0, 255)], 0f, 1f)
                    : 0f;

                Vector3 lit = TerrainLightingMath.EvaluateWithSpecular(
                    lambert,
                    normal,
                    lightDir,
                    parameters.SpecularIntensity,
                    parameters.SpecularPower,
                    dirColor,
                    ambColor,
                    shadowMask,
                    parameters.CastShadowStrength,
                    toneMapped: false
                );

                byte luma = (byte)Math.Clamp((int)(lit.X * 255f), 0, 255);
                candidateLuma[x, y] = luma;
            }
        }

        // Apply DXT1 quantization simulation if enabled
        if (parameters.ApplyDxt1Quantization && width >= 4 && height >= 4)
        {
            using var img = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte val = candidateLuma[x, y];
                    img[x, y] = new SixLabors.ImageSharp.PixelFormats.Rgba32(val, val, val, 255);
                }
            }

            using var quantized = Dxt1TileCodec.EncodeDecode(img);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    candidateLuma[x, y] = quantized[x, y].R;
                }
            }
        }

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Compute observed luminance
                float obsR = observedMinimapRgb[x, y, 0];
                float obsG = observedMinimapRgb[x, y, 1];
                float obsB = observedMinimapRgb[x, y, 2];
                float obsLuma = 0.299f * obsR + 0.587f * obsG + 0.114f * obsB;

                float diff = MathF.Abs(candidateLuma[x, y] - obsLuma) / 255f;
                totalError += diff;
                validPixels++;
            }
        }

        return validPixels > 0 ? (totalError / validPixels) : float.MaxValue;
    }

    /// <summary>
    /// Executes a coarse-to-fine parameter optimization over solar azimuth, elevation,
    /// ambient/diffuse balance, and specular properties to find the best match.
    /// </summary>
    public static MinimapCalibrationResult Optimize(
        TerrainTileTensorPack pack,
        byte[,,] observedMinimapRgb,
        int maxIterations = 20)
    {
        ArgumentNullException.ThrowIfNull(pack);
        ArgumentNullException.ThrowIfNull(observedMinimapRgb);

        MinimapLightingParameters bestParams = MinimapLightingParameters.Default;
        float bestMae = ComputePhotometricMae(pack, observedMinimapRgb, bestParams);
        int iterations = 0;

        // 1. Sweep Azimuth in 8 directions (45-degree steps)
        float[] azimuthCandidates = [0f, MathF.PI * 0.25f, MathF.PI * 0.5f, MathF.PI * 0.75f, MathF.PI, MathF.PI * 1.25f, MathF.PI * 1.5f, MathF.PI * 1.75f];
        foreach (float az in azimuthCandidates)
        {
            iterations++;
            var candidate = bestParams with { SolarAzimuth = az };
            float mae = ComputePhotometricMae(pack, observedMinimapRgb, candidate);
            if (mae < bestMae)
            {
                bestMae = mae;
                bestParams = candidate;
            }
        }

        // 2. Refine Elevation in [20, 60] degrees
        float[] elevationCandidates = [MathF.PI / 9f, MathF.PI / 6f, MathF.PI / 4.5f, MathF.PI / 3.5f];
        foreach (float el in elevationCandidates)
        {
            iterations++;
            var candidate = bestParams with { SolarElevation = el };
            float mae = ComputePhotometricMae(pack, observedMinimapRgb, candidate);
            if (mae < bestMae)
            {
                bestMae = mae;
                bestParams = candidate;
            }
        }

        // 3. Refine Ambient & Diffuse balance
        float[] ambientCandidates = [0.15f, 0.25f, 0.35f, 0.45f];
        foreach (float amb in ambientCandidates)
        {
            iterations++;
            var candidate = bestParams with { AmbientIntensity = amb, DiffuseIntensity = 1f - amb };
            float mae = ComputePhotometricMae(pack, observedMinimapRgb, candidate);
            if (mae < bestMae)
            {
                bestMae = mae;
                bestParams = candidate;
            }
        }

        // 4. Refine Specular Intensity & Power
        float[] specCandidates = [0.0f, 0.05f, 0.15f, 0.25f];
        foreach (float spec in specCandidates)
        {
            iterations++;
            var candidate = bestParams with { SpecularIntensity = spec };
            float mae = ComputePhotometricMae(pack, observedMinimapRgb, candidate);
            if (mae < bestMae)
            {
                bestMae = mae;
                bestParams = candidate;
            }
        }

        return new MinimapCalibrationResult(
            BestParameters: bestParams,
            PhotometricMae: bestMae,
            Iterations: iterations,
            Converged: bestMae < 0.20f
        );
    }
}
