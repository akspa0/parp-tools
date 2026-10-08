using System;
using System.Numerics;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

public sealed class MinimapLightingOptimizerTests
{
    private static TerrainTileTensorPack CreateSyntheticTilePack(int size = 16)
    {
        float[,,] normals = new float[size, size, 3];

        // Sloped surface tilted towards +X/+Y
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                Vector3 n = Vector3.Normalize(new Vector3(0.5f, 0.5f, 0.707f));
                normals[x, y, 0] = n.X;
                normals[x, y, 1] = n.Y;
                normals[x, y, 2] = n.Z;
            }
        }

        return new TerrainTileTensorPack
        {
            TileX = 32,
            TileY = 32,
            McnrNormalXyz = normals
        };
    }

    [Fact]
    public void EvaluateWithSpecular_ProducesPositiveSpecularLobe()
    {
        Vector3 normal = Vector3.UnitZ;
        Vector3 lightDir = new(0f, 0f, 1f); // Sun directly overhead
        Vector3 dirColor = Vector3.One;
        Vector3 ambColor = new(0.2f);

        Vector3 withoutSpec = TerrainLightingMath.EvaluateWithSpecular(
            interpolatedLambert: 1f,
            normal: normal,
            lightDirection: lightDir,
            specularIntensity: 0f,
            specularPower: 16f,
            directionalColor: dirColor,
            ambientColor: ambColor,
            shadowMask: 0f
        );

        Vector3 withSpec = TerrainLightingMath.EvaluateWithSpecular(
            interpolatedLambert: 1f,
            normal: normal,
            lightDirection: lightDir,
            specularIntensity: 0.25f,
            specularPower: 16f,
            directionalColor: dirColor,
            ambientColor: ambColor,
            shadowMask: 0f
        );

        Assert.True(withSpec.X > withoutSpec.X, "Specular component must brighten lit surface");
        Assert.True(withSpec.Y > withoutSpec.Y);
        Assert.True(withSpec.Z > withoutSpec.Z);
    }

    [Fact]
    public void ComputePhotometricMae_IdenticalParameters_HasLowError()
    {
        int res = 16;
        var pack = CreateSyntheticTilePack(res);
        var targetParams = new MinimapLightingParameters(
            SolarAzimuth: MathF.PI * 0.25f,
            SolarElevation: MathF.PI / 4f,
            AmbientIntensity: 0.25f,
            DiffuseIntensity: 0.75f,
            CastShadowStrength: 0.30f,
            SpecularIntensity: 0.05f,
            SpecularPower: 8.0f,
            ApplyDxt1Quantization: false
        );

        // Generate synthetic reference image under targetParams
        byte[,,] refImg = new byte[res, res, 3];
        float cosEl = MathF.Cos(targetParams.SolarElevation);
        Vector3 lightDir = Vector3.Normalize(new Vector3(
            cosEl * MathF.Cos(targetParams.SolarAzimuth),
            cosEl * MathF.Sin(targetParams.SolarAzimuth),
            MathF.Sin(targetParams.SolarElevation)));

        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                Vector3 normal = new(pack.McnrNormalXyz![x, y, 0], pack.McnrNormalXyz[x, y, 1], pack.McnrNormalXyz[x, y, 2]);
                float lambert = Math.Clamp(Vector3.Dot(normal, lightDir), 0f, 1f);
                Vector3 lit = TerrainLightingMath.EvaluateWithSpecular(
                    lambert,
                    normal,
                    lightDir,
                    targetParams.SpecularIntensity,
                    targetParams.SpecularPower,
                    new Vector3(targetParams.DiffuseIntensity),
                    new Vector3(targetParams.AmbientIntensity),
                    0f,
                    0.30f,
                    false);

                byte luma = (byte)Math.Clamp((int)(lit.X * 255f), 0, 255);
                refImg[x, y, 0] = luma;
                refImg[x, y, 1] = luma;
                refImg[x, y, 2] = luma;
            }
        }

        float errorMatch = MinimapLightingOptimizer.ComputePhotometricMae(pack, refImg, targetParams);
        Assert.InRange(errorMatch, 0f, 0.01f);

        // Perturb azimuth and verify error increases significantly
        var wrongParams = targetParams with { SolarAzimuth = targetParams.SolarAzimuth + MathF.PI * 0.5f };
        float errorWrong = MinimapLightingOptimizer.ComputePhotometricMae(pack, refImg, wrongParams);
        Assert.True(errorWrong > errorMatch * 5f, $"Wrong lighting parameters ({errorWrong}) must have substantially higher error than matching parameters ({errorMatch})");
    }

    [Fact]
    public void Optimize_ConvergesTowardTrueParameters()
    {
        int res = 16;
        var pack = CreateSyntheticTilePack(res);
        var groundTruth = new MinimapLightingParameters(
            SolarAzimuth: MathF.PI * 0.25f, // 45 degrees
            SolarElevation: MathF.PI / 4.5f,
            AmbientIntensity: 0.25f,
            DiffuseIntensity: 0.75f,
            CastShadowStrength: 0.30f,
            SpecularIntensity: 0.05f,
            SpecularPower: 8.0f,
            ApplyDxt1Quantization: false
        );

        byte[,,] refImg = new byte[res, res, 3];
        float cosEl = MathF.Cos(groundTruth.SolarElevation);
        Vector3 lightDir = Vector3.Normalize(new Vector3(
            cosEl * MathF.Cos(groundTruth.SolarAzimuth),
            cosEl * MathF.Sin(groundTruth.SolarAzimuth),
            MathF.Sin(groundTruth.SolarElevation)));

        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
            {
                Vector3 normal = new(pack.McnrNormalXyz![x, y, 0], pack.McnrNormalXyz[x, y, 1], pack.McnrNormalXyz[x, y, 2]);
                float lambert = Math.Clamp(Vector3.Dot(normal, lightDir), 0f, 1f);
                Vector3 lit = TerrainLightingMath.EvaluateWithSpecular(
                    lambert,
                    normal,
                    lightDir,
                    groundTruth.SpecularIntensity,
                    groundTruth.SpecularPower,
                    new Vector3(groundTruth.DiffuseIntensity),
                    new Vector3(groundTruth.AmbientIntensity),
                    0f,
                    0.30f,
                    false);

                byte luma = (byte)Math.Clamp((int)(lit.X * 255f), 0, 255);
                refImg[x, y, 0] = luma;
                refImg[x, y, 1] = luma;
                refImg[x, y, 2] = luma;
            }
        }

        var result = MinimapLightingOptimizer.Optimize(pack, refImg);

        Assert.True(result.Converged);
        Assert.InRange(result.PhotometricMae, 0f, 0.08f);
        Assert.Equal(groundTruth.SolarAzimuth, result.BestParameters.SolarAzimuth, 2);
    }
}
