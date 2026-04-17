using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.Tests;

public sealed class MdxRenderCharacteristicsAnalyzerTests
{
    [Fact]
    public void Analyze_OpaqueMaterialOnly_ReportsOpaqueOnly()
    {
        MdxSummary summary = CreateSummary(
            geosets: [CreateGeoset(materialId: 0)],
            materials: [new MdxMaterialSummary(0, 0, [new MdxMaterialLayerSummary(0, 0u, 0u, 0, 0, 0, 1.0f)])]);

        MdxRenderCharacteristics characteristics = MdxRenderCharacteristicsAnalyzer.Analyze(summary);

        Assert.True(characteristics.HasOpaqueRenderContent);
        Assert.False(characteristics.HasTransparentRenderContent);
    }

    [Fact]
    public void Analyze_BlendedMaterial_ReportsTransparentOnlyWhenNoOpaqueLayerExists()
    {
        MdxSummary summary = CreateSummary(
            geosets: [CreateGeoset(materialId: 0)],
            materials: [new MdxMaterialSummary(0, 0, [new MdxMaterialLayerSummary(0, 1u, 0u, 0, 0, 0, 1.0f)])]);

        MdxRenderCharacteristics characteristics = MdxRenderCharacteristicsAnalyzer.Analyze(summary);

        Assert.False(characteristics.HasOpaqueRenderContent);
        Assert.True(characteristics.HasTransparentRenderContent);
    }

    [Fact]
    public void Analyze_GeosetsWithoutMaterials_ConservativelyReportsOpaqueContent()
    {
        MdxSummary summary = CreateSummary(
            geosets: [CreateGeoset(materialId: -1)],
            materials: []);

        MdxRenderCharacteristics characteristics = MdxRenderCharacteristicsAnalyzer.Analyze(summary);

        Assert.True(characteristics.HasOpaqueRenderContent);
        Assert.False(characteristics.HasTransparentRenderContent);
    }

    [Fact]
    public void Analyze_BlendedMaterialWithoutGeosets_ReportsTransparentOnly()
    {
        MdxSummary summary = CreateSummary(
            geosets: [],
            materials: [new MdxMaterialSummary(0, 0, [new MdxMaterialLayerSummary(0, 2u, 0u, 0, 0, 0, 0.75f)])]);

        MdxRenderCharacteristics characteristics = MdxRenderCharacteristicsAnalyzer.Analyze(summary);

        Assert.False(characteristics.HasOpaqueRenderContent);
        Assert.True(characteristics.HasTransparentRenderContent);
    }

    private static MdxSummary CreateSummary(IReadOnlyList<MdxGeosetSummary> geosets, IReadOnlyList<MdxMaterialSummary> materials)
    {
        return new MdxSummary(
            sourcePath: "synthetic.mdx",
            signature: "MDLX",
            version: 1300,
            modelName: "Synthetic",
            blendTime: 0,
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            globalSequences: [],
            sequences: [],
            geosets: geosets,
            geosetAnimations: [],
            bones: [],
            lights: [],
            helpers: [],
            attachments: [],
            particleEmitters2: [],
            ribbons: [],
            cameras: [],
            events: [],
            hitTestShapes: [],
            collision: null,
            pivotPoints: [],
            textures: [],
            materials: materials,
            chunks: [],
            knownChunkCount: 0,
            unknownChunkCount: 0);
    }

    private static MdxGeosetSummary CreateGeoset(int materialId)
    {
        return new MdxGeosetSummary(0, 3, 3, 1, 1, 1, 1, 3, 0, 0, 0, 0, 0, materialId, 0u, 0u, 1f, Vector3.Zero, Vector3.One, 0);
    }
}