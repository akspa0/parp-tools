using System.Numerics;
using WowViewer.Core.IO.Dbc;
using WowViewer.Core.Runtime.DetailDoodads;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests.DetailDoodads;

public sealed class GroundEffectPlacementTests
{
    [Fact]
    public void ComputeNormalAlignment_AlignsUnitZToTargetNormal()
    {
        // 45 degree slope: normal = (0.7071, 0, 0.7071)
        var targetNorm = Vector3.Normalize(new Vector3(1.0f, 0.0f, 1.0f));
        var q = GroundEffectPlacementGenerator.ComputeNormalAlignment(targetNorm);

        // Rotating (0, 0, 1) by q must produce targetNorm
        var transformed = Vector3.Transform(Vector3.UnitZ, q);
        Assert.True(MathF.Abs(transformed.X - targetNorm.X) < 1e-4f);
        Assert.True(MathF.Abs(transformed.Y - targetNorm.Y) < 1e-4f);
        Assert.True(MathF.Abs(transformed.Z - targetNorm.Z) < 1e-4f);
    }

    [Fact]
    public void GenerateChunkDoodads_EnforcesSlopeCulling()
    {
        // Chunk with steep normals (normal Z = 0.2 < 0.4)
        var steepNormals = new Vector3[145];
        for (int i = 0; i < 145; i++)
            steepNormals[i] = Vector3.Normalize(new Vector3(0.9f, 0.0f, 0.2f));

        var heights = new float[145];
        var input = new TerrainChunkPlacementInput
        {
            WorldPosition = Vector3.Zero,
            Heights = heights,
            Normals = steepNormals,
            Layers = new List<TerrainChunkLayerInput>
            {
                new() { EffectId = 1, TextureIndex = 0 }
            },
            ChunkX = 0,
            ChunkY = 0,
            TileX = 30,
            TileY = 30,
        };

        var texRec = new GroundEffectTextureRecord(1, new uint[] { 10 }, Density: 32);
        var doodadRec = new GroundEffectDoodadRecord(10, "plant.m2", 0, GroundEffectDoodadFlags.None, 1.0f, 1.0f);

        var instances = GroundEffectPlacementGenerator.GenerateChunkDoodads(
            input,
            _ => texRec,
            _ => doodadRec,
            densityMultiplier: 1.0f);

        // All candidates should have been culled by the slope limit
        Assert.Empty(instances);

        // Now test with gentle slope normals (normal Z = 0.95 >= 0.4)
        var gentleNormals = new Vector3[145];
        for (int i = 0; i < 145; i++)
            gentleNormals[i] = Vector3.Normalize(new Vector3(0.1f, 0.0f, 0.95f));

        var gentleInput = new TerrainChunkPlacementInput
        {
            WorldPosition = Vector3.Zero,
            Heights = heights,
            Normals = gentleNormals,
            Layers = new List<TerrainChunkLayerInput>
            {
                new() { EffectId = 1, TextureIndex = 0 }
            },
            ChunkX = 0,
            ChunkY = 0,
            TileX = 30,
            TileY = 30,
        };

        var gentleInstances = GroundEffectPlacementGenerator.GenerateChunkDoodads(
            gentleInput,
            _ => texRec,
            _ => doodadRec,
            densityMultiplier: 1.0f);

        // Candidates should now be placed
        Assert.NotEmpty(gentleInstances);
    }

    [Fact]
    public void GenerateChunkDoodads_RespectsFlagsAndMccvAndShadow()
    {
        var heights = new float[145];
        var normals = new Vector3[145];
        for (int i = 0; i < 145; i++)
            normals[i] = Vector3.UnitZ;

        // MCCV: set vertex colors to bright green (B=0, G=200, R=0, A=255)
        var mccv = new byte[145 * 4];
        for (int i = 0; i < 145; i++)
        {
            mccv[i * 4 + 0] = 0;   // B
            mccv[i * 4 + 1] = 200; // G
            mccv[i * 4 + 2] = 0;   // R
            mccv[i * 4 + 3] = 255; // A
        }

        // Shadow map: all shadowed (255)
        var shadowMap = new byte[64 * 64];
        Array.Fill(shadowMap, (byte)255);

        // Case 1: Flag 0x0 (interpolates MCCV, darkened by shadow)
        var inputNormal = new TerrainChunkPlacementInput
        {
            WorldPosition = Vector3.Zero,
            Heights = heights,
            Normals = normals,
            MccvColors = mccv,
            ShadowMap = shadowMap,
            Layers = new List<TerrainChunkLayerInput>
            {
                new() { EffectId = 1, TextureIndex = 0 }
            },
            ChunkX = 0,
            ChunkY = 0,
            TileX = 30,
            TileY = 30,
        };

        var texRec = new GroundEffectTextureRecord(1, new uint[] { 10 }, Density: 16);
        var doodadRecNormal = new GroundEffectDoodadRecord(10, "plant.m2", 0, GroundEffectDoodadFlags.None, 1.0f, 1.0f);

        var instancesNormal = GroundEffectPlacementGenerator.GenerateChunkDoodads(
            inputNormal,
            _ => texRec,
            _ => doodadRecNormal,
            densityMultiplier: 1.0f);

        Assert.NotEmpty(instancesNormal);
        var first = instancesNormal[0];
        // Green should be around 200 * 0.7 = 140
        byte g = (byte)((first.ColorBgra >> 8) & 0xFF);
        Assert.InRange(g, 130, 150);

        // Case 2: Flag 0x2 (IgnoreMCCV -> white, darkened by shadow = 255 * 0.7 = ~178)
        var doodadRecIgnoreMccv = new GroundEffectDoodadRecord(10, "plant.m2", 0, GroundEffectDoodadFlags.IgnoreMCCV, 1.0f, 1.0f);
        var instancesWhite = GroundEffectPlacementGenerator.GenerateChunkDoodads(
            inputNormal,
            _ => texRec,
            _ => doodadRecIgnoreMccv,
            densityMultiplier: 1.0f);

        Assert.NotEmpty(instancesWhite);
        var firstWhite = instancesWhite[0];
        byte rw = (byte)((firstWhite.ColorBgra >> 16) & 0xFF);
        byte gw = (byte)((firstWhite.ColorBgra >> 8) & 0xFF);
        byte bw = (byte)(firstWhite.ColorBgra & 0xFF);
        Assert.InRange(rw, 170, 185);
        Assert.InRange(gw, 170, 185);
        Assert.InRange(bw, 170, 185);
    }

    [Fact]
    public void WmoDetailDoodadDecoder_DecodesGroupPlacementsCorrectly()
    {
        // Simple triangle: 3 vertices
        var vertices = new Vector3[]
        {
            new(0f, 0f, 10f),
            new(10f, 0f, 10f),
            new(0f, 10f, 10f),
        };
        var normals = new Vector3[] { Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ };
        var indices = new ushort[] { 0, 1, 2 };

        var layers = new List<WmoDetailDoodadLayer>
        {
            new(Density: 16, Doodads: new List<WmoDetailDoodadEntry> { new(DoodadId: 77, Weight: 100) })
        };
        var commands = new List<WmoDetailDoodadDecodedCommand>
        {
            new(LayerIndex: 0, BatchIndex: 0, RollAllLocations: false, LocRangeIndex: 0, SingleLocation: true, Locations: new List<int> { 0 })
        };

        var input = new WmoGroupPlacementInput
        {
            GroupIndex = 0,
            Vertices = vertices,
            Normals = normals,
            Indices = indices,
            WorldTransform = Matrix4x4.Identity,
            Layers = layers,
            Commands = commands,
        };

        var doodad = new GroundEffectDoodadRecord(77, "doodad.m2", 0, GroundEffectDoodadFlags.AlignToNormal, 1.0f, 1.0f);

        var instances = WmoDetailDoodadDecoder.DecodeGroupDoodads(input, _ => doodad);
        Assert.Single(instances);
        Assert.Equal(77u, instances[0].DoodadId);
        Assert.Equal("doodad.m2", instances[0].ModelPath);
        Assert.True(instances[0].AlignToNormal);
    }
}
