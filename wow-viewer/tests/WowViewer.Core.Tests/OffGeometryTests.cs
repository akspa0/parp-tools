using System;
using System.Numerics;
using WowViewer.Core.Geometry;
using Xunit;

namespace WowViewer.Core.Tests;

public class OffGeometryTests
{
    private const string CubeOffSample = @"
# OpenSCAD 2021.01 generated OFF file
OFF
8 6 12
-1.0 -1.0 -1.0
 1.0 -1.0 -1.0
 1.0  1.0 -1.0
-1.0  1.0 -1.0
-1.0 -1.0  1.0
 1.0 -1.0  1.0
 1.0  1.0  1.0
-1.0  1.0  1.0
4 0 1 2 3
4 4 5 6 7
4 0 4 7 3
4 1 5 6 2
4 3 2 6 7
4 0 1 5 4
";

    [Fact]
    public void Parse_ValidCubeOff_ExtractsVerticesAndTriangulatedFaces()
    {
        var geom = OffGeometry.Parse(CubeOffSample);

        Assert.NotNull(geom);
        Assert.Equal(8, geom.VertexCount);
        Assert.Equal(6, geom.FaceCount);

        // 6 quads triangulated into 2 triangles each = 12 triangles = 36 indices
        Assert.Equal(36, geom.Indices.Length);

        // 8 vertices * 6 floats (pos XYZ + normal XYZ) = 48 floats
        Assert.Equal(48, geom.VertexData.Length);

        // Check bounds
        Assert.Equal(new Vector3(-1.0f, -1.0f, -1.0f), geom.BoundsMin);
        Assert.Equal(new Vector3(1.0f, 1.0f, 1.0f), geom.BoundsMax);
    }

    [Fact]
    public void Parse_HeaderOnSingleLine_ParsesCorrectly()
    {
        const string SingleLineHeaderOff = @"OFF 3 1 3
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
3 0 1 2
";
        var geom = OffGeometry.Parse(SingleLineHeaderOff);

        Assert.Equal(3, geom.VertexCount);
        Assert.Equal(1, geom.FaceCount);
        Assert.Equal(3, geom.Indices.Length);
        Assert.Equal(0u, geom.Indices[0]);
        Assert.Equal(1u, geom.Indices[1]);
        Assert.Equal(2u, geom.Indices[2]);
    }

    [Fact]
    public void Parse_MissingOffSignature_ThrowsFormatException()
    {
        const string InvalidOff = @"
PLY
format ascii 1.0
";
        Assert.Throws<FormatException>(() => OffGeometry.Parse(InvalidOff));
    }
}
