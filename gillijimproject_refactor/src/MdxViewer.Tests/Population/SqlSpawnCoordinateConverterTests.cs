using System.Numerics;
using MdxViewer.Population;
using Xunit;

namespace MdxViewer.Tests.Population;

public sealed class SqlSpawnCoordinateConverterTests
{
    [Fact]
    public void ToRendererPosition_PreservesSqlSpawnCoordinates()
    {
        var wowPosition = new Vector3(1200f, 3400f, 55f);
        var position = SqlSpawnCoordinateConverter.ToRendererPosition(wowPosition);

        Assert.Equal(wowPosition, position);
    }
}