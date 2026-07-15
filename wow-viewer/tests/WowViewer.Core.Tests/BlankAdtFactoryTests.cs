using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class BlankAdtFactoryTests
{
    [Fact]
    public void CreateBlank_EncodesFlatMcnrAsSignedXzyUnitUp()
    {
        LkAdtData adt = BlankAdtFactory.CreateBlank("Synthetic", 0, 0);

        LkMcnkData chunk = Assert.IsType<LkMcnkData>(adt.Chunks[0]);
        Assert.Equal(448, chunk.Normals.Length);
        for (int sample = 0; sample < 145; sample++)
        {
            int offset = sample * 3;
            Assert.Equal((byte)0, chunk.Normals[offset]);
            Assert.Equal((byte)127, chunk.Normals[offset + 1]);
            Assert.Equal((byte)0, chunk.Normals[offset + 2]);
        }
    }
}
