using System.Numerics;
using WoWToolbox.Core.v2.Foundation.PM4;
using WoWToolbox.Core.v2.Foundation.PM4.Chunks;
using Warcraft.NET.Files.Structures;
using WoWToolbox.Core.v2.Foundation.Transforms;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.PM4
{
    public class CoordinateTransformsTests
    {
        [Fact]
        public void FromMsvtVertex_Flips_Zup_To_Yup()
        {
            var raw = new MSVT_Vertex { Position = new Vector3(10, 20, 30) };
            var result = CoordinateTransforms.FromMsvtVertex(raw);
            Assert.Equal(new Vector3(20, 10, 30), result);
        }

        [Fact]
        public void FromMsvtVertexSimple_LegacyStruct()
        {
            var raw = new MsvtVertex { X = 1, Y = 2, Z = 3 };
            var result = CoordinateTransforms.FromMsvtVertexSimple(raw);
            Assert.Equal(new Vector3(2, 1, 3), result);
        }

        [Fact]
        public void FromMscnVertex_CorrectFlip()
        {
            var raw = new Vector3(-5, 7, 9);
            var result = CoordinateTransforms.FromMscnVertex(raw);
            Assert.Equal(new Vector3(7, 5, 9), result);
        }

        [Fact]
        public void FromMspvVertex_Uses_C3Vector()
        {
            var raw = new C3Vector { X = 4, Y = 5, Z = 6 };
            var expected = new Vector3(4, 5, 6);
            var result = CoordinateTransforms.FromMspvVertex(raw);
            Assert.Equal(expected, result);
        }

        [Fact]
        public void FromMprlEntry_RotatesAxes()
        {
            var entry = new MprlEntry
            {
                Position = new C3Vector { X = 1, Y = 2, Z = 3 }
            };
            var expected = new Vector3(-3, 2, 1);
            var result = CoordinateTransforms.FromMprlEntry(entry);
            Assert.Equal(expected, result);
        }
    }
}
