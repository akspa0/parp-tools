using System;
using System.Linq;
using WoWToolbox.Core.v2.Foundation.WMO.V14.Parsers;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.Foundation.WMO.V14
{
    public class MoviParserTests
    {
        [Fact]
        public void Parse_ShouldReturnSingleTriangle()
        {
            // Arrange – indices 1,2,3 little-endian
            byte[] payload = { 0x01, 0x00, 0x02, 0x00, 0x03, 0x00 };

            // Act
            var tris = MOVIParser.Parse(payload);

            // Assert
            Assert.Single(tris);
            var t = tris.First();
            Assert.Equal((ushort)1, t.A);
            Assert.Equal((ushort)2, t.B);
            Assert.Equal((ushort)3, t.C);
        }
    }
}
