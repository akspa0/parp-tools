using System;
using System.IO;
using Xunit;
using WoWToolbox.Core.Legacy.Liquid;
using Moq;
using DBCD;

namespace WoWToolbox.Core.Tests.Legacy.Liquid
{
    public class MCLQChunkTests
    {
        [Fact]
        public void SerializeDeserialize_RoundTrip_PreservesData()
        {
            // Arrange
            var original = new MCLQChunk
            {
                HeightLevel1 = 100.5f,
                HeightLevel2 = 200.5f,
                Flags = MCLQFlags.HasLiquid | MCLQFlags.HasAlpha,
                Data = 0x42,
                X = 1.0f,
                Y = 2.0f,
                XOffset = 3,
                YOffset = 4,
                Width = 5,
                Height = 6,
                LiquidEntry = 7,
                LiquidVertexFormat = 8,
                LiquidFlags = 9,
                LiquidType = LiquidType.Ocean
            };

            // Fill height map with test data
            for (int y = 0; y < MCLQChunk.HEIGHT_MAP_SIZE; y++)
            {
                for (int x = 0; x < MCLQChunk.HEIGHT_MAP_SIZE; x++)
                {
                    original.HeightMap[x, y] = x * 10.0f + y;
                }
            }

            // Fill alpha map with test data
            for (int y = 0; y < MCLQChunk.ALPHA_MAP_SIZE; y++)
            {
                for (int x = 0; x < MCLQChunk.ALPHA_MAP_SIZE; x++)
                {
                    original.AlphaMap[x, y] = (byte)(x + y);
                }
            }

            // Act
            byte[] serialized;
            using (var ms = new MemoryStream())
            {
                using (var bw = new BinaryWriter(ms))
                {
                    original.Serialize(bw);
                }
                serialized = ms.ToArray();
            }

            MCLQChunk deserialized;
            using (var ms = new MemoryStream(serialized))
            {
                using (var br = new BinaryReader(ms))
                {
                    deserialized = MCLQReader.ReadChunk(br);
                }
            }

            // Assert
            Assert.Equal(original.HeightLevel1, deserialized.HeightLevel1);
            Assert.Equal(original.HeightLevel2, deserialized.HeightLevel2);
            Assert.Equal(original.Flags, deserialized.Flags);
            Assert.Equal(original.Data, deserialized.Data);
            Assert.Equal(original.X, deserialized.X);
            Assert.Equal(original.Y, deserialized.Y);
            Assert.Equal(original.XOffset, deserialized.XOffset);
            Assert.Equal(original.YOffset, deserialized.YOffset);
            Assert.Equal(original.Width, deserialized.Width);
            Assert.Equal(original.Height, deserialized.Height);
            Assert.Equal(original.LiquidEntry, deserialized.LiquidEntry);
            Assert.Equal(original.LiquidVertexFormat, deserialized.LiquidVertexFormat);
            Assert.Equal(original.LiquidFlags, deserialized.LiquidFlags);
            Assert.Equal(original.LiquidType, deserialized.LiquidType);

            // Check height map
            for (int y = 0; y < MCLQChunk.HEIGHT_MAP_SIZE; y++)
            {
                for (int x = 0; x < MCLQChunk.HEIGHT_MAP_SIZE; x++)
                {
                    Assert.Equal(original.HeightMap[x, y], deserialized.HeightMap[x, y]);
                }
            }

            // Check alpha map
            for (int y = 0; y < MCLQChunk.ALPHA_MAP_SIZE; y++)
            {
                for (int x = 0; x < MCLQChunk.ALPHA_MAP_SIZE; x++)
                {
                    Assert.Equal(original.AlphaMap[x, y], deserialized.AlphaMap[x, y]);
                }
            }
        }

        [Fact]
        public void HasAlphaMap_WhenFlagSet_ReturnsTrue()
        {
            // Arrange
            var chunk = new MCLQChunk { Flags = MCLQFlags.HasAlpha };

            // Act & Assert
            Assert.True(chunk.HasAlphaMap);
        }

        [Fact]
        public void HasAlphaMap_WhenFlagNotSet_ReturnsFalse()
        {
            // Arrange
            var chunk = new MCLQChunk { Flags = MCLQFlags.None };

            // Act & Assert
            Assert.False(chunk.HasAlphaMap);
        }

        [Fact]
        public void IsValidLiquidEntry_WhenValidationDisabled_ReturnsTrue()
        {
            // Arrange
            MCLQChunk.DisableValidation();
            var chunk = new MCLQChunk { LiquidEntry = 999999 }; // Invalid entry

            // Act & Assert
            Assert.True(chunk.IsValidLiquidEntry);
        }

        [Fact]
        public void IsValidLiquidEntry_WhenValidatorConfigured_UsesValidator()
        {
            // Arrange
            // Create a mock IDBCDStorage using Moq
            var mockStorage = new Mock<IDBCDStorage>();
            ushort validEntry = 1;
            ushort invalidEntry = 999;

            // Setup the mock: Return true for ContainsKey(1), false otherwise
            mockStorage.Setup(s => s.ContainsKey(validEntry)).Returns(true);
            mockStorage.Setup(s => s.ContainsKey(It.Is<int>(k => k != validEntry))).Returns(false);

            // Configure MCLQChunk with the mock storage
            MCLQChunk.ConfigureValidator(mockStorage.Object);
            
            var chunkValid = new MCLQChunk { LiquidEntry = validEntry };
            var chunkInvalid = new MCLQChunk { LiquidEntry = invalidEntry };

            // Act & Assert
            Assert.True(chunkValid.IsValidLiquidEntry); 
            Assert.False(chunkInvalid.IsValidLiquidEntry);
            
            // Verify ContainsKey was called on the mock
            mockStorage.Verify(s => s.ContainsKey(validEntry), Times.Once);
            mockStorage.Verify(s => s.ContainsKey(invalidEntry), Times.Once);
        }
    }
} 