using System;
using System.IO;
using Xunit;
using DBCD.Providers;
using WoWToolbox.Core.Legacy.Liquid;

namespace WoWToolbox.Core.Tests.Legacy.Liquid
{
    public class LiquidTypeValidatorTests
    {
        [Fact]
        public void ValidateLiquidEntry_WhenValidatorNotConfigured_ReturnsTrue()
        {
            // Arrange
            var validator = new LiquidTypeValidator();

            // Act
            bool isValid = validator.ValidateLiquidEntry(1);

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void ValidateLiquidEntry_WhenDBCLoadFails_ReturnsTrue()
        {
            // Arrange
            var validator = new LiquidTypeValidator(new MockDBCProvider(), "nonexistent.dbc");

            // Act
            bool isValid = validator.ValidateLiquidEntry(1);

            // Assert
            Assert.True(isValid);
        }

        [Fact]
        public void ClearCache_ClearsValidationCache()
        {
            // Arrange
            var validator = new LiquidTypeValidator();
            validator.ValidateLiquidEntry(1); // Add to cache

            // Act
            validator.ClearCache();

            // Act & Assert - Implementation detail: will cause cache to be rebuilt
            Assert.True(validator.ValidateLiquidEntry(1));
        }
    }

    // Mock DBC provider for testing
    public class MockDBCProvider : IDBCProvider
    {
        public Stream StreamForTableName(string tableName, string build = null)
        {
            throw new FileNotFoundException();
        }

        public bool TableExists(string tableName, string build = null)
        {
            return false;
        }
    }
} 