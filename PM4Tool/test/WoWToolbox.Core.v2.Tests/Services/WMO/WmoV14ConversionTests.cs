using System.IO;
using System.Threading.Tasks;
using WoWToolbox.Core.v2.Services.WMO;
using WoWToolbox.Core.v2.Services.WMO.Legacy;
using Xunit;

namespace WoWToolbox.Core.v2.Tests.Services.WMO
{
    public class WmoV14ConversionTests
    {
        private const string TestWmoPath = @"..\..\..\..\test_data\335_wmo\World\wmo\transports\WMO_elevators\UL_Ulduar_Trapdoor_02_transport.wmo";
        private const string OutputDirectory = @"..\..\..\..\project_output\WmoV14ConversionTests";

        [Fact]
        public async Task ConvertV14ToV17_ShouldSucceed_And_Create_Output_Files()
        {
            // Arrange
            if (!File.Exists(TestWmoPath))
                return; // skip if sample asset not present
            if (Directory.Exists(OutputDirectory))
            {
                Directory.Delete(OutputDirectory, true);
            }
            Directory.CreateDirectory(OutputDirectory);

            var converter = new FullV14Converter();

            // Act
            var result = await converter.ConvertAsync(TestWmoPath, OutputDirectory);

            // Assert
            Assert.True(result.Success, result.ErrorMessage);
            Assert.True(File.Exists(result.ConvertedWmoPath));
            Assert.True(File.Exists(result.ObjFilePath));
            Assert.True(File.Exists(result.LogFilePath));

            var logContent = await File.ReadAllTextAsync(result.LogFilePath);
            Assert.Contains("MOVI chunk not found", logContent);
        }
    }
}
