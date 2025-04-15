using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using WoWToolbox.Core.ADT; // For Placement type
using Xunit;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace WoWToolbox.Tests.AnalysisTool
{
    public class ProgramTests
    {
        private const string TestAdtPath = "test_data/original_development/development_22_18.adt";
        private const string TestPm4Path = "test_data/original_development/development_22_18.pm4";

        [Fact]
        public void RunCorrelation_ShouldProduceCorrectYaml()
        {
            Console.WriteLine("--- ProgramTests.RunCorrelation_ShouldProduceCorrectYaml START ---");

            // Arrange
            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            var adtInputPath = Path.Combine(baseDir, TestAdtPath);
            var pm4InputPath = Path.Combine(baseDir, TestPm4Path);
            var expectedYamlOutputPath = Path.Combine(Path.GetDirectoryName(adtInputPath) ?? ".", "correlated_placements.yaml");

            Console.WriteLine($"ADT Input: {adtInputPath}");
            Console.WriteLine($"PM4 Input: {pm4InputPath}");
            Console.WriteLine($"Expected YAML Output: {expectedYamlOutputPath}");

            Assert.True(File.Exists(adtInputPath), $"Test ADT file not found: {adtInputPath}");
            Assert.True(File.Exists(pm4InputPath), $"Test PM4 file not found: {pm4InputPath}");

            // Clean up previous output file if it exists
            if (File.Exists(expectedYamlOutputPath))
            {
                Console.WriteLine("Deleting existing output YAML file.");
                File.Delete(expectedYamlOutputPath);
            }

            // Act
            // Note: Program.Main might hang on Console.ReadLine().
            // If tests fail to complete, refactor Program.cs to make core logic testable without ReadLine.
            try
            {
                 Console.WriteLine("Running AnalysisTool Program.Main...");
                 // Redirect Console output to avoid blocking/capture if needed
                 // var currentOut = Console.Out;
                 // using (var writer = new StringWriter())
                 // {
                 //    Console.SetOut(writer);
                     WoWToolbox.AnalysisTool.Program.Main(new string[] { adtInputPath, pm4InputPath });
                 //    Console.SetOut(currentOut);
                 //    var consoleOutput = writer.ToString();
                 //    Console.WriteLine("Captured Console Output:\n" + consoleOutput); // Log captured output
                 // }
                 Console.WriteLine("AnalysisTool Program.Main finished.");
            }
            catch (Exception ex)
            {
                Assert.Fail($"AnalysisTool Program.Main threw an exception: {ex.Message}");
            }

            // Assert
            Assert.True(File.Exists(expectedYamlOutputPath), $"Expected YAML output file was not created: {expectedYamlOutputPath}");

            var yamlContent = File.ReadAllText(expectedYamlOutputPath);
            Assert.False(string.IsNullOrWhiteSpace(yamlContent), "Output YAML file is empty.");
            Console.WriteLine("YAML file created and is not empty.");

            // Deserialize and perform further checks
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(PascalCaseNamingConvention.Instance)
                .Build();
            
            List<Placement>? correlatedPlacements = null;
            try
            {
                correlatedPlacements = deserializer.Deserialize<List<Placement>>(yamlContent);
            }
            catch (Exception ex)
            {
                Assert.Fail($"Failed to deserialize YAML content: {ex.Message}\nContent:\n{yamlContent}");
            }
            
            Assert.NotNull(correlatedPlacements); 
            Console.WriteLine($"Deserialized {correlatedPlacements.Count} placements from YAML.");
            Assert.True(correlatedPlacements.Count > 0, "Expected at least one correlated placement in YAML output.");

            // TODO: Add assertions for specific known correlated placement data if available.
            // Example:
            // var specificPlacement = correlatedPlacements.FirstOrDefault(p => p.UniqueId == EXPECTED_ID);
            // Assert.NotNull(specificPlacement);
            // Assert.Equal(EXPECTED_NAME_ID, specificPlacement.NameId);

            Console.WriteLine("--- ProgramTests.RunCorrelation_ShouldProduceCorrectYaml END ---");
        }
    }
} 