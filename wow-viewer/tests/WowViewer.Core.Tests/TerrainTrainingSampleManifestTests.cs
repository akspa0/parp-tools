using WowViewer.Core.Datasets;

namespace WowViewer.Core.Tests;

public sealed class TerrainTrainingSampleManifestTests
{
	[Fact]
	public void Descriptor_PreservesBrushSignalAvailabilityAndMetrics()
	{
		TerrainTrainingSampleDescriptor descriptor = new(
			sampleId: "3_3_5_12340:development_0_0",
			sourceKind: TerrainTrainingSampleSourceKind.ClientRoot,
			buildLabel: "3_3_5_12340",
			mapName: "development",
			tileX: 0,
			tileY: 0,
			sourceRoot: @"H:\CLIENTS\3.3.5.12340",
			rootAdtPath: @"H:\CLIENTS\3.3.5.12340\Data\World\Maps\development\development_0_0.adt")
		{
			Signals = new TerrainTrainingSignalAvailability
			{
				HasRootAdt = true,
				HasBrushMask = true,
				HasLiquidMask = true,
			},
			Metrics = new TerrainTrainingSampleMetrics
			{
				LiquidCoverage = 0.35f,
				BrushCoverage = 0.12f,
				ObjectCoverage = 0.08f,
			},
		};

		Assert.Equal("development_0_0", descriptor.TileName);
		Assert.True(descriptor.Signals.HasBrushMask);
		Assert.Equal(0.12f, descriptor.Metrics.BrushCoverage);
		Assert.Equal(0.35f, descriptor.Metrics.LiquidCoverage);
	}
}