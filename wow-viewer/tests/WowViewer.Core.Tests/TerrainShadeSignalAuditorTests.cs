using WowViewer.Tools.ValidationCapture;

namespace WowViewer.Core.Tests;

public sealed class TerrainShadeSignalAuditorTests
{
    [Fact]
    public void Pearson_ReportsPerfectPositiveAndNegativeCorrelation()
    {
        double[] values = [1, 2, 3, 4, 5];
        Assert.Equal(1.0, TerrainShadeSignalAuditor.Pearson(values, [2, 4, 6, 8, 10]), 10);
        Assert.Equal(-1.0, TerrainShadeSignalAuditor.Pearson(values, [10, 8, 6, 4, 2]), 10);
    }
}
