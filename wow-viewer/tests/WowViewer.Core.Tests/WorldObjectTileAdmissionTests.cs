using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests;

public sealed class WorldObjectTileAdmissionTests
{
    [Fact]
    public void RetainedNeighborIsAdmittedEvenWhenOutsideDirectionalDetailSet()
    {
        var detailed = new List<(int tileX, int tileY)> { (32, 32) };
        var retained = new List<(int tileX, int tileY)> { (31, 32) };

        Assert.True(WorldObjectTileAdmission.IsResident(detailed, retained, (31, 32)));
    }

    [Fact]
    public void UnresidentTileRemainsRejected()
    {
        var detailed = new List<(int tileX, int tileY)> { (32, 32) };
        var retained = new List<(int tileX, int tileY)> { (31, 32) };

        Assert.False(WorldObjectTileAdmission.IsResident(detailed, retained, (30, 30)));
    }
}
