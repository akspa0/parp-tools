using WowViewer.Core.M2;
using Xunit;

namespace WowViewer.Core.Tests;

public class IdleAnimationSequenceSelectorTests
{
    [Fact]
    public void FindDefaultIdleSequenceIndex_ExactStand_SelectedFirst()
    {
        string[] sequences = ["Walk", "Run", "Stand", "Attack1H"];
        int index = M2AnimationNameResolver.FindDefaultIdleSequenceIndex(sequences);
        Assert.Equal(2, index);
    }

    [Fact]
    public void FindDefaultIdleSequenceIndex_ExactIdle_SelectedFirst()
    {
        string[] sequences = ["Walk", "Run", "Idle", "Death"];
        int index = M2AnimationNameResolver.FindDefaultIdleSequenceIndex(sequences);
        Assert.Equal(2, index);
    }

    [Fact]
    public void FindDefaultIdleSequenceIndex_ContainsStandOrIdle_SelectedOverMovement()
    {
        string[] sequences = ["Walk", "Run", "Stand_01", "Attack2H"];
        int index = M2AnimationNameResolver.FindDefaultIdleSequenceIndex(sequences);
        Assert.Equal(2, index);
    }

    [Fact]
    public void FindDefaultIdleSequenceIndex_NoStandOrIdle_AvoidsWalkAndRun()
    {
        string[] sequences = ["Walk", "Run", "Ready1H", "Death"];
        int index = M2AnimationNameResolver.FindDefaultIdleSequenceIndex(sequences);
        Assert.Equal(2, index); // Ready1H is chosen over Walk and Run
    }

    [Fact]
    public void FindDefaultIdleSequenceIndex_OnlyWalkAndRun_FallsBackToZero()
    {
        string[] sequences = ["Walk", "Run"];
        int index = M2AnimationNameResolver.FindDefaultIdleSequenceIndex(sequences);
        Assert.Equal(0, index);
    }

    [Fact]
    public void FindDefaultIdleSequenceIndex_EmptyOrNull_ReturnsNegativeOne()
    {
        Assert.Equal(-1, M2AnimationNameResolver.FindDefaultIdleSequenceIndex(null!));
        Assert.Equal(-1, M2AnimationNameResolver.FindDefaultIdleSequenceIndex(Array.Empty<string>()));
    }
}
