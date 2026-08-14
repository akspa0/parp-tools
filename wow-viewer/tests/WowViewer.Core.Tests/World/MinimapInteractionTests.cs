using System.Numerics;
using WowViewer.Core.Runtime.World.Minimap;

namespace WowViewer.Core.Tests.World;

public sealed class MinimapInteractionTests
{
    [Fact]
    public void SmallReleaseIsASequentialClickAndThirdSameTargetTeleportsExactlyOnce()
    {
        var state = new MinimapInteractionState();

        AssertClick(state, 100, 28.25f, 29.5f, 1, false);
        AssertClick(state, 500, 28.25f, 29.5f, 2, false);
        MinimapInteractionResult third = AssertClick(state, 900, 28.25f, 29.5f, 3, true);

        Assert.Equal(28.25f, third.TargetTileX);
        Assert.Equal(29.5f, third.TargetTileY);
        Assert.Equal(0, state.PendingClickCount);
    }

    [Fact]
    public void DragProducesPanDeltaAndCancelsTeleportSequence()
    {
        var state = new MinimapInteractionState();
        AssertClick(state, 100, 28f, 29f, 1, false);

        state.Process(MinimapPointerPhase.Pressed, new Vector2(100f, 100f));
        MinimapInteractionResult moved = state.Process(MinimapPointerPhase.Moved, new Vector2(120f, 90f));
        MinimapInteractionResult released = state.Process(
            MinimapPointerPhase.Released,
            new Vector2(120f, 90f),
            hasTarget: true,
            targetTileX: 28f,
            targetTileY: 29f,
            timestampMilliseconds: 200);

        Assert.True(moved.DragStarted);
        Assert.Equal(new Vector2(20f, -10f), moved.PanDeltaPixels);
        Assert.True(released.WasDragging);
        Assert.False(released.TeleportExecuted);
        Assert.Equal(0, state.PendingClickCount);
    }

    [Fact]
    public void ChangedTargetAndTimeoutRestartAtOneClick()
    {
        var state = new MinimapInteractionState();

        AssertClick(state, 100, 1f, 2f, 1, false);
        AssertClick(state, 200, 1f, 3f, 1, false);
        AssertClick(state, 3201, 1f, 3f, 1, false);
    }

    [Fact]
    public void InvalidTargetDoesNotArmTeleport()
    {
        var state = new MinimapInteractionState();
        state.Process(MinimapPointerPhase.Pressed, Vector2.Zero);
        MinimapInteractionResult result = state.Process(
            MinimapPointerPhase.Released,
            Vector2.Zero,
            hasTarget: true,
            targetTileX: 64f,
            targetTileY: 2f,
            timestampMilliseconds: 100);

        Assert.False(result.ClickAccepted);
        Assert.False(result.TeleportExecuted);
        Assert.Equal(0, state.PendingClickCount);
    }

    private static MinimapInteractionResult AssertClick(
        MinimapInteractionState state,
        long timestamp,
        float tileX,
        float tileY,
        int expectedCount,
        bool teleports)
    {
        state.Process(MinimapPointerPhase.Pressed, Vector2.Zero);
        MinimapInteractionResult result = state.Process(
            MinimapPointerPhase.Released,
            Vector2.Zero,
            hasTarget: true,
            targetTileX: tileX,
            targetTileY: tileY,
            timestampMilliseconds: timestamp);

        Assert.True(result.ClickAccepted);
        Assert.Equal(expectedCount, result.ClickCount);
        Assert.Equal(teleports, result.TeleportExecuted);
        return result;
    }
}
