using System.Numerics;

namespace WowViewer.Core.Runtime.World.Minimap;

public enum MinimapPointerPhase
{
    Pressed,
    Moved,
    Released,
    Cancelled,
}

public readonly record struct MinimapInteractionResult(
    bool PointerCaptured,
    bool DragStarted,
    bool WasDragging,
    Vector2 PanDeltaPixels,
    bool ClickAccepted,
    int ClickCount,
    bool TeleportExecuted,
    float TargetTileX,
    float TargetTileY);

/// <summary>
/// Pure pointer state for the shared minimap surface.
/// ImGui and camera mutation stay in the viewer adapter; this type only classifies
/// press/move/release sequences and owns the exact three-click teleport sequence.
/// </summary>
public sealed class MinimapInteractionState
{
    public const float DefaultClickMovementThresholdPixels = 6f;
    public const long DefaultTeleportWindowMilliseconds = 3000;
    public const int TeleportConfirmClicks = 3;
    public const int MapTileCount = 64;

    private readonly float _clickMovementThresholdPixels;
    private readonly long _teleportWindowMilliseconds;
    private bool _pointerDown;
    private bool _dragging;
    private Vector2 _lastPointerPosition;
    private Vector2 _pressPosition;
    private (int TileX, int TileY)? _pendingTarget;
    private int _clickCount;
    private long _lastClickTimestampMilliseconds = long.MinValue;

    public MinimapInteractionState(
        float clickMovementThresholdPixels = DefaultClickMovementThresholdPixels,
        long teleportWindowMilliseconds = DefaultTeleportWindowMilliseconds)
    {
        if (!float.IsFinite(clickMovementThresholdPixels) || clickMovementThresholdPixels < 0f)
            throw new ArgumentOutOfRangeException(nameof(clickMovementThresholdPixels));
        if (teleportWindowMilliseconds < 0)
            throw new ArgumentOutOfRangeException(nameof(teleportWindowMilliseconds));

        _clickMovementThresholdPixels = clickMovementThresholdPixels;
        _teleportWindowMilliseconds = teleportWindowMilliseconds;
    }

    public bool PointerDown => _pointerDown;

    public bool IsDragging => _dragging;

    public int PendingClickCount => _clickCount;

    public void Reset()
    {
        _pointerDown = false;
        _dragging = false;
        _lastPointerPosition = Vector2.Zero;
        _pressPosition = Vector2.Zero;
        ClearClickSequence();
    }

    public MinimapInteractionResult Process(
        MinimapPointerPhase phase,
        Vector2 pointerPosition,
        bool hasTarget = false,
        float targetTileX = 0f,
        float targetTileY = 0f,
        long timestampMilliseconds = 0)
    {
        return phase switch
        {
            MinimapPointerPhase.Pressed => Press(pointerPosition),
            MinimapPointerPhase.Moved => Move(pointerPosition),
            MinimapPointerPhase.Released => Release(
                pointerPosition,
                hasTarget,
                targetTileX,
                targetTileY,
                timestampMilliseconds),
            MinimapPointerPhase.Cancelled => Cancel(),
            _ => throw new ArgumentOutOfRangeException(nameof(phase)),
        };
    }

    private MinimapInteractionResult Press(Vector2 pointerPosition)
    {
        _pointerDown = true;
        _dragging = false;
        _pressPosition = pointerPosition;
        _lastPointerPosition = pointerPosition;
        return new MinimapInteractionResult(true, false, false, Vector2.Zero, false, _clickCount, false, 0f, 0f);
    }

    private MinimapInteractionResult Move(Vector2 pointerPosition)
    {
        if (!_pointerDown)
            return default;

        Vector2 delta = pointerPosition - _lastPointerPosition;
        _lastPointerPosition = pointerPosition;
        bool dragStarted = false;
        if (!_dragging && Vector2.DistanceSquared(pointerPosition, _pressPosition)
            > _clickMovementThresholdPixels * _clickMovementThresholdPixels)
        {
            _dragging = true;
            dragStarted = true;
            ClearClickSequence();
        }

        return new MinimapInteractionResult(true, dragStarted, _dragging,
            _dragging ? delta : Vector2.Zero, false, 0, false, 0f, 0f);
    }

    private MinimapInteractionResult Release(
        Vector2 pointerPosition,
        bool hasTarget,
        float targetTileX,
        float targetTileY,
        long timestampMilliseconds)
    {
        if (!_pointerDown)
            return default;

        bool wasDragging = _dragging || Vector2.DistanceSquared(pointerPosition, _pressPosition)
            > _clickMovementThresholdPixels * _clickMovementThresholdPixels;
        _pointerDown = false;
        _dragging = false;

        if (wasDragging)
        {
            ClearClickSequence();
            return new MinimapInteractionResult(false, false, true, Vector2.Zero, false, 0, false, 0f, 0f);
        }

        if (!hasTarget || !IsValidTarget(targetTileX, targetTileY))
        {
            ClearClickSequence();
            return default;
        }

        int tileX = (int)MathF.Floor(targetTileX);
        int tileY = (int)MathF.Floor(targetTileY);
        bool sameTarget = _pendingTarget is { } pending
            && pending.TileX == tileX
            && pending.TileY == tileY
            && timestampMilliseconds - _lastClickTimestampMilliseconds <= _teleportWindowMilliseconds;

        _clickCount = sameTarget ? _clickCount + 1 : 1;
        _pendingTarget = (tileX, tileY);
        _lastClickTimestampMilliseconds = timestampMilliseconds;
        bool teleportExecuted = _clickCount >= TeleportConfirmClicks;
        int completedClickCount = _clickCount;
        if (teleportExecuted)
            ClearClickSequence();

        return new MinimapInteractionResult(false, false, false, Vector2.Zero, true,
            completedClickCount, teleportExecuted, targetTileX, targetTileY);
    }

    private MinimapInteractionResult Cancel()
    {
        bool wasDragging = _dragging;
        Reset();
        return new MinimapInteractionResult(false, false, wasDragging, Vector2.Zero, false, 0, false, 0f, 0f);
    }

    private void ClearClickSequence()
    {
        _pendingTarget = null;
        _clickCount = 0;
        _lastClickTimestampMilliseconds = long.MinValue;
    }

    private static bool IsValidTarget(float tileX, float tileY)
    {
        return float.IsFinite(tileX)
            && float.IsFinite(tileY)
            && tileX >= 0f
            && tileX < MapTileCount
            && tileY >= 0f
            && tileY < MapTileCount;
    }
}
