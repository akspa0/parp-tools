# WoW Alpha 0.5.3 Player Movement

## Overview

The WoW Alpha 0.5.3 movement system is event-driven, with movement commands being queued and processed each frame.

## Movement Events

**Address:** [`CMovement::UpdatePlayerMovement`](0x004c4d90) (0x004c4d90)

### Movement Event Types

```c
enum MovementEventType {
    MOVE_START_FORWARD = 0,      // Start moving forward
    MOVE_START_BACKWARD = 1,     // Start moving backward
    MOVE_STOP = 2,               // Stop moving
    MOVE_START_STRAFE_LEFT = 3,  // Start strafing left
    MOVE_START_STRAFE_RIGHT = 4, // Start strafing right
    MOVE_STOP_STRAFE = 5,       // Stop strafing
    MOVE_START_FALLING = 6,      // Start falling
    MOVE_JUMP = 7,               // Jump
    MOVE_START_TURN_LEFT = 8,     // Start turning left
    MOVE_START_TURN_RIGHT = 9,    // Start turning right
    MOVE_STOP_TURN = 10,         // Stop turning
    MOVE_START_PITCH_UP = 11,    // Start pitching up
    MOVE_START_PITCH_DOWN = 12,  // Start pitching down
    MOVE_STOP_PITCH = 13,        // Stop pitching
    MOVE_SET_RUN = 14,          // Set run mode
    MOVE_SET_WALK = 15,         // Set walk mode
    MOVE_SET_FACING = 16,        // Set facing direction
    MOVE_SET_PITCH = 17,         // Set pitch
    MOVE_START_SWIM = 18,        // Start swimming
    MOVE_STOP_SWIM = 19,         // Stop swimming
};
```

### Movement Event Structure

```c
struct CPlayerMoveEvent {
    ulong timestamp;           // Event timestamp
    uint type;                // Movement event type
    float facing;             // Facing direction (for MOVE_SET_FACING)
    float pitch;               // Pitch angle (for MOVE_SET_PITCH)
    void* object;             // Associated object
};
```

## Movement Processing

### Purpose

Process player movement events from the event queue.

### Algorithm

1. Get movement globals
2. Iterate through movement events
3. For each event:
   - Check if event is in the past
   - Remove event from list
   - Process event based on type
   - Free event object
4. Check if there are any active movements
5. Return true if there are active movements

### Pseudocode

```c
int UpdatePlayerMovement(CMovement* movement, ulong currentTime) {
    void* globals = MovementGetGlobals();
    
    // Process movement events
    while (globals->moveEventCount > 0) {
        TSLink<CPlayerMoveEvent>* event = globals->moveEventList;
        
        // Check if event is in the past
        if ((int)(currentTime - event->timestamp) < 0) {
            break;
        }
        
        // Remove event from list
        RemoveMoveEvent(event);
        
        // Process event based on type
        switch (event->type) {
            case MOVE_START_FORWARD:
                StartMove(movement, currentTime, true);
                break;
            case MOVE_START_BACKWARD:
                StartMove(movement, currentTime, false);
                break;
            case MOVE_STOP:
                StopMove(movement, currentTime);
                break;
            case MOVE_START_STRAFE_LEFT:
                StartStrafe(movement, currentTime, true);
                break;
            case MOVE_START_STRAFE_RIGHT:
                StartStrafe(movement, currentTime, false);
                break;
            case MOVE_STOP_STRAFE:
                StopStrafe(movement, currentTime);
                break;
            case MOVE_START_FALLING:
                StartFalling(movement, currentTime);
                OnCollideFalling(movement->object, currentTime);
                break;
            case MOVE_JUMP:
                Jump(movement, currentTime);
                break;
            case MOVE_START_TURN_LEFT:
                StartTurn(movement, currentTime, true);
                break;
            case MOVE_START_TURN_RIGHT:
                StartTurn(movement, currentTime, false);
                break;
            case MOVE_STOP_TURN:
                StopTurn(movement, currentTime);
                break;
            case MOVE_START_PITCH_UP:
                StartPitch(movement, currentTime, true);
                break;
            case MOVE_START_PITCH_DOWN:
                StartPitch(movement, currentTime, false);
                break;
            case MOVE_STOP_PITCH:
                StopPitch(movement, currentTime);
                break;
            case MOVE_SET_RUN:
                SetRunMode(movement, currentTime, true);
                break;
            case MOVE_SET_WALK:
                SetRunMode(movement, currentTime, false);
                break;
            case MOVE_SET_FACING:
                SetFacing(movement, currentTime, event->facing);
                break;
            case MOVE_SET_PITCH:
                SetPitch(movement, currentTime, event->pitch);
                break;
            case MOVE_START_SWIM:
                StartSwimLocal(movement, currentTime);
                break;
            case MOVE_STOP_SWIM:
                StopSwimLocal(movement, currentTime);
                break;
        }
        
        // Free event
        ObjectFree(event->object);
    }
    
    // Check if there are any active movements
    if (globals->moveEventCount == 0 && (movement->flags & 0x40ff) == 0) {
        return 0;
    }
    
    return 1;
}
```

## Collision Handling

### Collision Functions

- [`OnCollideFalling`](0x005f3540) (0x005f3540) - Handle collision while falling
- [`OnCollideFallLand`](0x005f34f0) (0x005f34f0) - Handle landing after fall
- [`OnCollideRedirected`](0x005f33b0) (0x005f33b0) - Handle redirected collision
- [`OnCollideStuck`](0x005f3410) (0x005f3410) - Handle stuck collision

### Collision States

```c
enum CollisionState {
    COLLISION_FALLING,      // Player is falling
    COLLISION_LANDED,       // Player has landed
    COLLISION_REDIRECTED,   // Player was redirected by collision
    COLLISION_STUCK,        // Player is stuck
};
```

## Movement Constants

### Speed Constants

```c
// Movement speeds
const float WALK_SPEED = 3.5f;      // Walking speed (units/second)
const float RUN_SPEED = 7.0f;       // Running speed (units/second)
const float SWIM_SPEED = 3.5f;     // Swimming speed (units/second)
const float TURN_SPEED = 2.0f;      // Turning speed (radians/second)
const float PITCH_SPEED = 1.0f;     // Pitching speed (radians/second)

// Jump constants
const float JUMP_VELOCITY = 8.0f;    // Initial jump velocity (units/second)
const float GRAVITY = 9.8f;          // Gravity acceleration (units/second²)
const float TERMINAL_VELOCITY = 50.0f; // Maximum falling velocity (units/second)
```

### Movement Flags

```c
// Movement flags
const uint FLAG_MOVING = 0x01;        // Player is moving
const uint FLAG_STRAFING = 0x02;     // Player is strafing
const uint FLAG_TURNING = 0x04;       // Player is turning
const uint FLAG_PITCHING = 0x08;      // Player is pitching
const uint FLAG_FALLING = 0x10;       // Player is falling
const uint FLAG_JUMPING = 0x20;       // Player is jumping
const uint FLAG_RUNNING = 0x40;       // Player is running
const uint FLAG_SWIMMING = 0x80;      // Player is swimming
```

## Implementation Guidelines

### C# Player Movement

```csharp
public class PlayerMovement
{
    private C3Vector position;
    private float facing;
    private float pitch;
    private bool isMoving;
    private bool isStrafing;
    private bool isFalling;
    private bool isJumping;
    private bool isRunning;
    private float velocity;
    private float strafeVelocity;
    private float verticalVelocity;
    
    // Constants
    private const float WALK_SPEED = 3.5f;
    private const float RUN_SPEED = 7.0f;
    private const float SWIM_SPEED = 3.5f;
    private const float TURN_SPEED = 2.0f;
    private const float PITCH_SPEED = 1.0f;
    private const float JUMP_VELOCITY = 8.0f;
    private const float GRAVITY = 9.8f;
    private const float TERMINAL_VELOCITY = 50.0f;
    
    public void Update(float deltaTime, TerrainCollision collision)
    {
        // Calculate movement direction
        C3Vector moveDirection = new C3Vector();
        
        if (isMoving)
        {
            moveDirection.X = (float)Math.Sin(facing);
            moveDirection.Y = (float)Math.Cos(facing);
        }
        
        if (isStrafing)
        {
            moveDirection.X += (float)Math.Sin(facing + Math.PI / 2);
            moveDirection.Y += (float)Math.Cos(facing + Math.PI / 2);
        }
        
        // Normalize movement direction
        if (moveDirection.Length() > 0)
        {
            moveDirection = Vector3.Normalize(moveDirection);
        }
        
        // Calculate speed
        float speed = isRunning ? RUN_SPEED : WALK_SPEED;
        float moveSpeed = speed * deltaTime;
        
        // Calculate new position
        C3Vector newPosition = position;
        newPosition.X += moveDirection.X * moveSpeed;
        newPosition.Y += moveDirection.Y * moveSpeed;
        
        // Apply gravity if falling
        if (isFalling)
        {
            verticalVelocity -= GRAVITY * deltaTime;
            verticalVelocity = Math.Max(verticalVelocity, -TERMINAL_VELOCITY);
            newPosition.Z += verticalVelocity * deltaTime;
        }
        
        // Apply jump velocity
        if (isJumping)
        {
            verticalVelocity = JUMP_VELOCITY;
            isJumping = false;
            isFalling = true;
        }
        
        // Check for terrain collision
        C3Vector groundPosition = newPosition;
        groundPosition.Z = position.Z + 100.0f;
        
        if (collision.RayCast(groundPosition, new C3Vector(newPosition.X, newPosition.Y, newPosition.Z - 200.0f), 
            out float t, out C3Vector point, out C4Plane plane))
        {
            // Check if we're below the ground
            if (newPosition.Z < point.Z)
            {
                // Snap to ground
                newPosition.Z = point.Z;
                
                // Stop falling
                if (isFalling)
                {
                    isFalling = false;
                    verticalVelocity = 0.0f;
                }
            }
        }
        else
        {
            // No ground below, start falling
            if (!isFalling)
            {
                isFalling = true;
                verticalVelocity = 0.0f;
            }
        }
        
        // Update position
        position = newPosition;
    }
    
    public void StartMove(bool forward)
    {
        isMoving = true;
        velocity = forward ? 1.0f : -1.0f;
    }
    
    public void StopMove()
    {
        isMoving = false;
        velocity = 0.0f;
    }
    
    public void StartStrafe(bool left)
    {
        isStrafing = true;
        strafeVelocity = left ? 1.0f : -1.0f;
    }
    
    public void StopStrafe()
    {
        isStrafing = false;
        strafeVelocity = 0.0f;
    }
    
    public void Jump()
    {
        if (!isFalling)
        {
            isJumping = true;
        }
    }
    
    public void SetFacing(float newFacing)
    {
        facing = newFacing;
    }
    
    public void SetPitch(float newPitch)
    {
        pitch = newPitch;
    }
    
    public void SetRunMode(bool run)
    {
        isRunning = run;
    }
    
    public void StartSwim()
    {
        isSwimming = true;
    }
    
    public void StopSwim()
    {
        isSwimming = false;
    }
}
```

## References

- [`CMovement::UpdatePlayerMovement`](0x004c4d90) (0x004c4d90) - Process player movement events
- [`OnCollideFalling`](0x005f3540) (0x005f3540) - Handle collision while falling
- [`OnCollideFallLand`](0x005f34f0) (0x005f34f0) - Handle landing after fall
- [`OnCollideRedirected`](0x005f33b0) (0x005f33b0) - Handle redirected collision
- [`OnCollideStuck`](0x005f3410) (0x005f3410) - Handle stuck collision
