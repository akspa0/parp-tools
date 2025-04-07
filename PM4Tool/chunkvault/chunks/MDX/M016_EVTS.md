# EVTS - MDX Events Chunk

## Type
MDX Main Chunk

## Source
MDX_index.md

## Description
The EVTS (Events) chunk defines time-based trigger events that occur during model animations. These events are used to trigger sounds, spawn particle effects, shake the camera, or execute other game actions at specific points in an animation sequence. Events are non-visual markers that signal to the game engine when to perform certain actions, making them essential for synchronizing audio and visual effects with model animations.

## Structure

```csharp
public struct EVTS
{
    /// <summary>
    /// Array of event definitions
    /// </summary>
    // MDLEVENT events[numEvents] follows
}

public struct MDLEVENT
{
    /// <summary>
    /// Name of the event (used to identify the event type)
    /// </summary>
    public string name;
    
    /// <summary>
    /// Event identifier/track
    /// </summary>
    public uint eventId;
    
    /// <summary>
    /// Animation frame when this event triggers
    /// </summary>
    public uint keyFrame;
    
    /// <summary>
    /// Event track timing information
    /// </summary>
    // MDLKEYTRACK track follows
}
```

## Properties

### MDLEVENT Structure

| Offset | Name | Type | Description |
|--------|------|------|-------------|
| 0x00 | name | string | Name of the event, null-terminated. Used to determine event type |
| varies | eventId | uint | Unique identifier for this event (track ID) |
| varies | keyFrame | uint | Frame when this event triggers during animation |
| varies | ... | ... | Animation tracks follow |

## Standard Event Types
The event type is determined by the name field. Common event names include:

| Event Name | Purpose |
|------------|---------|
| "SND" | Play a sound effect |
| "FTP" | Footstep or impact sound |
| "SHP" | Play a weapon swing/swoosh sound |
| "BGND" | Background/ambient sound start/stop |
| "SPL" | Spawn a spell effect or particle system |
| "UBR" | Vibrate/shake the camera |

## Animation Tracks
After the base properties, an event track may follow:
- Visibility/activation track (int, 0 or 1) - Controls when the event is active

## Version Differences

| Version | Changes |
|---------|---------|
| 800-1000 (WC3) | Base structure as described |
| 1300-1500 (WoW Alpha) | Extended event types for MMO gameplay |

## Dependencies
- MDLKEYTRACK - Used for animation tracks within the structure
- SEQS - Events are triggered during animation sequences
- PRE2/PREM - Events may trigger particle effects (referenced by name)

## Implementation Notes
- Events are non-visual markers that signal when to perform certain actions
- The game engine is responsible for implementing appropriate responses to each event type
- Events are typically associated with specific animation sequences
- For sound events, the actual sound file is not stored in the MDX but referenced externally
- Multiple events can be triggered at the same keyFrame
- Events can be enabled/disabled via their visibility track
- The eventId field can be used to categorize events or match them with specific listeners
- Event names are case-sensitive and follow a convention defined by the game engine
- Custom event types can be defined for game-specific interactions
- Events are processed during animation playback when the current frame matches keyFrame
- Some events may have additional data stored in custom fields or referenced by name

## Usage Context
Events in MDX models are used for:
- Attack sounds (sword swing, arrow release, spell cast)
- Footstep sounds during walking/running animations
- Impact sounds (hitting target, landing on ground)
- Starting/stopping looping sounds (machine hum, flame burn)
- Particle effect spawning (blood splatter, dust cloud)
- Screen shake for powerful attacks or impacts
- Triggering game logic (damage application, state change)
- Synchronizing additional animations with primary animation
- Environmental interactions (footsteps on different surfaces)
- Unit voice lines during specific animations

## Implementation Example

```csharp
public class EVTSChunk : IMdxChunk
{
    public string ChunkId => "EVTS";
    
    public List<MdxEvent> Events { get; private set; } = new List<MdxEvent>();
    
    public void Parse(BinaryReader reader, long totalSize)
    {
        long startPosition = reader.BaseStream.Position;
        long endPosition = startPosition + totalSize;
        
        // Clear existing events
        Events.Clear();
        
        // Read events until we reach the end of the chunk
        while (reader.BaseStream.Position < endPosition)
        {
            var mdlEvent = new MdxEvent();
            
            // Read event properties
            mdlEvent.Name = reader.ReadCString();
            mdlEvent.EventId = reader.ReadUInt32();
            mdlEvent.KeyFrame = reader.ReadUInt32();
            
            // Read animation tracks
            mdlEvent.VisibilityTrack = new MdxKeyTrack<int>();
            mdlEvent.VisibilityTrack.Parse(reader, r => r.ReadInt32());
            
            Events.Add(mdlEvent);
        }
    }
    
    public void Write(BinaryWriter writer)
    {
        foreach (var mdlEvent in Events)
        {
            // Write event properties
            writer.WriteCString(mdlEvent.Name);
            writer.Write(mdlEvent.EventId);
            writer.Write(mdlEvent.KeyFrame);
            
            // Write animation tracks
            mdlEvent.VisibilityTrack.Write(writer, (w, i) => w.Write(i));
        }
    }
    
    /// <summary>
    /// Gets all events that should trigger at the specified animation frame
    /// </summary>
    /// <param name="frame">Current animation frame</param>
    /// <param name="time">Current animation time in milliseconds</param>
    /// <param name="sequenceDuration">Duration of the current sequence</param>
    /// <param name="globalSequences">Dictionary of global sequence durations</param>
    /// <returns>List of active events that trigger at this frame</returns>
    public List<MdxEvent> GetEventsAtFrame(uint frame, uint time, uint sequenceDuration, Dictionary<uint, uint> globalSequences)
    {
        List<MdxEvent> result = new List<MdxEvent>();
        
        foreach (var mdlEvent in Events)
        {
            // Check if this event triggers at the current frame
            if (mdlEvent.KeyFrame == frame)
            {
                // Check if the event is visible/active
                bool isActive = true;
                
                if (mdlEvent.VisibilityTrack.NumKeys > 0)
                {
                    isActive = mdlEvent.VisibilityTrack.Evaluate(time, sequenceDuration, globalSequences) > 0;
                }
                
                if (isActive)
                {
                    result.Add(mdlEvent);
                }
            }
        }
        
        return result;
    }
    
    /// <summary>
    /// Finds all events of a specific type
    /// </summary>
    /// <param name="eventType">Event type to search for (e.g., "SND", "FTP")</param>
    /// <returns>List of events matching the specified type</returns>
    public List<MdxEvent> FindEventsByType(string eventType)
    {
        return Events.Where(e => e.Name.StartsWith(eventType, StringComparison.OrdinalIgnoreCase)).ToList();
    }
}

public class MdxEvent
{
    public string Name { get; set; }
    public uint EventId { get; set; }
    public uint KeyFrame { get; set; }
    public MdxKeyTrack<int> VisibilityTrack { get; set; }
    
    /// <summary>
    /// Gets the base event type from the name
    /// </summary>
    public string EventType
    {
        get
        {
            if (string.IsNullOrEmpty(Name)) return "Unknown";
            
            // Event names are typically a 3-letter code followed by additional data
            if (Name.Length >= 3)
            {
                return Name.Substring(0, 3).ToUpperInvariant();
            }
            
            return Name;
        }
    }
    
    /// <summary>
    /// Gets additional event parameters from the name
    /// </summary>
    public string EventParameters
    {
        get
        {
            if (string.IsNullOrEmpty(Name) || Name.Length <= 3) return string.Empty;
            
            return Name.Substring(3);
        }
    }
    
    /// <summary>
    /// Determines if this is a sound event
    /// </summary>
    public bool IsSoundEvent => EventType == "SND" || EventType == "FTP" || EventType == "SHP" || EventType == "BGND";
    
    /// <summary>
    /// Determines if this is a particle effect event
    /// </summary>
    public bool IsParticleEvent => EventType == "SPL";
    
    /// <summary>
    /// Determines if this is a camera effect event
    /// </summary>
    public bool IsCameraEvent => EventType == "UBR";
}
``` 