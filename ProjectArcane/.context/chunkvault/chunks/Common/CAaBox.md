# CAaBox

## Type
Common Type

## Source
Common_Types.md

## Description
An axis-aligned bounding box, defined by its minimum and maximum corners.

## Structure
```csharp
struct CAaBox
{
    C3Vector min;  // minimum corner of the box
    C3Vector max;  // maximum corner of the box
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| min | C3Vector | Minimum corner of the box (lowest x, y, z values) |
| max | C3Vector | Maximum corner of the box (highest x, y, z values) |

## Dependencies
- C3Vector - Used for the minimum and maximum corners

## Implementation Notes
- Used for collision detection and rendering culling
- In the MODF chunk, it represents the transformed WMO bounding box
- The box is axis-aligned, meaning its edges are parallel to the coordinate axes

## Implementation Example
```csharp
public class CAaBox
{
    public C3Vector Min { get; set; } = new C3Vector();
    public C3Vector Max { get; set; } = new C3Vector();
    
    public CAaBox()
    {
        Min = new C3Vector();
        Max = new C3Vector();
    }
    
    public CAaBox(C3Vector min, C3Vector max)
    {
        Min = min;
        Max = max;
    }
    
    public override string ToString()
    {
        return $"Min: {Min}, Max: {Max}";
    }
    
    // Helper method to calculate center of the box
    public C3Vector GetCenter()
    {
        return new C3Vector(
            (Min.X + Max.X) / 2.0f,
            (Min.Y + Max.Y) / 2.0f,
            (Min.Z + Max.Z) / 2.0f
        );
    }
    
    // Helper method to calculate extent of the box
    public C3Vector GetExtent()
    {
        return new C3Vector(
            (Max.X - Min.X) / 2.0f,
            (Max.Y - Min.Y) / 2.0f,
            (Max.Z - Min.Z) / 2.0f
        );
    }
}
```

## Usage Context
CAaBox is used in several chunks, most notably:
- MODF chunk to define the bounds of WMO models for collision and rendering culling
- Other bounding box needs throughout the file formats 