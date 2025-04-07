# C3Vector

## Type
Common Type

## Source
Common_Types.md

## Description
A 3D vector commonly used for positions, rotations, and other 3D values.

## Structure
```csharp
struct C3Vector
{
    float x;
    float y;
    float z;
}
```

## Properties
| Name | Type | Description |
|------|------|-------------|
| x | float | X coordinate |
| y | float | Y coordinate |
| z | float | Z coordinate |

## Dependencies
None

## Implementation Notes
- Used throughout the WoW file formats for 3D positions, rotations, and scaling
- In the World of Warcraft coordinate system:
  - The positive X-axis points north
  - The positive Y-axis points west
  - The Z-axis is vertical height, with 0 being sea level
- When used for rotation, values are typically in degrees

## Implementation Example
```csharp
public class C3Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
    
    public C3Vector()
    {
        X = 0.0f;
        Y = 0.0f;
        Z = 0.0f;
    }
    
    public C3Vector(float x, float y, float z)
    {
        X = x;
        Y = y;
        Z = z;
    }
    
    public override string ToString()
    {
        return $"({X}, {Y}, {Z})";
    }
}
```

## Usage Context
C3Vector is a fundamental type used across multiple chunk types, particularly for:
- Object positions in the world (MDDF, MODF)
- Object rotations (MDDF, MODF)
- Bounding box corners (CAaBox in MODF)
- Many other 3D vector needs throughout the file formats 