# Common Vector Types

## Source
Common_Types.md

## Description
These are common vector types used across multiple WoW file formats.

## Types

### C2Vector
A two-component float vector.

```csharp
struct C2Vector 
{ 
    float x; 
    float y; 
};
```

### C2iVector
A two-component int vector.

```csharp
struct C2iVector 
{ 
    int x; 
    int y; 
};
```

### C3Vector
A three-component float vector.

```csharp
struct C3Vector 
{ 
    /*0x00*/ float x; 
    /*0x04*/ float y; 
    /*0x08*/ float z; 
};
```

### C3iVector
A three-component int vector.

```csharp
struct C3iVector 
{ 
    int x; 
    int y; 
    int z; 
};
```

### C4Vector
A four-component float vector.

```csharp
struct C4Vector 
{ 
    float x; 
    float y; 
    float z; 
    float w; 
};
```

## Implementation Examples

```csharp
public struct C2Vector
{
    public float X { get; set; }
    public float Y { get; set; }
}

public struct C2IVector
{
    public int X { get; set; }
    public int Y { get; set; }
}

public struct C3Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
}

public struct C3IVector
{
    public int X { get; set; }
    public int Y { get; set; }
    public int Z { get; set; }
}

public struct C4Vector
{
    public float X { get; set; }
    public float Y { get; set; }
    public float Z { get; set; }
    public float W { get; set; }
}
```

## Usage Notes
- These vector types are used throughout various WoW file formats
- C3Vector is especially common for positions and normals
- C2Vector is often used for texture coordinates
- Pay attention to the coordinate system when interpreting these vectors 