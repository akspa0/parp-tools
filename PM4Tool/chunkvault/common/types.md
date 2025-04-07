# Common Types

This document describes the common types used across various WoW file formats. These types are fundamental building blocks used throughout the file format specifications.

## Vector Types

### C2Vector
A two-component float vector.
```cpp
struct C2Vector {
    float x;
    float y;
};
```

### C2iVector
A two-component integer vector.
```cpp
struct C2iVector {
    int x;
    int y;
};
```

### C3Vector
A three-component float vector.
```cpp
struct C3Vector {
    float x;
    float y;
    float z;
};
```

### C3iVector
A three-component integer vector.
```cpp
struct C3iVector {
    int x;
    int y;
    int z;
};
```

### C4Vector
A four-component float vector.
```cpp
struct C4Vector {
    float x;
    float y;
    float z;
    float w;
};
```

### C4iVector
A four-component integer vector.
```cpp
struct C4iVector {
    int x;
    int y;
    int z;
    int w;
};
```

### C3sVector
A three-component vector of shorts.
```cpp
struct C3sVector {
    int16_t x;
    int16_t y;
    int16_t z;
};
```

## Matrix Types

### C33Matrix
A 3x3 matrix.
```cpp
struct C33Matrix {
    C3Vector columns[3];
};
```

### C34Matrix
A 3x4 matrix.
```cpp
struct C34Matrix {
    C3Vector columns[4];
};
```

### C44Matrix
A 4x4 column-major matrix.
```cpp
struct C44Matrix {
    C4Vector columns[4];
};
```

## Geometric Types

### C4Plane
A 3D plane defined by four floats (a, b, c, d from general plane equation).
```cpp
struct C4Plane {
    C3Vector normal;
    float distance;
};
```

### C4Quaternion
A quaternion. Note: Unlike quaternions elsewhere, the scalar part ('w') is the last element.
```cpp
struct C4Quaternion {
    float x;
    float y;
    float z;
    float w;
};
```

### CAaBox
An axis-aligned box described by minimum and maximum points.
```cpp
struct CAaBox {
    C3Vector min;
    C3Vector max;
};
```

### CAaSphere
An axis-aligned sphere described by position and radius.
```cpp
struct CAaSphere {
    C3Vector position;
    float radius;
};
```

### C3Segment
A line segment defined by start and end points.
```cpp
struct C3Segment {
    C3Vector start;
    C3Vector end;
};
```

### CFacet
A triangular facet with a plane equation and vertices.
```cpp
struct CFacet {
    C4Plane plane;
    C3Vector vertices[3];
};
```

### C3Ray
A ray defined by origin and direction.
```cpp
struct C3Ray {
    C3Vector origin;
    C3Vector dir;
};
```

## Range Types

### CRange
A one-dimensional float range defined by bounds.
```cpp
struct CRange {
    float min;
    float max;
};
```

### CiRange
A one-dimensional integer range defined by bounds.
```cpp
struct CiRange {
    int min;
    int max;
};
```

## Rectangle Types

### CRect
A floating-point rectangle. Members can be accessed as either min/max or top/left/bottom/right.
```cpp
struct CRect {
    float miny;  // top
    float minx;  // left
    float maxy;  // bottom
    float maxx;  // right
};
```

### CiRect
An integer rectangle. Members can be accessed as either min/max or top/left/bottom/right.
```cpp
struct CiRect {
    int miny;  // top
    int minx;  // left
    int maxy;  // bottom
    int maxx;  // right
};
```

## Color Types

### CArgb
A color given in values of red, green, blue and alpha.
```cpp
struct CArgb {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};
```

### CImVector
A color given in values of blue, green, red and alpha.
```cpp
struct CImVector {
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;
};
```

## Implementation Notes

1. All vector types should implement:
   - Basic construction
   - Binary reading/writing
   - ToString() for debugging
   - Equality comparison where appropriate

2. Color types should additionally support:
   - Normalization (0-255 to 0.0-1.0)
   - Construction from normalized values
   - Color space conversion where needed

3. Geometric types should include:
   - Appropriate mathematical operations
   - Intersection testing where applicable
   - Distance calculations where applicable

4. All types should follow WoW's binary layout exactly for file format compatibility 