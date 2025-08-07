# PM4 Core API Specification

## Overview
This document proposes a canonical API for handling PM4 files in the Core library, supporting mesh extraction, boundary analysis, and chunk relationship mapping.

## API Surface
```csharp
public class PM4File : IDisposable
{
    // Loads and parses all relevant chunks from a PM4 file
    public static PM4File FromFile(string path);

    // Returns the full mesh geometry (vertices, faces, etc.)
    public MeshGeometry GetMeshGeometry();

    // Returns the list of exterior (boundary) vertices from MSCN
    public IReadOnlyList<Vector3> GetBoundaryVertices();

    // Returns all doodad placements (nodes) from MSLK, mapped to anchor points
    public IReadOnlyList<DoodadPlacement> GetDoodadPlacements();

    // Returns diagnostic information about chunk relationships
    public PM4Diagnostics GetDiagnostics();

    // Proper resource management
    public void Dispose();
}
```

## Data Models
- `MeshGeometry`: Vertices, faces, normals, UVs, etc.
- `DoodadPlacement`: Group/object ID, anchor point, model reference, transform.
- `PM4Diagnostics`: Chunk presence, match statistics, error/warning logs.

## Example Usage
```csharp
using (var pm4 = PM4File.FromFile("tile_00_00.pm4"))
{
    var mesh = pm4.GetMeshGeometry();
    var boundaries = pm4.GetBoundaryVertices();
    var doodads = pm4.GetDoodadPlacements();
    var diagnostics = pm4.GetDiagnostics();
    // ... further processing ...
}
``` 