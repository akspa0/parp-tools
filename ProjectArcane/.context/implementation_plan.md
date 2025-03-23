# Arcane File Parser Implementation Plan

## Project Overview

The Arcane File Parser is a .NET 8 library designed to parse and manipulate various World of Warcraft file formats. The parser handles the unique big-endian chunk identifiers while maintaining little-endian data format, providing human-readable output for analysis and modification.

## Core Architecture

### Common Infrastructure (Implemented)
- ✓ `ChunkUtils`: Handles endianness conversion for chunk IDs
- ✓ `ChunkBase`: Abstract base class and interfaces for chunk handling
- ✓ `ChunkParser`: Generic chunk parsing framework
- ✓ Basic project structure and namespaces

### Format-Specific Components (In Progress)
- ✓ Base classes for each format type (ADT, WDT, WMO, M2, MDX)
- ✓ Initial MVER chunk implementation for ADT

## Implementation Phases

### Phase 1: Core Format Infrastructure
1. **Common Data Structures**
   - [ ] Vector2/3/4 structures
   - [ ] Color structures (BGRA, RGBA)
   - [ ] Matrix structures
   - [ ] Quaternion structure
   - [ ] Bounds/BoundingBox structures

2. **Binary Reading Extensions**
   - [ ] Extension methods for common data types
   - [ ] Fixed-point number handling
   - [ ] Array reading helpers
   - [ ] String reading (null-terminated, fixed length)
   - [ ] Bitfield handling

3. **Error Handling & Validation**
   - [ ] Custom exception types
   - [ ] Chunk validation framework
   - [ ] Size validation helpers
   - [ ] Version validation system

### Phase 2: ADT Format Implementation
1. **Main Chunks**
   - [x] MVER (Version)
   - [ ] MHDR (Header)
   - [ ] MCIN (Chunk Index)
   - [ ] MTEX (Textures)
   - [ ] MMDX (Models)
   - [ ] MMID (Model IDs)
   - [ ] MWMO (WMO Files)
   - [ ] MWID (WMO IDs)
   - [ ] MDDF (Doodad Placement)
   - [ ] MODF (Object Placement)

2. **MCNK Subchunks**
   - [ ] MCNK (Map Chunk)
   - [ ] MCVT (Height Map)
   - [ ] MCNR (Normals)
   - [ ] MCLY (Layers)
   - [ ] MCRF (References)
   - [ ] MCSH (Shadow Map)
   - [ ] MCAL (Alpha Maps)
   - [ ] MCLQ (Liquid)

3. **Expansion-Specific Chunks**
   - [ ] MH2O (Water Data)
   - [ ] MFBO (Flight Bounds)
   - [ ] MTXF (Texture Flags)
   - [ ] MTXP (Texture Params)

### Phase 3: WDT Format Implementation
1. **Core Chunks**
   - [ ] MVER (Version)
   - [ ] MAIN (Tile Table)
   - [ ] MWMO (WMO Files)
   - [ ] MODF (WMO Placement)

2. **Optional Chunks**
   - [ ] MPHD (Map Header)
   - [ ] MWID (WMO IDs)
   - [ ] MDDF (Doodad Placement)
   - [ ] MODF (Object Placement)

### Phase 4: WMO Format Implementation
1. **Root File Chunks**
   - [ ] MVER (Version)
   - [ ] MOHD (Header)
   - [ ] MOTX (Textures)
   - [ ] MOMT (Materials)
   - [ ] MOGN (Group Names)
   - [ ] MOGI (Group Info)
   - [ ] MOSB (Skybox)
   - [ ] MOPV (Portal Vertices)
   - [ ] MOPT (Portal Data)

2. **Group File Chunks**
   - [ ] MOGP (Group Header)
   - [ ] MOPY (Material Info)
   - [ ] MOVI (Indices)
   - [ ] MOVT (Vertices)
   - [ ] MONR (Normals)
   - [ ] MOTV (TexCoords)

### Phase 5: M2 Format Implementation
1. **Core Chunks**
   - [ ] MD20/MD21 (Main Data)
   - [ ] SFID (Skin Profiles)
   - [ ] AFID (Animations)
   - [ ] BFID (Bones)
   - [ ] TXAC (Textures)

2. **Animation System**
   - [ ] Animation tracks
   - [ ] Bone hierarchies
   - [ ] Vertex deformation
   - [ ] Texture animation

### Phase 6: Database Format Support
1. **DBC/DB2**
   - [ ] Header parsing
   - [ ] Record structure
   - [ ] String block handling
   - [ ] Field definitions

2. **Common Features**
   - [ ] Record iteration
   - [ ] Field access
   - [ ] String lookup
   - [ ] Relationship tracking

## Testing Strategy

### Unit Tests
1. **Core Components**
   - [ ] Endianness conversion
   - [ ] Chunk identification
   - [ ] Size validation
   - [ ] Data structure handling

2. **Format-Specific**
   - [ ] Chunk parsing accuracy
   - [ ] Version validation
   - [ ] Cross-references
   - [ ] Data integrity

### Integration Tests
1. **File Processing**
   - [ ] Complete file parsing
   - [ ] Cross-chunk references
   - [ ] Error handling
   - [ ] Memory management

2. **Format Compatibility**
   - [ ] Version compatibility
   - [ ] Expansion features
   - [ ] Data consistency

## Documentation

### API Documentation
- [ ] XML documentation for public APIs
- [ ] Usage examples
- [ ] Best practices
- [ ] Error handling guidelines

### Format Documentation
- [ ] Chunk structure details
- [ ] Version differences
- [ ] Dependencies
- [ ] Validation rules

## Tools and Utilities

### Development Tools
1. **Debugging**
   - [ ] Chunk viewer
   - [ ] Hex comparison
   - [ ] Structure visualization

2. **Testing**
   - [ ] Sample file generator
   - [ ] Corruption simulator
   - [ ] Performance profiler

### User Tools
1. **Analysis**
   - [ ] File structure viewer
   - [ ] Data exporters
   - [ ] Format converters

2. **Validation**
   - [ ] Format checker
   - [ ] Reference validator
   - [ ] Integrity verifier

## Implementation Notes

### Coding Standards
- Use C# 12 features where appropriate
- Follow .NET naming conventions
- Implement nullable reference types
- Use modern collection types
- Implement proper IDisposable patterns

### Performance Considerations
- Minimize allocations in parsing loops
- Use spans for binary data
- Implement lazy loading where appropriate
- Consider memory pooling for large files
- Use async IO for file operations

### Error Handling
- Detailed exception messages
- Proper exception hierarchy
- Validation at chunk boundaries
- Recovery mechanisms
- Logging infrastructure

## Future Enhancements

### Planned Features
1. **Format Support**
   - Additional file formats
   - New expansion features
   - Custom format extensions

2. **Tooling**
   - Visual editors
   - Batch processors
   - Format converters

3. **Performance**
   - Parallel processing
   - Memory optimization
   - Caching systems

### Research Areas
- New format variations
- Compression techniques
- Optimization strategies
- Tool integration

## WMO Format Implementation Plan

## Phase Status Legend
✅ Complete
🔄 In Progress
⏳ Pending

## Implementation Phases

### Phase 1: Core Functionality ✅
- [x] Basic chunk parsing infrastructure
- [x] File reading/writing framework
- [x] Version detection and handling
- [x] Validation system

### Phase 2: Portal System ✅
- [x] MOPT (Portals)
- [x] MOPR (Portal References)
- [x] Portal culling logic
- [x] Portal validation

### Phase 3: Group Data ✅
- [x] MOGP (Group Header)
- [x] MOGI (Group Info)
- [x] Group flags and properties
- [x] Group validation

### Phase 4: Visibility System ✅
- [x] MOVV (Visible Vertices)
- [x] MOVB (Visible Blocks)
- [x] MOBA (Render Batches)
- [x] Visibility validation

### Phase 5: Legacy Lighting ✅
- [x] MOLV (Lightmap Vertices)
- [x] MOIN (Index List)
- [x] MOMA (Material Attributes)
- [x] Legacy lighting validation

### Phase 6: Version-Specific Features ✅
- [x] MOUV (UV Animation)
- [x] MOSI (Skybox)
- [x] MODI (Doodad Info)
- [x] Version-specific validation

### Phase 7: Volume System ✅
- [x] MBVD (Box Volumes)
- [x] MAVD (Ambient Volumes)
- [x] MAVG (Ambient Volume Groups)
- [x] Volume validation

### Phase 8: Modern Lighting System ✅
- [x] MOLS (Spot Lights)
- [x] MOLP (Light Points)
- [x] MOLR (Light References)
- [x] Modern lighting validation

### Phase 9: Volume References ⏳
- [ ] MBVR (Box Volume References)
- [ ] MFVR (Fog Volume References)
- [ ] MAVR (Ambient Volume References)
- [ ] Reference validation

### Phase 10: Testing and Validation ⏳
- [ ] Unit test suite
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case handling

### Phase 11: Documentation ⏳
- [ ] API documentation
- [ ] Usage examples
- [ ] Version compatibility guide
- [ ] Best practices guide

## Current Status
- Phases 1-8 complete
- Phase 9 pending
- Testing and documentation ongoing

## Next Steps
1. Implement volume reference chunks (MBVR, MFVR, MAVR)
2. Develop comprehensive test suite
3. Complete documentation
4. Performance optimization

## Legend
- ✅ Complete
- 🔄 In Progress
- ⏳ Pending

## Phase Status

### Completed Phases
1. ✅ Core WMO Structure
2. ✅ Portal System Implementation
3. ✅ Group System Implementation
4. ✅ Visibility System
5. ✅ Legacy Lighting (v14)
6. ✅ Modern Lighting System
7. ✅ Volume System
8. ✅ Volume References

### Current Phase
9. 🔄 Conversion Utilities
   - WMO v17 to v14 converter
   - Path validation and conversion
   - Asset reference handling
   - Chunk compatibility checks

### Future Phases
10. 📅 Optimization
    - Chunk processing optimization
    - Memory usage improvements
    - Performance benchmarking
    - Validation enhancements

11. 📅 Documentation
    - Cross-chunk dependencies
    - Version compatibility matrix
    - Conversion guidelines
    - API documentation

## Implementation Notes
- All 48 WMO chunks successfully implemented
- Both v14 and v17 formats fully supported
- Legacy features preserved for backward compatibility
- Modern features properly implemented with version checks
- Validation and error handling in place 