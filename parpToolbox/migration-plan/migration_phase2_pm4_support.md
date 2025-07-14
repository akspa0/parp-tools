# Phase 2: PM4/PD4 Support Implementation

## Goals
1. Add PM4/PD4 format support to WoWFormatLib
2. Ensure compatibility with existing file loading infrastructure
3. Provide adapters for PM4Tool's domain models

## Tasks

### 2.1 PM4/PD4 Struct Definitions
- [ ] Create `PM4.Struct.cs` in `WoWFormatLib/Structs/`
  - [ ] Define chunk headers and data structures
  - [ ] Add documentation for each field
  - [ ] Include version-specific layouts if needed

### 2.2 PM4Reader Implementation
- [ ] Create `PM4Reader.cs` in `WoWFormatLib/FileReaders/`
  - [ ] Implement `Load` method for PM4 files
  - [ ] Add support for all known PM4 chunks
  - [ ] Handle endianness and alignment

### 2.3 Integration with File Provider System
- [ ] Register PM4 file extension with WoWFormatLib
- [ ] Add PM4 to the list of supported formats
- [ ] Update file type detection

### 2.4 Testing
- [ ] Add test PM4 files to test assets
- [ ] Create unit tests for PM4Reader
- [ ] Verify data integrity through round-trip tests

## Success Criteria
- [ ] PM4 files can be loaded through WoWFormatLib
- [ ] All known PM4 chunks are parsed correctly
- [ ] Tests pass with sample PM4 files
- [ ] No performance regressions in file loading

## Dependencies
- PM4 file format specification
- Sample PM4 files for testing
- WoWFormatLib core updates (if needed)

## Risks & Mitigation
- **Risk**: Incomplete format specification
  - **Mitigation**: Document known fields and mark others as unknown
- **Risk**: Performance issues with large PM4 files
  - **Mitigation**: Implement streaming or chunked reading if needed
