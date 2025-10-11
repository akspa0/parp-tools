# Phase 1: Setup & WoWFormatLib Integration

## Goals
1. Add WoWFormatLib as a project reference to the solution.
2. Create a thin adapter layer to bridge WoWFormatLib's models to PM4Tool's domain models.
3. Verify WMO loading and OBJ export functionality works end-to-end.

## Tasks

### 1.1 Add WoWFormatLib to Solution
- [ ] Add `WoWFormatLib` project to the solution under `lib/wow.tools.local/WoWFormatLib`
- [ ] Add project reference from `WoWToolbox.Core.v2` to `WoWFormatLib`
- [ ] Resolve any namespace conflicts (e.g., `WoWFormatLib.Structs.WMO` vs `WoWToolbox.Core.v2.Foundation.WMO`)

### 1.2 Create Adapter Layer
- [ ] Create new project: `WoWToolbox.Adapters`
  - [ ] Add reference to `WoWFormatLib`
  - [ ] Add reference to `WoWToolbox.Core.v2`
- [ ] Implement `WmoFormatAdapter`
  ```csharp
  public static class WmoFormatAdapter
  {
      public static WoWToolbox.Core.v2.Models.Wmo ToDomainModel(this WoWFormatLib.Structs.WMO.WMO wmo) { ... }
      public static WoWToolbox.Core.v2.Models.WmoGroup ToDomainModel(this WoWFormatLib.Structs.WMO.WMOGroup group) { ... }
  }
  ```

### 1.3 Update WmoObjExporter
- [ ] Modify `WmoObjExporter` to use `WoWFormatLib.WMOReader` + adapter
- [ ] Ensure material and texture paths are handled correctly
- [ ] Add validation to compare output with existing implementation

### 1.4 Testing
- [ ] Create integration test project: `WoWToolbox.IntegrationTests`
  - [ ] Add test assets (sample WMO files)
  - [ ] Add tests for WMO loading and OBJ export
  - [ ] Compare output hashes with wow.export

## Success Criteria
- [ ] Solution builds with WoWFormatLib reference
- [ ] WMO files can be loaded and exported to OBJ using the new adapter
- [ ] No regressions in existing functionality
- [ ] Tests pass with matching output hashes

## Dependencies
- WoWFormatLib project
- Sample WMO files for testing
- wow.export for output comparison

## Risks & Mitigation
- **Risk**: Performance degradation due to adapter layer
  - **Mitigation**: Profile and optimize hot paths; consider direct mapping where possible
- **Risk**: Missing features in WoWFormatLib
  - **Mitigation**: Document gaps and plan for extensions or workarounds
