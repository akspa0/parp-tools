# System Patterns (Next)

- Architecture: Library-first (`GillijimProject.Next.Core`) with CLI wrapper (`GillijimProject.Next.Cli`).
- Layers:
  - Domain: immutable-ish models with explicit dimensionality (e.g., 9x9, 8x8) and XML docs.
  - IO: Readers/Writers for Alpha/LK as adapters; writers ensure MCNK/MH2O/MCLQ ordering/policies.
  - Transform: Converters (e.g., Liquids) that operate on domain models.
  - Services: Crosswalks and translators (e.g., AreaID + Map crosswalk via DBCD).
- Conventions:
  - FourCC forward in memory; reversed on disk by serializers.
  - MH2O omitted when empty; MCLQ written last in MCNK (Alpha writer path).
  - LVF Case 0/2 supported initially; Case 1/3 deferred with TODO(PORT).
- Error Handling: Exceptions; validate inputs (bounds, masks) and log actionable warnings.
- Testing: Unit + round-trip + CLI integration with skip-if-missing fixtures.
