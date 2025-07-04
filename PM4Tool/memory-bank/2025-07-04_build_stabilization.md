# 2025-07-04 – Core v2 Build Stabilization

We achieved a clean compile of `WoWToolbox.Core.v2`.

Key fixes today:
1. **Type mismatch corrections in `MslkExporter`**
   * Added explicit casts from `uint → ushort` and `uint → int`.
   * Replaced placeholder alias with real `MsvtVertex` struct.
2. **Interface alignment**
   * `IMslkExporter` updated to use `ISet<int>`; implementation updated accordingly.
3. **Null-safety polish** – path validations added to silence nullable warnings.
4. **Final rebuild** – project now builds with zero errors (external package warnings remain).

Next focus: migrate all test projects in `test/` to reference the new `WoWToolbox.Core.v2` library and remove legacy tests.
