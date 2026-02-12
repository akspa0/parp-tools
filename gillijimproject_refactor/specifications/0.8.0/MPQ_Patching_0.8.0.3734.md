# MPQ Patching Analysis — WoW 0.8.0.3734

## Summary
Build `0.8.0.3734` has explicit MPQ patch archive support, including discovery/loading of `patch.MPQ` and `wow-patch.mpq`, plus signature/version validation logic before continuing startup.

## Build
- **Build**: `0.8.0.3734`
- **Source confidence**: High (direct decompiler evidence)

## Key String Evidence
- `"patch.MPQ"` @ `0x00856e1c`
- `"wow-patch.mpq"` @ `0x00861090`
- Core archive set includes `Data\model.MPQ`, `Data\texture.MPQ`, `Data\terrain.MPQ`, `Data\wmo.MPQ`, etc.

## Loader / Patch Flow (Observed)

### 1) Archive list construction and open
- **Function**: `FUN_004035c0`
- Behavior:
  - Builds archive list using dynamic archive names + fixed `Data\*.MPQ` list.
  - Calls `FUN_00657830(...)` per entry.
  - On failure logs `Failed to open archive` and keeps slot null.

### 2) Archive open abstraction
- **Function**: `FUN_00657830`
- Behavior:
  - Allocates archive object.
  - Uses `FUN_00667e40(...)` for standard archive opening path.
  - Has alternate branch for `"flat-"` style input (non-MPQ path), else MPQ-backed path.

### 3) Startup patch archive handling
- **Function**: `FUN_004b1be0`
- Behavior:
  - Constructs patch target path/name (`Patch`, fallback `wow-patch.mpq`).
  - Invokes `FUN_005c8900(...)` (file/open + partial handling path).

### 4) Patch sanity / version gate
- **Function**: `FUN_004b1ee0`
- Behavior:
  - Opens `wow-patch.mpq` via `FUN_00667e40(...)`.
  - Reads metadata through `FUN_006655c0(...)`.
  - If returned version/status is out-of-range (`0` or `>4`), branches into `FUN_004b1dd0()` recovery/update path.

### 5) Signature verification path
- **Function**: `FUN_006655c0` (large Storm-style validation routine)
- Behavior:
  - Loads `advapi32.dll` crypto APIs dynamically (`CryptAcquireContextA`, `CryptCreateHash`, `CryptVerifySignatureA`, etc.).
  - Uses embedded `BLIZZARDKEY` resource.
  - Hashes mapped archive regions and verifies signature.
  - Returns status used by patch gating logic.

## Interpretation: “MPQ patching ability”
Confirmed capabilities in this build:
- Patch archive discovery (`patch.MPQ`, `wow-patch.mpq`).
- Patch archive open/integration in startup flow.
- Patch archive verification/signature checks before acceptance.
- Recovery/update branch when patch metadata/signature status is invalid.

## Open Questions / Unknowns
- Exact semantic meaning of status values `1..4` from `FUN_006655c0` (only range checks are directly visible here).
- Exact conflict-resolution order between `patch.MPQ` and `wow-patch.mpq` without deeper runtime tracing.

## Confidence
- **High** for existence of patch MPQ support and cryptographic validation.
- **Medium** for status-code semantics (requires deeper reverse of helper routines).