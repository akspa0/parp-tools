# World Context Contract

This feature has no network API. The contract is an in-process boundary between shared readers,
runtime world state, and the viewer.

## Inputs

- Active world map identity and build/profile.
- Camera head snapshot for the current frame.
- Resident terrain chunk metadata, including the parser-owned raw MCNK area ID.
- Resident WMO placement/group candidates and any profile-proven WMO area evidence.
- Active client DBC/DB2 plus matching DBD schema through the existing provider.
- Profile-scoped lighting inputs from Specs 106/138 and existing WMO/M2 data.

## Outputs

`WorldContextSnapshot` must be serializable for diagnostics and must include:

- camera frame ID and eye state;
- ADT raw area ID and exact tile/chunk source;
- WMO identity/group and WMO area evidence when available;
- AreaTable row, localized name, parent chain, map validation, build/locale, and logical columns;
- selected source, confidence, candidate count, and explicit unresolved/fallback reason;
- lighting selection and shader/effect fallback diagnostics.

## Consumer rules

1. The status bar shows the selected AreaName only when the result is resolved and displays a compact
   diagnostic marker for unresolved context.
2. Visibility, fog, WMO containment, terrain context, and WMO/M2 lighting consume the same snapshot.
3. A missing WMO area field falls back to the ADT result and reports `UnavailableForProfile`.
4. A map mismatch is not silently converted into `Unknown`; it is retained in the diagnostic result.
5. A BLS/effect fallback is named as fallback and never labeled native parity.
6. Snapshot evaluation is bounded to resident candidates and does not trigger whole-map loading.

## Versioning

The snapshot and camera state carry a version. Adding a profile-specific source requires a focused
fixture and an updated research note; it must not reinterpret an existing field for all builds.
