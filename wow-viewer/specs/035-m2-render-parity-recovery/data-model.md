# Data Model: M2 Render Parity Recovery

## Entity: M2RouteDecision

- **Purpose**: Record exactly how a world M2 instance reached its draw path.
- **Fields**:
  - `ModelPath` (string): canonical virtual path.
  - `BuildProfileId` (string): resolved M2 profile id.
  - `PrimaryRoute` (enum): declared first-choice route.
  - `AppliedRoute` (enum): actual route used.
  - `SelectedSkinPath` (string|null): resolved skin path if applicable.
  - `FallbackReason` (string|null): reason primary route was not used.
  - `TimestampUtc` (datetime): decision capture time.
- **Validation Rules**:
  - `AppliedRoute` must always be populated.
  - `FallbackReason` required when `AppliedRoute != PrimaryRoute`.

## Entity: M2MaterialPassProfile

- **Purpose**: Capture world-pass semantics at material/layer level for parity checks.
- **Fields**:
  - `ModelPath` (string)
  - `SectionIndex` (int)
  - `MaterialIndex` (int)
  - `LayerIndex` (int)
  - `BlendDeclaration` (string)
  - `PassClass` (enum: Opaque, Cutout, Blended)
  - `DepthWrite` (bool)
  - `BlendEnabled` (bool)
  - `AlphaThreshold` (float|null)
- **Validation Rules**:
  - `PassClass` must be one of the declared enum values.
  - `AlphaThreshold` required when `PassClass == Cutout`.

## Entity: M2ParitySample

- **Purpose**: Define and track each regression-check sample model/tile.
- **Fields**:
  - `SampleId` (string)
  - `Build` (string)
  - `Map` (string)
  - `TileX` (int)
  - `TileY` (int)
  - `ModelPath` (string)
  - `ExpectedVisible` (bool)
  - `ProbeEvidencePath` (string)
  - `RuntimeEvidencePath` (string)
  - `LastResult` (enum: Pass, Fail, Unknown)
- **Validation Rules**:
  - `ModelPath` and tile coordinates required.
  - `LastResult` must be updated after each parity run.

## Relationships

- One `M2RouteDecision` can emit many `M2MaterialPassProfile` rows.
- One `M2ParitySample` run references one `M2RouteDecision` and many `M2MaterialPassProfile` rows as evidence.
