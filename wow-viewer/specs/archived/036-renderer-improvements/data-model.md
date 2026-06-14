# Data Model: Renderer Improvements Convergence

## RendererConvergencePhase

- **Purpose**: Represents one bounded modernization phase in the convergence owner plan.
- **Fields**:
  - `Id`: Stable phase identifier such as `phase-1-lighting-foundation`
  - `Title`: Short human-readable phase name
  - `Goal`: Single primary concern
  - `Dependencies`: Other phase ids that must validate first
  - `OwnerLayers`: One or more of `Core.IO`, `Core.Runtime`, `WowViewer.App`
  - `ValidationScenarios`: Linked `RendererValidationScenario` ids
  - `SourceMappings`: Linked `SourceSpecMapping` ids
  - `OutOfScopeNotes`: Optional list of adjacent but excluded concerns

## RendererCapabilitySlice

- **Purpose**: Represents a renderer subsystem capability inherited from a source spec.
- **Fields**:
  - `Id`: Stable slice id such as `wmo-interior-exterior-dispatch`
  - `SourceSpec`: One of `030`, `031`, `032`
  - `Subsystem`: Terrain, WMO, Lighting, Fog, Sky, Liquid, Viewer
  - `Description`: Short statement of the capability
  - `CanonicalOwnerLayer`: Where the behavior belongs
  - `Prerequisites`: Other slice ids or phase ids
  - `ConvergencePhaseId`: Phase that owns implementation
  - `ValidationExpectation`: Summary of how proof is established

## SourceSpecMapping

- **Purpose**: Traceability record linking a source spec area into the new convergence owner plan.
- **Fields**:
  - `Id`: Stable mapping id
  - `SourceSpecId`: `030`, `031`, or `032`
  - `SourceSection`: Human-readable source section or theme
  - `ConvergencePhaseId`: Target phase in spec 036
  - `Status`: `mapped`, `deferred`, or `out-of-scope`
  - `Notes`: Rationale for the mapping decision

## RendererValidationScenario

- **Purpose**: Defines a staged-client proof case for a convergence phase.
- **Fields**:
  - `Id`: Stable scenario id
  - `BuildProfile`: Client build string such as `3.3.5.12340`
  - `ClientRoot`: Staged client path
  - `MapContext`: Map/tile/WMO/sample description
  - `Subsystems`: Related slice ids or subsystem tags
  - `EvidenceType`: Screenshot, trace log, probe output, render comparison
  - `PassCriteria`: Concrete observable outcomes
  - `BlockedBy`: Optional phase or dependency ids

## RendererOwnershipBoundary

- **Purpose**: States which layer owns a class of renderer behavior.
- **Fields**:
  - `Concern`: Example `terrain-cell-decoding`, `wmo-pass-dispatch`, `time-of-day-slider`
  - `OwnerLayer`: `Core.IO`, `Core.Runtime`, or `WowViewer.App`
  - `NonOwnerLayers`: Layers that may consume but not redefine the behavior
  - `Why`: Brief rationale
