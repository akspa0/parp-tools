# GV-00B Base Engine Repo Extraction Boundary

## Intent

Define the future extraction boundary where the engine becomes its own repository and `wow-viewer` becomes one supported profile/library family instead of the accidental top-level identity.

## Scope

- future repo boundary
- `BASE` engine ownership
- profile/personality library ownership
- migration-safe naming rules

## Touched Surfaces

- future top-level repo layout docs
- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md`
- `wow-viewer/docs/architecture/game-viewer-host-plan-2026-05-13.md`
- plan-pack index and follow-up micro-plans

## Inputs And Assumptions

- `wow-viewer` remains the current development host because the new repo does not exist yet
- once `v0.5.0` is complete, the engine will be extracted and the last legacy `MdxViewer` binary will stay with the old repo
- `wow-viewer` should survive that extraction as a supported profile/library family, not as the whole product definition

## Outputs

- one explicit extraction boundary
- one explicit rule that `BASE` owns engine-neutral runtime, rendering, audio, scene, and content-service contracts
- one explicit rule that WoW/WC3/Museums support lives in profile/personality libraries layered on top

## Dependencies

- GV-00
- GV-00A
- GV-01

## Proof

- future plans can refer to `BASE` vs profile ownership without re-explaining repo extraction every time

## Stop Conditions

- the boundary is clear enough that a smaller model can answer "does this belong in BASE or in a profile library?" without guessing

## Non-Goals

- no immediate file moves
- no new repo creation
- no namespace refactor yet
