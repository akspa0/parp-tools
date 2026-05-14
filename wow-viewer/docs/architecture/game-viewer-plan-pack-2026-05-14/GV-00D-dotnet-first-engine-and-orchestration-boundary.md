# GV-00D Dotnet First Engine And Orchestration Boundary

## Intent

Make `.NET 10` and `C#` the primary implementation and orchestration backbone for the engine program while keeping Python in explicit downstream roles.

## Scope

- language/runtime ownership
- orchestration ownership
- service/tool boundary
- Python containment rules

## Touched Surfaces

- engine modernization docs
- host/editor docs
- future tool/service boundaries
- future extraction/repo-topology docs

## Inputs And Assumptions

- the user's other project currently shows Python orchestration strain
- repo-local evidence from the WoW tooling lane strongly favors C# for performance-sensitive content and runtime work
- Python remains useful for training, ML experiments, and some external orchestration or analysis helpers

## Outputs

- one rule that engine/runtime/editor/tool orchestration is `C#` and `.NET 10` first
- one rule that Python stays primarily in:
  - training and inference
  - dataset analysis and ML utilities
  - optional external automation that does not redefine engine ownership
- one rule that performance-sensitive extraction, conversion, playback, and runtime flows should default to the `C#` side unless there is a concrete reason not to

## Dependencies

- GV-00B
- GV-00C

## Proof

- future planning can answer "should this orchestration live in Python or C#?" from one small contract

## Stop Conditions

- a smaller model can route new work to the right runtime without reopening the language choice every time

## Non-Goals

- no benchmark suite in this plan
- no claim that Python disappears entirely
