# GV-00C Base Solution And Folder Topology

## Intent

Describe the future repo shape so the current `wow-viewer` work can converge toward it instead of improvising folder ownership later.

## Scope

- future solution layout
- `BASE` folder purpose
- host application placement
- profile/personality library placement
- tool and dataset seam placement

## Touched Surfaces

- architecture docs only for now
- future `.slnx` / project layout decisions
- future extraction checklist

## Inputs And Assumptions

- the future engine repo should treat engine layers and profile libraries as first-class siblings
- `game-viewer` is one host app, not the only one
- `wow-viewer` should become one profile/personality family in that future repo

## Outputs

- a proposed future layout such as:
  - `BASE/Runtime`
  - `BASE/Rendering`
  - `BASE/Audio`
  - `BASE/Content`
  - `Hosts/game-viewer`
  - `Profiles/WoW`
  - `Profiles/Warcraft3`
  - `Profiles/Museums`
  - `Tools/...`
- folder-ownership rules for each lane

## Dependencies

- GV-00B
- GV-06

## Proof

- future micro-plans can name exact intended home folders even before physical extraction happens

## Stop Conditions

- the future repo shape is specific enough to steer new code and docs without pretending the migration has already happened

## Non-Goals

- no current `.csproj` reshuffle
- no promise that final names are immutable
