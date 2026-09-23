# T007 — Solver Selection: License, Determinism, Collision Scope, Cloth Route

**Evaluated**: 2026-09-03. **Method**: primary sources only — NuGet registration and `.nuspec`
metadata, and the projects' own `LICENSE` files and repository contents. No licensing claim in this
file rests on recollection or on a third-party summary.

**Constraint being satisfied** (from [research.md](../research.md)): the operator requires an
existing permissively licensed C# solver, and **cloth must be a selection criterion rather than a
discovery**, because flag and banner motion is the primary visible outcome (US4). A rigid-only
selection that discovered missing cloth later would strand that user story.

---

## Result

**Selected: Jitter2, pinned to version 2.8.10.**

**Rejected: BepuPhysics v2.** Not on licensing — Apache-2.0 is perfectly acceptable — but because it
fails the cloth criterion outright and offers no determinism evidence, which are the two properties
this feature is chosen for.

**This selection does not add a package reference.** That is Phase 3 work and it needs the operator's
go-ahead, since it is a permanent third-party dependency.

## Comparison — all rows verified against primary sources

| Criterion | BepuPhysics v2 | Jitter2 |
|---|---|---|
| Newest stable on NuGet | **2.4.0**, published **2022-02-18** | **2.8.10**, published **2026-08-31** |
| License | **Apache-2.0** (declared `licenseExpression`; `LICENSE.md` in repo is the Apache 2.0 text) | **MIT** (`LICENSE` in repo: "MIT License / Copyright (c) Thorben Linneweber and contributors") |
| Target frameworks | .NET 8 per repo README; the 2.4.0 package targets .NET 6.0 | **`net8.0`, `net9.0`, `net10.0`** |
| Cloth / soft body | **Absent.** The README enumerates spheres, capsules, boxes, triangles, cylinders, convex hulls, compounds and meshes — all rigid. Neither cloth nor soft bodies are mentioned | **Present as a first-class library area.** `src/Jitter2/SoftBodies/` ships `SoftBody`, `SoftBodyShape`, `SoftBodyTetrahedron`, `SoftBodyTriangle`, `SpringConstraint` |
| Determinism | **No claim in the README.** Neither determinism nor cross-platform determinism is discussed | **Claimed, implemented, tested and CI-enforced** — see below |
| Maintenance | NuGet releases stop in 2022 | Released 3 days before this evaluation |

## Determinism — the decisive evidence

Jitter2's README states: *"Optional cross-platform deterministic solver mode for reproducible
simulation."* A README claim is not evidence, so it was checked against the repository. The
implementation and its enforcement both exist:

| Path | What it is |
|---|---|
| `src/Jitter2/World.Deterministic.cs` | The deterministic solver mode, in the library |
| `src/Jitter2/LinearMath/StableMath.cs` | The stable-math routines it depends on |
| `src/JitterTests/Robustness/ReproducibilityTest.cs` | A reproducibility test in the project's own suite |
| `src/JitterTests/Constraints/DeterministicConstraintSolverTests.cs` | Deterministic constraint-solver tests |
| **`.github/workflows/deterministic-hash.yml`** | **A CI workflow that hashes simulation output** |

The CI workflow is the strongest item: determinism is a **gated, regression-tested property of the
project**, not a documentation promise. That is the difference between "should be reproducible" and
"is checked to be reproducible on every change", and it is what SC-004 and SC-005 require.

BepuPhysics v2 offers no comparable artifact. Determinism there would be **our** problem to establish
and **our** problem to keep.

## Cloth route — verified, and honestly characterised

Jitter2 provides the primitives in the library and **demonstrates cloth built from them**:

- Library: `SoftBodyTriangle` (the per-triangle collision shape) and `SpringConstraint` (the
  structural spring network) in `src/Jitter2/SoftBodies/`.
- Demonstration: **`src/JitterDemo/Demos/SoftBody/SoftBodyCloth.cs`**, declared
  `public class SoftBodyCloth : SoftBody` — roughly 115 lines that build a cloth from triangle data
  by making each triangle a collision shape and connecting adjacent vertices with springs.

**The precise status matters, so state it exactly**: `SoftBodyCloth` lives in the *demo* project, not
in the shipped library. It is **sample code, not public API**. The route is therefore "adapt ~115
lines of first-party MIT-licensed sample code onto shipped library primitives" — not "call a
supported `Cloth` class".

That is still a decisively better position than BepuPhysics v2, where cloth would have to be designed
from rigid-body constraints with no first-party reference at all. And because Jitter2 is MIT,
adapting that sample into our own adapter is unambiguously permitted, with attribution.

Recording this distinction now is the point of the criterion. Discovering later that "soft bodies are
supported" meant "a demo exists" would be exactly the stranding that
[research.md](../research.md) set out to prevent.

## Licensing — verified, including one metadata trap

**Jitter2 is MIT.** Verified from the repository `LICENSE` file: *"MIT License / Copyright (c)
Thorben Linneweber and contributors"*.

**The trap**: NuGet's registration index reports an **empty `licenseExpression`** for Jitter2 2.7.x
and later, while older versions declared `MIT`. Read carelessly, that looks like a license change or
a package with no license.

It is neither. The 2.8.10 `.nuspec` declares:

```xml
<license type="file">LICENSE</license>
<licenseUrl>https://aka.ms/deprecateLicenseUrl</licenseUrl>
<repository type="git" url="https://github.com/notgiven688/jitterphysics2.git"
            branch="refs/tags/2.8.10"
            commit="781fbb4a50155870e9fe3b84350448c2737ea21a" />
```

The project moved from an SPDX **expression** to a bundled **license file**, which is why the
expression field is empty — Microsoft deprecated `licenseUrl` in favour of exactly this. The licence
is the `LICENSE` file inside the package, and the nuspec pins the **exact commit**
`781fbb4a50155870e9fe3b84350448c2737ea21a`, so the authoritative text is reproducibly identifiable
rather than inferred from `main`.

**BepuPhysics v2 is Apache-2.0**, declared as a `licenseExpression` on the package and matching the
Apache 2.0 text in the repository's `LICENSE.md`. Both candidates are permissively licensed and
either would have satisfied the legal constraint on its own.

## Framework fit

Jitter2 2.8.10 ships `net10.0` among its target framework groups. This project is .NET 10, so the fit
is exact and there is no down-level dependency. BepuPhysics 2.4.0 on NuGet targets .NET 6.0 —
consumable from .NET 10, but four years stale against a runtime that has moved twice since.

## What this selection does not settle

- **Collision scope against real `.phys` data is unproven.** The measured shape set is boxes,
  capsules and spheres ([physics-adapter-contract.md](physics-adapter-contract.md) §5), all of which
  Jitter2 covers. But no real sidecar has been parsed yet (T002), so shape coverage is verified
  against the *decoded format*, not against *observed assets*.
- **Joint mapping is unverified.** The client's three joint families — spherical, shoulder, weld —
  must each map to a Jitter2 constraint. Only `spherical` has an obvious counterpart. **Shoulder and
  weld joints are an open risk** and belong in Phase 6, not assumed here.
- **The client's own solver behaviour still governs the port**, not Jitter2's defaults: gravity
  `(0, 0, -10.0)`, the pinned FP control state, distance culling at the per-instance update, and the
  teleport-on-first-update rule are all adapter obligations regardless of solver.
- **No performance claim.** Nothing has been benchmarked.

## Gate status

**T007 is complete, with a selection rather than a blocker.** Combined with T003/T004/T005, the two
Phase 0 gates that block Phase 2 and Phase 3 are now both answered on evidence.

**T002 remains open** and still blocks real-byte validation. **T012** may now generate the
sidecar-parser and solver-adapter task set, per the rule in [tasks.md](../tasks.md), since T005 has
proven the sidecar boundary and T007 has recorded the selected solver's license and cloth route.

## Sources

- [BEPUphysics on NuGet (registration index)](https://api.nuget.org/v3/registration5-semver1/bepuphysics/index.json)
- [bepuphysics2 LICENSE.md](https://raw.githubusercontent.com/bepu/bepuphysics2/master/LICENSE.md)
- [bepuphysics2 README.md](https://raw.githubusercontent.com/bepu/bepuphysics2/master/README.md)
- [Jitter2 on NuGet (registration index)](https://api.nuget.org/v3/registration5-semver1/jitter2/index.json)
- [Jitter2 2.8.10 .nuspec](https://api.nuget.org/v3-flatcontainer/jitter2/2.8.10/jitter2.nuspec)
- [jitterphysics2 LICENSE](https://raw.githubusercontent.com/notgiven688/jitterphysics2/main/LICENSE)
- [jitterphysics2 README.md](https://raw.githubusercontent.com/notgiven688/jitterphysics2/main/README.md)
- [jitterphysics2 src/Jitter2/SoftBodies](https://github.com/notgiven688/jitterphysics2/tree/main/src/Jitter2/SoftBodies)
- [jitterphysics2 SoftBodyCloth.cs](https://github.com/notgiven688/jitterphysics2/blob/main/src/JitterDemo/Demos/SoftBody/SoftBodyCloth.cs)
